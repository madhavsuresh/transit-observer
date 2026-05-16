"""Direction-label correctness audit.

For each resolved forecast we re-load the arrivals the predictor saw,
re-apply the direction filter the predictor used, and compare against
the boarded run's direction_code / destination_name. Per-forecast row
goes into `direction_audit`. Two metrics emerge:

- **Recall** = boarded_was_kept rate. Did the filter let the right
  train through? Want ~100%.
- **Direction-precision (proxy)** = of the arrivals the filter kept,
  what fraction shared the boarded run's direction_code? Low values
  mean the filter is keeping noise (wrong-direction trains).

When this precision rate stabilizes per (line, boarding_map_id,
alighting_map_id), it becomes the empirical truth — the modal
direction_code for that (line, A, B) is the right direction, and we
can replace the dot-product heuristic with a hard direction_code
filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import duckdb

from .catalog import LStation, by_name, load_catalog
from .trip_generator import _heads_toward_alighting  # type: ignore[attr-defined]
from .trip_generator import TripSpec


_NAME_LOOKUP: dict[str, LStation] | None = None
_CATALOG_BY_ID: dict[int, LStation] | None = None


def _ensure_catalog() -> tuple[dict[str, LStation], dict[int, LStation]]:
    global _NAME_LOOKUP, _CATALOG_BY_ID
    if _NAME_LOOKUP is None or _CATALOG_BY_ID is None:
        cat = load_catalog()
        _NAME_LOOKUP = by_name(cat)
        _CATALOG_BY_ID = {s.map_id: s for s in cat}
    return _NAME_LOOKUP, _CATALOG_BY_ID


@dataclass(frozen=True)
class AuditResult:
    forecast_id: str
    candidate_arrivals_count: int
    kept_arrivals_count: int
    kept_direction_codes: str
    kept_destination_names: str
    boarded_direction_code: str | None
    boarded_destination_name: str | None
    boarded_was_kept: bool
    kept_matching_boarded_direction: int


def audit_resolved_forecast(
    conn: duckdb.DuckDBPyConnection,
    *,
    forecast_id: str,
    now: datetime,
    mode: str = "L",
    outcome: dict | None = None,
) -> AuditResult | None:
    """Compute and persist an audit row for one resolved forecast.

    Dispatches per `mode`. Each mode has a different idea of "direction":
    - L: directionCode + destinationName at the boarding station
    - bus: directionName + destinationName at the boarding stop
    - metra: direction_id (0/1)
    - intercampus: direction ('northbound' | 'southbound')
    """
    if mode == "bus":
        return _audit_bus(conn, forecast_id=forecast_id, now=now, outcome=outcome)
    if mode == "metra":
        return _audit_metra(conn, forecast_id=forecast_id, now=now, outcome=outcome)
    if mode == "intercampus":
        return _audit_intercampus(conn, forecast_id=forecast_id, now=now, outcome=outcome)
    return _audit_l(conn, forecast_id=forecast_id, now=now, outcome=outcome)


def _audit_l(
    conn: duckdb.DuckDBPyConnection,
    *,
    forecast_id: str,
    now: datetime,
    outcome: dict | None,
) -> AuditResult | None:
    row = conn.execute(
        """
        SELECT q.line, q.boarding_map_id, q.alighting_map_id,
               q.direction_code, q.leave_at, q.snapshot_polled_at,
               o.boarded_run_number, o.boarded_at
          FROM forecast_queue q
          JOIN forecast_outcomes o USING (forecast_id)
         WHERE forecast_id = ?
        """,
        [forecast_id],
    ).fetchone()
    if row is None:
        return None
    line, boarding_id, alighting_id, _, leave_at, snapshot_at, run_number, boarded_at = row

    # Reload the arrivals the predictor would have seen.
    candidate_rows = conn.execute(
        """
        SELECT arrival_at, is_approaching, destination_name, direction_code
          FROM train_arrivals_raw
         WHERE line = ?
           AND map_id = ?
           AND polled_at >= ?
           AND arrival_at >= ?
           AND arrival_at <= ?
           AND is_fault = FALSE
         ORDER BY polled_at DESC
        """,
        [
            line, boarding_id,
            leave_at - timedelta(minutes=5),
            leave_at,
            leave_at + timedelta(minutes=30),
        ],
    ).fetchall()
    candidate_count = len(candidate_rows)

    _, catalog_by_id = _ensure_catalog()
    boarding = catalog_by_id.get(boarding_id)
    alighting = catalog_by_id.get(alighting_id)
    if boarding is None or alighting is None:
        return None

    spec = TripSpec(
        line_catalog="",
        line_api=line,
        boarding=boarding,
        alighting=alighting,
        direction_label="",
        leave_at=leave_at,
    )

    kept: list[tuple[datetime, bool, str | None, str | None]] = []
    for arrival_at, is_app, destination, direction_code in candidate_rows:
        if _heads_toward_alighting(spec, destination):
            kept.append((arrival_at, bool(is_app), destination, direction_code))

    # The boarded run's direction + destination — look it up in raw arrivals.
    boarded_row = conn.execute(
        """
        SELECT destination_name, direction_code
          FROM train_arrivals_raw
         WHERE line = ? AND map_id = ? AND run_number = ?
           AND polled_at >= ? AND polled_at <= ?
         ORDER BY polled_at DESC
         LIMIT 1
        """,
        [
            line, boarding_id, run_number,
            leave_at - timedelta(minutes=10),
            (boarded_at or leave_at) + timedelta(minutes=5),
        ],
    ).fetchone() or (None, None)
    boarded_destination, boarded_direction = boarded_row

    kept_destinations = sorted({d for _, _, d, _ in kept if d})
    kept_direction_codes = sorted({c for _, _, _, c in kept if c})
    boarded_was_kept = bool(boarded_destination) and any(
        d == boarded_destination for _, _, d, _ in kept
    )
    matching = sum(
        1 for _, _, _, c in kept if c is not None and c == boarded_direction
    )

    result = AuditResult(
        forecast_id=forecast_id,
        candidate_arrivals_count=candidate_count,
        kept_arrivals_count=len(kept),
        kept_direction_codes=",".join(kept_direction_codes),
        kept_destination_names=",".join(kept_destinations),
        boarded_direction_code=boarded_direction,
        boarded_destination_name=boarded_destination,
        boarded_was_kept=boarded_was_kept,
        kept_matching_boarded_direction=matching,
    )

    _insert_audit(
        conn,
        forecast_id=forecast_id,
        mode="L",
        now=now,
        candidate_count=candidate_count,
        kept_count=len(kept),
        kept_direction_codes=result.kept_direction_codes,
        kept_destination_names=result.kept_destination_names,
        boarded_direction_code=boarded_direction,
        boarded_destination_name=boarded_destination,
        boarded_was_kept=boarded_was_kept,
        kept_matching_boarded_direction=matching,
    )
    return result


def _audit_bus(
    conn: duckdb.DuckDBPyConnection,
    *,
    forecast_id: str,
    now: datetime,
    outcome: dict | None,
) -> AuditResult | None:
    """For bus: keep arrivals whose direction_name matches the boarding stop's
    direction_label (as the predictor does). Compare to the boarded vehicle's
    direction_name."""
    row = conn.execute(
        """
        SELECT q.line, q.boarding_text_id, q.direction_code, q.leave_at
          FROM forecast_queue q
         WHERE forecast_id = ?
        """,
        [forecast_id],
    ).fetchone()
    if row is None:
        return None
    route, boarding_text_id, expected_direction, leave_at = row
    if boarding_text_id is None:
        return None
    boarding_stop_id = int(boarding_text_id)

    candidate_rows = conn.execute(
        """
        SELECT direction_name, destination_name
          FROM bus_predictions_raw
         WHERE route = ? AND stop_id = ?
           AND polled_at >= ? AND arrival_at >= ? AND arrival_at <= ?
        """,
        [route, boarding_stop_id, leave_at - timedelta(minutes=5), leave_at, leave_at + timedelta(minutes=45)],
    ).fetchall()
    candidate_count = len(candidate_rows)
    target = (expected_direction or "").lower()
    kept = [
        (direction_name, destination)
        for direction_name, destination in candidate_rows
        if not target or (direction_name and direction_name.lower() == target)
    ]
    boarded_direction = (outcome or {}).get("boarded_direction_code")
    boarded_destination = (outcome or {}).get("boarded_destination_name")
    matching = sum(
        1 for direction_name, _ in kept if direction_name and boarded_direction and direction_name == boarded_direction
    )
    boarded_was_kept = bool(boarded_direction) and any(
        direction_name and direction_name == boarded_direction for direction_name, _ in kept
    )
    _insert_audit(
        conn,
        forecast_id=forecast_id,
        mode="bus",
        now=now,
        candidate_count=candidate_count,
        kept_count=len(kept),
        kept_direction_codes=",".join(sorted({d for d, _ in kept if d})),
        kept_destination_names=",".join(sorted({dest for _, dest in kept if dest})),
        boarded_direction_code=boarded_direction,
        boarded_destination_name=boarded_destination,
        boarded_was_kept=boarded_was_kept,
        kept_matching_boarded_direction=matching,
    )
    return AuditResult(
        forecast_id=forecast_id,
        candidate_arrivals_count=candidate_count,
        kept_arrivals_count=len(kept),
        kept_direction_codes=",".join(sorted({d for d, _ in kept if d})),
        kept_destination_names=",".join(sorted({dest for _, dest in kept if dest})),
        boarded_direction_code=boarded_direction,
        boarded_destination_name=boarded_destination,
        boarded_was_kept=boarded_was_kept,
        kept_matching_boarded_direction=matching,
    )


def _audit_metra(
    conn: duckdb.DuckDBPyConnection,
    *,
    forecast_id: str,
    now: datetime,
    outcome: dict | None,
) -> AuditResult | None:
    """For Metra: the predictor doesn't apply a destination filter — it
    picks viable trips by trip_id join. Audit on direction_id (0/1)."""
    row = conn.execute(
        """
        SELECT q.line, q.direction_code
          FROM forecast_queue q WHERE forecast_id = ?
        """,
        [forecast_id],
    ).fetchone()
    if row is None:
        return None
    _, expected_direction = row
    boarded_direction = (outcome or {}).get("boarded_direction_code")
    boarded_was_kept = (
        expected_direction is None
        or boarded_direction is None
        or str(boarded_direction) == str(expected_direction)
    )
    _insert_audit(
        conn,
        forecast_id=forecast_id,
        mode="metra",
        now=now,
        candidate_count=1,
        kept_count=1,
        kept_direction_codes=str(expected_direction) if expected_direction is not None else "",
        kept_destination_names="",
        boarded_direction_code=str(boarded_direction) if boarded_direction is not None else None,
        boarded_destination_name=(outcome or {}).get("boarded_destination_name"),
        boarded_was_kept=boarded_was_kept,
        kept_matching_boarded_direction=1 if boarded_was_kept else 0,
    )
    return None


def _audit_intercampus(
    conn: duckdb.DuckDBPyConnection,
    *,
    forecast_id: str,
    now: datetime,
    outcome: dict | None,
) -> AuditResult | None:
    """For Intercampus: the predictor restricts trips by direction
    ('northbound' | 'southbound'). Audit by direction."""
    row = conn.execute(
        """
        SELECT q.direction_code FROM forecast_queue q WHERE forecast_id = ?
        """,
        [forecast_id],
    ).fetchone()
    if row is None:
        return None
    expected_direction = row[0]
    boarded_direction = (outcome or {}).get("boarded_direction_code")
    boarded_was_kept = (
        expected_direction is None
        or boarded_direction is None
        or str(boarded_direction).lower() == str(expected_direction).lower()
    )
    _insert_audit(
        conn,
        forecast_id=forecast_id,
        mode="intercampus",
        now=now,
        candidate_count=1,
        kept_count=1,
        kept_direction_codes=str(expected_direction or ""),
        kept_destination_names="",
        boarded_direction_code=str(boarded_direction) if boarded_direction is not None else None,
        boarded_destination_name=None,
        boarded_was_kept=boarded_was_kept,
        kept_matching_boarded_direction=1 if boarded_was_kept else 0,
    )
    return None


def _insert_audit(
    conn: duckdb.DuckDBPyConnection,
    *,
    forecast_id: str,
    mode: str,
    now: datetime,
    candidate_count: int,
    kept_count: int,
    kept_direction_codes: str,
    kept_destination_names: str,
    boarded_direction_code: str | None,
    boarded_destination_name: str | None,
    boarded_was_kept: bool,
    kept_matching_boarded_direction: int,
) -> None:
    conn.execute(
        """
        INSERT INTO direction_audit (
            forecast_id, mode, audited_at, candidate_arrivals_count, kept_arrivals_count,
            kept_direction_codes, kept_destination_names,
            boarded_direction_code, boarded_destination_name,
            boarded_was_kept, kept_matching_boarded_direction, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (forecast_id) DO NOTHING
        """,
        [
            forecast_id, mode, now, candidate_count, kept_count,
            kept_direction_codes, kept_destination_names,
            boarded_direction_code, boarded_destination_name,
            boarded_was_kept, kept_matching_boarded_direction, None,
        ],
    )


@dataclass(frozen=True)
class DirectionAuditSummary:
    line: str
    mode: str
    n_audited: int
    recall_rate: float
    avg_direction_precision: float


def audit_summary(
    conn: duckdb.DuckDBPyConnection,
    *,
    min_samples: int = 5,
) -> list[DirectionAuditSummary]:
    rows = conn.execute(
        """
        SELECT q.line, a.mode,
               COUNT(*) AS n,
               AVG(CASE WHEN a.boarded_was_kept THEN 1 ELSE 0 END) AS recall,
               AVG(
                 CASE WHEN a.kept_arrivals_count > 0
                      THEN CAST(a.kept_matching_boarded_direction AS DOUBLE) / a.kept_arrivals_count
                      ELSE NULL END
               ) AS precision_proxy
          FROM direction_audit a
          JOIN forecast_queue q USING (forecast_id)
         WHERE a.boarded_direction_code IS NOT NULL
         GROUP BY q.line, a.mode
        HAVING n >= ?
         ORDER BY a.mode, q.line
        """,
        [min_samples],
    ).fetchall()
    return [
        DirectionAuditSummary(
            line=line,
            mode=mode,
            n_audited=n,
            recall_rate=recall or 0.0,
            avg_direction_precision=precision or 0.0,
        )
        for line, mode, n, recall, precision in rows
    ]
