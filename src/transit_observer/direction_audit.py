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
) -> AuditResult | None:
    """Compute and persist an audit row for one resolved forecast.

    Looks up the boarded run from `forecast_outcomes`, re-fetches the
    arrivals the predictor saw at `boarding_map_id` around `leave_at`,
    re-applies the direction filter the predictor used, compares to the
    realized boarded run, and inserts into `direction_audit`.
    """
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

    conn.execute(
        """
        INSERT INTO direction_audit (
            forecast_id, audited_at, candidate_arrivals_count, kept_arrivals_count,
            kept_direction_codes, kept_destination_names,
            boarded_direction_code, boarded_destination_name,
            boarded_was_kept, kept_matching_boarded_direction, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (forecast_id) DO NOTHING
        """,
        [
            forecast_id, now, candidate_count, len(kept),
            result.kept_direction_codes, result.kept_destination_names,
            boarded_direction, boarded_destination,
            boarded_was_kept, matching, None,
        ],
    )
    return result


@dataclass(frozen=True)
class DirectionAuditSummary:
    line: str
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
        SELECT q.line,
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
         GROUP BY q.line
        HAVING n >= ?
         ORDER BY q.line
        """,
        [min_samples],
    ).fetchall()
    return [
        DirectionAuditSummary(
            line=line,
            n_audited=n,
            recall_rate=recall or 0.0,
            avg_direction_precision=precision or 0.0,
        )
        for line, n, recall, precision in rows
    ]
