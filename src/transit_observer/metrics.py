"""Coverage / calibration / sharpness metrics, bucketed by corridor.

A "corridor bucket" is `(line, direction, hour_of_day, weekday|weekend)`.
Use this module to answer:

- Which buckets have ≥N samples? (corridor inventory)
- Per bucket: what fraction of actuals fell inside p80? (coverage)
- Per bucket: median(p80 − p50)? (sharpness)
- Across all samples, when we predict failure probability p, what's the
  realized failure rate? (calibration curve)

Reads from the read replica; doesn't write.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import duckdb


@dataclass(frozen=True)
class CorridorCoverage:
    line: str
    direction_label: str
    hour_of_day: int
    weekday: bool
    n_samples: int
    coverage_p80: float
    coverage_p90: float
    median_sharpness_seconds: float
    median_p50_residual_seconds: float


@dataclass(frozen=True)
class CalibrationBin:
    predicted_lower: float
    predicted_upper: float
    n: int
    actual_failure_rate: float


def corridor_coverage(
    conn: duckdb.DuckDBPyConnection,
    *,
    min_samples: int = 5,
) -> list[CorridorCoverage]:
    rows = conn.execute(
        """
        SELECT q.line, q.direction_code,
               EXTRACT(hour FROM q.leave_at)::INTEGER AS hod,
               (EXTRACT(dow FROM q.leave_at) BETWEEN 1 AND 5) AS weekday,
               COUNT(*) AS n,
               AVG(CASE WHEN o.in_p80_window THEN 1 ELSE 0 END) AS cov80,
               AVG(CASE WHEN o.in_p90_window THEN 1 ELSE 0 END) AS cov90,
               MEDIAN(q.predicted_total_p80 - q.predicted_total_p50) AS sharpness,
               MEDIAN(o.p50_residual_seconds) AS median_residual
          FROM forecast_outcomes o
          JOIN forecast_queue q USING (forecast_id)
         GROUP BY q.line, q.direction_code, hod, weekday
        HAVING COUNT(*) >= ?
         ORDER BY q.line, q.direction_code, hod, weekday
        """,
        [min_samples],
    ).fetchall()
    return [
        CorridorCoverage(
            line=line,
            direction_label=direction or "?",
            hour_of_day=hod,
            weekday=bool(weekday),
            n_samples=n,
            coverage_p80=cov80 or 0.0,
            coverage_p90=cov90 or 0.0,
            median_sharpness_seconds=sharp or 0.0,
            median_p50_residual_seconds=median_residual or 0.0,
        )
        for line, direction, hod, weekday, n, cov80, cov90, sharp, median_residual in rows
    ]


def uncovered_buckets(
    conn: duckdb.DuckDBPyConnection,
    *,
    target_samples: int = 5,
) -> list[tuple[str, str, int, bool, int]]:
    """Return (line, direction, hour_of_day, weekday, n) for buckets below target.

    Includes buckets with zero samples too, by joining against the
    cross-product of possible buckets implicit in the line catalog.
    """
    rows = conn.execute(
        """
        SELECT q.line, q.direction_code,
               EXTRACT(hour FROM q.leave_at)::INTEGER AS hod,
               (EXTRACT(dow FROM q.leave_at) BETWEEN 1 AND 5) AS weekday,
               COUNT(*) AS n
          FROM forecast_queue q
         GROUP BY q.line, q.direction_code, hod, weekday
        HAVING COUNT(*) < ?
         ORDER BY n ASC, q.line, hod
        """,
        [target_samples],
    ).fetchall()
    return [(line, direction or "?", hod, bool(weekday), n) for line, direction, hod, weekday, n in rows]


def calibration_bins(
    conn: duckdb.DuckDBPyConnection,
    *,
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Calibration of `predicted_failure_prob` vs actual unresolved/failed.

    With the v1 single-leg simulator, `predicted_failure_prob` is 0.0 for
    successful predictions, so this is mostly informational until the
    multi-leg phase. We still produce the bin shape so the dashboard can
    plot it once it's meaningful.
    """
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        row = conn.execute(
            """
            SELECT COUNT(*) AS n,
                   AVG(CASE WHEN o.failed OR o.actual_total_seconds > q.predicted_total_p90 THEN 1 ELSE 0 END) AS actual_rate
              FROM forecast_outcomes o
              JOIN forecast_queue q USING (forecast_id)
             WHERE q.predicted_failure_prob >= ?
               AND q.predicted_failure_prob < ?
            """,
            [lower, upper],
        ).fetchone()
        n, actual = row or (0, 0.0)
        bins.append(CalibrationBin(predicted_lower=lower, predicted_upper=upper, n=n or 0, actual_failure_rate=actual or 0.0))
    return bins


@dataclass(frozen=True)
class Status:
    raw_arrivals_count: int
    runs_observed_count: int
    positions_count: int
    bus_predictions_count: int
    metra_arrivals_count: int
    intercampus_arrivals_count: int
    forecasts_pending: int
    forecasts_resolved: int
    forecasts_unresolvable: int
    latest_poll: datetime | None
    oldest_pending: datetime | None
    overall_p80_coverage: float | None


def _has_column(conn: duckdb.DuckDBPyConnection, *, table: str, column: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM pragma_table_info(?) WHERE name = ? LIMIT 1",
        [table, column],
    ).fetchone()
    return bool(row)


@dataclass(frozen=True)
class CorpusRow:
    forecast_id: str
    leave_at: datetime
    predictor_version: str | None
    direction_code: str | None
    predicted_total_p50: float
    predicted_total_p80: float
    predicted_total_p90: float
    actual_total_seconds: float | None
    in_p80_window: bool | None
    p50_residual_seconds: float | None
    truth_confidence: float | None
    status: str


def corpus_corridor_rows(
    conn: duckdb.DuckDBPyConnection,
    *,
    corridor_id: str,
    limit: int = 50,
) -> list[CorpusRow]:
    """Recent forecasts for one corridor, joined with outcomes if resolved."""
    if not _has_column(conn, table="forecast_queue", column="corridor_id"):
        return []
    rows = conn.execute(
        """
        SELECT q.forecast_id, q.leave_at, q.predictor_version, q.direction_code,
               q.predicted_total_p50, q.predicted_total_p80, q.predicted_total_p90,
               o.actual_total_seconds, o.in_p80_window, o.p50_residual_seconds,
               o.truth_confidence, q.status
          FROM forecast_queue q
          LEFT JOIN forecast_outcomes o USING (forecast_id)
         WHERE q.corridor_id = ?
         ORDER BY q.leave_at DESC
         LIMIT ?
        """,
        [corridor_id, limit],
    ).fetchall()
    return [
        CorpusRow(
            forecast_id=fid, leave_at=leave_at, predictor_version=pv,
            direction_code=dir_code,
            predicted_total_p50=p50, predicted_total_p80=p80, predicted_total_p90=p90,
            actual_total_seconds=actual, in_p80_window=in_p80,
            p50_residual_seconds=resid, truth_confidence=trustconf, status=stat,
        )
        for (
            fid, leave_at, pv, dir_code, p50, p80, p90,
            actual, in_p80, resid, trustconf, stat,
        ) in rows
    ]


@dataclass(frozen=True)
class CorpusCorridorSummary:
    corridor_id: str
    mode: str
    line: str
    direction: str
    origin_label: str
    destination_label: str
    n_predictions: int
    n_resolved: int
    n_unresolvable: int
    coverage_p80: float | None
    median_p50_residual_seconds: float | None
    median_truth_confidence: float | None
    last_predicted_at: datetime | None


def corpus_summary(
    conn: duckdb.DuckDBPyConnection,
    *,
    high_confidence_only: bool = False,
    confidence_threshold: float = 0.5,
) -> list[CorpusCorridorSummary]:
    """Per-corridor coverage + residual stats, joining corridors -> outcomes.

    ``high_confidence_only`` excludes outcomes with ``truth_confidence``
    below ``confidence_threshold`` (spec: low-confidence truths should be
    excluded from headline metrics).
    """
    # Reader may be hitting an older DB whose schema predates the
    # corridors table; return empty rather than blowing up the dashboard.
    has_corridors = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'corridors'"
    ).fetchone()
    if not has_corridors or not has_corridors[0]:
        return []

    threshold = confidence_threshold if high_confidence_only else -1.0
    rows = conn.execute(
        """
        SELECT c.corridor_id, c.mode, c.line, c.direction,
               c.origin_label, c.destination_label,
               COUNT(q.forecast_id) AS n_predictions,
               SUM(CASE WHEN q.status = 'resolved' THEN 1 ELSE 0 END) AS n_resolved,
               SUM(CASE WHEN q.status = 'unresolvable' THEN 1 ELSE 0 END) AS n_unresolvable,
               AVG(CASE WHEN o.in_p80_window IS NOT NULL
                            AND COALESCE(o.truth_confidence, 0) >= ?
                       THEN (CASE WHEN o.in_p80_window THEN 1.0 ELSE 0.0 END) END) AS cov80,
               MEDIAN(CASE WHEN COALESCE(o.truth_confidence, 0) >= ?
                            THEN o.p50_residual_seconds END) AS median_p50_residual,
               MEDIAN(o.truth_confidence) AS median_tc,
               c.last_predicted_at
          FROM corridors c
          LEFT JOIN forecast_queue q ON q.corridor_id = c.corridor_id
          LEFT JOIN forecast_outcomes o ON o.forecast_id = q.forecast_id
         GROUP BY c.corridor_id, c.mode, c.line, c.direction,
                  c.origin_label, c.destination_label, c.last_predicted_at,
                  c.priority
         ORDER BY c.priority ASC, c.corridor_id
        """,
        [threshold, threshold],
    ).fetchall()
    return [
        CorpusCorridorSummary(
            corridor_id=cid, mode=mode, line=line, direction=direction,
            origin_label=origin, destination_label=destination,
            n_predictions=n_pred or 0,
            n_resolved=n_resolved or 0,
            n_unresolvable=n_unresolvable or 0,
            coverage_p80=cov80,
            median_p50_residual_seconds=median_resid,
            median_truth_confidence=median_tc,
            last_predicted_at=last_at,
        )
        for (
            cid, mode, line, direction, origin, destination,
            n_pred, n_resolved, n_unresolvable,
            cov80, median_resid, median_tc, last_at,
        ) in rows
    ]


def status(conn: duckdb.DuckDBPyConnection) -> Status:
    raw = conn.execute("SELECT COUNT(*), MAX(polled_at) FROM train_arrivals_raw").fetchone() or (0, None)
    runs = conn.execute("SELECT COUNT(*) FROM train_runs_observed").fetchone() or (0,)
    positions = conn.execute("SELECT COUNT(*) FROM train_positions_raw").fetchone() or (0,)
    bus = conn.execute("SELECT COUNT(*) FROM bus_predictions_raw").fetchone() or (0,)
    metra = conn.execute("SELECT COUNT(*) FROM metra_arrivals_raw").fetchone() or (0,)
    intercampus = conn.execute("SELECT COUNT(*) FROM intercampus_arrivals_raw").fetchone() or (0,)
    queue = conn.execute(
        """
        SELECT
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END),
            SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END),
            SUM(CASE WHEN status = 'unresolvable' THEN 1 ELSE 0 END),
            MIN(CASE WHEN status = 'pending' THEN enqueued_at END)
          FROM forecast_queue
        """
    ).fetchone() or (0, 0, 0, None)
    coverage = conn.execute(
        "SELECT AVG(CASE WHEN in_p80_window THEN 1 ELSE 0 END) FROM forecast_outcomes"
    ).fetchone()
    return Status(
        raw_arrivals_count=raw[0] or 0,
        runs_observed_count=runs[0] or 0,
        positions_count=positions[0] or 0,
        bus_predictions_count=bus[0] or 0,
        metra_arrivals_count=metra[0] or 0,
        intercampus_arrivals_count=intercampus[0] or 0,
        forecasts_pending=queue[0] or 0,
        forecasts_resolved=queue[1] or 0,
        forecasts_unresolvable=queue[2] or 0,
        latest_poll=raw[1],
        oldest_pending=queue[3],
        overall_p80_coverage=(coverage[0] if coverage else None),
    )
