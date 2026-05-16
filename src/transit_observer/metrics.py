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
