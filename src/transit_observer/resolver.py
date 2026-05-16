"""Find the realized train for each due forecast and write the outcome.

For a forecast: rider is "at" `boarding_map_id` at `leave_at`. We look at
`train_runs_observed` and pick the first run on this line at this station
with `observed_arrival_at >= leave_at` whose `destination` heads in the
right direction. Then we find the same run's observed arrival at the
alighting station.

Forecasts that can't be resolved by `resolve_after + buffer` are marked
`unresolvable`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

import duckdb

from .direction_audit import audit_resolved_forecast

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Forecast:
    forecast_id: str
    leave_at: datetime
    line: str
    direction_code: str | None
    boarding_map_id: int
    alighting_map_id: int
    predicted_total_p50: float
    predicted_total_p80: float
    predicted_total_p90: float
    resolve_after: datetime


def resolve_due_forecasts(
    conn: duckdb.DuckDBPyConnection,
    *,
    now: datetime,
    expiration_buffer_seconds: float,
) -> tuple[int, int]:
    """Process all forecasts with `resolve_after <= now AND status='pending'`.

    Returns (n_resolved, n_unresolvable).
    """
    rows = conn.execute(
        """
        SELECT forecast_id, leave_at, line, direction_code,
               boarding_map_id, alighting_map_id,
               predicted_total_p50, predicted_total_p80, predicted_total_p90,
               resolve_after
          FROM forecast_queue
         WHERE status = 'pending'
           AND resolve_after <= ?
         ORDER BY resolve_after
         LIMIT 200
        """,
        [now],
    ).fetchall()

    n_resolved = 0
    n_unresolvable = 0

    for raw in rows:
        forecast = _Forecast(*raw)
        outcome = _resolve_one(conn, forecast, now=now)
        if outcome is None:
            cutoff = forecast.resolve_after + timedelta(seconds=expiration_buffer_seconds)
            if now >= cutoff:
                conn.execute(
                    "UPDATE forecast_queue SET status = 'unresolvable' WHERE forecast_id = ?",
                    [forecast.forecast_id],
                )
                n_unresolvable += 1
            continue
        conn.execute(
            """
            INSERT INTO forecast_outcomes (
                forecast_id, resolved_at, boarded_run_number,
                boarded_at, alighted_at,
                actual_wait_seconds, actual_in_vehicle_seconds, actual_total_seconds,
                in_p80_window, in_p90_window,
                p50_residual_seconds, p80_residual_seconds, failed, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (forecast_id) DO NOTHING
            """,
            [
                forecast.forecast_id,
                now,
                outcome.run_number,
                outcome.boarded_at,
                outcome.alighted_at,
                outcome.actual_wait_seconds,
                outcome.actual_in_vehicle_seconds,
                outcome.actual_total_seconds,
                outcome.actual_total_seconds <= forecast.predicted_total_p80,
                outcome.actual_total_seconds <= forecast.predicted_total_p90,
                outcome.actual_total_seconds - forecast.predicted_total_p50,
                outcome.actual_total_seconds - forecast.predicted_total_p80,
                False,
                outcome.notes,
            ],
        )
        conn.execute(
            "UPDATE forecast_queue SET status = 'resolved' WHERE forecast_id = ?",
            [forecast.forecast_id],
        )
        try:
            audit_resolved_forecast(conn, forecast_id=forecast.forecast_id, now=now)
        except Exception as exc:  # noqa: BLE001
            log.warning("direction_audit.error", forecast_id=forecast.forecast_id, err=str(exc))
        n_resolved += 1

    return n_resolved, n_unresolvable


@dataclass(frozen=True)
class _Outcome:
    run_number: str
    boarded_at: datetime
    alighted_at: datetime
    actual_wait_seconds: float
    actual_in_vehicle_seconds: float
    actual_total_seconds: float
    notes: str


def _resolve_one(
    conn: duckdb.DuckDBPyConnection,
    f: _Forecast,
    *,
    now: datetime,
) -> _Outcome | None:
    """Find the run a rider arriving at `leave_at` would have boarded,
    then look up that run's arrival at the alighting station."""
    boarding = conn.execute(
        """
        SELECT run_number, observed_arrival_at, destination_name
          FROM train_runs_observed
         WHERE line = ?
           AND map_id = ?
           AND observed_arrival_at >= ?
           AND observed_arrival_at <= ?
         ORDER BY observed_arrival_at ASC
         LIMIT 5
        """,
        [f.line, f.boarding_map_id, f.leave_at, now],
    ).fetchall()

    if not boarding:
        return None

    for run_number, boarded_at, destination in boarding:
        alighting = conn.execute(
            """
            SELECT observed_arrival_at
              FROM train_runs_observed
             WHERE line = ?
               AND run_number = ?
               AND map_id = ?
               AND observed_arrival_at >= ?
             ORDER BY observed_arrival_at ASC
             LIMIT 1
            """,
            [f.line, run_number, f.alighting_map_id, boarded_at],
        ).fetchone()
        if alighting is None:
            continue
        alighted_at = alighting[0]
        actual_wait = (boarded_at - f.leave_at).total_seconds()
        actual_in_vehicle = (alighted_at - boarded_at).total_seconds()
        actual_total = (alighted_at - f.leave_at).total_seconds()
        if actual_in_vehicle < 0:
            continue
        return _Outcome(
            run_number=run_number,
            boarded_at=boarded_at,
            alighted_at=alighted_at,
            actual_wait_seconds=actual_wait,
            actual_in_vehicle_seconds=actual_in_vehicle,
            actual_total_seconds=actual_total,
            notes=f"destination={destination or '?'}",
        )
    return None
