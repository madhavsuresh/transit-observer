"""Find the realized run for each due forecast across all modes.

Dispatches per `forecast_queue.mode`:
- L: existing logic against `train_runs_observed`
- bus: against `bus_runs_observed`, keyed by vehicle_id
- metra: against `metra_trips_observed`, keyed by trip_id
- intercampus: against `intercampus_trips_observed`, keyed by trip_id

Forecasts that can't be resolved by `resolve_after + buffer` are marked
`unresolvable`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

import duckdb

from .bus_predictor import resolve_bus_forecast
from .direction_audit import audit_resolved_forecast
from .intercampus_predictor import resolve_intercampus_forecast
from .metra_predictor import resolve_metra_forecast

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Forecast:
    forecast_id: str
    leave_at: datetime
    mode: str
    line: str
    direction_code: str | None
    boarding_map_id: int
    boarding_text_id: str | None
    alighting_map_id: int
    alighting_text_id: str | None
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
        SELECT forecast_id, leave_at, mode, line, direction_code,
               boarding_map_id, boarding_text_id,
               alighting_map_id, alighting_text_id,
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
        outcome = _resolve_mode(conn, forecast, now=now)
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
                outcome["boarded_run_number"],
                outcome["boarded_at"],
                outcome["alighted_at"],
                outcome["actual_wait_seconds"],
                outcome["actual_in_vehicle_seconds"],
                outcome["actual_total_seconds"],
                outcome["actual_total_seconds"] <= forecast.predicted_total_p80,
                outcome["actual_total_seconds"] <= forecast.predicted_total_p90,
                outcome["actual_total_seconds"] - forecast.predicted_total_p50,
                outcome["actual_total_seconds"] - forecast.predicted_total_p80,
                False,
                outcome.get("notes"),
            ],
        )
        conn.execute(
            "UPDATE forecast_queue SET status = 'resolved' WHERE forecast_id = ?",
            [forecast.forecast_id],
        )
        try:
            audit_resolved_forecast(
                conn,
                forecast_id=forecast.forecast_id,
                now=now,
                mode=forecast.mode,
                outcome=outcome,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("direction_audit.error", forecast_id=forecast.forecast_id, err=str(exc))
        n_resolved += 1

    return n_resolved, n_unresolvable


def _resolve_mode(conn: duckdb.DuckDBPyConnection, f: _Forecast, *, now: datetime) -> dict | None:
    if f.mode == "L":
        return _resolve_l(conn, f, now=now)
    if f.mode == "bus":
        if f.boarding_text_id is None or f.alighting_text_id is None:
            return None
        return resolve_bus_forecast(
            conn,
            forecast_id=f.forecast_id,
            leave_at=f.leave_at,
            route=f.line,
            boarding_stop_id=int(f.boarding_text_id),
            alighting_stop_id=int(f.alighting_text_id),
            now=now,
        )
    if f.mode == "metra":
        if f.boarding_text_id is None or f.alighting_text_id is None:
            return None
        return resolve_metra_forecast(
            conn,
            forecast_id=f.forecast_id,
            leave_at=f.leave_at,
            route_id=f.line,
            boarding_station_id=f.boarding_text_id,
            alighting_station_id=f.alighting_text_id,
            now=now,
        )
    if f.mode == "intercampus":
        if f.boarding_text_id is None or f.alighting_text_id is None:
            return None
        return resolve_intercampus_forecast(
            conn,
            forecast_id=f.forecast_id,
            leave_at=f.leave_at,
            boarding_stop_id=f.boarding_text_id,
            alighting_stop_id=f.alighting_text_id,
            now=now,
        )
    log.warning("resolver.unknown_mode", mode=f.mode)
    return None


def _resolve_l(conn: duckdb.DuckDBPyConnection, f: _Forecast, *, now: datetime) -> dict | None:
    boarding = conn.execute(
        """
        SELECT run_number, observed_arrival_at, destination_name, direction_code
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

    for run_number, boarded_at, destination, direction in boarding:
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
        if actual_in_vehicle < 0:
            continue
        return {
            "boarded_run_number": run_number,
            "boarded_at": boarded_at,
            "alighted_at": alighted_at,
            "actual_wait_seconds": actual_wait,
            "actual_in_vehicle_seconds": actual_in_vehicle,
            "actual_total_seconds": (alighted_at - f.leave_at).total_seconds(),
            "boarded_direction_code": direction,
            "boarded_destination_name": destination,
            "notes": f"destination={destination or '?'}",
        }
    return None
