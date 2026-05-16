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
from .predictors import conformal
from .predictors.journey_kernel import (
    KERNEL_EB_VERSION,
    update_eb_state,
)
from .predictors.quantile_gbm import GBM_VERSION

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
    predicted_wait_p50: float | None = None
    predicted_wait_p80: float | None = None
    predicted_wait_p90: float | None = None
    predictor_version: str | None = None


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
               resolve_after,
               predicted_wait_p50, predicted_wait_p80, predicted_wait_p90,
               predictor_version
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
        truth_conf = _truth_confidence(conn, forecast, outcome)
        conn.execute(
            """
            INSERT INTO forecast_outcomes (
                forecast_id, resolved_at, boarded_run_number,
                boarded_at, alighted_at,
                actual_wait_seconds, actual_in_vehicle_seconds, actual_total_seconds,
                in_p80_window, in_p90_window,
                p50_residual_seconds, p80_residual_seconds,
                truth_confidence, failed, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                truth_conf,
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
        try:
            _update_predictor_state(conn, forecast, outcome, now=now)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "predictor_state.error",
                forecast_id=forecast.forecast_id, err=str(exc),
            )
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


# Truth confidence -----------------------------------------------------
#
# How cleanly do the raw feed samples bracket the boarded run? More
# samples near boarded_at and alighted_at => higher confidence in the
# *truth* (independent of the prediction). Used by metrics to weight or
# exclude noisy outcomes from headline accuracy figures.
#
# Heuristic per mode: count raw rows in a ±10-min window around each
# endpoint, weighted by `is_approaching` when the feed has that signal
# (L, bus). Return a value in [0, 1] capped at 1.0 once we have enough
# evidence at both endpoints.

_TRUTH_WINDOW_SECONDS = 600.0
_TRUTH_SCORE_CAP = 4.0  # 2 approaching + 2 dropoff samples at each endpoint


def _truth_confidence(
    conn: duckdb.DuckDBPyConnection, f: _Forecast, outcome: dict,
) -> float:
    boarded_at = outcome.get("boarded_at")
    alighted_at = outcome.get("alighted_at")
    if boarded_at is None or alighted_at is None:
        return 0.0
    board_signal = _endpoint_signal(conn, f, ts=boarded_at, side="boarding")
    alight_signal = _endpoint_signal(conn, f, ts=alighted_at, side="alighting")
    if board_signal is None or alight_signal is None:
        return 0.0
    score = min(board_signal, alight_signal) / _TRUTH_SCORE_CAP
    return max(0.0, min(1.0, score))


def _endpoint_signal(
    conn: duckdb.DuckDBPyConnection, f: _Forecast, *, ts: datetime, side: str,
) -> float | None:
    """Sample-density score at one endpoint of the boarded run.

    Approaching samples count 1.5x, scheduled-only rows 0.5x, plain
    arrivals 1.0x. Mode-specific because each feed has different signals.
    """
    lo = ts - timedelta(seconds=_TRUTH_WINDOW_SECONDS)
    hi = ts + timedelta(seconds=_TRUTH_WINDOW_SECONDS)

    if f.mode == "L":
        map_id = f.boarding_map_id if side == "boarding" else f.alighting_map_id
        rows = conn.execute(
            """
            SELECT COUNT(*) FILTER (WHERE is_approaching = TRUE),
                   COUNT(*) FILTER (WHERE COALESCE(is_approaching, FALSE) = FALSE
                                    AND COALESCE(is_scheduled, FALSE) = FALSE),
                   COUNT(*) FILTER (WHERE COALESCE(is_scheduled, FALSE) = TRUE)
              FROM train_arrivals_raw
             WHERE line = ? AND map_id = ?
               AND arrival_at BETWEEN ? AND ?
            """,
            [f.line, map_id, lo, hi],
        ).fetchone()
        if rows is None:
            return 0.0
        approaching, plain, scheduled = (r or 0 for r in rows)
        return 1.5 * approaching + 1.0 * plain + 0.5 * scheduled

    if f.mode == "bus":
        stop_id = int(f.boarding_text_id if side == "boarding" else (f.alighting_text_id or 0) or 0)
        rows = conn.execute(
            """
            SELECT COUNT(*) FILTER (WHERE is_approaching = TRUE),
                   COUNT(*) FILTER (WHERE COALESCE(is_approaching, FALSE) = FALSE)
              FROM bus_predictions_raw
             WHERE route = ? AND stop_id = ?
               AND arrival_at BETWEEN ? AND ?
            """,
            [f.line, stop_id, lo, hi],
        ).fetchone()
        if rows is None:
            return 0.0
        approaching, plain = (r or 0 for r in rows)
        return 1.5 * approaching + 1.0 * plain

    if f.mode == "metra":
        station_id = f.boarding_text_id if side == "boarding" else f.alighting_text_id
        if station_id is None:
            return 0.0
        n = conn.execute(
            """
            SELECT COUNT(*)
              FROM metra_arrivals_raw
             WHERE route_id = ? AND station_id = ?
               AND predicted_at BETWEEN ? AND ?
            """,
            [f.line, station_id, lo, hi],
        ).fetchone()
        return float((n[0] if n else 0) or 0)

    if f.mode == "intercampus":
        stop_id = f.boarding_text_id if side == "boarding" else f.alighting_text_id
        if stop_id is None:
            return 0.0
        n = conn.execute(
            """
            SELECT COUNT(*)
              FROM intercampus_arrivals_raw
             WHERE stop_id = ?
               AND predicted_at BETWEEN ? AND ?
            """,
            [stop_id, lo, hi],
        ).fetchone()
        return float((n[0] if n else 0) or 0)

    return None


def _update_predictor_state(
    conn: duckdb.DuckDBPyConnection,
    f: _Forecast,
    outcome: dict,
    *,
    now: datetime,
) -> None:
    """After scoring an outcome, update the EB / DtACI state for the
    relevant predictor. No-op for predictor versions we don't manage
    online state for (e.g. the parity-pure kernel-v1)."""
    if f.predictor_version is None or f.mode != "L":
        return
    actual_wait = outcome.get("actual_wait_seconds")
    if actual_wait is None:
        return
    line = f.line or ""
    direction = f.direction_code or ""
    if f.predictor_version == KERNEL_EB_VERSION and f.predicted_wait_p50 is not None:
        residual = float(actual_wait) - float(f.predicted_wait_p50)
        update_eb_state(
            conn,
            line=line, direction_code=direction,
            residual_seconds=residual, now=now,
        )
    if f.predictor_version == GBM_VERSION or f.predictor_version.startswith("gbm-"):
        # Update DtACI offsets for whichever quantiles the schema stores.
        for quantile, raw in (
            (0.8, f.predicted_wait_p80),
            (0.9, f.predicted_wait_p90),
        ):
            if raw is None:
                continue
            conformal.update(
                conn,
                predictor_version=f.predictor_version,
                line=line, direction_code=direction,
                leg="wait", quantile=quantile,
                raw_quantile_seconds=float(raw),
                observed_seconds=float(actual_wait),
                now=now,
            )
        # Also DtACI on the total leg using the predicted_total_p* fields
        # (carried separately on the forecast row).
        actual_total = outcome.get("actual_total_seconds")
        if actual_total is not None:
            for quantile, raw in (
                (0.8, f.predicted_total_p80),
                (0.9, f.predicted_total_p90),
            ):
                if raw is None:
                    continue
                conformal.update(
                    conn,
                    predictor_version=f.predictor_version,
                    line=line, direction_code=direction,
                    leg="total", quantile=quantile,
                    raw_quantile_seconds=float(raw),
                    observed_seconds=float(actual_total),
                    now=now,
                )


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
