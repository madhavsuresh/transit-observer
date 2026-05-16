"""Corridor-driven prediction entry point.

Public surface (unchanged for back-compat):

  - ``PREDICTOR_VERSION`` — default kernel version constant.
  - ``AdHocPrediction``    — dataclass returned by ``predict_for_od``.
  - ``CorpusPrediction``   — dataclass returned by ``predict_and_enqueue_corridor``.
  - ``predict_for_od``     — on-demand prediction (no DB write).
  - ``predict_and_enqueue_corridor`` — corridor scheduler entry point;
    writes a forecast_queue row.

Internally, both routes delegate to the predictor registry
(:mod:`transit_observer.predictors.registry`). The registry chooses
which predictor (kernel-v1 / kernel-v1+eb / gbm-v1) is active for the
corridor and returns a :class:`Prediction`. We write that prediction
into ``forecast_queue`` with the predictor's own ``predictor_version``
tag, so ``metrics.py`` can rank predictors against each other on the
same outcomes.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import duckdb

from .corridors import Corridor, mark_predicted
from .predictors import registry as pred_registry
from .predictors.journey_kernel import KERNEL_VERSION
from .predictors.protocol import Prediction


log = logging.getLogger(__name__)


# Kept for callers that imported it directly (api.py, tests).
PREDICTOR_VERSION = KERNEL_VERSION


@dataclass(frozen=True)
class CorpusPrediction:
    forecast_id: str
    corridor_id: str
    predictor_version: str
    wait_mean: float
    wait_p50: float
    wait_p80: float
    wait_p90: float
    in_vehicle_mean: float
    total_p80: float


@dataclass(frozen=True)
class AdHocPrediction:
    """On-demand prediction for a single (mode, line, boarding, alighting).

    Returned by ``predict_for_od``. Does **not** write to forecast_queue;
    use for live API queries where you want a prediction without
    enqueuing a graded forecast for every request.
    """

    mode: str
    line: str
    direction_code: str | None
    boarding_label: str
    alighting_label: str
    predicted_wait_mean: float
    predicted_wait_p50: float
    predicted_wait_p80: float
    predicted_wait_p90: float
    predicted_in_vehicle_mean: float
    predicted_total_p50: float
    predicted_total_p80: float
    predicted_total_p90: float
    predictor_version: str


def predict_for_od(
    conn: duckdb.DuckDBPyConnection,
    *,
    mode: str,
    line: str,
    boarding_int_id: int = 0,
    boarding_text_id: str | None = None,
    alighting_int_id: int = 0,
    alighting_text_id: str | None = None,
    now: datetime,
    predictor_version: str | None = None,
) -> AdHocPrediction | None:
    """Run the active predictor for an arbitrary OD pair without persisting.

    With ``predictor_version=None``, uses the kernel (no corridor =
    no rolling promotion context). Pass an explicit ``predictor_version``
    to force a specific predictor.
    """
    pseudo = Corridor(
        corridor_id="__adhoc__",
        mode=mode, line=line, direction="",
        origin_label="", origin_latitude=0.0, origin_longitude=0.0,
        destination_label="", destination_latitude=0.0, destination_longitude=0.0,
        boarding_int_id=boarding_int_id, boarding_text_id=boarding_text_id,
        alighting_int_id=alighting_int_id, alighting_text_id=alighting_text_id,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=99,
    )
    version = predictor_version or KERNEL_VERSION
    predictor = pred_registry.predictor(version) or pred_registry.predictor(KERNEL_VERSION)
    if predictor is None:
        return None
    prediction = predictor.predict(conn, pseudo, now=now)
    if prediction is None:
        return None
    boarding_label = prediction.feature_snapshot.get("boarding_station_name") or ""
    alighting_label = prediction.feature_snapshot.get("alighting_station_name") or ""
    direction_code = prediction.feature_snapshot.get("direction_code")
    total = prediction.total
    return AdHocPrediction(
        mode=mode, line=line,
        direction_code=direction_code,
        boarding_label=str(boarding_label or ""),
        alighting_label=str(alighting_label or ""),
        predicted_wait_mean=prediction.wait.mean,
        predicted_wait_p50=prediction.wait.p50,
        predicted_wait_p80=prediction.wait.p80,
        predicted_wait_p90=prediction.wait.p90,
        predicted_in_vehicle_mean=prediction.in_vehicle.mean,
        predicted_total_p50=total.p50,
        predicted_total_p80=total.p80,
        predicted_total_p90=total.p90,
        predictor_version=prediction.predictor_version,
    )


def predict_and_enqueue_corridor(
    conn: duckdb.DuckDBPyConnection,
    corridor: Corridor,
    *,
    now: datetime,
) -> CorpusPrediction | None:
    """Issue one prediction for ``corridor`` at ``now`` and persist it.

    Returns ``None`` if the active predictor lacks data. ``last_predicted_at``
    is always advanced so a starved corridor doesn't hot-loop and starve
    every other one of poll budget.
    """
    mark_predicted(conn, corridor_id=corridor.corridor_id, at=now)

    predictor = pred_registry.active_predictor_for(conn, corridor=corridor)
    prediction = predictor.predict(conn, corridor, now=now)
    if prediction is None:
        # Active predictor declined (e.g., GBM lacked features). Fall
        # back to the kernel so the corridor at least logs a kernel row.
        kernel = pred_registry.predictor(KERNEL_VERSION)
        prediction = kernel.predict(conn, corridor, now=now) if kernel else None
    if prediction is None:
        return None

    forecast_id = str(uuid.uuid4())
    snap = dict(prediction.feature_snapshot)
    snap.setdefault("boarding_station_name", _resolve_boarding_label(corridor, snap))
    snap.setdefault("alighting_station_name", _resolve_alighting_label(corridor, snap))
    _insert_forecast(
        conn,
        forecast_id=forecast_id,
        corridor=corridor,
        prediction=prediction,
        now=now,
        feature_json=json.dumps(snap, default=str),
    )

    row = conn.execute(
        """
        SELECT predicted_wait_mean, predicted_wait_p50, predicted_wait_p80, predicted_wait_p90,
               predicted_in_vehicle_mean, predicted_total_p80
          FROM forecast_queue
         WHERE forecast_id = ?
        """,
        [forecast_id],
    ).fetchone()
    if row is None:
        return None
    return CorpusPrediction(
        forecast_id=forecast_id,
        corridor_id=corridor.corridor_id,
        predictor_version=prediction.predictor_version,
        wait_mean=row[0], wait_p50=row[1], wait_p80=row[2], wait_p90=row[3],
        in_vehicle_mean=row[4], total_p80=row[5],
    )


def _resolve_boarding_label(corridor: Corridor, snap: dict) -> str:
    return snap.get("boarding_label") or corridor.origin_label or ""


def _resolve_alighting_label(corridor: Corridor, snap: dict) -> str:
    return snap.get("alighting_label") or corridor.destination_label or ""


def _insert_forecast(
    conn: duckdb.DuckDBPyConnection,
    *,
    forecast_id: str,
    corridor: Corridor,
    prediction: Prediction,
    now: datetime,
    feature_json: str,
) -> None:
    wait = prediction.wait
    in_vehicle = prediction.in_vehicle
    total = prediction.total
    leave_at = now  # corridor scheduler always asks "now"
    resolve_after = leave_at + timedelta(seconds=total.p90 + 10 * 60)
    snap = prediction.feature_snapshot or {}
    mode = snap.get("mode", corridor.mode)
    line = snap.get("line", corridor.line)
    direction_code = snap.get("direction_code", corridor.direction)
    boarding_map_id = int(snap.get("boarding_map_id") or corridor.boarding_int_id or 0)
    alighting_map_id = int(snap.get("alighting_map_id") or corridor.alighting_int_id or 0)
    boarding_text_id = snap.get("boarding_text_id") or corridor.boarding_text_id
    alighting_text_id = snap.get("alighting_text_id") or corridor.alighting_text_id
    boarding_name = snap.get("boarding_station_name")
    alighting_name = snap.get("alighting_station_name")

    conn.execute(
        """
        INSERT INTO forecast_queue (
            forecast_id, enqueued_at, snapshot_polled_at, leave_at,
            mode, line, direction_code,
            corridor_id, predictor_version, feature_json,
            boarding_map_id, boarding_text_id, boarding_station_name,
            alighting_map_id, alighting_text_id, alighting_station_name,
            predicted_wait_mean, predicted_wait_p50, predicted_wait_p80, predicted_wait_p90,
            predicted_in_vehicle_mean,
            predicted_total_mean, predicted_total_p50, predicted_total_p80, predicted_total_p90,
            predicted_failure_prob, resolve_after, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        """,
        [
            forecast_id, now, now, leave_at,
            mode, line, direction_code,
            corridor.corridor_id, prediction.predictor_version, feature_json,
            boarding_map_id, boarding_text_id, boarding_name,
            alighting_map_id, alighting_text_id, alighting_name,
            wait.mean, wait.p50, wait.p80, wait.p90,
            in_vehicle.mean,
            total.mean, total.p50, total.p80, total.p90,
            0.0, resolve_after,
        ],
    )
