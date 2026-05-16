"""Corridor-driven prediction entry point and feature snapshotting.

This is the corpus side of the prediction loop: take a ``Corridor``, adapt
it to the existing per-mode ``predict_*`` functions, capture the feature
vector at prediction time, and write the forecast row tagged with
``corridor_id`` + ``predictor_version``. The resolver later grades these
rows the same way it grades the legacy random forecasts.

Feature vector: small JSON snapshot of what the predictor saw at prediction
time -- enough to reconstruct the inputs for retrospective replay against
a future predictor. We intentionally do not dump the full feed; only the
quantities the current predictor actually consumes.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import duckdb

from .bus_predictor import BusTripSpec, predict_bus_trip
from .catalog import (
    BusStop,
    IntercampusStop,
    LStation,
    MetraStation,
    bus_by_id,
    intercampus_by_id,
    load_bus_catalog,
    load_catalog,
    load_intercampus_catalog,
    load_metra_catalog,
    metra_by_id,
)
from .corridors import Corridor, mark_predicted
from .intercampus_predictor import IntercampusTripSpec, predict_intercampus_trip
from .journey.time_distribution import TimeDistributionSummary
from .metra_predictor import MetraTripSpec, predict_metra_trip
from .trip_generator import (
    CATALOG_LINE_TO_API_CODE,
    TripSpec,
    direction_label,
    predict_trip,
)


log = logging.getLogger(__name__)


PREDICTOR_VERSION = "kernel-v1"  # bump when the predictor changes shape


_API_TO_CATALOG: dict[str, str] = {v: k for k, v in CATALOG_LINE_TO_API_CODE.items()}


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


@dataclass
class _Catalogs:
    l_by_map: dict[int, LStation]
    bus_by_id: dict[tuple[str, int], BusStop]
    metra_by_id: dict[str, MetraStation]
    intercampus_by_id: dict[str, IntercampusStop]


_CATALOGS: _Catalogs | None = None


def _catalogs() -> _Catalogs:
    global _CATALOGS
    if _CATALOGS is None:
        l_cat = load_catalog()
        _CATALOGS = _Catalogs(
            l_by_map={s.map_id: s for s in l_cat},
            bus_by_id=bus_by_id(load_bus_catalog()),
            metra_by_id=metra_by_id(load_metra_catalog()),
            intercampus_by_id=intercampus_by_id(load_intercampus_catalog()),
        )
    return _CATALOGS


@dataclass(frozen=True)
class AdHocPrediction:
    """On-demand prediction for a single (mode, line, boarding, alighting).

    Returned by ``predict_for_od``. Does **not** write to forecast_queue --
    use this for live API queries where we want a prediction but don't
    want to enqueue a graded forecast for every query.
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
) -> AdHocPrediction | None:
    """Run the per-mode predictor for an arbitrary OD pair and return the
    prediction without persisting anything to forecast_queue.

    The caller is responsible for logging the query separately if desired
    (see ``query_log.append_query``).
    """
    cats = _catalogs()
    pseudo = Corridor(
        corridor_id="__adhoc__",
        mode=mode,
        line=line,
        direction="",
        origin_label="", origin_latitude=0.0, origin_longitude=0.0,
        destination_label="", destination_latitude=0.0, destination_longitude=0.0,
        boarding_int_id=boarding_int_id, boarding_text_id=boarding_text_id,
        alighting_int_id=alighting_int_id, alighting_text_id=alighting_text_id,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=99,
    )
    out = _predict(conn, pseudo, cats, now)
    if out is None:
        return None

    if mode == "L":
        spec, wait_forecast, in_vehicle = out
        wait = wait_forecast.wait_distribution
        boarding_label, alighting_label = spec.boarding.name, spec.alighting.name
        direction = spec.direction_label
    elif mode == "bus":
        spec, wait_forecast, in_vehicle = out
        wait = wait_forecast.wait_distribution
        boarding_label, alighting_label = spec.boarding.name, spec.alighting.name
        direction = spec.boarding.direction_label
    elif mode == "metra":
        spec, wait, in_vehicle, direction_id = out
        boarding_label, alighting_label = spec.boarding.name, spec.alighting.name
        direction = str(direction_id) if direction_id is not None else None
    elif mode == "intercampus":
        spec, wait, in_vehicle, direction = out
        boarding_label, alighting_label = spec.boarding.name, spec.alighting.name
    else:
        return None

    return AdHocPrediction(
        mode=mode,
        line=line,
        direction_code=direction,
        boarding_label=boarding_label,
        alighting_label=alighting_label,
        predicted_wait_mean=wait.mean,
        predicted_wait_p50=wait.p50,
        predicted_wait_p80=wait.p80,
        predicted_wait_p90=wait.p90,
        predicted_in_vehicle_mean=in_vehicle.mean,
        predicted_total_p50=wait.p50 + in_vehicle.p50,
        predicted_total_p80=wait.p80 + in_vehicle.p80,
        predicted_total_p90=wait.p90 + in_vehicle.p90,
        predictor_version=PREDICTOR_VERSION,
    )


def predict_and_enqueue_corridor(
    conn: duckdb.DuckDBPyConnection,
    corridor: Corridor,
    *,
    now: datetime,
) -> CorpusPrediction | None:
    """Issue one prediction for ``corridor`` at ``now`` and persist it.

    Returns ``None`` if the per-mode predictor lacks the data to produce a
    forecast (e.g. no recent arrivals at the boarding stop). The corridor's
    ``last_predicted_at`` is always advanced so a starved corridor doesn't
    hot-loop and starve every other one of poll budget.
    """
    mark_predicted(conn, corridor_id=corridor.corridor_id, at=now)

    cats = _catalogs()
    spec_and_pred = _predict(conn, corridor, cats, now)
    if spec_and_pred is None:
        return None

    forecast_id = str(uuid.uuid4())
    mode = corridor.mode

    if mode == "L":
        spec, wait_forecast, in_vehicle = spec_and_pred
        wait = wait_forecast.wait_distribution
        feature_json = _feature_snapshot_l(conn, spec, now)
        _insert_forecast(
            conn,
            forecast_id=forecast_id,
            mode="L", line=spec.line_api, direction_code=spec.direction_label,
            corridor=corridor, now=now, leave_at=spec.leave_at,
            boarding_map_id=spec.boarding.map_id, boarding_text_id=None,
            boarding_name=spec.boarding.name,
            alighting_map_id=spec.alighting.map_id, alighting_text_id=None,
            alighting_name=spec.alighting.name,
            wait=wait, in_vehicle=in_vehicle, feature_json=feature_json,
        )
    elif mode == "bus":
        spec, wait_forecast, in_vehicle = spec_and_pred
        wait = wait_forecast.wait_distribution
        feature_json = _feature_snapshot_bus(conn, spec, now)
        _insert_forecast(
            conn,
            forecast_id=forecast_id,
            mode="bus", line=spec.route,
            direction_code=spec.boarding.direction_label,
            corridor=corridor, now=now, leave_at=spec.leave_at,
            boarding_map_id=0, boarding_text_id=str(spec.boarding.stop_id),
            boarding_name=spec.boarding.name,
            alighting_map_id=0, alighting_text_id=str(spec.alighting.stop_id),
            alighting_name=spec.alighting.name,
            wait=wait, in_vehicle=in_vehicle, feature_json=feature_json,
        )
    elif mode == "metra":
        spec, wait, in_vehicle, direction_id = spec_and_pred
        feature_json = _feature_snapshot_metra(conn, spec, now, direction_id)
        _insert_forecast(
            conn,
            forecast_id=forecast_id,
            mode="metra", line=spec.route_id,
            direction_code=str(direction_id) if direction_id is not None else None,
            corridor=corridor, now=now, leave_at=spec.leave_at,
            boarding_map_id=0, boarding_text_id=spec.boarding.station_id,
            boarding_name=spec.boarding.name,
            alighting_map_id=0, alighting_text_id=spec.alighting.station_id,
            alighting_name=spec.alighting.name,
            wait=wait, in_vehicle=in_vehicle, feature_json=feature_json,
        )
    elif mode == "intercampus":
        spec, wait, in_vehicle, direction = spec_and_pred
        feature_json = _feature_snapshot_intercampus(conn, spec, now)
        _insert_forecast(
            conn,
            forecast_id=forecast_id,
            mode="intercampus", line="intercampus", direction_code=direction,
            corridor=corridor, now=now, leave_at=spec.leave_at,
            boarding_map_id=0, boarding_text_id=spec.boarding.stop_id,
            boarding_name=spec.boarding.name,
            alighting_map_id=0, alighting_text_id=spec.alighting.stop_id,
            alighting_name=spec.alighting.name,
            wait=wait, in_vehicle=in_vehicle, feature_json=feature_json,
        )
    else:
        return None

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
        predictor_version=PREDICTOR_VERSION,
        wait_mean=row[0], wait_p50=row[1], wait_p80=row[2], wait_p90=row[3],
        in_vehicle_mean=row[4], total_p80=row[5],
    )


# Dispatch ------------------------------------------------------------


def _predict(
    conn: duckdb.DuckDBPyConnection,
    corridor: Corridor,
    cats: _Catalogs,
    now: datetime,
) -> tuple | None:
    """Dispatch to the right per-mode predictor.

    Returns a per-mode tuple:
    - L:           (TripSpec, WaitForecast, in_vehicle TimeDistributionSummary)
    - bus:         (BusTripSpec, WaitForecast, in_vehicle)
    - metra:       (MetraTripSpec, wait TimeDistributionSummary, in_vehicle, direction_id)
    - intercampus: (IntercampusTripSpec, wait, in_vehicle, direction)
    or ``None`` if catalog lookup or prediction failed.
    """
    if corridor.mode == "L":
        boarding = cats.l_by_map.get(corridor.boarding_int_id)
        alighting = cats.l_by_map.get(corridor.alighting_int_id)
        if boarding is None or alighting is None:
            return None
        line_catalog = _API_TO_CATALOG.get(corridor.line, corridor.line.lower())
        spec = TripSpec(
            line_catalog=line_catalog, line_api=corridor.line,
            boarding=boarding, alighting=alighting,
            direction_label=corridor.direction or direction_label(boarding, alighting),
            leave_at=now,
        )
        out = predict_trip(conn, spec, now=now)
        if out is None:
            return None
        wait, in_vehicle = out
        return spec, wait, in_vehicle

    if corridor.mode == "bus":
        boarding = cats.bus_by_id.get((corridor.line, corridor.boarding_int_id))
        alighting = cats.bus_by_id.get((corridor.line, corridor.alighting_int_id))
        if boarding is None or alighting is None:
            return None
        spec = BusTripSpec(route=corridor.line, boarding=boarding, alighting=alighting, leave_at=now)
        out = predict_bus_trip(conn, spec, now=now)
        if out is None:
            return None
        wait, in_vehicle = out
        return spec, wait, in_vehicle

    if corridor.mode == "metra":
        boarding = cats.metra_by_id.get(corridor.boarding_text_id or "")
        alighting = cats.metra_by_id.get(corridor.alighting_text_id or "")
        if boarding is None or alighting is None:
            return None
        spec = MetraTripSpec(route_id=corridor.line, boarding=boarding, alighting=alighting, leave_at=now)
        out = predict_metra_trip(conn, spec, now=now)
        if out is None:
            return None
        wait, in_vehicle, direction_id = out
        return spec, wait, in_vehicle, direction_id

    if corridor.mode == "intercampus":
        boarding = cats.intercampus_by_id.get(corridor.boarding_text_id or "")
        alighting = cats.intercampus_by_id.get(corridor.alighting_text_id or "")
        if boarding is None or alighting is None:
            return None
        spec = IntercampusTripSpec(
            direction=corridor.direction, boarding=boarding, alighting=alighting, leave_at=now,
        )
        out = predict_intercampus_trip(conn, spec, now=now)
        if out is None:
            return None
        wait, in_vehicle, direction = out
        return spec, wait, in_vehicle, direction

    log.warning("corpus.skip", corridor=corridor.corridor_id, reason=f"unknown mode {corridor.mode}")
    return None


def _insert_forecast(
    conn: duckdb.DuckDBPyConnection,
    *,
    forecast_id: str,
    mode: str,
    line: str,
    direction_code: str | None,
    corridor: Corridor,
    now: datetime,
    leave_at: datetime,
    boarding_map_id: int,
    boarding_text_id: str | None,
    boarding_name: str | None,
    alighting_map_id: int,
    alighting_text_id: str | None,
    alighting_name: str | None,
    wait: TimeDistributionSummary,
    in_vehicle: TimeDistributionSummary,
    feature_json: str,
) -> None:
    total_mean = wait.mean + in_vehicle.mean
    total_p50 = wait.p50 + in_vehicle.p50
    total_p80 = wait.p80 + in_vehicle.p80
    total_p90 = wait.p90 + in_vehicle.p90
    resolve_after = leave_at + timedelta(seconds=total_p90 + 10 * 60)
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
            corridor.corridor_id, PREDICTOR_VERSION, feature_json,
            boarding_map_id, boarding_text_id, boarding_name,
            alighting_map_id, alighting_text_id, alighting_name,
            wait.mean, wait.p50, wait.p80, wait.p90,
            in_vehicle.mean,
            total_mean, total_p50, total_p80, total_p90,
            0.0, resolve_after,
        ],
    )


# Feature snapshots ---------------------------------------------------
#
# Goal: persist enough JSON to (a) understand which inputs drove the
# prediction at retrospective-debug time, and (b) feed a future predictor
# without re-fetching the feed. The full raw feed is already in *_raw
# tables; this captures the derived view the predictor actually used.


def _feature_snapshot_l(
    conn: duckdb.DuckDBPyConnection, spec: TripSpec, now: datetime,
) -> str:
    cutoff = spec.leave_at - timedelta(minutes=5)
    horizon = spec.leave_at + timedelta(minutes=30)
    rows = conn.execute(
        """
        SELECT arrival_at, is_approaching, is_scheduled
          FROM train_arrivals_raw
         WHERE line = ? AND map_id = ?
           AND polled_at >= ? AND arrival_at >= ? AND arrival_at <= ?
         ORDER BY arrival_at
         LIMIT 10
        """,
        [spec.line_api, spec.boarding.map_id, cutoff, spec.leave_at, horizon],
    ).fetchall()
    return json.dumps({
        "mode": "L",
        "line": spec.line_api,
        "boarding_map_id": spec.boarding.map_id,
        "alighting_map_id": spec.alighting.map_id,
        "haversine_meters": round(_haversine(
            spec.boarding.latitude, spec.boarding.longitude,
            spec.alighting.latitude, spec.alighting.longitude,
        ), 1),
        "live_departures": [
            {
                "arrival_at": arrival_at.isoformat() if arrival_at else None,
                "is_approaching": bool(is_app),
                "is_scheduled": bool(is_sched) if is_sched is not None else None,
            }
            for arrival_at, is_app, is_sched in rows
        ],
        "hour_of_day": now.hour,
        "minute_of_hour": now.minute,
        "dow": now.weekday(),
    })


def _feature_snapshot_bus(
    conn: duckdb.DuckDBPyConnection, spec: BusTripSpec, now: datetime,
) -> str:
    cutoff = spec.leave_at - timedelta(minutes=5)
    horizon = spec.leave_at + timedelta(minutes=45)
    rows = conn.execute(
        """
        SELECT arrival_at, is_approaching, direction_name
          FROM bus_predictions_raw
         WHERE route = ? AND stop_id = ?
           AND polled_at >= ? AND arrival_at >= ? AND arrival_at <= ?
         ORDER BY arrival_at
         LIMIT 10
        """,
        [spec.route, spec.boarding.stop_id, cutoff, spec.leave_at, horizon],
    ).fetchall()
    return json.dumps({
        "mode": "bus",
        "route": spec.route,
        "boarding_stop_id": spec.boarding.stop_id,
        "alighting_stop_id": spec.alighting.stop_id,
        "direction_label": spec.boarding.direction_label,
        "haversine_meters": round(_haversine(
            spec.boarding.latitude, spec.boarding.longitude,
            spec.alighting.latitude, spec.alighting.longitude,
        ), 1),
        "live_departures": [
            {
                "arrival_at": arrival_at.isoformat() if arrival_at else None,
                "is_approaching": bool(is_app),
                "direction_name": direction_name,
            }
            for arrival_at, is_app, direction_name in rows
        ],
        "hour_of_day": now.hour,
        "minute_of_hour": now.minute,
        "dow": now.weekday(),
    })


def _feature_snapshot_metra(
    conn: duckdb.DuckDBPyConnection, spec: MetraTripSpec, now: datetime, direction_id: int | None,
) -> str:
    rows = conn.execute(
        """
        WITH latest AS (
            SELECT trip_id, station_id, MAX(predicted_at) AS predicted_at
              FROM metra_arrivals_raw
             WHERE route_id = ?
               AND station_id IN (?, ?)
               AND predicted_at IS NOT NULL
             GROUP BY trip_id, station_id
        )
        SELECT b.trip_id, b.predicted_at, a.predicted_at
          FROM latest b
          JOIN latest a USING (trip_id)
         WHERE b.station_id = ?
           AND a.station_id = ?
         ORDER BY b.predicted_at
         LIMIT 6
        """,
        [
            spec.route_id, spec.boarding.station_id, spec.alighting.station_id,
            spec.boarding.station_id, spec.alighting.station_id,
        ],
    ).fetchall()
    return json.dumps({
        "mode": "metra",
        "route_id": spec.route_id,
        "boarding_station_id": spec.boarding.station_id,
        "alighting_station_id": spec.alighting.station_id,
        "direction_id": direction_id,
        "viable_trips": [
            {
                "trip_id": trip_id,
                "boarding_predicted_at": b.isoformat() if b else None,
                "alighting_predicted_at": a.isoformat() if a else None,
            }
            for trip_id, b, a in rows
        ],
        "hour_of_day": now.hour,
        "minute_of_hour": now.minute,
        "dow": now.weekday(),
    })


def _feature_snapshot_intercampus(
    conn: duckdb.DuckDBPyConnection, spec: IntercampusTripSpec, now: datetime,
) -> str:
    rows = conn.execute(
        """
        WITH latest AS (
            SELECT trip_id, stop_id, MAX(predicted_at) AS predicted_at
              FROM intercampus_arrivals_raw
             WHERE stop_id IN (?, ?)
               AND predicted_at IS NOT NULL
             GROUP BY trip_id, stop_id
        )
        SELECT b.trip_id, b.predicted_at, a.predicted_at
          FROM latest b
          JOIN latest a USING (trip_id)
         WHERE b.stop_id = ?
           AND a.stop_id = ?
         ORDER BY b.predicted_at
         LIMIT 6
        """,
        [
            spec.boarding.stop_id, spec.alighting.stop_id,
            spec.boarding.stop_id, spec.alighting.stop_id,
        ],
    ).fetchall()
    return json.dumps({
        "mode": "intercampus",
        "boarding_stop_id": spec.boarding.stop_id,
        "alighting_stop_id": spec.alighting.stop_id,
        "direction": spec.direction,
        "viable_trips": [
            {
                "trip_id": trip_id,
                "boarding_predicted_at": b.isoformat() if b else None,
                "alighting_predicted_at": a.isoformat() if a else None,
            }
            for trip_id, b, a in rows
        ],
        "hour_of_day": now.hour,
        "minute_of_hour": now.minute,
        "dow": now.weekday(),
    })


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))
