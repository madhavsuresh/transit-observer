"""Adapters that present the existing per-mode heuristic kernels as
``Predictor``s.

Two variants:

- ``JourneyKernelPredictor``: parity-pure. Wraps the per-mode
  ``predict_*`` functions verbatim. ``predictor_version = "kernel-v1"``.
- ``JourneyKernelEBPredictor``: kernel + empirical-Bayes residual mean
  shift per (line, direction). Reads the running residual mean from
  ``predictor_state``. ``predictor_version = "kernel-v1+eb"``.

Both implement the same ``Predictor`` protocol so the registry can swap
between them without callers caring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import duckdb

from ..bus_predictor import BusTripSpec, predict_bus_trip
from ..catalog import (
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
from ..corridors import Corridor
from ..intercampus_predictor import IntercampusTripSpec, predict_intercampus_trip
from ..journey.stop_arrival import WaitForecast
from ..journey.time_distribution import TimeDistributionSummary
from ..metra_predictor import MetraTripSpec, predict_metra_trip
from ..trip_generator import (
    CATALOG_LINE_TO_API_CODE,
    TripSpec,
    direction_label,
    predict_trip,
)

from . import features as feats
from .protocol import Prediction, PredictionLeg


KERNEL_VERSION = "kernel-v1"
KERNEL_EB_VERSION = "kernel-v1+eb"


_API_TO_CATALOG: dict[str, str] = {v: k for k, v in CATALOG_LINE_TO_API_CODE.items()}


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


def reset_catalogs() -> None:
    """Test hook — drops the cached catalogs."""
    global _CATALOGS
    _CATALOGS = None


@dataclass(frozen=True)
class KernelDispatch:
    """Per-mode kernel output, unified into a single shape."""

    mode: str
    line: str
    direction_code: str | None
    boarding_label: str | None
    alighting_label: str | None
    boarding_map_id: int
    boarding_text_id: str | None
    alighting_map_id: int
    alighting_text_id: str | None
    leave_at: datetime
    wait: TimeDistributionSummary
    in_vehicle: TimeDistributionSummary
    wait_forecast: WaitForecast | None       # L/bus only — carries state + explanation
    feature_bundle: feats.FeatureBundle | None  # L only — features for the GBM
    legacy_snapshot: dict[str, Any] | None = None  # mode-specific live_departures / viable_trips


def _dispatch(
    conn: duckdb.DuckDBPyConnection,
    corridor: Corridor,
    *,
    now: datetime,
) -> KernelDispatch | None:
    """Run the per-mode kernel for a Corridor and return a unified result.

    Mirrors what ``corpus._predict`` does, but produces a single
    structured object rather than a per-mode tuple. Returns ``None`` if
    the kernel can't produce a forecast (no live data, catalog miss).
    """
    cats = _catalogs()
    mode = corridor.mode

    if mode == "L":
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
        wait_forecast, in_vehicle = out
        bundle = feats.extract_features_live_l(conn, spec, now=now)
        legacy = feature_snapshot_l(conn, spec, now)
        legacy["haversine_meters"] = bundle.values.get("haversine_meters", 0.0)
        return KernelDispatch(
            mode="L", line=spec.line_api, direction_code=spec.direction_label,
            boarding_label=spec.boarding.name, alighting_label=spec.alighting.name,
            boarding_map_id=spec.boarding.map_id, boarding_text_id=None,
            alighting_map_id=spec.alighting.map_id, alighting_text_id=None,
            leave_at=spec.leave_at,
            wait=wait_forecast.wait_distribution,
            in_vehicle=in_vehicle,
            wait_forecast=wait_forecast, feature_bundle=bundle,
            legacy_snapshot=legacy,
        )

    if mode == "bus":
        boarding = cats.bus_by_id.get((corridor.line, corridor.boarding_int_id))
        alighting = cats.bus_by_id.get((corridor.line, corridor.alighting_int_id))
        if boarding is None or alighting is None:
            return None
        spec = BusTripSpec(route=corridor.line, boarding=boarding, alighting=alighting, leave_at=now)
        out = predict_bus_trip(conn, spec, now=now)
        if out is None:
            return None
        wait_forecast, in_vehicle = out
        return KernelDispatch(
            mode="bus", line=spec.route, direction_code=spec.boarding.direction_label,
            boarding_label=spec.boarding.name, alighting_label=spec.alighting.name,
            boarding_map_id=0, boarding_text_id=str(spec.boarding.stop_id),
            alighting_map_id=0, alighting_text_id=str(spec.alighting.stop_id),
            leave_at=spec.leave_at,
            wait=wait_forecast.wait_distribution,
            in_vehicle=in_vehicle,
            wait_forecast=wait_forecast, feature_bundle=None,
            legacy_snapshot=feature_snapshot_bus(conn, spec, now),
        )

    if mode == "metra":
        boarding = cats.metra_by_id.get(corridor.boarding_text_id or "")
        alighting = cats.metra_by_id.get(corridor.alighting_text_id or "")
        if boarding is None or alighting is None:
            return None
        spec = MetraTripSpec(route_id=corridor.line, boarding=boarding, alighting=alighting, leave_at=now)
        out = predict_metra_trip(conn, spec, now=now)
        if out is None:
            return None
        wait, in_vehicle, direction_id = out
        return KernelDispatch(
            mode="metra", line=spec.route_id,
            direction_code=str(direction_id) if direction_id is not None else None,
            boarding_label=spec.boarding.name, alighting_label=spec.alighting.name,
            boarding_map_id=0, boarding_text_id=spec.boarding.station_id,
            alighting_map_id=0, alighting_text_id=spec.alighting.station_id,
            leave_at=spec.leave_at,
            wait=wait, in_vehicle=in_vehicle,
            wait_forecast=None, feature_bundle=None,
            legacy_snapshot=feature_snapshot_metra(conn, spec, now, direction_id),
        )

    if mode == "intercampus":
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
        return KernelDispatch(
            mode="intercampus", line="intercampus", direction_code=direction,
            boarding_label=spec.boarding.name, alighting_label=spec.alighting.name,
            boarding_map_id=0, boarding_text_id=spec.boarding.stop_id,
            alighting_map_id=0, alighting_text_id=spec.alighting.stop_id,
            leave_at=spec.leave_at,
            wait=wait, in_vehicle=in_vehicle,
            wait_forecast=None, feature_bundle=None,
            legacy_snapshot=feature_snapshot_intercampus(conn, spec, now),
        )

    return None


def _bundle_to_snapshot(d: KernelDispatch) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "mode": d.mode, "line": d.line, "direction_code": d.direction_code,
        "boarding_map_id": d.boarding_map_id, "alighting_map_id": d.alighting_map_id,
        "boarding_text_id": d.boarding_text_id, "alighting_text_id": d.alighting_text_id,
        "boarding_station_name": d.boarding_label,
        "alighting_station_name": d.alighting_label,
        "boarding_label": d.boarding_label,
        "alighting_label": d.alighting_label,
        "leave_at": d.leave_at.isoformat(),
    }
    if d.feature_bundle is not None:
        snap.update(d.feature_bundle.values)
        snap["feature_completeness"] = d.feature_bundle.completeness
    if d.wait_forecast is not None and d.wait_forecast.next_departure_at is not None:
        snap["next_departure_at"] = d.wait_forecast.next_departure_at.isoformat()
    if d.legacy_snapshot:
        snap.update(d.legacy_snapshot)
    # Legacy time fields kept for retrospective-debug compatibility.
    snap.setdefault("hour_of_day", d.leave_at.hour)
    snap.setdefault("minute_of_hour", d.leave_at.minute)
    snap.setdefault("dow", d.leave_at.weekday())
    if d.mode == "L":
        snap.setdefault("haversine_meters", 0.0)
    return snap


def feature_snapshot_l(
    conn: duckdb.DuckDBPyConnection, spec, now: datetime,
) -> dict[str, Any]:
    """Legacy live_departures snapshot for L mode.

    Retained for ``corpus._insert_forecast`` so the feature_json shape
    matches what downstream tests and the dashboard already expect.
    """
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
    return {
        "live_departures": [
            {
                "arrival_at": arrival_at.isoformat() if arrival_at else None,
                "is_approaching": bool(is_app),
                "is_scheduled": bool(is_sched) if is_sched is not None else None,
            }
            for arrival_at, is_app, is_sched in rows
        ],
    }


def feature_snapshot_bus(
    conn: duckdb.DuckDBPyConnection, spec, now: datetime,
) -> dict[str, Any]:
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
    return {
        "direction_label": spec.boarding.direction_label,
        "live_departures": [
            {
                "arrival_at": arrival_at.isoformat() if arrival_at else None,
                "is_approaching": bool(is_app),
                "direction_name": direction_name,
            }
            for arrival_at, is_app, direction_name in rows
        ],
    }


def feature_snapshot_metra(
    conn: duckdb.DuckDBPyConnection, spec, now: datetime, direction_id: int | None,
) -> dict[str, Any]:
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
    return {
        "direction_id": direction_id,
        "viable_trips": [
            {
                "trip_id": trip_id,
                "boarding_predicted_at": b.isoformat() if b else None,
                "alighting_predicted_at": a.isoformat() if a else None,
            }
            for trip_id, b, a in rows
        ],
    }


def feature_snapshot_intercampus(
    conn: duckdb.DuckDBPyConnection, spec, now: datetime,
) -> dict[str, Any]:
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
    return {
        "direction": spec.direction,
        "viable_trips": [
            {
                "trip_id": trip_id,
                "boarding_predicted_at": b.isoformat() if b else None,
                "alighting_predicted_at": a.isoformat() if a else None,
            }
            for trip_id, b, a in rows
        ],
    }


def _dispatch_to_prediction(
    d: KernelDispatch,
    *,
    version: str,
    wait_adjustment: float = 0.0,
) -> Prediction:
    """Apply an optional residual mean shift to the kernel's wait and
    wrap into a Prediction. ``wait_adjustment`` is added to mean and to
    every quantile; clipped to non-negative."""
    wait_leg = PredictionLeg.from_summary(d.wait)
    if wait_adjustment:
        adj_quantiles = {
            q: max(0.0, v + wait_adjustment) for q, v in wait_leg.quantiles.items()
        }
        wait_leg = PredictionLeg(
            quantiles=adj_quantiles,
            mean=max(0.0, wait_leg.mean + wait_adjustment),
            confidence=wait_leg.confidence,
            sample_count=wait_leg.sample_count,
        )
    in_vehicle_leg = PredictionLeg.from_summary(d.in_vehicle)
    state_label = d.wait_forecast.state.value if d.wait_forecast else None
    explanation = d.wait_forecast.explanation if d.wait_forecast else None
    feature_completeness = 1.0
    if d.feature_bundle is not None:
        feature_completeness = d.feature_bundle.completeness
    return Prediction(
        predictor_version=version,
        wait=wait_leg,
        in_vehicle=in_vehicle_leg,
        feature_snapshot=_bundle_to_snapshot(d),
        feature_completeness=feature_completeness,
        state_label=state_label,
        explanation=explanation,
        schedule_fallback=(state_label == "feedUnreliable"),
    )


class JourneyKernelPredictor:
    """Parity-pure adapter. Same output as the legacy ``corpus._predict``."""

    predictor_version = KERNEL_VERSION

    def predict(
        self,
        conn: duckdb.DuckDBPyConnection,
        corridor: Corridor,
        *,
        now: datetime,
    ) -> Prediction | None:
        d = _dispatch(conn, corridor, now=now)
        if d is None:
            return None
        return _dispatch_to_prediction(d, version=self.predictor_version)


class JourneyKernelEBPredictor:
    """Kernel + empirical-Bayes residual mean shift on the wait leg.

    Reads the running ``residual_mean`` and ``n_observations`` from
    ``predictor_state`` for the (line, direction) bucket and applies a
    Beta-binomial-style shrinkage toward zero (no shift) when the bucket
    is sparse:

        n_eff = max(50, n)
        shrunk = residual_mean * (n / n_eff)

    The resolver updates ``residual_mean`` incrementally each time it
    scores a kernel-v1 outcome (Welford's online algorithm).
    """

    predictor_version = KERNEL_EB_VERSION
    SHRINKAGE_FLOOR = 50  # n_effective floor — strong shrinkage until the bucket has 50+ samples

    def predict(
        self,
        conn: duckdb.DuckDBPyConnection,
        corridor: Corridor,
        *,
        now: datetime,
    ) -> Prediction | None:
        d = _dispatch(conn, corridor, now=now)
        if d is None:
            return None
        shift = self._lookup_shift(
            conn, line=d.line, direction_code=d.direction_code or "",
        )
        return _dispatch_to_prediction(
            d, version=self.predictor_version, wait_adjustment=shift,
        )

    @classmethod
    def _lookup_shift(
        cls,
        conn: duckdb.DuckDBPyConnection,
        *,
        line: str,
        direction_code: str,
    ) -> float:
        row = conn.execute(
            """
            SELECT residual_mean, n_observations
              FROM predictor_state
             WHERE predictor_version = ?
               AND line = ? AND direction_code = ?
               AND leg = 'wait' AND quantile = -1
            """,
            [KERNEL_EB_VERSION, line, direction_code],
        ).fetchone()
        if row is None or row[0] is None:
            return 0.0
        residual_mean, n_obs = row[0], int(row[1] or 0)
        if n_obs <= 0:
            return 0.0
        n_eff = max(cls.SHRINKAGE_FLOOR, n_obs)
        shrunk = residual_mean * (n_obs / n_eff)
        if not math.isfinite(shrunk):
            return 0.0
        return float(shrunk)


def update_eb_state(
    conn: duckdb.DuckDBPyConnection,
    *,
    line: str,
    direction_code: str,
    residual_seconds: float,
    now: datetime,
) -> None:
    """Welford's online mean/variance update for the EB shrinkage.

    Called by the resolver after each scored kernel outcome. The
    leg='wait', quantile=-1 row is the canonical mean-shift entry; DtACI
    stores per-quantile offsets in separate rows.
    """
    if not math.isfinite(residual_seconds):
        return
    row = conn.execute(
        """
        SELECT n_observations, residual_mean, residual_var
          FROM predictor_state
         WHERE predictor_version = ?
           AND line = ? AND direction_code = ?
           AND leg = 'wait' AND quantile = -1
        """,
        [KERNEL_EB_VERSION, line, direction_code],
    ).fetchone()
    if row is None:
        conn.execute(
            """
            INSERT INTO predictor_state
                (predictor_version, line, direction_code, leg, quantile,
                 offset_seconds, residual_mean, residual_var, n_observations,
                 coverage_target, step_size, updated_at)
            VALUES (?, ?, ?, 'wait', -1, 0.0, ?, 0.0, 1, 0.8, 0.0, ?)
            """,
            [KERNEL_EB_VERSION, line, direction_code, residual_seconds, now],
        )
        return
    n_prev, mean_prev, var_prev = int(row[0] or 0), float(row[1] or 0.0), float(row[2] or 0.0)
    n_new = n_prev + 1
    delta = residual_seconds - mean_prev
    mean_new = mean_prev + delta / n_new
    delta2 = residual_seconds - mean_new
    m2_prev = var_prev * n_prev
    m2_new = m2_prev + delta * delta2
    var_new = m2_new / n_new
    conn.execute(
        """
        UPDATE predictor_state
           SET n_observations = ?, residual_mean = ?, residual_var = ?, updated_at = ?
         WHERE predictor_version = ?
           AND line = ? AND direction_code = ?
           AND leg = 'wait' AND quantile = -1
        """,
        [n_new, mean_new, var_new, now,
         KERNEL_EB_VERSION, line, direction_code],
    )
