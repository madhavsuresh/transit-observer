"""Telemetry-based bus arrival predictor.

Wraps :mod:`transit_observer.bus_v3.estimator` so it conforms to the
existing ``Predictor`` protocol. Returns ``None`` (registry falls back
to incumbent) when the estimator abstains, when no fresh prediction
snapshot exists for the boarding stop, or when no usable vehicle was
seen recently.

In-vehicle leg uses the same haversine + stop-penalty model the legacy
``bus_predictor`` does — the validator only estimates wait time, and
that's strictly the harder leg.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import duckdb

from ..bus_predictor import (
    BUS_AVG_SPEED_MPS,
    BUS_STOP_PENALTY_S_PER_KM,
)
from ..bus_v3.estimator import estimate_stop
from ..bus_v3.models import DisplayState, EstimateResult
from ..bus_v3.util import now_ms
from ..catalog import bus_by_id, load_bus_catalog
from ..corridors import Corridor
from ..journey.time_distribution import TimeDistributionSummary
from ..trip_generator import haversine_meters
from .protocol import SCHEMA_QUANTILES, Prediction, PredictionLeg


BUS_TELEMETRY_VERSION = "bus-telemetry-v1"


_BUS_CATALOG: Optional[dict[tuple[str, int], Any]] = None


def _catalog() -> dict[tuple[str, int], Any]:
    global _BUS_CATALOG
    if _BUS_CATALOG is None:
        _BUS_CATALOG = {(s.route, s.stop_id): s for s in load_bus_catalog()}
    return _BUS_CATALOG


def reset_catalog() -> None:
    """Test hook — drops the cached catalog."""
    global _BUS_CATALOG
    _BUS_CATALOG = None


def _in_vehicle_summary(
    boarding_lat: float,
    boarding_lon: float,
    alighting_lat: float,
    alighting_lon: float,
) -> TimeDistributionSummary:
    """Same haversine-+-stop-penalty model the kernel uses, so head-to-head
    metrics aren't biased by an in-vehicle difference."""
    meters = haversine_meters(boarding_lat, boarding_lon, alighting_lat, alighting_lon)
    mean = meters / BUS_AVG_SPEED_MPS + (meters / 1000.0) * BUS_STOP_PENALTY_S_PER_KM
    return TimeDistributionSummary.analytic(
        mean=mean,
        sigma=max(60.0, mean * 0.18),
        confidence=0.55,
    )


def _wait_leg_from_estimate(
    estimate: EstimateResult,
    *,
    now_ms_local: int,
) -> Optional[PredictionLeg]:
    if estimate.predicted_arrival_ms is None:
        return None
    p50_s = max(0.0, (estimate.predicted_arrival_ms - now_ms_local) / 1000.0)
    # Wait quantiles ⇒ same shape as the kernel's PredictionLeg.
    # interval80_high is the upper bound of the 80% prediction interval; treat
    # it as the p80 wait. Similarly interval90_high ⇒ p90.
    p80_s = p50_s
    if estimate.interval80_high_ms is not None:
        p80_s = max(p50_s, (estimate.interval80_high_ms - now_ms_local) / 1000.0)
    p90_s = p50_s
    if estimate.interval90_high_ms is not None:
        p90_s = max(p80_s, (estimate.interval90_high_ms - now_ms_local) / 1000.0)
    quantiles = {0.5: p50_s, 0.8: p80_s, 0.9: p90_s}
    return PredictionLeg(
        quantiles=quantiles,
        mean=p50_s,
        confidence=float(estimate.reliability),
        sample_count=int(estimate.features.get("n_speed_samples") or 0),
    )


class BusTelemetryPredictor:
    """Vehicle-telemetry + CTA-prediction hybrid.

    For bus corridors only. Returns ``None`` for non-bus modes and for
    cases the estimator abstains — the registry's ``active_predictor_for``
    fallback then hands the corridor back to the kernel.
    """

    predictor_version = BUS_TELEMETRY_VERSION

    def predict(
        self,
        conn: duckdb.DuckDBPyConnection,
        corridor: Corridor,
        *,
        now: datetime,
    ) -> Optional[Prediction]:
        if corridor.mode != "bus":
            return None

        catalog = _catalog()
        boarding = catalog.get((corridor.line, corridor.boarding_int_id))
        alighting = catalog.get((corridor.line, corridor.alighting_int_id))
        if boarding is None or alighting is None:
            return None

        # CTA's rtdir field is title-case ("Southbound"); ``boarding.direction_label``
        # mirrors the CTA case, while ``corridor.direction`` is lowercase
        # ("southbound"). Prefer the catalog-derived label so the v3 query
        # matches CTA's response shape.
        rtdir = boarding.direction_label or (
            corridor.direction.title() if corridor.direction else None
        )
        estimates = estimate_stop(
            conn,
            stpid=str(boarding.stop_id),
            rt=corridor.line,
            rtdir=rtdir,
            top=1,
        )
        if not estimates:
            return None
        best = estimates[0]
        if best.display_state == DisplayState.DO_NOT_DISPLAY_AS_ARRIVING:
            return None
        now_ms_local = best.generated_at_ms or now_ms()
        wait_leg = _wait_leg_from_estimate(best, now_ms_local=now_ms_local)
        if wait_leg is None:
            return None

        in_vehicle_summary = _in_vehicle_summary(
            boarding.latitude, boarding.longitude,
            alighting.latitude, alighting.longitude,
        )
        in_vehicle_leg = PredictionLeg.from_summary(in_vehicle_summary)

        snapshot: dict[str, Any] = {
            "mode": "bus",
            "line": corridor.line,
            "direction_code": rtdir,
            "boarding_text_id": str(boarding.stop_id),
            "alighting_text_id": str(alighting.stop_id),
            "boarding_station_name": boarding.name,
            "alighting_station_name": alighting.name,
            "leave_at": now.isoformat(),
            # Telemetry-specific diagnostics. Carried in feature_snapshot
            # so the dashboard can show display_state / reason_codes
            # without changing the Prediction / PredictionLeg dataclasses.
            "reliability": best.reliability,
            "display_state": best.display_state.value,
            "data_quality": best.data_quality.value,
            "reason_codes": best.reason_codes,
            "rider_message": best.rider_message,
            "predicted_arrival_ms": best.predicted_arrival_ms,
            "interval80_low_ms": best.interval80_low_ms,
            "interval80_high_ms": best.interval80_high_ms,
            "interval90_low_ms": best.interval90_low_ms,
            "interval90_high_ms": best.interval90_high_ms,
            "vid": best.vid,
            "tatripid": best.tatripid,
            "tablockid": best.tablockid,
            "calibration_source": best.features.get("calibration_source"),
            "cta_eta_s": best.features.get("cta_eta_s"),
            "geometry_eta_s": best.features.get("geometry_eta_s"),
            "vehicle_age_s": best.features.get("vehicle_age_s"),
            "prediction_age_s": best.features.get("prediction_age_s"),
            "pdist_trend": best.features.get("pdist_trend"),
            "dstp_trend": best.features.get("dstp_trend"),
            "prediction_volatility_s": best.features.get("prediction_volatility_s"),
        }

        return Prediction(
            predictor_version=self.predictor_version,
            wait=wait_leg,
            in_vehicle=in_vehicle_leg,
            feature_snapshot=snapshot,
            feature_completeness=float(best.reliability),
            state_label=best.data_quality.value,
            explanation=best.rider_message,
            schedule_fallback=False,
        )
