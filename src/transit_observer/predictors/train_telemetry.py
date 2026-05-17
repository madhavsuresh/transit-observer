"""Telemetry-based L (train) arrival predictor.

Wraps :mod:`transit_observer.train_v2.estimator` so it conforms to the
shared ``Predictor`` protocol. Returns ``None`` (registry falls back to
the kernel) when the estimator abstains, when no fresh prediction
exists for the boarding station, or when the train has already
advanced past the station.

The in-vehicle leg reuses the existing kernel's haversine model so the
head-to-head comparison stays focused on the wait-leg (which is the
harder thing to estimate well).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import duckdb

from ..catalog import LStation, load_catalog
from ..corridors import Corridor
from ..journey.time_distribution import TimeDistributionSummary
from ..train_v2.estimator import estimate_station
from ..train_v2.models import DisplayState, EstimateResult
from ..train_v2.util import now_ms
from ..trip_generator import haversine_meters
from .protocol import Prediction, PredictionLeg


TRAIN_TELEMETRY_VERSION = "train-telemetry-v1"


# Approximate average L speed including dwells. Used for the in-vehicle
# leg only; the wait leg comes from the v2 estimator.
L_AVG_SPEED_MPS = 12.0
L_STOP_PENALTY_S_PER_KM = 18.0


_CATALOG_BY_MAP: Optional[dict[int, LStation]] = None


def _catalog() -> dict[int, LStation]:
    global _CATALOG_BY_MAP
    if _CATALOG_BY_MAP is None:
        _CATALOG_BY_MAP = {int(s.map_id): s for s in load_catalog()}
    return _CATALOG_BY_MAP


def reset_catalog() -> None:
    """Test hook — drop the cached catalog."""
    global _CATALOG_BY_MAP
    _CATALOG_BY_MAP = None


def _wait_leg_from_estimate(
    estimate: EstimateResult,
    *,
    now_ms_local: int,
) -> Optional[PredictionLeg]:
    if estimate.predicted_arrival_ms is None:
        return None
    p50_s = max(0.0, (estimate.predicted_arrival_ms - now_ms_local) / 1000.0)
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
        sample_count=0,
    )


def _in_vehicle_summary(boarding: LStation, alighting: LStation) -> TimeDistributionSummary:
    meters = haversine_meters(
        boarding.latitude, boarding.longitude,
        alighting.latitude, alighting.longitude,
    )
    mean = meters / L_AVG_SPEED_MPS + (meters / 1000.0) * L_STOP_PENALTY_S_PER_KM
    return TimeDistributionSummary.analytic(
        mean=mean,
        sigma=max(45.0, mean * 0.15),
        confidence=0.65,
    )


class TrainTelemetryPredictor:
    """Cross-stream L predictor (ttarrivals + ttfollow + ttpositions + GTFS-RT).

    Returns ``None`` for non-L corridors and for cases the estimator
    abstains — the registry's ``active_predictor_for`` fallback then
    hands the corridor back to the kernel.
    """

    predictor_version = TRAIN_TELEMETRY_VERSION

    def predict(
        self,
        conn: duckdb.DuckDBPyConnection,
        corridor: Corridor,
        *,
        now: datetime,
    ) -> Optional[Prediction]:
        if corridor.mode != "L":
            return None

        catalog = _catalog()
        boarding = catalog.get(int(corridor.boarding_int_id))
        alighting = catalog.get(int(corridor.alighting_int_id))
        if boarding is None or alighting is None:
            return None

        estimates = estimate_station(
            conn,
            map_id=str(boarding.map_id),
            line=corridor.line,
            direction_code=None,  # corridor.direction is human-text; let estimator pick the best
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

        in_vehicle_leg = PredictionLeg.from_summary(_in_vehicle_summary(boarding, alighting))

        snapshot: dict[str, Any] = {
            "mode": "L",
            "line": corridor.line,
            "direction_code": best.direction_code or corridor.direction,
            "boarding_map_id": int(boarding.map_id),
            "alighting_map_id": int(alighting.map_id),
            "boarding_station_name": boarding.name,
            "alighting_station_name": alighting.name,
            "leave_at": now.isoformat(),
            # Telemetry-specific diagnostics live in feature_snapshot so
            # the Prediction / PredictionLeg dataclasses stay unchanged.
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
            "run_number": best.run_number,
            "calibration_source": best.features.get("calibration_source"),
            "follow_disagreement_s": best.features.get("follow_disagreement_s"),
            "gtfsrt_disagreement_s": best.features.get("gtfsrt_disagreement_s"),
            "gtfsrt_delay_seconds": best.features.get("gtfsrt_delay_seconds"),
            "prediction_age_s": best.features.get("prediction_age_s"),
            "position_age_s": best.features.get("position_age_s"),
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
