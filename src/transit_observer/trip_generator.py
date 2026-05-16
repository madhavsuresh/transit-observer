"""Sample random CTA L trips and enqueue forecast predictions.

For v1 we pick a single-line trip:
- choose a line uniformly at random
- choose two distinct stations on that line at random
- assume the rider is at the boarding station NOW (no walking leg yet)
- predict the wait + in-vehicle distribution
- enqueue a forecast that the resolver picks up later

Coverage bias note: uniform line + uniform station sampling under-weights
high-traffic corridors. A later iteration can weight by ridership.
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import duckdb

from .catalog import LStation, by_line
from .journey.stop_arrival import (
    FeedState,
    LiveDeparture,
    StopArrivalProcess,
    WaitForecast,
)
from .journey.time_distribution import TimeDistributionSummary

log = logging.getLogger(__name__)


# Per-line average ground speed for the in-vehicle estimate (m/s).
# These are calibration starting points; the simulator's job is to find
# out whether they're right.
LINE_AVG_SPEED_MPS: dict[str, float] = {
    "Red": 12.0,
    "Blue": 13.0,
    "Brn": 11.0,
    "G": 11.0,
    "Org": 12.0,
    "P": 13.5,
    "Pink": 11.0,
    "Y": 11.0,
}
# Per-stop dwell penalty (seconds per km of straight-line distance).
LINE_STOP_PENALTY_S_PER_KM = 25.0


SCHEDULE_HEADWAY_S = 600.0


# The catalog uses spelled-out line names ("red", "blue", "brown", "green",
# "orange", "purple", "pink", "yellow"); the CTA API uses short codes ("Red",
# "Blue", "Brn", "G", "Org", "P", "Pink", "Y").
CATALOG_LINE_TO_API_CODE = {
    "red": "Red",
    "blue": "Blue",
    "brown": "Brn",
    "green": "G",
    "orange": "Org",
    "purple": "P",
    "pink": "Pink",
    "yellow": "Y",
}


@dataclass(frozen=True)
class TripSpec:
    line_catalog: str          # e.g. "red"
    line_api: str              # e.g. "Red"
    boarding: LStation
    alighting: LStation
    direction_label: str       # "north" | "south" | "east" | "west" — coarse
    leave_at: datetime


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math

    r = 6_371_000
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def direction_label(boarding: LStation, alighting: LStation) -> str:
    dlat = alighting.latitude - boarding.latitude
    dlon = alighting.longitude - boarding.longitude
    if abs(dlat) > abs(dlon):
        return "north" if dlat > 0 else "south"
    return "east" if dlon > 0 else "west"


def sample_trip(
    catalog: list[LStation],
    *,
    rng: random.Random,
    leave_at: datetime,
) -> TripSpec | None:
    by_l = by_line(catalog)
    eligible_lines = [line for line, stations in by_l.items() if len(stations) >= 2]
    if not eligible_lines:
        return None
    line = rng.choice(eligible_lines)
    stations = by_l[line]
    a, b = rng.sample(stations, 2)
    return TripSpec(
        line_catalog=line,
        line_api=CATALOG_LINE_TO_API_CODE.get(line, line),
        boarding=a,
        alighting=b,
        direction_label=direction_label(a, b),
        leave_at=leave_at,
    )


def predict_trip(
    conn: duckdb.DuckDBPyConnection,
    spec: TripSpec,
    *,
    now: datetime,
    wait_window_minutes: float = 30.0,
) -> tuple[WaitForecast, TimeDistributionSummary] | None:
    """Build a wait forecast + in-vehicle distribution for `spec`.

    Returns None if there's not enough data to predict (e.g. no recent
    arrivals at the boarding station for this line).
    """
    cutoff = spec.leave_at - timedelta(minutes=5)
    horizon = spec.leave_at + timedelta(minutes=wait_window_minutes)

    rows = conn.execute(
        """
        SELECT arrival_at, is_approaching, destination_name
          FROM train_arrivals_raw
         WHERE line = ?
           AND map_id = ?
           AND polled_at >= ?
           AND arrival_at >= ?
           AND arrival_at <= ?
           AND is_fault = FALSE
         ORDER BY polled_at DESC, arrival_at ASC
        """,
        [spec.line_api, spec.boarding.map_id, cutoff, spec.leave_at, horizon],
    ).fetchall()

    if not rows:
        return None

    # Deduplicate by predicted arrival_at, keeping the most recent prediction.
    seen: dict[datetime, LiveDeparture] = {}
    for arrival_at, is_app, destination in rows:
        if _heads_toward_alighting(spec, destination):
            seen.setdefault(arrival_at, LiveDeparture(arrival_at=arrival_at, is_approaching=bool(is_app)))
    deps = sorted(seen.values(), key=lambda d: d.arrival_at)
    if not deps:
        return None

    process = StopArrivalProcess.make(
        route=spec.line_api,
        direction=spec.direction_label,
        generated_at=now,
        departures=deps,
        schedule_headway_seconds=SCHEDULE_HEADWAY_S,
        feed_state=FeedState.fresh,
    )
    wait = process.wait_distribution(spec.leave_at)

    distance_m = haversine_meters(
        spec.boarding.latitude, spec.boarding.longitude,
        spec.alighting.latitude, spec.alighting.longitude,
    )
    speed = LINE_AVG_SPEED_MPS.get(spec.line_api, 12.0)
    in_vehicle_mean = distance_m / speed + (distance_m / 1000.0) * LINE_STOP_PENALTY_S_PER_KM
    in_vehicle_sigma = max(60.0, in_vehicle_mean * 0.12)
    in_vehicle = TimeDistributionSummary.analytic(
        mean=in_vehicle_mean,
        sigma=in_vehicle_sigma,
        confidence=0.6,
    )
    return wait, in_vehicle


def _heads_toward_alighting(spec: TripSpec, destination_name: str | None) -> bool:
    if not destination_name:
        return True
    # Lat/lon-based direction filter using catalog name lookup. The catalog
    # may not contain composite destinations like "Loop"; conservatively keep.
    dest_lower = destination_name.strip().lower()
    if dest_lower == "loop" or not dest_lower:
        return True
    return _dir_dot(spec.boarding, dest_lower, spec) > 0


# Module-level catalog cache keyed by id so we can pass references inside
# the dot-product helper without rebuilding it per call.
_NAME_LOOKUP: dict[str, LStation] | None = None


def _dir_dot(boarding: LStation, dest_lower: str, spec: TripSpec) -> float:
    global _NAME_LOOKUP
    if _NAME_LOOKUP is None:
        from .catalog import by_name, load_catalog
        _NAME_LOOKUP = by_name(load_catalog())
    dest = _NAME_LOOKUP.get(dest_lower)
    if dest is None:
        return 1.0  # unknown — conservatively keep
    trip_dlat = spec.alighting.latitude - boarding.latitude
    trip_dlon = spec.alighting.longitude - boarding.longitude
    dest_dlat = dest.latitude - boarding.latitude
    dest_dlon = dest.longitude - boarding.longitude
    return trip_dlat * dest_dlat + trip_dlon * dest_dlon


def enqueue_forecast(
    conn: duckdb.DuckDBPyConnection,
    *,
    spec: TripSpec,
    wait: WaitForecast,
    in_vehicle: TimeDistributionSummary,
    now: datetime,
    snapshot_polled_at: datetime,
) -> str:
    """Combine wait + in-vehicle into a total distribution and persist."""
    total_p50 = wait.wait_distribution.p50 + in_vehicle.p50
    total_p80 = wait.wait_distribution.p80 + in_vehicle.p80
    total_p90 = wait.wait_distribution.p90 + in_vehicle.p90
    total_mean = wait.wait_distribution.mean + in_vehicle.mean

    resolve_after = spec.leave_at + timedelta(seconds=total_p90 + 5 * 60)

    forecast_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO forecast_queue (
            forecast_id, enqueued_at, snapshot_polled_at, leave_at,
            line, direction_code,
            boarding_map_id, boarding_station_name,
            alighting_map_id, alighting_station_name,
            predicted_wait_mean, predicted_wait_p50, predicted_wait_p80, predicted_wait_p90,
            predicted_in_vehicle_mean,
            predicted_total_mean, predicted_total_p50, predicted_total_p80, predicted_total_p90,
            predicted_failure_prob,
            resolve_after, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        """,
        [
            forecast_id, now, snapshot_polled_at, spec.leave_at,
            spec.line_api, spec.direction_label,
            spec.boarding.map_id, spec.boarding.name,
            spec.alighting.map_id, spec.alighting.name,
            wait.wait_distribution.mean, wait.wait_distribution.p50,
            wait.wait_distribution.p80, wait.wait_distribution.p90,
            in_vehicle.mean,
            total_mean, total_p50, total_p80, total_p90,
            0.0,
            resolve_after,
        ],
    )
    return forecast_id
