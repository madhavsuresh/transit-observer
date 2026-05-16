"""CTA Bus prediction + trajectory + resolver.

For v1 we restrict trips to the monitored stop set so we always have
data on both ends. A trip is (route, boarding_stop_id, alighting_stop_id)
with both stops on the same route.

Wait kernel: StopArrivalProcess over recent `bus_predictions_raw` rows
filtered to (route, boarding_stop_id), heading "toward" the alighting
stop (direction filter via direction_name).

In-vehicle estimate: haversine(boarding, alighting) / 6 m/s + 30 s/km
stop penalty. Bus is the slowest mode by far.

Trajectory: per (route, vehicle_id, stop_id), infer observed arrival
from the prediction stream using the same approaching/dropoff signals
the L uses.

Resolver: find the first vehicle_id at boarding_stop with observed
arrival >= leave_at; then find that same vehicle at alighting_stop.
"""

from __future__ import annotations

import logging
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

import duckdb

from .catalog import BusStop, bus_by_route, load_bus_catalog
from .config import settings
from .journey.stop_arrival import (
    FeedState,
    LiveDeparture,
    StopArrivalProcess,
    WaitForecast,
)
from .journey.time_distribution import TimeDistributionSummary
from .trip_generator import haversine_meters


log = logging.getLogger(__name__)

BUS_SCHEDULE_HEADWAY_S = 720.0
BUS_AVG_SPEED_MPS = 6.0
BUS_STOP_PENALTY_S_PER_KM = 30.0


@dataclass(frozen=True)
class BusTripSpec:
    route: str
    boarding: BusStop
    alighting: BusStop
    leave_at: datetime

    @property
    def direction_label(self) -> str:
        return self.boarding.direction_label or "?"


def sample_bus_trip(
    *,
    monitored_stops: list[tuple[str, int]],
    rng: random.Random,
    leave_at: datetime,
) -> BusTripSpec | None:
    """Pick a (route, boarding_stop, alighting_stop) from the monitored set.

    A trip needs at least two stops on the same route inside the monitored
    list. We pair them deterministically so direction_label matches across
    boarding and alighting (avoids picking a NB-boarding + SB-alighting
    that the predictor can't make sense of).
    """
    catalog = load_bus_catalog()
    lookup = {(s.route, s.stop_id): s for s in catalog}

    by_route: dict[str, list[BusStop]] = defaultdict(list)
    for route, stop_id in monitored_stops:
        stop = lookup.get((route, stop_id))
        if stop is None:
            continue
        by_route[route].append(stop)

    eligible = [r for r, stops in by_route.items() if len(stops) >= 2]
    if not eligible:
        return None
    route = rng.choice(eligible)
    candidates = by_route[route]
    a, b = rng.sample(candidates, 2)
    return BusTripSpec(route=route, boarding=a, alighting=b, leave_at=leave_at)


def predict_bus_trip(
    conn: duckdb.DuckDBPyConnection,
    spec: BusTripSpec,
    *,
    now: datetime,
    wait_window_minutes: float = 45.0,
) -> tuple[WaitForecast, TimeDistributionSummary] | None:
    cutoff = spec.leave_at - timedelta(minutes=5)
    horizon = spec.leave_at + timedelta(minutes=wait_window_minutes)

    rows = conn.execute(
        """
        SELECT arrival_at, is_approaching, destination_name, direction_name
          FROM bus_predictions_raw
         WHERE route = ?
           AND stop_id = ?
           AND polled_at >= ?
           AND arrival_at >= ?
           AND arrival_at <= ?
         ORDER BY polled_at DESC, arrival_at ASC
        """,
        [spec.route, spec.boarding.stop_id, cutoff, spec.leave_at, horizon],
    ).fetchall()

    if not rows:
        return None

    # Direction filter: only keep arrivals whose direction_name matches
    # the boarding stop's direction_label.
    seen: dict[datetime, LiveDeparture] = {}
    target_dir = spec.boarding.direction_label.lower()
    for arrival_at, is_app, _destination, direction_name in rows:
        if target_dir and direction_name and direction_name.lower() != target_dir:
            continue
        seen.setdefault(arrival_at, LiveDeparture(arrival_at=arrival_at, is_approaching=bool(is_app)))
    deps = sorted(seen.values(), key=lambda d: d.arrival_at)
    if not deps:
        return None

    process = StopArrivalProcess.make(
        route=spec.route,
        direction=spec.boarding.direction_label,
        generated_at=now,
        departures=deps,
        schedule_headway_seconds=BUS_SCHEDULE_HEADWAY_S,
        feed_state=FeedState.fresh,
    )
    wait = process.wait_distribution(spec.leave_at)

    meters = haversine_meters(
        spec.boarding.latitude, spec.boarding.longitude,
        spec.alighting.latitude, spec.alighting.longitude,
    )
    in_vehicle_mean = meters / BUS_AVG_SPEED_MPS + (meters / 1000.0) * BUS_STOP_PENALTY_S_PER_KM
    in_vehicle = TimeDistributionSummary.analytic(
        mean=in_vehicle_mean,
        sigma=max(60.0, in_vehicle_mean * 0.18),
        confidence=0.55,
    )
    return wait, in_vehicle


def enqueue_bus_forecast(
    conn: duckdb.DuckDBPyConnection,
    *,
    spec: BusTripSpec,
    wait: WaitForecast,
    in_vehicle: TimeDistributionSummary,
    now: datetime,
    snapshot_polled_at: datetime,
) -> str:
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
            mode, line, direction_code,
            boarding_map_id, boarding_text_id, boarding_station_name,
            alighting_map_id, alighting_text_id, alighting_station_name,
            predicted_wait_mean, predicted_wait_p50, predicted_wait_p80, predicted_wait_p90,
            predicted_in_vehicle_mean,
            predicted_total_mean, predicted_total_p50, predicted_total_p80, predicted_total_p90,
            predicted_failure_prob, resolve_after, status
        ) VALUES (?, ?, ?, ?, 'bus', ?, ?, 0, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, ?, 'pending')
        """,
        [
            forecast_id, now, snapshot_polled_at, spec.leave_at,
            spec.route, spec.boarding.direction_label,
            str(spec.boarding.stop_id), spec.boarding.name,
            str(spec.alighting.stop_id), spec.alighting.name,
            wait.wait_distribution.mean, wait.wait_distribution.p50,
            wait.wait_distribution.p80, wait.wait_distribution.p90,
            in_vehicle.mean,
            total_mean, total_p50, total_p80, total_p90,
            resolve_after,
        ],
    )
    return forecast_id


# Trajectory builder --------------------------------------------------


@dataclass(frozen=True)
class _BusSample:
    polled_at: datetime
    predicted_arrival_at: datetime
    is_approaching: bool
    destination: str | None
    direction: str | None


def build_observed_bus_runs(conn: duckdb.DuckDBPyConnection, *, now: datetime, horizon_hours: float = 6.0) -> int:
    cutoff = now - timedelta(hours=horizon_hours)
    rows = conn.execute(
        """
        SELECT route, vehicle_id, stop_id, polled_at, arrival_at, is_approaching,
               destination_name, direction_name
          FROM bus_predictions_raw
         WHERE polled_at >= ?
         ORDER BY polled_at
        """,
        [cutoff],
    ).fetchall()

    grouped: dict[tuple[str, str, int], list[_BusSample]] = defaultdict(list)
    for route, vid, stop_id, polled_at, arrival_at, is_app, dest, direction in rows:
        if not vid:
            continue
        grouped[(route, vid, int(stop_id))].append(
            _BusSample(
                polled_at=polled_at,
                predicted_arrival_at=arrival_at,
                is_approaching=bool(is_app),
                destination=dest,
                direction=direction,
            )
        )

    inserts: list[tuple] = []
    for (route, vid, stop_id), samples in grouped.items():
        observed = _resolve_bus_observed(samples, now=now)
        if observed is None:
            continue
        inserts.append(
            (
                route, vid, stop_id,
                observed["destination"], observed["direction"],
                observed["observed_arrival_at"],
                observed["first_seen_at"], observed["last_seen_at"],
                observed["sample_count"], observed["inferred_from"],
            )
        )

    if not inserts:
        return 0

    conn.executemany(
        """
        INSERT INTO bus_runs_observed
            (route, vehicle_id, stop_id, destination_name, direction_name,
             observed_arrival_at, first_seen_at, last_seen_at, sample_count, inferred_from)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (route, vehicle_id, stop_id, observed_arrival_at) DO NOTHING
        """,
        inserts,
    )
    return len(inserts)


def _resolve_bus_observed(samples: list[_BusSample], *, now: datetime) -> dict | None:
    seq = sorted(samples, key=lambda s: s.polled_at)
    if not seq:
        return None
    destination = next((s.destination for s in reversed(seq) if s.destination), None)
    direction = next((s.direction for s in reversed(seq) if s.direction), None)
    approaching = [s for s in seq if s.is_approaching]
    if approaching:
        latest = approaching[-1]
        return {
            "observed_arrival_at": latest.predicted_arrival_at,
            "first_seen_at": seq[0].polled_at,
            "last_seen_at": seq[-1].polled_at,
            "sample_count": len(seq),
            "inferred_from": "approaching",
            "destination": destination,
            "direction": direction,
        }
    latest = seq[-1]
    if latest.predicted_arrival_at <= now:
        return {
            "observed_arrival_at": latest.predicted_arrival_at,
            "first_seen_at": seq[0].polled_at,
            "last_seen_at": seq[-1].polled_at,
            "sample_count": len(seq),
            "inferred_from": "dropoff",
            "destination": destination,
            "direction": direction,
        }
    return None


# Resolver ------------------------------------------------------------


def resolve_bus_forecast(
    conn: duckdb.DuckDBPyConnection,
    *,
    forecast_id: str,
    leave_at: datetime,
    route: str,
    boarding_stop_id: int,
    alighting_stop_id: int,
    now: datetime,
) -> dict | None:
    """Find the vehicle a rider arriving at boarding at `leave_at` would
    have boarded, then find its arrival at alighting."""
    boarding = conn.execute(
        """
        SELECT vehicle_id, observed_arrival_at, destination_name, direction_name
          FROM bus_runs_observed
         WHERE route = ?
           AND stop_id = ?
           AND observed_arrival_at >= ?
           AND observed_arrival_at <= ?
         ORDER BY observed_arrival_at ASC
         LIMIT 5
        """,
        [route, boarding_stop_id, leave_at, now],
    ).fetchall()
    if not boarding:
        return None

    for vid, boarded_at, destination, direction in boarding:
        alighting = conn.execute(
            """
            SELECT observed_arrival_at
              FROM bus_runs_observed
             WHERE route = ?
               AND vehicle_id = ?
               AND stop_id = ?
               AND observed_arrival_at >= ?
             ORDER BY observed_arrival_at ASC
             LIMIT 1
            """,
            [route, vid, alighting_stop_id, boarded_at],
        ).fetchone()
        if alighting is None:
            continue
        alighted_at = alighting[0]
        actual_wait = (boarded_at - leave_at).total_seconds()
        actual_in_vehicle = (alighted_at - boarded_at).total_seconds()
        if actual_in_vehicle < 0:
            continue
        return {
            "boarded_run_number": vid,
            "boarded_at": boarded_at,
            "alighted_at": alighted_at,
            "actual_wait_seconds": actual_wait,
            "actual_in_vehicle_seconds": actual_in_vehicle,
            "actual_total_seconds": (alighted_at - leave_at).total_seconds(),
            "boarded_direction_code": direction,
            "boarded_destination_name": destination,
        }
    return None
