"""Intercampus (Northwestern shuttle) prediction + trajectory + resolver.

Same shape as Metra: trip-id based. The catalog has two directions
('northbound' from Chicago campus, 'southbound' from Evanston campus)
and ~24 stops between them. We pick a direction and two stops served
by that direction; predictor finds the next trip_id at the boarding
stop with a later arrival at the alighting stop.

GTFS-RT trip.direction_id is mapped to the 'northbound'/'southbound'
direction strings via the catalog; we don't try to be clever about it
since this is a tiny network.
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import duckdb

from .catalog import (
    IntercampusStop,
    intercampus_by_direction,
    load_intercampus_catalog,
)
from .journey.time_distribution import TimeDistributionSummary


log = logging.getLogger(__name__)

INTERCAMPUS_SCHEDULE_HEADWAY_S = 1200.0


@dataclass(frozen=True)
class IntercampusTripSpec:
    direction: str
    boarding: IntercampusStop
    alighting: IntercampusStop
    leave_at: datetime


def sample_intercampus_trip(
    *,
    catalog: list[IntercampusStop] | None = None,
    rng: random.Random,
    leave_at: datetime,
) -> IntercampusTripSpec | None:
    catalog = catalog if catalog is not None else load_intercampus_catalog()
    by_d = intercampus_by_direction(catalog)
    eligible = [d for d, stops in by_d.items() if len(stops) >= 2]
    if not eligible:
        return None
    direction = rng.choice(eligible)
    a, b = rng.sample(by_d[direction], 2)
    return IntercampusTripSpec(direction=direction, boarding=a, alighting=b, leave_at=leave_at)


@dataclass(frozen=True)
class _IntercampusTripView:
    trip_id: str
    direction: str | None
    boarding_predicted_at: datetime
    alighting_predicted_at: datetime


def _viable_trips(
    conn: duckdb.DuckDBPyConnection,
    *,
    boarding_id: str,
    alighting_id: str,
    direction: str,
    leave_at: datetime,
    horizon: datetime,
) -> list[_IntercampusTripView]:
    rows = conn.execute(
        """
        WITH latest AS (
            SELECT trip_id, stop_id, direction,
                   MAX(predicted_at) AS predicted_at
              FROM intercampus_arrivals_raw
             WHERE stop_id IN (?, ?)
               AND predicted_at IS NOT NULL
             GROUP BY trip_id, stop_id, direction
        )
        SELECT b.trip_id, b.direction, b.predicted_at, a.predicted_at
          FROM latest b
          JOIN latest a USING (trip_id)
         WHERE b.stop_id = ?
           AND a.stop_id = ?
           AND b.predicted_at >= ?
           AND b.predicted_at <= ?
           AND a.predicted_at > b.predicted_at
         ORDER BY b.predicted_at
         LIMIT 6
        """,
        [boarding_id, alighting_id, boarding_id, alighting_id, leave_at, horizon],
    ).fetchall()
    return [
        _IntercampusTripView(
            trip_id=trip_id,
            direction=dir_str,
            boarding_predicted_at=board_at,
            alighting_predicted_at=alight_at,
        )
        for trip_id, dir_str, board_at, alight_at in rows
    ]


def predict_intercampus_trip(
    conn: duckdb.DuckDBPyConnection,
    spec: IntercampusTripSpec,
    *,
    now: datetime,
    wait_window_minutes: float = 60.0,
) -> tuple[TimeDistributionSummary, TimeDistributionSummary, str | None] | None:
    horizon = spec.leave_at + timedelta(minutes=wait_window_minutes)
    viable = _viable_trips(
        conn,
        boarding_id=spec.boarding.stop_id,
        alighting_id=spec.alighting.stop_id,
        direction=spec.direction,
        leave_at=spec.leave_at,
        horizon=horizon,
    )
    if not viable:
        return None
    primary = viable[0]
    wait_seconds = max(0.0, (primary.boarding_predicted_at - spec.leave_at).total_seconds())
    if len(viable) >= 2:
        gap = (viable[1].boarding_predicted_at - primary.boarding_predicted_at).total_seconds()
        sigma = max(120.0, gap * 0.4)
    else:
        sigma = max(120.0, wait_seconds * 0.2)
    wait = TimeDistributionSummary.analytic(
        mean=wait_seconds, sigma=sigma, confidence=min(0.85, 0.5 + len(viable) * 0.1)
    )
    in_v_seconds = (primary.alighting_predicted_at - primary.boarding_predicted_at).total_seconds()
    in_vehicle = TimeDistributionSummary.analytic(
        mean=max(0.0, in_v_seconds), sigma=max(60.0, in_v_seconds * 0.15), confidence=0.6
    )
    return wait, in_vehicle, primary.direction


def enqueue_intercampus_forecast(
    conn: duckdb.DuckDBPyConnection,
    *,
    spec: IntercampusTripSpec,
    wait: TimeDistributionSummary,
    in_vehicle: TimeDistributionSummary,
    direction: str | None,
    now: datetime,
    snapshot_polled_at: datetime,
) -> str:
    total_mean = wait.mean + in_vehicle.mean
    total_p50 = wait.p50 + in_vehicle.p50
    total_p80 = wait.p80 + in_vehicle.p80
    total_p90 = wait.p90 + in_vehicle.p90
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
        ) VALUES (?, ?, ?, ?, 'intercampus', 'intercampus', ?, 0, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, ?, 'pending')
        """,
        [
            forecast_id, now, snapshot_polled_at, spec.leave_at,
            direction,
            spec.boarding.stop_id, spec.boarding.name,
            spec.alighting.stop_id, spec.alighting.name,
            wait.mean, wait.p50, wait.p80, wait.p90,
            in_vehicle.mean,
            total_mean, total_p50, total_p80, total_p90,
            resolve_after,
        ],
    )
    return forecast_id


# Trajectory ----------------------------------------------------------


def build_observed_intercampus_trips(
    conn: duckdb.DuckDBPyConnection, *, now: datetime, horizon_hours: float = 6.0
) -> int:
    cutoff = now - timedelta(hours=horizon_hours)
    rows = conn.execute(
        """
        SELECT route_id, trip_id, stop_id, direction,
               MAX(predicted_at) AS observed,
               MIN(polled_at) AS first_seen,
               MAX(polled_at) AS last_seen,
               COUNT(*) AS samples
          FROM intercampus_arrivals_raw
         WHERE polled_at >= ?
           AND predicted_at IS NOT NULL
         GROUP BY route_id, trip_id, stop_id, direction
        """,
        [cutoff],
    ).fetchall()
    inserts: list[tuple] = []
    for route_id, trip_id, stop_id, direction, observed, first_seen, last_seen, samples in rows:
        if observed is None or observed > now:
            continue
        inserts.append(
            (route_id, trip_id, stop_id, direction, observed, first_seen, last_seen, samples, "dropoff")
        )
    if not inserts:
        return 0
    conn.executemany(
        """
        INSERT INTO intercampus_trips_observed
            (route_id, trip_id, stop_id, direction,
             observed_arrival_at, first_seen_at, last_seen_at, sample_count, inferred_from)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (route_id, trip_id, stop_id, observed_arrival_at) DO NOTHING
        """,
        inserts,
    )
    return len(inserts)


# Resolver ------------------------------------------------------------


def resolve_intercampus_forecast(
    conn: duckdb.DuckDBPyConnection,
    *,
    forecast_id: str,
    leave_at: datetime,
    boarding_stop_id: str,
    alighting_stop_id: str,
    now: datetime,
) -> dict | None:
    boarding = conn.execute(
        """
        SELECT trip_id, observed_arrival_at, direction
          FROM intercampus_trips_observed
         WHERE stop_id = ?
           AND observed_arrival_at >= ?
           AND observed_arrival_at <= ?
         ORDER BY observed_arrival_at ASC
         LIMIT 5
        """,
        [boarding_stop_id, leave_at, now],
    ).fetchall()
    if not boarding:
        return None
    for trip_id, boarded_at, direction in boarding:
        alighting = conn.execute(
            """
            SELECT observed_arrival_at
              FROM intercampus_trips_observed
             WHERE trip_id = ? AND stop_id = ?
               AND observed_arrival_at >= ?
             ORDER BY observed_arrival_at ASC
             LIMIT 1
            """,
            [trip_id, alighting_stop_id, boarded_at],
        ).fetchone()
        if alighting is None:
            continue
        alighted_at = alighting[0]
        if alighted_at <= boarded_at:
            continue
        return {
            "boarded_run_number": trip_id,
            "boarded_at": boarded_at,
            "alighted_at": alighted_at,
            "actual_wait_seconds": (boarded_at - leave_at).total_seconds(),
            "actual_in_vehicle_seconds": (alighted_at - boarded_at).total_seconds(),
            "actual_total_seconds": (alighted_at - leave_at).total_seconds(),
            "boarded_direction_code": direction,
            "boarded_destination_name": None,
        }
    return None
