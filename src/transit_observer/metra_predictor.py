"""Metra prediction + trajectory + resolver.

Schedule-anchored. A "trip" is a specific Metra train (trip_id). For a
ride from station A → B we need to find the same trip_id at both
stations with predicted_at(A) < predicted_at(B).

Wait = (next viable trip's predicted_at at A) − leave_at.
In-vehicle = predicted_at(B) − predicted_at(A) for that trip.

Trajectory: per (route, trip, station), accept the most-recent
predicted_at once that time has passed as the observed arrival. No
isApp signal in GTFS-RT, so this is dropoff-only.

Resolver: the boarded trip is the next viable one after leave_at.
"""

from __future__ import annotations

import logging
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

import duckdb

from .catalog import MetraStation, load_metra_catalog, metra_by_route
from .journey.time_distribution import TimeDistributionSummary


log = logging.getLogger(__name__)

METRA_SCHEDULE_HEADWAY_S = 1800.0


@dataclass(frozen=True)
class MetraTripSpec:
    route_id: str
    boarding: MetraStation
    alighting: MetraStation
    leave_at: datetime


def sample_metra_trip(
    *,
    catalog: list[MetraStation] | None = None,
    rng: random.Random,
    leave_at: datetime,
) -> MetraTripSpec | None:
    catalog = catalog if catalog is not None else load_metra_catalog()
    by_r = metra_by_route(catalog)
    eligible = [r for r, stations in by_r.items() if len(stations) >= 2]
    if not eligible:
        return None
    route = rng.choice(eligible)
    a, b = rng.sample(by_r[route], 2)
    return MetraTripSpec(route_id=route, boarding=a, alighting=b, leave_at=leave_at)


@dataclass(frozen=True)
class _MetraTripView:
    trip_id: str
    direction_id: int | None
    boarding_predicted_at: datetime
    alighting_predicted_at: datetime


def _viable_trips(
    conn: duckdb.DuckDBPyConnection,
    *,
    route_id: str,
    boarding_id: str,
    alighting_id: str,
    leave_at: datetime,
    horizon: datetime,
) -> list[_MetraTripView]:
    """Find trips whose latest prediction has the boarding stop in the
    future (after leave_at) and the alighting stop strictly after that."""
    rows = conn.execute(
        """
        WITH latest AS (
            SELECT route_id, trip_id, station_id, direction_id,
                   MAX(predicted_at) AS predicted_at
              FROM metra_arrivals_raw
             WHERE route_id = ?
               AND station_id IN (?, ?)
               AND predicted_at IS NOT NULL
             GROUP BY route_id, trip_id, station_id, direction_id
        )
        SELECT b.trip_id, b.direction_id, b.predicted_at AS board_at, a.predicted_at AS alight_at
          FROM latest b
          JOIN latest a USING (route_id, trip_id)
         WHERE b.station_id = ?
           AND a.station_id = ?
           AND b.predicted_at >= ?
           AND b.predicted_at <= ?
           AND a.predicted_at > b.predicted_at
         ORDER BY b.predicted_at
         LIMIT 6
        """,
        [route_id, boarding_id, alighting_id, boarding_id, alighting_id, leave_at, horizon],
    ).fetchall()
    return [
        _MetraTripView(
            trip_id=trip_id,
            direction_id=direction_id,
            boarding_predicted_at=board_at,
            alighting_predicted_at=alight_at,
        )
        for trip_id, direction_id, board_at, alight_at in rows
    ]


def predict_metra_trip(
    conn: duckdb.DuckDBPyConnection,
    spec: MetraTripSpec,
    *,
    now: datetime,
    wait_window_minutes: float = 90.0,
) -> tuple[TimeDistributionSummary, TimeDistributionSummary, int | None] | None:
    """Returns (wait_distribution, in_vehicle_distribution, expected_direction_id)."""
    horizon = spec.leave_at + timedelta(minutes=wait_window_minutes)
    viable = _viable_trips(
        conn,
        route_id=spec.route_id,
        boarding_id=spec.boarding.station_id,
        alighting_id=spec.alighting.station_id,
        leave_at=spec.leave_at,
        horizon=horizon,
    )
    if not viable:
        return None

    primary = viable[0]
    wait_seconds = max(0.0, (primary.boarding_predicted_at - spec.leave_at).total_seconds())
    if len(viable) >= 2:
        gap_seconds = (viable[1].boarding_predicted_at - primary.boarding_predicted_at).total_seconds()
        sigma = max(120.0, gap_seconds * 0.4)
    else:
        sigma = max(120.0, wait_seconds * 0.2)
    wait_summary = TimeDistributionSummary.analytic(
        mean=wait_seconds, sigma=sigma, confidence=min(0.85, 0.5 + len(viable) * 0.1)
    )

    in_vehicle_seconds = (primary.alighting_predicted_at - primary.boarding_predicted_at).total_seconds()
    in_vehicle_summary = TimeDistributionSummary.analytic(
        mean=max(0.0, in_vehicle_seconds),
        sigma=max(120.0, in_vehicle_seconds * 0.1),
        confidence=0.65,
    )
    return wait_summary, in_vehicle_summary, primary.direction_id


def enqueue_metra_forecast(
    conn: duckdb.DuckDBPyConnection,
    *,
    spec: MetraTripSpec,
    wait: TimeDistributionSummary,
    in_vehicle: TimeDistributionSummary,
    direction_id: int | None,
    now: datetime,
    snapshot_polled_at: datetime,
) -> str:
    total_mean = wait.mean + in_vehicle.mean
    total_p50 = wait.p50 + in_vehicle.p50
    total_p80 = wait.p80 + in_vehicle.p80
    total_p90 = wait.p90 + in_vehicle.p90
    resolve_after = spec.leave_at + timedelta(seconds=total_p90 + 10 * 60)
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
        ) VALUES (?, ?, ?, ?, 'metra', ?, ?, 0, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, ?, 'pending')
        """,
        [
            forecast_id, now, snapshot_polled_at, spec.leave_at,
            spec.route_id, str(direction_id) if direction_id is not None else None,
            spec.boarding.station_id, spec.boarding.name,
            spec.alighting.station_id, spec.alighting.name,
            wait.mean, wait.p50, wait.p80, wait.p90,
            in_vehicle.mean,
            total_mean, total_p50, total_p80, total_p90,
            resolve_after,
        ],
    )
    return forecast_id


# Trajectory ----------------------------------------------------------


def build_observed_metra_trips(conn: duckdb.DuckDBPyConnection, *, now: datetime, horizon_hours: float = 12.0) -> int:
    cutoff = now - timedelta(hours=horizon_hours)
    rows = conn.execute(
        """
        SELECT route_id, trip_id, station_id, direction_id,
               MAX(predicted_at) AS observed,
               MIN(polled_at) AS first_seen,
               MAX(polled_at) AS last_seen,
               COUNT(*) AS samples
          FROM metra_arrivals_raw
         WHERE polled_at >= ?
           AND predicted_at IS NOT NULL
         GROUP BY route_id, trip_id, station_id, direction_id
        """,
        [cutoff],
    ).fetchall()
    inserts: list[tuple] = []
    for route_id, trip_id, station_id, direction_id, observed, first_seen, last_seen, samples in rows:
        if observed is None or observed > now:
            continue
        inserts.append(
            (route_id, trip_id, station_id, direction_id, observed, first_seen, last_seen, samples, "dropoff")
        )
    if not inserts:
        return 0
    conn.executemany(
        """
        INSERT INTO metra_trips_observed
            (route_id, trip_id, station_id, direction_id,
             observed_arrival_at, first_seen_at, last_seen_at, sample_count, inferred_from)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (route_id, trip_id, station_id, observed_arrival_at) DO NOTHING
        """,
        inserts,
    )
    return len(inserts)


# Resolver ------------------------------------------------------------


def resolve_metra_forecast(
    conn: duckdb.DuckDBPyConnection,
    *,
    forecast_id: str,
    leave_at: datetime,
    route_id: str,
    boarding_station_id: str,
    alighting_station_id: str,
    now: datetime,
) -> dict | None:
    boarding = conn.execute(
        """
        SELECT trip_id, observed_arrival_at, direction_id
          FROM metra_trips_observed
         WHERE route_id = ?
           AND station_id = ?
           AND observed_arrival_at >= ?
           AND observed_arrival_at <= ?
         ORDER BY observed_arrival_at ASC
         LIMIT 5
        """,
        [route_id, boarding_station_id, leave_at, now],
    ).fetchall()
    if not boarding:
        return None
    for trip_id, boarded_at, direction_id in boarding:
        alighting = conn.execute(
            """
            SELECT observed_arrival_at
              FROM metra_trips_observed
             WHERE route_id = ? AND trip_id = ? AND station_id = ?
               AND observed_arrival_at >= ?
             ORDER BY observed_arrival_at ASC
             LIMIT 1
            """,
            [route_id, trip_id, alighting_station_id, boarded_at],
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
            "boarded_direction_code": str(direction_id) if direction_id is not None else None,
            "boarded_destination_name": None,
        }
    return None
