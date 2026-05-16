"""Main loop. Polls the CTA API round-robin, writes raw arrivals, builds
observed runs, generates random trip forecasts, and resolves due ones.

Rate budget: 100 requests per 5-minute window per key (~1 per 3s).
`Settings.station_round_robin_batch` controls how many stations we poll
each tick (default 18 per 30s tick = 36/min, well under the cap with
headroom for retries).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import random
import signal
from datetime import datetime, timedelta
from typing import Iterable

import duckdb
import structlog

from . import db, trajectory
from .bus_client import CTABusClient
from .catalog import LStation, load_catalog
from .config import CHICAGO, Settings, settings
from .cta_train_client import ArrivalRaw, CTATrainClient
from .intercampus_client import IntercampusClient
from .metra_client import MetraClient
from .resolver import resolve_due_forecasts
from .trip_generator import enqueue_forecast, predict_trip, sample_trip

log = structlog.get_logger(__name__)


async def run(stngs: Settings = settings) -> None:
    if not stngs.cta_train_api_key:
        raise SystemExit("error: CTA_TRAIN_API_KEY env var is required")

    structlog.configure(processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.KeyValueRenderer(key_order=["timestamp", "level", "event"]),
    ])

    catalog = load_catalog()
    log.info("collector.startup", stations=len(catalog), poll_interval=stngs.poll_interval_seconds)

    client = CTATrainClient(stngs.cta_train_api_key)
    bus_client = CTABusClient(stngs.cta_bus_api_key) if stngs.cta_bus_api_key else None
    metra_client = MetraClient(stngs.metra_api_key) if stngs.metra_api_key else None
    intercampus_client = IntercampusClient()  # no auth
    rotation = itertools.cycle(catalog)
    bus_rotation = itertools.cycle(stngs.monitored_bus_stops) if stngs.monitored_bus_stops else None

    last_replica_refresh = datetime.now(CHICAGO)
    last_trip_gen = datetime.now(CHICAGO)
    last_resolver = datetime.now(CHICAGO)
    last_trajectory = datetime.now(CHICAGO)
    last_positions_poll = datetime.now(CHICAGO)
    last_bus_poll = datetime.now(CHICAGO)
    last_metra_poll = datetime.now(CHICAGO)
    last_intercampus_poll = datetime.now(CHICAGO)
    rng = random.Random()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    with db.writer() as conn:
        try:
            while not stop_event.is_set():
                tick_started = datetime.now(CHICAGO)
                batch = _take(rotation, stngs.station_round_robin_batch)
                await _poll_and_persist(conn, client, batch=batch)

                if (tick_started - last_positions_poll).total_seconds() >= stngs.poll_interval_seconds:
                    n_positions = await _poll_positions(conn, client, lines=stngs.line_codes)
                    if n_positions:
                        log.info("positions.polled", n=n_positions)
                    last_positions_poll = tick_started

                if bus_client and bus_rotation and (tick_started - last_bus_poll).total_seconds() >= stngs.poll_interval_seconds:
                    bus_batch = _take(bus_rotation, stngs.bus_round_robin_batch)
                    n_bus = await _poll_bus(conn, bus_client, batch=bus_batch)
                    if n_bus:
                        log.info("bus.polled", n=n_bus)
                    last_bus_poll = tick_started

                if metra_client and (tick_started - last_metra_poll).total_seconds() >= stngs.metra_poll_interval_seconds:
                    n_metra = await _poll_metra(conn, metra_client)
                    if n_metra:
                        log.info("metra.polled", n=n_metra)
                    last_metra_poll = tick_started

                if (tick_started - last_intercampus_poll).total_seconds() >= stngs.intercampus_poll_interval_seconds:
                    n_ic = await _poll_intercampus(conn, intercampus_client)
                    if n_ic:
                        log.info("intercampus.polled", n=n_ic)
                    last_intercampus_poll = tick_started

                if (tick_started - last_trajectory).total_seconds() >= stngs.poll_interval_seconds * 4:
                    written = trajectory.build_observed_runs(conn, now=tick_started)
                    log.info("trajectory.built", rows=written)
                    last_trajectory = tick_started

                if (tick_started - last_trip_gen).total_seconds() >= stngs.trip_generation_interval_seconds:
                    n_enqueued = _generate_trips(conn, catalog, rng=rng, now=tick_started, count=stngs.trips_per_generation_tick)
                    if n_enqueued:
                        log.info("trips.enqueued", n=n_enqueued)
                    last_trip_gen = tick_started

                if (tick_started - last_resolver).total_seconds() >= stngs.resolver_interval_seconds:
                    n_resolved, n_unresolvable = resolve_due_forecasts(
                        conn, now=tick_started, expiration_buffer_seconds=stngs.forecast_resolution_buffer_seconds
                    )
                    if n_resolved or n_unresolvable:
                        log.info("forecasts.resolved", resolved=n_resolved, unresolvable=n_unresolvable)
                    last_resolver = tick_started

                if (tick_started - last_replica_refresh).total_seconds() >= stngs.read_replica_refresh_seconds:
                    db.refresh_read_replica()
                    last_replica_refresh = tick_started

                elapsed = (datetime.now(CHICAGO) - tick_started).total_seconds()
                sleep_for = max(1.0, stngs.poll_interval_seconds - elapsed)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=sleep_for)
                except asyncio.TimeoutError:
                    pass
        finally:
            await client.aclose()
            if bus_client:
                await bus_client.aclose()
            if metra_client:
                await metra_client.aclose()
            await intercampus_client.aclose()
            log.info("collector.shutdown")


def _take(it: Iterable, n: int) -> list:
    out = []
    for _ in range(n):
        out.append(next(iter(it)))
    return out


async def _poll_and_persist(
    conn: duckdb.DuckDBPyConnection,
    client: CTATrainClient,
    *,
    batch: list[LStation],
) -> None:
    tasks = [client.fetch_arrivals(map_id=s.map_id) for s in batch]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    polled_at = datetime.now(CHICAGO)
    inserts: list[tuple] = []
    for station, arrivals in zip(batch, results, strict=True):
        if isinstance(arrivals, Exception):
            log.warning("poll.error", station=station.name, err=str(arrivals))
            continue
        for a in arrivals:
            inserts.append((
                polled_at, a.line, a.run_number, a.map_id, a.stop_id, a.station_name,
                a.direction_code, a.destination_name, a.predicted_at, a.arrival_at,
                a.is_approaching, a.is_delayed, a.is_fault, a.is_scheduled,
            ))
    if not inserts:
        return
    conn.executemany(
        """
        INSERT INTO train_arrivals_raw (
            polled_at, line, run_number, map_id, stop_id, station_name,
            direction_code, destination_name, predicted_at, arrival_at,
            is_approaching, is_delayed, is_fault, is_scheduled
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        inserts,
    )


async def _poll_positions(
    conn: duckdb.DuckDBPyConnection,
    client: CTATrainClient,
    *,
    lines: tuple[str, ...],
) -> int:
    """One ttpositions request covers all in-flight runs across the listed
    lines. ~8 requests every 30s; way cheaper than per-station polling and
    a more authoritative trajectory signal."""
    try:
        positions = await client.fetch_positions(line_codes=lines)
    except Exception as exc:  # noqa: BLE001
        log.warning("positions.error", err=str(exc))
        return 0
    polled_at = datetime.now(CHICAGO)
    if not positions:
        return 0
    rows = [
        (
            polled_at, p.line, p.run_number, p.destination_name, p.direction_code,
            p.next_station_map_id, p.next_station_name,
            p.predicted_at, p.next_arrival_at,
            p.is_approaching, p.is_delayed,
        )
        for p in positions
    ]
    conn.executemany(
        """
        INSERT INTO train_positions_raw (
            polled_at, line, run_number, destination_name, direction_code,
            next_station_map_id, next_station_name,
            predicted_at, next_arrival_at, is_approaching, is_delayed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


async def _poll_bus(
    conn: duckdb.DuckDBPyConnection,
    client: CTABusClient,
    *,
    batch: list[tuple[str, int]],
) -> int:
    """Poll CTA Bus predictions for a small rotating set of monitored
    (route, stop_id) pairs. CTA Bus has ~14k stops and a 10k/day budget;
    cover a curated subset rather than the whole catalog."""
    tasks = [client.fetch_predictions(route=route, stop_id=stop_id) for route, stop_id in batch]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    polled_at = datetime.now(CHICAGO)
    rows: list[tuple] = []
    for (route, stop_id), preds in zip(batch, results, strict=True):
        if isinstance(preds, Exception):
            log.warning("bus.error", route=route, stop_id=stop_id, err=str(preds))
            continue
        for p in preds:
            rows.append((
                polled_at, p.route, p.route_name, p.vehicle_id, p.stop_id, p.stop_name,
                p.destination_name, p.direction_name, p.generated_at, p.arrival_at,
                p.is_delayed, p.is_approaching,
            ))
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO bus_predictions_raw (
            polled_at, route, route_name, vehicle_id, stop_id, stop_name,
            destination_name, direction_name, generated_at, arrival_at,
            is_delayed, is_approaching
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


async def _poll_metra(
    conn: duckdb.DuckDBPyConnection,
    client: MetraClient,
) -> int:
    try:
        updates = await client.fetch_trip_updates()
    except Exception as exc:  # noqa: BLE001
        log.warning("metra.error", err=str(exc))
        return 0
    polled_at = datetime.now(CHICAGO)
    if not updates:
        return 0
    rows = [
        (
            polled_at, u.route_id, u.trip_id, u.station_id, u.direction_id,
            u.schedule_relationship, u.scheduled_at, u.predicted_at, u.delay_seconds,
        )
        for u in updates
    ]
    conn.executemany(
        """
        INSERT INTO metra_arrivals_raw (
            polled_at, route_id, trip_id, station_id, direction_id,
            schedule_relationship, scheduled_at, predicted_at, delay_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


async def _poll_intercampus(
    conn: duckdb.DuckDBPyConnection,
    client: IntercampusClient,
) -> int:
    try:
        updates = await client.fetch_trip_updates()
    except Exception as exc:  # noqa: BLE001
        log.warning("intercampus.error", err=str(exc))
        return 0
    polled_at = datetime.now(CHICAGO)
    if not updates:
        return 0
    rows = [
        (
            polled_at, u.route_id, u.trip_id, str(u.direction_id) if u.direction_id is not None else None,
            u.stop_id, None, None,
            u.predicted_at, u.predicted_at, u.delay_seconds,
            (u.delay_seconds or 0) > 60, "gtfs-rt",
        )
        for u in updates
    ]
    conn.executemany(
        """
        INSERT INTO intercampus_arrivals_raw (
            polled_at, route_id, trip_id, direction,
            stop_id, stop_name, destination_name,
            predicted_at, arrival_at, delay_seconds, is_delayed, time_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def _generate_trips(
    conn: duckdb.DuckDBPyConnection,
    catalog: list[LStation],
    *,
    rng: random.Random,
    now: datetime,
    count: int,
) -> int:
    n = 0
    for _ in range(count):
        spec = sample_trip(catalog, rng=rng, leave_at=now)
        if spec is None:
            continue
        forecast = predict_trip(conn, spec, now=now)
        if forecast is None:
            continue
        wait, in_vehicle = forecast
        enqueue_forecast(conn, spec=spec, wait=wait, in_vehicle=in_vehicle, now=now, snapshot_polled_at=now)
        n += 1
    return n


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
