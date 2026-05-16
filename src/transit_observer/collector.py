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
from .catalog import LStation, load_catalog
from .config import CHICAGO, Settings, settings
from .cta_train_client import ArrivalRaw, CTATrainClient
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
    rotation = itertools.cycle(catalog)

    last_replica_refresh = datetime.now(CHICAGO)
    last_trip_gen = datetime.now(CHICAGO)
    last_resolver = datetime.now(CHICAGO)
    last_trajectory = datetime.now(CHICAGO)
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
