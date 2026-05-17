"""Main loop. Polls the CTA API round-robin, writes raw arrivals, builds
observed runs, issues corridor-driven forecasts, and resolves due ones.

Rate budget: 100 requests per 5-minute window per key (~1 per 3s).
`Settings.station_round_robin_batch` controls how many stations we poll
each tick (default 18 per 30s tick = 36/min, well under the cap with
headroom for retries).

Trip generation: each tick, corridors whose ``cadence_seconds`` has
elapsed since their last prediction are predicted in priority order, up
to ``trips_per_generation_tick``. Random sampling is gone; the corpus
is corridor-driven now (see ``corridors.py`` / ``corpus.py``).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import signal
from datetime import datetime, timedelta
from typing import Iterable

import duckdb
import structlog

from . import db, query_log, trajectory
from .air_quality_client import AirQualityClient
from .alerts_client import CTAAlertsClient
from .auto_upgrade import promote_popular
from .bus_client import CTABusClient
from .bus_predictor import build_observed_bus_runs
from .catalog import LStation, load_catalog
from .config import CHICAGO, Settings, settings
from .corpus import predict_and_enqueue_corridor
from .corridors import due_corridors, seed_corridors
from .cta_gtfsrt_client import CTAGtfsRtClient
from .cta_train_client import ArrivalRaw, CTATrainClient
from .gtfs_static_archive import snapshot_gtfs_feeds
from .intercampus_client import IntercampusClient
from .intercampus_predictor import build_observed_intercampus_trips
from .metra_client import MetraClient
from .metra_predictor import build_observed_metra_trips
from .payload_archive import make_response_recorder
from .resolver import resolve_due_forecasts
from .social_client import SocialAccount, SocialClient
from .sports_client import SportsClient
from .venue_client import VenueClient
from .weather_client import WeatherClient
from .web_snapshot import snapshot_urls

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
    _log_acceleration_backends()

    last_replica_refresh = datetime.now(CHICAGO)
    last_trip_gen = datetime.now(CHICAGO)
    last_resolver = datetime.now(CHICAGO)
    last_trajectory = datetime.now(CHICAGO)
    last_positions_poll = datetime.now(CHICAGO)
    last_bus_poll = datetime.now(CHICAGO)
    last_bus_avl_poll = datetime.now(CHICAGO) - timedelta(seconds=stngs.bus_avl_poll_interval_seconds)
    last_metra_poll = datetime.now(CHICAGO)
    last_intercampus_poll = datetime.now(CHICAGO)
    last_query_import = datetime.now(CHICAGO)
    last_promotion = datetime.now(CHICAGO)
    last_train_attempt = datetime.now(CHICAGO)
    # External-signal cadences. Initialized in the past so each fires once at startup.
    last_alerts_poll = datetime.now(CHICAGO) - timedelta(seconds=stngs.cta_alerts_poll_interval_seconds)
    last_gtfsrt_poll = datetime.now(CHICAGO) - timedelta(seconds=stngs.cta_gtfsrt_poll_interval_seconds)
    last_weather_poll = datetime.now(CHICAGO) - timedelta(seconds=stngs.weather_poll_interval_seconds)
    last_aqi_poll = datetime.now(CHICAGO) - timedelta(seconds=stngs.air_quality_poll_interval_seconds)
    last_social_poll = datetime.now(CHICAGO) - timedelta(seconds=stngs.social_poll_interval_seconds)
    last_slow_zone_poll = datetime.now(CHICAGO) - timedelta(seconds=stngs.slow_zone_poll_interval_seconds)
    last_gtfs_static_poll = datetime.now(CHICAGO) - timedelta(seconds=stngs.gtfs_static_poll_interval_seconds)
    last_sports_poll = datetime.now(CHICAGO) - timedelta(seconds=stngs.sports_poll_interval_seconds)
    last_venue_poll = datetime.now(CHICAGO) - timedelta(seconds=stngs.venue_calendar_poll_interval_seconds)
    last_open_data_poll = datetime.now(CHICAGO) - timedelta(seconds=stngs.chicago_open_data_poll_interval_seconds)
    last_academic_poll = datetime.now(CHICAGO) - timedelta(seconds=stngs.academic_calendar_poll_interval_seconds)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    with db.writer() as conn:
        # Construct clients inside the writer block so payload recorders
        # can capture every HTTP response into api_payloads_raw.
        client = CTATrainClient(
            stngs.cta_train_api_key,
            payload_recorder=make_response_recorder(conn, source="cta_train"),
        )
        bus_client = (
            CTABusClient(
                stngs.cta_bus_api_key,
                payload_recorder=make_response_recorder(conn, source="cta_bus"),
            )
            if stngs.cta_bus_api_key
            else None
        )
        metra_client = (
            MetraClient(
                stngs.metra_api_key,
                payload_recorder=make_response_recorder(conn, source="metra"),
            )
            if stngs.metra_api_key
            else None
        )
        intercampus_client = IntercampusClient(
            payload_recorder=make_response_recorder(conn, source="intercampus_nu"),
        )
        alerts_client = CTAAlertsClient(
            payload_recorder=make_response_recorder(conn, source="cta_alerts"),
        )
        social_client = SocialClient(
            payload_recorder=make_response_recorder(conn, source="social"),
        )
        cta_gtfsrt_client = CTAGtfsRtClient(
            payload_recorder=make_response_recorder(conn, source="cta_gtfsrt"),
        )
        weather_client = WeatherClient(
            payload_recorder=make_response_recorder(conn, source="weather"),
        )
        airnow_client = (
            AirQualityClient(
                stngs.airnow_api_key,
                payload_recorder=make_response_recorder(conn, source="airnow"),
            )
            if stngs.airnow_api_key
            else None
        )
        sports_client = SportsClient(
            payload_recorder=make_response_recorder(conn, source="espn_sports"),
        )
        venue_client = (
            VenueClient(
                stngs.ticketmaster_api_key,
                payload_recorder=make_response_recorder(conn, source="ticketmaster"),
            )
            if stngs.ticketmaster_api_key
            else None
        )
        rotation = itertools.cycle(catalog)
        bus_rotation = (
            itertools.cycle(stngs.monitored_bus_stops) if stngs.monitored_bus_stops else None
        )
        n_seeded = seed_corridors(conn, now=datetime.now(CHICAGO))
        log.info("corridors.seeded", n=n_seeded)
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

                if (
                    bus_client
                    and stngs.monitored_bus_stops
                    and (tick_started - last_bus_avl_poll).total_seconds() >= stngs.bus_avl_poll_interval_seconds
                ):
                    avl_routes = sorted({route for route, _ in stngs.monitored_bus_stops})
                    n_avl = await _poll_bus_avl(conn, bus_client, routes=avl_routes)
                    if n_avl:
                        log.info("bus_avl.polled", n=n_avl)
                    last_bus_avl_poll = tick_started

                if metra_client and (tick_started - last_metra_poll).total_seconds() >= stngs.metra_poll_interval_seconds:
                    n_metra = await _poll_metra(conn, metra_client)
                    if n_metra:
                        log.info("metra.polled", n=n_metra)
                    last_metra_poll = tick_started

                if (
                    _intercampus_service_active(tick_started)
                    and (tick_started - last_intercampus_poll).total_seconds() >= stngs.intercampus_poll_interval_seconds
                ):
                    n_ic = await _poll_intercampus(conn, intercampus_client)
                    if n_ic:
                        log.info("intercampus.polled", n=n_ic)
                    last_intercampus_poll = tick_started

                if (tick_started - last_alerts_poll).total_seconds() >= stngs.cta_alerts_poll_interval_seconds:
                    n_alerts = await _poll_cta_alerts(conn, alerts_client)
                    if n_alerts:
                        log.info("alerts.polled", n=n_alerts)
                    last_alerts_poll = tick_started

                if (
                    stngs.cta_gtfsrt_feeds
                    and (tick_started - last_gtfsrt_poll).total_seconds() >= stngs.cta_gtfsrt_poll_interval_seconds
                ):
                    n_tu, n_vp = await _poll_cta_gtfsrt(
                        conn, cta_gtfsrt_client, feeds=stngs.cta_gtfsrt_feeds
                    )
                    if n_tu or n_vp:
                        log.info("cta_gtfsrt.polled", trip_updates=n_tu, vehicle_positions=n_vp)
                    last_gtfsrt_poll = tick_started

                if (
                    stngs.monitored_social_accounts
                    and (tick_started - last_social_poll).total_seconds() >= stngs.social_poll_interval_seconds
                ):
                    accounts = [SocialAccount(p, ident) for p, ident in stngs.monitored_social_accounts]
                    n_social = await _poll_social(conn, social_client, accounts=accounts)
                    if n_social:
                        log.info("social.polled", n=n_social)
                    last_social_poll = tick_started

                if (tick_started - last_slow_zone_poll).total_seconds() >= stngs.slow_zone_poll_interval_seconds:
                    n_sz = await snapshot_urls(
                        conn,
                        source="cta_slow_zones_page",
                        urls=("https://www.transitchicago.com/yourtimedeservesatrain/",),
                    )
                    if n_sz:
                        log.info("slow_zones.snapshot", n=n_sz)
                    last_slow_zone_poll = tick_started

                if (
                    stngs.gtfs_static_feeds
                    and (tick_started - last_gtfs_static_poll).total_seconds() >= stngs.gtfs_static_poll_interval_seconds
                ):
                    n_gtfs = await snapshot_gtfs_feeds(
                        conn,
                        feeds=stngs.gtfs_static_feeds,
                        archive_dir=stngs.gtfs_archive_dir,
                    )
                    if n_gtfs:
                        log.info("gtfs_static.archived", n=n_gtfs)
                    last_gtfs_static_poll = tick_started

                if (
                    stngs.weather_sites
                    and (tick_started - last_weather_poll).total_seconds() >= stngs.weather_poll_interval_seconds
                ):
                    n_w = await _poll_weather(conn, weather_client, sites=stngs.weather_sites)
                    if n_w:
                        log.info("weather.polled", n=n_w)
                    last_weather_poll = tick_started

                if (
                    airnow_client
                    and stngs.air_quality_zips
                    and (tick_started - last_aqi_poll).total_seconds() >= stngs.air_quality_poll_interval_seconds
                ):
                    n_aqi = await _poll_aqi(conn, airnow_client, zips=stngs.air_quality_zips)
                    if n_aqi:
                        log.info("aqi.polled", n=n_aqi)
                    last_aqi_poll = tick_started

                if (
                    stngs.sports_teams
                    and (tick_started - last_sports_poll).total_seconds() >= stngs.sports_poll_interval_seconds
                ):
                    n_sp = await _poll_sports(conn, sports_client, teams=stngs.sports_teams)
                    if n_sp:
                        log.info("sports.polled", n=n_sp)
                    last_sports_poll = tick_started

                if (tick_started - last_venue_poll).total_seconds() >= stngs.venue_calendar_poll_interval_seconds:
                    n_ve = 0
                    if venue_client:
                        n_ve = await _poll_venues(conn, venue_client)
                    n_mc = 0
                    if stngs.mccormick_urls:
                        n_mc = await snapshot_urls(
                            conn,
                            source="mccormick_calendar_page",
                            urls=stngs.mccormick_urls,
                        )
                    if n_ve or n_mc:
                        log.info("venues.polled", ticketmaster=n_ve, mccormick=n_mc)
                    last_venue_poll = tick_started

                if (
                    stngs.academic_calendar_urls
                    and (tick_started - last_academic_poll).total_seconds() >= stngs.academic_calendar_poll_interval_seconds
                ):
                    n_ac = await snapshot_urls(
                        conn,
                        source="academic_calendar_page",
                        urls=stngs.academic_calendar_urls,
                    )
                    if n_ac:
                        log.info("academic.snapshot", n=n_ac)
                    last_academic_poll = tick_started

                if (tick_started - last_open_data_poll).total_seconds() >= stngs.chicago_open_data_poll_interval_seconds:
                    n_od = await _poll_chicago_open_data(
                        conn, datasets=stngs.chicago_open_datasets
                    )
                    if n_od:
                        log.info("chicago_open_data.polled", n=n_od)
                    last_open_data_poll = tick_started

                if (tick_started - last_trajectory).total_seconds() >= stngs.poll_interval_seconds * 4:
                    n_l = trajectory.build_observed_runs(conn, now=tick_started)
                    n_bus = build_observed_bus_runs(conn, now=tick_started)
                    n_metra = build_observed_metra_trips(conn, now=tick_started)
                    n_ic = build_observed_intercampus_trips(conn, now=tick_started)
                    log.info("trajectory.built", l=n_l, bus=n_bus, metra=n_metra, intercampus=n_ic)
                    last_trajectory = tick_started

                if (tick_started - last_trip_gen).total_seconds() >= stngs.trip_generation_interval_seconds:
                    enabled_modes = ["L"]
                    if bus_client:
                        enabled_modes.append("bus")
                    if metra_client:
                        enabled_modes.append("metra")
                    if _intercampus_service_active(tick_started):
                        enabled_modes.append("intercampus")  # no key needed; M-F only
                    n_enqueued = _generate_corridor_predictions(
                        conn,
                        now=tick_started,
                        enabled_modes=enabled_modes,
                        max_per_tick=stngs.trips_per_generation_tick,
                    )
                    if n_enqueued:
                        log.info("corpus.enqueued", n=n_enqueued)
                    last_trip_gen = tick_started

                if (tick_started - last_resolver).total_seconds() >= stngs.resolver_interval_seconds:
                    n_resolved, n_unresolvable = resolve_due_forecasts(
                        conn, now=tick_started, expiration_buffer_seconds=stngs.forecast_resolution_buffer_seconds
                    )
                    if n_resolved or n_unresolvable:
                        log.info("forecasts.resolved", resolved=n_resolved, unresolvable=n_unresolvable)
                    last_resolver = tick_started

                if (tick_started - last_query_import).total_seconds() >= stngs.query_import_interval_seconds:
                    n_imported = query_log.import_pending(conn)
                    if n_imported:
                        log.info("queries.imported", n=n_imported)
                    last_query_import = tick_started

                if (tick_started - last_promotion).total_seconds() >= stngs.promotion_interval_seconds:
                    promoted = promote_popular(
                        conn, now=tick_started,
                        min_count=stngs.promotion_min_count,
                    )
                    if promoted:
                        log.info("corridors.promoted", n=len(promoted), ids=promoted)
                    last_promotion = tick_started

                if (
                    stngs.train_enabled
                    and (tick_started - last_train_attempt).total_seconds() >= stngs.train_interval_seconds
                ):
                    _maybe_train(conn, now=tick_started, window_days=stngs.train_window_days)
                    last_train_attempt = tick_started

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
            await alerts_client.aclose()
            await social_client.aclose()
            await cta_gtfsrt_client.aclose()
            await weather_client.aclose()
            if airnow_client:
                await airnow_client.aclose()
            await sports_client.aclose()
            if venue_client:
                await venue_client.aclose()
            log.info("collector.shutdown")


def _intercampus_service_active(now: datetime) -> bool:
    """Northwestern Intercampus shuttle runs Monday-Friday only.

    Skipping weekend polls saves ~2,880 round-trips per weekend. We still
    poll on holidays -- the feed is usually just empty then, and a
    full Chicago academic calendar isn't worth wiring in.
    """
    return now.weekday() < 5  # Mon=0 .. Fri=4


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
                a.flags, a.stop_description,
            ))
    if not inserts:
        return
    conn.executemany(
        """
        INSERT INTO train_arrivals_raw (
            polled_at, line, run_number, map_id, stop_id, station_name,
            direction_code, destination_name, predicted_at, arrival_at,
            is_approaching, is_delayed, is_fault, is_scheduled,
            flags, stop_description
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            p.lat, p.lon, p.heading,
        )
        for p in positions
    ]
    conn.executemany(
        """
        INSERT INTO train_positions_raw (
            polled_at, line, run_number, destination_name, direction_code,
            next_station_map_id, next_station_name,
            predicted_at, next_arrival_at, is_approaching, is_delayed,
            lat, lon, heading
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


async def _poll_chicago_open_data(
    conn: duckdb.DuckDBPyConnection,
    *,
    datasets: tuple[tuple[str, str], ...],
) -> int:
    """Snapshot each configured Socrata dataset. One row per record per poll."""
    if not datasets:
        return 0
    import httpx as _httpx  # local import; this is the only place we use it directly
    import json as _json
    polled_at = datetime.now(CHICAGO)
    recorder = make_response_recorder(conn, source="chicago_open_data")
    rows: list[tuple] = []
    async with _httpx.AsyncClient(
        timeout=30.0,
        event_hooks={"response": [recorder]},
        follow_redirects=True,
    ) as http:
        for table_id, label in datasets:
            url = f"https://data.cityofchicago.org/resource/{table_id}.json"
            try:
                resp = await http.get(url, params={"$limit": "1000"})
                resp.raise_for_status()
                records = resp.json()
            except Exception as exc:  # noqa: BLE001
                log.warning("chicago_open_data.error", table=table_id, err=str(exc))
                continue
            if not isinstance(records, list):
                continue
            for rec in records:
                record_id = rec.get(":id") or rec.get("id") or None
                rows.append((
                    polled_at, label, table_id,
                    str(record_id) if record_id is not None else None,
                    _json.dumps(rec),
                ))
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO chicago_open_data_raw (
            snapshot_polled_at, dataset, table_id, record_id, raw_payload_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


async def _poll_venues(
    conn: duckdb.DuckDBPyConnection,
    client: VenueClient,
) -> int:
    """Snapshot Chicago-area Ticketmaster events for the coming 30 days."""
    polled_at = datetime.now(CHICAGO)
    try:
        events = await client.fetch_chicago_events()
    except Exception as exc:  # noqa: BLE001
        log.warning("venues.error", err=str(exc))
        return 0
    if not events:
        return 0
    rows = [
        (
            polled_at, e.event_id, e.name, e.venue_name, e.venue_city,
            e.scheduled_start, e.sales_start, e.sales_end,
            e.classification, e.genre, e.url, e.raw_payload_json,
        )
        for e in events
    ]
    conn.executemany(
        """
        INSERT INTO venue_events_raw (
            snapshot_polled_at, event_id, name, venue_name, venue_city,
            scheduled_start, sales_start, sales_end,
            classification, genre, url, raw_payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


async def _poll_sports(
    conn: duckdb.DuckDBPyConnection,
    client: SportsClient,
    *,
    teams: tuple[tuple[str, str], ...],
) -> int:
    """Snapshot each configured team's full schedule. First poll to see
    ``completed=true`` for a given event approximates its end time."""
    polled_at = datetime.now(CHICAGO)
    rows: list[tuple] = []
    for league, team_abbr in teams:
        try:
            events = await client.fetch_team_schedule(league, team_abbr)
        except Exception as exc:  # noqa: BLE001
            log.warning("sports.error", league=league, team=team_abbr, err=str(exc))
            continue
        for e in events:
            rows.append((
                polled_at, e.event_id, e.league, e.sport, e.home_team, e.away_team,
                e.venue, e.scheduled_start, e.status, e.completed, e.attendance,
                e.home_score, e.away_score, e.raw_payload_json,
            ))
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO sports_events_raw (
            snapshot_polled_at, event_id, league, sport, home_team, away_team,
            venue, scheduled_start, status, completed, attendance,
            home_score, away_score, raw_payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


async def _poll_aqi(
    conn: duckdb.DuckDBPyConnection,
    client: AirQualityClient,
    *,
    zips: tuple[str, ...],
) -> int:
    """Snapshot current AQI for each configured zip code (AirNow)."""
    polled_at = datetime.now(CHICAGO)
    rows: list[tuple] = []
    for zip_code in zips:
        try:
            obs_list = await client.fetch_zip(zip_code)
        except Exception as exc:  # noqa: BLE001
            log.warning("aqi.error", zip=zip_code, err=str(exc))
            continue
        for obs in obs_list:
            rows.append((
                polled_at, obs.site_id, obs.parameter, obs.aqi, obs.raw_value, obs.unit,
                obs.category, obs.observation_time, obs.reporting_area,
                obs.latitude, obs.longitude,
            ))
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO air_quality_raw (
            polled_at, site_id, parameter, aqi, raw_value, unit,
            category, observation_time, reporting_area, latitude, longitude
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


async def _poll_weather(
    conn: duckdb.DuckDBPyConnection,
    client: WeatherClient,
    *,
    sites: tuple[tuple[str, float, float], ...],
) -> int:
    """Snapshot current weather at each configured site (Open-Meteo)."""
    if not sites:
        return 0
    polled_at = datetime.now(CHICAGO)
    rows: list[tuple] = []
    for site_id, lat, lon in sites:
        try:
            obs = await client.fetch_current(site_id=site_id, lat=lat, lon=lon)
        except Exception as exc:  # noqa: BLE001
            log.warning("weather.error", site=site_id, err=str(exc))
            continue
        if obs is None:
            continue
        rows.append((
            polled_at, obs.site_id, obs.lat, obs.lon, obs.observation_time,
            obs.temperature_c, obs.apparent_temperature_c, obs.humidity_pct,
            obs.precipitation_mm, obs.rain_mm, obs.snowfall_cm,
            obs.wind_speed_kph, obs.wind_gust_kph, obs.wind_direction_deg,
            obs.cloud_cover_pct, obs.pressure_hpa, obs.weather_code,
        ))
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO weather_observations_raw (
            polled_at, site_id, lat, lon, observation_time,
            temperature_c, apparent_temperature_c, humidity_pct,
            precipitation_mm, rain_mm, snowfall_cm,
            wind_speed_kph, wind_gust_kph, wind_direction_deg,
            cloud_cover_pct, pressure_hpa, weather_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


async def _poll_social(
    conn: duckdb.DuckDBPyConnection,
    client: SocialClient,
    *,
    accounts: list[SocialAccount],
) -> int:
    """Snapshot recent posts from configured transit-related accounts."""
    try:
        posts = await client.fetch_posts(accounts)
    except Exception as exc:  # noqa: BLE001
        log.warning("social.error", err=str(exc))
        return 0
    polled_at = datetime.now(CHICAGO)
    if not posts:
        return 0
    rows = [
        (
            polled_at, p.platform, p.handle, p.post_id, p.posted_at, p.body,
            p.url, p.in_reply_to, p.media_urls_json, p.raw_payload_json,
        )
        for p in posts
    ]
    conn.executemany(
        """
        INSERT INTO transit_social_raw (
            polled_at, platform, handle, post_id, posted_at, body,
            url, in_reply_to, media_urls_json, raw_payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


async def _poll_bus_avl(
    conn: duckdb.DuckDBPyConnection,
    client: CTABusClient,
    *,
    routes: list[str],
) -> int:
    """One getvehicles call covers up to ~10 routes. AVL gives ground-truth
    bus positions for segment-timing reconstruction."""
    if not routes:
        return 0
    # CTA Bus getvehicles caps at 10 routes per call; chunk if we ever exceed.
    chunks = [routes[i:i + 10] for i in range(0, len(routes), 10)]
    polled_at = datetime.now(CHICAGO)
    rows: list[tuple] = []
    for chunk in chunks:
        try:
            vehicles = await client.fetch_vehicles(routes=chunk)
        except Exception as exc:  # noqa: BLE001
            log.warning("bus_avl.error", routes=chunk, err=str(exc))
            continue
        for v in vehicles:
            rows.append((
                polled_at, v.route, v.vehicle_id, v.vehicle_timestamp,
                v.lat, v.lon, v.heading, v.speed_mph,
                v.pattern_id, v.pattern_distance,
                v.trip_id, v.block_id, v.destination, v.is_delayed, v.zone,
            ))
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO bus_positions_raw (
            polled_at, route, vehicle_id, vehicle_timestamp,
            lat, lon, heading, speed_mph,
            pattern_id, pattern_distance,
            trip_id, block_id, destination, is_delayed, zone
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


async def _poll_cta_gtfsrt(
    conn: duckdb.DuckDBPyConnection,
    client: CTAGtfsRtClient,
    *,
    feeds: tuple[tuple[str, str, str], ...],
) -> tuple[int, int]:
    """Poll each configured (mode, kind, url) CTA GTFS-RT feed.

    Returns (trip_update_rows, vehicle_position_rows).
    """
    polled_at = datetime.now(CHICAGO)
    tu_rows: list[tuple] = []
    vp_rows: list[tuple] = []
    for mode, kind, url in feeds:
        try:
            if kind == "trip_updates":
                updates = await client.fetch_trip_updates(url=url, mode=mode)
                for u in updates:
                    tu_rows.append((
                        polled_at, u.mode, u.route_id, u.trip_id, u.stop_id, u.stop_sequence,
                        u.arrival_time, u.arrival_delay_seconds,
                        u.departure_time, u.departure_delay_seconds,
                        u.schedule_relationship, u.vehicle_id,
                    ))
            elif kind == "vehicle_positions":
                positions = await client.fetch_vehicle_positions(url=url, mode=mode)
                for p in positions:
                    vp_rows.append((
                        polled_at, p.mode, p.route_id, p.trip_id, p.vehicle_id, p.vehicle_label,
                        p.lat, p.lon, p.bearing, p.speed_mps,
                        p.current_stop_sequence, p.current_status,
                        p.congestion_level, p.occupancy_status,
                    ))
        except Exception as exc:  # noqa: BLE001
            log.warning("cta_gtfsrt.error", mode=mode, kind=kind, err=str(exc))
            continue
    if tu_rows:
        conn.executemany(
            """
            INSERT INTO cta_gtfsrt_trip_updates_raw (
                polled_at, mode, route_id, trip_id, stop_id, stop_sequence,
                arrival_time, arrival_delay_seconds,
                departure_time, departure_delay_seconds,
                schedule_relationship, vehicle_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tu_rows,
        )
    if vp_rows:
        conn.executemany(
            """
            INSERT INTO cta_gtfsrt_vehicle_positions_raw (
                polled_at, mode, route_id, trip_id, vehicle_id, vehicle_label,
                lat, lon, bearing, speed_mps,
                current_stop_sequence, current_status,
                congestion_level, occupancy_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            vp_rows,
        )
    return len(tu_rows), len(vp_rows)


async def _poll_cta_alerts(
    conn: duckdb.DuckDBPyConnection,
    client: CTAAlertsClient,
) -> int:
    """Snapshot all currently-active CTA alerts. One row per (poll, alert)."""
    try:
        alerts = await client.fetch_alerts()
    except Exception as exc:  # noqa: BLE001
        log.warning("alerts.error", err=str(exc))
        return 0
    polled_at = datetime.now(CHICAGO)
    if not alerts:
        return 0
    rows = [
        (
            polled_at, a.alert_id, a.headline, a.short_description, a.full_description,
            a.severity_score, a.impact, a.event_start, a.event_end,
            a.tbd, a.major_alert, a.alert_url, a.impacted_services_json, a.guid,
        )
        for a in alerts
    ]
    conn.executemany(
        """
        INSERT INTO cta_alerts_raw (
            polled_at, alert_id, headline, short_description, full_description,
            severity_score, impact, event_start, event_end,
            tbd, major_alert, alert_url, impacted_services_json, guid
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


def _generate_corridor_predictions(
    conn: duckdb.DuckDBPyConnection,
    *,
    now: datetime,
    enabled_modes: list[str],
    max_per_tick: int,
) -> int:
    """Cycle through corridors whose cadence has elapsed and predict each.

    Each corridor produces at most one synthetic prediction per cadence
    window. ``max_per_tick`` caps how many we issue per generation tick so
    a long backlog doesn't burst-write hundreds of rows; corridors not
    serviced this tick come back next tick.
    """
    due = due_corridors(conn, now=now, enabled_modes=enabled_modes)
    n = 0
    for corridor in due[:max_per_tick]:
        result = predict_and_enqueue_corridor(conn, corridor, now=now)
        if result is not None:
            n += 1
    return n


def _log_acceleration_backends() -> None:
    """Surface which hardware-accelerated backends are wired up.

    Run once at collector startup so the operator can verify the right
    BLAS / threading is engaged. Pure diagnostic — never raises.
    """
    import os as _os
    import platform

    info: dict[str, str] = {
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu_count": str(_os.cpu_count() or 0),
    }
    try:
        import numpy as np  # type: ignore
        info["numpy"] = np.__version__
        # Detect Apple Accelerate vs OpenBLAS via the build dep dict
        cfg = np.__config__.show(mode="dicts")
        blas_name = cfg.get("Build Dependencies", {}).get("blas", {}).get("name") or "unknown"
        info["numpy_blas"] = blas_name
    except Exception:  # noqa: BLE001
        info["numpy"] = "missing"
    try:
        import duckdb as _duckdb  # type: ignore
        # Probe threads from a temporary connection (don't reuse a writer)
        c = _duckdb.connect(":memory:")
        info["duckdb_threads"] = str(c.execute("SELECT current_setting('threads')").fetchone()[0])
        info["duckdb"] = _duckdb.__version__
        c.close()
    except Exception:  # noqa: BLE001
        info["duckdb"] = "missing"
    try:
        import lightgbm as _lgb  # type: ignore
        info["lightgbm"] = _lgb.__version__
    except Exception:  # noqa: BLE001
        info["lightgbm"] = "not_installed"
    log.info("acceleration.backends", **info)


def _maybe_train(
    conn: duckdb.DuckDBPyConnection,
    *,
    now: datetime,
    window_days: int,
) -> None:
    """Try fitting the learned GBM in-process.

    Single-writer-safe: runs on the collector's writable connection so
    artifacts are registered without contending for the DB lock.
    No-ops cleanly when:
      - the cold-start gate isn't met (not enough resolved outcomes)
      - the ``learned`` dependency group isn't installed (LightGBM
        ImportError is caught and logged)
      - the training frame is empty after horizon filtering
    """
    try:
        from .training import dataset, fit
    except ImportError as exc:
        log.info("train.skip", reason="learned_deps_missing", err=str(exc))
        return

    ready, diag = dataset.cold_start_threshold(conn)
    if not ready:
        log.info(
            "train.skip",
            reason="cold_start",
            total_resolved=diag["total_resolved"],
            global_threshold=diag["global_threshold"],
            n_strong_buckets=diag["n_strong_buckets"],
        )
        return

    since = now - timedelta(days=window_days)
    frame = dataset.build_training_frame_l(conn, since=since, until=now)
    if len(frame) < 500:
        log.info("train.skip", reason="too_few_rows", n=len(frame))
        return

    try:
        report = fit.fit_quantile_gbm(conn, frame, now=now)
    except ImportError as exc:
        log.info("train.skip", reason="learned_deps_missing", err=str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        log.warning("train.error", err=str(exc))
        return

    warm = fit.warmup_dtaci(conn, n_warmup_rows=500)
    log.info(
        "train.fit",
        boosters=len(report.boosters),
        rows_train=report.rows_train,
        rows_val=report.rows_val,
        dtaci_updates=warm["n_warmup_updates"],
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
