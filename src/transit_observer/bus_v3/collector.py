"""Single-cycle orchestrator for the v3 bus pipeline.

Modeled on the validator's ``collector.Collector.poll_once`` but adapted
for async httpx + DuckDB. Each call to :func:`poll_once` produces one
``run_id``-tagged batch of api_poll rows + normalized rows.

The host (transit_observer.collector) is responsible for the schedule.
This module is pure orchestration.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import duckdb
import structlog

from .catalog import BusV3Target, unique_routes, unique_stops
from .client import CTABusV3Client
from .normalize import record_poll
from .util import chunked, now_ms, parse_prdctdn_minutes, pick_first, root_of


log = structlog.get_logger(__name__)


@dataclass
class BusV3CycleConfig:
    targets: list[BusV3Target]
    metadata_refresh_seconds: float = 3600.0
    detour_refresh_seconds: float = 300.0
    max_stop_ids_per_request: int = 10
    max_vehicle_ids_per_request: int = 10
    max_route_ids_per_request: int = 10
    top_predictions: Optional[int] = None
    predictions_by_vid_enabled: bool = True
    extra_directions: list[str] = field(default_factory=list)


@dataclass
class BusV3CycleState:
    last_metadata_ms: int = 0
    last_detours_ms: int = 0
    cycle_index: int = 0
    last_run_id: Optional[str] = None


@dataclass
class BusV3CycleResult:
    run_id: str
    cycle_index: int
    server_ms: Optional[int]
    prediction_vids: list[str]
    near_arrival: bool
    polls_recorded: int


async def poll_once(
    conn: duckdb.DuckDBPyConnection,
    client: CTABusV3Client,
    *,
    config: BusV3CycleConfig,
    state: BusV3CycleState,
) -> BusV3CycleResult:
    """Run one v3 cycle, write api_poll + normalized rows, return summary.

    ``run_id`` is fresh per cycle. The validator's ``--replace`` semantics
    rely on this granularity, and the inference module joins by run_id +
    stpid + vid for crossing detection.
    """
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    state.last_run_id = run_id
    cycle_index = state.cycle_index
    state.cycle_index += 1

    routes = unique_routes(config.targets)
    stops = unique_stops(config.targets)
    polls_recorded = 0

    # gettime → server clock alignment.
    r = await client.gettime()
    server_ms = client.extract_server_time_ms(r)
    r.cta_server_time_ms = server_ms
    record_poll(conn, r, run_id=run_id, cycle_index=cycle_index)
    polls_recorded += 1

    t_ms = now_ms()
    if t_ms - state.last_metadata_ms >= config.metadata_refresh_seconds * 1000:
        polls_recorded += await _refresh_static_metadata(
            conn, client, routes=routes, stops=stops,
            config=config, server_ms=server_ms,
            run_id=run_id, cycle_index=cycle_index,
        )
        state.last_metadata_ms = t_ms
    if t_ms - state.last_detours_ms >= config.detour_refresh_seconds * 1000:
        polls_recorded += await _refresh_detours(
            conn, client, routes=routes, config=config,
            server_ms=server_ms, run_id=run_id, cycle_index=cycle_index,
        )
        state.last_detours_ms = t_ms

    # getpredictions by stop, in 10-stpid chunks.
    prediction_vids: set[str] = set()
    near_arrival = False
    for chunk in chunked(stops, config.max_stop_ids_per_request):
        result = await client.getpredictions(
            stpids=chunk,
            routes=routes or None,
            top=config.top_predictions,
            cta_server_time_ms=server_ms,
        )
        record_poll(conn, result, run_id=run_id, cycle_index=cycle_index)
        polls_recorded += 1
        root = root_of(result.json_data)
        for p in pick_first(root, "prd", "prds", "prediction", "predictions"):
            if not isinstance(p, dict):
                continue
            vid = p.get("vid")
            if vid is not None:
                prediction_vids.add(str(vid))
            m = parse_prdctdn_minutes(p.get("prdctdn"))
            if m is not None and m <= 3:
                near_arrival = True

    # getvehicles by route, in 10-route chunks.
    for chunk in chunked(routes, config.max_route_ids_per_request):
        result = await client.getvehicles(routes=chunk, cta_server_time_ms=server_ms)
        record_poll(conn, result, run_id=run_id, cycle_index=cycle_index)
        polls_recorded += 1

    # getpredictions by vid for vehicles that appeared in target stop predictions.
    if config.predictions_by_vid_enabled and prediction_vids:
        for chunk in chunked(sorted(prediction_vids), config.max_vehicle_ids_per_request):
            result = await client.getpredictions(
                vids=chunk,
                top=config.top_predictions,
                cta_server_time_ms=server_ms,
            )
            record_poll(conn, result, run_id=run_id, cycle_index=cycle_index)
            polls_recorded += 1

    return BusV3CycleResult(
        run_id=run_id,
        cycle_index=cycle_index,
        server_ms=server_ms,
        prediction_vids=sorted(prediction_vids),
        near_arrival=near_arrival,
        polls_recorded=polls_recorded,
    )


async def _refresh_static_metadata(
    conn: duckdb.DuckDBPyConnection,
    client: CTABusV3Client,
    *,
    routes: list[str],
    stops: list[str],
    config: BusV3CycleConfig,
    server_ms: Optional[int],
    run_id: str,
    cycle_index: int,
) -> int:
    n = 0
    # getroutes once. Cheap; refreshed on the metadata cadence.
    result = await client.getroutes(cta_server_time_ms=server_ms)
    record_poll(conn, result, run_id=run_id, cycle_index=cycle_index)
    n += 1
    for rt in routes:
        # getdirections per route.
        result = await client.getdirections(rt, cta_server_time_ms=server_ms)
        record_poll(conn, result, run_id=run_id, cycle_index=cycle_index)
        n += 1
        # getpatterns per route — full pattern point geometry.
        result = await client.getpatterns(rt=rt, cta_server_time_ms=server_ms)
        record_poll(conn, result, run_id=run_id, cycle_index=cycle_index)
        n += 1
        # getstops per known direction (or configured directions).
        rows = conn.execute(
            "SELECT dir_id FROM bus_v3_direction WHERE rt = ?",
            [rt],
        ).fetchall()
        directions = config.extra_directions or [str(r[0]) for r in rows if r[0]]
        for d in directions:
            result = await client.getstops(rt=rt, direction=d, cta_server_time_ms=server_ms)
            record_poll(conn, result, run_id=run_id, cycle_index=cycle_index)
            n += 1
    # Also fetch monitored stops by id so we always have coordinates.
    for chunk in chunked(stops, config.max_stop_ids_per_request):
        result = await client.getstops(stpids=chunk, cta_server_time_ms=server_ms)
        record_poll(conn, result, run_id=run_id, cycle_index=cycle_index)
        n += 1
    return n


async def _refresh_detours(
    conn: duckdb.DuckDBPyConnection,
    client: CTABusV3Client,
    *,
    routes: list[str],
    config: BusV3CycleConfig,
    server_ms: Optional[int],
    run_id: str,
    cycle_index: int,
) -> int:
    n = 0
    if routes:
        for rt in routes:
            result = await client.getdetours(rt=rt, cta_server_time_ms=server_ms)
            record_poll(conn, result, run_id=run_id, cycle_index=cycle_index)
            n += 1
    else:
        result = await client.getdetours(cta_server_time_ms=server_ms)
        record_poll(conn, result, run_id=run_id, cycle_index=cycle_index)
        n += 1
    result = await client.getenhanceddetours(cta_server_time_ms=server_ms)
    record_poll(conn, result, run_id=run_id, cycle_index=cycle_index)
    n += 1
    return n
