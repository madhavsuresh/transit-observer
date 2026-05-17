"""Single-cycle orchestrator for the v2 train pipeline.

Modeled on ``bus_v3.collector``: each call to :func:`poll_once` does one
round-robin batch of ttarrivals + ttpositions + ttfollow, returning a
``run_id``-tagged summary. The host (transit_observer.collector) owns
the schedule and budget — this module is pure orchestration.

ttarrivals stations are taken in rotating batches so the 100-req-per-5
min budget isn't blown. ttpositions costs one call per cycle (covers
all lines). ttfollow is queried only for the union of run numbers
observed in the cycle's ttarrivals responses, deduped — that's how the
estimator gets per-run trajectory context.
"""

from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import duckdb
import structlog

from .catalog import TrainV2Target
from .client import CTATrainV2Client
from .normalize import record_poll
from .util import now_ms


log = structlog.get_logger(__name__)


@dataclass
class TrainV2CycleConfig:
    targets: list[TrainV2Target]
    line_codes: tuple[str, ...]
    arrivals_batch_size: int = 12          # how many stations per cycle
    arrivals_max_predictions: int = 12     # ``max`` param per ttarrivals call
    follow_max_runs_per_cycle: int = 6     # cap on ttfollow calls per cycle
    rotation: Optional[itertools.cycle] = None


@dataclass
class TrainV2CycleState:
    last_run_id: Optional[str] = None
    cycle_index: int = 0
    # Persistent round-robin over the full station list. Reused across
    # cycles so we cover every station without re-randomizing.
    _rotation: Optional[itertools.cycle] = None
    last_followed_runs: set[str] = field(default_factory=set)


@dataclass
class TrainV2CycleResult:
    run_id: str
    cycle_index: int
    server_ms: Optional[int]
    arrivals_polled: int
    positions_polled: int
    follow_polled: int
    run_numbers: list[str]
    polls_recorded: int


def _take(rotation: itertools.cycle, n: int) -> list[TrainV2Target]:
    return [next(rotation) for _ in range(n)]


async def poll_once(
    conn: duckdb.DuckDBPyConnection,
    client: CTATrainV2Client,
    *,
    config: TrainV2CycleConfig,
    state: TrainV2CycleState,
) -> TrainV2CycleResult:
    if state._rotation is None:
        if not config.targets:
            raise ValueError("train_v2 cycle requires at least one station target")
        state._rotation = itertools.cycle(config.targets)

    run_id = f"trainv2_{uuid.uuid4().hex[:12]}"
    state.last_run_id = run_id
    cycle_index = state.cycle_index
    state.cycle_index += 1

    polls_recorded = 0
    server_ms_seen: list[int] = []
    run_numbers: set[str] = set()

    # 1. ttarrivals for a rotating batch of stations.
    batch = _take(state._rotation, config.arrivals_batch_size)
    arrivals_polled = 0
    for target in batch:
        result = await client.ttarrivals(
            map_id=target.map_id,
            max_predictions=config.arrivals_max_predictions,
        )
        poll_id = record_poll(conn, result, run_id=run_id, cycle_index=cycle_index)
        polls_recorded += 1
        arrivals_polled += 1
        if result.cta_server_time_ms is not None:
            server_ms_seen.append(result.cta_server_time_ms)
        # Collect run numbers seen in this station's response for ttfollow.
        if result.json_data:
            for eta in (result.json_data.get("ctatt") or {}).get("eta") or []:
                rn = eta.get("rn") if isinstance(eta, dict) else None
                if rn:
                    run_numbers.add(str(rn))

    # 2. ttpositions for all configured lines — one call.
    positions_polled = 0
    if config.line_codes:
        result = await client.ttpositions(line_codes=config.line_codes)
        record_poll(conn, result, run_id=run_id, cycle_index=cycle_index)
        polls_recorded += 1
        positions_polled = 1
        if result.cta_server_time_ms is not None:
            server_ms_seen.append(result.cta_server_time_ms)
        # Positions also reveal active runs. Useful when stations rotated
        # in this batch had no live predictions but a train is mid-route.
        if result.json_data:
            for route in (result.json_data.get("ctatt") or {}).get("route") or []:
                for train in (route or {}).get("train") or []:
                    rn = train.get("rn") if isinstance(train, dict) else None
                    if rn:
                        run_numbers.add(str(rn))

    # 3. ttfollow for a deduped subset of runs (cap to budget).
    # Prefer runs we haven't followed in the previous cycle so coverage
    # rotates.
    follow_candidates = sorted(run_numbers - state.last_followed_runs)
    if len(follow_candidates) < config.follow_max_runs_per_cycle:
        follow_candidates += sorted(run_numbers - set(follow_candidates))
    follow_candidates = follow_candidates[: config.follow_max_runs_per_cycle]
    follow_polled = 0
    followed_now: set[str] = set()
    for rn in follow_candidates:
        result = await client.ttfollow(run_number=rn)
        record_poll(conn, result, run_id=run_id, cycle_index=cycle_index)
        polls_recorded += 1
        follow_polled += 1
        followed_now.add(rn)
        if result.cta_server_time_ms is not None:
            server_ms_seen.append(result.cta_server_time_ms)
    state.last_followed_runs = followed_now

    server_ms = max(server_ms_seen) if server_ms_seen else None
    return TrainV2CycleResult(
        run_id=run_id,
        cycle_index=cycle_index,
        server_ms=server_ms,
        arrivals_polled=arrivals_polled,
        positions_polled=positions_polled,
        follow_polled=follow_polled,
        run_numbers=sorted(run_numbers),
        polls_recorded=polls_recorded,
    )
