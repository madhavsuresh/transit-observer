"""Reconstruct observed train arrivals from raw prediction + position snapshots.

Three signals in priority order:

1. **positions:isApp** — `ttpositions` reports `isApp=true` for run R at
   `nextStaId = S`. The train is physically pulling in. Strongest signal.
   We use the position poll's `polled_at` as the observed arrival — the
   moment we directly saw the train at the station.
2. **arrivals:approaching** — `ttarrivals` reports `isApp=true` for run R
   at `staId = S`. Same semantics from per-station polling. We use the
   prediction's `arrival_at`.
3. **arrivals:dropoff** — the prediction's `arrival_at` has passed and
   the run stopped appearing in the feed at that station. Weakest signal.

`inferred_from` records which path was used. Downstream metrics can
trust positions-derived rows more than dropoff-derived ones.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

import duckdb

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RunSample:
    polled_at: datetime
    predicted_arrival_at: datetime
    is_approaching: bool
    direction_code: str | None
    destination_name: str | None


def build_observed_runs(
    conn: duckdb.DuckDBPyConnection,
    *,
    horizon_hours: float = 6.0,
    now: datetime,
) -> int:
    """Re-derive `train_runs_observed` rows from arrivals + positions data
    inside the horizon. Returns the number of rows written."""
    cutoff = now - timedelta(hours=horizon_hours)

    arrival_rows = conn.execute(
        """
        SELECT line, run_number, map_id, polled_at, arrival_at, is_approaching, direction_code, destination_name
          FROM train_arrivals_raw
         WHERE polled_at >= ?
           AND is_fault = FALSE
         ORDER BY polled_at
        """,
        [cutoff],
    ).fetchall()

    position_rows = conn.execute(
        """
        SELECT line, run_number, next_station_map_id, polled_at, next_arrival_at, is_approaching,
               direction_code, destination_name
          FROM train_positions_raw
         WHERE polled_at >= ?
           AND next_station_map_id IS NOT NULL
         ORDER BY polled_at
        """,
        [cutoff],
    ).fetchall()

    arrivals_grouped: dict[tuple[str, str, int], list[_RunSample]] = defaultdict(list)
    for line, run, map_id, polled_at, arrival_at, is_app, dir_code, dest in arrival_rows:
        arrivals_grouped[(line, run, map_id)].append(
            _RunSample(
                polled_at=polled_at,
                predicted_arrival_at=arrival_at,
                is_approaching=bool(is_app),
                direction_code=dir_code,
                destination_name=dest,
            )
        )

    positions_grouped: dict[tuple[str, str, int], list[_PositionSample]] = defaultdict(list)
    for line, run, next_map_id, polled_at, next_arr, is_app, dir_code, dest in position_rows:
        if next_map_id is None:
            continue
        positions_grouped[(line, run, int(next_map_id))].append(
            _PositionSample(
                polled_at=polled_at,
                predicted_arrival_at=next_arr,
                is_approaching=bool(is_app),
                direction_code=dir_code,
                destination_name=dest,
            )
        )

    keys = set(arrivals_grouped.keys()) | set(positions_grouped.keys())
    inserts: list[tuple] = []
    for key in keys:
        line, run, map_id = key
        observed = _resolve_observed(
            arrival_samples=arrivals_grouped.get(key, []),
            position_samples=positions_grouped.get(key, []),
            now=now,
        )
        if observed is None:
            continue
        inserts.append(
            (
                line, run, map_id,
                observed.direction_code, observed.destination_name,
                observed.observed_arrival_at, observed.first_seen_at, observed.last_seen_at,
                observed.sample_count, observed.inferred_from,
            )
        )

    if not inserts:
        return 0

    conn.executemany(
        """
        INSERT INTO train_runs_observed
            (line, run_number, map_id, direction_code, destination_name,
             observed_arrival_at, first_seen_at, last_seen_at, sample_count, inferred_from)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (line, run_number, map_id, observed_arrival_at) DO NOTHING
        """,
        inserts,
    )
    return len(inserts)


@dataclass(frozen=True)
class _PositionSample:
    polled_at: datetime
    predicted_arrival_at: datetime | None
    is_approaching: bool
    direction_code: str | None
    destination_name: str | None


@dataclass(frozen=True)
class _ObservedArrival:
    observed_arrival_at: datetime
    first_seen_at: datetime
    last_seen_at: datetime
    sample_count: int
    inferred_from: str
    direction_code: str | None
    destination_name: str | None


def _resolve_observed(
    *,
    arrival_samples: list[_RunSample],
    position_samples: list[_PositionSample],
    now: datetime,
) -> _ObservedArrival | None:
    """Pick the best estimate of when the run arrived at the station,
    consulting both signal streams.

    Priority order:
    1. `positions:isApp` — train is physically pulling in per the position
       feed. observed = `polled_at` of that sample.
    2. `arrivals:approaching` — per-station feed reports the same.
       observed = `predicted_arrival_at`.
    3. `arrivals:dropoff` — prediction's `predicted_arrival_at` has passed
       and the run stopped appearing in the feed.
    """
    direction = (
        _latest_non_null(reversed(position_samples), "direction_code")
        or _latest_non_null(reversed(arrival_samples), "direction_code")
    )
    destination = (
        _latest_non_null(reversed(position_samples), "destination_name")
        or _latest_non_null(reversed(arrival_samples), "destination_name")
    )

    position_app = [p for p in position_samples if p.is_approaching]
    if position_app:
        latest = max(position_app, key=lambda s: s.polled_at)
        return _ObservedArrival(
            observed_arrival_at=latest.polled_at,
            first_seen_at=_min_polled(arrival_samples, position_samples),
            last_seen_at=_max_polled(arrival_samples, position_samples),
            sample_count=len(arrival_samples) + len(position_samples),
            inferred_from="positions:approaching",
            direction_code=direction,
            destination_name=destination,
        )

    arrival_app = [s for s in arrival_samples if s.is_approaching]
    if arrival_app:
        latest = max(arrival_app, key=lambda s: s.polled_at)
        return _ObservedArrival(
            observed_arrival_at=latest.predicted_arrival_at,
            first_seen_at=_min_polled(arrival_samples, position_samples),
            last_seen_at=_max_polled(arrival_samples, position_samples),
            sample_count=len(arrival_samples) + len(position_samples),
            inferred_from="arrivals:approaching",
            direction_code=direction,
            destination_name=destination,
        )

    if arrival_samples:
        latest = max(arrival_samples, key=lambda s: s.polled_at)
        if latest.predicted_arrival_at <= now:
            return _ObservedArrival(
                observed_arrival_at=latest.predicted_arrival_at,
                first_seen_at=_min_polled(arrival_samples, position_samples),
                last_seen_at=_max_polled(arrival_samples, position_samples),
                sample_count=len(arrival_samples) + len(position_samples),
                inferred_from="arrivals:dropoff",
                direction_code=direction,
                destination_name=destination,
            )

    return None


def _latest_non_null(samples: Iterable, attr: str) -> str | None:
    for s in samples:
        value = getattr(s, attr)
        if value:
            return value
    return None


def _min_polled(arr: list, pos: list) -> datetime:
    timestamps = [s.polled_at for s in arr] + [s.polled_at for s in pos]
    return min(timestamps)


def _max_polled(arr: list, pos: list) -> datetime:
    timestamps = [s.polled_at for s in arr] + [s.polled_at for s in pos]
    return max(timestamps)
