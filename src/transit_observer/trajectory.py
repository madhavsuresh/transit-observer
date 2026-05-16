"""Reconstruct observed train arrivals from raw prediction snapshots.

CTA doesn't expose a clean "train arrived at station X at time T" signal.
We approximate by tracking each (line, run_number, map_id) tuple through
the prediction stream:

- The arrival is **observed** as `predicted_arrival_at` of the most-recent
  prediction we saw with `is_approaching = true`, OR the most-recent
  predicted_arrival_at before the prediction stopped appearing in the feed.

This is heuristic. Real GTFS-RT vehicle positions would be more precise.
We mark the source with `inferred_from = 'approaching' | 'dropoff'` so
downstream metrics can weight by reliability.

The builder is **incremental**: each call processes only raw arrivals
newer than the latest `last_seen_at` already in `train_runs_observed`.
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
    """Re-derive `train_runs_observed` rows for any (line, run, station)
    tuples whose final-prediction window falls inside the horizon.

    Returns the number of rows written.
    """
    cutoff = now - timedelta(hours=horizon_hours)

    rows = conn.execute(
        """
        SELECT line, run_number, map_id, polled_at, arrival_at, is_approaching, direction_code, destination_name
          FROM train_arrivals_raw
         WHERE polled_at >= ?
           AND is_fault = FALSE
         ORDER BY polled_at
        """,
        [cutoff],
    ).fetchall()

    grouped: dict[tuple[str, str, int], list[_RunSample]] = defaultdict(list)
    for line, run, map_id, polled_at, arrival_at, is_app, dir_code, dest in rows:
        grouped[(line, run, map_id)].append(
            _RunSample(
                polled_at=polled_at,
                predicted_arrival_at=arrival_at,
                is_approaching=bool(is_app),
                direction_code=dir_code,
                destination_name=dest,
            )
        )

    inserts: list[tuple] = []
    for (line, run, map_id), samples in grouped.items():
        observed = _resolve_observed(samples, now=now)
        if observed is None:
            continue
        inserts.append(
            (
                line,
                run,
                map_id,
                observed.direction_code,
                observed.destination_name,
                observed.observed_arrival_at,
                observed.first_seen_at,
                observed.last_seen_at,
                observed.sample_count,
                observed.inferred_from,
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
class _ObservedArrival:
    observed_arrival_at: datetime
    first_seen_at: datetime
    last_seen_at: datetime
    sample_count: int
    inferred_from: str
    direction_code: str | None
    destination_name: str | None


def _resolve_observed(samples: Iterable[_RunSample], *, now: datetime) -> _ObservedArrival | None:
    """Pick the best estimate of when the run arrived at the station.

    Rules in priority order:
    1. The latest sample where `is_approaching` was true — the train was
       reported as pulling in. Use its `predicted_arrival_at`.
    2. Otherwise, if the latest sample's `predicted_arrival_at` is in the
       past relative to `now`, the prediction "dropped off" the feed
       implying the train passed. Use that predicted_arrival_at.
    3. Otherwise, the train hasn't arrived yet — skip.
    """
    seq = sorted(samples, key=lambda s: s.polled_at)
    if not seq:
        return None
    direction = next((s.direction_code for s in reversed(seq) if s.direction_code), None)
    destination = next((s.destination_name for s in reversed(seq) if s.destination_name), None)
    approaching = [s for s in seq if s.is_approaching]
    if approaching:
        latest = approaching[-1]
        return _ObservedArrival(
            observed_arrival_at=latest.predicted_arrival_at,
            first_seen_at=seq[0].polled_at,
            last_seen_at=seq[-1].polled_at,
            sample_count=len(seq),
            inferred_from="approaching",
            direction_code=direction,
            destination_name=destination,
        )

    latest = seq[-1]
    # The prediction last said the train would arrive at this time; if that's
    # already in the past, the run almost certainly already arrived.
    if latest.predicted_arrival_at <= now:
        return _ObservedArrival(
            observed_arrival_at=latest.predicted_arrival_at,
            first_seen_at=seq[0].polled_at,
            last_seen_at=seq[-1].polled_at,
            sample_count=len(seq),
            inferred_from="dropoff",
            direction_code=direction,
            destination_name=destination,
        )

    return None
