"""Resolver writes a `truth_confidence` score per outcome.

We seed a forecast that's resolvable from `train_runs_observed`, plus
varying numbers of raw arrival rows near the boarded_at / alighted_at
moments, and check the score rises with sample density.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from transit_observer import db
from transit_observer.resolver import resolve_due_forecasts


T0 = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    db.init_schema(c)
    try:
        yield c
    finally:
        c.close()


def _insert_run(conn, *, run: str, line: str, map_id: int, arrival_at: datetime) -> None:
    conn.execute(
        """
        INSERT INTO train_runs_observed (
            line, run_number, map_id, direction_code, destination_name,
            observed_arrival_at, first_seen_at, last_seen_at, sample_count, inferred_from
        ) VALUES (?, ?, ?, '1', 'Howard', ?, ?, ?, 1, 'approaching')
        """,
        [line, run, map_id, arrival_at, arrival_at, arrival_at],
    )


def _seed_arrival_raw(conn, *, line: str, map_id: int, polled_at: datetime,
                      arrival_at: datetime, is_approaching: bool) -> None:
    conn.execute(
        """
        INSERT INTO train_arrivals_raw (
            polled_at, line, run_number, map_id, stop_id, station_name,
            direction_code, destination_name, predicted_at, arrival_at,
            is_approaching, is_delayed, is_fault, is_scheduled
        ) VALUES (?, ?, 'R1', ?, 0, 'A', '1', 'Howard', ?, ?, ?, FALSE, FALSE, FALSE)
        """,
        [polled_at, line, map_id, polled_at, arrival_at, is_approaching],
    )


def _enqueue(conn, *, leave_at: datetime, boarding: int, alighting: int) -> str:
    fid = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO forecast_queue (
            forecast_id, enqueued_at, snapshot_polled_at, leave_at,
            mode, line, direction_code,
            boarding_map_id, boarding_station_name,
            alighting_map_id, alighting_station_name,
            predicted_wait_mean, predicted_wait_p50, predicted_wait_p80, predicted_wait_p90,
            predicted_in_vehicle_mean,
            predicted_total_mean, predicted_total_p50, predicted_total_p80, predicted_total_p90,
            predicted_failure_prob, resolve_after, status
        ) VALUES (?, ?, ?, ?, 'L', 'Red', '1', ?, 'A', ?, 'B',
                  60, 60, 90, 120, 600, 660, 660, 780, 840, 0.0, ?, 'pending')
        """,
        [
            fid, leave_at, leave_at, leave_at, boarding, alighting,
            leave_at + timedelta(seconds=900),
        ],
    )
    return fid


def test_truth_confidence_with_dense_arrivals(conn: duckdb.DuckDBPyConnection):
    leave = T0
    boarding = 41320
    alighting = 41660
    boarded_at = leave + timedelta(seconds=120)
    alighted_at = leave + timedelta(seconds=720)

    _insert_run(conn, run="R1", line="Red", map_id=boarding, arrival_at=boarded_at)
    _insert_run(conn, run="R1", line="Red", map_id=alighting, arrival_at=alighted_at)

    # Seed several approaching + plain arrival samples bracketing both events.
    for i in range(3):
        _seed_arrival_raw(
            conn, line="Red", map_id=boarding,
            polled_at=boarded_at - timedelta(seconds=60 * (i + 1)),
            arrival_at=boarded_at, is_approaching=True,
        )
        _seed_arrival_raw(
            conn, line="Red", map_id=alighting,
            polled_at=alighted_at - timedelta(seconds=60 * (i + 1)),
            arrival_at=alighted_at, is_approaching=True,
        )

    fid = _enqueue(conn, leave_at=leave, boarding=boarding, alighting=alighting)
    n_res, _ = resolve_due_forecasts(
        conn, now=leave + timedelta(seconds=900), expiration_buffer_seconds=300,
    )
    assert n_res == 1
    tc = conn.execute(
        "SELECT truth_confidence FROM forecast_outcomes WHERE forecast_id = ?", [fid],
    ).fetchone()[0]
    assert tc is not None
    assert tc >= 0.9


def test_truth_confidence_with_sparse_arrivals(conn: duckdb.DuckDBPyConnection):
    leave = T0
    boarding = 41320
    alighting = 41660
    boarded_at = leave + timedelta(seconds=120)
    alighted_at = leave + timedelta(seconds=720)

    _insert_run(conn, run="R2", line="Red", map_id=boarding, arrival_at=boarded_at)
    _insert_run(conn, run="R2", line="Red", map_id=alighting, arrival_at=alighted_at)

    # Only one weak (non-approaching) sample at each endpoint.
    _seed_arrival_raw(
        conn, line="Red", map_id=boarding,
        polled_at=boarded_at - timedelta(seconds=30),
        arrival_at=boarded_at, is_approaching=False,
    )
    _seed_arrival_raw(
        conn, line="Red", map_id=alighting,
        polled_at=alighted_at - timedelta(seconds=30),
        arrival_at=alighted_at, is_approaching=False,
    )

    fid = _enqueue(conn, leave_at=leave, boarding=boarding, alighting=alighting)
    n_res, _ = resolve_due_forecasts(
        conn, now=leave + timedelta(seconds=900), expiration_buffer_seconds=300,
    )
    assert n_res == 1
    tc = conn.execute(
        "SELECT truth_confidence FROM forecast_outcomes WHERE forecast_id = ?", [fid],
    ).fetchone()[0]
    assert tc is not None
    assert 0.0 < tc < 0.5  # weak bracketing -> low confidence


def test_truth_confidence_with_no_arrivals_is_zero(conn: duckdb.DuckDBPyConnection):
    leave = T0
    boarding = 41320
    alighting = 41660
    boarded_at = leave + timedelta(seconds=120)
    alighted_at = leave + timedelta(seconds=720)

    _insert_run(conn, run="R3", line="Red", map_id=boarding, arrival_at=boarded_at)
    _insert_run(conn, run="R3", line="Red", map_id=alighting, arrival_at=alighted_at)
    # Deliberately no raw arrivals.

    fid = _enqueue(conn, leave_at=leave, boarding=boarding, alighting=alighting)
    n_res, _ = resolve_due_forecasts(
        conn, now=leave + timedelta(seconds=900), expiration_buffer_seconds=300,
    )
    assert n_res == 1
    tc = conn.execute(
        "SELECT truth_confidence FROM forecast_outcomes WHERE forecast_id = ?", [fid],
    ).fetchone()[0]
    assert tc == 0.0
