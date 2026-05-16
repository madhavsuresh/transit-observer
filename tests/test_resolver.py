"""Resolver finds the realized run for a forecast and writes the outcome."""

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


def _insert_run(conn, *, run: str, line: str, map_id: int, arrival_at: datetime,
                destination: str = "Howard", direction: str = "1") -> None:
    conn.execute(
        """
        INSERT INTO train_runs_observed (
            line, run_number, map_id, direction_code, destination_name,
            observed_arrival_at, first_seen_at, last_seen_at, sample_count, inferred_from
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 'approaching')
        """,
        [line, run, map_id, direction, destination, arrival_at, arrival_at, arrival_at],
    )


def _enqueue(conn, *, leave_at: datetime, boarding: int, alighting: int,
             line: str = "Red", direction: str = "1") -> str:
    fid = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO forecast_queue (
            forecast_id, enqueued_at, snapshot_polled_at, leave_at,
            line, direction_code,
            boarding_map_id, boarding_station_name,
            alighting_map_id, alighting_station_name,
            predicted_wait_mean, predicted_wait_p50, predicted_wait_p80, predicted_wait_p90,
            predicted_in_vehicle_mean,
            predicted_total_mean, predicted_total_p50, predicted_total_p80, predicted_total_p90,
            predicted_failure_prob, resolve_after, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'A', ?, 'B', 60, 60, 90, 120, 600, 660, 660, 780, 840, 0.0, ?, 'pending')
        """,
        [fid, leave_at, leave_at, leave_at, line, direction, boarding, alighting,
         leave_at + timedelta(seconds=900)],
    )
    return fid


def test_resolves_when_run_arrives_at_both_stations(conn: duckdb.DuckDBPyConnection):
    leave = T0
    boarding = 40330
    alighting = 40540
    _insert_run(conn, run="R1", line="Red", map_id=boarding,
                arrival_at=leave + timedelta(seconds=120))
    _insert_run(conn, run="R1", line="Red", map_id=alighting,
                arrival_at=leave + timedelta(seconds=120 + 600))
    fid = _enqueue(conn, leave_at=leave, boarding=boarding, alighting=alighting)
    n_res, n_unr = resolve_due_forecasts(
        conn, now=leave + timedelta(seconds=900), expiration_buffer_seconds=300
    )
    assert n_res == 1
    assert n_unr == 0
    row = conn.execute(
        """
        SELECT actual_wait_seconds, actual_in_vehicle_seconds, actual_total_seconds, in_p80_window, in_p90_window
          FROM forecast_outcomes WHERE forecast_id = ?
        """,
        [fid],
    ).fetchone()
    assert row[0] == 120
    assert row[1] == 600
    assert row[2] == 720
    assert row[3] is True  # 720 <= predicted_p80 (780)
    assert row[4] is True  # 720 <= predicted_p90 (840)
    assert (
        conn.execute("SELECT status FROM forecast_queue WHERE forecast_id = ?", [fid]).fetchone()[0]
        == "resolved"
    )


def test_skips_when_run_not_yet_at_alighting(conn: duckdb.DuckDBPyConnection):
    leave = T0
    _insert_run(conn, run="R2", line="Red", map_id=40330, arrival_at=leave + timedelta(seconds=60))
    fid = _enqueue(conn, leave_at=leave, boarding=40330, alighting=40540)
    n_res, _ = resolve_due_forecasts(
        conn, now=leave + timedelta(seconds=900), expiration_buffer_seconds=10_000
    )
    assert n_res == 0
    assert (
        conn.execute("SELECT status FROM forecast_queue WHERE forecast_id = ?", [fid]).fetchone()[0]
        == "pending"
    )


def test_expires_after_buffer_when_unresolvable(conn: duckdb.DuckDBPyConnection):
    leave = T0
    fid = _enqueue(conn, leave_at=leave, boarding=40330, alighting=40540)
    n_res, n_unr = resolve_due_forecasts(
        conn, now=leave + timedelta(seconds=2000), expiration_buffer_seconds=300
    )
    assert n_unr == 1
    assert (
        conn.execute("SELECT status FROM forecast_queue WHERE forecast_id = ?", [fid]).fetchone()[0]
        == "unresolvable"
    )
