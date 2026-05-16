"""Trajectory reconstruction from raw arrival snapshots."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from transit_observer import db
from transit_observer.trajectory import build_observed_runs


T0 = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    db.init_schema(c)
    try:
        yield c
    finally:
        c.close()


def _insert(conn: duckdb.DuckDBPyConnection, *, polled_at: datetime, run: str, map_id: int,
            arrival_at: datetime, is_approaching: bool = False, line: str = "Red",
            destination: str = "Howard") -> None:
    conn.execute(
        """
        INSERT INTO train_arrivals_raw (
            polled_at, line, run_number, map_id, stop_id, station_name,
            direction_code, destination_name, predicted_at, arrival_at,
            is_approaching, is_delayed, is_fault, is_scheduled
        ) VALUES (?, ?, ?, ?, 0, 'Test', '1', ?, ?, ?, ?, FALSE, FALSE, FALSE)
        """,
        [polled_at, line, run, map_id, destination, polled_at, arrival_at, is_approaching],
    )


def test_approaching_sample_observes_arrival(conn: duckdb.DuckDBPyConnection):
    _insert(conn, polled_at=T0, run="R1", map_id=40330, arrival_at=T0 + timedelta(seconds=120))
    _insert(conn, polled_at=T0 + timedelta(seconds=90), run="R1", map_id=40330,
            arrival_at=T0 + timedelta(seconds=120), is_approaching=True)
    rows = build_observed_runs(conn, now=T0 + timedelta(minutes=10))
    assert rows == 1
    result = conn.execute(
        "SELECT inferred_from, observed_arrival_at FROM train_runs_observed"
    ).fetchone()
    assert result[0] == "approaching"
    assert result[1] == T0 + timedelta(seconds=120)


def test_dropoff_when_prediction_passes_observed_time(conn: duckdb.DuckDBPyConnection):
    _insert(conn, polled_at=T0, run="R2", map_id=40330, arrival_at=T0 + timedelta(seconds=60))
    _insert(conn, polled_at=T0 + timedelta(seconds=30), run="R2", map_id=40330,
            arrival_at=T0 + timedelta(seconds=60))
    rows = build_observed_runs(conn, now=T0 + timedelta(minutes=10))
    assert rows == 1
    result = conn.execute(
        "SELECT inferred_from, observed_arrival_at FROM train_runs_observed"
    ).fetchone()
    assert result[0] == "dropoff"


def test_future_prediction_is_not_yet_observed(conn: duckdb.DuckDBPyConnection):
    _insert(conn, polled_at=T0, run="R3", map_id=40330, arrival_at=T0 + timedelta(minutes=5))
    rows = build_observed_runs(conn, now=T0 + timedelta(minutes=1))
    assert rows == 0


def test_each_run_station_pair_records_once(conn: duckdb.DuckDBPyConnection):
    for offset in range(0, 60, 10):
        _insert(conn, polled_at=T0 + timedelta(seconds=offset), run="R4", map_id=40330,
                arrival_at=T0 + timedelta(seconds=60))
    _insert(conn, polled_at=T0 + timedelta(seconds=70), run="R4", map_id=40330,
            arrival_at=T0 + timedelta(seconds=60), is_approaching=True)
    build_observed_runs(conn, now=T0 + timedelta(minutes=10))
    n = conn.execute("SELECT COUNT(*) FROM train_runs_observed").fetchone()[0]
    assert n == 1


def test_multiple_runs_recorded_independently(conn: duckdb.DuckDBPyConnection):
    _insert(conn, polled_at=T0, run="A", map_id=40330, arrival_at=T0 + timedelta(seconds=60), is_approaching=True)
    _insert(conn, polled_at=T0, run="B", map_id=40330, arrival_at=T0 + timedelta(seconds=420), is_approaching=True)
    _insert(conn, polled_at=T0, run="A", map_id=40540, arrival_at=T0 + timedelta(seconds=540), is_approaching=True)
    build_observed_runs(conn, now=T0 + timedelta(minutes=10))
    n = conn.execute("SELECT COUNT(*) FROM train_runs_observed").fetchone()[0]
    assert n == 3
