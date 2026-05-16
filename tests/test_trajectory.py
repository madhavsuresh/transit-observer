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
    assert result[0] == "arrivals:approaching"
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
    assert result[0] == "arrivals:dropoff"


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


def _insert_position(
    conn: duckdb.DuckDBPyConnection,
    *,
    polled_at: datetime,
    run: str,
    next_map_id: int,
    next_arrival_at: datetime,
    is_approaching: bool = False,
    line: str = "Red",
    destination: str = "Howard",
    direction: str = "1",
) -> None:
    conn.execute(
        """
        INSERT INTO train_positions_raw (
            polled_at, line, run_number, destination_name, direction_code,
            next_station_map_id, next_station_name,
            predicted_at, next_arrival_at, is_approaching, is_delayed
        ) VALUES (?, ?, ?, ?, ?, ?, 'Next', ?, ?, ?, FALSE)
        """,
        [polled_at, line, run, destination, direction, next_map_id, polled_at, next_arrival_at, is_approaching],
    )


def test_positions_approaching_wins_over_arrivals(conn: duckdb.DuckDBPyConnection):
    # Arrivals say the train was approaching at T0+90, predicted to arrive at T0+120.
    _insert(conn, polled_at=T0 + timedelta(seconds=90), run="R5", map_id=40330,
            arrival_at=T0 + timedelta(seconds=120), is_approaching=True)
    # Positions say the train physically arrived earlier — polled at T0+105 with isApp.
    _insert_position(conn, polled_at=T0 + timedelta(seconds=105), run="R5", next_map_id=40330,
                     next_arrival_at=T0 + timedelta(seconds=110), is_approaching=True)
    build_observed_runs(conn, now=T0 + timedelta(minutes=10))
    row = conn.execute(
        "SELECT inferred_from, observed_arrival_at FROM train_runs_observed"
    ).fetchone()
    assert row[0] == "positions:approaching"
    assert row[1] == T0 + timedelta(seconds=105)


def test_position_only_sample_resolves_to_polled_at(conn: duckdb.DuckDBPyConnection):
    _insert_position(conn, polled_at=T0 + timedelta(seconds=200), run="R6", next_map_id=40330,
                     next_arrival_at=T0 + timedelta(seconds=210), is_approaching=True)
    rows = build_observed_runs(conn, now=T0 + timedelta(minutes=10))
    assert rows == 1
    row = conn.execute(
        "SELECT inferred_from, observed_arrival_at FROM train_runs_observed"
    ).fetchone()
    assert row[0] == "positions:approaching"
    assert row[1] == T0 + timedelta(seconds=200)


def test_multiple_runs_recorded_independently(conn: duckdb.DuckDBPyConnection):
    _insert(conn, polled_at=T0, run="A", map_id=40330, arrival_at=T0 + timedelta(seconds=60), is_approaching=True)
    _insert(conn, polled_at=T0, run="B", map_id=40330, arrival_at=T0 + timedelta(seconds=420), is_approaching=True)
    _insert(conn, polled_at=T0, run="A", map_id=40540, arrival_at=T0 + timedelta(seconds=540), is_approaching=True)
    build_observed_runs(conn, now=T0 + timedelta(minutes=10))
    n = conn.execute("SELECT COUNT(*) FROM train_runs_observed").fetchone()[0]
    assert n == 3
