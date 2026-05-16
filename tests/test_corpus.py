"""Corridor-driven prediction: build forecast_queue rows tagged with
corridor_id, predictor_version, and feature_json."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from transit_observer import db
from transit_observer.corpus import PREDICTOR_VERSION, predict_and_enqueue_corridor
from transit_observer.corridors import SEED_CORRIDORS, by_id, seed_corridors


T0 = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    db.init_schema(c)
    seed_corridors(c, now=T0)
    try:
        yield c
    finally:
        c.close()


def _seed_l_arrivals(conn, *, line: str, map_id: int, leave_at: datetime,
                     departures_offset_s: list[int], destination: str = "Loop") -> None:
    """Seed train_arrivals_raw with several predictions for a single station."""
    for offset in departures_offset_s:
        conn.execute(
            """
            INSERT INTO train_arrivals_raw (
                polled_at, line, run_number, map_id, stop_id, station_name,
                direction_code, destination_name, predicted_at, arrival_at,
                is_approaching, is_delayed, is_fault, is_scheduled
            ) VALUES (?, ?, 'R1', ?, 0, 'A', '1', ?, ?, ?, ?, FALSE, FALSE, ?)
            """,
            [
                leave_at - timedelta(seconds=30),
                line, map_id, destination,
                leave_at - timedelta(seconds=30), leave_at + timedelta(seconds=offset),
                offset < 60, False,  # is_approaching when very close
            ],
        )


def test_predict_and_enqueue_l_corridor(conn: duckdb.DuckDBPyConnection):
    corridor = by_id()["cta-red-belmont-lake-sb"]
    _seed_l_arrivals(
        conn,
        line=corridor.line,
        map_id=corridor.boarding_int_id,
        leave_at=T0,
        departures_offset_s=[120, 600, 1200],
    )

    result = predict_and_enqueue_corridor(conn, corridor, now=T0)
    assert result is not None
    assert result.corridor_id == corridor.corridor_id
    assert result.predictor_version == PREDICTOR_VERSION

    row = conn.execute(
        """
        SELECT corridor_id, predictor_version, feature_json, mode, line,
               boarding_map_id, alighting_map_id, status
          FROM forecast_queue
         WHERE forecast_id = ?
        """,
        [result.forecast_id],
    ).fetchone()
    assert row is not None
    assert row[0] == corridor.corridor_id
    assert row[1] == PREDICTOR_VERSION
    feature = json.loads(row[2])
    assert feature["mode"] == "L"
    assert feature["line"] == corridor.line
    assert len(feature["live_departures"]) >= 1
    assert row[3] == "L"
    assert row[4] == corridor.line
    assert row[5] == corridor.boarding_int_id
    assert row[6] == corridor.alighting_int_id
    assert row[7] == "pending"


def test_predict_and_enqueue_returns_none_without_data(conn: duckdb.DuckDBPyConnection):
    corridor = by_id()["cta-red-belmont-lake-sb"]
    # No arrivals seeded -- predictor has nothing to work with.
    result = predict_and_enqueue_corridor(conn, corridor, now=T0)
    assert result is None
    # last_predicted_at still bumped so we don't hot-loop on a starved corridor.
    last = conn.execute(
        "SELECT last_predicted_at FROM corridors WHERE corridor_id = ?",
        [corridor.corridor_id],
    ).fetchone()[0]
    assert last == T0


def test_predict_and_enqueue_metra_corridor(conn: duckdb.DuckDBPyConnection):
    corridor = by_id()["metra-upn-evanston-otc-ib"]
    # Seed two viable trips for UP-N at both stations.
    for trip_id, board_off, alight_off in [("T-A", 300, 2400), ("T-B", 900, 3000)]:
        for station_id, off in [
            (corridor.boarding_text_id, board_off),
            (corridor.alighting_text_id, alight_off),
        ]:
            conn.execute(
                """
                INSERT INTO metra_arrivals_raw (
                    polled_at, route_id, trip_id, station_id, direction_id,
                    schedule_relationship, scheduled_at, predicted_at, delay_seconds
                ) VALUES (?, ?, ?, ?, 0, 'SCHEDULED', ?, ?, 0)
                """,
                [
                    T0 - timedelta(seconds=30),
                    corridor.line, trip_id, station_id,
                    T0 + timedelta(seconds=off), T0 + timedelta(seconds=off),
                ],
            )

    result = predict_and_enqueue_corridor(conn, corridor, now=T0)
    assert result is not None
    row = conn.execute(
        """
        SELECT corridor_id, predictor_version, mode, line,
               boarding_text_id, alighting_text_id, feature_json
          FROM forecast_queue WHERE forecast_id = ?
        """,
        [result.forecast_id],
    ).fetchone()
    assert row[0] == corridor.corridor_id
    assert row[2] == "metra"
    assert row[3] == "UP-N"
    assert row[4] == corridor.boarding_text_id
    assert row[5] == corridor.alighting_text_id
    feature = json.loads(row[6])
    assert feature["mode"] == "metra"
    assert any(t["trip_id"] == "T-A" for t in feature["viable_trips"])


def test_feature_json_records_time_features(conn: duckdb.DuckDBPyConnection):
    corridor = by_id()["cta-red-belmont-lake-sb"]
    _seed_l_arrivals(
        conn, line=corridor.line, map_id=corridor.boarding_int_id,
        leave_at=T0, departures_offset_s=[120, 600],
    )
    result = predict_and_enqueue_corridor(conn, corridor, now=T0)
    assert result is not None
    feature_json = conn.execute(
        "SELECT feature_json FROM forecast_queue WHERE forecast_id = ?", [result.forecast_id]
    ).fetchone()[0]
    feature = json.loads(feature_json)
    # 2026-01-01 is a Thursday -> weekday 3.
    assert feature["hour_of_day"] == T0.hour
    assert feature["minute_of_hour"] == T0.minute
    assert feature["dow"] == T0.weekday()
