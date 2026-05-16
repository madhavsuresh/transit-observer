"""End-to-end: synthetic arrivals → predict → enqueue → resolve."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from transit_observer import db
from transit_observer.catalog import LStation
from transit_observer.trip_generator import (
    TripSpec,
    direction_label,
    enqueue_forecast,
    haversine_meters,
    predict_trip,
    sample_trip,
)


T0 = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)


def _station(map_id: int, name: str, lat: float, lon: float, line: str = "red") -> LStation:
    return LStation(map_id=map_id, name=name, latitude=lat, longitude=lon, served_lines=(line,))


CHICAGO_STATE = _station(41450, "Chicago", 41.896671, -87.628176)
HOWARD = _station(40900, "Howard", 42.019063, -87.672892)
NINETY_FIFTH = _station(40450, "95th/Dan Ryan", 41.722377, -87.624342)


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    db.init_schema(c)
    try:
        yield c
    finally:
        c.close()


def _insert_arrivals(conn, *, line: str, map_id: int, minutes_ahead: list[float],
                    destination: str = "Howard", polled_at: datetime = T0) -> None:
    for m in minutes_ahead:
        arr = polled_at + timedelta(minutes=m)
        conn.execute(
            """
            INSERT INTO train_arrivals_raw (
                polled_at, line, run_number, map_id, stop_id, station_name,
                direction_code, destination_name, predicted_at, arrival_at,
                is_approaching, is_delayed, is_fault, is_scheduled
            ) VALUES (?, ?, ?, ?, 0, 'Test', '1', ?, ?, ?, FALSE, FALSE, FALSE, FALSE)
            """,
            [polled_at, line, f"R{int(m)}", map_id, destination, polled_at, arr],
        )


def test_haversine_chicago_to_howard_is_about_14km():
    distance = haversine_meters(
        CHICAGO_STATE.latitude, CHICAGO_STATE.longitude,
        HOWARD.latitude, HOWARD.longitude,
    )
    assert 13_000 < distance < 16_000


def test_direction_label_north_for_evanston_bound():
    assert direction_label(CHICAGO_STATE, HOWARD) == "north"
    assert direction_label(HOWARD, CHICAGO_STATE) == "south"


def test_sample_trip_returns_two_distinct_stations():
    catalog = [CHICAGO_STATE, HOWARD, NINETY_FIFTH]
    rng = random.Random(42)
    spec = sample_trip(catalog, rng=rng, leave_at=T0)
    assert spec is not None
    assert spec.boarding != spec.alighting


def test_predict_trip_yields_wait_and_invehicle(conn: duckdb.DuckDBPyConnection):
    spec = TripSpec(
        line_catalog="red",
        line_api="Red",
        boarding=CHICAGO_STATE,
        alighting=HOWARD,
        direction_label="north",
        leave_at=T0,
    )
    _insert_arrivals(conn, line="Red", map_id=CHICAGO_STATE.map_id,
                     minutes_ahead=[5, 13, 21, 29])
    forecast = predict_trip(conn, spec, now=T0)
    assert forecast is not None
    wait, in_vehicle = forecast
    assert wait.next_departure_at == T0 + timedelta(minutes=5)
    assert in_vehicle.mean > 0


def test_predict_trip_returns_none_without_arrivals(conn: duckdb.DuckDBPyConnection):
    spec = TripSpec(
        line_catalog="red",
        line_api="Red",
        boarding=CHICAGO_STATE,
        alighting=HOWARD,
        direction_label="north",
        leave_at=T0,
    )
    forecast = predict_trip(conn, spec, now=T0)
    assert forecast is None


def test_enqueue_forecast_persists_row(conn: duckdb.DuckDBPyConnection):
    spec = TripSpec(
        line_catalog="red",
        line_api="Red",
        boarding=CHICAGO_STATE,
        alighting=HOWARD,
        direction_label="north",
        leave_at=T0,
    )
    _insert_arrivals(conn, line="Red", map_id=CHICAGO_STATE.map_id,
                     minutes_ahead=[5, 13, 21, 29])
    forecast = predict_trip(conn, spec, now=T0)
    assert forecast is not None
    wait, in_vehicle = forecast
    fid = enqueue_forecast(conn, spec=spec, wait=wait, in_vehicle=in_vehicle,
                           now=T0, snapshot_polled_at=T0)
    row = conn.execute(
        "SELECT line, boarding_map_id, alighting_map_id, predicted_total_p80 FROM forecast_queue WHERE forecast_id = ?",
        [fid],
    ).fetchone()
    assert row[0] == "Red"
    assert row[1] == CHICAGO_STATE.map_id
    assert row[2] == HOWARD.map_id
    assert row[3] > 0
