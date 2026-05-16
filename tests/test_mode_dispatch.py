"""Resolver dispatches per-mode and audits the right way for each."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from transit_observer import db
from transit_observer.bus_predictor import (
    BusTripSpec,
    build_observed_bus_runs,
    enqueue_bus_forecast,
)
from transit_observer.catalog import BusStop, IntercampusStop, MetraStation
from transit_observer.intercampus_predictor import (
    IntercampusTripSpec,
    build_observed_intercampus_trips,
    enqueue_intercampus_forecast,
)
from transit_observer.journey.stop_arrival import LiveDeparture, WaitForecast
from transit_observer.journey.time_distribution import TimeDistributionSummary
from transit_observer.metra_predictor import (
    MetraTripSpec,
    build_observed_metra_trips,
    enqueue_metra_forecast,
)
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


def _wait(seconds: float) -> WaitForecast:
    summary = TimeDistributionSummary.analytic(mean=seconds, sigma=30, confidence=0.6)
    return WaitForecast(
        wait_distribution=summary,
        state=__import__("transit_observer.journey.stop_arrival", fromlist=["WaitReasonableness"]).WaitReasonableness.acceptable_wait,
        next_departure_at=T0 + timedelta(seconds=seconds),
        p_board_within_5_min=1.0,
        p_board_within_10_min=1.0,
        p_board_within_15_min=1.0,
        explanation=None,
    )


def test_bus_resolution_finds_vehicle_at_both_stops(conn: duckdb.DuckDBPyConnection):
    boarding = BusStop(stop_id=1106, route="22", name="Clark Belmont", latitude=41.94, longitude=-87.66, direction_label="Northbound")
    alighting = BusStop(stop_id=1107, route="22", name="Clark Howard", latitude=42.02, longitude=-87.67, direction_label="Northbound")
    spec = BusTripSpec(route="22", boarding=boarding, alighting=alighting, leave_at=T0)
    fid = enqueue_bus_forecast(
        conn,
        spec=spec,
        wait=_wait(60),
        in_vehicle=TimeDistributionSummary.analytic(mean=300, sigma=30, confidence=0.5),
        now=T0,
        snapshot_polled_at=T0,
    )

    # Seed bus_runs_observed
    for stop_id, offset in [(1106, 60), (1107, 360)]:
        conn.execute(
            """
            INSERT INTO bus_runs_observed (
                route, vehicle_id, stop_id, destination_name, direction_name,
                observed_arrival_at, first_seen_at, last_seen_at, sample_count, inferred_from
            ) VALUES ('22', '7777', ?, 'Howard', 'Northbound', ?, ?, ?, 3, 'approaching')
            """,
            [stop_id, T0 + timedelta(seconds=offset), T0, T0 + timedelta(seconds=offset)],
        )
    conn.execute("UPDATE forecast_queue SET resolve_after = ?", [T0 + timedelta(seconds=900)])
    n_res, _ = resolve_due_forecasts(conn, now=T0 + timedelta(seconds=900), expiration_buffer_seconds=300)
    assert n_res == 1
    row = conn.execute(
        "SELECT boarded_run_number, actual_total_seconds FROM forecast_outcomes WHERE forecast_id = ?",
        [fid],
    ).fetchone()
    assert row[0] == "7777"
    assert row[1] == 360


def test_metra_resolution_finds_trip_at_both_stations(conn: duckdb.DuckDBPyConnection):
    boarding = MetraStation(station_id="OGILVIE", name="Ogilvie", latitude=41.88, longitude=-87.64, served_routes=("UP-N",))
    alighting = MetraStation(station_id="DAVIS", name="Davis", latitude=42.04, longitude=-87.68, served_routes=("UP-N",))
    spec = MetraTripSpec(route_id="UP-N", boarding=boarding, alighting=alighting, leave_at=T0)
    fid = enqueue_metra_forecast(
        conn,
        spec=spec,
        wait=TimeDistributionSummary.analytic(mean=120, sigma=60, confidence=0.6),
        in_vehicle=TimeDistributionSummary.analytic(mean=1800, sigma=120, confidence=0.6),
        direction_id=1,
        now=T0,
        snapshot_polled_at=T0,
    )

    for station_id, offset in [("OGILVIE", 120), ("DAVIS", 2200)]:
        conn.execute(
            """
            INSERT INTO metra_trips_observed (
                route_id, trip_id, station_id, direction_id,
                observed_arrival_at, first_seen_at, last_seen_at, sample_count, inferred_from
            ) VALUES ('UP-N', 'UP-N-1234', ?, 1, ?, ?, ?, 3, 'dropoff')
            """,
            [station_id, T0 + timedelta(seconds=offset), T0, T0 + timedelta(seconds=offset)],
        )
    conn.execute("UPDATE forecast_queue SET resolve_after = ?", [T0 + timedelta(seconds=3000)])
    n_res, _ = resolve_due_forecasts(conn, now=T0 + timedelta(seconds=3000), expiration_buffer_seconds=300)
    assert n_res == 1
    row = conn.execute(
        "SELECT boarded_run_number, actual_total_seconds FROM forecast_outcomes WHERE forecast_id = ?",
        [fid],
    ).fetchone()
    assert row[0] == "UP-N-1234"
    assert row[1] == 2200


def test_intercampus_resolution(conn: duckdb.DuckDBPyConnection):
    boarding = IntercampusStop(stop_id="CHIC1", name="Chicago Campus", latitude=41.89, longitude=-87.61, served_directions=("northbound",))
    alighting = IntercampusStop(stop_id="EVAN1", name="Evanston Campus", latitude=42.05, longitude=-87.67, served_directions=("northbound",))
    spec = IntercampusTripSpec(direction="northbound", boarding=boarding, alighting=alighting, leave_at=T0)
    fid = enqueue_intercampus_forecast(
        conn,
        spec=spec,
        wait=TimeDistributionSummary.analytic(mean=120, sigma=60, confidence=0.6),
        in_vehicle=TimeDistributionSummary.analytic(mean=1500, sigma=120, confidence=0.6),
        direction="northbound",
        now=T0,
        snapshot_polled_at=T0,
    )

    for stop_id, offset in [("CHIC1", 120), ("EVAN1", 1700)]:
        conn.execute(
            """
            INSERT INTO intercampus_trips_observed (
                route_id, trip_id, stop_id, direction,
                observed_arrival_at, first_seen_at, last_seen_at, sample_count, inferred_from
            ) VALUES ('intercampus', 'IC-1', ?, 'northbound', ?, ?, ?, 3, 'dropoff')
            """,
            [stop_id, T0 + timedelta(seconds=offset), T0, T0 + timedelta(seconds=offset)],
        )
    conn.execute("UPDATE forecast_queue SET resolve_after = ?", [T0 + timedelta(seconds=2000)])
    n_res, _ = resolve_due_forecasts(conn, now=T0 + timedelta(seconds=2000), expiration_buffer_seconds=300)
    assert n_res == 1
    row = conn.execute(
        "SELECT boarded_run_number, actual_total_seconds FROM forecast_outcomes WHERE forecast_id = ?",
        [fid],
    ).fetchone()
    assert row[0] == "IC-1"
    assert row[1] == 1700


def test_per_mode_audit_records_each(conn: duckdb.DuckDBPyConnection):
    """The resolver should produce a direction_audit row marked mode='bus'."""
    boarding = BusStop(stop_id=10, route="X", name="A", latitude=0, longitude=0, direction_label="Northbound")
    alighting = BusStop(stop_id=11, route="X", name="B", latitude=0.1, longitude=0, direction_label="Northbound")
    spec = BusTripSpec(route="X", boarding=boarding, alighting=alighting, leave_at=T0)
    enqueue_bus_forecast(
        conn, spec=spec,
        wait=_wait(60),
        in_vehicle=TimeDistributionSummary.analytic(mean=300, sigma=30, confidence=0.5),
        now=T0, snapshot_polled_at=T0,
    )
    for stop_id, offset in [(10, 60), (11, 360)]:
        conn.execute(
            """
            INSERT INTO bus_runs_observed (
                route, vehicle_id, stop_id, destination_name, direction_name,
                observed_arrival_at, first_seen_at, last_seen_at, sample_count, inferred_from
            ) VALUES ('X', 'V', ?, 'End', 'Northbound', ?, ?, ?, 1, 'approaching')
            """,
            [stop_id, T0 + timedelta(seconds=offset), T0, T0 + timedelta(seconds=offset)],
        )
    # Seed a matching bus prediction so the audit has something to keep.
    conn.execute(
        """
        INSERT INTO bus_predictions_raw (
            polled_at, route, route_name, vehicle_id, stop_id, stop_name,
            destination_name, direction_name, generated_at, arrival_at,
            is_delayed, is_approaching
        ) VALUES (?, 'X', 'X', 'V', 10, 'A', 'End', 'Northbound', ?, ?, FALSE, FALSE)
        """,
        [T0, T0, T0 + timedelta(seconds=60)],
    )
    conn.execute("UPDATE forecast_queue SET resolve_after = ?", [T0 + timedelta(seconds=900)])
    resolve_due_forecasts(conn, now=T0 + timedelta(seconds=900), expiration_buffer_seconds=300)
    audited = conn.execute("SELECT mode, boarded_was_kept FROM direction_audit").fetchall()
    assert len(audited) == 1
    assert audited[0][0] == "bus"
    assert audited[0][1] is True
