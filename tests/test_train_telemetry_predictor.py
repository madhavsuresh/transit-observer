"""Unit tests for ``predictors.train_telemetry.TrainTelemetryPredictor``."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from types import SimpleNamespace

import duckdb
import pytest

from transit_observer.config import CHICAGO
from transit_observer.corridors import Corridor
from transit_observer.db import init_schema
from transit_observer.predictors import train_telemetry
from transit_observer.predictors.train_telemetry import (
    TRAIN_TELEMETRY_VERSION,
    TrainTelemetryPredictor,
)
from transit_observer.predictors.registry import predictor, reset


BASE_MS = 1_700_000_000_000


@pytest.fixture()
def conn():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test.duckdb")
    conn = duckdb.connect(path)
    init_schema(conn)
    try:
        yield conn
    finally:
        conn.close()
        import shutil

        shutil.rmtree(tmpdir)


@pytest.fixture(autouse=True)
def reset_caches():
    reset()
    train_telemetry.reset_catalog()
    yield
    reset()
    train_telemetry.reset_catalog()


@pytest.fixture()
def fake_catalog(monkeypatch):
    """Inject an in-memory L station catalog. Real catalog has ~145
    stations; tests don't need them."""
    fake = [
        SimpleNamespace(
            map_id=40380, name="UIC-Halsted", latitude=41.8755, longitude=-87.6494,
            served_lines=("Blue",), stop_ids=(30074, 30075),
        ),
        SimpleNamespace(
            map_id=40570, name="Clinton", latitude=41.8769, longitude=-87.6411,
            served_lines=("Blue",), stop_ids=(30076, 30077),
        ),
    ]
    monkeypatch.setattr(
        "transit_observer.predictors.train_telemetry.load_catalog",
        lambda: fake,
    )
    train_telemetry.reset_catalog()
    yield fake


def _bus_corridor() -> Corridor:
    return Corridor(
        corridor_id="cta-bus-test",
        mode="bus", line="22", direction="southbound",
        origin_label="A", origin_latitude=0.0, origin_longitude=0.0,
        destination_label="B", destination_latitude=0.0, destination_longitude=0.0,
        boarding_int_id=0, boarding_text_id="1828",
        alighting_int_id=0, alighting_text_id="1869",
        schedule_headway_seconds=600.0,
    )


def _l_corridor() -> Corridor:
    return Corridor(
        corridor_id="cta-l-blue-test",
        mode="L", line="Blue", direction="inbound",
        origin_label="UIC-Halsted", origin_latitude=41.8755, origin_longitude=-87.6494,
        destination_label="Clinton", destination_latitude=41.8769, destination_longitude=-87.6411,
        boarding_int_id=40380, boarding_text_id=None,
        alighting_int_id=40570, alighting_text_id=None,
        schedule_headway_seconds=300.0,
    )


def _seed_arrival_and_position(
    conn: duckdb.DuckDBPyConnection,
    *,
    line: str = "Blue",
    run_number: str = "812",
    map_id: str = "40380",
    server_ms: int = BASE_MS,
    eta_seconds: int = 180,
    is_fault: bool = False,
    next_station_map_id: str | None = None,
) -> None:
    # api_poll for the ttarrivals row.
    conn.execute(
        """
        INSERT INTO train_v2_api_poll(poll_id, run_id, source, endpoint, params_json_redacted,
                                      local_request_start_ms, local_response_end_ms,
                                      cta_server_time_ms, ok, created_at_ms)
        VALUES (1, 'r1', 'train_tracker', 'ttarrivals.aspx', '{}', ?, ?, ?, TRUE, ?)
        """,
        [server_ms, server_ms + 500, server_ms, server_ms],
    )
    arr = server_ms + eta_seconds * 1000
    conn.execute(
        """
        INSERT INTO train_v2_arrival_observation(
            poll_id, run_id, cta_server_time_ms, local_response_end_ms, query_kind,
            line, run_number, map_id, stop_id, station_name, direction_code, destination_name,
            predicted_at_ms, arrival_at_ms, eta_s, prediction_age_s,
            is_approaching, is_delayed, is_fault, is_scheduled, raw_json
        ) VALUES (1, 'r1', ?, ?, 'arrivals_by_station', ?, ?, ?, '30074', 'UIC-Halsted',
                  '5', 'O''Hare', ?, ?, ?, 5.0, ?, FALSE, ?, FALSE, '{}')
        """,
        [
            server_ms, server_ms, line, run_number, map_id,
            server_ms - 5_000, arr, float(eta_seconds),
            True, is_fault,
        ],
    )

    # Position row for the same run.
    next_map = next_station_map_id or map_id
    conn.execute(
        """
        INSERT INTO train_v2_api_poll(poll_id, run_id, source, endpoint, params_json_redacted,
                                      local_request_start_ms, local_response_end_ms,
                                      cta_server_time_ms, ok, created_at_ms)
        VALUES (2, 'r1', 'train_tracker', 'ttpositions.aspx', '{}', ?, ?, ?, TRUE, ?)
        """,
        [server_ms, server_ms + 500, server_ms, server_ms],
    )
    conn.execute(
        """
        INSERT INTO train_v2_position_observation(
            poll_id, run_id, cta_server_time_ms, local_response_end_ms,
            line, run_number, direction_code, destination_name, destination_map_id,
            next_station_map_id, next_station_name,
            predicted_at_ms, next_arrival_at_ms,
            is_approaching, is_delayed, is_fault, lat, lon, heading, raw_json
        ) VALUES (2, 'r1', ?, ?, ?, ?, '5', 'O''Hare', '30171',
                  ?, 'UIC-Halsted',
                  ?, ?, TRUE, FALSE, FALSE, 41.875, -87.649, 270, '{}')
        """,
        [server_ms, server_ms, line, run_number, next_map, server_ms - 5_000, arr],
    )


def test_predict_returns_prediction_with_reliability_for_l_corridor(conn, fake_catalog):
    _seed_arrival_and_position(conn)
    pred = TrainTelemetryPredictor().predict(
        conn, _l_corridor(), now=datetime.fromtimestamp(BASE_MS / 1000, CHICAGO)
    )
    assert pred is not None
    assert pred.predictor_version == TRAIN_TELEMETRY_VERSION
    assert pred.feature_snapshot["display_state"]
    assert isinstance(pred.feature_snapshot["reliability"], float)
    assert pred.feature_snapshot["run_number"] == "812"
    assert pred.wait.p50 > 0
    assert pred.in_vehicle.p50 > 0


def test_predict_abstains_on_isFlt(conn, fake_catalog):
    _seed_arrival_and_position(conn, is_fault=True)
    pred = TrainTelemetryPredictor().predict(
        conn, _l_corridor(), now=datetime.fromtimestamp(BASE_MS / 1000, CHICAGO)
    )
    assert pred is None


def test_predict_returns_none_for_bus_corridor(conn, fake_catalog):
    pred = TrainTelemetryPredictor().predict(
        conn, _bus_corridor(), now=datetime.fromtimestamp(BASE_MS / 1000, CHICAGO)
    )
    assert pred is None


def test_registry_can_instantiate_train_telemetry():
    instance = predictor(TRAIN_TELEMETRY_VERSION)
    assert instance is not None
    assert isinstance(instance, TrainTelemetryPredictor)
    assert instance.predictor_version == TRAIN_TELEMETRY_VERSION
