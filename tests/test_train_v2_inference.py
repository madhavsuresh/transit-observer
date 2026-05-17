"""Synthetic-fixture tests for train_v2 arrival inference.

Seed ``train_v2_position_observation`` directly (skipping the API
layer) so we can drive the algorithm under controlled conditions. Each
test asserts the label and high_confidence flag produced by
``infer_train_arrivals``.
"""

from __future__ import annotations

import os
import tempfile

import duckdb
import pytest

from transit_observer.db import init_schema
from transit_observer.train_v2.inference import infer_train_arrivals


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


def _seed_position(
    conn: duckdb.DuckDBPyConnection,
    *,
    poll_id: int,
    run_number: str,
    line: str,
    direction_code: str,
    destination_map_id: str,
    next_station_map_id: str,
    predicted_at_ms: int,
    server_ms: int | None = None,
    is_fault: bool = False,
    run_id: str = "r1",
) -> None:
    server_ms = server_ms or predicted_at_ms
    conn.execute(
        """
        INSERT INTO train_v2_api_poll(poll_id, run_id, source, endpoint, params_json_redacted,
                                      local_request_start_ms, local_response_end_ms,
                                      cta_server_time_ms, ok, created_at_ms)
        VALUES (?, ?, 'train_tracker', 'ttpositions.aspx', '{}', ?, ?, ?, TRUE, ?)
        """,
        [poll_id, run_id, predicted_at_ms, predicted_at_ms + 500, server_ms, predicted_at_ms],
    )
    conn.execute(
        """
        INSERT INTO train_v2_position_observation(
            poll_id, run_id, cta_server_time_ms, local_response_end_ms,
            line, run_number, direction_code, destination_map_id, destination_name,
            next_station_map_id, next_station_name,
            predicted_at_ms, next_arrival_at_ms,
            is_approaching, is_delayed, is_fault, lat, lon, heading, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'O''Hare', ?, 'NextStation',
                  ?, ?, FALSE, FALSE, ?, 41.875, -87.649, 270, '{}')
        """,
        [
            poll_id, run_id, server_ms, predicted_at_ms + 500,
            line, run_number, direction_code, destination_map_id,
            next_station_map_id,
            predicted_at_ms, predicted_at_ms + 90_000,
            is_fault,
        ],
    )


def test_high_confidence_arrival_on_next_station_transition(conn):
    """Two consecutive obs with different next_station → ARRIVED_CONFIRMED."""
    _seed_position(conn, poll_id=1, run_number="812", line="Blue", direction_code="5",
                   destination_map_id="30171", next_station_map_id="40380",
                   predicted_at_ms=BASE_MS)
    _seed_position(conn, poll_id=2, run_number="812", line="Blue", direction_code="5",
                   destination_map_id="30171", next_station_map_id="40570",
                   predicted_at_ms=BASE_MS + 30_000)
    n = infer_train_arrivals(conn, run_id="r1", replace=True)
    assert n == 1
    label, high, actual = conn.execute(
        "SELECT label, high_confidence, actual_arrival_ms FROM train_v2_arrival_event"
    ).fetchone()
    assert label == "ARRIVED_CONFIRMED"
    assert high is True
    # Midpoint interpolation.
    assert actual == BASE_MS + 15_000


def test_faulted_tracking_blocks_high_confidence(conn):
    _seed_position(conn, poll_id=1, run_number="812", line="Blue", direction_code="5",
                   destination_map_id="30171", next_station_map_id="40380",
                   predicted_at_ms=BASE_MS, is_fault=True)
    _seed_position(conn, poll_id=2, run_number="812", line="Blue", direction_code="5",
                   destination_map_id="30171", next_station_map_id="40570",
                   predicted_at_ms=BASE_MS + 30_000)
    n = infer_train_arrivals(conn, run_id="r1", replace=True)
    assert n == 1
    label, high = conn.execute(
        "SELECT label, high_confidence FROM train_v2_arrival_event"
    ).fetchone()
    assert label == "FAULTED_TRACKING"
    assert high is False


def test_stale_predictions_downgrade(conn):
    """Both predictions are too old vs server clock → STALE_DATA."""
    # predicted_at is 5 min before server_ms = 300 s, exceeds 60 s freshness.
    _seed_position(
        conn, poll_id=1, run_number="812", line="Blue", direction_code="5",
        destination_map_id="30171", next_station_map_id="40380",
        predicted_at_ms=BASE_MS - 300_000, server_ms=BASE_MS,
    )
    _seed_position(
        conn, poll_id=2, run_number="812", line="Blue", direction_code="5",
        destination_map_id="30171", next_station_map_id="40570",
        predicted_at_ms=BASE_MS - 270_000, server_ms=BASE_MS,
    )
    n = infer_train_arrivals(conn, run_id="r1", replace=True)
    assert n == 1
    label, high = conn.execute(
        "SELECT label, high_confidence FROM train_v2_arrival_event"
    ).fetchone()
    assert label == "STALE_DATA"
    assert high is False


def test_destination_flip_marks_rerouted(conn):
    _seed_position(conn, poll_id=1, run_number="812", line="Blue", direction_code="5",
                   destination_map_id="30171", next_station_map_id="40380",
                   predicted_at_ms=BASE_MS)
    _seed_position(conn, poll_id=2, run_number="812", line="Blue", direction_code="5",
                   destination_map_id="30172", next_station_map_id="40570",
                   predicted_at_ms=BASE_MS + 30_000)
    n = infer_train_arrivals(conn, run_id="r1", replace=True)
    label, high = conn.execute(
        "SELECT label, high_confidence FROM train_v2_arrival_event"
    ).fetchone()
    assert label == "REROUTED_OR_SHORT_TURN"
    assert high is False


def test_same_next_station_no_event(conn):
    """Two consecutive obs with the SAME next station → no event."""
    _seed_position(conn, poll_id=1, run_number="812", line="Blue", direction_code="5",
                   destination_map_id="30171", next_station_map_id="40380",
                   predicted_at_ms=BASE_MS)
    _seed_position(conn, poll_id=2, run_number="812", line="Blue", direction_code="5",
                   destination_map_id="30171", next_station_map_id="40380",
                   predicted_at_ms=BASE_MS + 30_000)
    n = infer_train_arrivals(conn, run_id="r1", replace=True)
    assert n == 0


def test_long_gap_between_obs_downgrades(conn):
    """Gap > 300 s → CENSORED_UNKNOWN (we missed too many positions)."""
    _seed_position(conn, poll_id=1, run_number="812", line="Blue", direction_code="5",
                   destination_map_id="30171", next_station_map_id="40380",
                   predicted_at_ms=BASE_MS)
    _seed_position(conn, poll_id=2, run_number="812", line="Blue", direction_code="5",
                   destination_map_id="30171", next_station_map_id="40570",
                   predicted_at_ms=BASE_MS + 400_000)  # 6:40 gap
    n = infer_train_arrivals(conn, run_id="r1", replace=True)
    label, high = conn.execute(
        "SELECT label, high_confidence FROM train_v2_arrival_event"
    ).fetchone()
    assert label == "CENSORED_UNKNOWN"
    assert high is False


def test_idempotent_inference(conn):
    """Re-running infer with the same run_id is a no-op (dedup)."""
    _seed_position(conn, poll_id=1, run_number="812", line="Blue", direction_code="5",
                   destination_map_id="30171", next_station_map_id="40380",
                   predicted_at_ms=BASE_MS)
    _seed_position(conn, poll_id=2, run_number="812", line="Blue", direction_code="5",
                   destination_map_id="30171", next_station_map_id="40570",
                   predicted_at_ms=BASE_MS + 30_000)
    assert infer_train_arrivals(conn, run_id="r1", replace=False) == 1
    assert infer_train_arrivals(conn, run_id="r1", replace=False) == 0
    assert conn.execute("SELECT COUNT(*) FROM train_v2_arrival_event").fetchone()[0] == 1
