"""Tier-3 data-capture tests.

Three pieces:

1. ID bridges:
   - refresh_run_vehicle_links links co-located Train Tracker positions
     to GTFS-RT vehicle positions.
   - refresh_station_id_map projects GTFS-static parent_station ↔
     child stop_id mappings into cta_station_id_map.
2. Pace GTFS-RT normalizers write into the pace_gtfsrt_* tables.
3. CDOT traffic client + storage writes both segment and region rows.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

import duckdb
import pytest

from transit_observer.cdot_traffic_client import (
    CDOTTrafficObservation,
    insert_cdot_observations,
)
from transit_observer.cta_id_bridges import (
    refresh_run_vehicle_links,
    refresh_station_id_map,
)
from transit_observer.db import init_schema
from transit_observer.pace_gtfsrt import (
    insert_pace_api_poll,
    normalize_pace_alerts,
    normalize_pace_trip_updates,
    normalize_pace_vehicle_positions,
)


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


BASE_MS = 1_700_000_000_000


# ---------------------------------------------------------------------------
# 1a. refresh_run_vehicle_links.
# ---------------------------------------------------------------------------

def _seed_train_position(conn, poll_id, run_number, line, lat, lon, t_ms):
    conn.execute(
        """
        INSERT INTO train_v2_api_poll(poll_id, run_id, source, endpoint, params_json_redacted,
                                      local_request_start_ms, local_response_end_ms, ok, created_at_ms)
        VALUES (?, 'r1', 'train_tracker', 'ttpositions.aspx', '{}', ?, ?, TRUE, ?)
        """,
        [poll_id, t_ms, t_ms, t_ms],
    )
    conn.execute(
        """
        INSERT INTO train_v2_position_observation(
            poll_id, run_id, cta_server_time_ms, local_response_end_ms,
            line, run_number, lat, lon
        ) VALUES (?, 'r1', ?, ?, ?, ?, ?, ?)
        """,
        [poll_id, t_ms, t_ms, line, run_number, lat, lon],
    )


def _seed_gtfsrt_vehicle(conn, poll_id, vehicle_id, route_id, lat, lon, t_ms):
    conn.execute(
        """
        INSERT INTO train_v2_api_poll(poll_id, run_id, source, endpoint, params_json_redacted,
                                      local_request_start_ms, local_response_end_ms, ok, created_at_ms)
        VALUES (?, 'r1', 'gtfsrt_train', 'gtfsrt', '{}', ?, ?, TRUE, ?)
        """,
        [poll_id, t_ms, t_ms, t_ms],
    )
    conn.execute(
        """
        INSERT INTO train_v2_gtfsrt_vehicle_position(
            poll_id, run_id, local_response_end_ms, route_id, vehicle_id, lat, lon
        ) VALUES (?, 'r1', ?, ?, ?, ?, ?)
        """,
        [poll_id, t_ms, route_id, vehicle_id, lat, lon],
    )


def test_run_vehicle_links_matches_close_lat_lon(conn):
    """Three co-located observation pairs → one link row."""
    # Same line, 3 timestamps, near-identical lat/lon.
    for i, (lat, lon, dt) in enumerate(
        [(41.875, -87.649, 0), (41.876, -87.650, 30_000), (41.877, -87.651, 60_000)],
        start=1,
    ):
        _seed_train_position(
            conn, poll_id=i * 2 - 1,
            run_number="812", line="Blue",
            lat=lat, lon=lon, t_ms=BASE_MS + dt,
        )
        _seed_gtfsrt_vehicle(
            conn, poll_id=i * 2,
            vehicle_id="V01", route_id="Blue",
            lat=lat + 0.0005, lon=lon + 0.0005,  # ~70m offset
            t_ms=BASE_MS + dt,
        )
    n = refresh_run_vehicle_links(
        conn, max_haversine_m=250.0, max_delta_s=90.0, cutoff_ms=0,
    )
    assert n == 1
    row = conn.execute(
        """
        SELECT line, run_number, gtfsrt_vehicle_id, n_observations, mean_haversine_m
          FROM train_v2_run_vehicle_link
        """
    ).fetchone()
    assert row[0:3] == ("Blue", "812", "V01")
    # The SQL JOIN cross-products positions × vehicles within the time
    # window, so the bucket's n_observations counts pairs, not raw
    # observations. What matters for confidence is the mean haversine
    # staying well under the 250m matching ceiling.
    assert row[3] >= 3
    assert row[4] < 250


def test_run_vehicle_links_rejects_far_pairs(conn):
    _seed_train_position(conn, 1, "812", "Blue", 41.875, -87.649, BASE_MS)
    # Vehicle ~5 km away — should be filtered out.
    _seed_gtfsrt_vehicle(conn, 2, "V99", "Blue", 41.925, -87.649, BASE_MS)
    _seed_train_position(conn, 3, "812", "Blue", 41.876, -87.650, BASE_MS + 30_000)
    _seed_gtfsrt_vehicle(conn, 4, "V99", "Blue", 41.926, -87.650, BASE_MS + 30_000)
    n = refresh_run_vehicle_links(conn, max_haversine_m=200.0, cutoff_ms=0)
    assert n == 0
    assert conn.execute("SELECT COUNT(*) FROM train_v2_run_vehicle_link").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# 1b. refresh_station_id_map.
# ---------------------------------------------------------------------------

def _seed_gtfs_snapshot(conn) -> int:
    sid = conn.execute(
        """
        INSERT INTO gtfs_static_snapshot(agency, archive_path, fetched_at_ms, archive_sha256)
        VALUES ('cta', '/tmp/dummy.zip', 1, 'sha')
        RETURNING snapshot_id
        """
    ).fetchone()[0]
    # Parent station (location_type = 1).
    conn.execute(
        """
        INSERT INTO gtfs_static_stops(snapshot_id, stop_id, stop_name, stop_lat, stop_lon, location_type)
        VALUES (?, '40380', 'UIC-Halsted', 41.8756, -87.6495, 1)
        """,
        [sid],
    )
    # Two child platforms (location_type = 0).
    conn.execute(
        """
        INSERT INTO gtfs_static_stops(snapshot_id, stop_id, stop_name, stop_lat, stop_lon, location_type, parent_station)
        VALUES (?, '30074', 'UIC-Halsted (Blue Line, O''Hare-bound)', 41.8757, -87.6494, 0, '40380'),
               (?, '30075', 'UIC-Halsted (Blue Line, Forest Park-bound)', 41.8755, -87.6496, 0, '40380')
        """,
        [sid, sid],
    )
    return int(sid)


def test_station_id_map_pairs_parent_to_each_child(conn):
    sid = _seed_gtfs_snapshot(conn)
    n = refresh_station_id_map(conn, snapshot_id=sid)
    assert n == 2
    rows = conn.execute(
        """
        SELECT map_id, parent_station, child_stop_id, child_stop_name
          FROM cta_station_id_map
         ORDER BY child_stop_id
        """
    ).fetchall()
    assert rows[0][:3] == ("40380", "40380", "30074")
    assert rows[1][:3] == ("40380", "40380", "30075")
    # Names from gtfs_static_stops carry through.
    assert "O'Hare-bound" in rows[0][3]


def test_station_id_map_replaces_on_rerun(conn):
    sid = _seed_gtfs_snapshot(conn)
    refresh_station_id_map(conn, snapshot_id=sid)
    refresh_station_id_map(conn, snapshot_id=sid)
    n = conn.execute("SELECT COUNT(*) FROM cta_station_id_map").fetchone()[0]
    assert n == 2  # not 4


# ---------------------------------------------------------------------------
# 2. Pace GTFS-RT normalizers.
# ---------------------------------------------------------------------------

def _pace_poll(conn) -> int:
    return insert_pace_api_poll(
        conn,
        run_id="pace_test",
        cycle_index=0,
        endpoint="gtfsrt",
        query_kind="vehicle_positions",
        request_url_redacted="http://example.test/pace",
        local_request_start_ms=BASE_MS,
        local_response_end_ms=BASE_MS + 500,
        http_status=200,
        latency_ms=500.0,
        ok=True,
        error_message=None,
    )


def test_pace_vehicle_positions_round_trip(conn):
    """A minimal vehicle-position object survives the normalizer."""
    from types import SimpleNamespace

    poll_id = _pace_poll(conn)
    row = SimpleNamespace(
        feed_timestamp=datetime.fromtimestamp(BASE_MS / 1000, tz=timezone.utc),
        feed_incrementality="full_dataset",
        vehicle_timestamp=datetime.fromtimestamp((BASE_MS + 100) / 1000, tz=timezone.utc),
        route_id="304",
        trip_id="P304-001",
        trip_start_date="20260517",
        trip_start_time="08:30:00",
        trip_direction_id=0,
        trip_schedule_relationship="scheduled",
        vehicle_id="P-7531",
        vehicle_label="7531",
        vehicle_license_plate=None,
        stop_id="STOP1",
        current_stop_sequence=4,
        current_status="in_transit_to",
        congestion_level="running_smoothly",
        occupancy_status="few_seats_available",
        occupancy_percentage=33,
        multi_carriage_details_json=None,
        lat=41.92,
        lon=-87.68,
        bearing=180.0,
        speed_mps=11.5,
        odometer_m=98765.4,
    )
    n = normalize_pace_vehicle_positions(
        conn, poll_id, "pace_test", [row], local_response_end_ms=BASE_MS + 500,
    )
    assert n == 1
    out = conn.execute(
        """
        SELECT route_id, vehicle_id, stop_id, occupancy_percentage, lat, lon,
               feed_timestamp_ms, trip_start_date, odometer_m
          FROM pace_gtfsrt_vehicle_position
        """
    ).fetchone()
    assert out[0] == "304"
    assert out[1] == "P-7531"
    assert out[2] == "STOP1"
    assert out[3] == 33
    assert out[4] == pytest.approx(41.92)
    assert out[5] == pytest.approx(-87.68)
    assert out[6] == BASE_MS  # tz-aware datetime → ms
    assert out[7] == "20260517"
    assert out[8] == pytest.approx(98765.4)


def test_pace_trip_update_round_trip(conn):
    from types import SimpleNamespace

    poll_id = _pace_poll(conn)
    row = SimpleNamespace(
        feed_timestamp=datetime.fromtimestamp(BASE_MS / 1000, tz=timezone.utc),
        feed_incrementality="full_dataset",
        trip_update_timestamp=datetime.fromtimestamp(BASE_MS / 1000, tz=timezone.utc),
        trip_update_delay_seconds=45,
        route_id="304",
        trip_id="P304-001",
        trip_start_date="20260517",
        trip_start_time="08:30:00",
        trip_direction_id=1,
        trip_schedule_relationship="scheduled",
        stop_id="S1",
        stop_sequence=3,
        arrival_time=datetime.fromtimestamp((BASE_MS + 60_000) / 1000, tz=timezone.utc),
        arrival_delay_seconds=30,
        arrival_uncertainty_seconds=15,
        departure_time=None,
        departure_delay_seconds=None,
        departure_uncertainty_seconds=None,
        schedule_relationship="scheduled",
        vehicle_id="P-7531",
        vehicle_label="7531",
        vehicle_license_plate=None,
    )
    normalize_pace_trip_updates(
        conn, poll_id, "pace_test", [row], local_response_end_ms=BASE_MS + 500,
    )
    out = conn.execute(
        """
        SELECT route_id, stop_id, arrival_time_ms, arrival_delay_seconds,
               trip_update_delay_seconds, arrival_uncertainty_seconds
          FROM pace_gtfsrt_trip_update
        """
    ).fetchone()
    assert out == ("304", "S1", BASE_MS + 60_000, 30, 45, 15)


def test_pace_alerts_round_trip(conn):
    from types import SimpleNamespace

    poll_id = _pace_poll(conn)
    row = SimpleNamespace(
        feed_timestamp=datetime.fromtimestamp(BASE_MS / 1000, tz=timezone.utc),
        feed_incrementality="full_dataset",
        entity_id="pace-alert-1",
        cause="maintenance",
        effect="detour",
        severity_level="warning",
        header_text="Route 304 detour",
        description_text="Construction on Cicero Ave.",
        tts_header_text=None,
        tts_description_text=None,
        url="https://pacebus.com/alerts/304",
        active_period_json="[]",
        informed_entity_json='[{"route_id":"304"}]',
    )
    normalize_pace_alerts(
        conn, poll_id, "pace_test", [row], local_response_end_ms=BASE_MS + 500,
    )
    out = conn.execute(
        """
        SELECT entity_id, cause, effect, severity_level, header_text
          FROM pace_gtfsrt_alert
        """
    ).fetchone()
    assert out == ("pace-alert-1", "maintenance", "detour", "warning", "Route 304 detour")


# ---------------------------------------------------------------------------
# 3. CDOT traffic insert path.
# ---------------------------------------------------------------------------

def test_cdot_observations_insert(conn):
    obs = [
        CDOTTrafficObservation(
            source="cdot_segments",
            polled_at_ms=BASE_MS,
            segment_id="seg-100",
            region="3",
            street="Lake Shore Dr",
            direction="NB",
            from_street="Roosevelt",
            to_street="Grand",
            speed=22.5,
            bus_count=4,
            message_count=3,
            hour_of_day=8,
            last_updated_ms=BASE_MS - 60_000,
            start_lat=41.86,
            start_lon=-87.62,
            end_lat=41.89,
            end_lon=-87.62,
            raw_payload_json=json.dumps({"segmentid": "seg-100"}),
        ),
        CDOTTrafficObservation(
            source="cdot_regions",
            polled_at_ms=BASE_MS,
            segment_id=None,
            region="3",
            street=None,
            direction=None,
            from_street=None,
            to_street=None,
            speed=24.0,
            bus_count=20,
            message_count=15,
            hour_of_day=8,
            last_updated_ms=BASE_MS - 60_000,
            start_lat=41.88,
            start_lon=-87.62,
            end_lat=None,
            end_lon=None,
            raw_payload_json=json.dumps({"region": "3"}),
        ),
    ]
    n = insert_cdot_observations(conn, obs)
    assert n == 2
    rows = conn.execute(
        "SELECT source, region, speed FROM cdot_traffic_observation ORDER BY source"
    ).fetchall()
    assert rows[0] == ("cdot_regions", "3", pytest.approx(24.0))
    assert rows[1] == ("cdot_segments", "3", pytest.approx(22.5))
