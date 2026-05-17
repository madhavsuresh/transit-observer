"""Tier-1 data-capture widening tests.

Three pieces in one suite:

1. GTFS-RT proto widening — the additional ``CTAGtfsRtTripUpdate``,
   ``CTAGtfsRtVehiclePosition``, ``CTAGtfsRtAlert`` fields land in their
   respective ``train_v2_gtfsrt_*`` tables.
2. GTFS-RT Alert entity parser — feed bytes → alert rows.
3. GTFS-static extractor — a synthetic GTFS zip lands in every
   ``gtfs_static_*`` table.
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest
from google.transit import gtfs_realtime_pb2 as pb

from transit_observer.config import CHICAGO
from transit_observer.cta_gtfsrt_client import CTAGtfsRtClient
from transit_observer.db import init_schema
from transit_observer.gtfs_static_extract import extract_gtfs_archive
from transit_observer.train_v2.normalize import (
    insert_api_poll,
    normalize_gtfsrt_alerts,
    normalize_gtfsrt_trip_updates,
    normalize_gtfsrt_vehicle_positions,
)
from transit_observer.train_v2.models import ApiCallResult as TrainV2ApiCallResult


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


def _poll_id(conn) -> int:
    result = TrainV2ApiCallResult(
        endpoint="gtfsrt",
        source="gtfsrt_train",
        params_redacted={},
        query_kind="vehicle_positions",
        request_url_redacted="http://example.test/gtfsrt",
        local_request_start_ms=1_700_000_000_000,
        local_response_end_ms=1_700_000_000_500,
        cta_server_time_ms=None,
        http_status=200,
        latency_ms=500.0,
        ok=True,
        json_data=None,
        raw_bytes=None,
        error_message=None,
    )
    return insert_api_poll(conn, result, run_id="r1")


# ---------------------------------------------------------------------------
# 1. GTFS-RT proto widening: confirm new dataclass fields are persisted.
# ---------------------------------------------------------------------------

def _build_vehicle_feed() -> bytes:
    feed = pb.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = 1_700_000_000
    feed.header.incrementality = pb.FeedHeader.FULL_DATASET
    entity = feed.entity.add()
    entity.id = "vp-1"
    vp = entity.vehicle
    vp.trip.trip_id = "T123"
    vp.trip.route_id = "Red"
    vp.trip.start_date = "20260517"
    vp.trip.start_time = "08:30:00"
    vp.trip.direction_id = 1
    vp.trip.schedule_relationship = pb.TripDescriptor.SCHEDULED
    vp.vehicle.id = "V01"
    vp.vehicle.label = "Train 812"
    vp.vehicle.license_plate = "CTA-812"
    vp.position.latitude = 41.875
    vp.position.longitude = -87.649
    vp.position.bearing = 270.0
    vp.position.speed = 12.5
    vp.position.odometer = 1234567.0
    vp.current_stop_sequence = 7
    vp.stop_id = "30074"
    vp.timestamp = 1_700_000_000
    vp.current_status = pb.VehiclePosition.STOPPED_AT
    vp.congestion_level = pb.VehiclePosition.RUNNING_SMOOTHLY
    vp.occupancy_status = pb.VehiclePosition.FEW_SEATS_AVAILABLE
    vp.occupancy_percentage = 42
    # multi_carriage_details requires a recent proto build; skip if not present.
    if hasattr(vp, "multi_carriage_details"):
        try:
            mc = vp.multi_carriage_details.add()
            mc.id = "car-1"
            mc.label = "1"
            mc.occupancy_status = pb.VehiclePosition.FEW_SEATS_AVAILABLE
            mc.occupancy_percentage = 40
            mc.carriage_sequence = 1
        except AttributeError:
            pass
    return feed.SerializeToString()


def _build_trip_feed() -> bytes:
    feed = pb.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = 1_700_000_000
    feed.header.incrementality = pb.FeedHeader.FULL_DATASET
    entity = feed.entity.add()
    entity.id = "tu-1"
    tu = entity.trip_update
    tu.trip.trip_id = "T123"
    tu.trip.route_id = "Red"
    tu.trip.start_date = "20260517"
    tu.trip.start_time = "08:30:00"
    tu.trip.direction_id = 1
    tu.trip.schedule_relationship = pb.TripDescriptor.SCHEDULED
    tu.vehicle.id = "V01"
    tu.vehicle.label = "Train 812"
    tu.vehicle.license_plate = "CTA-812"
    tu.timestamp = 1_700_000_000
    tu.delay = 90
    stu = tu.stop_time_update.add()
    stu.stop_id = "30074"
    stu.stop_sequence = 7
    stu.arrival.time = 1_700_000_060
    stu.arrival.delay = 30
    stu.arrival.uncertainty = 15
    stu.departure.time = 1_700_000_090
    stu.departure.delay = 25
    stu.departure.uncertainty = 12
    stu.schedule_relationship = pb.TripUpdate.StopTimeUpdate.SCHEDULED
    return feed.SerializeToString()


def _build_alert_feed() -> bytes:
    feed = pb.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = 1_700_000_000
    feed.header.incrementality = pb.FeedHeader.FULL_DATASET
    entity = feed.entity.add()
    entity.id = "alert-1"
    a = entity.alert
    a.cause = pb.Alert.MAINTENANCE
    a.effect = pb.Alert.REDUCED_SERVICE
    a.severity_level = pb.Alert.WARNING
    a.header_text.translation.add(text="Red Line delays", language="en")
    a.description_text.translation.add(text="Trains running on a reduced schedule.", language="en")
    a.url.translation.add(text="https://example.com/alert", language="en")
    period = a.active_period.add()
    period.start = 1_699_990_000
    period.end = 1_700_010_000
    selector = a.informed_entity.add()
    selector.route_id = "Red"
    selector.stop_id = "30074"
    return feed.SerializeToString()


@pytest.mark.asyncio
async def test_vehicle_position_widening_lands_new_columns(conn, monkeypatch):
    blob = _build_vehicle_feed()

    class _Resp:
        status_code = 200
        content = blob
        def raise_for_status(self):
            return None

    class _HTTP:
        async def get(self, *a, **k):
            return _Resp()
        async def aclose(self):
            return None

    client = CTAGtfsRtClient(http=_HTTP())
    rows = await client.fetch_vehicle_positions(url="http://example.test/vp", mode="train")
    assert len(rows) == 1
    r = rows[0]
    poll_id = _poll_id(conn)
    normalize_gtfsrt_vehicle_positions(
        conn, poll_id, "r1", rows, local_response_end_ms=1_700_000_000_500,
    )

    row = conn.execute(
        """
        SELECT stop_id, occupancy_percentage, odometer_m, vehicle_timestamp_ms,
               vehicle_license_plate, trip_start_date, trip_start_time, trip_direction_id,
               feed_timestamp_ms, feed_incrementality
          FROM train_v2_gtfsrt_vehicle_position
        """
    ).fetchone()
    assert row[0] == "30074"
    assert row[1] == 42
    assert row[2] == pytest.approx(1234567.0)
    assert row[3] == 1_700_000_000_000  # vehicle ts in ms
    assert row[4] == "CTA-812"
    assert row[5] == "20260517"
    assert row[6] == "08:30:00"
    assert row[7] == 1
    assert row[8] == 1_700_000_000_000  # feed ts in ms
    assert row[9] == "full_dataset"


@pytest.mark.asyncio
async def test_trip_update_widening_lands_new_columns(conn):
    blob = _build_trip_feed()

    class _Resp:
        status_code = 200
        content = blob
        def raise_for_status(self):
            return None

    class _HTTP:
        async def get(self, *a, **k):
            return _Resp()
        async def aclose(self):
            return None

    client = CTAGtfsRtClient(http=_HTTP())
    rows = await client.fetch_trip_updates(url="http://example.test/tu", mode="train")
    assert len(rows) == 1
    poll_id = _poll_id(conn)
    normalize_gtfsrt_trip_updates(
        conn, poll_id, "r1", rows, local_response_end_ms=1_700_000_000_500,
    )

    row = conn.execute(
        """
        SELECT trip_start_date, trip_start_time, trip_direction_id, trip_schedule_relationship,
               trip_update_timestamp_ms, trip_update_delay_seconds,
               arrival_uncertainty_seconds, departure_uncertainty_seconds,
               feed_timestamp_ms, feed_incrementality, vehicle_label, vehicle_license_plate
          FROM train_v2_gtfsrt_trip_update
        """
    ).fetchone()
    assert row[0] == "20260517"
    assert row[1] == "08:30:00"
    assert row[2] == 1
    assert row[3] == "scheduled"
    assert row[4] == 1_700_000_000_000
    assert row[5] == 90
    assert row[6] == 15
    assert row[7] == 12
    assert row[8] == 1_700_000_000_000
    assert row[9] == "full_dataset"
    assert row[10] == "Train 812"
    assert row[11] == "CTA-812"


# ---------------------------------------------------------------------------
# 2. GTFS-RT Alert parser.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_alert_parser_writes_alert_row(conn):
    blob = _build_alert_feed()

    class _Resp:
        status_code = 200
        content = blob
        def raise_for_status(self):
            return None

    class _HTTP:
        async def get(self, *a, **k):
            return _Resp()
        async def aclose(self):
            return None

    client = CTAGtfsRtClient(http=_HTTP())
    rows = await client.fetch_alerts(url="http://example.test/alerts", mode="train")
    assert len(rows) == 1
    poll_id = _poll_id(conn)
    n = normalize_gtfsrt_alerts(
        conn, poll_id, "r1", rows, local_response_end_ms=1_700_000_000_500,
    )
    assert n == 1
    row = conn.execute(
        """
        SELECT entity_id, cause, effect, severity_level,
               header_text, description_text, url,
               feed_timestamp_ms, active_period_json, informed_entity_json
          FROM train_v2_gtfsrt_alert
        """
    ).fetchone()
    assert row[0] == "alert-1"
    assert row[1] == "maintenance"
    assert row[2] == "reduced_service"
    assert row[3] == "warning"
    assert row[4] == "Red Line delays"
    assert row[5] == "Trains running on a reduced schedule."
    assert row[6] == "https://example.com/alert"
    assert row[7] == 1_700_000_000_000
    import json

    period_json = json.loads(row[8])
    assert period_json == [{"start": 1_699_990_000, "end": 1_700_010_000}]
    informed = json.loads(row[9])
    assert informed[0]["route_id"] == "Red"
    assert informed[0]["stop_id"] == "30074"


# ---------------------------------------------------------------------------
# 3. GTFS-static extraction from a synthetic zip.
# ---------------------------------------------------------------------------

def _make_synthetic_gtfs_zip(out_path: Path) -> None:
    files = {
        "agency.txt": (
            "agency_id,agency_name,agency_url,agency_timezone\n"
            "CTA,Chicago Transit Authority,http://transitchicago.com,America/Chicago\n"
        ),
        "stops.txt": (
            "stop_id,stop_code,stop_name,stop_lat,stop_lon,parent_station,location_type,wheelchair_boarding\n"
            "30074,30074,UIC-Halsted (Blue Line),41.8755,-87.6494,40380,0,1\n"
            "30075,30075,UIC-Halsted (O'Hare-bound),41.8757,-87.6496,40380,0,1\n"
            "40380,,UIC-Halsted,41.8756,-87.6495,,1,1\n"
        ),
        "routes.txt": (
            "route_id,agency_id,route_short_name,route_long_name,route_type,route_color,route_text_color\n"
            "Red,CTA,Red,Red Line,1,C60C30,FFFFFF\n"
            "Blue,CTA,Blue,Blue Line,1,00A1DE,FFFFFF\n"
        ),
        "trips.txt": (
            "route_id,service_id,trip_id,trip_headsign,direction_id,block_id,shape_id\n"
            "Blue,WKDY,T1,O'Hare,0,B1,SHP1\n"
            "Blue,WKDY,T2,Forest Park,1,B2,SHP2\n"
        ),
        "stop_times.txt": (
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type\n"
            "T1,08:00:00,08:00:30,30074,1,0,0\n"
            "T1,08:03:00,08:03:30,30075,2,0,0\n"
            "T2,08:30:00,08:30:30,30075,1,0,0\n"
        ),
        "shapes.txt": (
            "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence,shape_dist_traveled\n"
            "SHP1,41.8755,-87.6494,1,0\n"
            "SHP1,41.8800,-87.6500,2,500\n"
            "SHP2,41.8757,-87.6496,1,0\n"
        ),
        "calendar.txt": (
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
            "WKDY,1,1,1,1,1,0,0,20260101,20261231\n"
        ),
        "calendar_dates.txt": (
            "service_id,date,exception_type\n"
            "WKDY,20260704,2\n"
        ),
        "feed_info.txt": (
            "feed_publisher_name,feed_version,feed_start_date,feed_end_date\n"
            "CTA,2026.05.17,20260101,20261231\n"
        ),
    }
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def test_gtfs_static_extract_loads_every_table(conn, tmp_path):
    zip_path = tmp_path / "cta_gtfs.zip"
    _make_synthetic_gtfs_zip(zip_path)
    snapshot_id = extract_gtfs_archive(
        conn, agency="cta", archive_path=zip_path,
        fetched_at_ms=1_700_000_000_000,
    )
    # Snapshot row populated with extraction metadata.
    snap = conn.execute(
        """
        SELECT agency, archive_sha256, n_stops, n_routes, n_trips, n_stop_times, n_shapes,
               n_calendar, n_calendar_dates, feed_publisher_name, feed_version,
               feed_start_date, feed_end_date
          FROM gtfs_static_snapshot WHERE snapshot_id = ?
        """,
        [snapshot_id],
    ).fetchone()
    agency, sha, n_stops, n_routes, n_trips, n_st, n_shapes, n_cal, n_cd, pub, ver, start, end = snap
    assert agency == "cta"
    assert sha and len(sha) == 64
    assert n_stops == 3
    assert n_routes == 2
    assert n_trips == 2
    assert n_st == 3
    assert n_shapes == 3
    assert n_cal == 1
    assert n_cd == 1
    assert pub == "CTA"
    assert ver == "2026.05.17"
    assert start == "20260101"
    assert end == "20261231"

    # Spot-check actual rows.
    parent = conn.execute(
        "SELECT stop_lat, parent_station FROM gtfs_static_stops WHERE stop_id = '30074'"
    ).fetchone()
    assert parent[0] == pytest.approx(41.8755)
    assert parent[1] == "40380"

    trip_count = conn.execute(
        "SELECT COUNT(*) FROM gtfs_static_stop_times WHERE trip_id = 'T1'"
    ).fetchone()[0]
    assert trip_count == 2

    weekday = conn.execute(
        "SELECT monday, friday, saturday, sunday FROM gtfs_static_calendar WHERE service_id = 'WKDY'"
    ).fetchone()
    assert weekday == (1, 1, 0, 0)

    holiday = conn.execute(
        "SELECT exception_type FROM gtfs_static_calendar_dates WHERE service_id = 'WKDY' AND date = '20260704'"
    ).fetchone()
    assert holiday[0] == 2


def test_gtfs_static_extract_is_idempotent(conn, tmp_path):
    zip_path = tmp_path / "cta_gtfs.zip"
    _make_synthetic_gtfs_zip(zip_path)
    sid_a = extract_gtfs_archive(conn, agency="cta", archive_path=zip_path)
    sid_b = extract_gtfs_archive(conn, agency="cta", archive_path=zip_path)
    # Same sha + replace_existing default = re-use snapshot, row counts stable.
    assert sid_a == sid_b
    n_snapshots = conn.execute(
        "SELECT COUNT(*) FROM gtfs_static_snapshot WHERE agency = 'cta'"
    ).fetchone()[0]
    assert n_snapshots == 1
    n_stops = conn.execute("SELECT COUNT(*) FROM gtfs_static_stops").fetchone()[0]
    assert n_stops == 3
