"""Unit tests for ``predictors.bus_telemetry.BusTelemetryPredictor``."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import duckdb
import pytest

from transit_observer.bus_v3.models import DisplayState
from transit_observer.config import CHICAGO
from transit_observer.corridors import Corridor
from transit_observer.db import init_schema
from transit_observer.predictors import bus_telemetry
from transit_observer.predictors.bus_telemetry import (
    BUS_TELEMETRY_VERSION,
    BusTelemetryPredictor,
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
    """Stop catalog/predictor singletons from leaking between tests."""
    reset()
    bus_telemetry.reset_catalog()
    yield
    reset()
    bus_telemetry.reset_catalog()


@pytest.fixture()
def fake_catalog(monkeypatch):
    """Inject a minimal in-memory bus catalog so we don't depend on
    CTABusStops.json. Real catalog has ~14k entries; tests don't need them."""

    class _FakeBusStop:
        def __init__(self, *, route, stop_id, name, latitude, longitude, direction_label):
            self.route = route
            self.stop_id = stop_id
            self.name = name
            self.latitude = latitude
            self.longitude = longitude
            self.direction_label = direction_label

    fake = [
        _FakeBusStop(
            route="22", stop_id=1828, name="Clark & Belmont",
            latitude=41.940, longitude=-87.651, direction_label="Southbound",
        ),
        _FakeBusStop(
            route="22", stop_id=1869, name="Clark & Adams",
            latitude=41.879, longitude=-87.631, direction_label="Southbound",
        ),
    ]
    monkeypatch.setattr(
        "transit_observer.predictors.bus_telemetry.load_bus_catalog",
        lambda: fake,
    )
    bus_telemetry.reset_catalog()
    yield fake


def _bus_corridor() -> Corridor:
    return Corridor(
        corridor_id="cta-bus-22-test-sb",
        mode="bus",
        line="22",
        direction="southbound",
        origin_label="Clark & Belmont",
        origin_latitude=41.940,
        origin_longitude=-87.651,
        destination_label="Clark & Adams",
        destination_latitude=41.879,
        destination_longitude=-87.631,
        boarding_int_id=1828,
        boarding_text_id=None,
        alighting_int_id=1869,
        alighting_text_id=None,
        schedule_headway_seconds=600.0,
    )


def _seed_prediction_and_vehicles(
    conn: duckdb.DuckDBPyConnection,
    *,
    rt: str = "22",
    stpid: str = "1828",
    rtdir: str = "Southbound",
    vid: str = "V1",
) -> None:
    """Seed one prediction snapshot + two bracketing vehicle obs so the
    estimator has enough data to produce a non-abstaining estimate."""
    # Pattern with the target stop at pdist 1000.
    conn.execute(
        """
        INSERT INTO bus_v3_pattern(pid, rt, rtdir, length_ft, dtrid, raw_json)
        VALUES (1, ?, ?, 10000, NULL, '{}')
        """,
        [rt, rtdir],
    )
    for seq, (lat, lon, pdist, typ, sid) in enumerate(
        [
            (41.0, -87.0, 0.0, "W", None),
            (41.0, -87.001, 1000.0, "S", stpid),
            (41.0, -87.002, 2000.0, "W", None),
        ],
        start=1,
    ):
        conn.execute(
            """
            INSERT INTO bus_v3_pattern_point(pid, seq, typ, stpid, stpnm, lat, lon, pdist_ft,
                                             is_detour_original_point, raw_json)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, 0, '{}')
            """,
            [seq, typ, sid, "stop" if sid else None, lat, lon, pdist],
        )
    conn.execute(
        """
        INSERT INTO bus_v3_stop(stpid, stpnm, lat, lon, rt, rtdir, raw_json)
        VALUES (?, 'Test Stop', 41.0, -87.001, ?, ?, '{}')
        """,
        [stpid, rt, rtdir],
    )
    # api_poll rows so the cta_server_time_ms lookup has something to find.
    conn.execute(
        """
        INSERT INTO bus_v3_api_poll(poll_id, run_id, endpoint, params_json_redacted,
                                    local_request_start_ms, local_response_end_ms,
                                    cta_server_time_ms, ok, created_at_ms)
        VALUES (1, 'r1', 'getpredictions', '{}', ?, ?, ?, TRUE, ?)
        """,
        [BASE_MS, BASE_MS + 500, BASE_MS, BASE_MS],
    )
    conn.execute(
        """
        INSERT INTO bus_v3_prediction_observation(
            poll_id, run_id, cta_server_time_ms, local_response_end_ms, query_kind,
            tmstmp_ms, prediction_age_s, typ, stpid, stpnm, vid, dstp_ft, rt, rtdir, des,
            prdtm_ms, eta_s, prdctdn_raw, prdctdn_min, dly, dyn, tablockid, tatripid,
            origtatripno, stst, stsd, flagstop, raw_json
        ) VALUES (1, 'r1', ?, ?, 'predictions_by_stop', ?, 5, 'A', ?, 'Test Stop',
                  ?, 600, ?, ?, 'Austin', ?, 60, '1', 1, 0, 0, 'b', 't', 'o', 0,
                  '2026-01-01', 0, '{}')
        """,
        [BASE_MS, BASE_MS, BASE_MS - 5_000, stpid, vid, rt, rtdir, BASE_MS + 60_000],
    )
    # Two vehicle obs, both BEFORE the stop (pdist 700 → 850 over 15s).
    # The estimator needs the latest vehicle to be approaching, not crossed.
    for poll_id, (t_offset, pdist, lon) in enumerate(
        [(10_000, 700.0, -87.0007), (25_000, 850.0, -87.00085)],
        start=2,
    ):
        conn.execute(
            """
            INSERT INTO bus_v3_api_poll(poll_id, run_id, endpoint, params_json_redacted,
                                        local_request_start_ms, local_response_end_ms,
                                        cta_server_time_ms, ok, created_at_ms)
            VALUES (?, 'r1', 'getvehicles', '{}', ?, ?, ?, TRUE, ?)
            """,
            [poll_id, BASE_MS + t_offset, BASE_MS + t_offset + 1000, BASE_MS + t_offset + 1000, BASE_MS + t_offset],
        )
        conn.execute(
            """
            INSERT INTO bus_v3_vehicle_observation(
                poll_id, run_id, cta_server_time_ms, local_response_end_ms, vid, tmstmp_ms,
                vehicle_age_s, lat, lon, hdg, pid, pdist_ft, rt, des, dly, tablockid, tatripid,
                origtatripno, stst, stsd, raw_json
            ) VALUES (?, 'r1', ?, ?, ?, ?, 5, 41.0, ?, 270, 1, ?, ?, 'Austin', 0, 'b', 't',
                      'o', 0, '2026-01-01', '{}')
            """,
            [poll_id, BASE_MS + t_offset + 1000, BASE_MS + t_offset + 1000, vid,
             BASE_MS + t_offset, lon, pdist, rt],
        )


def test_predict_returns_prediction_with_reliability_in_feature_snapshot(conn, fake_catalog):
    _seed_prediction_and_vehicles(conn)
    pred = BusTelemetryPredictor().predict(
        conn, _bus_corridor(), now=datetime.fromtimestamp(BASE_MS / 1000, CHICAGO)
    )
    assert pred is not None
    assert pred.predictor_version == BUS_TELEMETRY_VERSION
    assert pred.feature_snapshot["display_state"] in {
        DisplayState.HIGH_CONFIDENCE.value,
        DisplayState.MEDIUM_CONFIDENCE.value,
        DisplayState.LOW_CONFIDENCE.value,
    }
    assert isinstance(pred.feature_snapshot["reliability"], float)
    assert pred.feature_snapshot["vid"] == "V1"
    assert pred.wait.p50 >= 0
    assert pred.in_vehicle.p50 > 0


def test_abstains_when_dyn_canceled(conn, fake_catalog):
    _seed_prediction_and_vehicles(conn)
    conn.execute("UPDATE bus_v3_prediction_observation SET dyn = 1")
    pred = BusTelemetryPredictor().predict(
        conn, _bus_corridor(), now=datetime.fromtimestamp(BASE_MS / 1000, CHICAGO)
    )
    assert pred is None


def test_non_bus_corridor_returns_none(conn, fake_catalog):
    train_corridor = Corridor(
        corridor_id="cta-train-test",
        mode="L",
        line="Red",
        direction="inbound",
        origin_label="A", origin_latitude=0.0, origin_longitude=0.0,
        destination_label="B", destination_latitude=0.0, destination_longitude=0.0,
        boarding_int_id=0, boarding_text_id=None,
        alighting_int_id=0, alighting_text_id=None,
        schedule_headway_seconds=300.0,
    )
    pred = BusTelemetryPredictor().predict(
        conn, train_corridor, now=datetime.fromtimestamp(BASE_MS / 1000, CHICAGO)
    )
    assert pred is None


def test_registry_can_instantiate_bus_telemetry():
    instance = predictor(BUS_TELEMETRY_VERSION)
    assert instance is not None
    assert isinstance(instance, BusTelemetryPredictor)
    assert instance.predictor_version == BUS_TELEMETRY_VERSION
