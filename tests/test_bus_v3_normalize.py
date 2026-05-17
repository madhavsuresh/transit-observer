"""Smoke tests for the bus_v3 normalizer.

Each test feeds a synthetic ``ApiCallResult`` through ``record_poll`` and
asserts the corresponding normalized table received the expected row(s).
DB-agnostic golden fixtures live inline; we never go to the network.
"""

from __future__ import annotations

import json
import os
import tempfile

import duckdb
import pytest

from transit_observer.bus_v3.models import ApiCallResult
from transit_observer.bus_v3.normalize import record_poll
from transit_observer.db import init_schema


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


def _call(endpoint: str, query_kind: str, json_data: dict) -> ApiCallResult:
    return ApiCallResult(
        endpoint=endpoint,
        params_redacted={"format": "json", "key": "<redacted>"},
        query_kind=query_kind,
        request_url_redacted=f"https://example.test/{endpoint}?key=<redacted>",
        local_request_start_ms=1_700_000_000_000,
        local_response_end_ms=1_700_000_000_500,
        cta_server_time_ms=1_700_000_000_250,
        http_status=200,
        latency_ms=500.0,
        ok=True,
        json_data=json_data,
        error_message=None,
    )


def test_getroutes_normalizes_to_route_table(conn):
    payload = {"bustime-response": {"routes": [
        {"rt": "20", "rtnm": "Madison", "rtclr": "#bb0000", "rtdd": "20"},
        {"rt": "66", "rtnm": "Chicago", "rtclr": "#00bb00", "rtdd": "66"},
    ]}}
    record_poll(conn, _call("getroutes", "routes", payload), run_id="r1")
    rows = conn.execute("SELECT rt, rtnm FROM bus_v3_route ORDER BY rt").fetchall()
    assert rows == [("20", "Madison"), ("66", "Chicago")]


def test_getroutes_is_idempotent(conn):
    payload = {"bustime-response": {"routes": [
        {"rt": "20", "rtnm": "Madison", "rtclr": "#bb0000", "rtdd": "20"},
    ]}}
    record_poll(conn, _call("getroutes", "routes", payload), run_id="r1")
    record_poll(conn, _call("getroutes", "routes", payload), run_id="r2")
    n = conn.execute("SELECT COUNT(*) FROM bus_v3_route").fetchone()[0]
    assert n == 1


def test_getpredictions_writes_observation_with_server_time(conn):
    server_ms = 1_700_000_000_250
    prdtm_ms = server_ms + 180_000
    tmstmp_ms = server_ms - 5_000
    payload = {"bustime-response": {"prd": [{
        "tmstmp": str(tmstmp_ms),
        "typ": "A",
        "stpid": "456",
        "stpnm": "Test Stop",
        "vid": "V1",
        "dstp": "1200",
        "rt": "20",
        "rtdd": "20",
        "rtdir": "Westbound",
        "des": "Austin",
        "prdtm": str(prdtm_ms),
        "prdctdn": "3",
        "dly": False,
        "dyn": 0,
        "tablockid": "b",
        "tatripid": "t",
        "origtatripno": "o",
        "stst": 0,
        "stsd": "2026-01-01",
        "flagstop": 0,
    }]}}
    record_poll(
        conn,
        _call("getpredictions", "predictions_by_stop", payload),
        run_id="r1",
        cycle_index=0,
    )
    rows = conn.execute(
        """
        SELECT stpid, vid, rt, rtdir, prdtm_ms, eta_s, prdctdn_min, dyn, flagstop
        FROM bus_v3_prediction_observation
        """
    ).fetchall()
    assert len(rows) == 1
    stpid, vid, rt, rtdir, prdtm, eta_s, prdctdn_min, dyn, flagstop = rows[0]
    assert (stpid, vid, rt, rtdir) == ("456", "V1", "20", "Westbound")
    assert prdtm == prdtm_ms
    assert eta_s == pytest.approx(180.0)
    assert prdctdn_min == 3.0
    assert dyn == 0
    assert flagstop == 0


def test_getvehicles_writes_pdist_and_age(conn):
    server_ms = 1_700_000_000_250
    payload = {"bustime-response": {"vehicle": [{
        "vid": "V1",
        "tmstmp": str(server_ms - 4_000),
        "lat": "41.0",
        "lon": "-87.001",
        "hdg": "270",
        "pid": 1,
        "pdist": "1200",
        "rt": "20",
        "des": "Austin",
        "dly": False,
        "tablockid": "b",
        "tatripid": "t",
        "origtatripno": "o",
        "zone": "",
        "mode": 0,
        "psgld": "EMPTY",
        "stst": 0,
        "stsd": "2026-01-01",
    }]}}
    record_poll(conn, _call("getvehicles", "vehicles_by_route", payload), run_id="r1")
    rows = conn.execute(
        "SELECT vid, pdist_ft, vehicle_age_s, pid FROM bus_v3_vehicle_observation"
    ).fetchall()
    assert len(rows) == 1
    vid, pdist, age, pid = rows[0]
    assert vid == "V1"
    assert pdist == pytest.approx(1200.0)
    assert age == pytest.approx(4.0)
    assert pid == 1


def test_getpatterns_writes_points_with_dedup(conn):
    payload = {"bustime-response": {"ptr": [{
        "pid": 1,
        "ln": 10000,
        "rtdir": "Westbound",
        "pt": [
            {"seq": 1, "typ": "W", "lat": "41.0", "lon": "-87.0", "pdist": "0"},
            {"seq": 2, "typ": "S", "stpid": "456", "stpnm": "Stop", "lat": "41.0", "lon": "-87.001", "pdist": "1000"},
        ],
    }]}}
    result = _call("getpatterns", "patterns_by_route", payload)
    result.params_redacted["rt"] = "20"
    record_poll(conn, result, run_id="r1")
    record_poll(conn, result, run_id="r2")  # idempotent
    n_pat = conn.execute("SELECT COUNT(*) FROM bus_v3_pattern").fetchone()[0]
    n_pts = conn.execute("SELECT COUNT(*) FROM bus_v3_pattern_point").fetchone()[0]
    assert n_pat == 1
    assert n_pts == 2


def test_api_poll_returning_assigns_unique_ids(conn):
    payload = {"bustime-response": {"tm": str(1_700_000_000_000)}}
    p1 = record_poll(conn, _call("gettime", "server_time", payload), run_id="r1")
    p2 = record_poll(conn, _call("gettime", "server_time", payload), run_id="r1")
    assert p2 > p1
