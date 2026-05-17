"""Smoke tests for the train_v2 normalizer.

Synthetic ``ttarrivals.aspx`` / ``ttfollow.aspx`` / ``ttpositions.aspx``
payloads → expected rows in the ``train_v2_*`` tables.
"""

from __future__ import annotations

import os
import tempfile

import duckdb
import pytest

from transit_observer.db import init_schema
from transit_observer.train_v2.models import ApiCallResult
from transit_observer.train_v2.normalize import record_poll


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


def _call(endpoint: str, query_kind: str, json_data: dict | None) -> ApiCallResult:
    return ApiCallResult(
        endpoint=endpoint,
        source="train_tracker",
        params_redacted={"outputType": "JSON", "key": "<redacted>"},
        query_kind=query_kind,
        request_url_redacted=f"http://example.test/{endpoint}?key=<redacted>",
        local_request_start_ms=1_700_000_000_000,
        local_response_end_ms=1_700_000_000_500,
        cta_server_time_ms=None,
        http_status=200,
        latency_ms=500.0,
        ok=True,
        json_data=json_data,
        raw_bytes=None,
        error_message=None,
    )


def test_ttarrivals_normalizes_to_arrival_observation(conn):
    payload = {
        "ctatt": {
            "tmst": "2026-05-13T08:23:00",
            "errCd": "0",
            "eta": [
                {
                    "staId": "40380",
                    "stpId": "30074",
                    "staNm": "UIC-Halsted",
                    "stpDe": "Service toward O'Hare",
                    "rn": "812",
                    "rt": "Blue",
                    "trDr": "5",
                    "destNm": "O'Hare",
                    "destSt": "30171",
                    "prdt": "2026-05-13T08:23:00",
                    "arrT": "2026-05-13T08:25:00",
                    "isApp": "0",
                    "isDly": "0",
                    "isFlt": "0",
                    "isSch": "0",
                    "flags": "",
                }
            ],
        }
    }
    record_poll(conn, _call("ttarrivals.aspx", "arrivals_by_station", payload), run_id="r1")
    rows = conn.execute(
        """
        SELECT line, run_number, map_id, stop_id, station_name, eta_s,
               prediction_age_s, is_approaching
          FROM train_v2_arrival_observation
        """
    ).fetchall()
    assert len(rows) == 1
    line, rn, map_id, stop_id, name, eta_s, pred_age_s, is_app = rows[0]
    assert (line, rn, map_id, stop_id) == ("Blue", "812", "40380", "30074")
    assert name == "UIC-Halsted"
    assert eta_s == pytest.approx(120.0)
    assert pred_age_s == pytest.approx(0.0, abs=0.01)
    assert is_app is False


def test_ttfollow_writes_one_row_per_eta_with_sequence(conn):
    payload = {
        "ctatt": {
            "tmst": "2026-05-13T08:23:00",
            "errCd": "0",
            "runno": "812",
            "eta": [
                {
                    "rn": "812", "rt": "Blue", "staId": "40380", "stpId": "30074",
                    "staNm": "UIC-Halsted", "trDr": "5", "destNm": "O'Hare",
                    "prdt": "2026-05-13T08:23:00", "arrT": "2026-05-13T08:25:00",
                    "isApp": "0", "isDly": "0", "isFlt": "0", "isSch": "0",
                },
                {
                    "rn": "812", "rt": "Blue", "staId": "40570", "stpId": "30075",
                    "staNm": "Clinton", "trDr": "5", "destNm": "O'Hare",
                    "prdt": "2026-05-13T08:23:00", "arrT": "2026-05-13T08:27:00",
                    "isApp": "0", "isDly": "0", "isFlt": "0", "isSch": "0",
                },
                {
                    "rn": "812", "rt": "Blue", "staId": "40050", "stpId": "30076",
                    "staNm": "LaSalle", "trDr": "5", "destNm": "O'Hare",
                    "prdt": "2026-05-13T08:23:00", "arrT": "2026-05-13T08:29:00",
                    "isApp": "0", "isDly": "0", "isFlt": "0", "isSch": "0",
                },
            ],
        }
    }
    record_poll(conn, _call("ttfollow.aspx", "follow_by_run", payload), run_id="r1")
    rows = conn.execute(
        """
        SELECT seq, run_number, map_id, station_name, eta_s
          FROM train_v2_follow_observation
         ORDER BY seq
        """
    ).fetchall()
    assert [r[0] for r in rows] == [0, 1, 2]
    assert all(r[1] == "812" for r in rows)
    assert rows[0][2] == "40380"
    assert rows[2][2] == "40050"
    # ETAs at 120 / 240 / 360 seconds from server time.
    assert rows[0][4] == pytest.approx(120.0)
    assert rows[2][4] == pytest.approx(360.0)


def test_ttpositions_writes_one_row_per_train_with_next_station(conn):
    payload = {
        "ctatt": {
            "tmst": "2026-05-13T08:23:00",
            "errCd": "0",
            "route": [
                {
                    "@name": "Blue",
                    "train": [
                        {
                            "rn": "812", "destSt": "30171", "destNm": "O'Hare",
                            "trDr": "5", "nextStaId": "40380", "nextStaNm": "UIC-Halsted",
                            "prdt": "2026-05-13T08:23:00", "arrT": "2026-05-13T08:25:00",
                            "isApp": "0", "isDly": "0", "isFlt": "0",
                            "lat": "41.875", "lon": "-87.649", "heading": "270",
                        },
                        {
                            "rn": "813", "destSt": "30171", "destNm": "O'Hare",
                            "trDr": "5", "nextStaId": "40570", "nextStaNm": "Clinton",
                            "prdt": "2026-05-13T08:23:00", "arrT": "2026-05-13T08:28:00",
                            "isApp": "0", "isDly": "0", "isFlt": "0",
                            "lat": "41.88", "lon": "-87.64", "heading": "270",
                        },
                    ],
                }
            ],
        }
    }
    record_poll(conn, _call("ttpositions.aspx", "positions_by_line", payload), run_id="r1")
    rows = conn.execute(
        """
        SELECT line, run_number, next_station_map_id, next_station_name, lat, lon, heading
          FROM train_v2_position_observation
         ORDER BY run_number
        """
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "Blue"
    assert rows[0][2] == "40380"
    assert rows[0][3] == "UIC-Halsted"
    assert rows[0][4] == pytest.approx(41.875)
    assert rows[1][2] == "40570"


def test_api_poll_records_server_time(conn):
    payload = {"ctatt": {"tmst": "2026-05-13T08:23:00", "errCd": "0", "eta": []}}
    record_poll(conn, _call("ttarrivals.aspx", "arrivals_by_station", payload), run_id="r1")
    row = conn.execute(
        "SELECT source, endpoint, cta_server_time_ms, ok FROM train_v2_api_poll"
    ).fetchone()
    assert row[0] == "train_tracker"
    assert row[1] == "ttarrivals.aspx"
    assert row[2] is not None and row[2] > 0
    assert row[3] is True


def test_errcd_nonzero_skips_normalization(conn):
    payload = {"ctatt": {"tmst": "2026-05-13T08:23:00", "errCd": "1", "errNm": "no service", "eta": []}}
    record_poll(conn, _call("ttarrivals.aspx", "arrivals_by_station", payload), run_id="r1")
    # api_poll still gets written for the audit trail, but no observation rows.
    assert conn.execute("SELECT COUNT(*) FROM train_v2_api_poll").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM train_v2_arrival_observation").fetchone()[0] == 0
