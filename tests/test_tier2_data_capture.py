"""Tier-2 data-capture tests.

Three pieces in one suite:

1. Slow-zone HTML parser — synthetic HTML → train_v2_slow_zone rows with
   line / direction / max_mph / station segment / dates extracted, with
   raw cells preserved for retrospective re-parsing.
2. ``ttarrivals_by_stop`` rotation — when platform polling is enabled
   and the arrival_observation table has known (map_id, stop_id) pairs,
   the collector calls the platform endpoint for the oldest-polled
   stops.
3. AirNow widening — ``Category.Number`` + ``StateCode`` lands per
   observation, and ``fetch_zip_forecast`` produces forecast rows with
   ``is_forecast=True``.
"""

from __future__ import annotations

import os
import tempfile

import duckdb
import pytest

from transit_observer.air_quality_client import AirQualityClient
from transit_observer.db import init_schema
from transit_observer.train_v2.client import CTATrainV2Client
from transit_observer.train_v2 import collector as train_v2_collector
from transit_observer.train_v2.slow_zone_parser import (
    classify_row,
    extract_rows,
    parse_slow_zones,
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


# ---------------------------------------------------------------------------
# 1. Slow-zone HTML parser.
# ---------------------------------------------------------------------------

_SLOW_ZONE_HTML = """
<html>
<body>
<h1>Slow Zones</h1>
<table>
<thead><tr><th>Line</th><th>Direction</th><th>Segment</th><th>Speed</th><th>Posted</th><th>Restore</th></tr></thead>
<tbody>
<tr>
  <td>Red Line</td><td>Southbound</td>
  <td>Between Howard and Jarvis</td><td>15 MPH</td>
  <td>03/15/2026</td><td>06/01/2026</td>
</tr>
<tr>
  <td>Blue Line</td><td>Both directions</td>
  <td>Between UIC-Halsted and Clinton</td><td>25 MPH</td>
  <td>04/01/2026</td><td>07/15/2026</td>
</tr>
<tr>
  <td>Brown Line</td><td>Inbound</td>
  <td>Between Western and Rockwell</td><td>10 MPH</td>
  <td>2026-02-20</td><td>2026-05-30</td>
</tr>
</tbody>
</table>
</body>
</html>
"""


def test_slow_zone_extract_rows_yields_every_tr():
    rows = extract_rows(_SLOW_ZONE_HTML)
    # 1 header row + 3 data rows.
    assert len(rows) == 4
    assert rows[0] == ["Line", "Direction", "Segment", "Speed", "Posted", "Restore"]
    assert "Red Line" in rows[1]


def test_slow_zone_classify_row_extracts_line_direction_mph_and_segment():
    cells = [
        "Red Line", "Southbound",
        "Between Howard and Jarvis", "15 MPH",
        "03/15/2026", "06/01/2026",
    ]
    record = classify_row(cells)
    assert record["line"] == "Red"
    assert record["direction_code"] == "SB"
    assert record["max_mph"] == 15.0
    assert record["from_station"] == "Howard"
    assert record["to_station"] == "Jarvis"
    assert record["posted_at_ms"] is not None
    assert record["expected_clear_at_ms"] is not None
    assert record["posted_at_ms"] < record["expected_clear_at_ms"]


def test_slow_zone_parse_inserts_train_v2_slow_zone_rows(conn):
    n = parse_slow_zones(conn, html=_SLOW_ZONE_HTML, polled_at_ms=1_700_000_000_000)
    assert n == 3
    lines = {
        r[0] for r in conn.execute(
            "SELECT line FROM train_v2_slow_zone ORDER BY line"
        ).fetchall()
    }
    assert lines == {"Brn", "Blue", "Red"}
    # Direction codes are normalized.
    directions = {
        r[0] for r in conn.execute(
            "SELECT direction_code FROM train_v2_slow_zone WHERE direction_code IS NOT NULL"
        ).fetchall()
    }
    assert "SB" in directions
    assert "BOTH" in directions
    assert "IB" in directions
    # Raw cells preserved.
    raw = conn.execute(
        "SELECT raw_payload_json FROM train_v2_slow_zone WHERE line = 'Red'"
    ).fetchone()[0]
    assert "Howard" in raw and "Jarvis" in raw


def test_slow_zone_parse_is_idempotent(conn):
    parse_slow_zones(conn, html=_SLOW_ZONE_HTML, polled_at_ms=1_700_000_000_000)
    parse_slow_zones(conn, html=_SLOW_ZONE_HTML, polled_at_ms=1_700_000_005_000)
    n = conn.execute("SELECT COUNT(*) FROM train_v2_slow_zone").fetchone()[0]
    assert n == 3


# ---------------------------------------------------------------------------
# 2. ttarrivals_by_stop rotation.
# ---------------------------------------------------------------------------

class _FakeTrainClient:
    """Records every call without hitting the network."""

    def __init__(self) -> None:
        self.gettime_calls = 0
        self.arrivals_calls: list[int] = []
        self.platform_calls: list[int] = []
        self.positions_calls = 0
        self.follow_calls: list[str] = []

    async def gettime(self):
        self.gettime_calls += 1
        return self._result("gettime", "server_time", {"ctatt": {"tmst": "2026-05-13T08:23:00"}})

    async def ttarrivals(self, *, map_id, max_predictions=12):
        self.arrivals_calls.append(int(map_id))
        return self._result("ttarrivals.aspx", "arrivals_by_station", {"ctatt": {"eta": []}})

    async def ttarrivals_by_stop(self, *, stop_id, max_predictions=12):
        self.platform_calls.append(int(stop_id))
        return self._result("ttarrivals.aspx", "arrivals_by_stop", {"ctatt": {"eta": []}})

    async def ttpositions(self, *, line_codes):
        self.positions_calls += 1
        return self._result("ttpositions.aspx", "positions_by_line", {"ctatt": {"route": []}})

    async def ttfollow(self, *, run_number):
        self.follow_calls.append(str(run_number))
        return self._result("ttfollow.aspx", "follow_by_run", {"ctatt": {"eta": []}})

    def _result(self, endpoint, query_kind, json_data):
        from transit_observer.train_v2.models import ApiCallResult
        return ApiCallResult(
            endpoint=endpoint, source="train_tracker",
            params_redacted={}, query_kind=query_kind,
            request_url_redacted=f"http://example.test/{endpoint}",
            local_request_start_ms=1_700_000_000_000,
            local_response_end_ms=1_700_000_000_500,
            cta_server_time_ms=None,
            http_status=200, latency_ms=500.0, ok=True,
            json_data=json_data, raw_bytes=None, error_message=None,
        )


def _seed_arrival_obs(conn, map_id, stop_id, polled_ms):
    conn.execute(
        """
        INSERT INTO train_v2_api_poll(poll_id, run_id, source, endpoint, params_json_redacted,
                                      local_request_start_ms, local_response_end_ms, ok, created_at_ms)
        VALUES (?, 'r1', 'train_tracker', 'ttarrivals.aspx', '{}', ?, ?, TRUE, ?)
        """,
        [polled_ms, polled_ms - 500, polled_ms, polled_ms],
    )
    conn.execute(
        """
        INSERT INTO train_v2_arrival_observation(
            poll_id, run_id, local_response_end_ms, query_kind,
            line, run_number, map_id, stop_id, station_name
        ) VALUES (?, 'r1', ?, 'arrivals_by_station', 'Blue', '812', ?, ?, 'X')
        """,
        [polled_ms, polled_ms, str(map_id), str(stop_id)],
    )


@pytest.mark.asyncio
async def test_platform_polling_picks_oldest_first(conn):
    # Three known platforms, oldest first.
    _seed_arrival_obs(conn, 40380, 30074, polled_ms=1_700_000_000_000)  # oldest
    _seed_arrival_obs(conn, 40380, 30075, polled_ms=1_700_000_010_000)
    _seed_arrival_obs(conn, 40570, 30076, polled_ms=1_700_000_020_000)
    state = train_v2_collector.TrainV2CycleState()
    config = train_v2_collector.TrainV2CycleConfig(
        targets=[train_v2_collector.TrainV2Target(map_id=99999, station_name="Nowhere")],
        line_codes=("red",),
        arrivals_batch_size=1,
        platform_polling_enabled=True,
        platforms_per_cycle=2,
        platform_max_predictions=8,
    )
    client = _FakeTrainClient()
    result = await train_v2_collector.poll_once(conn, client, config=config, state=state)
    assert result.platforms_polled == 2
    # The two oldest-polled stops should have been picked, in order.
    assert client.platform_calls[:2] == [30074, 30075]


@pytest.mark.asyncio
async def test_platform_polling_disabled_by_default(conn):
    _seed_arrival_obs(conn, 40380, 30074, polled_ms=1_700_000_000_000)
    state = train_v2_collector.TrainV2CycleState()
    config = train_v2_collector.TrainV2CycleConfig(
        targets=[train_v2_collector.TrainV2Target(map_id=99999, station_name="Nowhere")],
        line_codes=("red",),
        arrivals_batch_size=1,
    )
    client = _FakeTrainClient()
    result = await train_v2_collector.poll_once(conn, client, config=config, state=state)
    assert result.platforms_polled == 0
    assert client.platform_calls == []


# ---------------------------------------------------------------------------
# 3. AirNow widening.
# ---------------------------------------------------------------------------

class _StubAirNow:
    """Stand-in for the AirNow HTTP layer that returns hand-rolled payloads."""

    def __init__(self, current_payload, forecast_payload) -> None:
        self.current_payload = current_payload
        self.forecast_payload = forecast_payload

    async def get(self, url, params=None):
        class _Resp:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        if "forecast" in url:
            return _Resp(self.forecast_payload)
        return _Resp(self.current_payload)

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_airnow_current_has_category_number_and_state_code():
    payload = [
        {
            "DateObserved": "2026-05-17",
            "HourObserved": 12,
            "ReportingArea": "Chicago",
            "StateCode": "IL",
            "Latitude": 41.88,
            "Longitude": -87.62,
            "ParameterName": "PM2.5",
            "AQI": 42,
            "Category": {"Number": 2, "Name": "Moderate"},
        }
    ]
    client = AirQualityClient("test-key", http=_StubAirNow(payload, []))
    rows = await client.fetch_zip("60601")
    assert len(rows) == 1
    obs = rows[0]
    assert obs.parameter == "pm2.5"
    assert obs.category == "Moderate"
    assert obs.category_number == 2
    assert obs.state_code == "IL"
    assert obs.is_forecast is False


@pytest.mark.asyncio
async def test_airnow_forecast_rows_marked_is_forecast():
    payload = [
        {
            "DateForecast": "2026-05-18",
            "ReportingArea": "Chicago",
            "StateCode": "IL",
            "ParameterName": "Ozone",
            "AQI": 68,
            "Category": {"Number": 2, "Name": "Moderate"},
            "ActionDay": False,
        }
    ]
    client = AirQualityClient("test-key", http=_StubAirNow([], payload))
    rows = await client.fetch_zip_forecast("60601")
    assert len(rows) == 1
    f = rows[0]
    assert f.is_forecast is True
    assert f.forecast_date == "2026-05-18"
    assert f.parameter == "ozone"
    assert f.category_number == 2
    assert f.action_day is False


def test_air_quality_raw_has_new_columns(conn):
    cols = {
        r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'air_quality_raw'"
        ).fetchall()
    }
    assert {"category_number", "state_code", "is_forecast", "forecast_date", "action_day"} <= cols
