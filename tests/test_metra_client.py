"""MetraClient protobuf parsing. Doesn't hit the network."""

from __future__ import annotations

from datetime import datetime

import pytest
from google.transit import gtfs_realtime_pb2

from transit_observer.config import CHICAGO
from transit_observer.metra_client import MetraClient, MetraStopUpdate


def _build_feed(now_epoch: int) -> bytes:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = now_epoch

    entity = feed.entity.add()
    entity.id = "trip-1"
    trip_update = entity.trip_update
    trip_update.trip.trip_id = "UP-N-1234"
    trip_update.trip.route_id = "UP-N"
    trip_update.trip.direction_id = 1

    stu = trip_update.stop_time_update.add()
    stu.stop_id = "DAVIS"
    stu.departure.time = now_epoch + 300
    stu.departure.delay = 30
    stu.schedule_relationship = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.SCHEDULED

    cancelled = feed.entity.add()
    cancelled.id = "trip-2"
    cancelled.trip_update.trip.trip_id = "UP-N-9999"
    cancelled.trip_update.trip.route_id = "UP-N"
    cstu = cancelled.trip_update.stop_time_update.add()
    cstu.stop_id = "OGILVIE"
    cstu.schedule_relationship = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.SKIPPED

    return feed.SerializeToString()


@pytest.mark.asyncio
async def test_parses_trip_updates_into_stop_update_records():
    now_epoch = 1_730_000_000
    expected_predicted = datetime.fromtimestamp(now_epoch + 300, tz=CHICAGO)

    class _StubResponse:
        content = _build_feed(now_epoch)
        status_code = 200

        def raise_for_status(self) -> None:
            return None

    class _StubHttp:
        async def get(self, url: str, headers: dict) -> _StubResponse:
            return _StubResponse()

        async def aclose(self) -> None:
            return None

    client = MetraClient("test-key", http=_StubHttp())  # type: ignore[arg-type]
    updates = await client.fetch_trip_updates()
    assert len(updates) == 2

    by_stop = {u.station_id: u for u in updates}
    davis = by_stop["DAVIS"]
    assert davis.route_id == "UP-N"
    assert davis.trip_id == "UP-N-1234"
    assert davis.predicted_at == expected_predicted
    assert davis.delay_seconds == 30
    assert davis.schedule_relationship == "scheduled"
    assert davis.direction_id == 1

    ogilvie = by_stop["OGILVIE"]
    assert ogilvie.schedule_relationship == "skipped"
    assert ogilvie.predicted_at is None


def test_constructor_rejects_missing_key():
    with pytest.raises(ValueError):
        MetraClient("")
