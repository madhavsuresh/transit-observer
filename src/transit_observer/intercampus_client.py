"""Northwestern Intercampus (TripShot) GTFS-Realtime client.

No auth required (public feed). Two endpoints:
- https://northwestern.tripshot.com/v1/gtfs/realtime/tripUpdate
- https://northwestern.tripshot.com/v1/gtfs/realtime/vehiclePosition

For v1 we only consume tripUpdate. Small network (~24 stops, 2
directions) — one poll every 60s covers everything.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import httpx
from google.transit import gtfs_realtime_pb2

from .config import CHICAGO


TRIP_UPDATE_URL = "https://northwestern.tripshot.com/v1/gtfs/realtime/tripUpdate"


@dataclass(frozen=True)
class IntercampusStopUpdate:
    route_id: str
    trip_id: str
    stop_id: str
    schedule_relationship: str
    predicted_at: datetime | None
    delay_seconds: int | None
    direction_id: int | None


class IntercampusClient:
    def __init__(self, *, http: httpx.AsyncClient | None = None) -> None:
        self._http = http or httpx.AsyncClient(timeout=15.0)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def fetch_trip_updates(self) -> list[IntercampusStopUpdate]:
        resp = await self._http.get(TRIP_UPDATE_URL, headers={"Accept": "application/x-protobuf"})
        resp.raise_for_status()
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(resp.content)
        out: list[IntercampusStopUpdate] = []
        for entity in feed.entity:
            if not entity.HasField("trip_update"):
                continue
            trip_update = entity.trip_update
            trip = trip_update.trip
            route_id = trip.route_id or ""
            trip_id = trip.trip_id or ""
            direction_id = trip.direction_id if trip.HasField("direction_id") else None
            for stu in trip_update.stop_time_update:
                if not stu.stop_id:
                    continue
                predicted = None
                delay = None
                for field in ("departure", "arrival"):
                    if stu.HasField(field):
                        event = getattr(stu, field)
                        if event.HasField("time") and event.time > 0:
                            predicted = datetime.fromtimestamp(event.time, tz=CHICAGO)
                        if event.HasField("delay"):
                            delay = int(event.delay)
                        if predicted is not None:
                            break
                relationship = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name(
                    stu.schedule_relationship
                ).lower()
                out.append(
                    IntercampusStopUpdate(
                        route_id=route_id,
                        trip_id=trip_id,
                        stop_id=stu.stop_id,
                        schedule_relationship=relationship,
                        predicted_at=predicted,
                        delay_seconds=delay,
                        direction_id=direction_id,
                    )
                )
        return out
