"""Metra GTFS-Realtime client.

Single endpoint pulls all in-flight trip updates (StopTimeUpdates per
trip). One poll every 60s gives full network coverage — much simpler
than the CTA per-station model.

Auth: Bearer header. Set METRA_API_KEY (same key Cozy Fox uses).
Endpoint: https://gtfspublic.metrarr.com/gtfs/public/tripupdates

Falls back to the legacy https://gtfsapi.metrarail.com endpoint if the
primary one fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import httpx
from google.transit import gtfs_realtime_pb2

from .config import CHICAGO


PRIMARY_URL = "https://gtfspublic.metrarr.com/gtfs/public/tripupdates"
FALLBACK_URL = "https://gtfsapi.metrarail.com/gtfs/raw/tripUpdates.dat"


@dataclass(frozen=True)
class MetraStopUpdate:
    """One StopTimeUpdate inside a TripUpdate for a Metra trip."""

    route_id: str
    trip_id: str
    station_id: str
    schedule_relationship: str
    scheduled_at: datetime | None
    predicted_at: datetime | None
    delay_seconds: int | None
    direction_id: int | None


class MetraClient:
    def __init__(
        self,
        api_key: str,
        *,
        http: httpx.AsyncClient | None = None,
        payload_recorder=None,
    ) -> None:
        if not api_key:
            raise ValueError("METRA_API_KEY is required")
        self._key = api_key
        if http is not None:
            self._http = http
        else:
            event_hooks: dict = {}
            if payload_recorder is not None:
                event_hooks["response"] = [payload_recorder]
            self._http = httpx.AsyncClient(timeout=15.0, event_hooks=event_hooks or None)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def fetch_trip_updates(self) -> list[MetraStopUpdate]:
        data = await self._fetch_with_fallback([PRIMARY_URL, FALLBACK_URL])
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(data)
        out: list[MetraStopUpdate] = []
        for entity in feed.entity:
            if not entity.HasField("trip_update"):
                continue
            trip_update = entity.trip_update
            trip = trip_update.trip
            route_id = trip.route_id or ""
            trip_id = trip.trip_id or ""
            direction_id = trip.direction_id if trip.HasField("direction_id") else None
            for stu in trip_update.stop_time_update:
                station_id = stu.stop_id or ""
                if not station_id:
                    continue
                scheduled_at = None  # GTFS-RT doesn't carry the schedule; we'd join it from static GTFS
                predicted_at = _select_predicted(stu)
                relationship = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name(
                    stu.schedule_relationship
                ).lower()
                delay = _select_delay(stu)
                out.append(
                    MetraStopUpdate(
                        route_id=route_id,
                        trip_id=trip_id,
                        station_id=station_id,
                        schedule_relationship=relationship,
                        scheduled_at=scheduled_at,
                        predicted_at=predicted_at,
                        delay_seconds=delay,
                        direction_id=direction_id,
                    )
                )
        return out

    async def _fetch_with_fallback(self, urls: Iterable[str]) -> bytes:
        last_err: Exception | None = None
        for url in urls:
            try:
                resp = await self._http.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {self._key}",
                        "Accept": "application/x-protobuf",
                    },
                )
                resp.raise_for_status()
                return resp.content
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        assert last_err is not None
        raise last_err


def _select_predicted(stu) -> datetime | None:
    """Prefer departure over arrival; many Metra stops only carry one."""
    for field in ("departure", "arrival"):
        if not stu.HasField(field):
            continue
        event = getattr(stu, field)
        if event.HasField("time") and event.time > 0:
            return datetime.fromtimestamp(event.time, tz=CHICAGO)
    return None


def _select_delay(stu) -> int | None:
    for field in ("departure", "arrival"):
        if not stu.HasField(field):
            continue
        event = getattr(stu, field)
        if event.HasField("delay"):
            return int(event.delay)
    return None
