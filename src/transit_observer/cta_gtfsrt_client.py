"""CTA GTFS-Realtime client (TripUpdates + VehiclePositions).

CTA publishes standard GTFS-RT feeds alongside the proprietary
ttarrivals/ttpositions and Bus Tracker APIs. The GTFS-RT side carries
canonical ``trip_id`` (joinable to GTFS-static) plus the ``delay`` field
on each stop_time_update — schedule-adherence signal that the
proprietary feeds don't expose for the L.

Two parallel feeds per mode (train, bus). URLs vary by region and have
shifted historically; treat them as configurable. ``url`` is passed at
fetch time rather than init so one client can serve both modes.

Uses the ``gtfs-realtime-bindings`` package already in this project for
Metra/intercampus.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import httpx
from google.transit import gtfs_realtime_pb2

from .config import CHICAGO


@dataclass(frozen=True)
class CTAGtfsRtTripUpdate:
    mode: str                       # 'train' | 'bus'
    route_id: str | None
    trip_id: str | None
    stop_id: str | None
    stop_sequence: int | None
    arrival_time: datetime | None
    arrival_delay_seconds: int | None
    departure_time: datetime | None
    departure_delay_seconds: int | None
    schedule_relationship: str | None
    vehicle_id: str | None


@dataclass(frozen=True)
class CTAGtfsRtVehiclePosition:
    mode: str
    route_id: str | None
    trip_id: str | None
    vehicle_id: str | None
    vehicle_label: str | None
    lat: float | None
    lon: float | None
    bearing: float | None
    speed_mps: float | None
    current_stop_sequence: int | None
    current_status: str | None
    congestion_level: str | None
    occupancy_status: str | None


class CTAGtfsRtClient:
    def __init__(
        self,
        *,
        http: httpx.AsyncClient | None = None,
        payload_recorder=None,
    ) -> None:
        if http is not None:
            self._http = http
        else:
            event_hooks: dict = {}
            if payload_recorder is not None:
                event_hooks["response"] = [payload_recorder]
            self._http = httpx.AsyncClient(timeout=15.0, event_hooks=event_hooks or None)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def fetch_trip_updates(self, *, url: str, mode: str) -> list[CTAGtfsRtTripUpdate]:
        resp = await self._http.get(url, headers={"Accept": "application/x-protobuf"})
        resp.raise_for_status()
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(resp.content)
        out: list[CTAGtfsRtTripUpdate] = []
        for entity in feed.entity:
            if not entity.HasField("trip_update"):
                continue
            trip_update = entity.trip_update
            trip = trip_update.trip
            vehicle_id = trip_update.vehicle.id if trip_update.HasField("vehicle") else None
            for stu in trip_update.stop_time_update:
                arrival_time, arrival_delay = _event_fields(stu, "arrival")
                departure_time, departure_delay = _event_fields(stu, "departure")
                relationship = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name(
                    stu.schedule_relationship
                ).lower()
                out.append(
                    CTAGtfsRtTripUpdate(
                        mode=mode,
                        route_id=trip.route_id or None,
                        trip_id=trip.trip_id or None,
                        stop_id=stu.stop_id or None,
                        stop_sequence=stu.stop_sequence if stu.HasField("stop_sequence") else None,
                        arrival_time=arrival_time,
                        arrival_delay_seconds=arrival_delay,
                        departure_time=departure_time,
                        departure_delay_seconds=departure_delay,
                        schedule_relationship=relationship,
                        vehicle_id=vehicle_id or None,
                    )
                )
        return out

    async def fetch_vehicle_positions(
        self, *, url: str, mode: str
    ) -> list[CTAGtfsRtVehiclePosition]:
        resp = await self._http.get(url, headers={"Accept": "application/x-protobuf"})
        resp.raise_for_status()
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(resp.content)
        out: list[CTAGtfsRtVehiclePosition] = []
        for entity in feed.entity:
            if not entity.HasField("vehicle"):
                continue
            vp = entity.vehicle
            trip = vp.trip if vp.HasField("trip") else None
            position = vp.position if vp.HasField("position") else None
            vehicle = vp.vehicle if vp.HasField("vehicle") else None
            current_status = (
                gtfs_realtime_pb2.VehiclePosition.VehicleStopStatus.Name(vp.current_status).lower()
                if vp.HasField("current_status")
                else None
            )
            congestion = (
                gtfs_realtime_pb2.VehiclePosition.CongestionLevel.Name(vp.congestion_level).lower()
                if vp.HasField("congestion_level")
                else None
            )
            occupancy = (
                gtfs_realtime_pb2.VehiclePosition.OccupancyStatus.Name(vp.occupancy_status).lower()
                if vp.HasField("occupancy_status")
                else None
            )
            out.append(
                CTAGtfsRtVehiclePosition(
                    mode=mode,
                    route_id=trip.route_id if trip and trip.route_id else None,
                    trip_id=trip.trip_id if trip and trip.trip_id else None,
                    vehicle_id=vehicle.id if vehicle and vehicle.id else None,
                    vehicle_label=vehicle.label if vehicle and vehicle.label else None,
                    lat=position.latitude if position and position.HasField("latitude") else None,
                    lon=position.longitude if position and position.HasField("longitude") else None,
                    bearing=position.bearing if position and position.HasField("bearing") else None,
                    speed_mps=position.speed if position and position.HasField("speed") else None,
                    current_stop_sequence=(
                        vp.current_stop_sequence if vp.HasField("current_stop_sequence") else None
                    ),
                    current_status=current_status,
                    congestion_level=congestion,
                    occupancy_status=occupancy,
                )
            )
        return out


def _event_fields(stu, name: str) -> tuple[datetime | None, int | None]:
    if not stu.HasField(name):
        return None, None
    event = getattr(stu, name)
    when = None
    if event.HasField("time") and event.time > 0:
        when = datetime.fromtimestamp(event.time, tz=CHICAGO)
    delay = int(event.delay) if event.HasField("delay") else None
    return when, delay
