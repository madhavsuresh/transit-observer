"""CTA GTFS-Realtime client (TripUpdates + VehiclePositions + Alerts).

CTA publishes standard GTFS-RT feeds alongside the proprietary
ttarrivals/ttpositions and Bus Tracker APIs. The GTFS-RT side carries
canonical ``trip_id`` (joinable to GTFS-static) plus the ``delay`` field
on each stop_time_update — schedule-adherence signal that the
proprietary feeds don't expose for the L.

Three entity types per feed (train, bus). URLs vary by region and have
shifted historically; treat them as configurable. ``url`` is passed at
fetch time rather than init so one client can serve both modes.

Uses the ``gtfs-realtime-bindings`` package already in this project for
Metra/intercampus.

This client captures *every* documented protobuf field. Downstream
storage chooses what to project; raw payloads should never be parsed
twice for fields we forgot the first time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from google.transit import gtfs_realtime_pb2

from .config import CHICAGO


@dataclass(frozen=True)
class CTAGtfsRtFeedMeta:
    """``FeedHeader`` fields, broadcast on every entity from the same feed."""

    feed_timestamp: datetime | None
    feed_incrementality: str | None         # 'full_dataset' | 'differential'
    feed_realtime_version: str | None


@dataclass(frozen=True)
class CTAGtfsRtTripUpdate:
    mode: str                               # 'train' | 'bus'
    # FeedHeader
    feed_timestamp: datetime | None
    feed_incrementality: str | None
    # TripUpdate
    trip_update_timestamp: datetime | None  # when this trip_update was last computed
    trip_update_delay_seconds: int | None   # trip-level delay (deprecated; some feeds use it)
    # TripDescriptor
    route_id: str | None
    trip_id: str | None
    trip_start_date: str | None             # "YYYYMMDD"
    trip_start_time: str | None             # "HH:MM:SS"
    trip_direction_id: int | None
    trip_schedule_relationship: str | None  # 'scheduled' | 'added' | 'unscheduled' | 'canceled' | ...
    # StopTimeUpdate
    stop_id: str | None
    stop_sequence: int | None
    arrival_time: datetime | None
    arrival_delay_seconds: int | None
    arrival_uncertainty_seconds: int | None
    departure_time: datetime | None
    departure_delay_seconds: int | None
    departure_uncertainty_seconds: int | None
    schedule_relationship: str | None       # stop-level schedule relationship
    # VehicleDescriptor (attached to TripUpdate)
    vehicle_id: str | None
    vehicle_label: str | None
    vehicle_license_plate: str | None


@dataclass(frozen=True)
class CTAGtfsRtVehiclePosition:
    mode: str
    # FeedHeader
    feed_timestamp: datetime | None
    feed_incrementality: str | None
    # VehiclePosition
    vehicle_timestamp: datetime | None       # vehicle's own report time
    stop_id: str | None                       # which stop the vehicle is currently associated with
    current_stop_sequence: int | None
    current_status: str | None
    congestion_level: str | None
    occupancy_status: str | None
    occupancy_percentage: int | None
    multi_carriage_details_json: str | None   # per-car occupancy if the agency populates it
    # TripDescriptor
    route_id: str | None
    trip_id: str | None
    trip_start_date: str | None
    trip_start_time: str | None
    trip_direction_id: int | None
    trip_schedule_relationship: str | None
    # VehicleDescriptor
    vehicle_id: str | None
    vehicle_label: str | None
    vehicle_license_plate: str | None
    # Position
    lat: float | None
    lon: float | None
    bearing: float | None
    speed_mps: float | None
    odometer_m: float | None


@dataclass(frozen=True)
class CTAGtfsRtAlert:
    """GTFS-RT ``Alert`` entity. CTA disruptions show up here in
    addition to the proprietary RSS feed; both should be captured."""

    mode: str
    entity_id: str | None
    feed_timestamp: datetime | None
    feed_incrementality: str | None
    cause: str | None
    effect: str | None
    severity_level: str | None
    header_text: str | None
    description_text: str | None
    tts_header_text: str | None
    tts_description_text: str | None
    url: str | None
    active_period_json: str | None      # list of {start_ms, end_ms}
    informed_entity_json: str | None    # list of EntitySelector dicts


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

    async def _fetch_feed(self, url: str) -> gtfs_realtime_pb2.FeedMessage:
        resp = await self._http.get(url, headers={"Accept": "application/x-protobuf"})
        resp.raise_for_status()
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(resp.content)
        return feed

    async def fetch_trip_updates(self, *, url: str, mode: str) -> list[CTAGtfsRtTripUpdate]:
        feed = await self._fetch_feed(url)
        meta = _feed_meta(feed)
        out: list[CTAGtfsRtTripUpdate] = []
        for entity in feed.entity:
            if not entity.HasField("trip_update"):
                continue
            tu = entity.trip_update
            trip = tu.trip
            tu_timestamp = (
                datetime.fromtimestamp(tu.timestamp, tz=CHICAGO)
                if tu.HasField("timestamp") and tu.timestamp > 0
                else None
            )
            tu_delay = int(tu.delay) if tu.HasField("delay") else None
            vehicle = tu.vehicle if tu.HasField("vehicle") else None
            trip_relationship = _trip_schedule_relationship(trip)
            for stu in tu.stop_time_update:
                arrival_time, arrival_delay, arrival_uncertainty = _stop_event(stu, "arrival")
                departure_time, departure_delay, departure_uncertainty = _stop_event(stu, "departure")
                relationship = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name(
                    stu.schedule_relationship
                ).lower()
                out.append(
                    CTAGtfsRtTripUpdate(
                        mode=mode,
                        feed_timestamp=meta.feed_timestamp,
                        feed_incrementality=meta.feed_incrementality,
                        trip_update_timestamp=tu_timestamp,
                        trip_update_delay_seconds=tu_delay,
                        route_id=trip.route_id or None,
                        trip_id=trip.trip_id or None,
                        trip_start_date=trip.start_date or None,
                        trip_start_time=trip.start_time or None,
                        trip_direction_id=trip.direction_id if trip.HasField("direction_id") else None,
                        trip_schedule_relationship=trip_relationship,
                        stop_id=stu.stop_id or None,
                        stop_sequence=stu.stop_sequence if stu.HasField("stop_sequence") else None,
                        arrival_time=arrival_time,
                        arrival_delay_seconds=arrival_delay,
                        arrival_uncertainty_seconds=arrival_uncertainty,
                        departure_time=departure_time,
                        departure_delay_seconds=departure_delay,
                        departure_uncertainty_seconds=departure_uncertainty,
                        schedule_relationship=relationship,
                        vehicle_id=(vehicle.id if vehicle and vehicle.id else None),
                        vehicle_label=(vehicle.label if vehicle and vehicle.label else None),
                        vehicle_license_plate=(
                            vehicle.license_plate if vehicle and vehicle.license_plate else None
                        ),
                    )
                )
        return out

    async def fetch_vehicle_positions(
        self, *, url: str, mode: str
    ) -> list[CTAGtfsRtVehiclePosition]:
        feed = await self._fetch_feed(url)
        meta = _feed_meta(feed)
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
            occupancy_pct = (
                int(vp.occupancy_percentage) if vp.HasField("occupancy_percentage") else None
            )
            multi_carriage = _multi_carriage_json(vp)
            vehicle_timestamp = (
                datetime.fromtimestamp(vp.timestamp, tz=CHICAGO)
                if vp.HasField("timestamp") and vp.timestamp > 0
                else None
            )
            out.append(
                CTAGtfsRtVehiclePosition(
                    mode=mode,
                    feed_timestamp=meta.feed_timestamp,
                    feed_incrementality=meta.feed_incrementality,
                    vehicle_timestamp=vehicle_timestamp,
                    stop_id=vp.stop_id or None,
                    current_stop_sequence=(
                        vp.current_stop_sequence if vp.HasField("current_stop_sequence") else None
                    ),
                    current_status=current_status,
                    congestion_level=congestion,
                    occupancy_status=occupancy,
                    occupancy_percentage=occupancy_pct,
                    multi_carriage_details_json=multi_carriage,
                    route_id=trip.route_id if trip and trip.route_id else None,
                    trip_id=trip.trip_id if trip and trip.trip_id else None,
                    trip_start_date=trip.start_date if trip and trip.start_date else None,
                    trip_start_time=trip.start_time if trip and trip.start_time else None,
                    trip_direction_id=(
                        trip.direction_id if trip and trip.HasField("direction_id") else None
                    ),
                    trip_schedule_relationship=_trip_schedule_relationship(trip) if trip else None,
                    vehicle_id=vehicle.id if vehicle and vehicle.id else None,
                    vehicle_label=vehicle.label if vehicle and vehicle.label else None,
                    vehicle_license_plate=(
                        vehicle.license_plate if vehicle and vehicle.license_plate else None
                    ),
                    lat=position.latitude if position and position.HasField("latitude") else None,
                    lon=position.longitude if position and position.HasField("longitude") else None,
                    bearing=position.bearing if position and position.HasField("bearing") else None,
                    speed_mps=position.speed if position and position.HasField("speed") else None,
                    odometer_m=(
                        position.odometer if position and position.HasField("odometer") else None
                    ),
                )
            )
        return out

    async def fetch_alerts(self, *, url: str, mode: str) -> list[CTAGtfsRtAlert]:
        """Parse GTFS-RT ``Alert`` entities. Each alert carries the
        list of impacted (route/trip/stop) selectors, severity, cause,
        effect, free-form headline + description, and one or more
        active time ranges."""
        feed = await self._fetch_feed(url)
        meta = _feed_meta(feed)
        out: list[CTAGtfsRtAlert] = []
        for entity in feed.entity:
            if not entity.HasField("alert"):
                continue
            a = entity.alert
            out.append(
                CTAGtfsRtAlert(
                    mode=mode,
                    entity_id=entity.id or None,
                    feed_timestamp=meta.feed_timestamp,
                    feed_incrementality=meta.feed_incrementality,
                    cause=_enum_name(gtfs_realtime_pb2.Alert.Cause, a.cause) if a.HasField("cause") else None,
                    effect=_enum_name(gtfs_realtime_pb2.Alert.Effect, a.effect) if a.HasField("effect") else None,
                    severity_level=_enum_name(gtfs_realtime_pb2.Alert.SeverityLevel, a.severity_level)
                        if a.HasField("severity_level") else None,
                    header_text=_translated_string(a.header_text),
                    description_text=_translated_string(a.description_text),
                    tts_header_text=_translated_string(a.tts_header_text),
                    tts_description_text=_translated_string(a.tts_description_text),
                    url=_translated_string(a.url),
                    active_period_json=_active_period_json(a.active_period),
                    informed_entity_json=_informed_entity_json(a.informed_entity),
                )
            )
        return out


def _feed_meta(feed: gtfs_realtime_pb2.FeedMessage) -> CTAGtfsRtFeedMeta:
    header = feed.header
    ts = (
        datetime.fromtimestamp(header.timestamp, tz=CHICAGO)
        if header.HasField("timestamp") and header.timestamp > 0
        else None
    )
    incrementality = (
        gtfs_realtime_pb2.FeedHeader.Incrementality.Name(header.incrementality).lower()
        if header.HasField("incrementality")
        else None
    )
    version = header.gtfs_realtime_version or None
    return CTAGtfsRtFeedMeta(
        feed_timestamp=ts,
        feed_incrementality=incrementality,
        feed_realtime_version=version,
    )


def _stop_event(stu, name: str) -> tuple[datetime | None, int | None, int | None]:
    if not stu.HasField(name):
        return None, None, None
    event = getattr(stu, name)
    when = None
    if event.HasField("time") and event.time > 0:
        when = datetime.fromtimestamp(event.time, tz=CHICAGO)
    delay = int(event.delay) if event.HasField("delay") else None
    uncertainty = int(event.uncertainty) if event.HasField("uncertainty") else None
    return when, delay, uncertainty


def _trip_schedule_relationship(trip) -> str | None:
    if trip is None or not trip.HasField("schedule_relationship"):
        return None
    return gtfs_realtime_pb2.TripDescriptor.ScheduleRelationship.Name(
        trip.schedule_relationship
    ).lower()


def _enum_name(enum_cls, value) -> str | None:
    try:
        return enum_cls.Name(value).lower()
    except Exception:  # noqa: BLE001
        return None


def _translated_string(ts) -> str | None:
    """``TranslatedString`` → pick the first translation. Most CTA
    alerts come in a single language so the heuristic is fine."""
    if ts is None or not ts.translation:
        return None
    return ts.translation[0].text or None


def _active_period_json(periods) -> str | None:
    if not periods:
        return None
    out = []
    for p in periods:
        out.append(
            {
                "start": int(p.start) if p.HasField("start") else None,
                "end": int(p.end) if p.HasField("end") else None,
            }
        )
    return json.dumps(out, sort_keys=True, separators=(",", ":"))


def _informed_entity_json(entities) -> str | None:
    if not entities:
        return None
    out: list[dict[str, Any]] = []
    for e in entities:
        item: dict[str, Any] = {}
        if e.agency_id:
            item["agency_id"] = e.agency_id
        if e.route_id:
            item["route_id"] = e.route_id
        if e.HasField("route_type"):
            item["route_type"] = int(e.route_type)
        if e.stop_id:
            item["stop_id"] = e.stop_id
        if e.HasField("direction_id"):
            item["direction_id"] = int(e.direction_id)
        if e.HasField("trip"):
            item["trip"] = {
                "trip_id": e.trip.trip_id or None,
                "route_id": e.trip.route_id or None,
                "start_date": e.trip.start_date or None,
                "start_time": e.trip.start_time or None,
                "direction_id": (
                    int(e.trip.direction_id) if e.trip.HasField("direction_id") else None
                ),
            }
        out.append(item)
    return json.dumps(out, sort_keys=True, separators=(",", ":"))


def _multi_carriage_json(vp) -> str | None:
    if not vp.multi_carriage_details:
        return None
    out = []
    for c in vp.multi_carriage_details:
        item: dict[str, Any] = {}
        if c.id:
            item["id"] = c.id
        if c.label:
            item["label"] = c.label
        if c.HasField("occupancy_status"):
            item["occupancy_status"] = _enum_name(
                gtfs_realtime_pb2.VehiclePosition.OccupancyStatus, c.occupancy_status
            )
        if c.HasField("occupancy_percentage"):
            item["occupancy_percentage"] = int(c.occupancy_percentage)
        if c.HasField("carriage_sequence"):
            item["carriage_sequence"] = int(c.carriage_sequence)
        out.append(item)
    return json.dumps(out, sort_keys=True, separators=(",", ":"))
