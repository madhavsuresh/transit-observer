"""City of Chicago traffic-congestion client (no auth).

Two endpoints, both backed by Chicago Open Data:

- ``/resource/n4j6-wkkf.json`` — per-segment ("Traffic Tracker by
  Street Segment"): street, direction, from/to cross-streets, current
  estimated speed in MPH, last_updt.
- ``/resource/8v9j-bter.json`` — per-region ("Traffic Tracker by
  Region"): aggregate speed averaged over each of 29 regions.

Updated by CDOT roughly every 10 minutes. No API key needed; no
documented rate limit but be polite.

Each successful poll lands as rows in ``cdot_traffic_observation``,
tagged by ``source`` so segment-level and region-level coexist.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


SEGMENT_URL = "https://data.cityofchicago.org/resource/n4j6-wkkf.json"
REGION_URL = "https://data.cityofchicago.org/resource/8v9j-bter.json"


@dataclass(frozen=True)
class CDOTTrafficObservation:
    source: str
    polled_at_ms: int
    segment_id: str | None
    region: str | None
    street: str | None
    direction: str | None
    from_street: str | None
    to_street: str | None
    speed: float | None
    bus_count: int | None
    message_count: int | None
    hour_of_day: int | None
    last_updated_ms: int | None
    start_lat: float | None
    start_lon: float | None
    end_lat: float | None
    end_lon: float | None
    raw_payload_json: str


class CDOTTrafficClient:
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
            self._http = httpx.AsyncClient(timeout=30.0, event_hooks=event_hooks or None)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def fetch_segments(self, *, limit: int = 5000) -> list[CDOTTrafficObservation]:
        resp = await self._http.get(SEGMENT_URL, params={"$limit": str(limit)})
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list):
            return []
        polled_at_ms = _now_ms()
        out: list[CDOTTrafficObservation] = []
        import json

        for raw in payload:
            out.append(
                CDOTTrafficObservation(
                    source="cdot_segments",
                    polled_at_ms=polled_at_ms,
                    segment_id=str(raw.get("segmentid"))
                        if raw.get("segmentid") is not None
                        else None,
                    region=raw.get("region_id") or raw.get("region"),
                    street=raw.get("street"),
                    direction=raw.get("direction"),
                    from_street=raw.get("fromstreet") or raw.get("from_street"),
                    to_street=raw.get("tostreet") or raw.get("to_street"),
                    speed=_to_float(raw.get("speed") or raw.get("current_speed")),
                    bus_count=_to_int(raw.get("bus_count")),
                    message_count=_to_int(raw.get("msg_count") or raw.get("message_count")),
                    hour_of_day=_to_int(raw.get("hour")),
                    last_updated_ms=_parse_iso_ms(raw.get("last_updt") or raw.get("_last_updt")),
                    start_lat=_to_float(raw.get("start_lat") or raw.get("_lat_start")),
                    start_lon=_to_float(raw.get("start_lon") or raw.get("_lon_start")),
                    end_lat=_to_float(raw.get("end_lat") or raw.get("_lat_end")),
                    end_lon=_to_float(raw.get("end_lon") or raw.get("_lon_end")),
                    raw_payload_json=json.dumps(raw, ensure_ascii=False),
                )
            )
        return out

    async def fetch_regions(self, *, limit: int = 100) -> list[CDOTTrafficObservation]:
        resp = await self._http.get(REGION_URL, params={"$limit": str(limit)})
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list):
            return []
        polled_at_ms = _now_ms()
        out: list[CDOTTrafficObservation] = []
        import json

        for raw in payload:
            out.append(
                CDOTTrafficObservation(
                    source="cdot_regions",
                    polled_at_ms=polled_at_ms,
                    segment_id=None,
                    region=raw.get("region") or raw.get("region_id"),
                    street=None,
                    direction=None,
                    from_street=None,
                    to_street=None,
                    speed=_to_float(raw.get("current_speed") or raw.get("speed")),
                    bus_count=_to_int(raw.get("bus_count")),
                    message_count=_to_int(raw.get("msg_count") or raw.get("message_count")),
                    hour_of_day=_to_int(raw.get("hour")),
                    last_updated_ms=_parse_iso_ms(raw.get("last_updt") or raw.get("_last_updt")),
                    start_lat=_to_float(raw.get("_lat") or raw.get("center_lat")),
                    start_lon=_to_float(raw.get("_lon") or raw.get("center_lon")),
                    end_lat=None,
                    end_lon=None,
                    raw_payload_json=json.dumps(raw, ensure_ascii=False),
                )
            )
        return out


def _now_ms() -> int:
    return int(time.time() * 1000)


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_iso_ms(value: Any) -> int | None:
    """Parse an ISO-ish timestamp (with or without timezone) into ms epoch."""
    if not value:
        return None
    from datetime import datetime, timezone

    s = str(value).strip().replace("Z", "+00:00")
    # Strip subsecond precision past microseconds and 'T' separators.
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    return None


def insert_cdot_observations(
    conn,
    observations: list[CDOTTrafficObservation],
) -> int:
    """Bulk insert into ``cdot_traffic_observation``."""
    if not observations:
        return 0
    rows = [
        (
            obs.polled_at_ms, obs.source, obs.segment_id, obs.region, obs.street,
            obs.direction, obs.from_street, obs.to_street, obs.speed,
            obs.bus_count, obs.message_count, obs.hour_of_day, obs.last_updated_ms,
            obs.start_lat, obs.start_lon, obs.end_lat, obs.end_lon,
            obs.raw_payload_json,
        )
        for obs in observations
    ]
    conn.executemany(
        """
        INSERT INTO cdot_traffic_observation(
            polled_at_ms, source, segment_id, region, street, direction,
            from_street, to_street, speed, bus_count, message_count,
            hour_of_day, last_updated_ms, start_lat, start_lon, end_lat, end_lon,
            raw_payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)
