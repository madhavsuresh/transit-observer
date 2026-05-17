"""Ticketmaster Discovery API client (concerts + venue events).

Free with API key (``TICKETMASTER_API_KEY`` env or ``[api_keys]
ticketmaster`` in ``config.toml``). 5000 requests/day on the public tier.

We pull events for Chicago (DMA 249) on a weekly forward-looking
cadence. Captures concerts at Salt Shed, Aragon, Riviera, Lincoln
Hall, Metro, House of Blues, United Center, Wintrust, etc. — major
transit-demand drivers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import httpx

from .config import CHICAGO


BASE_URL = "https://app.ticketmaster.com/discovery/v2/events.json"
CHICAGO_DMA_ID = 249


@dataclass(frozen=True)
class VenueEvent:
    event_id: str
    name: str | None
    venue_name: str | None
    venue_city: str | None
    scheduled_start: datetime | None
    sales_start: datetime | None
    sales_end: datetime | None
    classification: str | None
    genre: str | None
    url: str | None
    raw_payload_json: str


def _parse_iso_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(CHICAGO)
    except (ValueError, AttributeError):
        return None


class VenueClient:
    def __init__(
        self,
        api_key: str,
        *,
        http: httpx.AsyncClient | None = None,
        payload_recorder=None,
    ) -> None:
        if not api_key:
            raise ValueError("TICKETMASTER_API_KEY is required")
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

    async def fetch_chicago_events(
        self,
        *,
        days_forward: int = 30,
        page_size: int = 200,
    ) -> list[VenueEvent]:
        now = datetime.now(CHICAGO)
        end = now + timedelta(days=days_forward)
        out: list[VenueEvent] = []
        page = 0
        while True:
            params = {
                "apikey": self._key,
                "dmaId": str(CHICAGO_DMA_ID),
                "startDateTime": now.astimezone().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "endDateTime": end.astimezone().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "size": str(page_size),
                "page": str(page),
                "sort": "date,asc",
            }
            resp = await self._http.get(BASE_URL, params=params)
            if resp.status_code == 429:
                # rate limited; stop polling this cycle
                break
            resp.raise_for_status()
            payload = resp.json()
            events = (payload.get("_embedded") or {}).get("events") or []
            for raw in events:
                e = _event_from_raw(raw)
                if e is not None:
                    out.append(e)
            page_info = payload.get("page") or {}
            total_pages = page_info.get("totalPages", 1)
            page += 1
            if page >= total_pages or page >= 5:
                # Cap at 5 pages (~1000 events) per poll to keep within budget.
                break
        return out


def _event_from_raw(raw: dict) -> VenueEvent | None:
    event_id = raw.get("id")
    if not event_id:
        return None
    venues = ((raw.get("_embedded") or {}).get("venues")) or [{}]
    venue = venues[0] if venues else {}
    city = (venue.get("city") or {}).get("name")
    classifications = raw.get("classifications") or [{}]
    classification = classifications[0] if classifications else {}
    segment = (classification.get("segment") or {}).get("name")
    genre = (classification.get("genre") or {}).get("name")
    dates = raw.get("dates") or {}
    start = (dates.get("start") or {}).get("dateTime")
    sales = (raw.get("sales") or {}).get("public") or {}
    return VenueEvent(
        event_id=str(event_id),
        name=raw.get("name"),
        venue_name=venue.get("name"),
        venue_city=city,
        scheduled_start=_parse_iso_dt(start),
        sales_start=_parse_iso_dt(sales.get("startDateTime")),
        sales_end=_parse_iso_dt(sales.get("endDateTime")),
        classification=segment,
        genre=genre,
        url=raw.get("url"),
        raw_payload_json=json.dumps(raw),
    )
