"""CTA Bus Tracker client.

Endpoints:
- /getpredictions?rt=<route>&stpid=<stop> — predictions at a stop
- /getvehicles?rt=<route_list> — vehicles per route(s)

Auth: API key (CTA_BUS_API_KEY env var; same key Cozy Fox uses).
Rate limit: 10,000 requests per day per key — much higher headroom
than the L's 100 per 5 min. The collector can poll predictions for a
monitored stop set every minute.

We use JSON (the API also supports XML). Times come back as
"YYYYMMDD HH:MM" in local Chicago time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import httpx

from .config import CHICAGO


BASE_URL = "https://ctabustracker.com/bustime/api/v2"


@dataclass(frozen=True)
class BusPrediction:
    route: str
    route_name: str | None
    vehicle_id: str
    stop_id: int
    stop_name: str
    destination_name: str
    direction_name: str
    generated_at: datetime
    arrival_at: datetime
    is_delayed: bool
    is_approaching: bool


def _parse_local_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        naive = datetime.strptime(value, "%Y%m%d %H:%M")
    except ValueError:
        return None
    return naive.replace(tzinfo=CHICAGO)


class CTABusClient:
    def __init__(self, api_key: str, *, http: httpx.AsyncClient | None = None) -> None:
        if not api_key:
            raise ValueError("CTA_BUS_API_KEY is required")
        self._key = api_key
        self._http = http or httpx.AsyncClient(timeout=10.0)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def fetch_predictions(self, *, route: str, stop_id: int, top: int = 4) -> list[BusPrediction]:
        params = {
            "key": self._key,
            "rt": route,
            "stpid": str(stop_id),
            "top": str(top),
            "format": "json",
        }
        resp = await self._http.get(f"{BASE_URL}/getpredictions", params=params)
        resp.raise_for_status()
        payload = resp.json()
        body = payload.get("bustime-response", {})
        out: list[BusPrediction] = []
        for raw in body.get("prd") or []:
            pred = _from_prediction(raw)
            if pred is not None:
                out.append(pred)
        return out


def _from_prediction(raw: dict) -> BusPrediction | None:
    generated = _parse_local_dt(raw.get("tmstmp"))
    arrival = _parse_local_dt(raw.get("prdtm"))
    if generated is None or arrival is None:
        return None
    try:
        stop_id = int(raw.get("stpid") or 0)
    except (TypeError, ValueError):
        return None
    countdown_raw = raw.get("prdctdn") or "99"
    if isinstance(countdown_raw, str) and countdown_raw.strip().upper() == "DUE":
        countdown = 0
    else:
        try:
            countdown = int(countdown_raw)
        except (TypeError, ValueError):
            countdown = 99
    return BusPrediction(
        route=str(raw.get("rt", "")),
        route_name=raw.get("rtdir"),
        vehicle_id=str(raw.get("vid", "")),
        stop_id=stop_id,
        stop_name=str(raw.get("stpnm", "")),
        destination_name=str(raw.get("des", "")),
        direction_name=str(raw.get("rtdir", "")),
        generated_at=generated,
        arrival_at=arrival,
        is_delayed=bool(raw.get("dly", False)),
        is_approaching=countdown <= 1,
    )
