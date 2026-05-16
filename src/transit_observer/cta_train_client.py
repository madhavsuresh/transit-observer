"""HTTP client for the CTA Train Tracker API.

API docs: https://www.transitchicago.com/developers/ttdocs/

The two endpoints we use:
- ttarrivals.aspx?mapid=<map_id>&max=N — predictions at a station
- ttpositions.aspx?rt=<lines> — current vehicle positions per line

Rate limit: 100 requests per 5-minute window per key. The collector
enforces this; this client just makes single requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import httpx

from .config import CHICAGO


BASE_URL = "http://lapi.transitchicago.com/api/1.0"


@dataclass(frozen=True)
class ArrivalRaw:
    """One predicted arrival at a station for a specific run."""

    line: str            # e.g. "Red", "Blue", "Brn", "G", "Org", "P", "Pink", "Y"
    run_number: str
    map_id: int          # station identifier
    stop_id: int         # platform identifier
    station_name: str
    direction_code: str | None
    destination_name: str
    predicted_at: datetime
    arrival_at: datetime
    is_approaching: bool
    is_delayed: bool
    is_fault: bool
    is_scheduled: bool


@dataclass(frozen=True)
class VehiclePositionRaw:
    line: str
    run_number: str
    destination_name: str | None
    direction_code: str | None
    next_station_map_id: int | None
    next_station_name: str | None
    predicted_at: datetime | None
    next_arrival_at: datetime | None
    is_approaching: bool
    is_delayed: bool


def _parse_local_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    # CTA returns "2026-05-13T08:23:00" — local time, no zone marker.
    try:
        naive = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    return naive.replace(tzinfo=CHICAGO)


def _bool_one(flag: str | None) -> bool:
    return flag == "1"


class CTATrainClient:
    """Thin async client. Doesn't enforce rate limits — collector does."""

    def __init__(self, api_key: str, *, http: httpx.AsyncClient | None = None) -> None:
        if not api_key:
            raise ValueError("CTA_TRAIN_API_KEY is required")
        self._key = api_key
        self._http = http or httpx.AsyncClient(timeout=10.0)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def fetch_arrivals(self, *, map_id: int, max_predictions: int = 12) -> list[ArrivalRaw]:
        params = {
            "key": self._key,
            "mapid": str(map_id),
            "max": str(max_predictions),
            "outputType": "JSON",
        }
        resp = await self._http.get(f"{BASE_URL}/ttarrivals.aspx", params=params)
        resp.raise_for_status()
        payload = resp.json()
        ctatt = payload.get("ctatt", {})
        if ctatt.get("errCd") not in (None, "0"):
            return []
        out: list[ArrivalRaw] = []
        for raw in ctatt.get("eta") or []:
            arrival = _arrival_from_eta(raw)
            if arrival is not None:
                out.append(arrival)
        return out

    async def fetch_positions(self, *, line_codes: Iterable[str]) -> list[VehiclePositionRaw]:
        params = {
            "key": self._key,
            "rt": ",".join(line_codes),
            "outputType": "JSON",
        }
        resp = await self._http.get(f"{BASE_URL}/ttpositions.aspx", params=params)
        resp.raise_for_status()
        payload = resp.json()
        ctatt = payload.get("ctatt", {})
        if ctatt.get("errCd") not in (None, "0"):
            return []
        out: list[VehiclePositionRaw] = []
        for route in ctatt.get("route") or []:
            line = route.get("@name") or route.get("name") or ""
            for train in route.get("train") or []:
                position = _position_from_train(line=line, raw=train)
                if position is not None:
                    out.append(position)
        return out


def _arrival_from_eta(raw: dict) -> ArrivalRaw | None:
    try:
        map_id = int(raw["staId"])
        stop_id = int(raw["stpId"])
    except (KeyError, ValueError, TypeError):
        return None
    predicted_at = _parse_local_dt(raw.get("prdt"))
    arrival_at = _parse_local_dt(raw.get("arrT"))
    if predicted_at is None or arrival_at is None:
        return None
    return ArrivalRaw(
        line=str(raw.get("rt", "")).strip(),
        run_number=str(raw.get("rn", "")).strip(),
        map_id=map_id,
        stop_id=stop_id,
        station_name=str(raw.get("staNm", "")),
        direction_code=raw.get("trDr"),
        destination_name=str(raw.get("destNm", "")),
        predicted_at=predicted_at,
        arrival_at=arrival_at,
        is_approaching=_bool_one(raw.get("isApp")),
        is_delayed=_bool_one(raw.get("isDly")),
        is_fault=_bool_one(raw.get("isFlt")),
        is_scheduled=_bool_one(raw.get("isSch")),
    )


def _position_from_train(*, line: str, raw: dict) -> VehiclePositionRaw | None:
    next_map_id: int | None = None
    if next_str := raw.get("nextStaId"):
        try:
            next_map_id = int(next_str)
        except ValueError:
            next_map_id = None
    return VehiclePositionRaw(
        line=line,
        run_number=str(raw.get("rn", "")).strip(),
        destination_name=raw.get("destNm"),
        direction_code=raw.get("trDr"),
        next_station_map_id=next_map_id,
        next_station_name=raw.get("nextStaNm"),
        predicted_at=_parse_local_dt(raw.get("prdt")),
        next_arrival_at=_parse_local_dt(raw.get("arrT")),
        is_approaching=_bool_one(raw.get("isApp")),
        is_delayed=_bool_one(raw.get("isDly")),
    )
