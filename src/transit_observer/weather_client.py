"""Open-Meteo weather client.

Free, no key. We capture *current* conditions at each configured site
on a slow cadence. Historical Open-Meteo is also free and joinable on
(lat, lon, timestamp), so we don't need a per-prediction snapshot —
but capturing live makes the join trivial and pins the exact field
set we use.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import httpx

from .config import CHICAGO


BASE_URL = "https://api.open-meteo.com/v1/forecast"

_CURRENT_FIELDS = (
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
)


@dataclass(frozen=True)
class WeatherObservation:
    site_id: str
    lat: float
    lon: float
    observation_time: datetime | None
    temperature_c: float | None
    apparent_temperature_c: float | None
    humidity_pct: float | None
    precipitation_mm: float | None
    rain_mm: float | None
    snowfall_cm: float | None
    wind_speed_kph: float | None
    wind_gust_kph: float | None
    wind_direction_deg: float | None
    cloud_cover_pct: float | None
    pressure_hpa: float | None
    weather_code: int | None


def _parse_iso_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # Open-Meteo returns "YYYY-MM-DDTHH:MM" — local-time-naive (UTC by default).
        return datetime.fromisoformat(value).replace(tzinfo=CHICAGO)
    except ValueError:
        return None


class WeatherClient:
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

    async def fetch_current(
        self, *, site_id: str, lat: float, lon: float
    ) -> WeatherObservation | None:
        params = {
            "latitude": str(lat),
            "longitude": str(lon),
            "current": ",".join(_CURRENT_FIELDS),
            "timezone": "America/Chicago",
            "wind_speed_unit": "kmh",
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
        }
        resp = await self._http.get(BASE_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()
        current = payload.get("current") or {}
        if not current:
            return None
        return WeatherObservation(
            site_id=site_id,
            lat=lat,
            lon=lon,
            observation_time=_parse_iso_dt(current.get("time")),
            temperature_c=current.get("temperature_2m"),
            apparent_temperature_c=current.get("apparent_temperature"),
            humidity_pct=current.get("relative_humidity_2m"),
            precipitation_mm=current.get("precipitation"),
            rain_mm=current.get("rain"),
            snowfall_cm=current.get("snowfall"),
            wind_speed_kph=current.get("wind_speed_10m"),
            wind_gust_kph=current.get("wind_gusts_10m"),
            wind_direction_deg=current.get("wind_direction_10m"),
            cloud_cover_pct=current.get("cloud_cover"),
            pressure_hpa=current.get("pressure_msl"),
            weather_code=current.get("weather_code"),
        )
