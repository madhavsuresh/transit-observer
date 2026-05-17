"""AirNow API client (EPA / US-only free AQI data).

Endpoints:
- ``/aq/observation/zipCode/current/`` — current observed AQI per pollutant
- ``/aq/forecast/zipCode/`` — same-day / next-day forecast AQI per pollutant

Auth: free API key (env ``AIRNOW_API_KEY`` or ``[api_keys] airnow`` in
``config.toml``). Modest rate limit (500/hour), so the daily poll across
a handful of zip codes is well within budget.

Each (zip, hour) returns 1-3 observation rows (one per parameter:
ozone, pm2.5, pm10) — we store them all. Tier-2 widening adds
``category_number`` (numeric AQI category 1-6, the actionable form;
Name is for display) and ``state_code`` (for cross-agency joins),
plus a separate forecast capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from .config import CHICAGO


BASE_URL = "https://www.airnowapi.org/aq/observation/zipCode/current/"
FORECAST_URL = "https://www.airnowapi.org/aq/forecast/zipCode/"


@dataclass(frozen=True)
class AirQualityObservation:
    site_id: str
    parameter: str
    aqi: int | None
    raw_value: float | None
    unit: str | None
    category: str | None
    category_number: int | None
    state_code: str | None
    observation_time: datetime | None
    reporting_area: str | None
    latitude: float | None
    longitude: float | None
    is_forecast: bool = False
    forecast_date: str | None = None         # YYYY-MM-DD when ``is_forecast=True``
    action_day: bool | None = None           # AirNow "action day" flag from forecast endpoint


def _parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_dt(date_str: str | None, hour: int | None) -> datetime | None:
    if not date_str or hour is None:
        return None
    try:
        d = datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError:
        return None
    return d.replace(hour=int(hour), tzinfo=CHICAGO)


class AirQualityClient:
    def __init__(
        self,
        api_key: str,
        *,
        http: httpx.AsyncClient | None = None,
        payload_recorder=None,
    ) -> None:
        if not api_key:
            raise ValueError("AIRNOW_API_KEY is required")
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

    async def fetch_zip(self, zip_code: str, *, distance_miles: int = 25) -> list[AirQualityObservation]:
        params = {
            "format": "application/json",
            "zipCode": zip_code,
            "distance": str(distance_miles),
            "API_KEY": self._key,
        }
        resp = await self._http.get(BASE_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list):
            return []
        out: list[AirQualityObservation] = []
        for raw in payload:
            parameter = raw.get("ParameterName")
            if not parameter:
                continue
            category = raw.get("Category") or {}
            category_name = category.get("Name") if isinstance(category, dict) else None
            category_number = (
                _parse_int(category.get("Number")) if isinstance(category, dict) else None
            )
            out.append(
                AirQualityObservation(
                    site_id=zip_code,
                    parameter=str(parameter).lower(),
                    aqi=_parse_int(raw.get("AQI")),
                    raw_value=_parse_float(raw.get("RawConcentration")),
                    unit=raw.get("Unit"),
                    category=category_name,
                    category_number=category_number,
                    state_code=raw.get("StateCode"),
                    observation_time=_parse_dt(raw.get("DateObserved"), raw.get("HourObserved")),
                    reporting_area=raw.get("ReportingArea"),
                    latitude=_parse_float(raw.get("Latitude")),
                    longitude=_parse_float(raw.get("Longitude")),
                    is_forecast=False,
                    forecast_date=None,
                    action_day=None,
                )
            )
        return out

    async def fetch_zip_forecast(
        self, zip_code: str, *, distance_miles: int = 25,
    ) -> list[AirQualityObservation]:
        """Same-day + next-day AQI forecast per pollutant.

        AirNow's forecast endpoint returns one row per (date, pollutant)
        with a Discussion / ActionDay flag. We project them through the
        same dataclass shape so they share the storage path; ``is_forecast``
        and ``forecast_date`` distinguish them from current observations.
        """
        params = {
            "format": "application/json",
            "zipCode": zip_code,
            "distance": str(distance_miles),
            "API_KEY": self._key,
        }
        resp = await self._http.get(FORECAST_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list):
            return []
        out: list[AirQualityObservation] = []
        for raw in payload:
            parameter = raw.get("ParameterName")
            if not parameter:
                continue
            category = raw.get("Category") or {}
            category_name = category.get("Name") if isinstance(category, dict) else None
            category_number = (
                _parse_int(category.get("Number")) if isinstance(category, dict) else None
            )
            date_str = raw.get("DateForecast") or raw.get("DateIssue")
            action_day_raw = raw.get("ActionDay")
            if isinstance(action_day_raw, str):
                action_day = action_day_raw.strip().lower() in {"true", "1", "yes", "y"}
            elif isinstance(action_day_raw, bool):
                action_day = action_day_raw
            else:
                action_day = None
            out.append(
                AirQualityObservation(
                    site_id=zip_code,
                    parameter=str(parameter).lower(),
                    aqi=_parse_int(raw.get("AQI")),
                    raw_value=None,
                    unit=None,
                    category=category_name,
                    category_number=category_number,
                    state_code=raw.get("StateCode"),
                    observation_time=None,
                    reporting_area=raw.get("ReportingArea"),
                    latitude=_parse_float(raw.get("Latitude")),
                    longitude=_parse_float(raw.get("Longitude")),
                    is_forecast=True,
                    forecast_date=date_str.strip() if isinstance(date_str, str) else None,
                    action_day=action_day,
                )
            )
        return out
