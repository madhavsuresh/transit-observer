"""CTA Customer Alerts client.

Endpoint: ``http://www.transitchicago.com/api/1.0/alerts.aspx`` (no auth).
We pull ``outputType=JSON`` and store each alert seen on every poll —
one row per (poll, alert) — so we can reconstruct the active alert set
at any historical timestamp. Alerts have no public archive; once an
alert is cleared from the live feed it's gone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from .config import CHICAGO


BASE_URL = "http://www.transitchicago.com/api/1.0/alerts.aspx"


@dataclass(frozen=True)
class CTAAlert:
    alert_id: str
    headline: str | None
    short_description: str | None
    full_description: str | None
    severity_score: int | None
    impact: str | None
    event_start: datetime | None
    event_end: datetime | None
    tbd: bool
    major_alert: bool
    alert_url: str | None
    impacted_services_json: str | None
    guid: str | None


def _parse_dt(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        naive = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    return naive.replace(tzinfo=CHICAGO)


def _parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return False


def _unwrap_text(value: Any) -> str | None:
    """CTA's JSON serializer wraps CDATA elements as ``{"#cdata-section": "..."}``."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        cdata = value.get("#cdata-section")
        if isinstance(cdata, str):
            return cdata
    return None


class CTAAlertsClient:
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

    async def fetch_alerts(self) -> list[CTAAlert]:
        params = {"outputType": "JSON", "accessibility": "false"}
        resp = await self._http.get(BASE_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()
        root = payload.get("CTAAlerts", {}) if isinstance(payload, dict) else {}
        alerts = root.get("Alert") or []
        if isinstance(alerts, dict):
            alerts = [alerts]
        out: list[CTAAlert] = []
        for raw in alerts:
            parsed = _alert_from_raw(raw)
            if parsed is not None:
                out.append(parsed)
        return out


def _alert_from_raw(raw: dict) -> CTAAlert | None:
    alert_id = raw.get("AlertId") or raw.get("GUID")
    if not alert_id:
        return None
    services = raw.get("ImpactedService") or {}
    if isinstance(services, dict):
        service_list = services.get("Service") or []
        if isinstance(service_list, dict):
            service_list = [service_list]
        impacted_services_json = json.dumps(service_list) if service_list else None
    else:
        impacted_services_json = json.dumps(services) if services else None
    return CTAAlert(
        alert_id=str(alert_id),
        headline=_unwrap_text(raw.get("Headline")),
        short_description=_unwrap_text(raw.get("ShortDescription")),
        full_description=_unwrap_text(raw.get("FullDescription")),
        severity_score=_parse_int(raw.get("SeverityScore")),
        impact=_unwrap_text(raw.get("Impact")),
        event_start=_parse_dt(raw.get("EventStart")),
        event_end=_parse_dt(raw.get("EventEnd")),
        tbd=_parse_bool(raw.get("TBD")),
        major_alert=_parse_bool(raw.get("MajorAlert")),
        alert_url=_unwrap_text(raw.get("AlertURL")),
        impacted_services_json=impacted_services_json,
        guid=_unwrap_text(raw.get("GUID")) or (str(raw.get("GUID")) if raw.get("GUID") else None),
    )
