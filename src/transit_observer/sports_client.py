"""ESPN unofficial site-api client.

Free, no auth. Used to capture home games for the local sports teams
whose venues drive significant transit demand spikes (Cubs at Wrigley
↔ Red Line, Bears at Soldier ↔ Metra Electric, etc.).

The schedule endpoint returns ~160 events per team-season including
status, attendance, and final scores. We snapshot it on a cadence; the
first poll that observes ``completed=true`` is our proxy for the
game-end time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from .config import CHICAGO


_LEAGUE_PATHS: dict[str, str] = {
    "mlb":  "baseball/mlb",
    "nba":  "basketball/nba",
    "wnba": "basketball/wnba",
    "nhl":  "hockey/nhl",
    "nfl":  "football/nfl",
    "mls":  "soccer/usa.1",
}


BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"


@dataclass(frozen=True)
class SportsEvent:
    event_id: str
    league: str
    sport: str | None
    home_team: str | None
    away_team: str | None
    venue: str | None
    scheduled_start: datetime | None
    status: str | None
    completed: bool
    attendance: int | None
    home_score: int | None
    away_score: int | None
    raw_payload_json: str


def _parse_iso_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(CHICAGO)
    except (ValueError, AttributeError):
        return None


def _parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class SportsClient:
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

    async def fetch_team_schedule(self, league: str, team_abbr: str) -> list[SportsEvent]:
        path = _LEAGUE_PATHS.get(league.lower())
        if not path:
            return []
        url = f"{BASE_URL}/{path}/teams/{team_abbr.lower()}/schedule"
        resp = await self._http.get(url)
        resp.raise_for_status()
        payload = resp.json()
        out: list[SportsEvent] = []
        for raw in payload.get("events") or []:
            event = _event_from_raw(raw, league=league)
            if event is not None:
                out.append(event)
        return out


def _event_from_raw(raw: dict, *, league: str) -> SportsEvent | None:
    event_id = raw.get("id")
    if not event_id:
        return None
    competitions = raw.get("competitions") or [{}]
    comp = competitions[0] if competitions else {}
    venue = (comp.get("venue") or {}).get("fullName")
    status = (comp.get("status") or {}).get("type") or {}
    completed = bool(status.get("completed"))
    competitors = comp.get("competitors") or []
    home_team = away_team = None
    home_score = away_score = None
    for c in competitors:
        team = (c.get("team") or {}).get("abbreviation")
        score = _parse_int(c.get("score"))
        if c.get("homeAway") == "home":
            home_team = team
            home_score = score
        elif c.get("homeAway") == "away":
            away_team = team
            away_score = score
    return SportsEvent(
        event_id=str(event_id),
        league=league.lower(),
        sport=_LEAGUE_PATHS.get(league.lower(), "").split("/", 1)[0] or None,
        home_team=home_team,
        away_team=away_team,
        venue=venue,
        scheduled_start=_parse_iso_dt(raw.get("date") or comp.get("date")),
        status=status.get("name"),
        completed=completed,
        attendance=_parse_int(comp.get("attendance")),
        home_score=home_score,
        away_score=away_score,
        raw_payload_json=json.dumps(raw),
    )
