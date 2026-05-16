"""CTA L station catalog. Loaded once at startup from a bundled JSON snapshot.

The bundled `CTAStations.json` is a copy of the Cozy Fox resource of the same
name. Refresh it from `../transit/Packages/TransitCore/Sources/TransitModels/Resources/CTAStations.json`
when the source updates (rare — the CTA L roster changes very seldom).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LStation:
    map_id: int
    name: str
    latitude: float
    longitude: float
    served_lines: tuple[str, ...]


_CATALOG_PATH = Path(__file__).parent / "CTAStations.json"


def load_catalog(path: Path = _CATALOG_PATH) -> list[LStation]:
    raw = json.loads(path.read_text())
    return [
        LStation(
            map_id=int(entry["id"]),
            name=str(entry["name"]),
            latitude=float(entry["latitude"]),
            longitude=float(entry["longitude"]),
            served_lines=tuple(entry.get("servedLines", [])),
        )
        for entry in raw
    ]


def by_map_id(catalog: list[LStation]) -> dict[int, LStation]:
    return {station.map_id: station for station in catalog}


def by_line(catalog: list[LStation]) -> dict[str, list[LStation]]:
    out: dict[str, list[LStation]] = {}
    for station in catalog:
        for line in station.served_lines:
            out.setdefault(line, []).append(station)
    return out


def by_name(catalog: list[LStation]) -> dict[str, LStation]:
    """Case-insensitive name lookup. Multiple stations share names (e.g.
    'Western' on 5 different lines); collisions keep the first."""
    out: dict[str, LStation] = {}
    for station in catalog:
        key = station.name.lower()
        out.setdefault(key, station)
    return out
