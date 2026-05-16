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


# Metra ---------------------------------------------------------------


@dataclass(frozen=True)
class MetraStation:
    station_id: str
    name: str
    latitude: float
    longitude: float
    served_routes: tuple[str, ...]


_METRA_PATH = Path(__file__).parent / "MetraCatalog.json"


def load_metra_catalog(path: Path = _METRA_PATH) -> list[MetraStation]:
    raw = json.loads(path.read_text())
    out: list[MetraStation] = []
    for entry in raw.get("stations", []):
        if len(entry) < 7:
            continue
        out.append(
            MetraStation(
                station_id=str(entry[0]),
                name=str(entry[1]),
                latitude=float(entry[2]),
                longitude=float(entry[3]),
                served_routes=tuple(entry[6] or ()),
            )
        )
    return out


def metra_by_id(catalog: list[MetraStation]) -> dict[str, MetraStation]:
    return {s.station_id: s for s in catalog}


def metra_by_route(catalog: list[MetraStation]) -> dict[str, list[MetraStation]]:
    out: dict[str, list[MetraStation]] = {}
    for s in catalog:
        for r in s.served_routes:
            out.setdefault(r, []).append(s)
    return out


# Intercampus ---------------------------------------------------------


@dataclass(frozen=True)
class IntercampusStop:
    stop_id: str
    name: str
    latitude: float
    longitude: float
    served_directions: tuple[str, ...]


_INTERCAMPUS_PATH = Path(__file__).parent / "IntercampusCatalog.json"


def load_intercampus_catalog(path: Path = _INTERCAMPUS_PATH) -> list[IntercampusStop]:
    raw = json.loads(path.read_text())
    out: list[IntercampusStop] = []
    for entry in raw.get("stops", []):
        if len(entry) < 5:
            continue
        out.append(
            IntercampusStop(
                stop_id=str(entry[0]),
                name=str(entry[1]),
                latitude=float(entry[2]),
                longitude=float(entry[3]),
                served_directions=tuple(entry[4] or ()),
            )
        )
    return out


def intercampus_by_id(catalog: list[IntercampusStop]) -> dict[str, IntercampusStop]:
    return {s.stop_id: s for s in catalog}


def intercampus_by_direction(catalog: list[IntercampusStop]) -> dict[str, list[IntercampusStop]]:
    out: dict[str, list[IntercampusStop]] = {}
    for s in catalog:
        for d in s.served_directions:
            out.setdefault(d, []).append(s)
    return out


# CTA Bus -------------------------------------------------------------


@dataclass(frozen=True)
class BusStop:
    stop_id: int
    route: str
    name: str
    latitude: float
    longitude: float
    direction_label: str


_BUS_PATH = Path(__file__).parent / "CTABusStops.json"


def load_bus_catalog(path: Path = _BUS_PATH) -> list[BusStop]:
    raw = json.loads(path.read_text())
    return [
        BusStop(
            stop_id=int(entry["id"]),
            route=str(entry["route"]),
            name=str(entry["name"]),
            latitude=float(entry["latitude"]),
            longitude=float(entry["longitude"]),
            direction_label=str(entry.get("directionLabel", "")),
        )
        for entry in raw
    ]


def bus_by_id(catalog: list[BusStop]) -> dict[tuple[str, int], BusStop]:
    """Bus stops are (route, stop_id) pairs; same stop appears on multiple
    routes with different rows."""
    return {(s.route, s.stop_id): s for s in catalog}


def bus_by_route(catalog: list[BusStop]) -> dict[str, list[BusStop]]:
    out: dict[str, list[BusStop]] = {}
    for s in catalog:
        out.setdefault(s.route, []).append(s)
    return out
