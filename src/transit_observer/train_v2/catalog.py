"""Resolve which stations to poll for the v2 train pipeline.

We start from the same L station catalog the legacy collector uses
(``CTAStations.json`` via ``catalog.load_catalog``). The v2 cycle polls
ttarrivals for each monitored ``map_id`` and ttpositions for each
configured line; ttfollow runs once per ``run_number`` seen in the
ttarrivals response.

For now "monitored" = "every L station in the catalog" — the L surface
is small enough (~145 stations) that we don't need a subset, and the
round-robin batching honors the 100-req-per-5-min budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..catalog import LStation, load_catalog


@dataclass(frozen=True)
class TrainV2Target:
    map_id: int
    station_name: str


def all_stations() -> list[TrainV2Target]:
    return [
        TrainV2Target(map_id=int(s.map_id), station_name=s.name)
        for s in load_catalog()
    ]


def stations_for_lines(lines: Iterable[str]) -> list[TrainV2Target]:
    """Filter the catalog to stations served by any of ``lines``.

    Lines are matched against ``LStation.lines`` membership; case-
    insensitive on API code (``'red'`` matches ``'Red'`` etc.).
    """
    wanted = {l.lower() for l in lines}
    out: list[TrainV2Target] = []
    for s in load_catalog():
        station_lines = {ln.lower() for ln in (s.served_lines or ())}
        if station_lines & wanted:
            out.append(TrainV2Target(map_id=int(s.map_id), station_name=s.name))
    return out
