"""Resolution of monitored (route, stpid, direction) tuples for the v3
bus pipeline.

For now we mirror the v2 ``monitored_bus_stops`` set in settings, but
re-keyed so direction is explicit. The v2 tuple is ``(route, stop_id)``;
direction comes from the per-stop ``BusStop.direction_label`` loaded by
``catalog.bus_by_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..catalog import bus_by_id, load_bus_catalog


@dataclass(frozen=True)
class BusV3Target:
    rt: str
    stpid: str
    direction: str | None  # human-friendly direction label, e.g. "Westbound"


def targets_from_monitored_stops(
    monitored: Iterable[tuple[str, int]],
) -> list[BusV3Target]:
    """Resolve the v2 ``(route, stop_id)`` tuples into v3 targets with
    a direction label attached when the catalog knows it."""
    lookup = bus_by_id(load_bus_catalog())
    out: list[BusV3Target] = []
    for route, stop_id in monitored:
        meta = lookup.get((str(route), int(stop_id)))
        out.append(
            BusV3Target(
                rt=str(route),
                stpid=str(stop_id),
                direction=(meta.direction_label if meta is not None else None),
            )
        )
    return out


def unique_routes(targets: Iterable[BusV3Target]) -> list[str]:
    return sorted({t.rt for t in targets})


def unique_stops(targets: Iterable[BusV3Target]) -> list[str]:
    return sorted({t.stpid for t in targets})


def unique_directions(targets: Iterable[BusV3Target]) -> list[str]:
    return sorted({t.direction for t in targets if t.direction})
