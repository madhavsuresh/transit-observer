"""Promote frequently-queried OD pairs from query_log into seeded corridors.

Goal: an OD pair that the API has answered >= N times in the rolling
window becomes a permanent corridor, so it starts producing graded
forecasts alongside the hand-seeded set. This is the auto-tuning side of
the corpus.

We look up the station's coordinates and labels from the relevant
catalog so the promoted corridor row carries the same metadata shape as
a hand-seeded one. ``source='auto_upgraded'`` distinguishes them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable

import duckdb

from .catalog import (
    bus_by_id,
    intercampus_by_id,
    load_bus_catalog,
    load_catalog,
    load_intercampus_catalog,
    load_metra_catalog,
    metra_by_id,
)
from .query_log import find_popular_ods


log = logging.getLogger(__name__)


DEFAULT_PROMOTION_MIN_COUNT = 50
DEFAULT_PROMOTION_WINDOW = timedelta(days=7)

# Per-mode default schedule headways (seconds). Cheap fallback when we
# don't have a better estimate.
_DEFAULT_HEADWAY: dict[str, float] = {
    "L": 600.0,
    "bus": 600.0,
    "metra": 1800.0,
    "intercampus": 900.0,
}


def promote_popular(
    conn: duckdb.DuckDBPyConnection,
    *,
    now: datetime,
    min_count: int = DEFAULT_PROMOTION_MIN_COUNT,
    window: timedelta = DEFAULT_PROMOTION_WINDOW,
) -> list[str]:
    """Promote queried-often ODs into corridors. Returns new corridor_ids."""
    popular = find_popular_ods(conn, now=now, window=window, min_count=min_count)
    if not popular:
        return []

    l_by_map = {s.map_id: s for s in load_catalog()}
    bus_lookup = bus_by_id(load_bus_catalog())
    metra_lookup = metra_by_id(load_metra_catalog())
    ic_lookup = intercampus_by_id(load_intercampus_catalog())

    new_ids: list[str] = []
    for row in popular:
        corridor = _build_corridor_row(
            row, now=now,
            l_by_map=l_by_map, bus_lookup=bus_lookup,
            metra_lookup=metra_lookup, ic_lookup=ic_lookup,
        )
        if corridor is None:
            continue
        try:
            conn.execute(
                """
                INSERT INTO corridors (
                    corridor_id, mode, line, direction,
                    origin_label, origin_latitude, origin_longitude,
                    destination_label, destination_latitude, destination_longitude,
                    boarding_int_id, boarding_text_id,
                    alighting_int_id, alighting_text_id,
                    schedule_headway_seconds, cadence_seconds, priority,
                    is_active, seeded_at, source, promoted_from_query_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'auto_upgraded', ?)
                ON CONFLICT (corridor_id) DO NOTHING
                """,
                [
                    corridor["corridor_id"], corridor["mode"], corridor["line"], corridor["direction"],
                    corridor["origin_label"], corridor["origin_latitude"], corridor["origin_longitude"],
                    corridor["destination_label"], corridor["destination_latitude"], corridor["destination_longitude"],
                    corridor["boarding_int_id"], corridor["boarding_text_id"],
                    corridor["alighting_int_id"], corridor["alighting_text_id"],
                    corridor["schedule_headway_seconds"], corridor["cadence_seconds"], corridor["priority"],
                    True, now, row["count"],
                ],
            )
            new_ids.append(corridor["corridor_id"])
            log.info(
                "corridor.auto_upgraded",
                corridor_id=corridor["corridor_id"], count=row["count"],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("corridor.auto_upgrade_failed", err=str(exc), row=row)
    return new_ids


def _build_corridor_row(
    row: dict,
    *,
    now: datetime,
    l_by_map: dict,
    bus_lookup: dict,
    metra_lookup: dict,
    ic_lookup: dict,
) -> dict | None:
    """Look up station coordinates and labels for a popular OD pair."""
    mode = row["mode"]
    line = row["line"]
    direction = row.get("direction_code") or _infer_direction(row)

    if mode == "L":
        boarding = l_by_map.get(row["boarding_int_id"])
        alighting = l_by_map.get(row["alighting_int_id"])
        if boarding is None or alighting is None:
            return None
        return _row(
            corridor_id=f"auto-{mode.lower()}-{line.lower()}-{boarding.map_id}-{alighting.map_id}",
            mode=mode, line=line, direction=direction,
            origin_label=boarding.name, origin_lat=boarding.latitude, origin_lon=boarding.longitude,
            dest_label=alighting.name, dest_lat=alighting.latitude, dest_lon=alighting.longitude,
            boarding_int_id=boarding.map_id, boarding_text_id=None,
            alighting_int_id=alighting.map_id, alighting_text_id=None,
        )

    if mode == "bus":
        boarding = bus_lookup.get((line, row["boarding_int_id"]))
        alighting = bus_lookup.get((line, row["alighting_int_id"]))
        if boarding is None or alighting is None:
            return None
        return _row(
            corridor_id=f"auto-bus-{line}-{boarding.stop_id}-{alighting.stop_id}",
            mode=mode, line=line, direction=direction,
            origin_label=boarding.name, origin_lat=boarding.latitude, origin_lon=boarding.longitude,
            dest_label=alighting.name, dest_lat=alighting.latitude, dest_lon=alighting.longitude,
            boarding_int_id=boarding.stop_id, boarding_text_id=None,
            alighting_int_id=alighting.stop_id, alighting_text_id=None,
        )

    if mode == "metra":
        b_id = row.get("boarding_text_id")
        a_id = row.get("alighting_text_id")
        if not b_id or not a_id:
            return None
        boarding = metra_lookup.get(b_id)
        alighting = metra_lookup.get(a_id)
        if boarding is None or alighting is None:
            return None
        return _row(
            corridor_id=f"auto-metra-{line.lower()}-{boarding.station_id}-{alighting.station_id}".lower(),
            mode=mode, line=line, direction=direction,
            origin_label=boarding.name, origin_lat=boarding.latitude, origin_lon=boarding.longitude,
            dest_label=alighting.name, dest_lat=alighting.latitude, dest_lon=alighting.longitude,
            boarding_int_id=0, boarding_text_id=boarding.station_id,
            alighting_int_id=0, alighting_text_id=alighting.station_id,
        )

    if mode == "intercampus":
        b_id = row.get("boarding_text_id")
        a_id = row.get("alighting_text_id")
        if not b_id or not a_id:
            return None
        boarding = ic_lookup.get(b_id)
        alighting = ic_lookup.get(a_id)
        if boarding is None or alighting is None:
            return None
        return _row(
            corridor_id=f"auto-ic-{boarding.stop_id[:8]}-{alighting.stop_id[:8]}",
            mode=mode, line="intercampus", direction=direction,
            origin_label=boarding.name, origin_lat=boarding.latitude, origin_lon=boarding.longitude,
            dest_label=alighting.name, dest_lat=alighting.latitude, dest_lon=alighting.longitude,
            boarding_int_id=0, boarding_text_id=boarding.stop_id,
            alighting_int_id=0, alighting_text_id=alighting.stop_id,
        )

    return None


def _row(
    *,
    corridor_id: str, mode: str, line: str, direction: str,
    origin_label: str, origin_lat: float, origin_lon: float,
    dest_label: str, dest_lat: float, dest_lon: float,
    boarding_int_id: int, boarding_text_id: str | None,
    alighting_int_id: int, alighting_text_id: str | None,
) -> dict:
    return {
        "corridor_id": corridor_id,
        "mode": mode, "line": line, "direction": direction,
        "origin_label": origin_label,
        "origin_latitude": origin_lat, "origin_longitude": origin_lon,
        "destination_label": dest_label,
        "destination_latitude": dest_lat, "destination_longitude": dest_lon,
        "boarding_int_id": boarding_int_id, "boarding_text_id": boarding_text_id,
        "alighting_int_id": alighting_int_id, "alighting_text_id": alighting_text_id,
        "schedule_headway_seconds": _DEFAULT_HEADWAY.get(mode, 600.0),
        "cadence_seconds": 300.0,
        "priority": 6,  # auto-upgraded < hand-seeded
    }


def _infer_direction(row: dict) -> str:
    """When the query log lacks a direction_code, fall back to lat/lon."""
    return "unknown"
