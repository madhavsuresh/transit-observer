"""Canonical Chicago corridors that drive the synthetic-route corpus.

A *corridor* is one direction of one origin-destination pair on one mode.
We always seed both directions of every OD (inbound + outbound) so the
directional asymmetry is preserved -- DOOR_TO_DOOR.md is explicit that
two corridors per OD is the rule, never one bidirectional record.

The collector cycles through corridors on a fixed cadence rather than
sampling random trips. Each corridor produces one synthetic prediction
per ``cadence_seconds``; that prediction is later graded against the
recorded feed stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import duckdb


@dataclass(frozen=True)
class Corridor:
    corridor_id: str
    mode: str                    # 'L' | 'bus' | 'metra' | 'intercampus'
    line: str                    # API line code (e.g. 'Red'), bus route ('22'), Metra route_id ('UP-N')
    direction: str               # 'inbound' | 'outbound' | 'northbound' | 'southbound' | 'eastbound' | 'westbound'
    origin_label: str
    origin_latitude: float
    origin_longitude: float
    destination_label: str
    destination_latitude: float
    destination_longitude: float
    boarding_int_id: int         # 0 when not applicable (non-L modes)
    boarding_text_id: str | None
    alighting_int_id: int
    alighting_text_id: str | None
    schedule_headway_seconds: float
    cadence_seconds: float = 300.0
    priority: int = 5


# Seed corridors. Two rows per OD: ``-ib`` is the toward-Loop / toward-Evanston
# / canonical-A direction, ``-ob`` is the reverse. Intercampus is a one-way
# loop in each direction, so each direction is its own corridor.
SEED_CORRIDORS: tuple[Corridor, ...] = (
    # --- Metra UP-N: Evanston (Davis) <-> Ogilvie ---
    Corridor(
        corridor_id="metra-upn-evanston-otc-ib",
        mode="metra", line="UP-N", direction="inbound",
        origin_label="Evanston (Davis St.)", origin_latitude=42.0467, origin_longitude=-87.6837,
        destination_label="Chicago OTC", destination_latitude=41.8855, destination_longitude=-87.6406,
        boarding_int_id=0, boarding_text_id="EVANSTON",
        alighting_int_id=0, alighting_text_id="OTC",
        schedule_headway_seconds=1800.0, cadence_seconds=300.0, priority=1,
    ),
    Corridor(
        corridor_id="metra-upn-evanston-otc-ob",
        mode="metra", line="UP-N", direction="outbound",
        origin_label="Chicago OTC", origin_latitude=41.8855, origin_longitude=-87.6406,
        destination_label="Evanston (Davis St.)", destination_latitude=42.0467, destination_longitude=-87.6837,
        boarding_int_id=0, boarding_text_id="OTC",
        alighting_int_id=0, alighting_text_id="EVANSTON",
        schedule_headway_seconds=1800.0, cadence_seconds=300.0, priority=1,
    ),
    Corridor(
        corridor_id="metra-upn-central-otc-ib",
        mode="metra", line="UP-N", direction="inbound",
        origin_label="Central St.", origin_latitude=42.0613, origin_longitude=-87.6929,
        destination_label="Chicago OTC", destination_latitude=41.8855, destination_longitude=-87.6406,
        boarding_int_id=0, boarding_text_id="CENTRALST",
        alighting_int_id=0, alighting_text_id="OTC",
        schedule_headway_seconds=1800.0, cadence_seconds=300.0, priority=2,
    ),
    Corridor(
        corridor_id="metra-upn-central-otc-ob",
        mode="metra", line="UP-N", direction="outbound",
        origin_label="Chicago OTC", origin_latitude=41.8855, origin_longitude=-87.6406,
        destination_label="Central St.", destination_latitude=42.0613, destination_longitude=-87.6929,
        boarding_int_id=0, boarding_text_id="OTC",
        alighting_int_id=0, alighting_text_id="CENTRALST",
        schedule_headway_seconds=1800.0, cadence_seconds=300.0, priority=2,
    ),

    # --- Intercampus loops ---
    Corridor(
        corridor_id="intercampus-central-loyola-sb",
        mode="intercampus", line="intercampus", direction="southbound",
        origin_label="Central/Jackson (IB)", origin_latitude=42.0613, origin_longitude=-87.6943,
        destination_label="Sheridan/Loyola (IB)", destination_latitude=41.9999, destination_longitude=-87.6595,
        boarding_int_id=0, boarding_text_id="b3f50cbe-621f-4664-934a-fe48d4901250",
        alighting_int_id=0, alighting_text_id="e5aa8b6f-44b5-4c4b-becd-1125a1fa4db4",
        schedule_headway_seconds=900.0, cadence_seconds=300.0, priority=3,
    ),
    Corridor(
        corridor_id="intercampus-loyola-central-nb",
        mode="intercampus", line="intercampus", direction="northbound",
        origin_label="Sheridan/Loyola (OB)", origin_latitude=41.9999, origin_longitude=-87.6595,
        destination_label="Central/Jackson (OB)", destination_latitude=42.0613, destination_longitude=-87.6943,
        boarding_int_id=0, boarding_text_id="e647afb1-e56d-4b28-b58a-b581f27b3e90",
        alighting_int_id=0, alighting_text_id="c28a43f2-95c6-442f-9077-adfcdda4a4cf",
        schedule_headway_seconds=900.0, cadence_seconds=300.0, priority=3,
    ),

    # --- CTA Red Line: Belmont <-> Lake (North leg) ---
    Corridor(
        corridor_id="cta-red-belmont-lake-sb",
        mode="L", line="Red", direction="southbound",
        origin_label="Belmont", origin_latitude=41.9398, origin_longitude=-87.6531,
        destination_label="Lake", destination_latitude=41.8848, destination_longitude=-87.6280,
        boarding_int_id=41320, boarding_text_id=None,
        alighting_int_id=41660, alighting_text_id=None,
        schedule_headway_seconds=540.0, cadence_seconds=300.0, priority=2,
    ),
    Corridor(
        corridor_id="cta-red-belmont-lake-nb",
        mode="L", line="Red", direction="northbound",
        origin_label="Lake", origin_latitude=41.8848, origin_longitude=-87.6280,
        destination_label="Belmont", destination_latitude=41.9398, destination_longitude=-87.6531,
        boarding_int_id=41660, boarding_text_id=None,
        alighting_int_id=41320, alighting_text_id=None,
        schedule_headway_seconds=540.0, cadence_seconds=300.0, priority=2,
    ),

    # --- CTA Red Line: 95th/Dan Ryan <-> Lake (South leg) ---
    Corridor(
        corridor_id="cta-red-95th-lake-nb",
        mode="L", line="Red", direction="northbound",
        origin_label="95th/Dan Ryan", origin_latitude=41.7224, origin_longitude=-87.6244,
        destination_label="Lake", destination_latitude=41.8848, destination_longitude=-87.6280,
        boarding_int_id=40450, boarding_text_id=None,
        alighting_int_id=41660, alighting_text_id=None,
        schedule_headway_seconds=540.0, cadence_seconds=300.0, priority=4,
    ),
    Corridor(
        corridor_id="cta-red-95th-lake-sb",
        mode="L", line="Red", direction="southbound",
        origin_label="Lake", origin_latitude=41.8848, origin_longitude=-87.6280,
        destination_label="95th/Dan Ryan", destination_latitude=41.7224, destination_longitude=-87.6244,
        boarding_int_id=41660, boarding_text_id=None,
        alighting_int_id=40450, alighting_text_id=None,
        schedule_headway_seconds=540.0, cadence_seconds=300.0, priority=4,
    ),

    # --- CTA Blue Line: O'Hare <-> Clark/Lake ---
    Corridor(
        corridor_id="cta-blue-ohare-loop-eb",
        mode="L", line="Blue", direction="eastbound",
        origin_label="O'Hare", origin_latitude=41.9777, origin_longitude=-87.9042,
        destination_label="Clark/Lake", destination_latitude=41.8857, destination_longitude=-87.6309,
        boarding_int_id=40890, boarding_text_id=None,
        alighting_int_id=40380, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=3,
    ),
    Corridor(
        corridor_id="cta-blue-ohare-loop-wb",
        mode="L", line="Blue", direction="westbound",
        origin_label="Clark/Lake", origin_latitude=41.8857, origin_longitude=-87.6309,
        destination_label="O'Hare", destination_latitude=41.9777, destination_longitude=-87.9042,
        boarding_int_id=40380, boarding_text_id=None,
        alighting_int_id=40890, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=3,
    ),

    # --- CTA Pink Line: 54th/Cermak <-> Clark/Lake ---
    Corridor(
        corridor_id="cta-pink-cermak-loop-eb",
        mode="L", line="Pink", direction="eastbound",
        origin_label="54th/Cermak", origin_latitude=41.8518, origin_longitude=-87.7567,
        destination_label="Clark/Lake", destination_latitude=41.8857, destination_longitude=-87.6309,
        boarding_int_id=40580, boarding_text_id=None,
        alighting_int_id=40380, alighting_text_id=None,
        schedule_headway_seconds=900.0, cadence_seconds=300.0, priority=4,
    ),
    Corridor(
        corridor_id="cta-pink-cermak-loop-wb",
        mode="L", line="Pink", direction="westbound",
        origin_label="Clark/Lake", origin_latitude=41.8857, origin_longitude=-87.6309,
        destination_label="54th/Cermak", destination_latitude=41.8518, destination_longitude=-87.7567,
        boarding_int_id=40380, boarding_text_id=None,
        alighting_int_id=40580, alighting_text_id=None,
        schedule_headway_seconds=900.0, cadence_seconds=300.0, priority=4,
    ),

    # --- CTA Bus 22 (Clark): Belmont <-> Adams ---
    Corridor(
        corridor_id="cta-bus-22-belmont-adams-sb",
        mode="bus", line="22", direction="southbound",
        origin_label="Clark & Belmont", origin_latitude=41.9401, origin_longitude=-87.6509,
        destination_label="Clark & Adams", destination_latitude=41.8791, destination_longitude=-87.6309,
        boarding_int_id=1828, boarding_text_id=None,
        alighting_int_id=1869, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=5,
    ),
    Corridor(
        corridor_id="cta-bus-22-belmont-adams-nb",
        mode="bus", line="22", direction="northbound",
        origin_label="Dearborn & Adams area", origin_latitude=41.8791, origin_longitude=-87.6296,
        destination_label="Clark & Belmont", destination_latitude=41.9399, destination_longitude=-87.6505,
        boarding_int_id=14767, boarding_text_id=None,
        alighting_int_id=1921, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=5,
    ),
)


def by_id() -> dict[str, Corridor]:
    return {c.corridor_id: c for c in SEED_CORRIDORS}


def by_mode(mode: str) -> tuple[Corridor, ...]:
    return tuple(c for c in SEED_CORRIDORS if c.mode == mode)


def seed_corridors(conn: duckdb.DuckDBPyConnection, *, now: datetime) -> int:
    """Insert any missing corridors into the ``corridors`` table.

    Upserts on ``corridor_id``: re-seeding is safe and refreshes metadata
    on any corridor row whose seed definition changed (e.g. corrected
    coordinates, adjusted cadence).
    """
    rows = [
        (
            c.corridor_id, c.mode, c.line, c.direction,
            c.origin_label, c.origin_latitude, c.origin_longitude,
            c.destination_label, c.destination_latitude, c.destination_longitude,
            c.boarding_int_id, c.boarding_text_id,
            c.alighting_int_id, c.alighting_text_id,
            c.schedule_headway_seconds, c.cadence_seconds, c.priority,
            True, now,
        )
        for c in SEED_CORRIDORS
    ]
    conn.executemany(
        """
        INSERT INTO corridors (
            corridor_id, mode, line, direction,
            origin_label, origin_latitude, origin_longitude,
            destination_label, destination_latitude, destination_longitude,
            boarding_int_id, boarding_text_id,
            alighting_int_id, alighting_text_id,
            schedule_headway_seconds, cadence_seconds, priority,
            is_active, seeded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (corridor_id) DO UPDATE SET
            mode = excluded.mode,
            line = excluded.line,
            direction = excluded.direction,
            origin_label = excluded.origin_label,
            origin_latitude = excluded.origin_latitude,
            origin_longitude = excluded.origin_longitude,
            destination_label = excluded.destination_label,
            destination_latitude = excluded.destination_latitude,
            destination_longitude = excluded.destination_longitude,
            boarding_int_id = excluded.boarding_int_id,
            boarding_text_id = excluded.boarding_text_id,
            alighting_int_id = excluded.alighting_int_id,
            alighting_text_id = excluded.alighting_text_id,
            schedule_headway_seconds = excluded.schedule_headway_seconds,
            cadence_seconds = excluded.cadence_seconds,
            priority = excluded.priority,
            is_active = excluded.is_active
        """,
        rows,
    )
    return len(rows)


def due_corridors(
    conn: duckdb.DuckDBPyConnection,
    *,
    now: datetime,
    enabled_modes: Iterable[str],
) -> list[Corridor]:
    """Return active corridors whose cadence window has elapsed since their
    last prediction.

    A corridor is due if ``last_predicted_at`` is NULL, or if
    ``(now - last_predicted_at) >= cadence_seconds``. Ordered by priority
    (lowest int first) then by how long they've been waiting.
    """
    enabled = tuple(enabled_modes)
    if not enabled:
        return []
    placeholders = ",".join(["?"] * len(enabled))
    rows = conn.execute(
        f"""
        SELECT corridor_id, mode, line, direction,
               origin_label, origin_latitude, origin_longitude,
               destination_label, destination_latitude, destination_longitude,
               boarding_int_id, boarding_text_id,
               alighting_int_id, alighting_text_id,
               schedule_headway_seconds, cadence_seconds, priority,
               last_predicted_at
          FROM corridors
         WHERE is_active = TRUE
           AND mode IN ({placeholders})
           AND (
                last_predicted_at IS NULL
                OR EPOCH(? - last_predicted_at) >= cadence_seconds
           )
         ORDER BY priority ASC,
                  COALESCE(last_predicted_at, TIMESTAMPTZ '1970-01-01') ASC
        """,
        list(enabled) + [now],
    ).fetchall()
    out: list[Corridor] = []
    for r in rows:
        out.append(
            Corridor(
                corridor_id=r[0], mode=r[1], line=r[2], direction=r[3],
                origin_label=r[4], origin_latitude=r[5], origin_longitude=r[6],
                destination_label=r[7], destination_latitude=r[8], destination_longitude=r[9],
                boarding_int_id=r[10], boarding_text_id=r[11],
                alighting_int_id=r[12], alighting_text_id=r[13],
                schedule_headway_seconds=r[14], cadence_seconds=r[15], priority=r[16],
            )
        )
    return out


def mark_predicted(
    conn: duckdb.DuckDBPyConnection,
    *,
    corridor_id: str,
    at: datetime,
) -> None:
    conn.execute(
        "UPDATE corridors SET last_predicted_at = ? WHERE corridor_id = ?",
        [at, corridor_id],
    )
