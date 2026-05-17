"""Cross-stream identifier bridges for the CTA pipelines.

The proprietary CTA feeds and the GTFS-RT feeds use disjoint ID
spaces. This module derives the joins from observed data:

- :func:`refresh_run_vehicle_links` walks recent
  ``train_v2_position_observation`` rows (Train Tracker run_number +
  lat/lon at time T) and matches them against
  ``train_v2_gtfsrt_vehicle_position`` (GTFS-RT vehicle_id + lat/lon at
  time T). When the haversine separation is small and the temporal
  separation is small, the two IDs refer to the same physical train.

- :func:`refresh_station_id_map` precomputes the
  ``ttarrivals.aspx.staId`` ↔ GTFS-static ``parent_station`` + child
  ``stop_id`` mapping from the most recent ``gtfs_static_*`` snapshot.
  This is a one-time precompute per GTFS-static snapshot.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import duckdb
import structlog


log = structlog.get_logger(__name__)


_EARTH_RADIUS_M = 6_371_000.0
DEFAULT_MAX_HAVERSINE_M = 250.0    # CTA trains are ~150m apart at closest; >250m is a likely mismatch
DEFAULT_MAX_DELTA_S = 90.0         # only pair observations within this temporal window
DEFAULT_WINDOW_HOURS = 24


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def refresh_run_vehicle_links(
    conn: duckdb.DuckDBPyConnection,
    *,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    max_haversine_m: float = DEFAULT_MAX_HAVERSINE_M,
    max_delta_s: float = DEFAULT_MAX_DELTA_S,
    cutoff_ms: Optional[int] = None,
) -> int:
    """Derive ``train_v2_run_vehicle_link`` rows from co-located
    Train Tracker positions and GTFS-RT vehicle positions.

    Returns the number of rows upserted.

    ``cutoff_ms`` overrides the default ``now() - window_hours`` floor;
    pass an explicit value to scan historical periods.
    """
    if cutoff_ms is None:
        cutoff_ms = int(time.time() * 1000) - window_hours * 3_600_000

    # Pull every candidate pair: same line, timestamps within max_delta_s,
    # and both lat/lon non-null. We compute haversine in Python since
    # DuckDB doesn't ship with one out of the box.
    pairs = conn.execute(
        """
        SELECT
            tt.line,
            tt.run_number,
            gv.vehicle_id,
            tt.lat AS tt_lat, tt.lon AS tt_lon, tt.local_response_end_ms AS tt_ms,
            gv.lat AS gv_lat, gv.lon AS gv_lon, gv.local_response_end_ms AS gv_ms
          FROM train_v2_position_observation tt
          JOIN train_v2_gtfsrt_vehicle_position gv
            ON gv.route_id = tt.line
           AND ABS(gv.local_response_end_ms - tt.local_response_end_ms) <= ?
         WHERE tt.lat IS NOT NULL AND tt.lon IS NOT NULL
           AND gv.lat IS NOT NULL AND gv.lon IS NOT NULL
           AND tt.run_number IS NOT NULL
           AND gv.vehicle_id IS NOT NULL
           AND tt.local_response_end_ms >= ?
           AND gv.local_response_end_ms >= ?
        """,
        [int(max_delta_s * 1000), cutoff_ms, cutoff_ms],
    ).fetchall()

    # Aggregate by (line, run_number, vehicle_id) keeping the spatial
    # min/max/avg + temporal span. Filter to pairs that consistently
    # have small haversine separation — those are likely the same train.
    from collections import defaultdict

    agg: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "first_seen_ms": math.inf,
            "last_seen_ms": -math.inf,
            "haversine_sum": 0.0,
            "haversine_max": 0.0,
            "n": 0,
        }
    )
    for line, run_number, vehicle_id, tt_lat, tt_lon, tt_ms, gv_lat, gv_lon, gv_ms in pairs:
        d_m = _haversine_m(tt_lat, tt_lon, gv_lat, gv_lon)
        if d_m > max_haversine_m:
            continue
        key = (str(line), str(run_number), str(vehicle_id))
        bucket = agg[key]
        ts = max(int(tt_ms or 0), int(gv_ms or 0))
        bucket["first_seen_ms"] = min(bucket["first_seen_ms"], ts)
        bucket["last_seen_ms"] = max(bucket["last_seen_ms"], ts)
        bucket["haversine_sum"] += d_m
        bucket["haversine_max"] = max(bucket["haversine_max"], d_m)
        bucket["n"] += 1

    n_written = 0
    for (line, run_number, vehicle_id), bucket in agg.items():
        if bucket["n"] < 2:
            # Require at least 2 co-occurrences to be confident.
            continue
        mean_m = bucket["haversine_sum"] / bucket["n"]
        conn.execute(
            """
            INSERT INTO train_v2_run_vehicle_link(
                line, run_number, gtfsrt_vehicle_id,
                first_seen_ms, last_seen_ms, n_observations,
                mean_haversine_m, max_haversine_m
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (line, run_number, gtfsrt_vehicle_id) DO UPDATE SET
                first_seen_ms = LEAST(train_v2_run_vehicle_link.first_seen_ms, excluded.first_seen_ms),
                last_seen_ms = GREATEST(train_v2_run_vehicle_link.last_seen_ms, excluded.last_seen_ms),
                n_observations = train_v2_run_vehicle_link.n_observations + excluded.n_observations,
                mean_haversine_m = (
                    train_v2_run_vehicle_link.mean_haversine_m * train_v2_run_vehicle_link.n_observations
                    + excluded.mean_haversine_m * excluded.n_observations
                ) / (train_v2_run_vehicle_link.n_observations + excluded.n_observations),
                max_haversine_m = GREATEST(train_v2_run_vehicle_link.max_haversine_m, excluded.max_haversine_m)
            """,
            [
                line, run_number, vehicle_id,
                int(bucket["first_seen_ms"]), int(bucket["last_seen_ms"]),
                bucket["n"], mean_m, bucket["haversine_max"],
            ],
        )
        n_written += 1
    log.info(
        "cta_id_bridges.run_vehicle_links",
        candidate_pairs=len(pairs),
        rows_upserted=n_written,
    )
    return n_written


def refresh_station_id_map(
    conn: duckdb.DuckDBPyConnection,
    *,
    agency: str = "cta",
    snapshot_id: Optional[int] = None,
) -> int:
    """Build ``cta_station_id_map`` from the most recent
    ``gtfs_static_*`` snapshot for the given agency.

    CTA's ttarrivals ``staId`` maps directly onto the GTFS
    ``parent_station`` stop_id (they're typically the same number).
    Each ``parent_station`` has 1-4 child stops (one per platform).
    The map enables exact joins between proprietary Train Tracker IDs
    and GTFS-RT / GTFS-static stop_ids.
    """
    if snapshot_id is None:
        row = conn.execute(
            """
            SELECT snapshot_id FROM gtfs_static_snapshot
             WHERE agency = ?
             ORDER BY COALESCE(extracted_at_ms, fetched_at_ms) DESC
             LIMIT 1
            """,
            [agency],
        ).fetchone()
        if row is None:
            log.info("cta_id_bridges.station_map.no_snapshot", agency=agency)
            return 0
        snapshot_id = int(row[0])

    # Clear previous mappings for this snapshot, then re-derive.
    conn.execute(
        "DELETE FROM cta_station_id_map WHERE snapshot_id = ?",
        [snapshot_id],
    )

    # Project: every stop whose parent_station is non-null becomes a
    # (parent → child) row keyed on parent's stop_id (== staId).
    rows = conn.execute(
        """
        SELECT
            parent.stop_id   AS map_id,
            parent.stop_id   AS parent_station,
            child.stop_id    AS child_stop_id,
            child.stop_name  AS child_stop_name,
            parent.stop_lat  AS lat,
            parent.stop_lon  AS lon
          FROM gtfs_static_stops parent
          JOIN gtfs_static_stops child
            ON child.parent_station = parent.stop_id
           AND child.snapshot_id = parent.snapshot_id
         WHERE parent.snapshot_id = ?
           AND parent.location_type = 1   -- 1 = station (per GTFS spec)
        """,
        [snapshot_id],
    ).fetchall()

    # Also try to attach a "line" guess by joining each child stop to
    # the routes its trips serve (most-frequent route_short_name).
    line_lookup_rows = conn.execute(
        """
        WITH child_routes AS (
            SELECT st.stop_id AS child_stop_id,
                   r.route_short_name AS route_short_name,
                   COUNT(*) AS n
              FROM gtfs_static_stop_times st
              JOIN gtfs_static_trips t ON t.trip_id = st.trip_id AND t.snapshot_id = st.snapshot_id
              JOIN gtfs_static_routes r ON r.route_id = t.route_id AND r.snapshot_id = t.snapshot_id
             WHERE st.snapshot_id = ?
             GROUP BY st.stop_id, r.route_short_name
        )
        SELECT child_stop_id, route_short_name
          FROM child_routes
         QUALIFY ROW_NUMBER() OVER (PARTITION BY child_stop_id ORDER BY n DESC) = 1
        """,
        [snapshot_id],
    ).fetchall()
    line_by_child = {str(c): str(r) for c, r in line_lookup_rows if c and r}

    n = 0
    for map_id, parent_station, child_stop_id, child_stop_name, lat, lon in rows:
        line = line_by_child.get(str(child_stop_id))
        conn.execute(
            """
            INSERT INTO cta_station_id_map(
                map_id, parent_station, child_stop_id, child_stop_name,
                line, lat, lon, snapshot_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [str(map_id), str(parent_station), str(child_stop_id),
             child_stop_name, line, lat, lon, snapshot_id],
        )
        n += 1
    log.info(
        "cta_id_bridges.station_id_map",
        snapshot_id=snapshot_id, rows=n, mapped_lines=len(line_by_child),
    )
    return n
