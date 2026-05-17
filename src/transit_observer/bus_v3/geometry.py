"""Pattern geometry helpers for the v3 bus pipeline.

Ported from the validator's ``geometry.py``. Pattern points are
discrete (lat, lon, pdist_ft) samples along a bus route; map-matching
projects an observed vehicle GPS onto the polyline and reports a
cross-track residual + along-track ``pdist``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import duckdb

EARTH_RADIUS_M = 6_371_000.0
FT_PER_M = 3.280839895
M_PER_FT = 1.0 / FT_PER_M


@dataclass
class MapMatch:
    projected_pdist_ft: Optional[float]
    cross_track_m: Optional[float]
    segment_index: Optional[int]
    segment_fraction: Optional[float]
    segment_bearing_deg: Optional[float]
    heading_error_deg: Optional[float]
    quality: str

    def as_dict(self) -> dict:
        return {
            "projected_pdist_ft": self.projected_pdist_ft,
            "cross_track_m": self.cross_track_m,
            "segment_index": self.segment_index,
            "segment_fraction": self.segment_fraction,
            "segment_bearing_deg": self.segment_bearing_deg,
            "heading_error_deg": self.heading_error_deg,
            "quality": self.quality,
        }


@dataclass
class PatternPoint:
    pid: int
    seq: int
    typ: Optional[str]
    stpid: Optional[str]
    stpnm: Optional[str]
    lat: float
    lon: float
    pdist_ft: float


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def heading_error(hdg: Optional[float], bearing: Optional[float]) -> Optional[float]:
    if hdg is None or bearing is None or not math.isfinite(hdg) or not math.isfinite(bearing):
        return None
    return abs((hdg - bearing + 180.0) % 360.0 - 180.0)


def _xy_m(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    x = math.radians(lon - ref_lon) * EARTH_RADIUS_M * math.cos(math.radians(ref_lat))
    y = math.radians(lat - ref_lat) * EARTH_RADIUS_M
    return x, y


def _project_point_to_segment(
    lat: float,
    lon: float,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> tuple[float, float]:
    ref_lat = (lat1 + lat2 + lat) / 3.0
    ref_lon = (lon1 + lon2 + lon) / 3.0
    px, py = _xy_m(lat, lon, ref_lat, ref_lon)
    x1, y1 = _xy_m(lat1, lon1, ref_lat, ref_lon)
    x2, y2 = _xy_m(lat2, lon2, ref_lat, ref_lon)
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    denom = vx * vx + vy * vy
    if denom <= 1e-9:
        t = 0.0
        dx, dy = px - x1, py - y1
    else:
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
        qx, qy = x1 + t * vx, y1 + t * vy
        dx, dy = px - qx, py - qy
    return t, math.hypot(dx, dy)


def map_match_to_pattern(
    points: list[PatternPoint],
    lat: Optional[float],
    lon: Optional[float],
    hdg: Optional[float] = None,
) -> MapMatch:
    if len(points) < 2 or lat is None or lon is None:
        return MapMatch(None, None, None, None, None, None, "UNUSABLE")
    best: Optional[tuple[float, float, float, int, float, float, Optional[float]]] = None
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        if a.lat is None or a.lon is None or b.lat is None or b.lon is None:
            continue
        frac, dist_m = _project_point_to_segment(lat, lon, a.lat, a.lon, b.lat, b.lon)
        pdist = a.pdist_ft + frac * (b.pdist_ft - a.pdist_ft)
        brg = bearing_deg(a.lat, a.lon, b.lat, b.lon)
        herr = heading_error(hdg, brg)
        score = dist_m + (0.4 * herr if herr is not None else 0.0)
        if best is None or score < best[0]:
            best = (score, pdist, dist_m, i, frac, brg, herr)
    if best is None:
        return MapMatch(None, None, None, None, None, None, "UNUSABLE")
    _, pdist, dist_m, idx, frac, brg, herr = best
    if dist_m <= 50 and (herr is None or herr <= 90):
        q = "HIGH"
    elif dist_m <= 100 and (herr is None or herr <= 120):
        q = "MEDIUM"
    elif dist_m <= 200:
        q = "LOW"
    else:
        q = "UNUSABLE"
    return MapMatch(pdist, dist_m, idx, frac, brg, herr, q)


def load_pattern_points(
    conn: duckdb.DuckDBPyConnection,
    pid: int,
    include_detour_original: bool = False,
) -> list[PatternPoint]:
    rows = conn.execute(
        """
        SELECT pid, seq, typ, stpid, stpnm, lat, lon, pdist_ft
        FROM bus_v3_pattern_point
        WHERE pid = ? AND is_detour_original_point = ?
              AND lat IS NOT NULL AND lon IS NOT NULL AND pdist_ft IS NOT NULL
        ORDER BY seq
        """,
        [pid, 1 if include_detour_original else 0],
    ).fetchall()
    return [
        PatternPoint(
            pid=int(r[0]),
            seq=int(r[1]),
            typ=r[2],
            stpid=r[3],
            stpnm=r[4],
            lat=float(r[5]),
            lon=float(r[6]),
            pdist_ft=float(r[7]),
        )
        for r in rows
    ]


def stop_pdist_for_pid(
    conn: duckdb.DuckDBPyConnection,
    stpid: str,
    pid: int,
) -> Optional[float]:
    row = conn.execute(
        """
        SELECT pdist_ft FROM bus_v3_pattern_point
        WHERE stpid = ? AND pid = ?
          AND is_detour_original_point = 0
          AND pdist_ft IS NOT NULL
        ORDER BY seq LIMIT 1
        """,
        [str(stpid), int(pid)],
    ).fetchone()
    return None if row is None else float(row[0])


def choose_pattern_for_stop(
    conn: duckdb.DuckDBPyConnection,
    stpid: str,
    rt: Optional[str] = None,
    rtdir: Optional[str] = None,
    preferred_pid: Optional[int] = None,
) -> tuple[Optional[int], Optional[float]]:
    if preferred_pid is not None:
        pd = stop_pdist_for_pid(conn, stpid, preferred_pid)
        if pd is not None:
            return int(preferred_pid), pd
    params: list = [str(stpid)]
    where = ["pp.stpid = ?", "pp.is_detour_original_point = 0"]
    if rt:
        where.append("p.rt = ?")
        params.append(rt)
    if rtdir:
        where.append("p.rtdir = ?")
        params.append(rtdir)
    row = conn.execute(
        f"""
        SELECT pp.pid, pp.pdist_ft
        FROM bus_v3_pattern_point pp
        JOIN bus_v3_pattern p ON p.pid = pp.pid
        WHERE {' AND '.join(where)} AND pp.pdist_ft IS NOT NULL
        ORDER BY CASE WHEN p.dtrid IS NULL THEN 0 ELSE 1 END, pp.pid
        LIMIT 1
        """,
        params,
    ).fetchone()
    if row is None:
        return None, None
    return int(row[0]), float(row[1])


def distance_to_stop_m(
    conn: duckdb.DuckDBPyConnection,
    stpid: str,
    lat: Optional[float],
    lon: Optional[float],
) -> Optional[float]:
    if lat is None or lon is None:
        return None
    row = conn.execute(
        "SELECT lat, lon FROM bus_v3_stop WHERE stpid = ?",
        [str(stpid)],
    ).fetchone()
    if row is None or row[0] is None or row[1] is None:
        row = conn.execute(
            """
            SELECT lat, lon FROM bus_v3_pattern_point
            WHERE stpid = ? AND lat IS NOT NULL AND lon IS NOT NULL
            LIMIT 1
            """,
            [str(stpid)],
        ).fetchone()
    if row is None or row[0] is None or row[1] is None:
        return None
    return haversine_m(float(lat), float(lon), float(row[0]), float(row[1]))


def path_remaining_ft(
    vehicle_pdist_ft: Optional[float],
    stop_pdist_ft: Optional[float],
    pattern_length_ft: Optional[float] = None,
) -> Optional[float]:
    if vehicle_pdist_ft is None or stop_pdist_ft is None:
        return None
    remaining = stop_pdist_ft - vehicle_pdist_ft
    if remaining < -100 and pattern_length_ft and pattern_length_ft > stop_pdist_ft:
        return remaining
    return remaining
