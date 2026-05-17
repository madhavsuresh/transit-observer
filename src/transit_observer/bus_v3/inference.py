"""High-confidence arrival inference from pdist crossings.

Ported from the validator's ``inference.py`` to DuckDB. A
``bus_v3_arrival_event`` row is marked ``high_confidence=TRUE`` only
when all of the following hold:

- Same vehicle (``vid``) appears in two consecutive observations on the
  same pattern (``pid``) bracketing the stop's pattern-point ``pdist_ft``.
- Both observations are fresh (vehicle_age_s ≤ 90 s).
- The gap between observations is short (≤ 120 s).
- Pdist speed is physically plausible (0.2 – 110 ft/s).
- Both GPS positions map-match to the same pattern (cross-track ≤ 125 m,
  pdist↔map-match ≤ 1500 ft).
- No severe ``dyn`` / ``flagstop`` / active-detour evidence contradicts
  the pickup.

Censored / disrupted candidates land as ``high_confidence=FALSE`` with a
descriptive label and reason codes; they're recorded for diagnostics but
never feed accuracy metrics.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import duckdb

from .geometry import (
    PatternPoint,
    distance_to_stop_m,
    load_pattern_points,
    map_match_to_pattern,
    stop_pdist_for_pid,
)
from .models import (
    DYN_SEVERE_ABSTAIN,
    FLAGSTOP_ONLY_DISCHARGE,
    ArrivalLabel,
    ReasonCode,
)
from .util import clamp, json_dumps, now_ms, safe_float, safe_int


GROUND_TRUTH_MAX_GAP_S = 120.0
GROUND_TRUTH_MAX_VEHICLE_AGE_S = 90.0
GROUND_TRUTH_MAX_CROSSTRACK_M = 125.0
GROUND_TRUTH_MAX_PDIST_MAPMATCH_DIFF_FT = 1_500.0
GROUND_TRUTH_MIN_SPEED_FTPS = 0.2
GROUND_TRUTH_MAX_SPEED_FTPS = 110.0  # ~75 mph; generous for GPS/pdist quantization


def _rows_to_dicts(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    params: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Run SQL and return rows as ``dict[column_name -> value]``.

    DuckDB doesn't expose a row factory like sqlite3, so we wrap each
    query result so the rest of the inference code can dereference rows
    by column name (matching the validator's style).
    """
    cur = conn.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _active_detour_ids(conn: duckdb.DuckDBPyConnection) -> set[str]:
    rows = conn.execute("SELECT id FROM bus_v3_detour WHERE state = 1").fetchall()
    return {str(r[0]).upper() for r in rows if r[0] is not None}


def _json_list(s: Optional[str]) -> list[Any]:
    if not s:
        return []
    try:
        import json

        x = json.loads(s)
        return x if isinstance(x, list) else [x]
    except Exception:
        return []


def stop_removed_by_active_detour(conn: duckdb.DuckDBPyConnection, stpid: str) -> bool:
    active = _active_detour_ids(conn)
    if not active:
        return False
    row = conn.execute(
        "SELECT dtrrem_json FROM bus_v3_stop WHERE stpid = ?",
        [str(stpid)],
    ).fetchone()
    if row is None or not row[0]:
        return False
    removed = {str(x).upper() for x in _json_list(row[0])}
    return bool(active.intersection(removed))


def stop_added_by_active_detour(conn: duckdb.DuckDBPyConnection, stpid: str) -> bool:
    active = _active_detour_ids(conn)
    if not active:
        return False
    row = conn.execute(
        "SELECT dtradd_json FROM bus_v3_stop WHERE stpid = ?",
        [str(stpid)],
    ).fetchone()
    if row is None or not row[0]:
        return False
    added = {str(x).upper() for x in _json_list(row[0])}
    return bool(active.intersection(added))


def detour_active_for_route_dir(
    conn: duckdb.DuckDBPyConnection,
    rt: Optional[str],
    rtdir: Optional[str],
) -> bool:
    if not rt:
        return False
    rows = conn.execute(
        "SELECT route_dirs_json FROM bus_v3_detour WHERE state = 1"
    ).fetchall()
    rt_norm = str(rt).lower()
    dir_norm = str(rtdir).lower() if rtdir else None
    for (raw,) in rows:
        for entry in _json_list(raw):
            entries: list[Any]
            if isinstance(entry, dict) and "rtdir" in entry:
                v = entry.get("rtdir")
                entries = v if isinstance(v, list) else [v]
            else:
                entries = [entry]
            for e in entries:
                if not isinstance(e, dict):
                    continue
                if str(e.get("rt", "")).lower() != rt_norm:
                    continue
                if dir_norm is None or str(e.get("dir", "")).lower() == dir_norm:
                    return True
    return False


def _prediction_group_severity(
    conn: duckdb.DuckDBPyConnection,
    cand: dict[str, Any],
) -> tuple[Optional[int], Optional[int], list[str]]:
    rows = conn.execute(
        """
        SELECT dyn, flagstop FROM bus_v3_prediction_observation
        WHERE run_id = ? AND stpid = ? AND vid = ?
          AND (? IS NULL OR rt = ?)
          AND (? IS NULL OR rtdir = ?)
          AND (? IS NULL OR tatripid = ?)
        ORDER BY local_response_end_ms
        """,
        [
            cand["run_id"], cand["stpid"], cand["vid"],
            cand["rt"], cand["rt"],
            cand["rtdir"], cand["rtdir"],
            cand["tatripid"], cand["tatripid"],
        ],
    ).fetchall()
    dyns = [r[0] for r in rows if r[0] is not None]
    flags = [r[1] for r in rows if r[1] is not None]
    reasons: list[str] = []
    if any(d in {1, 12, 18} for d in dyns):
        reasons.append(ReasonCode.DYN_CANCELED.value)
    if any(d in {16, 17} for d in dyns):
        reasons.append(ReasonCode.DYN_INVALIDATED.value)
    if any(d == 4 for d in dyns):
        reasons.append(ReasonCode.DYN_EXPRESSED.value)
    if any(d == 2 for d in dyns):
        reasons.append(ReasonCode.DYN_REASSIGNED.value)
    if any(d in {9, 10} for d in dyns):
        reasons.append(ReasonCode.DYN_PARTIAL_TRIP.value)
    if any(f == FLAGSTOP_ONLY_DISCHARGE for f in flags):
        reasons.append(ReasonCode.FLAGSTOP_ONLY_DISCHARGE.value)
    max_dyn = max(dyns) if dyns else None
    max_flag = max(flags) if flags else None
    return max_dyn, max_flag, reasons


def _candidate_predictions(
    conn: duckdb.DuckDBPyConnection,
    run_id: Optional[str],
    route: Optional[str],
    stop_id: Optional[str],
    direction: Optional[str],
) -> list[dict[str, Any]]:
    where = ["vid IS NOT NULL", "stpid IS NOT NULL"]
    params: list[Any] = []
    if run_id:
        where.append("run_id = ?")
        params.append(run_id)
    if route:
        where.append("rt = ?")
        params.append(route)
    if stop_id:
        where.append("stpid = ?")
        params.append(str(stop_id))
    if direction:
        where.append("rtdir = ?")
        params.append(direction)
    sql = f"""
        SELECT run_id, stpid, rt, rtdir, vid, tatripid, origtatripno, tablockid, stst, stsd,
               MIN(local_response_end_ms) AS first_seen_ms,
               MAX(COALESCE(prdtm_ms, local_response_end_ms)) AS last_prediction_ms,
               COUNT(*) AS n_predictions
        FROM bus_v3_prediction_observation
        WHERE {' AND '.join(where)}
        GROUP BY run_id, stpid, rt, rtdir, vid, tatripid, origtatripno, tablockid, stst, stsd
        ORDER BY first_seen_ms
    """
    return _rows_to_dicts(conn, sql, params)


def _vehicle_obs_for_candidate(
    conn: duckdb.DuckDBPyConnection,
    cand: dict[str, Any],
    lookback_ms: int = 10 * 60_000,
    lookahead_ms: int = 45 * 60_000,
) -> list[dict[str, Any]]:
    start = int(cand["first_seen_ms"] or 0) - lookback_ms
    end = int(cand["last_prediction_ms"] or cand["first_seen_ms"] or 0) + lookahead_ms
    params: list[Any] = [cand["run_id"], cand["vid"], start, end]
    where = [
        "run_id = ?",
        "vid = ?",
        "COALESCE(tmstmp_ms, local_response_end_ms) BETWEEN ? AND ?",
        "pdist_ft IS NOT NULL",
        "pid IS NOT NULL",
    ]
    if cand["rt"]:
        where.append("rt = ?")
        params.append(cand["rt"])
    rows = _rows_to_dicts(
        conn,
        f"""
        SELECT * FROM bus_v3_vehicle_observation
        WHERE {' AND '.join(where)}
        ORDER BY COALESCE(tmstmp_ms, local_response_end_ms), vehicle_obs_id
        """,
        params,
    )
    dedup: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for r in rows:
        key = (r["tmstmp_ms"], r["pid"], r["pdist_ft"])
        dedup[key] = r
    return sorted(
        dedup.values(),
        key=lambda r: ((r["tmstmp_ms"] or r["local_response_end_ms"]), r["vehicle_obs_id"]),
    )


def _high_conf_crossing(
    conn: duckdb.DuckDBPyConnection,
    cand: dict[str, Any],
    obs: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    for a, b in zip(obs, obs[1:]):
        if a["pid"] != b["pid"]:
            continue
        stop_pdist = stop_pdist_for_pid(conn, cand["stpid"], int(a["pid"]))
        if stop_pdist is None:
            continue
        p0, p1 = safe_float(a["pdist_ft"]), safe_float(b["pdist_ft"])
        t0 = a["tmstmp_ms"] or a["local_response_end_ms"]
        t1 = b["tmstmp_ms"] or b["local_response_end_ms"]
        if p0 is None or p1 is None or t0 is None or t1 is None or t1 <= t0:
            continue
        delta_p = p1 - p0
        dt_s = (t1 - t0) / 1000.0
        if not (0 < dt_s <= GROUND_TRUTH_MAX_GAP_S):
            continue
        speed_ftps = delta_p / dt_s
        if not (GROUND_TRUTH_MIN_SPEED_FTPS <= speed_ftps <= GROUND_TRUTH_MAX_SPEED_FTPS):
            continue
        if not (p0 <= stop_pdist <= p1):
            continue
        if a["vehicle_age_s"] is not None and a["vehicle_age_s"] > GROUND_TRUTH_MAX_VEHICLE_AGE_S:
            continue
        if b["vehicle_age_s"] is not None and b["vehicle_age_s"] > GROUND_TRUTH_MAX_VEHICLE_AGE_S:
            continue

        points = load_pattern_points(conn, int(a["pid"]))
        mm0 = map_match_to_pattern(points, a["lat"], a["lon"], a["hdg"]) if points else None
        mm1 = map_match_to_pattern(points, b["lat"], b["lon"], b["hdg"]) if points else None
        mm_ok = False
        if mm0 and mm1 and mm0.cross_track_m is not None and mm1.cross_track_m is not None:
            pdiff0 = abs((mm0.projected_pdist_ft or 0.0) - p0) if mm0.projected_pdist_ft is not None else math.inf
            pdiff1 = abs((mm1.projected_pdist_ft or 0.0) - p1) if mm1.projected_pdist_ft is not None else math.inf
            mm_ok = (
                mm0.cross_track_m <= GROUND_TRUTH_MAX_CROSSTRACK_M
                and mm1.cross_track_m <= GROUND_TRUTH_MAX_CROSSTRACK_M
                and pdiff0 <= GROUND_TRUTH_MAX_PDIST_MAPMATCH_DIFF_FT
                and pdiff1 <= GROUND_TRUTH_MAX_PDIST_MAPMATCH_DIFF_FT
            )
        if not mm_ok:
            continue
        frac = clamp((stop_pdist - p0) / delta_p, 0.0, 1.0)
        arrival_ms = int(t0 + frac * (t1 - t0))
        dist_stop0 = distance_to_stop_m(conn, cand["stpid"], a["lat"], a["lon"])
        dist_stop1 = distance_to_stop_m(conn, cand["stpid"], b["lat"], b["lon"])
        return {
            "actual_arrival_ms": arrival_ms,
            "pid": int(a["pid"]),
            "stop_pdist_ft": stop_pdist,
            "first_vehicle_obs_id": int(a["vehicle_obs_id"]),
            "second_vehicle_obs_id": int(b["vehicle_obs_id"]),
            "pdist_before_ft": p0,
            "pdist_after_ft": p1,
            "vehicle_time_before_ms": int(t0),
            "vehicle_time_after_ms": int(t1),
            "delta_t_s": dt_s,
            "speed_ftps": speed_ftps,
            "interpolation_fraction": frac,
            "map_match_before": mm0.as_dict() if mm0 else None,
            "map_match_after": mm1.as_dict() if mm1 else None,
            "distance_to_stop_before_m": dist_stop0,
            "distance_to_stop_after_m": dist_stop1,
        }
    return None


def infer_bus_arrivals(
    conn: duckdb.DuckDBPyConnection,
    *,
    run_id: Optional[str] = None,
    route: Optional[str] = None,
    stop_id: Optional[str] = None,
    direction: Optional[str] = None,
    replace: bool = False,
) -> int:
    """Infer arrival events for the given filter scope.

    High-confidence rows are emitted only when the crossing test passes.
    Censored / disrupted candidates land with ``high_confidence=FALSE``
    and a label/reason-codes blob for diagnostics. Returns the count of
    rows inserted in this call (existing rows are skipped if identical).
    """
    if replace and run_id:
        conn.execute(
            "DELETE FROM bus_v3_arrival_event WHERE run_id = ?", [run_id]
        )
    candidates = _candidate_predictions(conn, run_id, route, stop_id, direction)
    inserted = 0
    for cand in candidates:
        dyn, flag, severity_reasons = _prediction_group_severity(conn, cand)
        removed = stop_removed_by_active_detour(conn, cand["stpid"])
        active_detour = detour_active_for_route_dir(conn, cand["rt"], cand["rtdir"])
        added = stop_added_by_active_detour(conn, cand["stpid"])
        pre_reasons = list(severity_reasons)
        if active_detour:
            pre_reasons.append(ReasonCode.DETOUR_ACTIVE.value)
        if removed:
            pre_reasons.append(ReasonCode.STOP_REMOVED_BY_DETOUR.value)
        if added:
            pre_reasons.append(ReasonCode.STOP_ADDED_BY_DETOUR.value)

        obs = _vehicle_obs_for_candidate(conn, cand)
        crossing = _high_conf_crossing(conn, cand, obs)

        if dyn in DYN_SEVERE_ABSTAIN or flag == FLAGSTOP_ONLY_DISCHARGE:
            label = (
                ArrivalLabel.PASSED_WITHOUT_PICKUP_OR_EXPRESSED
                if (dyn == 4 or flag == FLAGSTOP_ONLY_DISCHARGE)
                else ArrivalLabel.CANCELED_OR_INVALIDATED
            )
            high = False
            conf = 0.0
            actual_ms = None
            evidence = {
                "n_vehicle_observations": len(obs),
                "dyn": dyn,
                "flagstop": flag,
                "crossing_ignored": crossing,
            }
            reasons = pre_reasons or [ReasonCode.ARRIVAL_ESTIMATE_ABSTAINED.value]
            pid = crossing.get("pid") if crossing else None
            stop_pdist = crossing.get("stop_pdist_ft") if crossing else None
            first_id = crossing.get("first_vehicle_obs_id") if crossing else None
            second_id = crossing.get("second_vehicle_obs_id") if crossing else None
        elif removed:
            label = ArrivalLabel.DETOUR_AMBIGUOUS
            high = False
            conf = 0.0
            actual_ms = None
            evidence = {
                "n_vehicle_observations": len(obs),
                "active_detour_stop_removed": True,
                "crossing_ignored": crossing,
            }
            reasons = pre_reasons
            pid = crossing.get("pid") if crossing else None
            stop_pdist = crossing.get("stop_pdist_ft") if crossing else None
            first_id = crossing.get("first_vehicle_obs_id") if crossing else None
            second_id = crossing.get("second_vehicle_obs_id") if crossing else None
        elif crossing:
            label = ArrivalLabel.ARRIVED_CONFIRMED
            high = True
            conf = 0.97
            actual_ms = crossing["actual_arrival_ms"]
            evidence = crossing
            reasons = pre_reasons + [
                ReasonCode.PDIST_CROSSED_STOP.value,
                ReasonCode.GPS_ON_EXPECTED_PATTERN.value,
                ReasonCode.GROUND_TRUTH_HIGH_CONFIDENCE.value,
            ]
            pid = crossing["pid"]
            stop_pdist = crossing["stop_pdist_ft"]
            first_id = crossing["first_vehicle_obs_id"]
            second_id = crossing["second_vehicle_obs_id"]
        else:
            if not obs:
                label = ArrivalLabel.NO_EVIDENCE_GHOST_CANDIDATE
                reasons = pre_reasons + [ReasonCode.VEHICLE_NOT_FOUND.value]
            elif all(
                (r["vehicle_age_s"] is not None and r["vehicle_age_s"] > GROUND_TRUTH_MAX_VEHICLE_AGE_S)
                for r in obs
            ):
                label = ArrivalLabel.STALE_DATA
                reasons = pre_reasons + [ReasonCode.VEHICLE_POSITION_STALE.value]
            else:
                label = ArrivalLabel.CENSORED_UNKNOWN
                reasons = pre_reasons + [ReasonCode.GROUND_TRUTH_LOW_CONFIDENCE.value]
            high = False
            conf = 0.0
            actual_ms = None
            last = obs[-1] if obs else None
            evidence = {
                "n_vehicle_observations": len(obs),
                "last_vehicle_obs_id": last["vehicle_obs_id"] if last else None,
                "last_pdist_ft": last["pdist_ft"] if last else None,
                "last_pid": last["pid"] if last else None,
            }
            pid = last["pid"] if last else None
            stop_pdist = (
                stop_pdist_for_pid(conn, cand["stpid"], int(pid))
                if pid is not None
                else None
            )
            first_id = None
            second_id = None

        # Avoid exact duplicate event rows after repeated inference runs.
        # COALESCE keeps NULL comparisons stable across rerun.
        duplicate = conn.execute(
            """
            SELECT event_id FROM bus_v3_arrival_event
            WHERE run_id = ?
              AND stpid = ?
              AND COALESCE(rt, '') = COALESCE(?, '')
              AND COALESCE(rtdir, '') = COALESCE(?, '')
              AND COALESCE(vid, '') = COALESCE(?, '')
              AND COALESCE(tatripid, '') = COALESCE(?, '')
              AND COALESCE(tablockid, '') = COALESCE(?, '')
              AND COALESCE(stsd, '') = COALESCE(?, '')
              AND COALESCE(stst, -1) = COALESCE(?, -1)
              AND label = ?
              AND COALESCE(actual_arrival_ms, -1) = COALESCE(?, -1)
            LIMIT 1
            """,
            [
                cand["run_id"], cand["stpid"], cand["rt"], cand["rtdir"], cand["vid"],
                cand["tatripid"], cand["tablockid"], cand["stsd"], cand["stst"],
                label.value, actual_ms,
            ],
        ).fetchone()
        if duplicate:
            continue
        conn.execute(
            """
            INSERT INTO bus_v3_arrival_event(
                run_id, stpid, rt, rtdir, vid, pid, tatripid, origtatripno, tablockid, stst, stsd,
                stop_pdist_ft, actual_arrival_ms, label, high_confidence, confidence,
                evidence_json, reason_codes_json, first_vehicle_obs_id, second_vehicle_obs_id, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                cand["run_id"], cand["stpid"], cand["rt"], cand["rtdir"], cand["vid"], pid,
                cand["tatripid"], cand["origtatripno"], cand["tablockid"], cand["stst"], cand["stsd"],
                stop_pdist, actual_ms, label.value, bool(high), conf,
                json_dumps(evidence), json_dumps(reasons), first_id, second_id, now_ms(),
            ],
        )
        inserted += 1
    return inserted
