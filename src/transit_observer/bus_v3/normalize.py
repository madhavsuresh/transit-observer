"""Raw-JSON -> DuckDB normalized rows for the v3 bus pipeline.

One ``normalize_poll`` dispatcher per endpoint, modeled on the
validator's ``normalizer.py`` but using DuckDB's ``ON CONFLICT … DO
UPDATE`` syntax and ``RETURNING`` for surrogate-key reads.

All inserts are idempotent on natural keys; rerunning the same poll
yields the same rows. Vehicle and prediction observations are
intentionally append-only (each poll is a fresh snapshot).
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

import duckdb

from .models import ApiCallResult
from .util import (
    as_list,
    json_dumps,
    json_sha256,
    now_ms,
    parse_cta_timestamp_ms,
    parse_prdctdn_minutes,
    pick_first,
    root_of,
    safe_bool_int,
    safe_float,
    safe_int,
)


def insert_api_poll(
    conn: duckdb.DuckDBPyConnection,
    result: ApiCallResult,
    *,
    run_id: str,
    cycle_index: Optional[int] = None,
) -> int:
    """Write one ``bus_v3_api_poll`` row and return its generated ``poll_id``."""
    raw_json = json_dumps(result.json_data) if result.json_data is not None else None
    raw_sha = json_sha256(result.json_data) if result.json_data is not None else None
    row = conn.execute(
        """
        INSERT INTO bus_v3_api_poll(
            run_id, cycle_index, endpoint, query_kind, request_url_redacted,
            params_json_redacted, local_request_start_ms, local_response_end_ms,
            cta_server_time_ms, http_status, latency_ms, ok, error_message,
            raw_json, raw_sha256, created_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING poll_id
        """,
        [
            run_id,
            cycle_index,
            result.endpoint,
            result.query_kind,
            result.request_url_redacted,
            json_dumps(result.params_redacted),
            result.local_request_start_ms,
            result.local_response_end_ms,
            result.cta_server_time_ms,
            result.http_status,
            result.latency_ms,
            bool(result.ok),
            result.error_message,
            raw_json,
            raw_sha,
            now_ms(),
        ],
    ).fetchone()
    if row is None or row[0] is None:
        raise RuntimeError("insert_api_poll did not return a poll_id")
    return int(row[0])


def normalize_poll(
    conn: duckdb.DuckDBPyConnection,
    poll_id: int,
    result: ApiCallResult,
    run_id: str,
) -> None:
    root = root_of(result.json_data)
    if not root:
        return
    _insert_errors(conn, poll_id, result.endpoint, root)
    endpoint = result.endpoint.lower()
    if endpoint == "getroutes":
        _normalize_routes(conn, poll_id, root)
    elif endpoint == "getdirections":
        _normalize_directions(conn, poll_id, root, result)
    elif endpoint == "getstops":
        _normalize_stops(conn, poll_id, root, result)
    elif endpoint == "getpatterns":
        _normalize_patterns(conn, poll_id, root, result)
    elif endpoint == "getvehicles":
        _normalize_vehicles(conn, poll_id, root, result, run_id)
    elif endpoint == "getpredictions":
        _normalize_predictions(conn, poll_id, root, result, run_id)
    elif endpoint == "getdetours":
        _normalize_detours(conn, poll_id, root, result, enhanced=False)
    elif endpoint == "getenhanceddetours":
        _normalize_detours(conn, poll_id, root, result, enhanced=True)


def record_poll(
    conn: duckdb.DuckDBPyConnection,
    result: ApiCallResult,
    *,
    run_id: str,
    cycle_index: Optional[int] = None,
) -> int:
    """Insert api_poll row + normalize. Returns the poll_id."""
    poll_id = insert_api_poll(conn, result, run_id=run_id, cycle_index=cycle_index)
    normalize_poll(conn, poll_id, result, run_id)
    return poll_id


def _insert_errors(
    conn: duckdb.DuckDBPyConnection,
    poll_id: int,
    endpoint: str,
    root: dict[str, Any],
) -> None:
    for err in pick_first(root, "error", "errors"):
        if not isinstance(err, dict):
            continue
        conn.execute(
            """
            INSERT INTO bus_v3_api_error(poll_id, endpoint, rt, stpid, vid, msg, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                poll_id,
                endpoint,
                err.get("rt"),
                err.get("stpid"),
                err.get("vid"),
                str(err.get("msg") or err),
                json_dumps(err),
            ],
        )


def _normalize_routes(conn: duckdb.DuckDBPyConnection, poll_id: int, root: dict[str, Any]) -> None:
    for r in pick_first(root, "routes", "route"):
        if not isinstance(r, dict):
            continue
        rt = str(r.get("rt", "")).strip()
        if not rt:
            continue
        conn.execute(
            """
            INSERT INTO bus_v3_route(rt, rtnm, rtclr, rtdd, first_seen_poll_id, last_seen_poll_id, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (rt) DO UPDATE SET
                rtnm = excluded.rtnm,
                rtclr = excluded.rtclr,
                rtdd = excluded.rtdd,
                last_seen_poll_id = excluded.last_seen_poll_id,
                raw_json = excluded.raw_json
            """,
            [rt, r.get("rtnm"), r.get("rtclr"), r.get("rtdd"), poll_id, poll_id, json_dumps(r)],
        )


def _normalize_directions(
    conn: duckdb.DuckDBPyConnection,
    poll_id: int,
    root: dict[str, Any],
    result: ApiCallResult,
) -> None:
    rt = result.params_redacted.get("rt")
    for d in pick_first(root, "directions", "dir"):
        if not isinstance(d, dict):
            continue
        dir_id = d.get("id") or d.get("name")
        if not rt or not dir_id:
            continue
        conn.execute(
            """
            INSERT INTO bus_v3_direction(rt, dir_id, name, first_seen_poll_id, last_seen_poll_id, raw_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (rt, dir_id) DO UPDATE SET
                name = excluded.name,
                last_seen_poll_id = excluded.last_seen_poll_id,
                raw_json = excluded.raw_json
            """,
            [str(rt), str(dir_id), d.get("name"), poll_id, poll_id, json_dumps(d)],
        )


def _normalize_stops(
    conn: duckdb.DuckDBPyConnection,
    poll_id: int,
    root: dict[str, Any],
    result: ApiCallResult,
) -> None:
    rt = result.params_redacted.get("rt")
    rtdir = result.params_redacted.get("dir")
    for s in pick_first(root, "stops", "stop"):
        if not isinstance(s, dict):
            continue
        stpid = str(s.get("stpid", "")).strip()
        if not stpid:
            continue
        conn.execute(
            """
            INSERT INTO bus_v3_stop(stpid, stpnm, lat, lon, rt, rtdir, dtradd_json, dtrrem_json,
                                    first_seen_poll_id, last_seen_poll_id, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (stpid) DO UPDATE SET
                stpnm = excluded.stpnm,
                lat = COALESCE(excluded.lat, bus_v3_stop.lat),
                lon = COALESCE(excluded.lon, bus_v3_stop.lon),
                rt = COALESCE(excluded.rt, bus_v3_stop.rt),
                rtdir = COALESCE(excluded.rtdir, bus_v3_stop.rtdir),
                dtradd_json = excluded.dtradd_json,
                dtrrem_json = excluded.dtrrem_json,
                last_seen_poll_id = excluded.last_seen_poll_id,
                raw_json = excluded.raw_json
            """,
            [
                stpid,
                s.get("stpnm"),
                safe_float(s.get("lat")),
                safe_float(s.get("lon")),
                rt,
                rtdir,
                json_dumps(as_list(s.get("dtradd"))) if s.get("dtradd") is not None else None,
                json_dumps(as_list(s.get("dtrrem"))) if s.get("dtrrem") is not None else None,
                poll_id,
                poll_id,
                json_dumps(s),
            ],
        )


def _normalize_patterns(
    conn: duckdb.DuckDBPyConnection,
    poll_id: int,
    root: dict[str, Any],
    result: ApiCallResult,
) -> None:
    rt = result.params_redacted.get("rt")
    for p in pick_first(root, "ptr", "ptrs", "pattern", "patterns"):
        if not isinstance(p, dict):
            continue
        pid = safe_int(p.get("pid"))
        if pid is None:
            continue
        rtdir = p.get("rtdir")
        conn.execute(
            """
            INSERT INTO bus_v3_pattern(pid, rt, rtdir, length_ft, dtrid, first_seen_poll_id, last_seen_poll_id, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (pid) DO UPDATE SET
                rt = COALESCE(excluded.rt, bus_v3_pattern.rt),
                rtdir = excluded.rtdir,
                length_ft = excluded.length_ft,
                dtrid = excluded.dtrid,
                last_seen_poll_id = excluded.last_seen_poll_id,
                raw_json = excluded.raw_json
            """,
            [pid, rt, rtdir, safe_float(p.get("ln")), p.get("dtrid"), poll_id, poll_id, json_dumps(p)],
        )
        for pt in as_list(p.get("pt")):
            if isinstance(pt, dict):
                _insert_pattern_point(conn, pid, pt, is_detour_original_point=0)
        for pt in as_list(p.get("dtrpt")):
            if isinstance(pt, dict):
                _insert_pattern_point(conn, pid, pt, is_detour_original_point=1)


def _insert_pattern_point(
    conn: duckdb.DuckDBPyConnection,
    pid: int,
    pt: dict[str, Any],
    is_detour_original_point: int,
) -> None:
    seq = safe_int(pt.get("seq"))
    if seq is None:
        return
    conn.execute(
        """
        INSERT INTO bus_v3_pattern_point(pid, seq, typ, stpid, stpnm, lat, lon, pdist_ft,
                                         is_detour_original_point, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (pid, seq, is_detour_original_point) DO UPDATE SET
            typ = excluded.typ,
            stpid = excluded.stpid,
            stpnm = excluded.stpnm,
            lat = excluded.lat,
            lon = excluded.lon,
            pdist_ft = excluded.pdist_ft,
            raw_json = excluded.raw_json
        """,
        [
            pid,
            seq,
            pt.get("typ"),
            str(pt.get("stpid")) if pt.get("stpid") is not None else None,
            pt.get("stpnm"),
            safe_float(pt.get("lat")),
            safe_float(pt.get("lon")),
            safe_float(pt.get("pdist")),
            is_detour_original_point,
            json_dumps(pt),
        ],
    )


def _normalize_vehicles(
    conn: duckdb.DuckDBPyConnection,
    poll_id: int,
    root: dict[str, Any],
    result: ApiCallResult,
    run_id: str,
) -> None:
    server_ms = result.cta_server_time_ms
    for v in pick_first(root, "vehicle", "vehicles"):
        if not isinstance(v, dict):
            continue
        vid = str(v.get("vid", "")).strip()
        if not vid:
            continue
        tmstmp = parse_cta_timestamp_ms(v.get("tmstmp") or v.get("tmpstmp"))
        age_s = None
        if tmstmp is not None and server_ms is not None:
            age_s = (server_ms - tmstmp) / 1000.0
        conn.execute(
            """
            INSERT INTO bus_v3_vehicle_observation(
                poll_id, run_id, cta_server_time_ms, local_response_end_ms, vid, tmstmp_ms, vehicle_age_s,
                lat, lon, hdg, pid, pdist_ft, rt, des, dly, tablockid, tatripid, origtatripno,
                zone, mode, psgld, stst, stsd, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                poll_id,
                run_id,
                server_ms,
                result.local_response_end_ms,
                vid,
                tmstmp,
                age_s,
                safe_float(v.get("lat")),
                safe_float(v.get("lon")),
                safe_float(v.get("hdg")),
                safe_int(v.get("pid")),
                safe_float(v.get("pdist")),
                str(v.get("rt")) if v.get("rt") is not None else None,
                v.get("des"),
                safe_bool_int(v.get("dly")),
                v.get("tablockid"),
                str(v.get("tatripid")) if v.get("tatripid") is not None else None,
                str(v.get("origtatripno")) if v.get("origtatripno") is not None else None,
                v.get("zone"),
                safe_int(v.get("mode")),
                v.get("psgld"),
                safe_int(v.get("stst")),
                v.get("stsd"),
                json_dumps(v),
            ],
        )


def _normalize_predictions(
    conn: duckdb.DuckDBPyConnection,
    poll_id: int,
    root: dict[str, Any],
    result: ApiCallResult,
    run_id: str,
) -> None:
    server_ms = result.cta_server_time_ms
    for p in pick_first(root, "prd", "prds", "prediction", "predictions"):
        if not isinstance(p, dict):
            continue
        stpid = str(p.get("stpid")) if p.get("stpid") is not None else None
        vid = str(p.get("vid")) if p.get("vid") is not None else None
        tmstmp = parse_cta_timestamp_ms(p.get("tmstmp"))
        prdtm = parse_cta_timestamp_ms(p.get("prdtm"))
        pred_age_s = None
        eta_s = None
        if tmstmp is not None and server_ms is not None:
            pred_age_s = (server_ms - tmstmp) / 1000.0
        if prdtm is not None and server_ms is not None:
            eta_s = (prdtm - server_ms) / 1000.0
        conn.execute(
            """
            INSERT INTO bus_v3_prediction_observation(
                poll_id, run_id, cta_server_time_ms, local_response_end_ms, query_kind, tmstmp_ms,
                prediction_age_s, typ, stpid, stpnm, vid, dstp_ft, rt, rtdd, rtdir, des, prdtm_ms,
                eta_s, prdctdn_raw, prdctdn_min, dly, dyn, tablockid, tatripid, origtatripno, zone,
                psgld, stst, stsd, flagstop, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                poll_id,
                run_id,
                server_ms,
                result.local_response_end_ms,
                result.query_kind,
                tmstmp,
                pred_age_s,
                p.get("typ"),
                stpid,
                p.get("stpnm"),
                vid,
                safe_float(p.get("dstp")),
                str(p.get("rt")) if p.get("rt") is not None else None,
                p.get("rtdd"),
                p.get("rtdir"),
                p.get("des"),
                prdtm,
                eta_s,
                str(p.get("prdctdn")) if p.get("prdctdn") is not None else None,
                parse_prdctdn_minutes(p.get("prdctdn")),
                safe_bool_int(p.get("dly")),
                safe_int(p.get("dyn")),
                p.get("tablockid"),
                str(p.get("tatripid")) if p.get("tatripid") is not None else None,
                str(p.get("origtatripno")) if p.get("origtatripno") is not None else None,
                p.get("zone"),
                p.get("psgld"),
                safe_int(p.get("stst")),
                p.get("stsd"),
                safe_int(p.get("flagstop")),
                json_dumps(p),
            ],
        )


def _normalize_detours(
    conn: duckdb.DuckDBPyConnection,
    poll_id: int,
    root: dict[str, Any],
    result: ApiCallResult,
    enhanced: bool,
) -> None:
    for d in pick_first(root, "dtrs", "dtr"):
        if not isinstance(d, dict):
            continue
        detour_id = str(d.get("id", "")).strip()
        if not detour_id:
            continue
        ver = safe_int(d.get("ver"))
        detour_pk = f"{detour_id}:{ver if ver is not None else ''}"
        route_dirs = as_list(d.get("rtdirs") or d.get("rtdir"))
        conn.execute(
            """
            INSERT INTO bus_v3_detour(detour_pk, id, ver, state, descr, route_dirs_json,
                                      startdt_ms, enddt_ms, moddt_ms,
                                      first_seen_poll_id, last_seen_poll_id, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (detour_pk) DO UPDATE SET
                state = excluded.state,
                descr = excluded.descr,
                route_dirs_json = excluded.route_dirs_json,
                startdt_ms = excluded.startdt_ms,
                enddt_ms = excluded.enddt_ms,
                moddt_ms = excluded.moddt_ms,
                last_seen_poll_id = excluded.last_seen_poll_id,
                raw_json = excluded.raw_json
            """,
            [
                detour_pk,
                detour_id,
                ver,
                safe_int(d.get("st")),
                d.get("desc"),
                json_dumps(route_dirs),
                parse_cta_timestamp_ms(d.get("startdt")),
                parse_cta_timestamp_ms(d.get("enddt")),
                parse_cta_timestamp_ms(d.get("moddt")),
                poll_id,
                poll_id,
                json_dumps(d),
            ],
        )
        if enhanced:
            _normalize_enhanced_detour_children(conn, detour_pk, d)


def _normalize_enhanced_detour_children(
    conn: duckdb.DuckDBPyConnection,
    detour_pk: str,
    d: dict[str, Any],
) -> None:
    for ptr in as_list(d.get("ptrs") or d.get("ptr")):
        if not isinstance(ptr, dict):
            continue
        origpid = safe_int(ptr.get("origpid"))
        dtrpid = safe_int(ptr.get("dtrpid"))
        if origpid is None or dtrpid is None:
            continue
        conn.execute(
            """
            INSERT INTO bus_v3_enhanced_detour_pattern(detour_pk, origpid, dtrpid, encoded_polyline,
                                                      delay_s, raw_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (detour_pk, origpid, dtrpid) DO UPDATE SET
                encoded_polyline = excluded.encoded_polyline,
                delay_s = excluded.delay_s,
                raw_json = excluded.raw_json
            """,
            [
                detour_pk,
                origpid,
                dtrpid,
                ptr.get("encpl"),
                safe_int(ptr.get("dtrdelay")),
                json_dumps(ptr),
            ],
        )
    for trip in as_list(d.get("trips") or d.get("trip")):
        if not isinstance(trip, dict):
            continue
        conn.execute(
            """
            INSERT INTO bus_v3_enhanced_detour_trip(detour_pk, tripid, tatripid, origtatripno,
                                                   dates_json, stst, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (detour_pk, tripid, tatripid, origtatripno, stst) DO UPDATE SET
                dates_json = excluded.dates_json,
                raw_json = excluded.raw_json
            """,
            [
                detour_pk,
                str(trip.get("tripid")) if trip.get("tripid") is not None else None,
                str(trip.get("tatripid")) if trip.get("tatripid") is not None else None,
                str(trip.get("origtatripno")) if trip.get("origtatripno") is not None else None,
                json_dumps(as_list(trip.get("dates") or trip.get("date"))),
                safe_int(trip.get("stst")),
                json_dumps(trip),
            ],
        )
    for role, key in [("start", "dtrstartstop"), ("end", "dtrendstop")]:
        stop = d.get(key)
        if isinstance(stop, dict):
            _insert_replacement_stop(conn, detour_pk, role, stop)
    for stop in as_list(d.get("repstops") or d.get("repstop")):
        if isinstance(stop, dict):
            _insert_replacement_stop(conn, detour_pk, "replacement", stop)


def _insert_replacement_stop(
    conn: duckdb.DuckDBPyConnection,
    detour_pk: str,
    role: str,
    stop: dict[str, Any],
) -> None:
    stpid = str(stop.get("stpid")) if stop.get("stpid") is not None else None
    seq = safe_int(stop.get("seq")) or 0
    conn.execute(
        """
        INSERT INTO bus_v3_enhanced_detour_replacement_stop(
            detour_pk, role, geoid, stpid, seq, stpnm, lat, lon, adhoc, relpasstime_s, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (detour_pk, role, stpid, seq) DO UPDATE SET
            geoid = excluded.geoid,
            stpnm = excluded.stpnm,
            lat = excluded.lat,
            lon = excluded.lon,
            adhoc = excluded.adhoc,
            relpasstime_s = excluded.relpasstime_s,
            raw_json = excluded.raw_json
        """,
        [
            detour_pk,
            role,
            str(stop.get("geoid")) if stop.get("geoid") is not None else None,
            stpid,
            seq,
            stop.get("stpnm"),
            safe_float(stop.get("lat")),
            safe_float(stop.get("lon")),
            safe_bool_int(stop.get("adhoc")),
            safe_int(stop.get("relpasstime")),
            json_dumps(stop),
        ],
    )
