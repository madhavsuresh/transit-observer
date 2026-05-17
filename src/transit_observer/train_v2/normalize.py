"""DuckDB normalizers for the train_v2 pipeline.

One ``record_poll`` dispatcher per endpoint (matching bus_v3.normalize).
Writes raw JSON to ``train_v2_api_poll`` and projected rows to the
endpoint-specific normalized tables.
"""

from __future__ import annotations

from typing import Any, Optional

import duckdb

from .models import ApiCallResult
from .util import (
    json_dumps,
    json_sha256,
    now_ms,
    parse_cta_train_dt_ms,
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
    raw_json = json_dumps(result.json_data) if result.json_data is not None else None
    raw_sha = json_sha256(result.json_data) if result.json_data is not None else None
    row = conn.execute(
        """
        INSERT INTO train_v2_api_poll(
            run_id, cycle_index, source, endpoint, query_kind, request_url_redacted,
            params_json_redacted, local_request_start_ms, local_response_end_ms,
            cta_server_time_ms, http_status, latency_ms, ok, error_message,
            raw_json, raw_sha256, created_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING poll_id
        """,
        [
            run_id,
            cycle_index,
            result.source,
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
        raise RuntimeError("train_v2: insert_api_poll did not return a poll_id")
    return int(row[0])


def normalize_poll(
    conn: duckdb.DuckDBPyConnection,
    poll_id: int,
    result: ApiCallResult,
    run_id: str,
) -> None:
    if not result.ok or result.json_data is None:
        return
    endpoint = result.endpoint.lower()
    if endpoint == "ttarrivals.aspx":
        _normalize_ttarrivals(conn, poll_id, result, run_id)
    elif endpoint == "ttfollow.aspx":
        _normalize_ttfollow(conn, poll_id, result, run_id)
    elif endpoint == "ttpositions.aspx":
        _normalize_ttpositions(conn, poll_id, result, run_id)


def record_poll(
    conn: duckdb.DuckDBPyConnection,
    result: ApiCallResult,
    *,
    run_id: str,
    cycle_index: Optional[int] = None,
) -> int:
    """Insert api_poll row + normalize. Returns the poll_id."""
    # ``ttarrivals.aspx`` / ``ttfollow.aspx`` / ``ttpositions.aspx`` carry
    # a ``tmst`` field — CTA's server clock at response generation time.
    # We fold it back into the ApiCallResult so the normalizer can derive
    # eta_s and prediction_age_s with the right clock.
    server_ms = _extract_server_time_ms(result)
    if server_ms is not None:
        result.cta_server_time_ms = server_ms
    poll_id = insert_api_poll(conn, result, run_id=run_id, cycle_index=cycle_index)
    normalize_poll(conn, poll_id, result, run_id)
    return poll_id


def _extract_server_time_ms(result: ApiCallResult) -> Optional[int]:
    if not result.json_data:
        return None
    ctatt = result.json_data.get("ctatt") or {}
    return parse_cta_train_dt_ms(ctatt.get("tmst"))


def _ctatt(result: ApiCallResult) -> dict[str, Any]:
    return (result.json_data or {}).get("ctatt", {}) or {}


def _err_ok(ctatt: dict[str, Any]) -> bool:
    code = ctatt.get("errCd")
    return code in (None, "0", 0)


def _normalize_ttarrivals(
    conn: duckdb.DuckDBPyConnection,
    poll_id: int,
    result: ApiCallResult,
    run_id: str,
) -> None:
    ctatt = _ctatt(result)
    if not _err_ok(ctatt):
        return
    server_ms = result.cta_server_time_ms
    etas = ctatt.get("eta") or []
    for raw in etas:
        if not isinstance(raw, dict):
            continue
        prdt = parse_cta_train_dt_ms(raw.get("prdt"))
        arr = parse_cta_train_dt_ms(raw.get("arrT"))
        eta_s = None
        if arr is not None and server_ms is not None:
            eta_s = (arr - server_ms) / 1000.0
        pred_age_s = None
        if prdt is not None and server_ms is not None:
            pred_age_s = (server_ms - prdt) / 1000.0
        conn.execute(
            """
            INSERT INTO train_v2_arrival_observation(
                poll_id, run_id, cta_server_time_ms, local_response_end_ms, query_kind,
                line, run_number, map_id, stop_id, station_name, stop_description,
                direction_code, destination_name, destination_map_id,
                predicted_at_ms, arrival_at_ms, eta_s, prediction_age_s,
                is_approaching, is_delayed, is_fault, is_scheduled, flags, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                poll_id, run_id, server_ms, result.local_response_end_ms, result.query_kind,
                str(raw.get("rt")) if raw.get("rt") is not None else None,
                str(raw.get("rn")) if raw.get("rn") is not None else None,
                str(raw.get("staId")) if raw.get("staId") is not None else None,
                str(raw.get("stpId")) if raw.get("stpId") is not None else None,
                raw.get("staNm"),
                raw.get("stpDe"),
                raw.get("trDr"),
                raw.get("destNm"),
                str(raw.get("destSt")) if raw.get("destSt") is not None else None,
                prdt, arr, eta_s, pred_age_s,
                _flag_bool(raw.get("isApp")),
                _flag_bool(raw.get("isDly")),
                _flag_bool(raw.get("isFlt")),
                _flag_bool(raw.get("isSch")),
                str(raw.get("flags")) if raw.get("flags") not in (None, "") else None,
                json_dumps(raw),
            ],
        )


def _normalize_ttfollow(
    conn: duckdb.DuckDBPyConnection,
    poll_id: int,
    result: ApiCallResult,
    run_id: str,
) -> None:
    ctatt = _ctatt(result)
    if not _err_ok(ctatt):
        return
    server_ms = result.cta_server_time_ms
    # ttfollow returns a list of eta rows plus the queried run number.
    # The run number sits at ctatt.runno (not on each eta).
    run_number = ctatt.get("runno")
    if run_number is None and ctatt.get("eta"):
        first = ctatt["eta"][0] if isinstance(ctatt["eta"], list) and ctatt["eta"] else None
        if isinstance(first, dict):
            run_number = first.get("rn")
    run_number = str(run_number) if run_number is not None else None
    etas = ctatt.get("eta") or []
    if not isinstance(etas, list):
        etas = [etas]
    for seq, raw in enumerate(etas):
        if not isinstance(raw, dict):
            continue
        prdt = parse_cta_train_dt_ms(raw.get("prdt"))
        arr = parse_cta_train_dt_ms(raw.get("arrT"))
        eta_s = None
        if arr is not None and server_ms is not None:
            eta_s = (arr - server_ms) / 1000.0
        conn.execute(
            """
            INSERT INTO train_v2_follow_observation(
                poll_id, run_id, cta_server_time_ms, local_response_end_ms,
                run_number, line, seq, map_id, stop_id, station_name,
                direction_code, destination_name, predicted_at_ms, arrival_at_ms, eta_s,
                is_approaching, is_delayed, is_fault, is_scheduled, flags, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                poll_id, run_id, server_ms, result.local_response_end_ms,
                run_number or str(raw.get("rn") or ""),
                str(raw.get("rt")) if raw.get("rt") is not None else None,
                seq,
                str(raw.get("staId")) if raw.get("staId") is not None else None,
                str(raw.get("stpId")) if raw.get("stpId") is not None else None,
                raw.get("staNm"),
                raw.get("trDr"),
                raw.get("destNm"),
                prdt, arr, eta_s,
                _flag_bool(raw.get("isApp")),
                _flag_bool(raw.get("isDly")),
                _flag_bool(raw.get("isFlt")),
                _flag_bool(raw.get("isSch")),
                str(raw.get("flags")) if raw.get("flags") not in (None, "") else None,
                json_dumps(raw),
            ],
        )


def _normalize_ttpositions(
    conn: duckdb.DuckDBPyConnection,
    poll_id: int,
    result: ApiCallResult,
    run_id: str,
) -> None:
    ctatt = _ctatt(result)
    if not _err_ok(ctatt):
        return
    server_ms = result.cta_server_time_ms
    for route in ctatt.get("route") or []:
        if not isinstance(route, dict):
            continue
        line = route.get("@name") or route.get("name") or ""
        trains = route.get("train") or []
        if isinstance(trains, dict):
            trains = [trains]
        for raw in trains:
            if not isinstance(raw, dict):
                continue
            prdt = parse_cta_train_dt_ms(raw.get("prdt"))
            arr = parse_cta_train_dt_ms(raw.get("arrT"))
            conn.execute(
                """
                INSERT INTO train_v2_position_observation(
                    poll_id, run_id, cta_server_time_ms, local_response_end_ms,
                    line, run_number, direction_code, destination_name, destination_map_id,
                    next_station_map_id, next_station_name,
                    predicted_at_ms, next_arrival_at_ms,
                    is_approaching, is_delayed, is_fault,
                    lat, lon, heading, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    poll_id, run_id, server_ms, result.local_response_end_ms,
                    str(line), str(raw.get("rn")) if raw.get("rn") is not None else None,
                    raw.get("trDr"),
                    raw.get("destNm"),
                    str(raw.get("destSt")) if raw.get("destSt") is not None else None,
                    str(raw.get("nextStaId")) if raw.get("nextStaId") is not None else None,
                    raw.get("nextStaNm"),
                    prdt, arr,
                    _flag_bool(raw.get("isApp")),
                    _flag_bool(raw.get("isDly")),
                    _flag_bool(raw.get("isFlt")),
                    safe_float(raw.get("lat")),
                    safe_float(raw.get("lon")),
                    safe_float(raw.get("heading")),
                    json_dumps(raw),
                ],
            )


def _flag_bool(raw: Any) -> Optional[bool]:
    """CTA boolean flags come back as strings: '1' / '0'. Treat None /
    empty as None so SQL can distinguish 'unknown' from 'false'."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip()
    if s == "1":
        return True
    if s == "0":
        return False
    return None


def normalize_gtfsrt_trip_updates(
    conn: duckdb.DuckDBPyConnection,
    poll_id: int,
    run_id: str,
    rows: list[Any],
    *,
    local_response_end_ms: int,
) -> int:
    """Normalize GTFS-RT TripUpdates into train_v2_gtfsrt_trip_update.

    Accepts ``cta_gtfsrt_client.CTAGtfsRtTripUpdate`` dataclasses (which
    have ``arrival_time`` as a tz-aware ``datetime``). Returns row count.
    """
    n = 0
    for r in rows:
        arrival_ms = _dt_to_ms(getattr(r, "arrival_time", None))
        departure_ms = _dt_to_ms(getattr(r, "departure_time", None))
        conn.execute(
            """
            INSERT INTO train_v2_gtfsrt_trip_update(
                poll_id, run_id, local_response_end_ms, route_id, trip_id, vehicle_id,
                stop_id, stop_sequence, arrival_time_ms, arrival_delay_seconds,
                departure_time_ms, departure_delay_seconds, schedule_relationship
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                poll_id, run_id, local_response_end_ms,
                getattr(r, "route_id", None), getattr(r, "trip_id", None),
                getattr(r, "vehicle_id", None), getattr(r, "stop_id", None),
                getattr(r, "stop_sequence", None),
                arrival_ms, getattr(r, "arrival_delay_seconds", None),
                departure_ms, getattr(r, "departure_delay_seconds", None),
                getattr(r, "schedule_relationship", None),
            ],
        )
        n += 1
    return n


def normalize_gtfsrt_vehicle_positions(
    conn: duckdb.DuckDBPyConnection,
    poll_id: int,
    run_id: str,
    rows: list[Any],
    *,
    local_response_end_ms: int,
) -> int:
    """Normalize GTFS-RT VehiclePositions into train_v2_gtfsrt_vehicle_position."""
    n = 0
    for r in rows:
        conn.execute(
            """
            INSERT INTO train_v2_gtfsrt_vehicle_position(
                poll_id, run_id, local_response_end_ms, route_id, trip_id, vehicle_id,
                vehicle_label, lat, lon, bearing, speed_mps, current_stop_sequence,
                current_status, congestion_level, occupancy_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                poll_id, run_id, local_response_end_ms,
                getattr(r, "route_id", None), getattr(r, "trip_id", None),
                getattr(r, "vehicle_id", None), getattr(r, "vehicle_label", None),
                getattr(r, "lat", None), getattr(r, "lon", None),
                getattr(r, "bearing", None), getattr(r, "speed_mps", None),
                getattr(r, "current_stop_sequence", None),
                getattr(r, "current_status", None), getattr(r, "congestion_level", None),
                getattr(r, "occupancy_status", None),
            ],
        )
        n += 1
    return n


def _dt_to_ms(dt: Any) -> Optional[int]:
    if dt is None:
        return None
    try:
        return int(dt.timestamp() * 1000)
    except Exception:
        return None
