"""Pace (suburban bus) GTFS-Realtime ingest.

Reuses :class:`cta_gtfsrt_client.CTAGtfsRtClient` (the GTFS-RT bindings
are agency-neutral) but writes to a separate ``pace_gtfsrt_*`` namespace
so Pace bus data doesn't muddy the CTA train tables. The collector
hooks the ``pace`` mode in ``cta_gtfsrt_feeds``.
"""

from __future__ import annotations

from typing import Any, Optional

import duckdb

from .train_v2.util import json_dumps, now_ms


def insert_pace_api_poll(
    conn: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    cycle_index: Optional[int],
    endpoint: str,
    query_kind: str,
    request_url_redacted: str,
    local_request_start_ms: int,
    local_response_end_ms: int,
    http_status: Optional[int],
    latency_ms: float,
    ok: bool,
    error_message: Optional[str],
) -> int:
    row = conn.execute(
        """
        INSERT INTO pace_gtfsrt_api_poll(
            run_id, cycle_index, endpoint, query_kind, request_url_redacted,
            local_request_start_ms, local_response_end_ms, http_status, latency_ms,
            ok, error_message, created_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING poll_id
        """,
        [
            run_id, cycle_index, endpoint, query_kind, request_url_redacted,
            local_request_start_ms, local_response_end_ms,
            http_status, latency_ms, bool(ok), error_message, now_ms(),
        ],
    ).fetchone()
    if row is None or row[0] is None:
        raise RuntimeError("pace_gtfsrt: insert_api_poll did not return a poll_id")
    return int(row[0])


def _dt_to_ms(dt: Any) -> Optional[int]:
    if dt is None:
        return None
    try:
        return int(dt.timestamp() * 1000)
    except Exception:  # noqa: BLE001
        return None


def normalize_pace_trip_updates(
    conn: duckdb.DuckDBPyConnection,
    poll_id: int,
    run_id: str,
    rows: list[Any],
    *,
    local_response_end_ms: int,
) -> int:
    """Same shape as :func:`train_v2.normalize.normalize_gtfsrt_trip_updates`."""
    n = 0
    for r in rows:
        arrival_ms = _dt_to_ms(getattr(r, "arrival_time", None))
        departure_ms = _dt_to_ms(getattr(r, "departure_time", None))
        feed_ts_ms = _dt_to_ms(getattr(r, "feed_timestamp", None))
        tu_ts_ms = _dt_to_ms(getattr(r, "trip_update_timestamp", None))
        conn.execute(
            """
            INSERT INTO pace_gtfsrt_trip_update(
                poll_id, run_id, local_response_end_ms,
                feed_timestamp_ms, feed_incrementality,
                trip_update_timestamp_ms, trip_update_delay_seconds,
                route_id, trip_id, trip_start_date, trip_start_time,
                trip_direction_id, trip_schedule_relationship,
                stop_id, stop_sequence,
                arrival_time_ms, arrival_delay_seconds, arrival_uncertainty_seconds,
                departure_time_ms, departure_delay_seconds, departure_uncertainty_seconds,
                schedule_relationship,
                vehicle_id, vehicle_label, vehicle_license_plate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                poll_id, run_id, local_response_end_ms,
                feed_ts_ms, getattr(r, "feed_incrementality", None),
                tu_ts_ms, getattr(r, "trip_update_delay_seconds", None),
                getattr(r, "route_id", None), getattr(r, "trip_id", None),
                getattr(r, "trip_start_date", None), getattr(r, "trip_start_time", None),
                getattr(r, "trip_direction_id", None),
                getattr(r, "trip_schedule_relationship", None),
                getattr(r, "stop_id", None), getattr(r, "stop_sequence", None),
                arrival_ms, getattr(r, "arrival_delay_seconds", None),
                getattr(r, "arrival_uncertainty_seconds", None),
                departure_ms, getattr(r, "departure_delay_seconds", None),
                getattr(r, "departure_uncertainty_seconds", None),
                getattr(r, "schedule_relationship", None),
                getattr(r, "vehicle_id", None), getattr(r, "vehicle_label", None),
                getattr(r, "vehicle_license_plate", None),
            ],
        )
        n += 1
    return n


def normalize_pace_vehicle_positions(
    conn: duckdb.DuckDBPyConnection,
    poll_id: int,
    run_id: str,
    rows: list[Any],
    *,
    local_response_end_ms: int,
) -> int:
    n = 0
    for r in rows:
        feed_ts_ms = _dt_to_ms(getattr(r, "feed_timestamp", None))
        vehicle_ts_ms = _dt_to_ms(getattr(r, "vehicle_timestamp", None))
        conn.execute(
            """
            INSERT INTO pace_gtfsrt_vehicle_position(
                poll_id, run_id, local_response_end_ms,
                feed_timestamp_ms, feed_incrementality,
                vehicle_timestamp_ms, route_id, trip_id,
                trip_start_date, trip_start_time, trip_direction_id, trip_schedule_relationship,
                vehicle_id, vehicle_label, vehicle_license_plate,
                stop_id, current_stop_sequence, current_status,
                congestion_level, occupancy_status, occupancy_percentage,
                multi_carriage_details_json, lat, lon, bearing, speed_mps, odometer_m
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                poll_id, run_id, local_response_end_ms,
                feed_ts_ms, getattr(r, "feed_incrementality", None),
                vehicle_ts_ms,
                getattr(r, "route_id", None), getattr(r, "trip_id", None),
                getattr(r, "trip_start_date", None), getattr(r, "trip_start_time", None),
                getattr(r, "trip_direction_id", None),
                getattr(r, "trip_schedule_relationship", None),
                getattr(r, "vehicle_id", None), getattr(r, "vehicle_label", None),
                getattr(r, "vehicle_license_plate", None),
                getattr(r, "stop_id", None),
                getattr(r, "current_stop_sequence", None),
                getattr(r, "current_status", None),
                getattr(r, "congestion_level", None),
                getattr(r, "occupancy_status", None),
                getattr(r, "occupancy_percentage", None),
                getattr(r, "multi_carriage_details_json", None),
                getattr(r, "lat", None), getattr(r, "lon", None),
                getattr(r, "bearing", None), getattr(r, "speed_mps", None),
                getattr(r, "odometer_m", None),
            ],
        )
        n += 1
    return n


def normalize_pace_alerts(
    conn: duckdb.DuckDBPyConnection,
    poll_id: int,
    run_id: str,
    rows: list[Any],
    *,
    local_response_end_ms: int,
) -> int:
    n = 0
    for r in rows:
        feed_ts_ms = _dt_to_ms(getattr(r, "feed_timestamp", None))
        conn.execute(
            """
            INSERT INTO pace_gtfsrt_alert(
                poll_id, run_id, local_response_end_ms,
                feed_timestamp_ms, feed_incrementality,
                entity_id, cause, effect, severity_level,
                header_text, description_text, tts_header_text, tts_description_text, url,
                active_period_json, informed_entity_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                poll_id, run_id, local_response_end_ms,
                feed_ts_ms, getattr(r, "feed_incrementality", None),
                getattr(r, "entity_id", None),
                getattr(r, "cause", None), getattr(r, "effect", None),
                getattr(r, "severity_level", None),
                getattr(r, "header_text", None), getattr(r, "description_text", None),
                getattr(r, "tts_header_text", None), getattr(r, "tts_description_text", None),
                getattr(r, "url", None),
                getattr(r, "active_period_json", None),
                getattr(r, "informed_entity_json", None),
            ],
        )
        n += 1
    return n
