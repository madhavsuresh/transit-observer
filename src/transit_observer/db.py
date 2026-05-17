"""DuckDB schema and connection helpers.

Single-writer pattern: only the collector opens a writable connection on
``settings.db_path``. Everything else reads ``settings.read_replica_path``,
which the collector refreshes every ``read_replica_refresh_seconds``.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

from .config import settings


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS train_arrivals_raw (
    polled_at        TIMESTAMPTZ NOT NULL,
    line             TEXT        NOT NULL,
    run_number       TEXT        NOT NULL,
    map_id           INTEGER     NOT NULL,
    stop_id          INTEGER     NOT NULL,
    station_name     TEXT,
    direction_code   TEXT,
    destination_name TEXT,
    predicted_at     TIMESTAMPTZ,
    arrival_at       TIMESTAMPTZ,
    is_approaching   BOOLEAN,
    is_delayed       BOOLEAN,
    is_fault         BOOLEAN,
    is_scheduled     BOOLEAN,
    flags            TEXT,
    stop_description TEXT
);

CREATE INDEX IF NOT EXISTS idx_arrivals_polled ON train_arrivals_raw(polled_at);
CREATE INDEX IF NOT EXISTS idx_arrivals_run_station ON train_arrivals_raw(run_number, line, map_id);

CREATE TABLE IF NOT EXISTS train_positions_raw (
    polled_at           TIMESTAMPTZ NOT NULL,
    line                TEXT NOT NULL,
    run_number          TEXT NOT NULL,
    destination_name    TEXT,
    direction_code      TEXT,
    next_station_map_id INTEGER,
    next_station_name   TEXT,
    predicted_at        TIMESTAMPTZ,
    next_arrival_at     TIMESTAMPTZ,
    is_approaching      BOOLEAN,
    is_delayed          BOOLEAN,
    lat                 DOUBLE,
    lon                 DOUBLE,
    heading             DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_positions_polled ON train_positions_raw(polled_at);
CREATE INDEX IF NOT EXISTS idx_positions_run_next ON train_positions_raw(line, run_number, next_station_map_id);

CREATE TABLE IF NOT EXISTS train_runs_observed (
    line                TEXT        NOT NULL,
    run_number          TEXT        NOT NULL,
    map_id              INTEGER     NOT NULL,
    direction_code      TEXT,
    destination_name    TEXT,
    observed_arrival_at TIMESTAMPTZ NOT NULL,
    first_seen_at       TIMESTAMPTZ,
    last_seen_at        TIMESTAMPTZ,
    sample_count        INTEGER,
    inferred_from       TEXT,       -- 'approaching' | 'dropoff'
    PRIMARY KEY (line, run_number, map_id, observed_arrival_at)
);

CREATE INDEX IF NOT EXISTS idx_runs_line_station ON train_runs_observed(line, map_id);

CREATE TABLE IF NOT EXISTS forecast_queue (
    forecast_id              TEXT PRIMARY KEY,
    enqueued_at              TIMESTAMPTZ NOT NULL,
    snapshot_polled_at       TIMESTAMPTZ NOT NULL,
    leave_at                 TIMESTAMPTZ NOT NULL,
    mode                     TEXT NOT NULL DEFAULT 'L',  -- L | bus | metra | intercampus
    line                     TEXT NOT NULL,              -- L line code / bus route / metra route_id / 'intercampus'
    direction_code           TEXT,
    corridor_id              TEXT,                       -- FK to corridors.corridor_id (NULL for legacy rows)
    predictor_version        TEXT,                       -- semver-ish hash for A/B between predictors
    feature_json             TEXT,                       -- feature snapshot at prediction time (JSON)
    boarding_map_id          INTEGER NOT NULL DEFAULT 0, -- L only
    boarding_text_id         TEXT,                       -- non-L modes use this
    boarding_station_name    TEXT,
    alighting_map_id         INTEGER NOT NULL DEFAULT 0, -- L only
    alighting_text_id        TEXT,                       -- non-L modes use this
    alighting_station_name   TEXT,
    predicted_wait_mean      DOUBLE,
    predicted_wait_p50       DOUBLE,
    predicted_wait_p80       DOUBLE,
    predicted_wait_p90       DOUBLE,
    predicted_in_vehicle_mean DOUBLE,
    predicted_total_mean     DOUBLE,
    predicted_total_p50      DOUBLE,
    predicted_total_p80      DOUBLE,
    predicted_total_p90      DOUBLE,
    predicted_failure_prob   DOUBLE,
    resolve_after            TIMESTAMPTZ NOT NULL,
    status                   TEXT NOT NULL DEFAULT 'pending'  -- pending | resolved | unresolvable
);

CREATE INDEX IF NOT EXISTS idx_forecast_status_resolve ON forecast_queue(status, resolve_after);
CREATE INDEX IF NOT EXISTS idx_forecast_corridor ON forecast_queue(corridor_id);

CREATE TABLE IF NOT EXISTS corridors (
    corridor_id              TEXT PRIMARY KEY,
    mode                     TEXT NOT NULL,
    line                     TEXT NOT NULL,
    direction                TEXT NOT NULL,
    origin_label             TEXT NOT NULL,
    origin_latitude          DOUBLE NOT NULL,
    origin_longitude         DOUBLE NOT NULL,
    destination_label        TEXT NOT NULL,
    destination_latitude     DOUBLE NOT NULL,
    destination_longitude    DOUBLE NOT NULL,
    boarding_int_id          INTEGER NOT NULL DEFAULT 0,
    boarding_text_id         TEXT,
    alighting_int_id         INTEGER NOT NULL DEFAULT 0,
    alighting_text_id        TEXT,
    schedule_headway_seconds DOUBLE NOT NULL,
    cadence_seconds          DOUBLE NOT NULL DEFAULT 300,
    priority                 INTEGER NOT NULL DEFAULT 5,
    is_active                BOOLEAN NOT NULL DEFAULT TRUE,
    seeded_at                TIMESTAMPTZ NOT NULL,
    last_predicted_at        TIMESTAMPTZ,
    source                   TEXT NOT NULL DEFAULT 'seed',  -- 'seed' | 'auto_upgraded'
    promoted_from_query_count INTEGER                       -- only set when source='auto_upgraded'
);

CREATE INDEX IF NOT EXISTS idx_corridors_mode ON corridors(mode, is_active);

CREATE TABLE IF NOT EXISTS query_log (
    query_id                 TEXT PRIMARY KEY,
    queried_at               TIMESTAMPTZ NOT NULL,
    client_id                TEXT,
    mode                     TEXT NOT NULL,
    line                     TEXT NOT NULL,
    direction_code           TEXT,
    boarding_int_id          INTEGER NOT NULL DEFAULT 0,
    boarding_text_id         TEXT,
    boarding_station_name    TEXT,
    alighting_int_id         INTEGER NOT NULL DEFAULT 0,
    alighting_text_id        TEXT,
    alighting_station_name   TEXT,
    predicted_wait_mean      DOUBLE,
    predicted_wait_p50       DOUBLE,
    predicted_wait_p80       DOUBLE,
    predicted_wait_p90       DOUBLE,
    predicted_in_vehicle_mean DOUBLE,
    predicted_total_p50      DOUBLE,
    predicted_total_p80      DOUBLE,
    predicted_total_p90      DOUBLE,
    predictor_version        TEXT,
    success                  BOOLEAN NOT NULL,
    error_reason             TEXT
);

CREATE INDEX IF NOT EXISTS idx_query_log_at ON query_log(queried_at);
CREATE INDEX IF NOT EXISTS idx_query_log_od ON query_log(mode, line, boarding_int_id, boarding_text_id, alighting_int_id, alighting_text_id);

CREATE TABLE IF NOT EXISTS metra_arrivals_raw (
    polled_at              TIMESTAMPTZ NOT NULL,
    route_id               TEXT NOT NULL,
    trip_id                TEXT NOT NULL,
    station_id             TEXT NOT NULL,
    direction_id           INTEGER,
    schedule_relationship  TEXT,
    scheduled_at           TIMESTAMPTZ,
    predicted_at           TIMESTAMPTZ,
    delay_seconds          INTEGER
);

CREATE INDEX IF NOT EXISTS idx_metra_polled ON metra_arrivals_raw(polled_at);
CREATE INDEX IF NOT EXISTS idx_metra_route_station ON metra_arrivals_raw(route_id, station_id);

CREATE TABLE IF NOT EXISTS intercampus_arrivals_raw (
    polled_at         TIMESTAMPTZ NOT NULL,
    route_id          TEXT NOT NULL,
    trip_id           TEXT NOT NULL,
    direction         TEXT,
    stop_id           TEXT NOT NULL,
    stop_name         TEXT,
    destination_name  TEXT,
    predicted_at      TIMESTAMPTZ,
    arrival_at        TIMESTAMPTZ,
    delay_seconds     INTEGER,
    is_delayed        BOOLEAN,
    time_source       TEXT
);

CREATE INDEX IF NOT EXISTS idx_intercampus_polled ON intercampus_arrivals_raw(polled_at);
CREATE INDEX IF NOT EXISTS idx_intercampus_stop ON intercampus_arrivals_raw(direction, stop_id);

CREATE TABLE IF NOT EXISTS bus_predictions_raw (
    polled_at          TIMESTAMPTZ NOT NULL,
    route              TEXT NOT NULL,
    route_name         TEXT,
    vehicle_id         TEXT,
    stop_id            INTEGER NOT NULL,
    stop_name          TEXT,
    destination_name   TEXT,
    direction_name     TEXT,
    generated_at       TIMESTAMPTZ,
    arrival_at         TIMESTAMPTZ,
    is_delayed         BOOLEAN,
    is_approaching     BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_bus_polled ON bus_predictions_raw(polled_at);
CREATE INDEX IF NOT EXISTS idx_bus_route_stop ON bus_predictions_raw(route, stop_id);

CREATE TABLE IF NOT EXISTS bus_positions_raw (
    polled_at         TIMESTAMPTZ NOT NULL,
    route             TEXT NOT NULL,
    vehicle_id        TEXT NOT NULL,
    vehicle_timestamp TIMESTAMPTZ,
    lat               DOUBLE,
    lon               DOUBLE,
    heading           DOUBLE,
    speed_mph         DOUBLE,
    pattern_id        INTEGER,
    pattern_distance  DOUBLE,
    trip_id           TEXT,
    block_id          TEXT,
    destination       TEXT,
    is_delayed        BOOLEAN,
    zone              TEXT
);

CREATE INDEX IF NOT EXISTS idx_bus_positions_polled ON bus_positions_raw(polled_at);
CREATE INDEX IF NOT EXISTS idx_bus_positions_route_vehicle
    ON bus_positions_raw(route, vehicle_id, polled_at);

CREATE TABLE IF NOT EXISTS bus_runs_observed (
    route               TEXT NOT NULL,
    vehicle_id          TEXT NOT NULL,
    stop_id             INTEGER NOT NULL,
    destination_name    TEXT,
    direction_name      TEXT,
    observed_arrival_at TIMESTAMPTZ NOT NULL,
    first_seen_at       TIMESTAMPTZ,
    last_seen_at        TIMESTAMPTZ,
    sample_count        INTEGER,
    inferred_from       TEXT,
    PRIMARY KEY (route, vehicle_id, stop_id, observed_arrival_at)
);

CREATE INDEX IF NOT EXISTS idx_bus_runs_route_stop ON bus_runs_observed(route, stop_id);

CREATE TABLE IF NOT EXISTS metra_trips_observed (
    route_id            TEXT NOT NULL,
    trip_id             TEXT NOT NULL,
    station_id          TEXT NOT NULL,
    direction_id        INTEGER,
    observed_arrival_at TIMESTAMPTZ NOT NULL,
    first_seen_at       TIMESTAMPTZ,
    last_seen_at        TIMESTAMPTZ,
    sample_count        INTEGER,
    inferred_from       TEXT,
    PRIMARY KEY (route_id, trip_id, station_id, observed_arrival_at)
);

CREATE INDEX IF NOT EXISTS idx_metra_trips_route_station ON metra_trips_observed(route_id, station_id);

CREATE TABLE IF NOT EXISTS intercampus_trips_observed (
    route_id            TEXT NOT NULL,
    trip_id             TEXT NOT NULL,
    stop_id             TEXT NOT NULL,
    direction           TEXT,
    observed_arrival_at TIMESTAMPTZ NOT NULL,
    first_seen_at       TIMESTAMPTZ,
    last_seen_at        TIMESTAMPTZ,
    sample_count        INTEGER,
    inferred_from       TEXT,
    PRIMARY KEY (route_id, trip_id, stop_id, observed_arrival_at)
);

CREATE INDEX IF NOT EXISTS idx_intercampus_trips_stop ON intercampus_trips_observed(direction, stop_id);

CREATE TABLE IF NOT EXISTS direction_audit (
    forecast_id                       TEXT PRIMARY KEY,
    mode                              TEXT NOT NULL DEFAULT 'L',
    audited_at                        TIMESTAMPTZ NOT NULL,
    candidate_arrivals_count          INTEGER NOT NULL,
    kept_arrivals_count               INTEGER NOT NULL,
    kept_direction_codes              TEXT,
    kept_destination_names            TEXT,
    boarded_direction_code            TEXT,
    boarded_destination_name          TEXT,
    boarded_was_kept                  BOOLEAN,
    kept_matching_boarded_direction   INTEGER,
    notes                             TEXT
);

CREATE TABLE IF NOT EXISTS forecast_outcomes (
    forecast_id              TEXT PRIMARY KEY,
    resolved_at              TIMESTAMPTZ NOT NULL,
    boarded_run_number       TEXT,
    boarded_at               TIMESTAMPTZ,
    alighted_at              TIMESTAMPTZ,
    actual_wait_seconds      DOUBLE,
    actual_in_vehicle_seconds DOUBLE,
    actual_total_seconds     DOUBLE,
    in_p80_window            BOOLEAN,
    in_p90_window            BOOLEAN,
    p50_residual_seconds     DOUBLE,
    p80_residual_seconds     DOUBLE,
    truth_confidence         DOUBLE,        -- 0.0..1.0: how cleanly the snapshots bracket the boarded run
    failed                   BOOLEAN,
    notes                    TEXT
);

-- DtACI / EB-shrinkage online state. One row per (predictor_version, line,
-- direction_code, leg, quantile). The resolver updates `offset_seconds`
-- after each scored outcome so the live predictor can adjust its raw
-- quantile output toward the empirical target coverage.
CREATE TABLE IF NOT EXISTS predictor_state (
    predictor_version  TEXT NOT NULL,
    line               TEXT NOT NULL,
    direction_code     TEXT NOT NULL,
    leg                TEXT NOT NULL,           -- 'wait' | 'in_vehicle' | 'total' | 'residual'
    quantile           DOUBLE NOT NULL,         -- 0.5 | 0.8 | 0.9 | -1 for mean-shift
    offset_seconds     DOUBLE NOT NULL DEFAULT 0.0,
    step_size          DOUBLE NOT NULL DEFAULT 0.01,
    coverage_target    DOUBLE NOT NULL DEFAULT 0.8,
    coverage_observed  DOUBLE,
    n_observations     BIGINT  NOT NULL DEFAULT 0,
    residual_mean      DOUBLE,                  -- running mean of residuals (for EB)
    residual_var       DOUBLE,                  -- running variance of residuals
    updated_at         TIMESTAMPTZ,
    PRIMARY KEY (predictor_version, line, direction_code, leg, quantile)
);

-- Which predictor is "active" for a given corridor. Empty for corridors
-- still on the bootstrap kernel. Filled by the registry's promote() with
-- anti-flap switch-margin logic.
CREATE TABLE IF NOT EXISTS predictor_active (
    corridor_id        TEXT PRIMARY KEY,
    predictor_version  TEXT NOT NULL,
    decided_at         TIMESTAMPTZ NOT NULL,
    decided_score      DOUBLE,                  -- CRPS + alpha*coverage_gap at decision
    incumbent_score    DOUBLE,                  -- score of the predictor it displaced
    margin             DOUBLE,                  -- decided_score - incumbent_score (negative = improvement)
    n_consecutive_wins INTEGER NOT NULL DEFAULT 1,
    pending_candidate  TEXT,                    -- challenger currently winning, not yet promoted
    pending_wins       INTEGER NOT NULL DEFAULT 0,
    pending_score      DOUBLE
);

-- Trained model artifacts on disk. Multiple rows per predictor_version
-- (one per leg x line). Only the trainer writes here.
CREATE TABLE IF NOT EXISTS model_artifacts (
    predictor_version  TEXT NOT NULL,
    leg                TEXT NOT NULL,           -- 'wait' | 'in_vehicle' | 'total'
    line               TEXT NOT NULL,           -- 'ALL' = global model, else line code
    quantile           DOUBLE NOT NULL,
    artifact_path      TEXT NOT NULL,
    trained_at         TIMESTAMPTZ NOT NULL,
    n_train_rows       BIGINT,
    n_val_rows         BIGINT,
    val_pinball_loss   DOUBLE,
    val_crps           DOUBLE,
    feature_columns    TEXT,                    -- JSON list, for schema-drift detection
    PRIMARY KEY (predictor_version, leg, line, quantile)
);

-- Local sports schedules (ESPN). One row per (team-poll, event). First
-- snapshot to see ``completed=true`` approximates the game-end time.
CREATE TABLE IF NOT EXISTS sports_events_raw (
    snapshot_polled_at TIMESTAMPTZ NOT NULL,
    event_id           TEXT NOT NULL,
    league             TEXT NOT NULL,
    sport              TEXT,
    home_team          TEXT,
    away_team          TEXT,
    venue              TEXT,
    scheduled_start    TIMESTAMPTZ,
    status             TEXT,
    completed          BOOLEAN,
    attendance         INTEGER,
    home_score         INTEGER,
    away_score         INTEGER,
    raw_payload_json   TEXT
);

CREATE INDEX IF NOT EXISTS idx_sports_polled ON sports_events_raw(snapshot_polled_at);
CREATE INDEX IF NOT EXISTS idx_sports_event
    ON sports_events_raw(league, event_id, snapshot_polled_at);

-- Venue events (Ticketmaster Discovery API). Concerts and major events
-- at music + sports venues.
CREATE TABLE IF NOT EXISTS venue_events_raw (
    snapshot_polled_at TIMESTAMPTZ NOT NULL,
    event_id           TEXT NOT NULL,
    name               TEXT,
    venue_name         TEXT,
    venue_city         TEXT,
    scheduled_start    TIMESTAMPTZ,
    sales_start        TIMESTAMPTZ,
    sales_end          TIMESTAMPTZ,
    classification     TEXT,
    genre              TEXT,
    url                TEXT,
    raw_payload_json   TEXT
);

CREATE INDEX IF NOT EXISTS idx_venue_polled ON venue_events_raw(snapshot_polled_at);
CREATE INDEX IF NOT EXISTS idx_venue_event
    ON venue_events_raw(event_id, snapshot_polled_at);

-- Chicago Open Data snapshots. One row per record per poll.
CREATE TABLE IF NOT EXISTS chicago_open_data_raw (
    snapshot_polled_at TIMESTAMPTZ NOT NULL,
    dataset            TEXT NOT NULL,    -- our local label, e.g. 'street_closures'
    table_id           TEXT NOT NULL,    -- Socrata id, e.g. 'dhk3-bs2g'
    record_id          TEXT,             -- Socrata `:id`
    raw_payload_json   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chi_open_polled ON chicago_open_data_raw(snapshot_polled_at);
CREATE INDEX IF NOT EXISTS idx_chi_open_dataset ON chicago_open_data_raw(dataset, snapshot_polled_at);

-- Weather observations (Open-Meteo). One row per (site, poll).
CREATE TABLE IF NOT EXISTS weather_observations_raw (
    polled_at              TIMESTAMPTZ NOT NULL,
    site_id                TEXT NOT NULL,
    lat                    DOUBLE NOT NULL,
    lon                    DOUBLE NOT NULL,
    observation_time       TIMESTAMPTZ,
    temperature_c          DOUBLE,
    apparent_temperature_c DOUBLE,
    humidity_pct           DOUBLE,
    precipitation_mm       DOUBLE,
    rain_mm                DOUBLE,
    snowfall_cm            DOUBLE,
    wind_speed_kph         DOUBLE,
    wind_gust_kph          DOUBLE,
    wind_direction_deg     DOUBLE,
    cloud_cover_pct        DOUBLE,
    pressure_hpa           DOUBLE,
    weather_code           INTEGER,
    source                 TEXT NOT NULL DEFAULT 'open_meteo'
);

CREATE INDEX IF NOT EXISTS idx_weather_polled ON weather_observations_raw(polled_at);
CREATE INDEX IF NOT EXISTS idx_weather_site ON weather_observations_raw(site_id, polled_at);

-- Air-quality observations (AirNow). One row per (site, poll).
CREATE TABLE IF NOT EXISTS air_quality_raw (
    polled_at        TIMESTAMPTZ NOT NULL,
    site_id          TEXT NOT NULL,        -- zip code or site name
    parameter        TEXT NOT NULL,        -- 'pm2.5' | 'pm10' | 'ozone' | 'co' | 'no2' | 'so2'
    aqi              INTEGER,
    raw_value        DOUBLE,
    unit             TEXT,
    category         TEXT,                 -- 'Good' | 'Moderate' | ...
    observation_time TIMESTAMPTZ,
    reporting_area   TEXT,
    latitude         DOUBLE,
    longitude        DOUBLE,
    source           TEXT NOT NULL DEFAULT 'airnow'
);

CREATE INDEX IF NOT EXISTS idx_aqi_polled ON air_quality_raw(polled_at);
CREATE INDEX IF NOT EXISTS idx_aqi_site ON air_quality_raw(site_id, polled_at);

-- GTFS-static feed version archive. One row per unique content hash
-- per agency. Zips are stored on disk under ``data/gtfs_snapshots/{agency}/``
-- (out of the DB) since they can be multi-MB each.
CREATE TABLE IF NOT EXISTS gtfs_feed_versions (
    agency        TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    downloaded_at TIMESTAMPTZ NOT NULL,
    file_size     BIGINT,
    feed_version  TEXT,           -- from feed_info.txt if present
    source_url    TEXT,
    archive_path  TEXT,
    PRIMARY KEY (agency, sha256)
);

CREATE INDEX IF NOT EXISTS idx_gtfs_versions_agency
    ON gtfs_feed_versions(agency, downloaded_at);

-- CTA GTFS-Realtime TripUpdates (train + bus). Parallel to the
-- proprietary ttarrivals/Bus Tracker feeds. Gives canonical trip_id
-- (joinable to GTFS-static) plus per-stop delay. One row per
-- stop_time_update.
CREATE TABLE IF NOT EXISTS cta_gtfsrt_trip_updates_raw (
    polled_at                TIMESTAMPTZ NOT NULL,
    mode                     TEXT NOT NULL,    -- 'train' | 'bus'
    route_id                 TEXT,
    trip_id                  TEXT,
    stop_id                  TEXT,
    stop_sequence            INTEGER,
    arrival_time             TIMESTAMPTZ,
    arrival_delay_seconds    INTEGER,
    departure_time           TIMESTAMPTZ,
    departure_delay_seconds  INTEGER,
    schedule_relationship    TEXT,
    vehicle_id               TEXT
);

CREATE INDEX IF NOT EXISTS idx_cta_gtfsrt_tu_polled
    ON cta_gtfsrt_trip_updates_raw(polled_at);
CREATE INDEX IF NOT EXISTS idx_cta_gtfsrt_tu_trip
    ON cta_gtfsrt_trip_updates_raw(mode, route_id, trip_id);

-- CTA GTFS-Realtime VehiclePositions. Parallel to ttpositions for trains
-- but with trip_id binding + occupancy/congestion fields where published.
CREATE TABLE IF NOT EXISTS cta_gtfsrt_vehicle_positions_raw (
    polled_at             TIMESTAMPTZ NOT NULL,
    mode                  TEXT NOT NULL,    -- 'train' | 'bus'
    route_id              TEXT,
    trip_id               TEXT,
    vehicle_id            TEXT,
    vehicle_label         TEXT,
    lat                   DOUBLE,
    lon                   DOUBLE,
    bearing               DOUBLE,
    speed_mps             DOUBLE,
    current_stop_sequence INTEGER,
    current_status        TEXT,
    congestion_level      TEXT,
    occupancy_status      TEXT
);

CREATE INDEX IF NOT EXISTS idx_cta_gtfsrt_vp_polled
    ON cta_gtfsrt_vehicle_positions_raw(polled_at);
CREATE INDEX IF NOT EXISTS idx_cta_gtfsrt_vp_vehicle
    ON cta_gtfsrt_vehicle_positions_raw(mode, vehicle_id, polled_at);

-- Transit-account social media posts (Bluesky, Mastodon). Captured
-- forward-only — historical timeline search is paywalled / unreliable.
CREATE TABLE IF NOT EXISTS transit_social_raw (
    polled_at        TIMESTAMPTZ NOT NULL,
    platform         TEXT NOT NULL,    -- 'bluesky' | 'mastodon'
    handle           TEXT NOT NULL,
    post_id          TEXT NOT NULL,
    posted_at        TIMESTAMPTZ,
    body             TEXT,
    url              TEXT,
    in_reply_to      TEXT,
    media_urls_json  TEXT,
    raw_payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_social_polled ON transit_social_raw(polled_at);
CREATE INDEX IF NOT EXISTS idx_social_post ON transit_social_raw(platform, handle, post_id);

-- CTA service alerts. Snapshotted on every poll (no de-duplication) so
-- we can reconstruct the active alert set at any historical timestamp.
-- The live feed has no public archive.
CREATE TABLE IF NOT EXISTS cta_alerts_raw (
    polled_at              TIMESTAMPTZ NOT NULL,
    alert_id               TEXT NOT NULL,
    headline               TEXT,
    short_description      TEXT,
    full_description       TEXT,
    severity_score         INTEGER,
    impact                 TEXT,
    event_start            TIMESTAMPTZ,
    event_end              TIMESTAMPTZ,
    tbd                    BOOLEAN,
    major_alert            BOOLEAN,
    alert_url              TEXT,
    impacted_services_json TEXT,
    guid                   TEXT
);

CREATE INDEX IF NOT EXISTS idx_cta_alerts_polled ON cta_alerts_raw(polled_at);
CREATE INDEX IF NOT EXISTS idx_cta_alerts_id ON cta_alerts_raw(alert_id, polled_at);

-- One row per HTTP call to any external API. Stores the raw response body
-- so future feature extraction can re-parse fields we currently drop,
-- without needing to re-poll (impossible for time-of-day signal).
-- API keys are scrubbed from request_params_json before storage.
CREATE TABLE IF NOT EXISTS api_payloads_raw (
    polled_at           TIMESTAMPTZ NOT NULL,
    source              TEXT NOT NULL,        -- e.g. 'cta_train_arrivals', 'cta_bus_predictions', 'weather_open_meteo'
    endpoint            TEXT NOT NULL,        -- URL path
    request_params_json TEXT,
    response_body       TEXT,
    http_status         INTEGER,
    latency_ms          DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_payloads_polled ON api_payloads_raw(polled_at);
CREATE INDEX IF NOT EXISTS idx_payloads_source ON api_payloads_raw(source, polled_at);

CREATE INDEX IF NOT EXISTS idx_predictor_state_v
    ON predictor_state(predictor_version, line, direction_code);
CREATE INDEX IF NOT EXISTS idx_arrivals_run_polled
    ON train_arrivals_raw(line, run_number, polled_at);
CREATE INDEX IF NOT EXISTS idx_positions_run_polled
    ON train_positions_raw(line, run_number, polled_at);
CREATE INDEX IF NOT EXISTS idx_outcomes_predictor
    ON forecast_queue(predictor_version, line, direction_code, leave_at);

-- ============================================================
-- CTA Bus Tracker v3 parallel pipeline.
-- See src/transit_observer/bus_v3/ for the ingest + estimator code.
-- All timestamps are stored as BIGINT ms epochs (CTA server-native;
-- avoids tz conversions in hot inference loops). All stop IDs are
-- TEXT to match the API surface (v2 INTEGER stop_id tables live in
-- a parallel namespace and are not migrated).
-- ============================================================

CREATE SEQUENCE IF NOT EXISTS bus_v3_api_poll_seq;
CREATE TABLE IF NOT EXISTS bus_v3_api_poll (
    poll_id                 BIGINT PRIMARY KEY DEFAULT nextval('bus_v3_api_poll_seq'),
    run_id                  TEXT NOT NULL,
    cycle_index             INTEGER,
    endpoint                TEXT NOT NULL,
    query_kind              TEXT,
    request_url_redacted    TEXT,
    params_json_redacted    TEXT NOT NULL,
    local_request_start_ms  BIGINT NOT NULL,
    local_response_end_ms   BIGINT NOT NULL,
    cta_server_time_ms      BIGINT,
    http_status             INTEGER,
    latency_ms              DOUBLE,
    ok                      BOOLEAN NOT NULL DEFAULT FALSE,
    error_message           TEXT,
    raw_json                TEXT,
    raw_sha256              TEXT,
    created_at_ms           BIGINT NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS bus_v3_api_error_seq;
CREATE TABLE IF NOT EXISTS bus_v3_api_error (
    api_error_id    BIGINT PRIMARY KEY DEFAULT nextval('bus_v3_api_error_seq'),
    poll_id         BIGINT NOT NULL,
    endpoint        TEXT NOT NULL,
    rt              TEXT,
    stpid           TEXT,
    vid             TEXT,
    msg             TEXT NOT NULL,
    raw_json        TEXT
);

CREATE TABLE IF NOT EXISTS bus_v3_route (
    rt                  TEXT PRIMARY KEY,
    rtnm                TEXT,
    rtclr               TEXT,
    rtdd                TEXT,
    first_seen_poll_id  BIGINT,
    last_seen_poll_id   BIGINT,
    raw_json            TEXT
);

CREATE TABLE IF NOT EXISTS bus_v3_direction (
    rt                  TEXT NOT NULL,
    dir_id              TEXT NOT NULL,
    name                TEXT,
    first_seen_poll_id  BIGINT,
    last_seen_poll_id   BIGINT,
    raw_json            TEXT,
    PRIMARY KEY (rt, dir_id)
);

CREATE TABLE IF NOT EXISTS bus_v3_stop (
    stpid               TEXT PRIMARY KEY,
    stpnm               TEXT,
    lat                 DOUBLE,
    lon                 DOUBLE,
    rt                  TEXT,
    rtdir               TEXT,
    dtradd_json         TEXT,
    dtrrem_json         TEXT,
    first_seen_poll_id  BIGINT,
    last_seen_poll_id   BIGINT,
    raw_json            TEXT
);

CREATE TABLE IF NOT EXISTS bus_v3_pattern (
    pid                 INTEGER PRIMARY KEY,
    rt                  TEXT,
    rtdir               TEXT,
    length_ft           DOUBLE,
    dtrid               TEXT,
    first_seen_poll_id  BIGINT,
    last_seen_poll_id   BIGINT,
    raw_json            TEXT
);

CREATE TABLE IF NOT EXISTS bus_v3_pattern_point (
    pid                         INTEGER NOT NULL,
    seq                         INTEGER NOT NULL,
    typ                         TEXT,
    stpid                       TEXT,
    stpnm                       TEXT,
    lat                         DOUBLE,
    lon                         DOUBLE,
    pdist_ft                    DOUBLE,
    is_detour_original_point    INTEGER DEFAULT 0,
    raw_json                    TEXT,
    PRIMARY KEY (pid, seq, is_detour_original_point)
);

CREATE TABLE IF NOT EXISTS bus_v3_detour (
    detour_pk           TEXT PRIMARY KEY,
    id                  TEXT NOT NULL,
    ver                 INTEGER,
    state               INTEGER,
    descr               TEXT,
    route_dirs_json     TEXT,
    startdt_ms          BIGINT,
    enddt_ms            BIGINT,
    moddt_ms            BIGINT,
    first_seen_poll_id  BIGINT,
    last_seen_poll_id   BIGINT,
    raw_json            TEXT
);

CREATE TABLE IF NOT EXISTS bus_v3_enhanced_detour_pattern (
    detour_pk           TEXT NOT NULL,
    origpid             INTEGER NOT NULL,
    dtrpid              INTEGER NOT NULL,
    encoded_polyline    TEXT,
    delay_s             INTEGER,
    raw_json            TEXT,
    PRIMARY KEY (detour_pk, origpid, dtrpid)
);

CREATE TABLE IF NOT EXISTS bus_v3_enhanced_detour_trip (
    detour_pk       TEXT NOT NULL,
    tripid          TEXT,
    tatripid        TEXT,
    origtatripno    TEXT,
    dates_json      TEXT,
    stst            INTEGER,
    raw_json        TEXT,
    PRIMARY KEY (detour_pk, tripid, tatripid, origtatripno, stst)
);

CREATE TABLE IF NOT EXISTS bus_v3_enhanced_detour_replacement_stop (
    detour_pk           TEXT NOT NULL,
    role                TEXT NOT NULL,    -- 'start' | 'end' | 'replacement'
    geoid               TEXT,
    stpid               TEXT,
    seq                 INTEGER,
    stpnm               TEXT,
    lat                 DOUBLE,
    lon                 DOUBLE,
    adhoc               INTEGER,
    relpasstime_s       INTEGER,
    raw_json            TEXT,
    PRIMARY KEY (detour_pk, role, stpid, seq)
);

CREATE SEQUENCE IF NOT EXISTS bus_v3_vehicle_observation_seq;
CREATE TABLE IF NOT EXISTS bus_v3_vehicle_observation (
    vehicle_obs_id          BIGINT PRIMARY KEY DEFAULT nextval('bus_v3_vehicle_observation_seq'),
    poll_id                 BIGINT NOT NULL,
    run_id                  TEXT NOT NULL,
    cta_server_time_ms      BIGINT,
    local_response_end_ms   BIGINT NOT NULL,
    vid                     TEXT NOT NULL,
    tmstmp_ms               BIGINT,
    vehicle_age_s           DOUBLE,
    lat                     DOUBLE,
    lon                     DOUBLE,
    hdg                     DOUBLE,
    pid                     INTEGER,
    pdist_ft                DOUBLE,
    rt                      TEXT,
    des                     TEXT,
    dly                     INTEGER,
    tablockid               TEXT,
    tatripid                TEXT,
    origtatripno            TEXT,
    zone                    TEXT,
    mode                    INTEGER,
    psgld                   TEXT,
    stst                    INTEGER,
    stsd                    TEXT,
    raw_json                TEXT
);

CREATE SEQUENCE IF NOT EXISTS bus_v3_prediction_observation_seq;
CREATE TABLE IF NOT EXISTS bus_v3_prediction_observation (
    prediction_obs_id       BIGINT PRIMARY KEY DEFAULT nextval('bus_v3_prediction_observation_seq'),
    poll_id                 BIGINT NOT NULL,
    run_id                  TEXT NOT NULL,
    cta_server_time_ms      BIGINT,
    local_response_end_ms   BIGINT NOT NULL,
    query_kind              TEXT,
    tmstmp_ms               BIGINT,
    prediction_age_s        DOUBLE,
    typ                     TEXT,
    stpid                   TEXT,
    stpnm                   TEXT,
    vid                     TEXT,
    dstp_ft                 DOUBLE,
    rt                      TEXT,
    rtdd                    TEXT,
    rtdir                   TEXT,
    des                     TEXT,
    prdtm_ms                BIGINT,
    eta_s                   DOUBLE,
    prdctdn_raw             TEXT,
    prdctdn_min             DOUBLE,
    dly                     INTEGER,
    dyn                     INTEGER,
    tablockid               TEXT,
    tatripid                TEXT,
    origtatripno            TEXT,
    zone                    TEXT,
    psgld                   TEXT,
    stst                    INTEGER,
    stsd                    TEXT,
    flagstop                INTEGER,
    raw_json                TEXT
);

CREATE SEQUENCE IF NOT EXISTS bus_v3_arrival_event_seq;
CREATE TABLE IF NOT EXISTS bus_v3_arrival_event (
    event_id                BIGINT PRIMARY KEY DEFAULT nextval('bus_v3_arrival_event_seq'),
    run_id                  TEXT NOT NULL,
    stpid                   TEXT NOT NULL,
    rt                      TEXT,
    rtdir                   TEXT,
    vid                     TEXT,
    pid                     INTEGER,
    tatripid                TEXT,
    origtatripno            TEXT,
    tablockid               TEXT,
    stst                    INTEGER,
    stsd                    TEXT,
    stop_pdist_ft           DOUBLE,
    actual_arrival_ms       BIGINT,
    label                   TEXT NOT NULL,
    high_confidence         BOOLEAN NOT NULL DEFAULT FALSE,
    confidence              DOUBLE NOT NULL DEFAULT 0,
    evidence_json           TEXT NOT NULL,
    reason_codes_json       TEXT NOT NULL,
    first_vehicle_obs_id    BIGINT,
    second_vehicle_obs_id   BIGINT,
    created_at_ms           BIGINT NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS bus_v3_online_estimate_seq;
CREATE TABLE IF NOT EXISTS bus_v3_online_estimate (
    estimate_id             BIGINT PRIMARY KEY DEFAULT nextval('bus_v3_online_estimate_seq'),
    run_id                  TEXT,
    generated_at_ms         BIGINT NOT NULL,
    stpid                   TEXT NOT NULL,
    rt                      TEXT,
    rtdir                   TEXT,
    vid                     TEXT,
    tatripid                TEXT,
    tablockid               TEXT,
    predicted_arrival_ms    BIGINT,
    interval80_low_ms       BIGINT,
    interval80_high_ms      BIGINT,
    interval90_low_ms       BIGINT,
    interval90_high_ms      BIGINT,
    interval95_low_ms       BIGINT,
    interval95_high_ms      BIGINT,
    reliability             DOUBLE NOT NULL,
    display_state           TEXT NOT NULL,
    data_quality            TEXT NOT NULL,
    rider_message           TEXT,
    reason_codes_json       TEXT NOT NULL,
    features_json           TEXT NOT NULL,
    raw_estimate_json       TEXT NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS bus_v3_residual_quantile_seq;
CREATE TABLE IF NOT EXISTS bus_v3_residual_quantile (
    cal_id          BIGINT PRIMARY KEY DEFAULT nextval('bus_v3_residual_quantile_seq'),
    created_at_ms   BIGINT NOT NULL,
    rt              TEXT,
    stpid           TEXT,
    rtdir           TEXT,
    horizon_bin     TEXT NOT NULL,
    quality_bin     TEXT NOT NULL,
    n               INTEGER NOT NULL,
    q05_s           DOUBLE,
    q10_s           DOUBLE,
    q25_s           DOUBLE,
    q50_s           DOUBLE,
    q75_s           DOUBLE,
    q90_s           DOUBLE,
    q95_s           DOUBLE,
    mae_s           DOUBLE,
    bias_s          DOUBLE
);

CREATE SEQUENCE IF NOT EXISTS bus_v3_reliability_bin_seq;
CREATE TABLE IF NOT EXISTS bus_v3_reliability_bin (
    bin_id                          BIGINT PRIMARY KEY DEFAULT nextval('bus_v3_reliability_bin_seq'),
    created_at_ms                   BIGINT NOT NULL,
    run_id                          TEXT,
    lower_bound                     DOUBLE NOT NULL,
    upper_bound                     DOUBLE NOT NULL,
    n                               INTEGER NOT NULL,
    mean_predicted_reliability      DOUBLE,
    empirical_success_rate          DOUBLE,
    brier                           DOUBLE,
    ece_component                   DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_bus_v3_api_poll_run_endpoint
    ON bus_v3_api_poll(run_id, endpoint, local_request_start_ms);
CREATE INDEX IF NOT EXISTS idx_bus_v3_api_poll_time
    ON bus_v3_api_poll(local_request_start_ms);
CREATE INDEX IF NOT EXISTS idx_bus_v3_stop_rt_dir
    ON bus_v3_stop(rt, rtdir);
CREATE INDEX IF NOT EXISTS idx_bus_v3_pattern_route_dir
    ON bus_v3_pattern(rt, rtdir);
CREATE INDEX IF NOT EXISTS idx_bus_v3_pattern_dtrid
    ON bus_v3_pattern(dtrid);
CREATE INDEX IF NOT EXISTS idx_bus_v3_pattern_point_stop
    ON bus_v3_pattern_point(stpid, pid);
CREATE INDEX IF NOT EXISTS idx_bus_v3_pattern_point_pid_pdist
    ON bus_v3_pattern_point(pid, pdist_ft);
CREATE INDEX IF NOT EXISTS idx_bus_v3_detour_id_state
    ON bus_v3_detour(id, state);
CREATE INDEX IF NOT EXISTS idx_bus_v3_vehicle_obs_vid_time
    ON bus_v3_vehicle_observation(vid, tmstmp_ms);
CREATE INDEX IF NOT EXISTS idx_bus_v3_vehicle_obs_run_rt
    ON bus_v3_vehicle_observation(run_id, rt, local_response_end_ms);
CREATE INDEX IF NOT EXISTS idx_bus_v3_vehicle_obs_trip
    ON bus_v3_vehicle_observation(vid, rt, pid, tatripid, tablockid, stsd, stst);
CREATE INDEX IF NOT EXISTS idx_bus_v3_prediction_obs_stop_time
    ON bus_v3_prediction_observation(stpid, rt, rtdir, local_response_end_ms);
CREATE INDEX IF NOT EXISTS idx_bus_v3_prediction_obs_vid_stop_time
    ON bus_v3_prediction_observation(vid, stpid, rt, tmstmp_ms);
CREATE INDEX IF NOT EXISTS idx_bus_v3_prediction_obs_trip
    ON bus_v3_prediction_observation(vid, rt, stpid, tatripid, tablockid, stsd, stst);
CREATE INDEX IF NOT EXISTS idx_bus_v3_arrival_lookup
    ON bus_v3_arrival_event(run_id, stpid, rt, rtdir, vid, actual_arrival_ms);
CREATE INDEX IF NOT EXISTS idx_bus_v3_arrival_high_conf
    ON bus_v3_arrival_event(high_confidence, label);
CREATE INDEX IF NOT EXISTS idx_bus_v3_online_estimate_stop_time
    ON bus_v3_online_estimate(stpid, rt, generated_at_ms);
CREATE INDEX IF NOT EXISTS idx_bus_v3_residual_quantile_lookup
    ON bus_v3_residual_quantile(rt, stpid, rtdir, horizon_bin, quality_bin, created_at_ms);

-- ============================================================
-- CTA L (train) v2 parallel pipeline.
-- See src/transit_observer/train_v2/ for the ingest + estimator code.
-- Pulls from three independent CTA streams:
--   1. Train Tracker ttarrivals.aspx (by-station predictions)
--   2. Train Tracker ttfollow.aspx   (per-run trajectory — NEW)
--   3. Train Tracker ttpositions.aspx (per-line vehicle positions)
--   4. CTA GTFS-RT TripUpdates + VehiclePositions (independent stream
--      with delay + congestion + current_status fields)
-- Timestamps are BIGINT ms epochs. Stop and map IDs are stored as TEXT
-- so future feeds with non-numeric IDs can land here unchanged.
-- Legacy train_arrivals_raw / train_positions_raw stay untouched.
-- ============================================================

CREATE SEQUENCE IF NOT EXISTS train_v2_api_poll_seq;
CREATE TABLE IF NOT EXISTS train_v2_api_poll (
    poll_id                 BIGINT PRIMARY KEY DEFAULT nextval('train_v2_api_poll_seq'),
    run_id                  TEXT NOT NULL,
    cycle_index             INTEGER,
    source                  TEXT NOT NULL,                  -- 'train_tracker' | 'gtfsrt_train'
    endpoint                TEXT NOT NULL,
    query_kind              TEXT,
    request_url_redacted    TEXT,
    params_json_redacted    TEXT NOT NULL,
    local_request_start_ms  BIGINT NOT NULL,
    local_response_end_ms   BIGINT NOT NULL,
    cta_server_time_ms      BIGINT,
    http_status             INTEGER,
    latency_ms              DOUBLE,
    ok                      BOOLEAN NOT NULL DEFAULT FALSE,
    error_message           TEXT,
    raw_json                TEXT,                           -- JSON body for tt*.aspx
    raw_sha256              TEXT,
    created_at_ms           BIGINT NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS train_v2_arrival_observation_seq;
CREATE TABLE IF NOT EXISTS train_v2_arrival_observation (
    arrival_obs_id          BIGINT PRIMARY KEY DEFAULT nextval('train_v2_arrival_observation_seq'),
    poll_id                 BIGINT NOT NULL,
    run_id                  TEXT NOT NULL,
    cta_server_time_ms      BIGINT,
    local_response_end_ms   BIGINT NOT NULL,
    query_kind              TEXT,
    line                    TEXT,                           -- API line code: 'Red' | 'Blue' | …
    run_number              TEXT,
    map_id                  TEXT,
    stop_id                 TEXT,
    station_name            TEXT,
    stop_description        TEXT,
    direction_code          TEXT,
    destination_name        TEXT,
    destination_map_id      TEXT,
    predicted_at_ms         BIGINT,                         -- ``prdt`` (when CTA produced the prediction)
    arrival_at_ms           BIGINT,                         -- ``arrT`` (predicted arrival)
    eta_s                   DOUBLE,                         -- arrival_at - server_time
    prediction_age_s        DOUBLE,                         -- server_time - predicted_at
    is_approaching          BOOLEAN,
    is_delayed              BOOLEAN,
    is_fault                BOOLEAN,
    is_scheduled            BOOLEAN,
    flags                   TEXT,
    raw_json                TEXT
);

CREATE SEQUENCE IF NOT EXISTS train_v2_follow_observation_seq;
CREATE TABLE IF NOT EXISTS train_v2_follow_observation (
    follow_obs_id           BIGINT PRIMARY KEY DEFAULT nextval('train_v2_follow_observation_seq'),
    poll_id                 BIGINT NOT NULL,
    run_id                  TEXT NOT NULL,
    cta_server_time_ms      BIGINT,
    local_response_end_ms   BIGINT NOT NULL,
    run_number              TEXT NOT NULL,
    line                    TEXT,
    seq                     INTEGER NOT NULL,                -- order of this stop in the follow response
    map_id                  TEXT,
    stop_id                 TEXT,
    station_name            TEXT,
    direction_code          TEXT,
    destination_name        TEXT,
    predicted_at_ms         BIGINT,
    arrival_at_ms           BIGINT,
    eta_s                   DOUBLE,
    is_approaching          BOOLEAN,
    is_delayed              BOOLEAN,
    is_fault                BOOLEAN,
    is_scheduled            BOOLEAN,
    flags                   TEXT,
    raw_json                TEXT
);

CREATE SEQUENCE IF NOT EXISTS train_v2_position_observation_seq;
CREATE TABLE IF NOT EXISTS train_v2_position_observation (
    position_obs_id         BIGINT PRIMARY KEY DEFAULT nextval('train_v2_position_observation_seq'),
    poll_id                 BIGINT NOT NULL,
    run_id                  TEXT NOT NULL,
    cta_server_time_ms      BIGINT,
    local_response_end_ms   BIGINT NOT NULL,
    line                    TEXT,
    run_number              TEXT,
    direction_code          TEXT,
    destination_name        TEXT,
    destination_map_id      TEXT,
    next_station_map_id     TEXT,
    next_station_name       TEXT,
    predicted_at_ms         BIGINT,
    next_arrival_at_ms      BIGINT,
    is_approaching          BOOLEAN,
    is_delayed              BOOLEAN,
    is_fault                BOOLEAN,
    lat                     DOUBLE,
    lon                     DOUBLE,
    heading                 DOUBLE,
    raw_json                TEXT
);

CREATE SEQUENCE IF NOT EXISTS train_v2_gtfsrt_trip_update_seq;
CREATE TABLE IF NOT EXISTS train_v2_gtfsrt_trip_update (
    trip_update_id          BIGINT PRIMARY KEY DEFAULT nextval('train_v2_gtfsrt_trip_update_seq'),
    poll_id                 BIGINT NOT NULL,
    run_id                  TEXT NOT NULL,
    local_response_end_ms   BIGINT NOT NULL,
    route_id                TEXT,
    trip_id                 TEXT,
    vehicle_id              TEXT,
    stop_id                 TEXT,
    stop_sequence           INTEGER,
    arrival_time_ms         BIGINT,
    arrival_delay_seconds   INTEGER,
    departure_time_ms       BIGINT,
    departure_delay_seconds INTEGER,
    schedule_relationship   TEXT
);

CREATE SEQUENCE IF NOT EXISTS train_v2_gtfsrt_vehicle_position_seq;
CREATE TABLE IF NOT EXISTS train_v2_gtfsrt_vehicle_position (
    vehicle_position_id     BIGINT PRIMARY KEY DEFAULT nextval('train_v2_gtfsrt_vehicle_position_seq'),
    poll_id                 BIGINT NOT NULL,
    run_id                  TEXT NOT NULL,
    local_response_end_ms   BIGINT NOT NULL,
    route_id                TEXT,
    trip_id                 TEXT,
    vehicle_id              TEXT,
    vehicle_label           TEXT,
    lat                     DOUBLE,
    lon                     DOUBLE,
    bearing                 DOUBLE,
    speed_mps               DOUBLE,
    current_stop_sequence   INTEGER,
    current_status          TEXT,
    congestion_level        TEXT,
    occupancy_status        TEXT
);

CREATE SEQUENCE IF NOT EXISTS train_v2_arrival_event_seq;
CREATE TABLE IF NOT EXISTS train_v2_arrival_event (
    event_id                BIGINT PRIMARY KEY DEFAULT nextval('train_v2_arrival_event_seq'),
    run_id                  TEXT NOT NULL,
    map_id                  TEXT NOT NULL,
    line                    TEXT,
    direction_code          TEXT,
    run_number              TEXT,
    destination_name        TEXT,
    actual_arrival_ms       BIGINT,
    label                   TEXT NOT NULL,
    high_confidence         BOOLEAN NOT NULL DEFAULT FALSE,
    confidence              DOUBLE NOT NULL DEFAULT 0,
    evidence_json           TEXT NOT NULL,
    reason_codes_json       TEXT NOT NULL,
    before_position_obs_id  BIGINT,                          -- position row where train was "incoming to" map_id
    after_position_obs_id   BIGINT,                          -- position row where nextStaId had advanced past map_id
    gtfsrt_corroboration_id BIGINT,                          -- optional: the GTFS-RT vehicle position that confirmed
    created_at_ms           BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS train_v2_line_topology (
    line                    TEXT NOT NULL,
    direction_code          TEXT NOT NULL,
    seq                     INTEGER NOT NULL,                -- 0 .. N-1 along the direction of travel
    map_id                  TEXT NOT NULL,
    station_name            TEXT,
    lat                     DOUBLE,
    lon                     DOUBLE,
    PRIMARY KEY (line, direction_code, seq)
);

CREATE TABLE IF NOT EXISTS train_v2_slow_zone (
    slow_zone_id            TEXT PRIMARY KEY,                -- usually CTA's id field or a hash if missing
    line                    TEXT NOT NULL,
    direction_code          TEXT,
    from_station            TEXT,
    to_station              TEXT,
    max_mph                 DOUBLE,
    posted_at_ms            BIGINT,
    expected_clear_at_ms    BIGINT,
    description             TEXT,
    raw_payload_json        TEXT,
    first_seen_poll_id      BIGINT,
    last_seen_poll_id       BIGINT
);

CREATE SEQUENCE IF NOT EXISTS train_v2_online_estimate_seq;
CREATE TABLE IF NOT EXISTS train_v2_online_estimate (
    estimate_id             BIGINT PRIMARY KEY DEFAULT nextval('train_v2_online_estimate_seq'),
    run_id                  TEXT,
    generated_at_ms         BIGINT NOT NULL,
    map_id                  TEXT NOT NULL,
    line                    TEXT,
    direction_code          TEXT,
    run_number              TEXT,
    predicted_arrival_ms    BIGINT,
    interval80_low_ms       BIGINT,
    interval80_high_ms      BIGINT,
    interval90_low_ms       BIGINT,
    interval90_high_ms      BIGINT,
    interval95_low_ms       BIGINT,
    interval95_high_ms      BIGINT,
    reliability             DOUBLE NOT NULL,
    display_state           TEXT NOT NULL,
    data_quality            TEXT NOT NULL,
    rider_message           TEXT,
    reason_codes_json       TEXT NOT NULL,
    features_json           TEXT NOT NULL,
    raw_estimate_json       TEXT NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS train_v2_residual_quantile_seq;
CREATE TABLE IF NOT EXISTS train_v2_residual_quantile (
    cal_id          BIGINT PRIMARY KEY DEFAULT nextval('train_v2_residual_quantile_seq'),
    created_at_ms   BIGINT NOT NULL,
    line            TEXT,
    map_id          TEXT,
    direction_code  TEXT,
    horizon_bin     TEXT NOT NULL,
    quality_bin     TEXT NOT NULL,
    n               INTEGER NOT NULL,
    q05_s           DOUBLE,
    q10_s           DOUBLE,
    q25_s           DOUBLE,
    q50_s           DOUBLE,
    q75_s           DOUBLE,
    q90_s           DOUBLE,
    q95_s           DOUBLE,
    mae_s           DOUBLE,
    bias_s          DOUBLE
);

CREATE SEQUENCE IF NOT EXISTS train_v2_reliability_bin_seq;
CREATE TABLE IF NOT EXISTS train_v2_reliability_bin (
    bin_id                          BIGINT PRIMARY KEY DEFAULT nextval('train_v2_reliability_bin_seq'),
    created_at_ms                   BIGINT NOT NULL,
    run_id                          TEXT,
    lower_bound                     DOUBLE NOT NULL,
    upper_bound                     DOUBLE NOT NULL,
    n                               INTEGER NOT NULL,
    mean_predicted_reliability      DOUBLE,
    empirical_success_rate          DOUBLE,
    brier                           DOUBLE,
    ece_component                   DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_train_v2_api_poll_run_endpoint
    ON train_v2_api_poll(run_id, source, endpoint, local_request_start_ms);
CREATE INDEX IF NOT EXISTS idx_train_v2_api_poll_time
    ON train_v2_api_poll(local_request_start_ms);
CREATE INDEX IF NOT EXISTS idx_train_v2_arrival_obs_station_time
    ON train_v2_arrival_observation(map_id, line, direction_code, local_response_end_ms);
CREATE INDEX IF NOT EXISTS idx_train_v2_arrival_obs_run
    ON train_v2_arrival_observation(run_number, map_id, local_response_end_ms);
CREATE INDEX IF NOT EXISTS idx_train_v2_follow_obs_run_time
    ON train_v2_follow_observation(run_number, local_response_end_ms, seq);
CREATE INDEX IF NOT EXISTS idx_train_v2_follow_obs_station
    ON train_v2_follow_observation(map_id, line, local_response_end_ms);
CREATE INDEX IF NOT EXISTS idx_train_v2_position_obs_run_time
    ON train_v2_position_observation(run_number, line, local_response_end_ms);
CREATE INDEX IF NOT EXISTS idx_train_v2_position_obs_next
    ON train_v2_position_observation(next_station_map_id, line, local_response_end_ms);
CREATE INDEX IF NOT EXISTS idx_train_v2_gtfsrt_trip_route
    ON train_v2_gtfsrt_trip_update(route_id, stop_id, arrival_time_ms);
CREATE INDEX IF NOT EXISTS idx_train_v2_gtfsrt_trip_vehicle
    ON train_v2_gtfsrt_trip_update(vehicle_id, stop_id, arrival_time_ms);
CREATE INDEX IF NOT EXISTS idx_train_v2_gtfsrt_pos_vehicle_time
    ON train_v2_gtfsrt_vehicle_position(vehicle_id, local_response_end_ms);
CREATE INDEX IF NOT EXISTS idx_train_v2_arrival_event_lookup
    ON train_v2_arrival_event(run_id, map_id, line, direction_code, actual_arrival_ms);
CREATE INDEX IF NOT EXISTS idx_train_v2_arrival_event_high_conf
    ON train_v2_arrival_event(high_confidence, label);
CREATE INDEX IF NOT EXISTS idx_train_v2_topology_lookup
    ON train_v2_line_topology(line, direction_code, map_id);
CREATE INDEX IF NOT EXISTS idx_train_v2_online_estimate_lookup
    ON train_v2_online_estimate(map_id, line, generated_at_ms);
CREATE INDEX IF NOT EXISTS idx_train_v2_residual_quantile_lookup
    ON train_v2_residual_quantile(line, map_id, direction_code, horizon_bin, quality_bin, created_at_ms);

-- ============================================================
-- GTFS-RT Alert entity. The proprietary CTA alerts RSS feed already
-- lands in cta_alerts_raw. This table captures the GTFS-RT side, which
-- carries machine-readable cause/effect/severity codes and a list of
-- informed_entity selectors (route/trip/stop) that the RSS does not.
-- ============================================================
CREATE SEQUENCE IF NOT EXISTS train_v2_gtfsrt_alert_seq;
CREATE TABLE IF NOT EXISTS train_v2_gtfsrt_alert (
    alert_id                BIGINT PRIMARY KEY DEFAULT nextval('train_v2_gtfsrt_alert_seq'),
    poll_id                 BIGINT NOT NULL,
    run_id                  TEXT NOT NULL,
    local_response_end_ms   BIGINT NOT NULL,
    feed_timestamp_ms       BIGINT,
    feed_incrementality     TEXT,
    entity_id               TEXT,
    cause                   TEXT,
    effect                  TEXT,
    severity_level          TEXT,
    header_text             TEXT,
    description_text        TEXT,
    tts_header_text         TEXT,
    tts_description_text    TEXT,
    url                     TEXT,
    active_period_json      TEXT,
    informed_entity_json    TEXT
);

CREATE INDEX IF NOT EXISTS idx_train_v2_gtfsrt_alert_time
    ON train_v2_gtfsrt_alert(local_response_end_ms);

-- ============================================================
-- GTFS-static extraction. We snapshot agency zips weekly via
-- gtfs_static_archive but never project them into tables. These rows
-- give us the canonical schedule (stop_times), official stop coords
-- (stops), line geometry (shapes), and service calendars — joinable
-- by trip_id and stop_id to GTFS-RT and (via id-bridge tables) to
-- the proprietary feeds. One snapshot row per (agency, fetched_at);
-- all extracted rows carry the snapshot_id FK.
-- ============================================================
CREATE SEQUENCE IF NOT EXISTS gtfs_static_snapshot_seq;
CREATE TABLE IF NOT EXISTS gtfs_static_snapshot (
    snapshot_id             BIGINT PRIMARY KEY DEFAULT nextval('gtfs_static_snapshot_seq'),
    agency                  TEXT NOT NULL,
    archive_path            TEXT NOT NULL,
    fetched_at_ms           BIGINT NOT NULL,
    extracted_at_ms         BIGINT,
    archive_sha256          TEXT,
    feed_start_date         TEXT,
    feed_end_date           TEXT,
    feed_publisher_name     TEXT,
    feed_version            TEXT,
    n_stops                 INTEGER,
    n_routes                INTEGER,
    n_trips                 INTEGER,
    n_stop_times            BIGINT,
    n_shapes                BIGINT,
    n_calendar              INTEGER,
    n_calendar_dates        INTEGER
);

CREATE TABLE IF NOT EXISTS gtfs_static_agency (
    snapshot_id             BIGINT NOT NULL,
    agency_id               TEXT NOT NULL,
    agency_name             TEXT,
    agency_url              TEXT,
    agency_timezone         TEXT,
    agency_lang             TEXT,
    agency_phone            TEXT,
    agency_fare_url         TEXT,
    agency_email            TEXT,
    PRIMARY KEY (snapshot_id, agency_id)
);

CREATE TABLE IF NOT EXISTS gtfs_static_stops (
    snapshot_id             BIGINT NOT NULL,
    stop_id                 TEXT NOT NULL,
    stop_code               TEXT,
    stop_name               TEXT,
    stop_desc               TEXT,
    stop_lat                DOUBLE,
    stop_lon                DOUBLE,
    zone_id                 TEXT,
    stop_url                TEXT,
    location_type           INTEGER,
    parent_station          TEXT,
    stop_timezone           TEXT,
    wheelchair_boarding     INTEGER,
    platform_code           TEXT,
    PRIMARY KEY (snapshot_id, stop_id)
);

CREATE TABLE IF NOT EXISTS gtfs_static_routes (
    snapshot_id             BIGINT NOT NULL,
    route_id                TEXT NOT NULL,
    agency_id               TEXT,
    route_short_name        TEXT,
    route_long_name         TEXT,
    route_desc              TEXT,
    route_type              INTEGER,
    route_url               TEXT,
    route_color             TEXT,
    route_text_color        TEXT,
    PRIMARY KEY (snapshot_id, route_id)
);

CREATE TABLE IF NOT EXISTS gtfs_static_trips (
    snapshot_id             BIGINT NOT NULL,
    route_id                TEXT NOT NULL,
    service_id              TEXT NOT NULL,
    trip_id                 TEXT NOT NULL,
    trip_headsign           TEXT,
    trip_short_name         TEXT,
    direction_id            INTEGER,
    block_id                TEXT,
    shape_id                TEXT,
    wheelchair_accessible   INTEGER,
    bikes_allowed           INTEGER,
    PRIMARY KEY (snapshot_id, trip_id)
);

CREATE TABLE IF NOT EXISTS gtfs_static_stop_times (
    snapshot_id             BIGINT NOT NULL,
    trip_id                 TEXT NOT NULL,
    stop_sequence           INTEGER NOT NULL,
    arrival_time            TEXT,
    departure_time          TEXT,
    stop_id                 TEXT NOT NULL,
    stop_headsign           TEXT,
    pickup_type             INTEGER,
    drop_off_type           INTEGER,
    continuous_pickup       INTEGER,
    continuous_drop_off     INTEGER,
    shape_dist_traveled     DOUBLE,
    timepoint               INTEGER,
    PRIMARY KEY (snapshot_id, trip_id, stop_sequence)
);

CREATE TABLE IF NOT EXISTS gtfs_static_shapes (
    snapshot_id             BIGINT NOT NULL,
    shape_id                TEXT NOT NULL,
    shape_pt_sequence       INTEGER NOT NULL,
    shape_pt_lat            DOUBLE,
    shape_pt_lon            DOUBLE,
    shape_dist_traveled     DOUBLE,
    PRIMARY KEY (snapshot_id, shape_id, shape_pt_sequence)
);

CREATE TABLE IF NOT EXISTS gtfs_static_calendar (
    snapshot_id             BIGINT NOT NULL,
    service_id              TEXT NOT NULL,
    monday                  INTEGER,
    tuesday                 INTEGER,
    wednesday               INTEGER,
    thursday                INTEGER,
    friday                  INTEGER,
    saturday                INTEGER,
    sunday                  INTEGER,
    start_date              TEXT,
    end_date                TEXT,
    PRIMARY KEY (snapshot_id, service_id)
);

CREATE TABLE IF NOT EXISTS gtfs_static_calendar_dates (
    snapshot_id             BIGINT NOT NULL,
    service_id              TEXT NOT NULL,
    date                    TEXT NOT NULL,
    exception_type          INTEGER,
    PRIMARY KEY (snapshot_id, service_id, date)
);

CREATE INDEX IF NOT EXISTS idx_gtfs_static_snapshot_agency
    ON gtfs_static_snapshot(agency, fetched_at_ms);
CREATE INDEX IF NOT EXISTS idx_gtfs_static_stops_parent
    ON gtfs_static_stops(snapshot_id, parent_station);
CREATE INDEX IF NOT EXISTS idx_gtfs_static_stop_times_stop
    ON gtfs_static_stop_times(snapshot_id, stop_id, arrival_time);
CREATE INDEX IF NOT EXISTS idx_gtfs_static_trips_route
    ON gtfs_static_trips(snapshot_id, route_id);
CREATE INDEX IF NOT EXISTS idx_gtfs_static_shapes_shape
    ON gtfs_static_shapes(snapshot_id, shape_id, shape_pt_sequence);
"""


def connect(path: Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    target = path or settings.db_path
    conn = duckdb.connect(str(target), read_only=read_only)
    # Pin DuckDB to all available cores. The default is already
    # ``cpu_count()`` on most platforms, but being explicit defends
    # against shells / sandboxes that drop ``OMP_NUM_THREADS=1`` or
    # similar. Cheap to set on every connection.
    cpu = os.cpu_count() or 4
    conn.execute(f"PRAGMA threads = {cpu}")
    return conn


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Bring an empty or pre-existing DB up to the current schema.

    Order matters for the upgrade-in-place case: an existing
    ``forecast_queue`` from PR #1 won't get new columns from
    ``CREATE TABLE IF NOT EXISTS`` (it's a no-op), so we must
    1) run CREATE TABLE first (creates missing tables fresh),
    2) ALTER TABLE in any columns the old schema lacks, and
    3) only then create indexes that reference the new columns.
    """
    tables: list[str] = []
    indexes: list[str] = []
    for statement in SCHEMA_SQL.split(";"):
        body = statement.strip()
        if not body:
            continue
        if body.upper().startswith("CREATE INDEX"):
            indexes.append(body)
        else:
            tables.append(body)
    for stmt in tables:
        conn.execute(stmt)
    _migrate(conn)
    for stmt in indexes:
        conn.execute(stmt)


# Schema migrations for tables that may pre-date a newer column.
# DuckDB doesn't have CREATE COLUMN IF NOT EXISTS, so we probe pragma_table_info
# and skip when the column already exists. Cheap to run on every startup.
# (table, column, type, backfill_expression).
# DuckDB rejects constraints (NOT NULL, DEFAULT) on ALTER TABLE ADD COLUMN, so
# we add the column unconstrained and run a separate UPDATE to backfill any
# NULL rows when a default is needed.
_MIGRATIONS: tuple[tuple[str, str, str, str | None], ...] = (
    ("forecast_queue", "corridor_id",        "TEXT",    None),
    ("forecast_queue", "predictor_version",  "TEXT",    None),
    ("forecast_queue", "feature_json",       "TEXT",    None),
    ("forecast_outcomes", "truth_confidence", "DOUBLE", None),
    ("corridors", "source",                  "TEXT",    "'seed'"),
    ("corridors", "promoted_from_query_count", "INTEGER", None),
    # Tier 0a: stop discarding train vehicle position lat/lon/heading.
    ("train_positions_raw", "lat",          "DOUBLE",  None),
    ("train_positions_raw", "lon",          "DOUBLE",  None),
    ("train_positions_raw", "heading",      "DOUBLE",  None),
    # Tier 0b: capture incident flags + stop description on arrival predictions.
    ("train_arrivals_raw", "flags",            "TEXT", None),
    ("train_arrivals_raw", "stop_description", "TEXT", None),
    # Tier-1 data-capture widening: GTFS-RT TripUpdate fields previously dropped.
    ("train_v2_gtfsrt_trip_update", "feed_timestamp_ms",          "BIGINT",  None),
    ("train_v2_gtfsrt_trip_update", "feed_incrementality",        "TEXT",    None),
    ("train_v2_gtfsrt_trip_update", "trip_update_timestamp_ms",   "BIGINT",  None),
    ("train_v2_gtfsrt_trip_update", "trip_update_delay_seconds",  "INTEGER", None),
    ("train_v2_gtfsrt_trip_update", "trip_start_date",            "TEXT",    None),
    ("train_v2_gtfsrt_trip_update", "trip_start_time",            "TEXT",    None),
    ("train_v2_gtfsrt_trip_update", "trip_direction_id",          "INTEGER", None),
    ("train_v2_gtfsrt_trip_update", "trip_schedule_relationship", "TEXT",    None),
    ("train_v2_gtfsrt_trip_update", "arrival_uncertainty_seconds",   "INTEGER", None),
    ("train_v2_gtfsrt_trip_update", "departure_uncertainty_seconds", "INTEGER", None),
    ("train_v2_gtfsrt_trip_update", "vehicle_label",              "TEXT",    None),
    ("train_v2_gtfsrt_trip_update", "vehicle_license_plate",      "TEXT",    None),
    # Tier-1 data-capture widening: GTFS-RT VehiclePosition fields previously dropped.
    ("train_v2_gtfsrt_vehicle_position", "feed_timestamp_ms",          "BIGINT",  None),
    ("train_v2_gtfsrt_vehicle_position", "feed_incrementality",        "TEXT",    None),
    ("train_v2_gtfsrt_vehicle_position", "vehicle_timestamp_ms",       "BIGINT",  None),
    ("train_v2_gtfsrt_vehicle_position", "stop_id",                    "TEXT",    None),
    ("train_v2_gtfsrt_vehicle_position", "occupancy_percentage",       "INTEGER", None),
    ("train_v2_gtfsrt_vehicle_position", "multi_carriage_details_json","TEXT",    None),
    ("train_v2_gtfsrt_vehicle_position", "trip_start_date",            "TEXT",    None),
    ("train_v2_gtfsrt_vehicle_position", "trip_start_time",            "TEXT",    None),
    ("train_v2_gtfsrt_vehicle_position", "trip_direction_id",          "INTEGER", None),
    ("train_v2_gtfsrt_vehicle_position", "trip_schedule_relationship", "TEXT",    None),
    ("train_v2_gtfsrt_vehicle_position", "vehicle_license_plate",      "TEXT",    None),
    ("train_v2_gtfsrt_vehicle_position", "odometer_m",                 "DOUBLE",  None),
)


def _migrate(conn: duckdb.DuckDBPyConnection) -> None:
    for table, column, ctype, backfill in _MIGRATIONS:
        existing = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM pragma_table_info(?)", [table]
            ).fetchall()
        }
        if column in existing:
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ctype}")
        if backfill is not None:
            conn.execute(
                f"UPDATE {table} SET {column} = {backfill} WHERE {column} IS NULL"
            )


def refresh_read_replica() -> None:
    """Copy the live DB to the read replica path. Cheap for small DBs;
    swap to ATTACH if size becomes a problem."""
    if not settings.db_path.exists():
        return
    tmp = settings.read_replica_path.with_suffix(".tmp")
    shutil.copy2(settings.db_path, tmp)
    tmp.replace(settings.read_replica_path)


@contextmanager
def writer() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = connect()
    try:
        init_schema(conn)
        yield conn
    finally:
        conn.close()


@contextmanager
def reader() -> Iterator[duckdb.DuckDBPyConnection]:
    path = settings.read_replica_path if settings.read_replica_path.exists() else settings.db_path
    conn = connect(path=path, read_only=True)
    try:
        yield conn
    finally:
        conn.close()
