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
