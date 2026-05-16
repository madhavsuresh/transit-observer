"""DuckDB schema and connection helpers.

Single-writer pattern: only the collector opens a writable connection on
``settings.db_path``. Everything else reads ``settings.read_replica_path``,
which the collector refreshes every ``read_replica_refresh_seconds``.
"""

from __future__ import annotations

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
    is_scheduled     BOOLEAN
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
    is_delayed          BOOLEAN
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
"""


def connect(path: Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    target = path or settings.db_path
    return duckdb.connect(str(target), read_only=read_only)


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
