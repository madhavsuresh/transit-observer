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
    line                     TEXT        NOT NULL,
    direction_code           TEXT,
    boarding_map_id          INTEGER     NOT NULL,
    boarding_station_name    TEXT,
    alighting_map_id         INTEGER     NOT NULL,
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
    failed                   BOOLEAN,
    notes                    TEXT
);
"""


def connect(path: Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    target = path or settings.db_path
    return duckdb.connect(str(target), read_only=read_only)


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    for statement in SCHEMA_SQL.split(";"):
        body = statement.strip()
        if body:
            conn.execute(body)


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
