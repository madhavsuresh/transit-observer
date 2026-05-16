"""Schema migration upgrades a pre-corpus DB in place.

Regression test for the bug where ``CREATE INDEX idx_forecast_corridor ON
forecast_queue(corridor_id)`` ran before the ``corridor_id`` column got
added to a pre-existing PR #1 database. The fix splits ``init_schema``
into table-creation, migration, then index-creation.
"""

from __future__ import annotations

import os
import tempfile

import duckdb
import pytest

from transit_observer import db


PR1_FORECAST_QUEUE_DDL = """
CREATE TABLE forecast_queue (
    forecast_id TEXT PRIMARY KEY,
    enqueued_at TIMESTAMPTZ,
    snapshot_polled_at TIMESTAMPTZ,
    leave_at TIMESTAMPTZ,
    mode TEXT, line TEXT, direction_code TEXT,
    boarding_map_id INTEGER, boarding_text_id TEXT, boarding_station_name TEXT,
    alighting_map_id INTEGER, alighting_text_id TEXT, alighting_station_name TEXT,
    predicted_wait_mean DOUBLE, predicted_wait_p50 DOUBLE,
    predicted_wait_p80 DOUBLE, predicted_wait_p90 DOUBLE,
    predicted_in_vehicle_mean DOUBLE,
    predicted_total_mean DOUBLE, predicted_total_p50 DOUBLE,
    predicted_total_p80 DOUBLE, predicted_total_p90 DOUBLE,
    predicted_failure_prob DOUBLE,
    resolve_after TIMESTAMPTZ, status TEXT
)
"""

PR1_FORECAST_OUTCOMES_DDL = """
CREATE TABLE forecast_outcomes (
    forecast_id TEXT PRIMARY KEY,
    resolved_at TIMESTAMPTZ,
    boarded_run_number TEXT, boarded_at TIMESTAMPTZ, alighted_at TIMESTAMPTZ,
    actual_wait_seconds DOUBLE, actual_in_vehicle_seconds DOUBLE, actual_total_seconds DOUBLE,
    in_p80_window BOOLEAN, in_p90_window BOOLEAN,
    p50_residual_seconds DOUBLE, p80_residual_seconds DOUBLE,
    failed BOOLEAN, notes TEXT
)
"""


def _columns(conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM pragma_table_info(?)", [table]).fetchall()}


@pytest.fixture
def pr1_db_path():
    """Create a temp DB file with the PR #1 schema (no corpus columns)."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "pr1.duckdb")
    c = duckdb.connect(path)
    c.execute(PR1_FORECAST_QUEUE_DDL)
    c.execute(PR1_FORECAST_OUTCOMES_DDL)
    c.close()
    yield path
    try:
        os.unlink(path)
        os.rmdir(tmpdir)
    except OSError:
        pass


def test_migrates_pr1_forecast_queue_in_place(pr1_db_path):
    conn = duckdb.connect(pr1_db_path)
    try:
        pre = _columns(conn, "forecast_queue")
        assert "corridor_id" not in pre
        assert "predictor_version" not in pre
        assert "feature_json" not in pre

        db.init_schema(conn)

        post = _columns(conn, "forecast_queue")
        assert "corridor_id" in post
        assert "predictor_version" in post
        assert "feature_json" in post
    finally:
        conn.close()


def test_migrates_pr1_forecast_outcomes_in_place(pr1_db_path):
    conn = duckdb.connect(pr1_db_path)
    try:
        assert "truth_confidence" not in _columns(conn, "forecast_outcomes")
        db.init_schema(conn)
        assert "truth_confidence" in _columns(conn, "forecast_outcomes")
    finally:
        conn.close()


def test_corridors_table_created_on_old_db(pr1_db_path):
    conn = duckdb.connect(pr1_db_path)
    try:
        db.init_schema(conn)
        n = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'corridors'"
        ).fetchone()[0]
        assert n == 1
    finally:
        conn.close()


def test_corridor_id_index_works_after_migration(pr1_db_path):
    """The original bug: idx_forecast_corridor was created before the
    corridor_id column existed, blowing up init_schema."""
    conn = duckdb.connect(pr1_db_path)
    try:
        db.init_schema(conn)
        # Index references corridor_id -- this query exercises it.
        conn.execute("SELECT forecast_id FROM forecast_queue WHERE corridor_id IS NULL").fetchall()
    finally:
        conn.close()


def test_init_schema_on_empty_db_is_clean():
    """Sanity: fresh DB still works (no tables to migrate)."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "fresh.duckdb")
    conn = duckdb.connect(path)
    try:
        db.init_schema(conn)
        # All expected tables present.
        names = {
            r[0] for r in conn.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
        assert "corridors" in names
        assert "forecast_queue" in names
        assert "forecast_outcomes" in names
        # New columns present.
        assert "corridor_id" in _columns(conn, "forecast_queue")
        assert "truth_confidence" in _columns(conn, "forecast_outcomes")
    finally:
        conn.close()
        os.unlink(path)
        os.rmdir(tmpdir)
