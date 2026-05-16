"""NDJSON query log -> DuckDB import path."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from transit_observer import db, query_log
from transit_observer.corpus import AdHocPrediction


T0 = datetime(2026, 5, 14, 8, 0, 0, tzinfo=timezone.utc)  # Thursday


@pytest.fixture
def patched_paths(tmp_path: Path, monkeypatch):
    """Point QUERIES_PATH + cursor at a temp dir for isolation."""
    monkeypatch.setattr(query_log, "QUERIES_PATH", tmp_path / "queries.ndjson")
    monkeypatch.setattr(query_log, "QUERIES_CURSOR_PATH", tmp_path / "queries.cursor")
    yield tmp_path


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    db.init_schema(c)
    try:
        yield c
    finally:
        c.close()


def _make_prediction() -> AdHocPrediction:
    return AdHocPrediction(
        mode="L", line="Red", direction_code="southbound",
        boarding_label="Belmont", alighting_label="Lake",
        predicted_wait_mean=120.0, predicted_wait_p50=120.0,
        predicted_wait_p80=180.0, predicted_wait_p90=240.0,
        predicted_in_vehicle_mean=600.0,
        predicted_total_p50=720.0, predicted_total_p80=780.0, predicted_total_p90=840.0,
        predictor_version="kernel-v1",
    )


def test_append_writes_ndjson_record(patched_paths: Path):
    qid = query_log.append_query(
        queried_at=T0, client_id="test",
        mode="L", line="Red",
        boarding_int_id=41320, boarding_text_id=None,
        alighting_int_id=41660, alighting_text_id=None,
        prediction=_make_prediction(), error_reason=None,
    )
    assert qid
    lines = query_log.QUERIES_PATH.read_text().strip().split("\n")
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["query_id"] == qid
    assert rec["mode"] == "L"
    assert rec["success"] is True
    assert rec["predicted_total_p80"] == 780.0


def test_append_failed_query_marks_unsuccessful(patched_paths: Path):
    qid = query_log.append_query(
        queried_at=T0, client_id=None,
        mode="L", line="Red",
        boarding_int_id=99999, boarding_text_id=None,
        alighting_int_id=99998, alighting_text_id=None,
        prediction=None, error_reason="No data",
    )
    rec = json.loads(query_log.QUERIES_PATH.read_text().strip())
    assert rec["query_id"] == qid
    assert rec["success"] is False
    assert rec["error_reason"] == "No data"


def test_import_pending_loads_records_into_query_log(
    patched_paths: Path, conn: duckdb.DuckDBPyConnection,
):
    for i in range(3):
        query_log.append_query(
            queried_at=T0 + timedelta(seconds=i), client_id=None,
            mode="L", line="Red",
            boarding_int_id=41320, boarding_text_id=None,
            alighting_int_id=41660, alighting_text_id=None,
            prediction=_make_prediction(), error_reason=None,
        )
    n = query_log.import_pending(conn)
    assert n == 3

    rows = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()
    assert rows[0] == 3


def test_import_pending_is_idempotent(
    patched_paths: Path, conn: duckdb.DuckDBPyConnection,
):
    query_log.append_query(
        queried_at=T0, client_id=None,
        mode="L", line="Red",
        boarding_int_id=41320, boarding_text_id=None,
        alighting_int_id=41660, alighting_text_id=None,
        prediction=_make_prediction(), error_reason=None,
    )
    query_log.import_pending(conn)
    # Second import: no new rows since cursor already moved past EOF.
    n = query_log.import_pending(conn)
    assert n == 0
    assert conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0] == 1


def test_import_pending_resets_cursor_when_file_shrinks(
    patched_paths: Path, conn: duckdb.DuckDBPyConnection,
):
    """If queries.ndjson is smaller than the saved cursor (rotation), the
    cursor resets to 0 so we re-import from the start."""
    # Seed cursor to a position beyond the file's actual size.
    query_log._write_cursor(99_999)
    query_log.append_query(
        queried_at=T0, client_id=None,
        mode="L", line="Red",
        boarding_int_id=41320, boarding_text_id=None,
        alighting_int_id=41660, alighting_text_id=None,
        prediction=_make_prediction(), error_reason=None,
    )
    n = query_log.import_pending(conn)
    assert n == 1


def test_find_popular_ods_filters_below_threshold(
    patched_paths: Path, conn: duckdb.DuckDBPyConnection,
):
    # 5 queries for one OD pair; 2 for another.
    for i in range(5):
        query_log.append_query(
            queried_at=T0 + timedelta(seconds=i), client_id=None,
            mode="L", line="Red",
            boarding_int_id=41320, boarding_text_id=None,
            alighting_int_id=41660, alighting_text_id=None,
            prediction=_make_prediction(), error_reason=None,
        )
    for i in range(2):
        query_log.append_query(
            queried_at=T0 + timedelta(seconds=i), client_id=None,
            mode="L", line="Red",
            boarding_int_id=41450, boarding_text_id=None,
            alighting_int_id=40240, alighting_text_id=None,
            prediction=_make_prediction(), error_reason=None,
        )
    query_log.import_pending(conn)

    popular = query_log.find_popular_ods(
        conn, now=T0 + timedelta(minutes=1), min_count=3,
    )
    assert len(popular) == 1
    assert popular[0]["boarding_int_id"] == 41320
    assert popular[0]["count"] == 5


def test_find_popular_ods_excludes_existing_corridors(
    patched_paths: Path, conn: duckdb.DuckDBPyConnection,
):
    from transit_observer.corridors import seed_corridors

    seed_corridors(conn, now=T0)
    # Query Belmont->Lake 10 times -- that's a SEEDED corridor, so it
    # should NOT be a promotion candidate.
    for i in range(10):
        query_log.append_query(
            queried_at=T0 + timedelta(seconds=i), client_id=None,
            mode="L", line="Red",
            boarding_int_id=41320, boarding_text_id=None,
            alighting_int_id=41660, alighting_text_id=None,
            prediction=_make_prediction(), error_reason=None,
        )
    query_log.import_pending(conn)
    popular = query_log.find_popular_ods(
        conn, now=T0 + timedelta(minutes=1), min_count=3,
    )
    assert popular == []
