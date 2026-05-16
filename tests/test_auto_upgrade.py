"""Auto-promote popular OD pairs from query_log into corridors."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from transit_observer import db, query_log
from transit_observer.auto_upgrade import promote_popular
from transit_observer.corpus import AdHocPrediction


T0 = datetime(2026, 5, 14, 8, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def patched_paths(tmp_path: Path, monkeypatch):
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


def _seed_queries(n: int, *, conn, mode="L", line="Red", boarding=41220, alighting=41450) -> None:
    pred = AdHocPrediction(
        mode=mode, line=line, direction_code="southbound",
        boarding_label="Fullerton", alighting_label="Chicago",
        predicted_wait_mean=120.0, predicted_wait_p50=120.0,
        predicted_wait_p80=180.0, predicted_wait_p90=240.0,
        predicted_in_vehicle_mean=600.0,
        predicted_total_p50=720.0, predicted_total_p80=780.0, predicted_total_p90=840.0,
        predictor_version="kernel-v1",
    )
    for i in range(n):
        query_log.append_query(
            queried_at=T0 + timedelta(seconds=i), client_id=None,
            mode=mode, line=line,
            boarding_int_id=boarding, boarding_text_id=None,
            alighting_int_id=alighting, alighting_text_id=None,
            prediction=pred, error_reason=None,
        )
    query_log.import_pending(conn)


def test_promote_inserts_l_corridor(patched_paths, conn):
    # Fullerton->Chicago is NOT in SEED_CORRIDORS (we have Wilson->Chicago but
    # not Fullerton->Chicago). Five queries with min_count=3 -> promote.
    _seed_queries(5, conn=conn, boarding=41220, alighting=41450)
    promoted = promote_popular(conn, now=T0 + timedelta(minutes=1), min_count=3)
    assert len(promoted) == 1
    cid = promoted[0]
    assert cid.startswith("auto-l-red-")

    row = conn.execute(
        "SELECT mode, line, source, promoted_from_query_count, origin_label, destination_label "
        "FROM corridors WHERE corridor_id = ?", [cid],
    ).fetchone()
    assert row is not None
    assert row[0] == "L"
    assert row[1] == "Red"
    assert row[2] == "auto_upgraded"
    assert row[3] == 5
    assert row[4] == "Fullerton"
    assert row[5] == "Chicago"


def test_promote_is_idempotent(patched_paths, conn):
    """Running promote twice doesn't double-insert the same OD."""
    _seed_queries(5, conn=conn, boarding=41220, alighting=41450)
    promote_popular(conn, now=T0 + timedelta(minutes=1), min_count=3)
    n_before = conn.execute("SELECT COUNT(*) FROM corridors").fetchone()[0]
    promote_popular(conn, now=T0 + timedelta(minutes=2), min_count=3)
    n_after = conn.execute("SELECT COUNT(*) FROM corridors").fetchone()[0]
    assert n_after == n_before


def test_promote_skips_below_threshold(patched_paths, conn):
    _seed_queries(2, conn=conn, boarding=41220, alighting=41450)
    promoted = promote_popular(conn, now=T0 + timedelta(minutes=1), min_count=3)
    assert promoted == []


def test_promote_respects_window(patched_paths, conn):
    """Queries older than the rolling window don't count."""
    _seed_queries(5, conn=conn, boarding=41220, alighting=41450)
    # Now is 8 days later -- everything is outside the 7-day window.
    promoted = promote_popular(
        conn, now=T0 + timedelta(days=8),
        min_count=3, window=timedelta(days=7),
    )
    assert promoted == []
