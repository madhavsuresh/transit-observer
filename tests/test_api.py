"""FastAPI predict endpoint smoke tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from transit_observer import api, db, query_log
from transit_observer.config import Settings


T0 = datetime(2026, 5, 14, 8, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch):
    """Point DB + queries.ndjson at temp paths so tests can't clobber prod."""
    new_settings = Settings(
        cta_train_api_key=None, cta_bus_api_key=None, metra_api_key=None,
        data_dir=tmp_path, logs_dir=tmp_path,
        db_path=tmp_path / "test.duckdb",
        read_replica_path=tmp_path / "test.duckdb",
    )
    monkeypatch.setattr("transit_observer.db.settings", new_settings)
    monkeypatch.setattr("transit_observer.config.settings", new_settings)
    monkeypatch.setattr(query_log, "QUERIES_PATH", tmp_path / "queries.ndjson")
    monkeypatch.setattr(query_log, "QUERIES_CURSOR_PATH", tmp_path / "queries.cursor")
    # Seed a fresh DB with some Red Line arrivals.
    c = duckdb.connect(str(new_settings.db_path))
    db.init_schema(c)
    for offset in [120, 600, 1200]:
        c.execute(
            """
            INSERT INTO train_arrivals_raw (
                polled_at, line, run_number, map_id, stop_id, station_name,
                direction_code, destination_name, predicted_at, arrival_at,
                is_approaching, is_delayed, is_fault, is_scheduled
            ) VALUES (?, 'Red', 'R1', 41320, 0, 'Belmont', '1', 'Loop',
                      ?, ?, FALSE, FALSE, FALSE, FALSE)
            """,
            [T0 - timedelta(seconds=30), T0 - timedelta(seconds=30), T0 + timedelta(seconds=offset)],
        )
    c.close()
    yield tmp_path


def test_healthz(isolated_paths):
    client = TestClient(api.app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_corridors_list(isolated_paths):
    client = TestClient(api.app)
    r = client.get("/corridors")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) > 30  # we have 60 seeded
    modes = {r["mode"] for r in rows}
    assert {"L", "bus", "metra", "intercampus"} <= modes


def test_predict_rejects_non_int_for_l(isolated_paths):
    client = TestClient(api.app)
    r = client.get("/predict?mode=L&line=Red&boarding=abc&alighting=41660")
    assert r.status_code == 400


def test_predict_logs_query_even_on_404(isolated_paths):
    """A failed prediction still appends to the NDJSON log."""
    client = TestClient(api.app)
    # No data exists for this OD -- should return 404 but still log.
    r = client.get("/predict?mode=L&line=Red&boarding=99999&alighting=99998")
    assert r.status_code == 404
    assert query_log.QUERIES_PATH.exists()
    assert query_log.QUERIES_PATH.read_text().strip()
