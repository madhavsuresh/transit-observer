"""Tests for ``train_v2.calibration.refresh_train_residual_quantiles``."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timedelta

import duckdb
import pytest

from transit_observer.config import CHICAGO
from transit_observer.db import init_schema
from transit_observer.train_v2.calibration import refresh_train_residual_quantiles


@pytest.fixture()
def conn():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test.duckdb")
    conn = duckdb.connect(path)
    init_schema(conn)
    try:
        yield conn
    finally:
        conn.close()
        import shutil

        shutil.rmtree(tmpdir)


def _seed_outcome(
    conn,
    *,
    line: str,
    map_id: int,
    direction_code: str,
    predicted_wait_p50: float,
    actual_wait_seconds: float,
    data_quality: str = "GOOD",
    truth_confidence: float = 1.0,
    predictor_version: str = "train-telemetry-v1",
) -> None:
    fid = uuid.uuid4().hex
    now = datetime.now(CHICAGO)
    feature_json = json.dumps({"data_quality": data_quality, "line": line, "map_id": map_id})
    conn.execute(
        """
        INSERT INTO forecast_queue(
            forecast_id, enqueued_at, snapshot_polled_at, leave_at,
            mode, line, direction_code, corridor_id, predictor_version, feature_json,
            boarding_map_id, boarding_text_id, boarding_station_name,
            alighting_map_id, alighting_text_id, alighting_station_name,
            predicted_wait_mean, predicted_wait_p50, predicted_wait_p80, predicted_wait_p90,
            predicted_in_vehicle_mean,
            predicted_total_mean, predicted_total_p50, predicted_total_p80, predicted_total_p90,
            predicted_failure_prob, resolve_after, status
        ) VALUES (?, ?, ?, ?, 'L', ?, ?, NULL, ?, ?,
                  ?, NULL, NULL, 0, NULL, NULL,
                  ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  0.0, ?, 'resolved')
        """,
        [
            fid, now, now, now, line, direction_code, predictor_version, feature_json,
            map_id,
            predicted_wait_p50, predicted_wait_p50, predicted_wait_p50 + 30, predicted_wait_p50 + 60,
            180.0,
            predicted_wait_p50 + 180, predicted_wait_p50 + 180, predicted_wait_p50 + 210, predicted_wait_p50 + 240,
            now + timedelta(seconds=600),
        ],
    )
    conn.execute(
        """
        INSERT INTO forecast_outcomes(
            forecast_id, resolved_at, boarded_run_number, boarded_at, alighted_at,
            actual_wait_seconds, actual_in_vehicle_seconds, actual_total_seconds,
            in_p80_window, in_p90_window, p50_residual_seconds, p80_residual_seconds,
            truth_confidence, failed
        ) VALUES (?, ?, NULL, ?, ?, ?, 180.0, ?, FALSE, FALSE, NULL, NULL, ?, FALSE)
        """,
        [
            fid, now, now, now + timedelta(seconds=10),
            actual_wait_seconds,
            actual_wait_seconds + 180.0,
            truth_confidence,
        ],
    )


def test_dense_cell_writes_quantile_row(conn):
    residuals = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15,
                 20, 25, 30, -28, -18, -8, 8, 18, 28, 0]
    for delta in residuals:
        _seed_outcome(
            conn, line="Blue", map_id=40380, direction_code="inbound",
            predicted_wait_p50=120.0, actual_wait_seconds=120.0 + delta,
        )
    n = refresh_train_residual_quantiles(conn, min_n=20)
    assert n == 1
    row = conn.execute(
        "SELECT line, map_id, direction_code, horizon_bin, quality_bin, n, q50_s, mae_s "
        "FROM train_v2_residual_quantile"
    ).fetchone()
    line, map_id, dir_c, hbin, qbin, count, q50, mae = row
    assert (line, map_id, dir_c, qbin) == ("Blue", "40380", "inbound", "high")
    assert hbin == "0_2m"  # 120s = 2.0 minutes, boundary inclusive
    assert count == 20
    assert q50 == pytest.approx(0.0, abs=2.0)
    assert mae > 0


def test_low_truth_confidence_outcomes_excluded(conn):
    for _ in range(25):
        _seed_outcome(
            conn, line="Blue", map_id=40380, direction_code="inbound",
            predicted_wait_p50=120.0, actual_wait_seconds=130.0,
            truth_confidence=0.2,
        )
    n = refresh_train_residual_quantiles(conn, min_n=20, truth_confidence_floor=0.5)
    assert n == 0


def test_distinct_quality_bins_split(conn):
    for _ in range(20):
        _seed_outcome(
            conn, line="Blue", map_id=40380, direction_code="inbound",
            predicted_wait_p50=120.0, actual_wait_seconds=120.0,
            data_quality="GOOD",
        )
    for _ in range(20):
        _seed_outcome(
            conn, line="Blue", map_id=40380, direction_code="inbound",
            predicted_wait_p50=120.0, actual_wait_seconds=130.0,
            data_quality="DEGRADED",
        )
    n = refresh_train_residual_quantiles(conn, min_n=20)
    assert n == 2
    bins = {r[0] for r in conn.execute("SELECT quality_bin FROM train_v2_residual_quantile").fetchall()}
    assert bins == {"high", "low"}
