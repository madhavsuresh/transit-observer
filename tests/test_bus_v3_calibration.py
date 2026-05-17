"""Tests for bus_v3 residual-quantile calibration refresh."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timedelta

import duckdb
import pytest

from transit_observer.bus_v3.calibration import refresh_bus_residual_quantiles
from transit_observer.config import CHICAGO
from transit_observer.db import init_schema


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
    conn: duckdb.DuckDBPyConnection,
    *,
    rt: str,
    stpid: str,
    rtdir: str,
    predicted_wait_p50: float,
    actual_wait_seconds: float,
    data_quality: str = "GOOD",
    truth_confidence: float = 1.0,
    predictor_version: str = "bus-telemetry-v1",
) -> None:
    fid = uuid.uuid4().hex
    now = datetime.now(CHICAGO)
    feature_json = json.dumps({"data_quality": data_quality, "rt": rt, "stpid": stpid, "rtdir": rtdir})
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
        ) VALUES (?, ?, ?, ?, 'bus', ?, ?, NULL, ?, ?,
                  0, ?, ?, 0, NULL, NULL,
                  ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  0.0, ?, 'resolved')
        """,
        [
            fid, now, now, now,
            rt, rtdir, predictor_version, feature_json,
            stpid, "Test Boarding",
            predicted_wait_p50, predicted_wait_p50, predicted_wait_p50 + 30, predicted_wait_p50 + 60,
            120.0,
            predicted_wait_p50 + 120, predicted_wait_p50 + 120, predicted_wait_p50 + 150, predicted_wait_p50 + 180,
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
        ) VALUES (?, ?, NULL, ?, ?, ?, 120.0, ?, FALSE, FALSE, NULL, NULL, ?, FALSE)
        """,
        [
            fid, now, now, now + timedelta(seconds=10), actual_wait_seconds,
            actual_wait_seconds + 120.0,
            truth_confidence,
        ],
    )


def test_refresh_writes_quantiles_for_a_dense_cell(conn):
    """20 outcomes in one (rt, stpid, rtdir, horizon_bin, quality_bin) cell
    yields one bus_v3_residual_quantile row."""
    # Predicted wait ~120s (horizon bin "2_5m"). Actuals jitter ±30s.
    residuals = [-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0,
                 20.0, 25.0, 30.0, -28.0, -18.0, -8.0, 8.0, 18.0, 28.0, 0.0]
    for delta in residuals:
        _seed_outcome(
            conn, rt="22", stpid="1828", rtdir="Southbound",
            predicted_wait_p50=120.0, actual_wait_seconds=120.0 + delta,
            data_quality="GOOD",
        )
    n = refresh_bus_residual_quantiles(conn, min_n=20)
    assert n == 1
    row = conn.execute(
        "SELECT rt, stpid, rtdir, horizon_bin, quality_bin, n, q50_s, mae_s FROM bus_v3_residual_quantile"
    ).fetchone()
    rt, stpid, rtdir, hbin, qbin, count, q50, mae = row
    assert (rt, stpid, rtdir, qbin) == ("22", "1828", "Southbound", "high")
    assert hbin == "0_2m"  # predicted_wait_p50 = 120s = 2.0 minutes → boundary inclusive
    assert count == 20
    assert q50 == pytest.approx(0.0, abs=2.0)
    assert mae > 0


def test_sparse_cells_are_skipped(conn):
    """A cell with fewer than min_n samples doesn't produce a row."""
    for delta in (-5.0, 5.0, 0.0):
        _seed_outcome(
            conn, rt="22", stpid="1828", rtdir="Southbound",
            predicted_wait_p50=120.0, actual_wait_seconds=120.0 + delta,
        )
    n = refresh_bus_residual_quantiles(conn, min_n=20)
    assert n == 0


def test_low_truth_confidence_outcomes_excluded(conn):
    """Outcomes with truth_confidence below the floor are filtered out."""
    for _ in range(25):
        _seed_outcome(
            conn, rt="22", stpid="1828", rtdir="Southbound",
            predicted_wait_p50=120.0, actual_wait_seconds=130.0,
            truth_confidence=0.2,
        )
    n = refresh_bus_residual_quantiles(conn, min_n=20, truth_confidence_floor=0.5)
    assert n == 0


def test_different_quality_bins_yield_separate_cells(conn):
    """GOOD vs DEGRADED quality goes into separate strata."""
    for delta in [0.0] * 20:
        _seed_outcome(
            conn, rt="22", stpid="1828", rtdir="Southbound",
            predicted_wait_p50=120.0, actual_wait_seconds=120.0 + delta,
            data_quality="GOOD",
        )
    for delta in [10.0] * 20:
        _seed_outcome(
            conn, rt="22", stpid="1828", rtdir="Southbound",
            predicted_wait_p50=120.0, actual_wait_seconds=120.0 + delta,
            data_quality="DEGRADED",
        )
    n = refresh_bus_residual_quantiles(conn, min_n=20)
    assert n == 2
    bins = {
        r[0] for r in conn.execute(
            "SELECT quality_bin FROM bus_v3_residual_quantile"
        ).fetchall()
    }
    assert bins == {"high", "low"}
