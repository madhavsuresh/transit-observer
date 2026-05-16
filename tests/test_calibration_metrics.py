"""reliability_curve and pit_histogram against an in-memory fixture DB."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from statistics import NormalDist

import duckdb
import pytest

from transit_observer import db
from transit_observer.metrics import (
    diagnose_pit_shape,
    historical_prediction,
    live_data_diagnostic,
    per_line_resolved_counts,
    pit_histogram,
    pit_histogram_aggregated,
    reliability_curve,
    reliability_curve_aggregated,
)
from transit_observer.metrics import PitBin


T0 = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
_Z80 = 0.8416212335729143


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    db.init_schema(c)
    try:
        yield c
    finally:
        c.close()


def _insert_calibrated_lognormal(
    conn: duckdb.DuckDBPyConnection,
    *,
    line: str,
    mu: float,
    sigma: float,
    n: int,
    seed: int = 0,
    truth_conf: float = 1.0,
) -> None:
    """Insert n resolved forecasts drawn from a log-normal that matches
    the stored (p50, p80) — i.e. a perfectly-calibrated kernel."""
    import random
    rng = random.Random(seed)
    z_norm = NormalDist()
    p50 = math.exp(mu)
    p80 = math.exp(mu + sigma * _Z80)
    p90 = math.exp(mu + sigma * z_norm.inv_cdf(0.9))
    for i in range(n):
        forecast_id = f"{line}-{i}"
        actual = math.exp(mu + sigma * z_norm.inv_cdf(rng.random()))
        leave_at = T0 + timedelta(minutes=i)
        conn.execute(
            """
            INSERT INTO forecast_queue (
                forecast_id, enqueued_at, snapshot_polled_at, leave_at,
                mode, line, direction_code,
                boarding_map_id, alighting_map_id,
                predicted_total_p50, predicted_total_p80, predicted_total_p90,
                resolve_after, status
            ) VALUES (?, ?, ?, ?, 'L', ?, '1', 0, 0, ?, ?, ?, ?, 'resolved')
            """,
            [forecast_id, leave_at, leave_at, leave_at, line,
             p50, p80, p90, leave_at + timedelta(minutes=60)],
        )
        conn.execute(
            """
            INSERT INTO forecast_outcomes (
                forecast_id, resolved_at, actual_total_seconds,
                in_p80_window, in_p90_window,
                p50_residual_seconds, p80_residual_seconds,
                truth_confidence, failed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, FALSE)
            """,
            [forecast_id, leave_at + timedelta(minutes=30), actual,
             actual <= p80, actual <= p90,
             actual - p50, actual - p80, truth_conf],
        )


def test_reliability_curve_well_calibrated_lognormal_is_near_diagonal(conn):
    """When actuals are drawn from the fitted log-normal, the empirical
    coverage at each nominal q should be close to q itself."""
    _insert_calibrated_lognormal(conn, line="Red", mu=6.2, sigma=0.4, n=1500)
    points = reliability_curve(conn, min_samples=100)
    red_points = [p for p in points if p.line == "Red"]
    assert len(red_points) == 9  # default grid is 9 points
    for p in red_points:
        assert abs(p.empirical_coverage - p.nominal_quantile) < 0.05, (
            f"empirical={p.empirical_coverage:.3f} far from "
            f"nominal={p.nominal_quantile:.3f}"
        )


def test_reliability_curve_skips_lines_below_min_samples(conn):
    _insert_calibrated_lognormal(conn, line="Red", mu=6.2, sigma=0.4, n=20)
    points = reliability_curve(conn, min_samples=30)
    assert points == []


def test_reliability_curve_filters_low_truth_confidence(conn):
    _insert_calibrated_lognormal(conn, line="Red", mu=6.2, sigma=0.4, n=100, truth_conf=0.2)
    points = reliability_curve(conn, min_samples=30, min_truth_confidence=0.5)
    assert points == []


def test_pit_histogram_well_calibrated_lognormal_is_approximately_flat(conn):
    _insert_calibrated_lognormal(conn, line="Red", mu=6.2, sigma=0.4, n=2000)
    bins = pit_histogram(conn, n_bins=10, min_samples=100)
    red_bins = [b for b in bins if b.line == "Red"]
    assert len(red_bins) == 10
    # Each bin should hold ≈10% of the mass; allow generous slack at this n.
    for b in red_bins:
        share = b.count / 2000
        assert 0.06 < share < 0.14, f"bin {b.bin_lower:.1f}-{b.bin_upper:.1f} share {share:.3f}"


def test_pit_histogram_distinguishes_lines(conn):
    _insert_calibrated_lognormal(conn, line="Red", mu=6.2, sigma=0.4, n=500, seed=1)
    _insert_calibrated_lognormal(conn, line="Blue", mu=5.5, sigma=0.6, n=500, seed=2)
    bins = pit_histogram(conn, n_bins=10, min_samples=100)
    lines_present = {b.line for b in bins}
    assert lines_present == {"Red", "Blue"}


def test_reliability_curve_filter_by_line(conn):
    _insert_calibrated_lognormal(conn, line="Red", mu=6.2, sigma=0.4, n=200, seed=3)
    _insert_calibrated_lognormal(conn, line="Blue", mu=5.5, sigma=0.6, n=200, seed=4)
    only_red = reliability_curve(conn, line="Red", min_samples=50)
    assert {p.line for p in only_red} == {"Red"}


def test_pit_aggregated_pools_all_lines(conn):
    _insert_calibrated_lognormal(conn, line="Red", mu=6.2, sigma=0.4, n=200, seed=5)
    _insert_calibrated_lognormal(conn, line="Blue", mu=5.5, sigma=0.6, n=200, seed=6)
    bins = pit_histogram_aggregated(conn, n_bins=10)
    assert len(bins) == 10
    assert all(b.line == "ALL" for b in bins)
    assert sum(b.count for b in bins) == 400


def test_reliability_aggregated_pools_all_lines(conn):
    _insert_calibrated_lognormal(conn, line="Red", mu=6.2, sigma=0.4, n=200, seed=7)
    _insert_calibrated_lognormal(conn, line="Blue", mu=5.5, sigma=0.6, n=200, seed=8)
    points = reliability_curve_aggregated(conn)
    assert len(points) == 9
    assert all(p.line == "ALL" for p in points)
    # Combined pool should still be approximately calibrated.
    for p in points:
        assert abs(p.empirical_coverage - p.nominal_quantile) < 0.07


def test_pit_aggregated_returns_empty_when_no_data(conn):
    assert pit_histogram_aggregated(conn, n_bins=10) == []


def test_reliability_aggregated_returns_empty_when_no_data(conn):
    assert reliability_curve_aggregated(conn) == []


def test_per_line_resolved_counts(conn):
    _insert_calibrated_lognormal(conn, line="Red", mu=6.2, sigma=0.4, n=50, seed=9)
    _insert_calibrated_lognormal(conn, line="Blue", mu=5.5, sigma=0.6, n=25, seed=10)
    counts = per_line_resolved_counts(conn)
    by_line = {(c.mode, c.line): c.n_resolved for c in counts}
    assert by_line.get(("L", "Red")) == 50
    assert by_line.get(("L", "Blue")) == 25


def test_per_line_resolved_counts_high_conf_filter(conn):
    _insert_calibrated_lognormal(conn, line="Red", mu=6.2, sigma=0.4, n=10, truth_conf=0.2, seed=11)
    _insert_calibrated_lognormal(conn, line="Blue", mu=6.2, sigma=0.4, n=10, truth_conf=0.9, seed=12)
    counts = per_line_resolved_counts(conn, min_truth_confidence=0.5)
    red = next(c for c in counts if c.line == "Red")
    blue = next(c for c in counts if c.line == "Blue")
    assert red.n_resolved == 10
    assert red.n_resolved_high_conf == 0   # truth_conf=0.2 below threshold
    assert blue.n_resolved == 10
    assert blue.n_resolved_high_conf == 10


def test_historical_prediction_empirical_quantiles(conn):
    # Insert 100 resolved L-mode trips for the same OD with known
    # actuals so we can check the empirical quantiles.
    for i in range(100):
        forecast_id = f"hist-{i}"
        leave_at = T0 + timedelta(minutes=i)
        actual_seconds = 600 + i * 6   # 600..1194s, evenly spaced
        conn.execute(
            """
            INSERT INTO forecast_queue (
                forecast_id, enqueued_at, snapshot_polled_at, leave_at,
                mode, line, direction_code,
                boarding_map_id, alighting_map_id,
                predicted_total_p50, predicted_total_p80, predicted_total_p90,
                resolve_after, status
            ) VALUES (?, ?, ?, ?, 'L', 'Red', '1', 1234, 5678, 700, 900, 1000, ?, 'resolved')
            """,
            [forecast_id, leave_at, leave_at, leave_at, leave_at + timedelta(minutes=60)],
        )
        conn.execute(
            """
            INSERT INTO forecast_outcomes (
                forecast_id, resolved_at, actual_total_seconds,
                in_p80_window, in_p90_window, truth_confidence, failed
            ) VALUES (?, ?, ?, TRUE, TRUE, 0.9, FALSE)
            """,
            [forecast_id, leave_at + timedelta(minutes=30), actual_seconds],
        )
    hist = historical_prediction(
        conn, mode="L", line="Red",
        boarding_int_id=1234, alighting_int_id=5678,
    )
    assert hist is not None
    assert hist.n_samples == 100
    # Linearly spaced 600..1194 → p50 ≈ midpoint, p80 ≈ 80%-ile
    assert 870 < hist.p50_seconds < 920
    assert 1070 < hist.p80_seconds < 1120
    assert 1130 < hist.p90_seconds < 1200


def test_historical_prediction_returns_none_below_threshold(conn):
    # Only 4 resolved rows — below the n>=5 cutoff.
    for i in range(4):
        forecast_id = f"sparse-{i}"
        leave_at = T0 + timedelta(minutes=i)
        conn.execute(
            """
            INSERT INTO forecast_queue (
                forecast_id, enqueued_at, snapshot_polled_at, leave_at,
                mode, line, direction_code, boarding_map_id, alighting_map_id,
                predicted_total_p50, predicted_total_p80, predicted_total_p90,
                resolve_after, status
            ) VALUES (?, ?, ?, ?, 'L', 'Red', '1', 99, 99, 700, 900, 1000, ?, 'resolved')
            """,
            [forecast_id, leave_at, leave_at, leave_at, leave_at + timedelta(minutes=60)],
        )
        conn.execute(
            """
            INSERT INTO forecast_outcomes (
                forecast_id, resolved_at, actual_total_seconds,
                in_p80_window, in_p90_window, truth_confidence, failed
            ) VALUES (?, ?, ?, TRUE, TRUE, 0.9, FALSE)
            """,
            [forecast_id, leave_at + timedelta(minutes=30), 800],
        )
    assert historical_prediction(
        conn, mode="L", line="Red",
        boarding_int_id=99, alighting_int_id=99,
    ) is None


def test_live_data_diagnostic_reports_zero_when_empty(conn):
    diag = live_data_diagnostic(
        conn, mode="L", line="Red", boarding_int_id=1234,
        now=T0,
    )
    assert diag.raw_rows_in_window == 0
    assert diag.future_rows == 0
    assert diag.last_raw_polled_at is None


def test_live_data_diagnostic_counts_l_arrivals(conn):
    # Seed 3 raw rows: one in the past, one approaching, one future.
    for i, offset_s in enumerate([-120, 30, 300]):
        conn.execute(
            """
            INSERT INTO train_arrivals_raw (
                polled_at, line, run_number, map_id, stop_id,
                direction_code, predicted_at, arrival_at,
                is_approaching, is_delayed, is_fault, is_scheduled
            ) VALUES (?, 'Red', ?, 1234, 0, '1', ?, ?, FALSE, FALSE, FALSE, FALSE)
            """,
            [
                T0 - timedelta(seconds=10), f"R{i}",
                T0 - timedelta(seconds=10), T0 + timedelta(seconds=offset_s),
            ],
        )
    diag = live_data_diagnostic(
        conn, mode="L", line="Red", boarding_int_id=1234, now=T0,
    )
    # All 3 within the polled_at >= T0-5min and arrival_at <= T0+30min window
    assert diag.raw_rows_in_window == 3
    # 2 with arrival_at >= now (the +30s and +300s ones)
    assert diag.future_rows == 2


def test_diagnose_pit_shape_flat_returns_calibrated():
    bins = [
        PitBin(line="ALL", bin_lower=i/10, bin_upper=(i+1)/10, count=100, density=1.0)
        for i in range(10)
    ]
    diagnosis = diagnose_pit_shape(bins)
    assert "calibrated" in diagnosis.lower()


def test_diagnose_pit_shape_u_shape_detected():
    # Heavy mass in first and last bins, light middle.
    counts = [400, 50, 50, 50, 50, 50, 50, 50, 50, 400]
    bins = [
        PitBin(line="ALL", bin_lower=i/10, bin_upper=(i+1)/10,
               count=c, density=c / sum(counts) / 0.1)
        for i, c in enumerate(counts)
    ]
    diagnosis = diagnose_pit_shape(bins)
    assert "U-shape" in diagnosis or "too tight" in diagnosis


def test_diagnose_pit_shape_right_skew_actuals_slower():
    counts = [10, 10, 10, 30, 50, 80, 150, 200, 250, 210]
    bins = [
        PitBin(line="ALL", bin_lower=i/10, bin_upper=(i+1)/10,
               count=c, density=c / sum(counts) / 0.1)
        for i, c in enumerate(counts)
    ]
    diagnosis = diagnose_pit_shape(bins)
    assert "slower" in diagnosis or "right" in diagnosis
