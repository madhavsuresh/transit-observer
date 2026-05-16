"""reliability_curve and pit_histogram against an in-memory fixture DB."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from statistics import NormalDist

import duckdb
import pytest

from transit_observer import db
from transit_observer.metrics import pit_histogram, reliability_curve


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
