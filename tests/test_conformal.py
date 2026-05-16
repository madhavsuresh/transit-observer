"""DtACI / online ACI conformal wrapper."""

from __future__ import annotations

import random
from datetime import datetime, timezone

import duckdb
import pytest

from transit_observer import db
from transit_observer.predictors import conformal


T0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    db.init_schema(c)
    yield c
    c.close()


def test_initial_state_zero_offset(conn):
    state = conformal.load_state(
        conn,
        predictor_version="gbm-v1",
        line="Red", direction_code="south",
        leg="wait", quantile=0.9,
    )
    assert state.offset_seconds == 0.0
    assert state.n == 0
    assert state.coverage_target == 0.9


def test_aci_widens_after_miss(conn):
    """A single miss increases the offset (widens the upper bound)."""
    state = conformal.load_state(
        conn, predictor_version="gbm-v1",
        line="Red", direction_code="south",
        leg="wait", quantile=0.9,
    )
    initial_offset = state.offset_seconds
    # Raw prediction = 60s, actual = 120s -> miss -> offset should grow
    state.step(raw_quantile_seconds=60.0, observed_seconds=120.0)
    assert state.offset_seconds > initial_offset
    assert state.n == 1
    assert state.miscoverage_count == 1


def test_aci_tightens_after_excess_coverage(conn):
    """Many consecutive covers should pull the offset down (tighter)."""
    state = conformal.load_state(
        conn, predictor_version="gbm-v1",
        line="Red", direction_code="south",
        leg="wait", quantile=0.9,
    )
    # Inflate offset first
    for _ in range(20):
        state.step(raw_quantile_seconds=60.0, observed_seconds=120.0)
    after_misses = state.offset_seconds
    # Now feed a long run of covered observations
    for _ in range(200):
        state.step(raw_quantile_seconds=120.0, observed_seconds=50.0)
    assert state.offset_seconds < after_misses


def test_persistence_roundtrip(conn):
    """Persist + reload yields the same offset."""
    state = conformal.load_state(
        conn, predictor_version="gbm-v1",
        line="Red", direction_code="south",
        leg="wait", quantile=0.8,
    )
    state.step(60.0, 120.0)
    state.step(60.0, 70.0)
    conformal.persist_state(conn, state, now=T0)
    reloaded = conformal.load_state(
        conn, predictor_version="gbm-v1",
        line="Red", direction_code="south",
        leg="wait", quantile=0.8,
    )
    assert reloaded.offset_seconds == pytest.approx(state.offset_seconds)
    assert reloaded.n == state.n


def test_coverage_converges_to_target_under_stationary_residuals(conn):
    """The whole point: empirical coverage on a stationary stream converges to target."""
    rng = random.Random(42)
    state = conformal.load_state(
        conn, predictor_version="gbm-v1",
        line="Red", direction_code="south",
        leg="wait", quantile=0.9,
    )
    # Synthetic: actual ~ Normal(mu=60, sigma=30); raw_q90 = 60 (i.e. naive median)
    # True 0.9 quantile of N(60, 30) is ~ 60 + 1.28*30 ≈ 98.4
    # The conformal layer should learn an offset ≈ +38s
    for _ in range(2000):
        actual = max(0.0, rng.gauss(60.0, 30.0))
        state.step(raw_quantile_seconds=60.0, observed_seconds=actual)
    # Over the last 1000 obs, empirical coverage should be ~0.9
    # Use the running miscoverage_count over all observations as a proxy
    # (after warmup the trailing window dominates)
    coverage = state.coverage_observed
    assert coverage is not None
    assert 0.83 < coverage < 0.95   # generous bounds; the warmup biases it


def test_update_recovers_under_distribution_shift(conn):
    """ACI's marquee property: coverage returns to target after a drift."""
    rng = random.Random(7)
    state = conformal.load_state(
        conn, predictor_version="gbm-v1",
        line="Red", direction_code="south",
        leg="wait", quantile=0.9,
    )
    # Phase 1: stationary at mu=60
    for _ in range(800):
        actual = max(0.0, rng.gauss(60.0, 30.0))
        state.step(raw_quantile_seconds=60.0, observed_seconds=actual)
    offset_phase1 = state.offset_seconds
    # Phase 2: drift to mu=120 (huge upward shift)
    for _ in range(800):
        actual = max(0.0, rng.gauss(120.0, 30.0))
        state.step(raw_quantile_seconds=60.0, observed_seconds=actual)
    # Offset should grow significantly to chase the new distribution
    assert state.offset_seconds > offset_phase1 + 30.0


def test_offset_bounded():
    """Sanity: even a catastrophic data burst can't drive the offset off to infinity."""
    state = conformal.DtACIState(
        predictor_version="gbm-v1", line="Red", direction_code="south",
        leg="wait", quantile=0.9, coverage_target=0.9,
    )
    for _ in range(10_000):
        state.step(raw_quantile_seconds=60.0, observed_seconds=60_000.0)
    assert abs(state.offset_seconds) <= 1200.0
