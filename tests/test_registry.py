"""Predictor registry promotion + anti-flap logic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from transit_observer import db
from transit_observer.predictors import registry as pred_registry
from transit_observer.predictors.journey_kernel import KERNEL_EB_VERSION, KERNEL_VERSION
from transit_observer.predictors.quantile_gbm import GBM_VERSION


T0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    db.init_schema(c)
    yield c
    c.close()


def test_active_version_defaults_to_kernel(conn):
    pred_registry.reset()
    v = pred_registry.active_version_for_corridor(conn, corridor_id="anywhere")
    assert v == KERNEL_VERSION


def test_set_active_manual_override(conn):
    pred_registry.reset()
    pred_registry.set_active(
        conn, corridor_id="c1", predictor_version=KERNEL_EB_VERSION, now=T0,
    )
    assert pred_registry.active_version_for_corridor(conn, corridor_id="c1") == KERNEL_EB_VERSION


def test_promote_requires_consecutive_wins(conn):
    """Anti-flap: one strong win isn't enough to switch."""
    pred_registry.reset()
    # Window 1: gbm beats kernel by 0.05 — should record streak=1, not switch
    switched = pred_registry.promote(
        conn, corridor_id="c1",
        candidate_version=GBM_VERSION,
        candidate_score=0.10, incumbent_score=0.15,
        now=T0,
    )
    assert switched is False
    assert pred_registry.active_version_for_corridor(conn, corridor_id="c1") == KERNEL_VERSION

    # Window 2: gbm wins again -> streak=2 -> switch
    switched = pred_registry.promote(
        conn, corridor_id="c1",
        candidate_version=GBM_VERSION,
        candidate_score=0.11, incumbent_score=0.15,
        now=T0 + timedelta(days=1),
    )
    assert switched is True
    assert pred_registry.active_version_for_corridor(conn, corridor_id="c1") == GBM_VERSION


def test_promote_resets_on_loss(conn):
    """If the candidate loses a window mid-streak, the streak resets."""
    pred_registry.reset()
    # Win 1
    pred_registry.promote(
        conn, corridor_id="c1",
        candidate_version=GBM_VERSION,
        candidate_score=0.10, incumbent_score=0.15,
        now=T0,
    )
    # Lose (candidate didn't beat incumbent)
    pred_registry.promote(
        conn, corridor_id="c1",
        candidate_version=GBM_VERSION,
        candidate_score=0.16, incumbent_score=0.15,
        now=T0 + timedelta(days=1),
    )
    # Win again — should NOT switch (streak was reset to 0, this is win 1)
    switched = pred_registry.promote(
        conn, corridor_id="c1",
        candidate_version=GBM_VERSION,
        candidate_score=0.10, incumbent_score=0.15,
        now=T0 + timedelta(days=2),
    )
    assert switched is False
    assert pred_registry.active_version_for_corridor(conn, corridor_id="c1") == KERNEL_VERSION


def test_promote_ignores_below_margin(conn):
    """Insufficient improvement (within margin) doesn't count as a win."""
    pred_registry.reset()
    # Margin defaults to 0.005; candidate beats by only 0.001
    switched = pred_registry.promote(
        conn, corridor_id="c1",
        candidate_version=GBM_VERSION,
        candidate_score=0.149, incumbent_score=0.150,
        now=T0,
    )
    assert switched is False
    switched = pred_registry.promote(
        conn, corridor_id="c1",
        candidate_version=GBM_VERSION,
        candidate_score=0.149, incumbent_score=0.150,
        now=T0 + timedelta(days=1),
    )
    assert switched is False
    assert pred_registry.active_version_for_corridor(conn, corridor_id="c1") == KERNEL_VERSION


def test_alternating_candidates_dont_flap(conn):
    """When two candidates take turns winning, neither stays in the running."""
    pred_registry.reset()
    # GBM wins
    pred_registry.promote(
        conn, corridor_id="c1",
        candidate_version=GBM_VERSION,
        candidate_score=0.10, incumbent_score=0.15,
        now=T0,
    )
    # EB wins next round (different candidate -> resets the streak)
    pred_registry.promote(
        conn, corridor_id="c1",
        candidate_version=KERNEL_EB_VERSION,
        candidate_score=0.11, incumbent_score=0.15,
        now=T0 + timedelta(days=1),
    )
    # Back to GBM (streak still 1 because EB interrupted)
    switched = pred_registry.promote(
        conn, corridor_id="c1",
        candidate_version=GBM_VERSION,
        candidate_score=0.10, incumbent_score=0.15,
        now=T0 + timedelta(days=2),
    )
    assert switched is False
    assert pred_registry.active_version_for_corridor(conn, corridor_id="c1") == KERNEL_VERSION


def test_candidate_versions_includes_kernel(conn):
    pred_registry.reset()
    vs = pred_registry.candidate_versions(conn)
    assert KERNEL_VERSION in vs
    assert KERNEL_EB_VERSION in vs


def test_predictor_singleton_returns_same_instance(conn):
    pred_registry.reset()
    a = pred_registry.predictor(KERNEL_VERSION)
    b = pred_registry.predictor(KERNEL_VERSION)
    assert a is b
