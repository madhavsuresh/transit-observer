"""Predictor-agnostic probabilistic-forecast diagnostics."""

from __future__ import annotations

import math

import pytest

from transit_observer.predictors.diagnostics import (
    aggregate_coverage,
    aggregate_crps,
    coverage_gap,
    crps_from_quantiles,
    crps_from_quantiles_batch,
    decision_score,
    diagnose_row,
    interval_score,
    pinball_loss,
    tail_miss,
)


def test_pinball_loss_is_zero_when_predicted_equals_actual():
    assert pinball_loss(100.0, 100.0, 0.5) == 0.0
    assert pinball_loss(100.0, 100.0, 0.9) == 0.0


def test_pinball_loss_asymmetric_around_quantile():
    # q=0.9: under-prediction (actual > predicted) costs more
    under = pinball_loss(110.0, 100.0, 0.9)
    over = pinball_loss(90.0, 100.0, 0.9)
    assert under == pytest.approx(0.9 * 10.0)
    assert over == pytest.approx(0.1 * 10.0)
    assert under > over


def test_crps_zero_when_distribution_concentrated_at_actual():
    quantiles = {0.5: 50.0, 0.8: 50.0, 0.9: 50.0}
    assert crps_from_quantiles(quantiles, 50.0) == pytest.approx(0.0, abs=1e-9)


def test_crps_positive_when_distribution_misses():
    quantiles = {0.5: 100.0, 0.8: 120.0, 0.9: 140.0}
    # actual=200 (way above the upper tail)
    crps = crps_from_quantiles(quantiles, 200.0)
    assert crps > 0


def test_crps_handles_nan_gracefully():
    quantiles = {0.5: float("nan"), 0.8: 100.0, 0.9: 120.0}
    # Should still produce a finite number from the finite quantiles
    out = crps_from_quantiles(quantiles, 110.0)
    assert math.isfinite(out)


def test_interval_score_penalizes_misses():
    inside = interval_score(50.0, 0.0, 100.0, alpha=0.2)   # actual inside
    outside_high = interval_score(200.0, 0.0, 100.0, alpha=0.2)
    outside_low = interval_score(-50.0, 0.0, 100.0, alpha=0.2)
    assert inside == 100.0  # just the width
    # Outside high: width + (2/0.2)*(200-100) = 100 + 1000 = 1100
    assert outside_high == pytest.approx(1100.0)
    # Outside low: width + (2/0.2)*(0 - (-50)) = 100 + 500 = 600
    assert outside_low == pytest.approx(600.0)


def test_interval_score_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        interval_score(50.0, 0.0, 100.0, alpha=0.0)
    with pytest.raises(ValueError):
        interval_score(50.0, 0.0, 100.0, alpha=1.0)


def test_tail_miss_returns_true_only_when_actual_above_upper():
    assert tail_miss(100.0, 90.0) is True
    assert tail_miss(50.0, 90.0) is False
    assert tail_miss(float("nan"), 90.0) is False


def test_diagnose_row_packages_everything():
    quantiles = {0.5: 100.0, 0.8: 130.0, 0.9: 160.0}
    d = diagnose_row(quantiles, actual=140.0)
    assert d.pinball_q80 > 0       # actual > p80
    assert d.p80_covered is False
    assert d.p90_covered is True
    assert d.tail_miss_p90 is False
    assert d.interval_score_central_40 > 0


def test_aggregate_crps_ignores_nans():
    assert aggregate_crps([1.0, float("nan"), 3.0]) == pytest.approx(2.0)
    assert math.isnan(aggregate_crps([]))
    assert math.isnan(aggregate_crps([float("nan")]))


def test_aggregate_coverage_simple():
    assert aggregate_coverage([True, True, False, False]) == 0.5
    assert aggregate_coverage([True, True, True, True]) == 1.0
    assert math.isnan(aggregate_coverage([]))


def test_coverage_gap_symmetric():
    assert coverage_gap(0.85, 0.8) == pytest.approx(0.05)
    assert coverage_gap(0.75, 0.8) == pytest.approx(0.05)


def test_decision_score_combines_crps_and_calibration():
    s1 = decision_score(100.0, 0.0)
    s2 = decision_score(100.0, 0.10)
    assert s2 > s1   # bigger gap hurts the score


def test_crps_decreases_when_distribution_recenters_to_actual():
    far = crps_from_quantiles({0.5: 100.0, 0.8: 120.0, 0.9: 140.0}, 200.0)
    near = crps_from_quantiles({0.5: 190.0, 0.8: 200.0, 0.9: 210.0}, 200.0)
    assert near < far


def test_crps_batch_matches_scalar():
    """The vectorized CRPS must agree with the scalar implementation."""
    import numpy as np
    rng = np.random.default_rng(0)
    alphas = np.array([0.5, 0.8, 0.9])
    n = 200
    # Random monotone quantile triples and actuals
    p50 = rng.uniform(30.0, 600.0, size=n)
    spread1 = rng.uniform(10.0, 200.0, size=n)
    spread2 = rng.uniform(5.0, 80.0, size=n)
    p80 = p50 + spread1
    p90 = p80 + spread2
    actuals = rng.uniform(0.0, 1000.0, size=n)
    qvals = np.stack([p50, p80, p90], axis=1)

    batch = crps_from_quantiles_batch(
        quantile_levels=alphas, quantile_values=qvals, actuals=actuals,
    )
    scalar = np.array([
        crps_from_quantiles({0.5: p50[i], 0.8: p80[i], 0.9: p90[i]}, actuals[i])
        for i in range(n)
    ])
    np.testing.assert_allclose(batch, scalar, rtol=1e-10, atol=1e-9)


def test_crps_batch_propagates_nan():
    import numpy as np
    alphas = np.array([0.5, 0.8, 0.9])
    qvals = np.array([
        [100.0, 120.0, 140.0],
        [np.nan, 120.0, 140.0],
    ])
    actuals = np.array([200.0, 200.0])
    out = crps_from_quantiles_batch(
        quantile_levels=alphas, quantile_values=qvals, actuals=actuals,
    )
    assert np.isfinite(out[0])
    assert np.isnan(out[1])


def test_crps_batch_unsorted_alphas():
    """The batched version should sort internally — caller doesn't have to."""
    import numpy as np
    alphas_sorted = np.array([0.5, 0.8, 0.9])
    alphas_shuffled = np.array([0.9, 0.5, 0.8])
    qvals_sorted = np.array([[100.0, 120.0, 140.0]])
    # Same quantile values, columns shuffled to match alphas_shuffled
    qvals_shuffled = np.array([[140.0, 100.0, 120.0]])
    actuals = np.array([130.0])
    a = crps_from_quantiles_batch(
        quantile_levels=alphas_sorted, quantile_values=qvals_sorted, actuals=actuals,
    )
    b = crps_from_quantiles_batch(
        quantile_levels=alphas_shuffled, quantile_values=qvals_shuffled, actuals=actuals,
    )
    np.testing.assert_allclose(a, b, rtol=1e-12)
