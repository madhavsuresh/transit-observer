"""Log-normal fit, PIT, and quantile dotplot positions are pure math
and tested in isolation. See the project plan for why we use log-normal."""

from __future__ import annotations

import math
import random
from statistics import NormalDist

import pytest

from transit_observer.journey.quantile_distribution import (
    fit_lognormal_from_p50_p80,
    lognormal_cdf,
    lognormal_quantile,
    p90_fit_residual,
    pit_value,
    quantile_dotplot_positions,
)


# --- fit_lognormal_from_p50_p80 -----------------------------------------


def test_fit_round_trip():
    mu, sigma = 6.2, 0.4  # exp(6.2) ≈ 493s ≈ 8.2 min
    p50 = math.exp(mu)
    p80 = math.exp(mu + sigma * 0.8416212335729143)
    mu_out, sigma_out = fit_lognormal_from_p50_p80(p50, p80)
    assert math.isclose(mu_out, mu, abs_tol=1e-9)
    assert math.isclose(sigma_out, sigma, abs_tol=1e-9)


def test_fit_rejects_non_positive_quantiles():
    with pytest.raises(ValueError):
        fit_lognormal_from_p50_p80(0.0, 100.0)
    with pytest.raises(ValueError):
        fit_lognormal_from_p50_p80(-1.0, 100.0)


def test_fit_rejects_p80_le_p50():
    with pytest.raises(ValueError):
        fit_lognormal_from_p50_p80(300.0, 300.0)
    with pytest.raises(ValueError):
        fit_lognormal_from_p50_p80(300.0, 250.0)


def test_fit_rejects_non_finite():
    with pytest.raises(ValueError):
        fit_lognormal_from_p50_p80(float("nan"), 100.0)
    with pytest.raises(ValueError):
        fit_lognormal_from_p50_p80(100.0, float("inf"))


# --- lognormal_cdf and lognormal_quantile -------------------------------


def test_cdf_quantile_inverse():
    mu, sigma = 6.2, 0.4
    for p in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95):
        x = lognormal_quantile(p, mu, sigma)
        assert math.isclose(lognormal_cdf(x, mu, sigma), p, abs_tol=1e-9)


def test_cdf_at_p50_is_half():
    mu, sigma = 5.0, 0.5
    assert math.isclose(lognormal_cdf(math.exp(mu), mu, sigma), 0.5, abs_tol=1e-12)


def test_cdf_zero_returns_zero():
    assert lognormal_cdf(0.0, 5.0, 0.5) == 0.0
    assert lognormal_cdf(-10.0, 5.0, 0.5) == 0.0


def test_quantile_rejects_p_out_of_range():
    with pytest.raises(ValueError):
        lognormal_quantile(0.0, 5.0, 0.5)
    with pytest.raises(ValueError):
        lognormal_quantile(1.0, 5.0, 0.5)
    with pytest.raises(ValueError):
        lognormal_quantile(-0.1, 5.0, 0.5)


# --- pit_value ----------------------------------------------------------


def test_pit_value_round_trip_at_p80():
    p50 = 600.0
    p80 = 900.0
    pit = pit_value(p80, p50, p80)
    assert math.isclose(pit, 0.8, abs_tol=1e-9)


def test_pit_value_at_p50_is_half():
    pit = pit_value(600.0, 600.0, 900.0)
    assert math.isclose(pit, 0.5, abs_tol=1e-12)


def test_pit_value_returns_nan_on_bad_quantiles():
    assert math.isnan(pit_value(500.0, 0.0, 100.0))
    assert math.isnan(pit_value(500.0, 100.0, 100.0))
    assert math.isnan(pit_value(float("nan"), 100.0, 200.0))


def test_pit_values_are_approximately_uniform_for_calibrated_samples():
    """If we draw actuals from the same log-normal we're fitting to, the
    PIT distribution should look uniform on [0, 1]. This is the basic
    calibration sanity check."""
    rng = random.Random(42)
    mu, sigma = 6.2, 0.4
    p50 = math.exp(mu)
    p80 = math.exp(mu + sigma * 0.8416212335729143)
    z_norm = NormalDist()
    actuals = [math.exp(mu + sigma * z_norm.inv_cdf(rng.random())) for _ in range(2000)]
    pits = [pit_value(a, p50, p80) for a in actuals]
    # All ten deciles should have count roughly 200 ± a bit of slack.
    buckets = [0] * 10
    for pit in pits:
        idx = min(int(pit * 10), 9)
        buckets[idx] += 1
    for b in buckets:
        assert 140 < b < 270, f"PIT bucket count {b} far from uniform 200"


# --- quantile_dotplot_positions ----------------------------------------


def test_dotplot_position_count_and_sorted():
    positions = quantile_dotplot_positions(600.0, 900.0, n=50)
    assert len(positions) == 50
    assert positions == sorted(positions)


def test_dotplot_positions_endpoints_match_extreme_quantiles():
    p50, p80 = 600.0, 900.0
    mu, sigma = fit_lognormal_from_p50_p80(p50, p80)
    positions = quantile_dotplot_positions(p50, p80, n=50)
    # First dot is at the (0.5/50)=1%-ile, last at the 99%-ile.
    assert math.isclose(positions[0], lognormal_quantile(0.01, mu, sigma), abs_tol=1e-9)
    assert math.isclose(positions[-1], lognormal_quantile(0.99, mu, sigma), abs_tol=1e-9)


def test_dotplot_positions_n_default_is_50():
    """50 is the dot count Fernandes et al. (CHI 2018) recommend."""
    positions = quantile_dotplot_positions(600.0, 900.0)
    assert len(positions) == 50


def test_dotplot_rejects_zero_n():
    with pytest.raises(ValueError):
        quantile_dotplot_positions(600.0, 900.0, n=0)


# --- p90_fit_residual --------------------------------------------------


def test_p90_residual_zero_when_p90_matches_fit():
    p50, p80 = 600.0, 900.0
    mu, sigma = fit_lognormal_from_p50_p80(p50, p80)
    p90_consistent = lognormal_quantile(0.9, mu, sigma)
    residual = p90_fit_residual(p50, p80, p90_consistent)
    assert abs(residual) < 1e-9


def test_p90_residual_positive_when_tail_heavier_than_fit():
    p50, p80 = 600.0, 900.0
    mu, sigma = fit_lognormal_from_p50_p80(p50, p80)
    p90_fit = lognormal_quantile(0.9, mu, sigma)
    # Stored p90 25% larger ⇒ heavier tail than log-normal predicts.
    residual = p90_fit_residual(p50, p80, p90_fit * 1.25)
    assert math.isclose(residual, 0.25, abs_tol=1e-9)
