"""TimeDistributionSummary parity with the Swift port.

Numbers come from running TimeDistributionSummaryTests.swift on the
canonical inputs. Any drift here means Swift and Python disagree on
the quantile primitive.
"""

from __future__ import annotations

import math

from transit_observer.journey.time_distribution import TimeDistributionSummary


def test_empirical_ten_samples_matches_swift_quantiles():
    samples = [float(i * 60) for i in range(1, 11)]
    s = TimeDistributionSummary.empirical(samples)
    assert s.sample_count == 10
    assert s.mean == 330
    assert s.p50 == 300
    assert s.p80 == 480
    assert s.p90 == 540


def test_empirical_empty_returns_zero():
    s = TimeDistributionSummary.empirical([])
    assert s == TimeDistributionSummary.zero()


def test_empirical_drops_negative_and_non_finite():
    s = TimeDistributionSummary.empirical([60, -10, math.inf, 120])
    assert s.sample_count == 2
    assert s.p50 == 60


def test_analytic_matches_gaussian_quantiles():
    s = TimeDistributionSummary.analytic(mean=600, sigma=120, confidence=0.7, sample_count=5)
    assert s.p50 == 600
    assert abs(s.p80 - (600 + 0.8416 * 120)) < 0.01
    assert abs(s.p90 - (600 + 1.2816 * 120)) < 0.01


def test_confidence_clamped_to_unit_interval():
    high = TimeDistributionSummary.analytic(mean=0, sigma=0, confidence=2.5)
    low = TimeDistributionSummary.analytic(mean=0, sigma=0, confidence=-0.5)
    assert high.confidence == 1.0
    assert low.confidence == 0.0
