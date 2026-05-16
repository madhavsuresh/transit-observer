"""Log-normal fit and PIT helpers used by the calibration dashboards.

We store only (p50, p80, p90) per forecast. To draw reliability diagrams,
PIT histograms, and quantile dotplots we need a continuous distribution.
Three quantiles over-determine a two-parameter family, so we fit a
log-normal from (p50, p80) in closed form and use p90 as a free
fit-check on the reliability diagram.

Why log-normal: travel times are positive and right-skewed; this is the
standard choice in the travel-time literature. If the kernel deviates,
the PIT histogram itself will show it (skew, U-shape, etc.).

All functions are pure and import nothing from the rest of the project,
so they are trivially testable.
"""

from __future__ import annotations

import math
from statistics import NormalDist

_STDNORM = NormalDist()

# Φ⁻¹(0.8) — the z-score corresponding to the 80th percentile of the
# standard normal. Hardcoded so the fit doesn't pay for repeated
# NormalDist().inv_cdf() calls in tight loops.
_Z_80 = 0.8416212335729143


def fit_lognormal_from_p50_p80(p50: float, p80: float) -> tuple[float, float]:
    """Return (mu, sigma) of a log-normal matching the given p50 and p80.

    Closed form:
        mu = log(p50)
        sigma = (log(p80) - mu) / Φ⁻¹(0.8)

    Raises ValueError if inputs are non-positive or p80 <= p50.
    """
    if not (math.isfinite(p50) and math.isfinite(p80)):
        raise ValueError(f"non-finite quantiles: p50={p50}, p80={p80}")
    if p50 <= 0 or p80 <= 0:
        raise ValueError(f"quantiles must be positive: p50={p50}, p80={p80}")
    if p80 <= p50:
        raise ValueError(f"p80 must exceed p50: p50={p50}, p80={p80}")
    mu = math.log(p50)
    sigma = (math.log(p80) - mu) / _Z_80
    return mu, sigma


def lognormal_cdf(x: float, mu: float, sigma: float) -> float:
    """P(X <= x) for X ~ LogNormal(mu, sigma). x must be positive."""
    if x <= 0 or not math.isfinite(x):
        return 0.0
    if sigma <= 0:
        return 1.0 if x >= math.exp(mu) else 0.0
    z = (math.log(x) - mu) / sigma
    return _STDNORM.cdf(z)


def lognormal_quantile(p: float, mu: float, sigma: float) -> float:
    """Inverse CDF: exp(mu + sigma · Φ⁻¹(p)). p ∈ (0, 1)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    z = _STDNORM.inv_cdf(p)
    return math.exp(mu + sigma * z)


def pit_value(actual: float, p50: float, p80: float) -> float:
    """PIT = F_predicted(actual) for the log-normal fit through (p50, p80).

    Returns NaN if the quantile triple is malformed (caller's choice on
    how to handle — usually drop the row).
    """
    try:
        mu, sigma = fit_lognormal_from_p50_p80(p50, p80)
    except ValueError:
        return math.nan
    if not math.isfinite(actual):
        return math.nan
    return lognormal_cdf(actual, mu, sigma)


def quantile_dotplot_positions(p50: float, p80: float, n: int = 50) -> list[float]:
    """Return n x-positions for a quantile dotplot of the fitted log-normal.

    Uses the (k - 0.5)/n grid recommended by Kay et al. — this puts the
    first dot at the (0.5/n)-th quantile and the last at ((n - 0.5)/n),
    which gives a symmetric, dense-in-the-middle layout. Each dot
    therefore represents a 1/n slice of probability mass.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    mu, sigma = fit_lognormal_from_p50_p80(p50, p80)
    return [lognormal_quantile((k + 0.5) / n, mu, sigma) for k in range(n)]


def p90_fit_residual(p50: float, p80: float, p90: float) -> float:
    """How far is the stored p90 from what the (p50, p80) log-normal predicts?

    Positive → stored p90 is wider than the fit (kernel's tail is heavier
    than log-normal). Negative → narrower (lighter tail). Reported in
    relative terms: (p90_stored − p90_fit) / p90_fit.
    """
    mu, sigma = fit_lognormal_from_p50_p80(p50, p80)
    p90_fit = lognormal_quantile(0.9, mu, sigma)
    if p90_fit <= 0:
        return math.nan
    return (p90 - p90_fit) / p90_fit
