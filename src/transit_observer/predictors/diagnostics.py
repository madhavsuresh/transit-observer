"""Predictor-agnostic probabilistic-forecast diagnostics.

Operates on arbitrary quantile dicts so it works equally well for the
kernel (lognormal-fit quantiles) and the GBM (empirical pinball
quantiles). The metrics module groups outcomes by ``predictor_version``
and pipes (quantiles, actual) tuples through these primitives.

Implemented:
  - pinball_loss      — single-quantile loss
  - crps_from_quantiles — Laio & Tamea 2007 trapezoid CRPS approximation
  - interval_score    — Winkler/Gneiting interval score for a central band
  - tail_miss_rate    — empirical P(y > q_upper) for the late-train tail
  - quantile_reliability — empirical-vs-nominal coverage table

These return scalars or tuples for one (prediction, actual) pair. The
metrics module aggregates across rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def pinball_loss(actual: float, predicted: float, quantile: float) -> float:
    """Standard pinball / quantile loss.

    Lower is better; calibrated quantile minimizes its own pinball loss
    in expectation.
    """
    if not math.isfinite(actual) or not math.isfinite(predicted):
        return float("nan")
    err = actual - predicted
    return quantile * err if err >= 0 else (quantile - 1.0) * err


def crps_from_quantiles(
    quantiles: dict[float, float],
    actual: float,
) -> float:
    """Trapezoid CRPS estimate from an arbitrary quantile dict.

    CRPS = ∫₀¹ pinball(y, F⁻¹(α), α) dα. For a piecewise-defined CDF given
    only at a few α points, integrate by trapezoid between adjacent
    points, plus constant extensions to α=0 and α=1 (the standard Laio &
    Tamea 2007 approach).

    Scaled by 2 to match the analytic CRPS scale: when the supplied
    grid is dense and uniform on (0, 1), this converges to true CRPS.
    """
    items: list[tuple[float, float]] = sorted(
        (a, q) for a, q in quantiles.items()
        if math.isfinite(a) and math.isfinite(q) and 0.0 < a < 1.0
    )
    if not items or not math.isfinite(actual):
        return float("nan")

    alphas = [a for a, _ in items]
    losses = [pinball_loss(actual, q, a) for a, q in items]

    # Inner trapezoids
    crps = 0.0
    for i in range(len(alphas) - 1):
        crps += 0.5 * (alphas[i + 1] - alphas[i]) * (losses[i] + losses[i + 1])
    # Boundary constant extensions: α∈[0, α₀) and α∈(α_last, 1]
    crps += alphas[0] * losses[0]
    crps += (1.0 - alphas[-1]) * losses[-1]
    return 2.0 * crps


def crps_from_quantiles_batch(
    *,
    quantile_levels: np.ndarray,
    quantile_values: np.ndarray,
    actuals: np.ndarray,
) -> np.ndarray:
    """Vectorized CRPS over many rows.

    Same formula as :func:`crps_from_quantiles` (Laio & Tamea 2007
    trapezoid), implemented as one BLAS-backed reduction so a 100k-row
    scoring pass is sub-second instead of seconds in a Python loop.

    Args:
        quantile_levels: 1-D float array of α values, shape ``(k,)``, in
            ``(0, 1)``. Need not be sorted — sorted internally.
        quantile_values: float array of predicted quantiles, shape
            ``(n, k)`` aligned with ``quantile_levels``.
        actuals: 1-D float array of realized outcomes, shape ``(n,)``.

    Returns:
        1-D array of per-row CRPS, shape ``(n,)``. NaN entries pass
        through.
    """
    alphas = np.asarray(quantile_levels, dtype=np.float64)
    qv = np.asarray(quantile_values, dtype=np.float64)
    y = np.asarray(actuals, dtype=np.float64)

    if alphas.ndim != 1 or qv.ndim != 2 or y.ndim != 1:
        raise ValueError(
            f"shape mismatch: alphas{alphas.shape}, qv{qv.shape}, y{y.shape}"
        )
    if qv.shape != (y.shape[0], alphas.shape[0]):
        raise ValueError(
            f"quantile_values must be (n, k) = ({y.shape[0]}, {alphas.shape[0]}); "
            f"got {qv.shape}"
        )

    order = np.argsort(alphas)
    alphas_s = alphas[order]
    qv_s = qv[:, order]

    # Pinball loss at each (row, quantile)
    err = y[:, None] - qv_s
    pos = np.maximum(0.0, err)
    neg = np.maximum(0.0, -err)
    losses = alphas_s[None, :] * pos + (1.0 - alphas_s[None, :]) * neg

    # Trapezoid integration across quantile axis
    dx = np.diff(alphas_s)
    inner = (losses[:, :-1] + losses[:, 1:]) * 0.5 * dx[None, :]
    inner_sum = inner.sum(axis=1)
    # Boundary constant extensions to α=0 and α=1
    boundary = alphas_s[0] * losses[:, 0] + (1.0 - alphas_s[-1]) * losses[:, -1]
    crps = 2.0 * (inner_sum + boundary)

    # Propagate NaNs from inputs
    nan_rows = ~np.isfinite(y) | ~np.isfinite(qv_s).all(axis=1)
    crps[nan_rows] = np.nan
    return crps


def interval_score(
    actual: float,
    lower: float,
    upper: float,
    *,
    alpha: float,
) -> float:
    """Gneiting & Raftery interval / Winkler score for a central (1 − α) interval.

    ``alpha`` is the *miscoverage* level (0.2 for an 80% interval). The
    score penalizes both interval width and miscoverage:

        IS_α = (u − l) + (2/α)·(l − y)·I(y < l) + (2/α)·(y − u)·I(y > u)

    Smaller is better; calibrated and sharp intervals minimize it.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not all(math.isfinite(v) for v in (actual, lower, upper)):
        return float("nan")
    width = max(0.0, upper - lower)
    miss_below = max(0.0, lower - actual)
    miss_above = max(0.0, actual - upper)
    return width + (2.0 / alpha) * (miss_below + miss_above)


def tail_miss(actual: float, upper_quantile_predicted: float) -> bool:
    """True iff the actual exceeded the predicted upper quantile.

    Aggregate over a sample and compare to the nominal miss rate
    (1 − q_upper) to diagnose tail calibration.
    """
    return math.isfinite(actual) and math.isfinite(upper_quantile_predicted) and actual > upper_quantile_predicted


def quantile_hits(
    predicted_quantiles: dict[float, float],
    actual: float,
) -> dict[float, bool]:
    """For each predicted quantile q, True iff actual ≤ q_predicted.

    Aggregated, the fraction-True per q is the empirical coverage that
    should equal q under calibration. Powers the reliability-curve
    aggregator in metrics.py.
    """
    if not math.isfinite(actual):
        return {q: False for q in predicted_quantiles}
    return {q: actual <= v for q, v in predicted_quantiles.items() if math.isfinite(v)}


@dataclass(frozen=True)
class CalibrationDiagnostic:
    """Bundle of summary numbers per row, before any aggregation."""

    crps: float
    pinball_q50: float
    pinball_q80: float
    pinball_q90: float
    interval_score_central_80: float    # for the (p10, p90) interval
    interval_score_central_40: float    # for the (p50, p90) interval — covers what we have natively
    tail_miss_p90: bool                 # 1 if actual > p90
    p80_covered: bool
    p90_covered: bool


def diagnose_row(
    quantiles: dict[float, float],
    *,
    actual: float,
    p10_lower: float | None = None,
) -> CalibrationDiagnostic:
    """Compute all per-row diagnostics for one (quantiles, actual) pair.

    If ``p10_lower`` is supplied, the 80% central interval score uses
    it. Otherwise the central-80 score is NaN (the schema doesn't store
    p10 directly).
    """
    p50 = quantiles.get(0.5, float("nan"))
    p80 = quantiles.get(0.8, float("nan"))
    p90 = quantiles.get(0.9, float("nan"))
    central_80 = (
        interval_score(actual, p10_lower, p90, alpha=0.2)
        if p10_lower is not None and math.isfinite(p10_lower)
        else float("nan")
    )
    return CalibrationDiagnostic(
        crps=crps_from_quantiles(quantiles, actual),
        pinball_q50=pinball_loss(actual, p50, 0.5),
        pinball_q80=pinball_loss(actual, p80, 0.8),
        pinball_q90=pinball_loss(actual, p90, 0.9),
        interval_score_central_80=central_80,
        interval_score_central_40=interval_score(actual, p50, p90, alpha=0.5),
        tail_miss_p90=tail_miss(actual, p90),
        p80_covered=(math.isfinite(p80) and math.isfinite(actual) and actual <= p80),
        p90_covered=(math.isfinite(p90) and math.isfinite(actual) and actual <= p90),
    )


def aggregate_crps(values: Iterable[float]) -> float:
    """Mean CRPS over a sample, ignoring NaNs."""
    total = 0.0
    n = 0
    for v in values:
        if math.isfinite(v):
            total += v
            n += 1
    return total / n if n > 0 else float("nan")


def aggregate_coverage(hits: Sequence[bool]) -> float:
    """Empirical coverage = fraction True."""
    if not hits:
        return float("nan")
    return sum(1 for h in hits if h) / len(hits)


def coverage_gap(empirical: float, nominal: float) -> float:
    """|empirical − nominal| — symmetric calibration gap."""
    if not (math.isfinite(empirical) and math.isfinite(nominal)):
        return float("nan")
    return abs(empirical - nominal)


def decision_score(crps: float, coverage_gap_p80: float, *, alpha: float = 0.05) -> float:
    """Composite decision-loss the registry promotes on.

    Lower is better. ``alpha`` mixes accuracy (CRPS) with calibration
    (coverage gap at the 80% interval). Matches divvy-observer's
    Brier+0.05·log_loss recipe in spirit.
    """
    if not (math.isfinite(crps) and math.isfinite(coverage_gap_p80)):
        return float("inf")
    return crps + alpha * coverage_gap_p80
