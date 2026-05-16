"""Port of Cozy Fox's TimeDistributionSummary."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TimeDistributionSummary:
    mean: float       # seconds
    p50: float
    p80: float
    p90: float
    confidence: float
    sample_count: int

    @staticmethod
    def zero() -> "TimeDistributionSummary":
        return TimeDistributionSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    @staticmethod
    def empirical(samples: list[float]) -> "TimeDistributionSummary":
        cleaned = sorted(s for s in samples if math.isfinite(s) and s >= 0)
        n = len(cleaned)
        if n == 0:
            return TimeDistributionSummary.zero()
        mean = sum(cleaned) / n
        return TimeDistributionSummary(
            mean=mean,
            p50=_nearest_rank(cleaned, 0.5),
            p80=_nearest_rank(cleaned, 0.8),
            p90=_nearest_rank(cleaned, 0.9),
            confidence=min(1.0, n / 30.0),
            sample_count=n,
        )

    @staticmethod
    def analytic(*, mean: float, sigma: float, confidence: float, sample_count: int = 0) -> "TimeDistributionSummary":
        sigma = max(0.0, sigma)
        mean = max(0.0, mean)
        return TimeDistributionSummary(
            mean=mean,
            p50=mean,
            p80=max(0.0, mean + 0.8416 * sigma),
            p90=max(0.0, mean + 1.2816 * sigma),
            confidence=max(0.0, min(1.0, confidence)),
            sample_count=max(0, sample_count),
        )


def _nearest_rank(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    rank = max(1, math.ceil(p * n))
    return sorted_values[min(rank, n) - 1]
