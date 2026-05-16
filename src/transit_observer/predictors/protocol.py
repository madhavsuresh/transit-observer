"""Predictor Protocol + the richer Prediction object.

Every predictor returns a Prediction with the same shape. The schema
stores only (mean, p50, p80, p90) per leg, so ``Prediction`` keeps that
columnar surface, plus a richer ``quantiles`` dict for diagnostics that
want more than three points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from ..journey.stop_arrival import WaitForecast, WaitReasonableness
from ..journey.time_distribution import TimeDistributionSummary


# The three quantiles the forecast_queue schema natively stores. Predictors
# that produce other quantiles still report these three; extras live in
# ``PredictionLeg.quantiles`` for diagnostics.
SCHEMA_QUANTILES: tuple[float, ...] = (0.5, 0.8, 0.9)


@dataclass(frozen=True)
class PredictionLeg:
    """A single leg (wait / in_vehicle / total) of a prediction.

    ``quantiles`` is the source of truth: a mapping from nominal quantile
    (e.g. 0.5) to predicted seconds. ``p50/p80/p90`` are convenience
    accessors for the storage-schema triple.

    ``confidence`` is a 0..1 self-rating the predictor sets — the kernel
    derives it from feed health; the GBM derives it from feature
    completeness. Stored in ``forecast_queue.predicted_wait_mean``'s
    sibling field… (only ``mean`` is in the schema, so confidence is for
    runtime ranking, not persistence).
    """

    quantiles: dict[float, float]
    mean: float
    confidence: float = 1.0
    sample_count: int = 0

    @property
    def p50(self) -> float:
        return self.quantiles.get(0.5, self.mean)

    @property
    def p80(self) -> float:
        return self.quantiles.get(0.8, self.mean)

    @property
    def p90(self) -> float:
        return self.quantiles.get(0.9, self.mean)

    def to_summary(self) -> TimeDistributionSummary:
        return TimeDistributionSummary(
            mean=self.mean,
            p50=self.p50,
            p80=self.p80,
            p90=self.p90,
            confidence=self.confidence,
            sample_count=self.sample_count,
        )

    @classmethod
    def from_summary(
        cls, s: TimeDistributionSummary, *, extra_quantiles: dict[float, float] | None = None,
    ) -> "PredictionLeg":
        q = {0.5: s.p50, 0.8: s.p80, 0.9: s.p90}
        if extra_quantiles:
            q.update(extra_quantiles)
        return cls(
            quantiles=q,
            mean=s.mean,
            confidence=s.confidence,
            sample_count=s.sample_count,
        )


@dataclass(frozen=True)
class Prediction:
    """Predictor output. Modes that lack an in-vehicle leg just supply
    a TimeDistributionSummary.zero() — the legacy code did the same."""

    predictor_version: str
    wait: PredictionLeg
    in_vehicle: PredictionLeg
    feature_snapshot: dict[str, Any] = field(default_factory=dict)
    feature_completeness: float = 1.0  # 0..1 — what fraction of expected features were populated
    state_label: str | None = None     # WaitReasonableness for kernel; None for GBM
    explanation: str | None = None
    schedule_fallback: bool = False

    @property
    def total(self) -> PredictionLeg:
        """Convolution of wait + in-vehicle assuming independence is a
        rough approximation; the legacy code adds quantile-wise. Kept
        the same here so storage stays comparable across predictors."""
        joined_q = {
            q: self.wait.quantiles.get(q, self.wait.mean) + self.in_vehicle.quantiles.get(q, self.in_vehicle.mean)
            for q in sorted(set(self.wait.quantiles) | set(self.in_vehicle.quantiles))
        }
        return PredictionLeg(
            quantiles=joined_q,
            mean=self.wait.mean + self.in_vehicle.mean,
            confidence=min(self.wait.confidence, self.in_vehicle.confidence),
            sample_count=min(self.wait.sample_count, self.in_vehicle.sample_count),
        )

    def to_wait_forecast(self) -> WaitForecast:
        """Back-compat for callers (corpus.py adhoc path, tests) that
        still want a WaitForecast. The state defaults to ``unknown`` if
        the predictor didn't supply one."""
        state = WaitReasonableness.unknown
        if self.state_label:
            try:
                state = WaitReasonableness(self.state_label)
            except ValueError:
                state = WaitReasonableness.unknown
        next_dep = self.feature_snapshot.get("next_departure_at")
        if isinstance(next_dep, str):
            try:
                next_dep_dt = datetime.fromisoformat(next_dep)
            except ValueError:
                next_dep_dt = None
        elif isinstance(next_dep, datetime):
            next_dep_dt = next_dep
        else:
            next_dep_dt = None
        return WaitForecast(
            wait_distribution=self.wait.to_summary(),
            state=state,
            next_departure_at=next_dep_dt,
            p_board_within_5_min=1.0 if self.wait.p50 <= 5 * 60 else 0.0,
            p_board_within_10_min=1.0 if self.wait.p50 <= 10 * 60 else 0.0,
            p_board_within_15_min=1.0 if self.wait.p50 <= 15 * 60 else 0.0,
            explanation=self.explanation,
        )


def quantiles_to_summary(
    quantiles: dict[float, float],
    *,
    mean: float | None = None,
    confidence: float = 1.0,
    sample_count: int = 0,
) -> TimeDistributionSummary:
    """Build a schema-shaped TimeDistributionSummary from a quantile dict.

    If a quantile isn't present, fall back to nearest available. The mean
    defaults to p50.
    """
    p50 = quantiles.get(0.5, mean or 0.0)
    p80 = quantiles.get(0.8, p50)
    p90 = quantiles.get(0.9, p80)
    return TimeDistributionSummary(
        mean=mean if mean is not None else p50,
        p50=p50, p80=p80, p90=p90,
        confidence=confidence,
        sample_count=sample_count,
    )


class Predictor(Protocol):
    """Pluggable predictor surface.

    Implementations:

    - ``JourneyKernelPredictor`` — parity-pure wrapper around the Swift
      port. ``predictor_version = "kernel-v1"``.
    - ``JourneyKernelEBPredictor`` — kernel + empirical-Bayes residual
      shrinkage. ``predictor_version = "kernel-v1+eb"``.
    - ``QuantileGBMPredictor`` — residual-target LightGBM.
      ``predictor_version = "gbm-v1"``.
    """

    predictor_version: str

    def predict(
        self,
        conn,
        spec,
        *,
        now: datetime,
    ) -> Prediction | None:
        """Run the predictor for one corridor at ``now``. Returns ``None``
        if there's not enough data (caller falls back to a kernel)."""
        ...
