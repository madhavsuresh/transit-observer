"""Port of Cozy Fox's StopArrivalProcess + WaitForecast.

Given a list of upcoming live departures at a station and a hypothetical
arrival time, returns the wait distribution, a WaitReasonableness label,
and explanation copy. Mirrors the Swift implementation in
`TransitCore/Journey/StopArrivalProcess.swift` line-for-line so the
golden-file parity test can pin behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .time_distribution import TimeDistributionSummary


class FeedState(str, Enum):
    fresh = "fresh"
    stale = "stale"
    missing = "missing"


class WaitReasonableness(str, Enum):
    good_wait = "goodWait"
    acceptable_wait = "acceptableWait"
    risky_wait = "riskyWait"
    bad_gap = "badGap"
    bunched = "bunched"
    feed_unreliable = "feedUnreliable"
    unknown = "unknown"


@dataclass(frozen=True)
class LiveDeparture:
    arrival_at: datetime
    is_approaching: bool = False
    is_scheduled: bool = False


@dataclass(frozen=True)
class WaitForecast:
    wait_distribution: TimeDistributionSummary
    state: WaitReasonableness
    next_departure_at: datetime | None
    p_board_within_5_min: float
    p_board_within_10_min: float
    p_board_within_15_min: float
    explanation: str | None


@dataclass(frozen=True)
class StopArrivalProcess:
    route: str
    direction: str | None
    generated_at: datetime
    departures: tuple[LiveDeparture, ...]
    schedule_headway_seconds: float | None = None
    feed_state: FeedState = FeedState.fresh

    @classmethod
    def make(
        cls,
        *,
        route: str,
        direction: str | None = None,
        generated_at: datetime,
        departures: list[LiveDeparture],
        schedule_headway_seconds: float | None = None,
        feed_state: FeedState = FeedState.fresh,
    ) -> "StopArrivalProcess":
        ordered = tuple(sorted(departures, key=lambda d: d.arrival_at))
        return cls(
            route=route,
            direction=direction,
            generated_at=generated_at,
            departures=ordered,
            schedule_headway_seconds=schedule_headway_seconds,
            feed_state=feed_state,
        )

    def wait_distribution(self, arriving_at: datetime) -> WaitForecast:
        upcoming = [d for d in self.departures if d.arrival_at >= arriving_at]

        if self.feed_state is FeedState.missing or (
            not upcoming and self.feed_state is FeedState.fresh and self.schedule_headway_seconds is None
        ):
            return self._schedule_fallback(arriving_at)

        if self.feed_state is FeedState.stale:
            return self._schedule_fallback(arriving_at)

        if not upcoming:
            return self._schedule_fallback(arriving_at)

        next_dep = upcoming[0]
        next_wait = max(0.0, (next_dep.arrival_at - arriving_at).total_seconds())
        gaps = _consecutive_gaps(upcoming)
        summary = _wait_summary(next_wait, gaps, sample_count=len(upcoming))
        state = _classify(next_wait, gaps, next_dep)
        explanation = _explain(state, next_wait)
        return WaitForecast(
            wait_distribution=summary,
            state=state,
            next_departure_at=next_dep.arrival_at,
            p_board_within_5_min=1.0 if next_wait <= 5 * 60 else 0.0,
            p_board_within_10_min=1.0 if next_wait <= 10 * 60 else 0.0,
            p_board_within_15_min=1.0 if next_wait <= 15 * 60 else 0.0,
            explanation=explanation,
        )

    def _schedule_fallback(self, arriving_at: datetime) -> WaitForecast:
        if not self.schedule_headway_seconds or self.schedule_headway_seconds <= 0:
            return WaitForecast(
                wait_distribution=TimeDistributionSummary.zero(),
                state=WaitReasonableness.unknown,
                next_departure_at=None,
                p_board_within_5_min=0.0,
                p_board_within_10_min=0.0,
                p_board_within_15_min=0.0,
                explanation="No live data and no schedule headway.",
            )
        headway = self.schedule_headway_seconds
        half = headway / 2
        summary = TimeDistributionSummary.analytic(
            mean=half, sigma=headway / 3, confidence=0.4
        )
        rate = 1.0 / max(60.0, half)
        return WaitForecast(
            wait_distribution=summary,
            state=WaitReasonableness.feed_unreliable,
            next_departure_at=None,
            p_board_within_5_min=_saturating(rate, 5 * 60),
            p_board_within_10_min=_saturating(rate, 10 * 60),
            p_board_within_15_min=_saturating(rate, 15 * 60),
            explanation="Schedule-only estimate — half-headway.",
        )


def _consecutive_gaps(deps: list[LiveDeparture]) -> list[float]:
    if len(deps) < 2:
        return []
    return [
        (deps[i].arrival_at - deps[i - 1].arrival_at).total_seconds()
        for i in range(1, len(deps))
    ]


def _wait_summary(next_wait: float, gaps: list[float], sample_count: int) -> TimeDistributionSummary:
    if not gaps:
        return TimeDistributionSummary.analytic(
            mean=next_wait,
            sigma=max(60.0, next_wait * 0.2),
            confidence=0.55,
            sample_count=sample_count,
        )
    median = _median(gaps)
    sigma = max(45.0, median * 0.3)
    return TimeDistributionSummary.analytic(
        mean=next_wait,
        sigma=sigma,
        confidence=min(1.0, 0.5 + min(len(gaps), 4) * 0.1),
        sample_count=sample_count,
    )


def _classify(next_wait: float, gaps: list[float], nxt: LiveDeparture) -> WaitReasonableness:
    if nxt.is_approaching:
        return WaitReasonableness.good_wait
    if next_wait <= 4 * 60:
        return WaitReasonableness.good_wait

    if len(gaps) >= 2:
        median_rest = _median(gaps[1:])
        first_gap = gaps[0]
        if median_rest > 0 and first_gap > 0 and first_gap < 0.5 * median_rest and first_gap <= 4 * 60:
            return WaitReasonableness.bunched
        if median_rest > 0 and first_gap > 2.0 * median_rest and first_gap > 12 * 60:
            return WaitReasonableness.bad_gap

    if next_wait > 12 * 60:
        return WaitReasonableness.bad_gap
    if next_wait > 7 * 60:
        return WaitReasonableness.risky_wait
    return WaitReasonableness.acceptable_wait


def _explain(state: WaitReasonableness, next_wait: float) -> str:
    minutes = max(1, round(next_wait / 60))
    if state is WaitReasonableness.good_wait:
        return f"Next departure in {minutes} min."
    if state is WaitReasonableness.acceptable_wait:
        return f"Reasonable wait — {minutes} min."
    if state is WaitReasonableness.risky_wait:
        return f"Cutting it close — {minutes} min away."
    if state is WaitReasonableness.bad_gap:
        return f"Long gap — {minutes} min wait."
    if state is WaitReasonableness.bunched:
        return f"Bunched arrivals — next {minutes} min, then another close behind."
    if state is WaitReasonableness.feed_unreliable:
        return "Feed unreliable — estimate widened."
    return "Not enough data to judge wait."


def _saturating(rate: float, seconds: float) -> float:
    p = 1.0 - math.exp(-rate * seconds)
    return max(0.0, min(1.0, p))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    n = len(sv)
    if n % 2 == 1:
        return sv[n // 2]
    return (sv[n // 2 - 1] + sv[n // 2]) / 2
