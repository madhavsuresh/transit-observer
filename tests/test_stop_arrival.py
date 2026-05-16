"""StopArrivalProcess parity with the Swift port.

Cases mirror StopArrivalProcessTests.swift one-for-one. Numbers stay
identical to within floating-point precision.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from transit_observer.journey.stop_arrival import (
    FeedState,
    LiveDeparture,
    StopArrivalProcess,
    WaitReasonableness,
)


T0 = datetime.fromisoformat("2026-01-01T08:00:00+00:00")


def _process(minutes_ahead: list[float], **kwargs) -> StopArrivalProcess:
    deps = [
        LiveDeparture(arrival_at=T0 + timedelta(minutes=m), is_approaching=False)
        for m in minutes_ahead
    ]
    return StopArrivalProcess.make(route="Red", generated_at=T0, departures=deps, **kwargs)


def test_normal_headways_produce_acceptable_wait():
    forecast = _process([5, 13, 21, 29]).wait_distribution(T0)
    assert forecast.state is WaitReasonableness.acceptable_wait
    assert forecast.next_departure_at == T0 + timedelta(minutes=5)


def test_approaching_departure_gives_good_wait():
    deps = [
        LiveDeparture(arrival_at=T0 + timedelta(seconds=120), is_approaching=True),
        LiveDeparture(arrival_at=T0 + timedelta(minutes=8)),
    ]
    process = StopArrivalProcess.make(route="Red", generated_at=T0, departures=deps)
    forecast = process.wait_distribution(T0)
    assert forecast.state is WaitReasonableness.good_wait


def test_long_gap_produces_bad_gap_classification():
    forecast = _process([18, 28, 38, 48]).wait_distribution(T0)
    assert forecast.state is WaitReasonableness.bad_gap


def test_bunched_first_gap_classifies_as_bunched():
    forecast = _process([8, 11, 22, 32]).wait_distribution(T0)
    assert forecast.state is WaitReasonableness.bunched


def test_stale_feed_falls_back_to_feed_unreliable():
    forecast = _process([5, 13], schedule_headway_seconds=480, feed_state=FeedState.stale).wait_distribution(T0)
    assert forecast.state is WaitReasonableness.feed_unreliable
    assert forecast.next_departure_at is None


def test_missing_feed_with_schedule_returns_unreliable():
    process = StopArrivalProcess.make(
        route="Red",
        generated_at=T0,
        departures=[],
        schedule_headway_seconds=600,
        feed_state=FeedState.missing,
    )
    forecast = process.wait_distribution(T0)
    assert forecast.state is WaitReasonableness.feed_unreliable


def test_missing_feed_without_schedule_is_unknown():
    process = StopArrivalProcess.make(
        route="Red",
        generated_at=T0,
        departures=[],
        schedule_headway_seconds=None,
        feed_state=FeedState.missing,
    )
    forecast = process.wait_distribution(T0)
    assert forecast.state is WaitReasonableness.unknown


def test_schedule_fallback_produces_nonzero_board_probabilities():
    process = StopArrivalProcess.make(
        route="Red",
        generated_at=T0,
        departures=[],
        schedule_headway_seconds=600,
        feed_state=FeedState.missing,
    )
    forecast = process.wait_distribution(T0)
    assert forecast.p_board_within_15_min > 0


def test_board_within_5_min_flags_immediate_departure():
    forecast = _process([2, 12, 22]).wait_distribution(T0)
    assert forecast.p_board_within_5_min == 1.0
    assert forecast.p_board_within_10_min == 1.0
