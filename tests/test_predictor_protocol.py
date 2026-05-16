"""Kernel adapter is byte-equivalent to the raw journey kernel for L mode."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from transit_observer import db
from transit_observer.corridors import by_id
from transit_observer.predictors import registry as pred_registry
from transit_observer.predictors.journey_kernel import (
    KERNEL_EB_VERSION,
    KERNEL_VERSION,
    JourneyKernelEBPredictor,
    JourneyKernelPredictor,
    update_eb_state,
)


T0 = datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    db.init_schema(c)
    yield c
    c.close()


def _seed_l(conn, *, line, map_id, leave_at, offsets, destination="Loop"):
    for offset in offsets:
        conn.execute(
            """
            INSERT INTO train_arrivals_raw (
                polled_at, line, run_number, map_id, stop_id, station_name,
                direction_code, destination_name, predicted_at, arrival_at,
                is_approaching, is_delayed, is_fault, is_scheduled
            ) VALUES (?, ?, 'R1', ?, 0, 'A', '1', ?, ?, ?, ?, FALSE, FALSE, ?)
            """,
            [
                leave_at - timedelta(seconds=30),
                line, map_id, destination,
                leave_at - timedelta(seconds=30),
                leave_at + timedelta(seconds=offset),
                offset < 60, False,
            ],
        )


def test_kernel_predictor_returns_prediction(conn):
    pred_registry.reset()
    corridor = by_id()["cta-red-belmont-lake-sb"]
    _seed_l(conn, line=corridor.line, map_id=corridor.boarding_int_id,
            leave_at=T0, offsets=[180, 600, 1200])
    predictor = JourneyKernelPredictor()
    out = predictor.predict(conn, corridor, now=T0)
    assert out is not None
    assert out.predictor_version == KERNEL_VERSION
    # All three legacy quantiles should be present
    assert set(out.wait.quantiles) == {0.5, 0.8, 0.9}
    assert out.wait.quantiles[0.5] <= out.wait.quantiles[0.8] <= out.wait.quantiles[0.9]
    # Feature snapshot should round-trip the legacy fields
    assert out.feature_snapshot["mode"] == "L"
    assert "live_departures" in out.feature_snapshot
    assert "hour_of_day" in out.feature_snapshot
    # And the new fields the GBM consumes
    assert "seconds_until_next_arrival" in out.feature_snapshot


def test_kernel_eb_with_no_state_matches_kernel(conn):
    """Before any EB updates land, EB predictor outputs == kernel predictor."""
    pred_registry.reset()
    corridor = by_id()["cta-red-belmont-lake-sb"]
    _seed_l(conn, line=corridor.line, map_id=corridor.boarding_int_id,
            leave_at=T0, offsets=[180, 600, 1200])
    kernel = JourneyKernelPredictor().predict(conn, corridor, now=T0)
    eb = JourneyKernelEBPredictor().predict(conn, corridor, now=T0)
    assert kernel is not None and eb is not None
    assert kernel.wait.p50 == pytest.approx(eb.wait.p50)
    assert kernel.wait.p80 == pytest.approx(eb.wait.p80)
    assert kernel.wait.p90 == pytest.approx(eb.wait.p90)


def test_kernel_eb_shifts_wait_after_state_updates(conn):
    """Once update_eb_state populates a residual mean, EB shifts the wait."""
    pred_registry.reset()
    corridor = by_id()["cta-red-belmont-lake-sb"]
    _seed_l(conn, line=corridor.line, map_id=corridor.boarding_int_id,
            leave_at=T0, offsets=[180, 600, 1200])
    line = corridor.line
    direction = corridor.direction
    # 25 observations is below the SHRINKAGE_FLOOR (50), so the shift
    # is shrunk: shift = 30 * (25/50) = 15s.
    for _ in range(25):
        update_eb_state(
            conn, line=line, direction_code=direction,
            residual_seconds=30.0, now=T0,
        )
    out = JourneyKernelEBPredictor().predict(conn, corridor, now=T0)
    base = JourneyKernelPredictor().predict(conn, corridor, now=T0)
    assert out is not None and base is not None
    # EB wait should be shifted upward but not all the way to the raw mean
    assert out.wait.mean > base.wait.mean
    assert out.wait.mean < base.wait.mean + 30.0   # shrinkage caps the shift below raw residual


def test_kernel_returns_none_with_no_arrivals(conn):
    pred_registry.reset()
    corridor = by_id()["cta-red-belmont-lake-sb"]
    # No arrivals seeded — predictor can't return anything
    out = JourneyKernelPredictor().predict(conn, corridor, now=T0)
    assert out is None


def test_to_wait_forecast_back_compat(conn):
    """Prediction.to_wait_forecast() preserves the kernel's surface."""
    pred_registry.reset()
    corridor = by_id()["cta-red-belmont-lake-sb"]
    _seed_l(conn, line=corridor.line, map_id=corridor.boarding_int_id,
            leave_at=T0, offsets=[180, 600, 1200])
    p = JourneyKernelPredictor().predict(conn, corridor, now=T0)
    wf = p.to_wait_forecast()
    assert wf.wait_distribution.p50 == pytest.approx(p.wait.p50)
    assert wf.wait_distribution.p80 == pytest.approx(p.wait.p80)
    assert wf.wait_distribution.p90 == pytest.approx(p.wait.p90)
