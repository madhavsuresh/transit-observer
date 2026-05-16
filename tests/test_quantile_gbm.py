"""QuantileGBMPredictor: isotonization + artifact loading + (if lightgbm) end-to-end fit."""

from __future__ import annotations

import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from transit_observer import db
from transit_observer.predictors.quantile_gbm import (
    GBM_QUANTILES,
    GBM_VERSION,
    MIN_FEATURE_COMPLETENESS,
    QuantileGBMPredictor,
    _isotonize,
)


def test_isotonize_sorts_monotonically():
    out = _isotonize({0.5: 100.0, 0.8: 90.0, 0.9: 120.0})
    assert out[0.5] == 100.0
    assert out[0.8] == 100.0   # was 90; clamped up to preceding
    assert out[0.9] == 120.0


def test_isotonize_no_op_when_sorted():
    out = _isotonize({0.5: 100.0, 0.8: 130.0, 0.9: 160.0})
    assert out == {0.5: 100.0, 0.8: 130.0, 0.9: 160.0}


def test_isotonize_handles_empty():
    assert _isotonize({}) == {}


def test_gbm_predictor_returns_none_when_no_artifacts(tmp_path):
    """Without trained artifacts on disk, the GBM predictor must defer to fallback."""
    predictor = QuantileGBMPredictor.from_root(tmp_path, version=GBM_VERSION)
    # No on-disk model -> get() returns None -> predict returns None
    assert predictor.artifact.get("wait", "Red") is None


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    db.init_schema(c)
    yield c
    c.close()


# ----------- Optional end-to-end test (needs lightgbm) ------------------


try:
    import lightgbm  # noqa: F401
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


requires_lightgbm = pytest.mark.skipif(
    not HAS_LIGHTGBM, reason="lightgbm not installed (uv sync --group learned)",
)


@requires_lightgbm
def test_fit_quantile_gbm_on_synthetic_fixture(conn, tmp_path, monkeypatch):
    """End-to-end synthetic fixture: pinball loss < naive baseline.

    Seeds 2,000 synthetic L outcomes (resolved forecasts) where the
    residual follows a known function of features. Fits the GBM and
    checks the validation pinball loss is < a naive zero-residual model.
    """
    import random

    # Redirect models root so artifacts land in tmp_path (Settings is frozen).
    from transit_observer.training import artifacts as art_mod
    monkeypatch.setattr(art_mod, "models_root", lambda: tmp_path / "models")

    rng = random.Random(123)
    T0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    n_rows = 2_000

    for i in range(n_rows):
        ts = T0 + timedelta(seconds=i * 60)
        line = rng.choice(["Red", "Blue", "Brn"])
        # The "true" wait depends on hour and line:
        #   Red faster, Blue medium, Brn slower
        # Kernel always predicts mean=300; the GBM should learn the residual.
        kernel_wait = 300.0
        true_extra = {"Red": -30.0, "Blue": 0.0, "Brn": 40.0}[line]
        hour_effect = 30.0 * math.sin(ts.hour * math.pi / 12)
        noise = rng.gauss(0.0, 30.0)
        actual_wait = max(0.0, kernel_wait + true_extra + hour_effect + noise)

        forecast_id = f"f{i:06d}"
        # Build a minimal feature_json matching the L_FEATURE_NAMES schema
        import json as _json
        feature_json = _json.dumps({
            "mode": "L", "line": line, "direction_code": "south",
            "boarding_map_id": "40000", "alighting_map_id": "41000",
            "hour_of_day": ts.hour, "weekday_or_weekend": "weekday",
            "haversine_meters": 5000.0,
            "seconds_until_next_arrival": 60.0 + rng.uniform(0, 600),
            "next_is_approaching": 0.0,
            "n_upcoming_arrivals_30m": rng.randint(2, 6),
            "headway_median_s": 600.0, "headway_iqr_s": 120.0, "headway_min_s": 240.0,
            "run_n_predictions_seen": rng.randint(1, 8),
            "run_arrival_variance_s": rng.uniform(0, 1000.0),
            "run_drift_s": rng.uniform(0, 60),
            "line_delayed_5m": 0.0, "line_fault_5m": 0.0, "line_n_runs_5m": 4.0,
            "position_age_s": 30.0, "position_next_arrival_offset_s": 60.0,
            "line_dir_hour_recent_mean_residual_s": hour_effect + true_extra,
        })
        conn.execute(
            """
            INSERT INTO forecast_queue (
                forecast_id, enqueued_at, snapshot_polled_at, leave_at,
                mode, line, direction_code,
                corridor_id, predictor_version, feature_json,
                boarding_map_id, boarding_text_id, boarding_station_name,
                alighting_map_id, alighting_text_id, alighting_station_name,
                predicted_wait_mean, predicted_wait_p50, predicted_wait_p80, predicted_wait_p90,
                predicted_in_vehicle_mean,
                predicted_total_mean, predicted_total_p50, predicted_total_p80, predicted_total_p90,
                predicted_failure_prob, resolve_after, status
            ) VALUES (?, ?, ?, ?, 'L', ?, 'south',
                     'syn', 'kernel-v1', ?,
                     40000, NULL, 'A', 41000, NULL, 'B',
                     ?, ?, ?, ?,
                     0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, ?, 'resolved')
            """,
            [
                forecast_id, ts, ts, ts, line, feature_json,
                kernel_wait, kernel_wait, kernel_wait * 1.2, kernel_wait * 1.4,
                ts + timedelta(seconds=2_000),
            ],
        )
        conn.execute(
            """
            INSERT INTO forecast_outcomes (
                forecast_id, resolved_at, boarded_run_number, boarded_at, alighted_at,
                actual_wait_seconds, actual_in_vehicle_seconds, actual_total_seconds,
                in_p80_window, in_p90_window,
                p50_residual_seconds, p80_residual_seconds,
                truth_confidence, failed, notes
            ) VALUES (?, ?, 'R1', ?, ?, ?, 0.0, ?, FALSE, FALSE, ?, ?, 1.0, FALSE, NULL)
            """,
            [
                forecast_id, ts + timedelta(seconds=2_000),
                ts + timedelta(seconds=actual_wait),
                ts + timedelta(seconds=actual_wait),
                actual_wait, actual_wait,
                actual_wait - kernel_wait, actual_wait - kernel_wait * 1.2,
            ],
        )

    # Fit
    from transit_observer.training import dataset, fit as fit_mod
    frame = dataset.build_training_frame_l(
        conn, since=T0 - timedelta(hours=1), until=T0 + timedelta(days=10),
    )
    assert len(frame) >= 1500   # filter for horizon >= 90s shouldn't drop most rows

    report = fit_mod.fit_quantile_gbm(
        conn, frame,
        predictor_version=GBM_VERSION,
        num_boost_round=80, early_stopping_rounds=15,
        per_line=False,                  # global model is enough for this fixture
    )
    assert report.boosters

    # The naive baseline is "predict residual = 0"; its pinball loss at q=0.5
    # for a residual stream with non-zero mean is the mean |residual|.
    import statistics
    residuals = [r["residual_wait_s"] for r in frame.rows if r["residual_wait_s"] is not None]
    naive_pinball_q50 = statistics.median(
        max(0.5 * r, -0.5 * r) for r in residuals
    )
    q50_boosters = [b for b in report.boosters if b.leg == "wait" and b.quantile == 0.5]
    assert q50_boosters
    val_pinball_q50 = q50_boosters[0].val_pinball
    # Trained model should beat naive baseline. Bound is loose because the
    # fixture has noise; the point is the GBM learns SOMETHING.
    assert val_pinball_q50 < naive_pinball_q50, (
        f"GBM did not beat naive baseline: val_pinball_q50={val_pinball_q50:.2f} >= "
        f"naive_pinball_q50={naive_pinball_q50:.2f}"
    )
