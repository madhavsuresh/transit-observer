"""Live feature extraction for the learned L predictor."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from transit_observer import db
from transit_observer.catalog import LStation
from transit_observer.predictors.features import (
    L_FEATURE_NAMES,
    extract_features_live_l,
    feature_completeness,
    normalize_for_model,
)
from transit_observer.trip_generator import TripSpec


T0 = datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    db.init_schema(c)
    yield c
    c.close()


def _station(map_id: int, name: str, lat: float, lon: float, line: str = "red") -> LStation:
    return LStation(
        map_id=map_id, name=name,
        latitude=lat, longitude=lon, served_lines=(line,),
    )


@pytest.fixture
def spec():
    boarding = _station(40640, "Belmont", 41.939, -87.653)
    alighting = _station(41000, "Lake/State", 41.886, -87.628)
    return TripSpec(
        line_catalog="red", line_api="Red",
        boarding=boarding, alighting=alighting,
        direction_label="south", leave_at=T0,
    )


def _seed(conn, *, line, map_id, polled_at, arrival_at, run="R1",
          is_app=False, is_dly=False, is_flt=False):
    conn.execute(
        """
        INSERT INTO train_arrivals_raw (
            polled_at, line, run_number, map_id, stop_id, station_name,
            direction_code, destination_name, predicted_at, arrival_at,
            is_approaching, is_delayed, is_fault, is_scheduled
        ) VALUES (?, ?, ?, ?, 0, 'B', '1', 'Loop', ?, ?, ?, ?, ?, FALSE)
        """,
        [polled_at, line, run, map_id, polled_at, arrival_at,
         is_app, is_dly, is_flt],
    )


def test_completeness_full_when_all_dynamic_present(conn, spec):
    # Seed several arrivals so all run-history / system-state features populate.
    for offset_s in (60, 180, 360, 540, 720, 900, 1080):
        _seed(conn,
              line=spec.line_api, map_id=spec.boarding.map_id,
              polled_at=T0 - timedelta(seconds=30),
              arrival_at=T0 + timedelta(seconds=offset_s),
              run="R1", is_app=offset_s < 90)
    # Add a position row so position_age_s / position_next_arrival_offset_s populate
    conn.execute(
        """
        INSERT INTO train_positions_raw (
            polled_at, line, run_number, destination_name, direction_code,
            next_station_map_id, next_station_name, predicted_at, next_arrival_at,
            is_approaching, is_delayed
        ) VALUES (?, ?, 'R1', 'Loop', '1', ?, 'B', ?, ?, FALSE, FALSE)
        """,
        [T0 - timedelta(seconds=10), spec.line_api, spec.boarding.map_id,
         T0 - timedelta(seconds=10), T0 + timedelta(seconds=60)],
    )

    bundle = extract_features_live_l(conn, spec, now=T0)
    # Most dynamic features should be populated (autoregressive bias is the
    # one that needs resolved forecasts; that's expected to be NaN).
    assert bundle.completeness >= 0.85
    assert bundle.values["seconds_until_next_arrival"] == pytest.approx(60.0)
    assert bundle.values["next_is_approaching"] == 1.0
    assert bundle.values["n_upcoming_arrivals_30m"] == 7
    assert bundle.values["run_n_predictions_seen"] >= 1
    assert bundle.values["line_n_runs_5m"] >= 1


def test_completeness_zero_when_no_data(conn, spec):
    bundle = extract_features_live_l(conn, spec, now=T0)
    # System state still returns COUNT(*)=0 for delayed/fault — these are
    # actual zeros, not NaNs, so completeness > 0. But run/position/headway
    # features are all NaN.
    assert bundle.completeness < 0.5


def test_categoricals_strings(conn, spec):
    for offset_s in (300, 900):
        _seed(conn, line=spec.line_api, map_id=spec.boarding.map_id,
              polled_at=T0, arrival_at=T0 + timedelta(seconds=offset_s))
    bundle = extract_features_live_l(conn, spec, now=T0)
    v = bundle.values
    assert isinstance(v["line"], str)
    assert isinstance(v["boarding_map_id"], str)   # cast to str for LightGBM categorical
    assert v["weekday_or_weekend"] in {"weekday", "weekend"}


def test_normalize_for_model_round_trip(conn, spec):
    for offset_s in (300, 900):
        _seed(conn, line=spec.line_api, map_id=spec.boarding.map_id,
              polled_at=T0, arrival_at=T0 + timedelta(seconds=offset_s))
    bundle = extract_features_live_l(conn, spec, now=T0)
    normalized = normalize_for_model(bundle.values)
    # All L_FEATURE_NAMES present in normalized
    assert set(normalized.keys()) == set(L_FEATURE_NAMES)
    # Categoricals are strings, numerics are floats
    for cat in ("line", "direction_code", "weekday_or_weekend", "mode",
                "boarding_map_id", "alighting_map_id"):
        assert isinstance(normalized[cat], str)
    for num in ("seconds_until_next_arrival", "haversine_meters"):
        v = normalized[num]
        assert isinstance(v, float)


def test_completeness_helper_matches_dynamic_subset():
    values = {name: 1.0 for name in L_FEATURE_NAMES}
    assert feature_completeness(values) == 1.0
    values["seconds_until_next_arrival"] = math.nan
    c = feature_completeness(values)
    assert 0 < c < 1
