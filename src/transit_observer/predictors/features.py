"""Feature extraction for the learned L predictor.

Shared between live serving (called from corpus.py at predict time) and
offline training (the same names appear in training/dataset.py's CTE).
Keep the schemas in lockstep — drift between the two is how learned
predictors silently degrade.

Categoricals are returned as strings; LightGBM handles them natively.
Numerics are floats. Missing values use NaN, not None — polars / pandas
both prefer NaN for downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import duckdb

from ..trip_generator import TripSpec


# Names of features the GBM expects. Used both as a schema check (does the
# trained model match this code?) and to compute feature_completeness.
L_FEATURE_NAMES: tuple[str, ...] = (
    # categoricals
    "line", "direction_code", "boarding_map_id", "alighting_map_id",
    "hour_of_day", "weekday_or_weekend", "mode",
    # static
    "haversine_meters",
    # live
    "seconds_until_next_arrival",
    "next_is_approaching",
    "n_upcoming_arrivals_30m",
    "headway_median_s",
    "headway_iqr_s",
    "headway_min_s",
    # run history (for the next candidate run)
    "run_n_predictions_seen",
    "run_arrival_variance_s",
    "run_drift_s",
    # system state
    "line_delayed_5m",
    "line_fault_5m",
    "line_n_runs_5m",
    # position state
    "position_age_s",
    "position_next_arrival_offset_s",
    # autoregressive bias signal
    "line_dir_hour_recent_mean_residual_s",
)


# Subset that participates in feature_completeness (excludes static
# categoricals which are always present).
_DYNAMIC_FEATURES: tuple[str, ...] = (
    "seconds_until_next_arrival", "next_is_approaching", "n_upcoming_arrivals_30m",
    "headway_median_s", "headway_iqr_s", "headway_min_s",
    "run_n_predictions_seen", "run_arrival_variance_s", "run_drift_s",
    "line_delayed_5m", "line_fault_5m", "line_n_runs_5m",
    "position_age_s", "position_next_arrival_offset_s",
    "line_dir_hour_recent_mean_residual_s",
)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass(frozen=True)
class FeatureBundle:
    values: dict[str, Any]
    completeness: float  # 0..1


def feature_completeness(values: dict[str, Any]) -> float:
    """Fraction of dynamic features that are present (not NaN/None)."""
    if not _DYNAMIC_FEATURES:
        return 1.0
    n_present = 0
    for name in _DYNAMIC_FEATURES:
        v = values.get(name)
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        n_present += 1
    return n_present / len(_DYNAMIC_FEATURES)


def extract_features_live_l(
    conn: duckdb.DuckDBPyConnection,
    spec: TripSpec,
    *,
    now: datetime,
    horizon_minutes: float = 30.0,
) -> FeatureBundle:
    """Snapshot features for one L prediction at ``now``.

    Returns a FeatureBundle whose ``values`` dict is a flat
    feature_name -> scalar map plus ``completeness`` in [0, 1]. Callers
    persist ``values`` into forecast_queue.feature_json verbatim so the
    same schema feeds both training and serving.
    """
    cutoff = now - timedelta(minutes=5)
    horizon = now + timedelta(minutes=horizon_minutes)

    # Static / time
    weekday_or_weekend = "weekday" if now.weekday() < 5 else "weekend"
    haversine_m = _haversine(
        spec.boarding.latitude, spec.boarding.longitude,
        spec.alighting.latitude, spec.alighting.longitude,
    )

    values: dict[str, Any] = {
        "line": spec.line_api,
        "direction_code": spec.direction_label,
        "boarding_map_id": str(spec.boarding.map_id),
        "alighting_map_id": str(spec.alighting.map_id),
        "hour_of_day": now.hour,
        "weekday_or_weekend": weekday_or_weekend,
        "mode": "L",
        "haversine_meters": haversine_m,
        # dynamic features default to NaN; populated below if data exists
        "seconds_until_next_arrival": math.nan,
        "next_is_approaching": math.nan,
        "n_upcoming_arrivals_30m": 0,
        "headway_median_s": math.nan,
        "headway_iqr_s": math.nan,
        "headway_min_s": math.nan,
        "run_n_predictions_seen": math.nan,
        "run_arrival_variance_s": math.nan,
        "run_drift_s": math.nan,
        "line_delayed_5m": math.nan,
        "line_fault_5m": math.nan,
        "line_n_runs_5m": math.nan,
        "position_age_s": math.nan,
        "position_next_arrival_offset_s": math.nan,
        "line_dir_hour_recent_mean_residual_s": math.nan,
    }

    # Upcoming arrivals at boarding for this line
    rows = conn.execute(
        """
        SELECT arrival_at, is_approaching, is_delayed, is_fault, run_number
          FROM train_arrivals_raw
         WHERE line = ? AND map_id = ?
           AND polled_at >= ?
           AND arrival_at >= ? AND arrival_at <= ?
           AND COALESCE(is_fault, FALSE) = FALSE
         ORDER BY arrival_at
        """,
        [spec.line_api, spec.boarding.map_id, cutoff, now, horizon],
    ).fetchall()

    # Deduplicate by arrival_at, keeping the most recent prediction
    seen: dict[datetime, tuple[bool, bool, bool, str]] = {}
    for arrival_at, is_app, is_dly, is_flt, run_num in rows:
        seen.setdefault(arrival_at, (bool(is_app), bool(is_dly), bool(is_flt), run_num))
    arrivals = sorted(seen.items(), key=lambda kv: kv[0])
    values["n_upcoming_arrivals_30m"] = len(arrivals)

    next_run_number: str | None = None
    if arrivals:
        next_arrival_at, (next_is_app, _next_dly, _next_flt, next_run_number) = arrivals[0]
        values["seconds_until_next_arrival"] = max(0.0, (next_arrival_at - now).total_seconds())
        values["next_is_approaching"] = 1.0 if next_is_app else 0.0

        if len(arrivals) >= 2:
            gaps = [
                (arrivals[i][0] - arrivals[i - 1][0]).total_seconds()
                for i in range(1, len(arrivals))
            ]
            gaps_sorted = sorted(gaps)
            values["headway_median_s"] = gaps_sorted[len(gaps_sorted) // 2]
            values["headway_min_s"] = gaps_sorted[0]
            if len(gaps_sorted) >= 4:
                q1 = gaps_sorted[len(gaps_sorted) // 4]
                q3 = gaps_sorted[(3 * len(gaps_sorted)) // 4]
                values["headway_iqr_s"] = q3 - q1
            else:
                values["headway_iqr_s"] = gaps_sorted[-1] - gaps_sorted[0]

    # Run-history for the next candidate run
    if next_run_number is not None:
        run_rows = conn.execute(
            """
            SELECT EPOCH(arrival_at) AS aep
              FROM train_arrivals_raw
             WHERE line = ? AND run_number = ? AND map_id = ?
               AND polled_at <= ?
            """,
            [spec.line_api, next_run_number, spec.boarding.map_id, now],
        ).fetchall()
        if run_rows:
            aeps = [r[0] for r in run_rows if r[0] is not None]
            values["run_n_predictions_seen"] = float(len(aeps))
            if len(aeps) >= 2:
                mean = sum(aeps) / len(aeps)
                var = sum((a - mean) ** 2 for a in aeps) / len(aeps)
                values["run_arrival_variance_s"] = var
                values["run_drift_s"] = max(aeps) - min(aeps)
            else:
                values["run_arrival_variance_s"] = 0.0
                values["run_drift_s"] = 0.0

    # System state: line-wide rolling 5-min counts
    sys_row = conn.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE COALESCE(is_delayed, FALSE))                   AS delayed_5m,
            COUNT(*) FILTER (WHERE COALESCE(is_fault, FALSE))                     AS fault_5m,
            COUNT(DISTINCT run_number)                                            AS n_runs_5m
          FROM train_arrivals_raw
         WHERE line = ? AND polled_at >= ? AND polled_at <= ?
        """,
        [spec.line_api, cutoff, now],
    ).fetchone()
    if sys_row is not None:
        delayed_5m, fault_5m, n_runs_5m = sys_row
        values["line_delayed_5m"] = float(delayed_5m or 0)
        values["line_fault_5m"] = float(fault_5m or 0)
        values["line_n_runs_5m"] = float(n_runs_5m or 0)

    # Position state for the next candidate run
    if next_run_number is not None:
        pos_row = conn.execute(
            """
            SELECT polled_at, next_arrival_at
              FROM train_positions_raw
             WHERE line = ? AND run_number = ? AND next_station_map_id = ?
               AND polled_at <= ?
             ORDER BY polled_at DESC
             LIMIT 1
            """,
            [spec.line_api, next_run_number, spec.boarding.map_id, now],
        ).fetchone()
        if pos_row is not None:
            pos_polled_at, pos_next_at = pos_row
            if pos_polled_at is not None:
                values["position_age_s"] = max(0.0, (now - pos_polled_at).total_seconds())
            if pos_next_at is not None:
                values["position_next_arrival_offset_s"] = (pos_next_at - now).total_seconds()

    # Autoregressive bias: recent mean residual on same (line, direction, hour-bucket)
    bias_row = conn.execute(
        """
        SELECT AVG(o.p50_residual_seconds) AS mean_resid
          FROM forecast_outcomes o
          JOIN forecast_queue q USING (forecast_id)
         WHERE q.line = ? AND q.direction_code = ?
           AND EXTRACT(hour FROM q.leave_at)::INTEGER = ?
           AND q.status = 'resolved'
           AND o.resolved_at >= ?
           AND o.resolved_at <= ?
        """,
        [
            spec.line_api, spec.direction_label, now.hour,
            now - timedelta(minutes=30), now,
        ],
    ).fetchone()
    if bias_row is not None and bias_row[0] is not None:
        values["line_dir_hour_recent_mean_residual_s"] = float(bias_row[0])

    return FeatureBundle(values=values, completeness=feature_completeness(values))


def normalize_for_model(values: dict[str, Any]) -> dict[str, Any]:
    """Cast feature values to LightGBM-friendly types.

    Categoricals stay as strings (LightGBM Pandas API recognizes
    ``category`` dtype). Numerics become floats, with NaN for missing.
    """
    out: dict[str, Any] = {}
    for name in L_FEATURE_NAMES:
        v = values.get(name)
        if name in {"line", "direction_code", "boarding_map_id", "alighting_map_id",
                    "weekday_or_weekend", "mode"}:
            out[name] = "" if v is None else str(v)
            continue
        if v is None:
            out[name] = math.nan
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            out[name] = math.nan
            continue
        out[name] = f
    return out
