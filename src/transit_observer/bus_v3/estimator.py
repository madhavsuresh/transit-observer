"""Online ETA estimator with reliability + reason-codes + display-state.

Ported from the validator's ``estimator.py`` to DuckDB. ``estimate_stop``
returns up to ``top`` candidate arrivals at a stop, each scored by:

- prediction freshness
- dyn/flagstop disruption signals
- detour state
- vehicle freshness + route/pattern match
- GPS map-match quality + pdist↔map-match consistency
- pdist / dstp monotonicity trends
- prediction volatility (IQR/1.349 ≈ σ over last ~12 min)
- agreement between CTA's ``prdtm`` and a geometry-derived ETA

Intervals come from the ``bus_v3_residual_quantile`` calibration table
when a populated cell exists, falling back to rule-based widths.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import duckdb

from .geometry import (
    distance_to_stop_m,
    load_pattern_points,
    map_match_to_pattern,
    path_remaining_ft,
    stop_pdist_for_pid,
)
from .inference import (
    detour_active_for_route_dir,
    stop_added_by_active_detour,
    stop_removed_by_active_detour,
)
from .models import (
    DYN_SEVERE_ABSTAIN,
    DYN_WARN,
    FLAGSTOP_ONLY_DISCHARGE,
    DataQuality,
    DisplayState,
    EstimateResult,
    ReasonCode,
)
from .util import clamp, horizon_bin, median, now_ms, quantile


FRESH_VEHICLE_S = 60.0
STALE_VEHICLE_S = 120.0
FRESH_PREDICTION_S = 60.0
STALE_PREDICTION_S = 120.0
MIN_REASONABLE_SPEED_FTPS = 1.0
MAX_REASONABLE_SPEED_FTPS = 90.0
DEFAULT_BUS_SPEED_FTPS = 18.0  # ~12.3 mph fallback


def _rows(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    params: list[Any] | None = None,
) -> list[dict[str, Any]]:
    cur = conn.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _one(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    params: list[Any] | None = None,
) -> Optional[dict[str, Any]]:
    rows = _rows(conn, sql, params)
    return rows[0] if rows else None


def _latest_prediction_rows(
    conn: duckdb.DuckDBPyConnection,
    stpid: str,
    rt: Optional[str],
    rtdir: Optional[str],
    max_age_s: float = 180.0,
) -> list[dict[str, Any]]:
    latest = _one(
        conn,
        """
        SELECT MAX(local_response_end_ms) AS t FROM bus_v3_prediction_observation
        WHERE stpid = ?
          AND (? IS NULL OR rt = ?)
          AND (? IS NULL OR rtdir = ?)
          AND query_kind = 'predictions_by_stop'
        """,
        [str(stpid), rt, rt, rtdir, rtdir],
    )
    if latest is None or latest["t"] is None:
        return []
    t = int(latest["t"])
    return _rows(
        conn,
        """
        SELECT * FROM bus_v3_prediction_observation
        WHERE stpid = ? AND local_response_end_ms = ?
          AND (? IS NULL OR rt = ?) AND (? IS NULL OR rtdir = ?)
        ORDER BY prdtm_ms, eta_s
        """,
        [str(stpid), t, rt, rt, rtdir, rtdir],
    )


def _latest_vehicle(
    conn: duckdb.DuckDBPyConnection,
    vid: Optional[str],
    rt: Optional[str] = None,
    as_of_local_ms: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    if not vid:
        return None
    params: list[Any] = [str(vid)]
    where = ["vid = ?"]
    if rt:
        where.append("rt = ?")
        params.append(rt)
    if as_of_local_ms is not None:
        where.append("local_response_end_ms <= ?")
        params.append(int(as_of_local_ms))
    return _one(
        conn,
        f"""
        SELECT * FROM bus_v3_vehicle_observation
        WHERE {' AND '.join(where)}
        ORDER BY COALESCE(tmstmp_ms, local_response_end_ms) DESC, vehicle_obs_id DESC
        LIMIT 1
        """,
        params,
    )


def _recent_vehicle_rows(
    conn: duckdb.DuckDBPyConnection,
    vid: str,
    pid: Optional[int],
    since_ms: int,
    rt: Optional[str] = None,
    as_of_local_ms: Optional[int] = None,
) -> list[dict[str, Any]]:
    params: list[Any] = [str(vid), since_ms]
    where = [
        "vid = ?",
        "COALESCE(tmstmp_ms, local_response_end_ms) >= ?",
        "pdist_ft IS NOT NULL",
    ]
    if pid is not None:
        where.append("pid = ?")
        params.append(pid)
    if rt:
        where.append("rt = ?")
        params.append(rt)
    if as_of_local_ms is not None:
        where.append("local_response_end_ms <= ?")
        params.append(int(as_of_local_ms))
    return _rows(
        conn,
        f"""
        SELECT * FROM bus_v3_vehicle_observation
        WHERE {' AND '.join(where)}
        ORDER BY COALESCE(tmstmp_ms, local_response_end_ms), vehicle_obs_id
        """,
        params,
    )


def _estimate_speed_ftps(vehicle_rows: list[dict[str, Any]]) -> tuple[Optional[float], dict[str, Any]]:
    speeds: list[float] = []
    deltas: list[dict[str, Any]] = []
    for a, b in zip(vehicle_rows, vehicle_rows[1:]):
        t0 = a["tmstmp_ms"] or a["local_response_end_ms"]
        t1 = b["tmstmp_ms"] or b["local_response_end_ms"]
        if t0 is None or t1 is None or t1 <= t0:
            continue
        if a["pdist_ft"] is None or b["pdist_ft"] is None:
            continue
        dp = b["pdist_ft"] - a["pdist_ft"]
        dt = (t1 - t0) / 1000.0
        if dp <= 0 or dt <= 0:
            continue
        speed = dp / dt
        if MIN_REASONABLE_SPEED_FTPS <= speed <= MAX_REASONABLE_SPEED_FTPS:
            speeds.append(float(speed))
            deltas.append({"dt_s": dt, "dp_ft": dp, "speed_ftps": speed})
    med = median(speeds)
    return med, {
        "speed_samples": deltas[-8:],
        "median_speed_ftps": med,
        "n_speed_samples": len(speeds),
    }


def _monotone_signal(
    values: list[Optional[float]],
    tolerance: float = 10.0,
    decreasing: bool = False,
) -> str:
    xs = [v for v in values if v is not None and math.isfinite(v)]
    if len(xs) < 3:
        return "unknown"
    deltas = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    if decreasing:
        good = sum(1 for d in deltas if d <= tolerance)
        bad = sum(1 for d in deltas if d > tolerance)
    else:
        good = sum(1 for d in deltas if d >= -tolerance)
        bad = sum(1 for d in deltas if d < -tolerance)
    if good >= len(deltas) - 1 and bad <= 1:
        return "monotone"
    if all(abs(d) <= tolerance for d in deltas):
        return "stalled"
    return "nonmonotone"


def _recent_prediction_rows(
    conn: duckdb.DuckDBPyConnection,
    pred: dict[str, Any],
    since_ms: int,
    as_of_local_ms: Optional[int] = None,
) -> list[dict[str, Any]]:
    params: list[Any] = [pred["stpid"], pred["vid"], pred["rt"], pred["rt"], since_ms]
    extra = ""
    if as_of_local_ms is not None:
        extra = " AND local_response_end_ms <= ?"
        params.append(int(as_of_local_ms))
    return _rows(
        conn,
        f"""
        SELECT * FROM bus_v3_prediction_observation
        WHERE stpid = ? AND vid = ? AND (? IS NULL OR rt = ?) AND local_response_end_ms >= ?{extra}
        ORDER BY local_response_end_ms, prediction_obs_id
        """,
        params,
    )


def _prediction_volatility_s(pred_rows: list[dict[str, Any]]) -> Optional[float]:
    vals = [r["prdtm_ms"] for r in pred_rows if r["prdtm_ms"] is not None]
    if len(vals) < 3:
        return None
    q75 = quantile([v / 1000.0 for v in vals], 0.75)
    q25 = quantile([v / 1000.0 for v in vals], 0.25)
    if q75 is None or q25 is None:
        return None
    return abs(q75 - q25) / 1.349


def _calibration_quantiles(
    conn: duckdb.DuckDBPyConnection,
    rt: Optional[str],
    stpid: str,
    rtdir: Optional[str],
    horizon_s: Optional[float],
    quality_bin: str,
    min_n: int = 20,
) -> Optional[dict[str, float]]:
    hbin = horizon_bin(horizon_s)
    queries = [
        (rt, stpid, rtdir, hbin, quality_bin),
        (rt, stpid, rtdir, hbin, "any"),
        (rt, None, rtdir, hbin, quality_bin),
        (rt, None, None, hbin, "any"),
        (None, None, None, hbin, "any"),
    ]
    for qrt, qstpid, qrtdir, qhbin, qq in queries:
        row = _one(
            conn,
            """
            SELECT * FROM bus_v3_residual_quantile
            WHERE (? IS NULL OR rt = ?)
              AND (? IS NULL OR stpid = ?)
              AND (? IS NULL OR rtdir = ?)
              AND horizon_bin = ? AND quality_bin = ?
            ORDER BY created_at_ms DESC, n DESC LIMIT 1
            """,
            [qrt, qrt, qstpid, qstpid, qrtdir, qrtdir, qhbin, qq],
        )
        if row and row["n"] is not None and row["n"] >= min_n:
            return {
                k: row[k]
                for k in ["q05_s", "q10_s", "q25_s", "q50_s", "q75_s", "q90_s", "q95_s"]
                if row.get(k) is not None
            }
    return None


def _display_state(score: float, abstain: bool) -> DisplayState:
    if abstain:
        return DisplayState.DO_NOT_DISPLAY_AS_ARRIVING
    if score >= 0.82:
        return DisplayState.HIGH_CONFIDENCE
    if score >= 0.62:
        return DisplayState.MEDIUM_CONFIDENCE
    if score >= 0.40:
        return DisplayState.LOW_CONFIDENCE
    return DisplayState.UNRELIABLE


def _data_quality(score: float, reasons: list[str], abstain: bool) -> DataQuality:
    if ReasonCode.API_ERROR.value in reasons:
        return DataQuality.API_ERROR
    if abstain:
        if (
            ReasonCode.STOP_REMOVED_BY_DETOUR.value in reasons
            or ReasonCode.DETOUR_ACTIVE.value in reasons
        ):
            return DataQuality.DETOUR_AMBIGUOUS
        return DataQuality.INSUFFICIENT
    if (
        ReasonCode.CTA_GEOMETRY_DISAGREE.value in reasons
        or ReasonCode.GPS_PDIST_INCONSISTENT.value in reasons
    ):
        return DataQuality.CONTRADICTORY
    if (
        ReasonCode.VEHICLE_POSITION_STALE.value in reasons
        or ReasonCode.PREDICTION_STALE.value in reasons
    ):
        return DataQuality.STALE
    if score >= 0.75:
        return DataQuality.GOOD
    if score >= 0.55:
        return DataQuality.ACCEPTABLE
    return DataQuality.DEGRADED


def _latest_server_time_ms(conn: duckdb.DuckDBPyConnection) -> Optional[int]:
    row = conn.execute(
        "SELECT MAX(cta_server_time_ms) FROM bus_v3_api_poll WHERE cta_server_time_ms IS NOT NULL"
    ).fetchone()
    return None if row is None else row[0]


def estimate_prediction(
    conn: duckdb.DuckDBPyConnection,
    pred: dict[str, Any],
    *,
    now_ms_override: Optional[int] = None,
    as_of_local_ms: Optional[int] = None,
) -> EstimateResult:
    current_ms = now_ms_override or pred.get("cta_server_time_ms") or now_ms()
    reasons: list[str] = []
    features: dict[str, Any] = {}
    evidence: dict[str, Any] = {
        "prediction_obs_id": pred.get("prediction_obs_id"),
        "poll_id": pred.get("poll_id"),
    }
    score = 0.52
    abstain = False

    pred_age_s = pred.get("prediction_age_s")
    if pred_age_s is None and pred.get("tmstmp_ms") is not None:
        pred_age_s = (current_ms - pred["tmstmp_ms"]) / 1000.0
    features["prediction_age_s"] = pred_age_s
    if pred_age_s is not None and pred_age_s <= FRESH_PREDICTION_S:
        reasons.append(ReasonCode.PREDICTION_FRESH.value)
        score += 0.06
    elif pred_age_s is not None and pred_age_s > STALE_PREDICTION_S:
        reasons.append(ReasonCode.PREDICTION_STALE.value)
        score -= 0.18

    dyn = pred.get("dyn")
    if dyn in DYN_SEVERE_ABSTAIN:
        if dyn in {1, 12, 18}:
            reasons.append(ReasonCode.DYN_CANCELED.value)
        elif dyn == 4:
            reasons.append(ReasonCode.DYN_EXPRESSED.value)
        else:
            reasons.append(ReasonCode.DYN_INVALIDATED.value)
        abstain = True
        score -= 0.85
    elif dyn in DYN_WARN:
        if dyn == 2:
            reasons.append(ReasonCode.DYN_REASSIGNED.value)
            score -= 0.18
        elif dyn in {9, 10}:
            reasons.append(ReasonCode.DYN_PARTIAL_TRIP.value)
            score -= 0.22
        elif dyn in {3, 14, 15}:
            reasons.append(ReasonCode.DYN_DELAYED_OR_SHIFTED.value)
            score -= 0.10
        else:
            score -= 0.06
    if pred.get("dly") == 1:
        reasons.append(ReasonCode.DLY_TRUE.value)
        score -= 0.04
    if pred.get("flagstop") == FLAGSTOP_ONLY_DISCHARGE:
        reasons.append(ReasonCode.FLAGSTOP_ONLY_DISCHARGE.value)
        abstain = True
        score -= 0.75

    if detour_active_for_route_dir(conn, pred.get("rt"), pred.get("rtdir")):
        reasons.append(ReasonCode.DETOUR_ACTIVE.value)
        score -= 0.10
    if stop_removed_by_active_detour(conn, pred["stpid"]):
        reasons.append(ReasonCode.STOP_REMOVED_BY_DETOUR.value)
        abstain = True
        score -= 0.85
    if stop_added_by_active_detour(conn, pred["stpid"]):
        reasons.append(ReasonCode.STOP_ADDED_BY_DETOUR.value)
        score -= 0.03

    vehicle = _latest_vehicle(conn, pred.get("vid"), pred.get("rt"), as_of_local_ms=as_of_local_ms)
    vehicle_age_s: Optional[float] = None
    stop_pdist: Optional[float] = None
    remaining_ft: Optional[float] = None
    geometry_eta_s: Optional[float] = None
    speed_ftps: Optional[float] = None
    distance_stop_m: Optional[float] = None

    if vehicle is None:
        reasons.append(ReasonCode.VEHICLE_NOT_FOUND.value)
        score -= 0.30
    else:
        reasons.append(ReasonCode.VEHICLE_FOUND.value)
        score += 0.06
        vehicle_age_s = vehicle["vehicle_age_s"]
        if vehicle_age_s is None and vehicle.get("tmstmp_ms") is not None:
            vehicle_age_s = (current_ms - vehicle["tmstmp_ms"]) / 1000.0
        features["vehicle_age_s"] = vehicle_age_s
        if vehicle_age_s is not None and vehicle_age_s <= FRESH_VEHICLE_S:
            reasons.append(ReasonCode.VEHICLE_POSITION_FRESH.value)
            score += 0.12
        elif vehicle_age_s is None or vehicle_age_s > STALE_VEHICLE_S:
            reasons.append(ReasonCode.VEHICLE_POSITION_STALE.value)
            score -= 0.30

        if vehicle.get("rt") == pred.get("rt"):
            reasons.append(ReasonCode.ROUTE_DIRECTION_MATCH.value)
            score += 0.04
        else:
            reasons.append(ReasonCode.ROUTE_DIRECTION_MISMATCH.value)
            score -= 0.18

        if vehicle.get("pid") is not None:
            stop_pdist = stop_pdist_for_pid(conn, pred["stpid"], int(vehicle["pid"]))
        if stop_pdist is not None:
            reasons.append(ReasonCode.PATTERN_MATCH.value)
            score += 0.08
            remaining_ft = path_remaining_ft(vehicle.get("pdist_ft"), stop_pdist)
            features["remaining_ft_by_pdist"] = remaining_ft
        else:
            reasons.append(ReasonCode.PATTERN_MISMATCH.value)
            score -= 0.14

        points = (
            load_pattern_points(conn, int(vehicle["pid"]))
            if vehicle.get("pid") is not None
            else []
        )
        if points and vehicle.get("lat") is not None and vehicle.get("lon") is not None:
            mm = map_match_to_pattern(points, vehicle["lat"], vehicle["lon"], vehicle.get("hdg"))
            features["map_match"] = mm.as_dict()
            if mm.quality in {"HIGH", "MEDIUM"}:
                reasons.append(ReasonCode.GPS_ON_EXPECTED_PATTERN.value)
                score += 0.04
            else:
                reasons.append(ReasonCode.GPS_OFF_EXPECTED_PATTERN.value)
                score -= 0.12
            if mm.projected_pdist_ft is not None and vehicle.get("pdist_ft") is not None:
                pdiff = abs(mm.projected_pdist_ft - vehicle["pdist_ft"])
                features["gps_pdist_diff_ft"] = pdiff
                if pdiff > 2_000:
                    reasons.append(ReasonCode.GPS_PDIST_INCONSISTENT.value)
                    score -= 0.14

        distance_stop_m = distance_to_stop_m(conn, pred["stpid"], vehicle.get("lat"), vehicle.get("lon"))
        features["distance_to_stop_m"] = distance_stop_m
        if distance_stop_m is not None and distance_stop_m <= 60:
            reasons.append(ReasonCode.GPS_NEAR_STOP.value)

        recent_vehicle = _recent_vehicle_rows(
            conn,
            pred["vid"],
            vehicle.get("pid"),
            current_ms - 12 * 60_000,
            pred.get("rt"),
            as_of_local_ms=as_of_local_ms,
        )
        speed_ftps, speed_debug = _estimate_speed_ftps(recent_vehicle)
        features.update(speed_debug)
        pdist_trend = _monotone_signal([r["pdist_ft"] for r in recent_vehicle], decreasing=False)
        features["pdist_trend"] = pdist_trend
        if pdist_trend == "monotone":
            reasons.append(ReasonCode.PDIST_INCREASING.value)
            score += 0.05
        elif pdist_trend == "stalled":
            reasons.append(ReasonCode.PDIST_STALLED.value)
            score -= 0.06

        if remaining_ft is not None and remaining_ft < -50:
            reasons.append(ReasonCode.PDIST_CROSSED_STOP.value)
            if str(pred.get("typ") or "A").upper() != "D":
                abstain = True
                reasons.append(ReasonCode.ARRIVAL_ESTIMATE_ABSTAINED.value)
                score -= 0.55
            else:
                score -= 0.25
        elif remaining_ft is not None and remaining_ft >= 0:
            if speed_ftps is not None:
                geometry_eta_s = remaining_ft / max(speed_ftps, 0.1)
            elif pred.get("dstp_ft") is not None:
                geometry_eta_s = float(pred["dstp_ft"]) / DEFAULT_BUS_SPEED_FTPS
            features["geometry_eta_s"] = geometry_eta_s

    recent_preds = _recent_prediction_rows(conn, pred, current_ms - 12 * 60_000, as_of_local_ms=as_of_local_ms)
    dstp_trend = _monotone_signal([r["dstp_ft"] for r in recent_preds], tolerance=25.0, decreasing=True)
    features["dstp_trend"] = dstp_trend
    if dstp_trend == "monotone":
        reasons.append(ReasonCode.DSTP_DECREASING.value)
        score += 0.05
    elif dstp_trend == "stalled":
        reasons.append(ReasonCode.DSTP_STALLED.value)
        score -= 0.08
    elif dstp_trend == "nonmonotone":
        reasons.append(ReasonCode.DSTP_INCREASING.value)
        score -= 0.10

    vol_s = _prediction_volatility_s(recent_preds)
    features["prediction_volatility_s"] = vol_s
    if vol_s is not None:
        if vol_s <= 45:
            reasons.append(ReasonCode.PREDICTION_STABLE.value)
            score += 0.03
        elif vol_s >= 120:
            reasons.append(ReasonCode.PREDICTION_VOLATILE.value)
            score -= 0.15

    cta_eta_s = pred.get("eta_s")
    if cta_eta_s is None and pred.get("prdtm_ms") is not None:
        cta_eta_s = (pred["prdtm_ms"] - current_ms) / 1000.0
    features["cta_eta_s"] = cta_eta_s

    if (
        cta_eta_s is not None
        and cta_eta_s <= 90
        and distance_stop_m is not None
        and distance_stop_m > 350
        and vehicle_age_s is not None
        and vehicle_age_s <= FRESH_VEHICLE_S
    ):
        reasons.append(ReasonCode.DUE_BUT_VEHICLE_NOT_NEAR_STOP.value)
        score -= 0.18

    predicted_ms: Optional[int] = None
    disagreement_s: Optional[float] = None
    if abstain:
        predicted_ms = None
    elif cta_eta_s is not None and geometry_eta_s is not None and geometry_eta_s >= 0:
        disagreement_s = abs(cta_eta_s - geometry_eta_s)
        features["cta_geometry_disagreement_s"] = disagreement_s
        if disagreement_s <= 75:
            reasons.append(ReasonCode.CTA_GEOMETRY_AGREE.value)
            score += 0.08
            eta = 0.68 * cta_eta_s + 0.32 * geometry_eta_s
        else:
            reasons.append(ReasonCode.CTA_GEOMETRY_DISAGREE.value)
            score -= min(0.28, disagreement_s / 600.0)
            cta_weight = 0.55
            if cta_eta_s <= 90 and geometry_eta_s > 180:
                cta_weight = 0.25
            elif pred_age_s is not None and pred_age_s > FRESH_PREDICTION_S:
                cta_weight = 0.35
            elif speed_ftps is None:
                cta_weight = 0.75
            eta = cta_weight * cta_eta_s + (1 - cta_weight) * geometry_eta_s
        predicted_ms = int(current_ms + max(0.0, eta) * 1000)
    elif not abstain and cta_eta_s is not None:
        eta = max(0.0, cta_eta_s)
        predicted_ms = int(current_ms + eta * 1000)
        score -= 0.06 if vehicle is None else 0.02
    elif not abstain and geometry_eta_s is not None:
        eta = max(0.0, geometry_eta_s)
        predicted_ms = int(current_ms + eta * 1000)
        score -= 0.10
    else:
        abstain = True
        reasons.append(ReasonCode.ARRIVAL_ESTIMATE_ABSTAINED.value)
        score -= 0.35

    score = clamp(score, 0.0, 1.0)
    display = _display_state(score, abstain)
    quality = _data_quality(score, reasons, abstain)

    interval80 = interval90 = interval95 = (None, None)
    if predicted_ms is not None:
        horizon_s = max(0.0, (predicted_ms - current_ms) / 1000.0)
        quality_bin = "high" if score >= 0.75 else "medium" if score >= 0.55 else "low"
        cal = _calibration_quantiles(conn, pred.get("rt"), pred["stpid"], pred.get("rtdir"), horizon_s, quality_bin)
        if cal:
            interval80 = (
                int(predicted_ms + cal.get("q10_s", -60.0) * 1000),
                int(predicted_ms + cal.get("q90_s", 60.0) * 1000),
            )
            interval90 = (
                int(predicted_ms + cal.get("q05_s", -90.0) * 1000),
                int(predicted_ms + cal.get("q95_s", 90.0) * 1000),
            )
            mid = predicted_ms
            lo90, hi90 = interval90
            interval95 = (
                int(mid - 1.25 * (mid - lo90)),
                int(mid + 1.25 * (hi90 - mid)),
            )
            features["calibration_source"] = "empirical"
        else:
            base = 28.0 + 0.12 * horizon_s
            if vehicle_age_s is not None:
                base += max(0.0, vehicle_age_s - 30.0) * 0.35
            if pred_age_s is not None:
                base += max(0.0, pred_age_s - 30.0) * 0.25
            if disagreement_s is not None:
                base += min(180.0, 0.35 * disagreement_s)
            if vol_s is not None:
                base += min(120.0, 0.45 * vol_s)
            if ReasonCode.DETOUR_ACTIVE.value in reasons:
                base += 60.0
            base *= 1.0 + max(0.0, 0.75 - score)
            base = clamp(base, 25.0, 600.0)
            interval80 = (int(predicted_ms - base * 1000), int(predicted_ms + base * 1000))
            interval90 = (int(predicted_ms - 1.55 * base * 1000), int(predicted_ms + 1.55 * base * 1000))
            interval95 = (int(predicted_ms - 2.05 * base * 1000), int(predicted_ms + 2.05 * base * 1000))
            features["calibration_source"] = "fallback_rule"
            features["fallback_interval80_half_width_s"] = base

    if abstain:
        if ReasonCode.STOP_REMOVED_BY_DETOUR.value in reasons:
            rider_message = "No reliable arrival estimate: the stop appears affected by an active detour."
        elif any(
            r in reasons
            for r in (
                ReasonCode.DYN_CANCELED.value,
                ReasonCode.DYN_INVALIDATED.value,
                ReasonCode.DYN_EXPRESSED.value,
            )
        ):
            rider_message = "No reliable arrival estimate: CTA disruption fields indicate this arrival should not be displayed as a normal pickup."
        else:
            rider_message = "No reliable arrival estimate from the available data."
    elif predicted_ms is not None:
        minutes = max(0, round((predicted_ms - current_ms) / 60_000))
        rider_message = f"Estimated arrival in about {minutes} minute{'s' if minutes != 1 else ''}."
        if display in (DisplayState.LOW_CONFIDENCE, DisplayState.UNRELIABLE):
            rider_message += " Evidence is weak; verify before relying on it."
    else:
        rider_message = "No reliable arrival estimate from the available data."

    return EstimateResult(
        generated_at_ms=current_ms,
        stpid=str(pred["stpid"]),
        rt=pred.get("rt"),
        rtdir=pred.get("rtdir"),
        vid=pred.get("vid"),
        destination=pred.get("des"),
        tatripid=pred.get("tatripid"),
        tablockid=pred.get("tablockid"),
        predicted_arrival_ms=predicted_ms,
        interval80_low_ms=interval80[0],
        interval80_high_ms=interval80[1],
        interval90_low_ms=interval90[0],
        interval90_high_ms=interval90[1],
        interval95_low_ms=interval95[0],
        interval95_high_ms=interval95[1],
        reliability=round(float(score), 4),
        display_state=display,
        data_quality=quality,
        rider_message=rider_message,
        reason_codes=sorted(set(reasons)),
        features=features,
        evidence=evidence,
    )


def estimate_stop(
    conn: duckdb.DuckDBPyConnection,
    stpid: str,
    rt: Optional[str] = None,
    rtdir: Optional[str] = None,
    top: int = 4,
    *,
    now_ms_override: Optional[int] = None,
) -> list[EstimateResult]:
    rows = _latest_prediction_rows(conn, stpid, rt, rtdir)
    current_ms = now_ms_override or _latest_server_time_ms(conn)
    estimates = [
        estimate_prediction(conn, row, now_ms_override=current_ms)
        for row in rows
    ]
    estimates.sort(
        key=lambda e: (
            e.predicted_arrival_ms is None,
            e.predicted_arrival_ms if e.predicted_arrival_ms is not None else 2**63 - 1,
            -e.reliability,
        )
    )
    return estimates[:top]
