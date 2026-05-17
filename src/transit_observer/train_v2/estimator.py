"""Online ETA estimator for the train_v2 pipeline.

For a given (line, map_id), pulls the latest ``ttarrivals`` predictions
and scores each one with cross-validation against:

- The same run's ttfollow trajectory (does the per-run snapshot agree
  about the predicted arrival at this stop?).
- The corresponding position observation (is the train actually
  approaching this station? what does ``isApp`` say? has the next
  station already advanced past us?).
- GTFS-RT TripUpdates (independent prediction stream; ``delay`` field
  gives schedule adherence).

Reliability is built from additive evidence (fresh / approaching /
multiple independent agree) minus disruption (isFlt / isDly / position
already past). Intervals come from the empirical ``train_v2_residual_quantile``
table when populated; otherwise a rule-based fallback widens with
prediction age + volatility + reliability discount.
"""

from __future__ import annotations

from typing import Any, Optional

import duckdb

from .models import (
    DataQuality,
    DisplayState,
    EstimateResult,
    ReasonCode,
)
from .util import clamp, horizon_bin, now_ms, quantile


FRESH_PREDICTION_S = 60.0
STALE_PREDICTION_S = 180.0
FRESH_POSITION_S = 90.0
STALE_POSITION_S = 240.0


def _rows(conn: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    cur = conn.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _one(conn: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None) -> Optional[dict[str, Any]]:
    rows = _rows(conn, sql, params)
    return rows[0] if rows else None


def _latest_server_time_ms(conn: duckdb.DuckDBPyConnection) -> Optional[int]:
    row = conn.execute(
        "SELECT MAX(cta_server_time_ms) FROM train_v2_api_poll WHERE cta_server_time_ms IS NOT NULL"
    ).fetchone()
    return None if row is None else row[0]


def _latest_arrival_rows(
    conn: duckdb.DuckDBPyConnection,
    map_id: str,
    line: Optional[str],
    direction_code: Optional[str],
) -> list[dict[str, Any]]:
    latest = _one(
        conn,
        """
        SELECT MAX(local_response_end_ms) AS t
          FROM train_v2_arrival_observation
         WHERE map_id = ?
           AND (? IS NULL OR line = ?)
           AND (? IS NULL OR direction_code = ?)
           AND query_kind = 'arrivals_by_station'
        """,
        [str(map_id), line, line, direction_code, direction_code],
    )
    if latest is None or latest["t"] is None:
        return []
    t = int(latest["t"])
    return _rows(
        conn,
        """
        SELECT * FROM train_v2_arrival_observation
         WHERE map_id = ? AND local_response_end_ms = ?
           AND (? IS NULL OR line = ?)
           AND (? IS NULL OR direction_code = ?)
         ORDER BY arrival_at_ms NULLS LAST, eta_s NULLS LAST
        """,
        [str(map_id), t, line, line, direction_code, direction_code],
    )


def _latest_position_for_run(
    conn: duckdb.DuckDBPyConnection,
    run_number: str,
    line: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    return _one(
        conn,
        """
        SELECT * FROM train_v2_position_observation
         WHERE run_number = ?
           AND (? IS NULL OR line = ?)
         ORDER BY COALESCE(predicted_at_ms, local_response_end_ms) DESC, position_obs_id DESC
         LIMIT 1
        """,
        [str(run_number), line, line],
    )


def _latest_follow_for_run_stop(
    conn: duckdb.DuckDBPyConnection,
    run_number: str,
    map_id: str,
) -> Optional[dict[str, Any]]:
    return _one(
        conn,
        """
        SELECT * FROM train_v2_follow_observation
         WHERE run_number = ? AND map_id = ?
         ORDER BY local_response_end_ms DESC, follow_obs_id DESC
         LIMIT 1
        """,
        [str(run_number), str(map_id)],
    )


def _recent_arrivals_for_run_stop(
    conn: duckdb.DuckDBPyConnection,
    run_number: str,
    map_id: str,
    *,
    since_ms: int,
) -> list[dict[str, Any]]:
    return _rows(
        conn,
        """
        SELECT * FROM train_v2_arrival_observation
         WHERE run_number = ? AND map_id = ? AND local_response_end_ms >= ?
         ORDER BY local_response_end_ms
        """,
        [str(run_number), str(map_id), since_ms],
    )


def _gtfsrt_for_stop(
    conn: duckdb.DuckDBPyConnection,
    map_id: str,
    *,
    around_ms: int,
    window_s: int = 180,
) -> Optional[dict[str, Any]]:
    """Look for a GTFS-RT TripUpdate predicting arrival at this stop
    near the ttarrivals prediction. We don't have a clean cross-stream
    ID map between ttarrivals.staId and GTFS-RT stop_id (different ID
    spaces), so this is a best-effort lookup keyed by stop_id-as-text."""
    lo = around_ms - window_s * 1000
    hi = around_ms + window_s * 1000
    return _one(
        conn,
        """
        SELECT * FROM train_v2_gtfsrt_trip_update
         WHERE stop_id = ? AND arrival_time_ms BETWEEN ? AND ?
         ORDER BY ABS(arrival_time_ms - ?)
         LIMIT 1
        """,
        [str(map_id), lo, hi, around_ms],
    )


def _calibration_quantiles(
    conn: duckdb.DuckDBPyConnection,
    line: Optional[str],
    map_id: str,
    direction_code: Optional[str],
    horizon_s: Optional[float],
    quality_bin: str,
    min_n: int = 20,
) -> Optional[dict[str, float]]:
    hbin = horizon_bin(horizon_s)
    queries = [
        (line, map_id, direction_code, hbin, quality_bin),
        (line, map_id, direction_code, hbin, "any"),
        (line, None, direction_code, hbin, quality_bin),
        (line, None, None, hbin, "any"),
        (None, None, None, hbin, "any"),
    ]
    for qline, qmap, qdir, qhbin, qq in queries:
        row = _one(
            conn,
            """
            SELECT * FROM train_v2_residual_quantile
             WHERE (? IS NULL OR line = ?)
               AND (? IS NULL OR map_id = ?)
               AND (? IS NULL OR direction_code = ?)
               AND horizon_bin = ? AND quality_bin = ?
             ORDER BY created_at_ms DESC, n DESC LIMIT 1
            """,
            [qline, qline, qmap, qmap, qdir, qdir, qhbin, qq],
        )
        if row and row.get("n") is not None and row["n"] >= min_n:
            return {
                k: row[k] for k in ("q05_s", "q10_s", "q25_s", "q50_s", "q75_s", "q90_s", "q95_s")
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
        return DataQuality.INSUFFICIENT
    if (
        ReasonCode.IS_FAULT_FLAG.value in reasons
        or ReasonCode.FOLLOW_TRAJECTORY_DIVERGENT.value in reasons
    ):
        return DataQuality.CONTRADICTORY
    if (
        ReasonCode.PREDICTION_STALE.value in reasons
        or ReasonCode.POSITION_STALE.value in reasons
    ):
        return DataQuality.STALE
    if score >= 0.75:
        return DataQuality.GOOD
    if score >= 0.55:
        return DataQuality.ACCEPTABLE
    return DataQuality.DEGRADED


def estimate_arrival(
    conn: duckdb.DuckDBPyConnection,
    pred: dict[str, Any],
    *,
    now_ms_override: Optional[int] = None,
) -> EstimateResult:
    """Score one ``train_v2_arrival_observation`` row.

    Returns an ``EstimateResult`` whose ``predicted_arrival_ms`` is the
    consensus of ttarrivals, ttfollow (when present), and GTFS-RT
    (when present), and whose reliability reflects the agreement
    between those streams plus position-stream corroboration.
    """
    current_ms = now_ms_override or pred.get("cta_server_time_ms") or now_ms()
    reasons: list[str] = []
    features: dict[str, Any] = {}
    evidence: dict[str, Any] = {
        "arrival_obs_id": pred.get("arrival_obs_id"),
        "poll_id": pred.get("poll_id"),
    }
    score = 0.55
    abstain = False

    # 1. Prediction freshness.
    pred_age_s = pred.get("prediction_age_s")
    if pred_age_s is None and pred.get("predicted_at_ms") is not None:
        pred_age_s = (current_ms - pred["predicted_at_ms"]) / 1000.0
    features["prediction_age_s"] = pred_age_s
    if pred_age_s is not None and pred_age_s <= FRESH_PREDICTION_S:
        reasons.append(ReasonCode.PREDICTION_FRESH.value)
        score += 0.10
    elif pred_age_s is not None and pred_age_s > STALE_PREDICTION_S:
        reasons.append(ReasonCode.PREDICTION_STALE.value)
        score -= 0.20

    # 2. Train Tracker flags.
    if pred.get("is_fault"):
        reasons.append(ReasonCode.IS_FAULT_FLAG.value)
        abstain = True
        score -= 0.60
    if pred.get("is_delayed"):
        reasons.append(ReasonCode.IS_DELAYED_FLAG.value)
        score -= 0.08
    if pred.get("is_approaching"):
        reasons.append(ReasonCode.IS_APPROACHING_TRUE.value)
        score += 0.10
    if pred.get("is_scheduled"):
        reasons.append(ReasonCode.IS_SCHEDULED_FALLBACK.value)
        score -= 0.15  # scheduled-only with no live tracking ⇒ weaker signal

    # 3. Position-stream corroboration.
    run_number = pred.get("run_number")
    map_id = str(pred["map_id"])
    line = pred.get("line")
    direction_code = pred.get("direction_code")
    position = _latest_position_for_run(conn, run_number, line) if run_number else None
    position_age_s: Optional[float] = None
    if position is None:
        reasons.append(ReasonCode.POSITION_NOT_FOUND.value)
        score -= 0.18
    else:
        pos_t = position.get("predicted_at_ms") or position.get("local_response_end_ms")
        if pos_t is not None:
            position_age_s = (current_ms - pos_t) / 1000.0
            features["position_age_s"] = position_age_s
            if position_age_s <= FRESH_POSITION_S:
                reasons.append(ReasonCode.POSITION_FOUND.value)
                score += 0.08
            elif position_age_s > STALE_POSITION_S:
                reasons.append(ReasonCode.POSITION_STALE.value)
                score -= 0.15
        next_map = position.get("next_station_map_id")
        if next_map == map_id:
            reasons.append(ReasonCode.NEXT_STATION_MATCH.value)
            score += 0.10
        elif next_map is not None:
            # Train's next station differs from the queried station —
            # either the train hasn't reached us yet (it's behind X with
            # X != our map_id but X precedes us on the line) or it has
            # passed us already (more concerning).
            from .topology import is_downstream
            if direction_code:
                topo = is_downstream(
                    conn,
                    line=str(line or ""),
                    direction_code=str(direction_code),
                    from_map=str(next_map),
                    to_map=map_id,
                )
                if topo is True:
                    reasons.append(ReasonCode.NEXT_STATION_MATCH.value)
                    score += 0.03  # behind us but still inbound
                elif topo is False:
                    reasons.append(ReasonCode.NEXT_STATION_ADVANCED.value)
                    abstain = True
                    score -= 0.55
                else:
                    reasons.append(ReasonCode.FOLLOW_MISSING.value)
            else:
                reasons.append(ReasonCode.FOLLOW_MISSING.value)
        if position.get("is_fault"):
            reasons.append(ReasonCode.IS_FAULT_FLAG.value)
            abstain = True
            score -= 0.4

    # 4. ttfollow trajectory cross-check.
    follow = _latest_follow_for_run_stop(conn, run_number, map_id) if run_number else None
    follow_arrival_ms = follow.get("arrival_at_ms") if follow else None
    if follow_arrival_ms is None:
        reasons.append(ReasonCode.FOLLOW_MISSING.value)
    else:
        ttarrivals_ms = pred.get("arrival_at_ms")
        if ttarrivals_ms is not None:
            disagreement_s = abs(follow_arrival_ms - ttarrivals_ms) / 1000.0
            features["follow_disagreement_s"] = disagreement_s
            if disagreement_s <= 30:
                reasons.append(ReasonCode.FOLLOW_TRAJECTORY_CONSISTENT.value)
                score += 0.10
            elif disagreement_s > 90:
                reasons.append(ReasonCode.FOLLOW_TRAJECTORY_DIVERGENT.value)
                score -= 0.12

    # 5. GTFS-RT cross-validation.
    arrival_ms_anchor = pred.get("arrival_at_ms") or 0
    gtfsrt = _gtfsrt_for_stop(conn, map_id, around_ms=arrival_ms_anchor) if arrival_ms_anchor else None
    if gtfsrt is None:
        reasons.append(ReasonCode.GTFSRT_MISSING.value)
    else:
        gtfsrt_ms = gtfsrt.get("arrival_time_ms")
        if gtfsrt_ms is not None and arrival_ms_anchor:
            disagreement_s = abs(gtfsrt_ms - arrival_ms_anchor) / 1000.0
            features["gtfsrt_disagreement_s"] = disagreement_s
            if disagreement_s <= 45:
                reasons.append(ReasonCode.GTFSRT_AGREE.value)
                score += 0.10
            elif disagreement_s > 120:
                reasons.append(ReasonCode.GTFSRT_DISAGREE.value)
                score -= 0.10
        if gtfsrt.get("arrival_delay_seconds") is not None:
            reasons.append(ReasonCode.GTFSRT_DELAY_REPORTED.value)
            features["gtfsrt_delay_seconds"] = gtfsrt["arrival_delay_seconds"]

    # 6. Three-way agreement bonus.
    if (
        ReasonCode.IS_APPROACHING_TRUE.value in reasons
        and ReasonCode.FOLLOW_TRAJECTORY_CONSISTENT.value in reasons
        and ReasonCode.GTFSRT_AGREE.value in reasons
    ):
        reasons.append(ReasonCode.THREE_WAY_AGREE.value)
        score += 0.08
    elif (
        sum(
            1
            for r in (
                ReasonCode.IS_APPROACHING_TRUE.value,
                ReasonCode.FOLLOW_TRAJECTORY_CONSISTENT.value,
                ReasonCode.GTFSRT_AGREE.value,
            )
            if r in reasons
        )
        >= 2
    ):
        reasons.append(ReasonCode.TWO_WAY_AGREE.value)
        score += 0.04

    # 7. Volatility: how much did the predicted arrival jitter across
    # the last ~5 ttarrivals snapshots for this (run, stop)?
    recent = _recent_arrivals_for_run_stop(
        conn, run_number, map_id, since_ms=current_ms - 5 * 60_000,
    ) if run_number else []
    if len(recent) >= 3:
        vals = [r["arrival_at_ms"] for r in recent if r["arrival_at_ms"] is not None]
        if len(vals) >= 3:
            q75 = quantile([v / 1000.0 for v in vals], 0.75) or 0.0
            q25 = quantile([v / 1000.0 for v in vals], 0.25) or 0.0
            vol_s = abs(q75 - q25) / 1.349
            features["prediction_volatility_s"] = vol_s
            if vol_s <= 30:
                reasons.append(ReasonCode.PREDICTION_STABLE.value)
                score += 0.04
            elif vol_s >= 90:
                reasons.append(ReasonCode.PREDICTION_VOLATILE.value)
                score -= 0.12

    score = clamp(score, 0.0, 1.0)
    display = _display_state(score, abstain)
    quality = _data_quality(score, reasons, abstain)

    predicted_arrival_ms = pred.get("arrival_at_ms") if not abstain else None
    if predicted_arrival_ms is not None and follow_arrival_ms is not None:
        # Modest pull toward ttfollow when both are present.
        predicted_arrival_ms = int(0.7 * predicted_arrival_ms + 0.3 * follow_arrival_ms)
    interval80 = interval90 = interval95 = (None, None)
    if predicted_arrival_ms is not None:
        horizon_s = max(0.0, (predicted_arrival_ms - current_ms) / 1000.0)
        quality_bin = "high" if score >= 0.75 else "medium" if score >= 0.55 else "low"
        cal = _calibration_quantiles(conn, line, map_id, direction_code, horizon_s, quality_bin)
        if cal:
            interval80 = (
                int(predicted_arrival_ms + cal.get("q10_s", -60.0) * 1000),
                int(predicted_arrival_ms + cal.get("q90_s", 60.0) * 1000),
            )
            interval90 = (
                int(predicted_arrival_ms + cal.get("q05_s", -90.0) * 1000),
                int(predicted_arrival_ms + cal.get("q95_s", 90.0) * 1000),
            )
            mid = predicted_arrival_ms
            lo90, hi90 = interval90
            interval95 = (int(mid - 1.25 * (mid - lo90)), int(mid + 1.25 * (hi90 - mid)))
            features["calibration_source"] = "empirical"
        else:
            base = 25.0 + 0.10 * horizon_s
            if pred_age_s is not None:
                base += max(0.0, pred_age_s - 30.0) * 0.25
            if position_age_s is not None:
                base += max(0.0, position_age_s - 30.0) * 0.20
            base *= 1.0 + max(0.0, 0.75 - score)
            base = clamp(base, 20.0, 480.0)
            interval80 = (int(predicted_arrival_ms - base * 1000), int(predicted_arrival_ms + base * 1000))
            interval90 = (int(predicted_arrival_ms - 1.55 * base * 1000), int(predicted_arrival_ms + 1.55 * base * 1000))
            interval95 = (int(predicted_arrival_ms - 2.05 * base * 1000), int(predicted_arrival_ms + 2.05 * base * 1000))
            features["calibration_source"] = "fallback_rule"

    if abstain:
        rider_message = "No reliable arrival estimate from the available train data."
    elif predicted_arrival_ms is not None:
        minutes = max(0, round((predicted_arrival_ms - current_ms) / 60_000))
        rider_message = f"Estimated arrival in about {minutes} minute{'s' if minutes != 1 else ''}."
        if display in (DisplayState.LOW_CONFIDENCE, DisplayState.UNRELIABLE):
            rider_message += " Evidence is weak; verify before relying on it."
    else:
        rider_message = "No reliable arrival estimate from the available train data."

    return EstimateResult(
        generated_at_ms=current_ms,
        map_id=int(map_id) if str(map_id).isdigit() else 0,
        line=line,
        direction_code=direction_code,
        run_number=run_number,
        destination=pred.get("destination_name"),
        predicted_arrival_ms=predicted_arrival_ms,
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


def estimate_station(
    conn: duckdb.DuckDBPyConnection,
    map_id: str,
    *,
    line: Optional[str] = None,
    direction_code: Optional[str] = None,
    top: int = 4,
    now_ms_override: Optional[int] = None,
) -> list[EstimateResult]:
    rows = _latest_arrival_rows(conn, str(map_id), line, direction_code)
    current_ms = now_ms_override or _latest_server_time_ms(conn)
    estimates = [estimate_arrival(conn, r, now_ms_override=current_ms) for r in rows]
    estimates.sort(
        key=lambda e: (
            e.predicted_arrival_ms is None,
            e.predicted_arrival_ms if e.predicted_arrival_ms is not None else 2**63 - 1,
            -e.reliability,
        )
    )
    return estimates[:top]
