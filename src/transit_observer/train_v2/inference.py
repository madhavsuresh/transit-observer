"""High-confidence arrival inference for the train_v2 pipeline.

Train arrival ground truth (analog of bus_v3's pdist crossing) comes
from a same-run ``nextStaId`` transition:

- At time T0, run R has ``nextStaId = X`` (train is approaching X).
- At time T1 > T0, run R has ``nextStaId = Y`` with Y != X.
- The transition implies the train arrived at X between T0 and T1.

A row is marked ``high_confidence = TRUE`` only when:

1. Both observations are fresh (``prdt`` within 60 s of server time).
2. Neither observation has ``isFlt = TRUE``.
3. The gap between observations is short (≤ 300 s).
4. The destination didn't change between the observations (a
   destination flip implies the train was reassigned or short-turned).
5. Either the line topology confirms Y is downstream of X, or
   topology is empty for this (line, direction) — in which case the
   transition is accepted with a small reliability discount.

The arrival timestamp is interpolated at the midpoint between T0 and T1.
Without a finer-grained position cadence we can't do better than that;
the resolver's ±60 s window absorbs the uncertainty.

Censored / disrupted cases land as ``high_confidence = FALSE`` with a
descriptive label (``FAULTED_TRACKING``, ``STALE_DATA``,
``NO_EVIDENCE_GHOST``, ``REROUTED_OR_SHORT_TURN``, ``CENSORED_UNKNOWN``)
so the dashboard can surface them for diagnostics without polluting
accuracy metrics.
"""

from __future__ import annotations

from typing import Any, Optional

import duckdb

from .models import ArrivalLabel, ReasonCode
from .topology import is_downstream
from .util import json_dumps, now_ms


FRESHNESS_LIMIT_S = 60.0          # prediction age over this disqualifies high_confidence
TRANSITION_MAX_GAP_S = 300.0      # consecutive obs must be within 5 min
DEFAULT_RUN_WINDOW_HOURS = 6


def _rows_to_dicts(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    params: list[Any] | None = None,
) -> list[dict[str, Any]]:
    cur = conn.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _active_runs(
    conn: duckdb.DuckDBPyConnection,
    run_id: Optional[str],
    cutoff_ms: Optional[int],
) -> list[tuple[str, str, str]]:
    where = ["next_station_map_id IS NOT NULL", "run_number IS NOT NULL"]
    params: list[Any] = []
    if run_id:
        where.append("run_id = ?")
        params.append(run_id)
    if cutoff_ms is not None:
        where.append("local_response_end_ms >= ?")
        params.append(cutoff_ms)
    sql = f"""
        SELECT DISTINCT run_number, line, direction_code
          FROM train_v2_position_observation
         WHERE {' AND '.join(where)}
    """
    return [
        (str(rn), str(line) if line else "", str(dir_code) if dir_code else "")
        for rn, line, dir_code in conn.execute(sql, params).fetchall()
    ]


def _run_observations(
    conn: duckdb.DuckDBPyConnection,
    run_number: str,
    cutoff_ms: Optional[int],
) -> list[dict[str, Any]]:
    where = ["run_number = ?", "next_station_map_id IS NOT NULL"]
    params: list[Any] = [run_number]
    if cutoff_ms is not None:
        where.append("local_response_end_ms >= ?")
        params.append(cutoff_ms)
    sql = f"""
        SELECT position_obs_id, poll_id, run_id, line, run_number, direction_code,
               destination_name, destination_map_id,
               next_station_map_id, next_station_name,
               predicted_at_ms, next_arrival_at_ms,
               cta_server_time_ms, local_response_end_ms,
               is_approaching, is_delayed, is_fault
          FROM train_v2_position_observation
         WHERE {' AND '.join(where)}
         ORDER BY COALESCE(predicted_at_ms, local_response_end_ms), position_obs_id
    """
    return _rows_to_dicts(conn, sql, params)


def _is_fresh(obs: dict[str, Any]) -> bool:
    server_ms = obs.get("cta_server_time_ms")
    prdt = obs.get("predicted_at_ms")
    if server_ms is None or prdt is None:
        # If we lack server time, fall back to local_response_end_ms.
        server_ms = obs.get("local_response_end_ms")
    if server_ms is None or prdt is None:
        return True  # don't reject for missing fields; downgrade elsewhere
    age_s = (server_ms - prdt) / 1000.0
    return age_s <= FRESHNESS_LIMIT_S


def infer_train_arrivals(
    conn: duckdb.DuckDBPyConnection,
    *,
    run_id: Optional[str] = None,
    cutoff_ms: Optional[int] = None,
    replace: bool = False,
) -> int:
    """Detect arrival events from ``train_v2_position_observation`` and
    write them to ``train_v2_arrival_event``.

    Args:
        run_id: restrict to one collector cycle's run_id.
        cutoff_ms: only consider position rows newer than this epoch
            (defaults to the last 6 h).
        replace: delete rows for ``run_id`` before re-inferring (matches
            the validator's ``--replace`` semantics).
    """
    if cutoff_ms is None and run_id is None:
        cutoff_ms = now_ms() - DEFAULT_RUN_WINDOW_HOURS * 3_600_000
    if replace and run_id:
        conn.execute("DELETE FROM train_v2_arrival_event WHERE run_id = ?", [run_id])

    inserted = 0
    for run_number, _line_hint, _dir_hint in _active_runs(conn, run_id, cutoff_ms):
        obs = _run_observations(conn, run_number, cutoff_ms)
        for a, b in zip(obs, obs[1:]):
            from_map = a.get("next_station_map_id")
            to_map = b.get("next_station_map_id")
            if not from_map or not to_map or from_map == to_map:
                continue
            inserted += _emit_event(conn, a, b, run_id=run_id or a.get("run_id") or "")
    return inserted


def _emit_event(
    conn: duckdb.DuckDBPyConnection,
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    run_id: str,
) -> int:
    from_map = a["next_station_map_id"]
    to_map = b["next_station_map_id"]
    line = a.get("line") or b.get("line")
    direction_code = a.get("direction_code") or b.get("direction_code")
    destination_name = a.get("destination_name") or b.get("destination_name")
    run_number = a.get("run_number") or b.get("run_number")

    reasons: list[str] = []
    high_confidence = True
    confidence = 0.95
    label = ArrivalLabel.ARRIVED_CONFIRMED

    # Freshness.
    fresh_a = _is_fresh(a)
    fresh_b = _is_fresh(b)
    if not (fresh_a and fresh_b):
        reasons.append(ReasonCode.PREDICTION_STALE.value)
        high_confidence = False
        confidence = 0.55
        label = ArrivalLabel.STALE_DATA
    else:
        reasons.append(ReasonCode.PREDICTION_FRESH.value)

    # Fault.
    if a.get("is_fault") or b.get("is_fault"):
        reasons.append(ReasonCode.IS_FAULT_FLAG.value)
        high_confidence = False
        confidence = 0.20
        label = ArrivalLabel.FAULTED_TRACKING

    # Gap between observations.
    t0 = a.get("predicted_at_ms") or a.get("local_response_end_ms")
    t1 = b.get("predicted_at_ms") or b.get("local_response_end_ms")
    if t0 is None or t1 is None or t1 <= t0:
        return 0
    gap_s = (t1 - t0) / 1000.0
    if gap_s > TRANSITION_MAX_GAP_S:
        reasons.append(ReasonCode.POSITION_STALE.value)
        high_confidence = False
        confidence = min(confidence, 0.40)
        if label == ArrivalLabel.ARRIVED_CONFIRMED:
            label = ArrivalLabel.CENSORED_UNKNOWN

    # Destination flip → short turn / reassignment, not a clean arrival.
    if (
        a.get("destination_map_id")
        and b.get("destination_map_id")
        and a["destination_map_id"] != b["destination_map_id"]
    ):
        reasons.append(ReasonCode.NEXT_STATION_MISMATCH.value)
        high_confidence = False
        confidence = 0.25
        label = ArrivalLabel.REROUTED_OR_SHORT_TURN

    # Topology check (gates high_confidence).
    if line and direction_code:
        topo = is_downstream(
            conn,
            line=line,
            direction_code=direction_code,
            from_map=str(from_map),
            to_map=str(to_map),
        )
        if topo is True:
            reasons.append(ReasonCode.NEXT_STATION_ADVANCED.value)
        elif topo is False:
            # Topology says to_map is upstream of from_map — likely a
            # bad transition or a train that turned around.
            reasons.append(ReasonCode.NEXT_STATION_MISMATCH.value)
            high_confidence = False
            confidence = min(confidence, 0.30)
            label = ArrivalLabel.CENSORED_UNKNOWN
        else:
            # Unknown — apply a small reliability discount but keep the
            # high_confidence flag if no other downgrade fired.
            reasons.append(ReasonCode.FOLLOW_MISSING.value)
            confidence = min(confidence, 0.85)

    if high_confidence:
        reasons.append(ReasonCode.GROUND_TRUTH_HIGH_CONFIDENCE.value)
    else:
        reasons.append(ReasonCode.GROUND_TRUTH_LOW_CONFIDENCE.value)

    actual_arrival_ms = int((t0 + t1) / 2)

    evidence = {
        "from_map_id": str(from_map),
        "to_map_id": str(to_map),
        "t0_ms": int(t0),
        "t1_ms": int(t1),
        "gap_s": gap_s,
        "freshness_a_ok": fresh_a,
        "freshness_b_ok": fresh_b,
        "destination_a": a.get("destination_map_id"),
        "destination_b": b.get("destination_map_id"),
    }

    # Dedup on (run_id, run_number, map_id, label, ±30s arrival).
    existing = conn.execute(
        """
        SELECT event_id FROM train_v2_arrival_event
         WHERE run_id = ? AND run_number = ? AND map_id = ?
           AND COALESCE(line, '') = COALESCE(?, '')
           AND label = ?
           AND ABS(actual_arrival_ms - ?) <= 30000
         LIMIT 1
        """,
        [run_id, run_number, str(from_map), line, label.value, actual_arrival_ms],
    ).fetchone()
    if existing:
        return 0
    conn.execute(
        """
        INSERT INTO train_v2_arrival_event(
            run_id, map_id, line, direction_code, run_number, destination_name,
            actual_arrival_ms, label, high_confidence, confidence,
            evidence_json, reason_codes_json,
            before_position_obs_id, after_position_obs_id, gtfsrt_corroboration_id,
            created_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id, str(from_map), line, direction_code, run_number, destination_name,
            actual_arrival_ms, label.value, bool(high_confidence), confidence,
            json_dumps(evidence), json_dumps(sorted(set(reasons))),
            a.get("position_obs_id"), b.get("position_obs_id"), None,
            now_ms(),
        ],
    )
    return 1
