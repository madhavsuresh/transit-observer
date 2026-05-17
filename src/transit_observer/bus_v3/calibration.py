"""Empirical residual-quantile calibration for the bus-telemetry predictor.

Reads resolved ``forecast_outcomes`` for the ``bus-telemetry-v1`` version,
groups by ``(rt, stpid, rtdir, horizon_bin, quality_bin)``, and writes
the q05/q10/q25/q50/q75/q90/q95 + mae + bias to
``bus_v3_residual_quantile``. Cells with fewer than ``min_n`` samples are
skipped; the estimator's lookup walks a fallback ladder (full stratum →
drop stpid → drop rtdir → drop quality_bin → global per horizon).

The runtime calibration sits on top of this: the existing online ACI in
``predictors.conformal`` keeps adjusting ``offset_seconds`` in
``predictor_state``. ``bus_v3_residual_quantile`` is the per-stratum
prior consulted by the estimator when no ACI offset is mature yet.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import duckdb
import structlog

from .util import horizon_bin, now_ms, quantile


log = structlog.get_logger(__name__)


_QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
_QUANTILE_KEYS: dict[float, str] = {
    0.05: "q05_s",
    0.10: "q10_s",
    0.25: "q25_s",
    0.50: "q50_s",
    0.75: "q75_s",
    0.90: "q90_s",
    0.95: "q95_s",
}


def _quality_bin(features: dict[str, Any]) -> str:
    state = features.get("data_quality") or ""
    state = str(state).upper()
    if state == "GOOD":
        return "high"
    if state == "ACCEPTABLE":
        return "medium"
    if state in ("DEGRADED", "STALE", "CONTRADICTORY"):
        return "low"
    return "any"


def refresh_bus_residual_quantiles(
    conn: duckdb.DuckDBPyConnection,
    *,
    predictor_version: str = "bus-telemetry-v1",
    min_n: int = 20,
    truth_confidence_floor: float = 0.5,
) -> int:
    """Recompute the empirical residual quantile table.

    Args:
        predictor_version: which forecast_queue rows to consider.
        min_n: minimum sample count per cell; sparser cells are skipped.
        truth_confidence_floor: lower bound on ``forecast_outcomes.truth_confidence``
            (default 0.5; the v3 pdist-crossing path saturates this to 1.0).

    Returns: number of rows inserted into ``bus_v3_residual_quantile``.
    """
    rows = conn.execute(
        """
        SELECT
            fq.line               AS rt,
            fq.boarding_text_id   AS stpid,
            fq.direction_code     AS rtdir,
            fq.predicted_wait_p50 AS predicted_wait_p50,
            fo.actual_wait_seconds AS actual_wait_seconds,
            fq.feature_json       AS feature_json
        FROM forecast_outcomes fo
        JOIN forecast_queue fq ON fq.forecast_id = fo.forecast_id
        WHERE fq.mode = 'bus'
          AND fq.predictor_version = ?
          AND fo.failed IS NOT TRUE
          AND fo.actual_wait_seconds IS NOT NULL
          AND fq.predicted_wait_p50 IS NOT NULL
          AND COALESCE(fo.truth_confidence, 0.0) >= ?
        """,
        [predictor_version, truth_confidence_floor],
    ).fetchall()

    if not rows:
        log.info("bus_v3.calibration.no_outcomes")
        return 0

    # Aggregate residuals by (rt, stpid, rtdir, horizon_bin, quality_bin).
    cells: dict[tuple[str, Optional[str], Optional[str], str, str], list[float]] = {}
    for rt, stpid, rtdir, predicted, actual, feature_json_raw in rows:
        features: dict[str, Any] = {}
        if feature_json_raw:
            try:
                features = json.loads(feature_json_raw) or {}
            except (TypeError, ValueError):
                features = {}
        residual_s = float(actual) - float(predicted)
        hbin = horizon_bin(float(predicted))
        qbin = _quality_bin(features)
        key = (str(rt), str(stpid) if stpid else None, str(rtdir) if rtdir else None, hbin, qbin)
        cells.setdefault(key, []).append(residual_s)

    created_at = now_ms()
    inserted = 0
    for (rt, stpid, rtdir, hbin, qbin), residuals in cells.items():
        if len(residuals) < min_n:
            continue
        quantiles = {key: quantile(residuals, q) for q, key in _QUANTILE_KEYS.items()}
        mae = sum(abs(r) for r in residuals) / len(residuals)
        bias = sum(residuals) / len(residuals)
        conn.execute(
            """
            INSERT INTO bus_v3_residual_quantile(
                created_at_ms, rt, stpid, rtdir, horizon_bin, quality_bin, n,
                q05_s, q10_s, q25_s, q50_s, q75_s, q90_s, q95_s, mae_s, bias_s
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                created_at, rt, stpid, rtdir, hbin, qbin, len(residuals),
                quantiles.get("q05_s"), quantiles.get("q10_s"), quantiles.get("q25_s"),
                quantiles.get("q50_s"), quantiles.get("q75_s"),
                quantiles.get("q90_s"), quantiles.get("q95_s"),
                mae, bias,
            ],
        )
        inserted += 1
    log.info("bus_v3.calibration.refreshed", inserted=inserted, total_cells=len(cells))
    return inserted
