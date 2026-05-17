"""Empirical residual-quantile calibration for the train-telemetry predictor.

Reads resolved ``forecast_outcomes`` for the ``train-telemetry-v1``
predictor, groups by ``(line, map_id, direction_code, horizon_bin,
quality_bin)``, and writes the q05/q10/q25/q50/q75/q90/q95 + mae + bias
to ``train_v2_residual_quantile``. The estimator consults this table
when an empirical cell has at least ``min_n`` samples; otherwise it
falls back to its rule-based interval widening.

Mirrors ``bus_v3.calibration`` end-to-end.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import duckdb
import structlog

from .util import horizon_bin, now_ms, quantile


log = structlog.get_logger(__name__)


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
    state = str(features.get("data_quality") or "").upper()
    if state == "GOOD":
        return "high"
    if state == "ACCEPTABLE":
        return "medium"
    if state in ("DEGRADED", "STALE", "CONTRADICTORY"):
        return "low"
    return "any"


def refresh_train_residual_quantiles(
    conn: duckdb.DuckDBPyConnection,
    *,
    predictor_version: str = "train-telemetry-v1",
    min_n: int = 20,
    truth_confidence_floor: float = 0.5,
) -> int:
    """Recompute the empirical residual quantile table for trains.

    Returns the number of rows inserted into ``train_v2_residual_quantile``.
    """
    rows = conn.execute(
        """
        SELECT
            fq.line                AS line,
            fq.boarding_map_id     AS map_id,
            fq.direction_code      AS direction_code,
            fq.predicted_wait_p50  AS predicted_wait_p50,
            fo.actual_wait_seconds AS actual_wait_seconds,
            fq.feature_json        AS feature_json
        FROM forecast_outcomes fo
        JOIN forecast_queue fq ON fq.forecast_id = fo.forecast_id
        WHERE fq.mode = 'L'
          AND fq.predictor_version = ?
          AND fo.failed IS NOT TRUE
          AND fo.actual_wait_seconds IS NOT NULL
          AND fq.predicted_wait_p50 IS NOT NULL
          AND COALESCE(fo.truth_confidence, 0.0) >= ?
        """,
        [predictor_version, truth_confidence_floor],
    ).fetchall()

    if not rows:
        log.info("train_v2.calibration.no_outcomes")
        return 0

    cells: dict[tuple[Optional[str], Optional[str], Optional[str], str, str], list[float]] = {}
    for line, map_id, dir_code, predicted, actual, feature_json_raw in rows:
        features: dict[str, Any] = {}
        if feature_json_raw:
            try:
                features = json.loads(feature_json_raw) or {}
            except (TypeError, ValueError):
                features = {}
        residual_s = float(actual) - float(predicted)
        hbin = horizon_bin(float(predicted))
        qbin = _quality_bin(features)
        key = (
            str(line) if line else None,
            str(map_id) if map_id else None,
            str(dir_code) if dir_code else None,
            hbin,
            qbin,
        )
        cells.setdefault(key, []).append(residual_s)

    created_at = now_ms()
    inserted = 0
    for (line, map_id, dir_code, hbin, qbin), residuals in cells.items():
        if len(residuals) < min_n:
            continue
        quantiles = {key: quantile(residuals, q) for q, key in _QUANTILE_KEYS.items()}
        mae = sum(abs(r) for r in residuals) / len(residuals)
        bias = sum(residuals) / len(residuals)
        conn.execute(
            """
            INSERT INTO train_v2_residual_quantile(
                created_at_ms, line, map_id, direction_code, horizon_bin, quality_bin, n,
                q05_s, q10_s, q25_s, q50_s, q75_s, q90_s, q95_s, mae_s, bias_s
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                created_at, line, map_id, dir_code, hbin, qbin, len(residuals),
                quantiles.get("q05_s"), quantiles.get("q10_s"), quantiles.get("q25_s"),
                quantiles.get("q50_s"), quantiles.get("q75_s"),
                quantiles.get("q90_s"), quantiles.get("q95_s"),
                mae, bias,
            ],
        )
        inserted += 1
    log.info("train_v2.calibration.refreshed", inserted=inserted, total_cells=len(cells))
    return inserted
