"""LightGBM training for the residual quantile predictor.

Public API:

- ``fit_quantile_gbm`` — train all (leg × line × quantile) boosters for
  one predictor_version. Returns per-booster validation metrics.
- ``warmup_dtaci`` — initialize DtACI offsets from the validation
  residuals so the conformal layer doesn't start at zero.

Heavy imports (lightgbm, pandas, numpy) live inside the functions so
the base collector keeps a slim dependency footprint.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import structlog

from ..predictors.diagnostics import (
    crps_from_quantiles,
    pinball_loss,
)
from ..predictors.features import L_FEATURE_NAMES
from ..predictors.quantile_gbm import GBM_QUANTILES, GBM_VERSION

from . import artifacts as art
from .dataset import TrainingFrame, time_based_split, to_pandas


log = structlog.get_logger(__name__)


# Hyperparameters chosen for the data scale (10k-100k rows). Conservative
# tree depth + bagging keeps the model honest at the lower end; LightGBM
# scales these gracefully upward.
#
# ``num_threads`` is pinned to all available cores instead of LightGBM's
# default (which can fall back to 1 on some shells). On Apple Silicon
# this routes through Accelerate / AMX for the heavy matrix ops; on x86
# it goes through OpenMP. LightGBM doesn't currently expose Apple GPU
# directly (no Metal backend; OpenCL is x86-only for macOS wheels), so
# we leave ``device_type`` at "cpu" — the AMX-backed CPU path is faster
# than GPU at our data scale anyway.
LGBM_NUM_THREADS = int(os.environ.get("LIGHTGBM_NUM_THREADS") or os.cpu_count() or 4)


LGBM_PARAMS_BASE: dict[str, Any] = {
    "objective": "quantile",
    "metric": "quantile",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "lambda_l2": 0.1,
    "verbose": -1,
    "num_threads": LGBM_NUM_THREADS,
    "device_type": "cpu",
    "deterministic": False,   # allow non-deterministic multi-threaded ops
    "force_col_wise": True,   # better cache behavior on small-medium frames
}


CATEGORICAL_FEATURES: tuple[str, ...] = (
    "line", "direction_code", "boarding_map_id", "alighting_map_id",
    "weekday_or_weekend", "mode",
)


@dataclass(frozen=True)
class BoosterMetrics:
    leg: str
    line: str
    quantile: float
    n_train: int
    n_val: int
    val_pinball: float
    val_crps: float                # CRPS computed across all three trained quantiles
    artifact_path: Path


@dataclass(frozen=True)
class FitReport:
    predictor_version: str
    boosters: list[BoosterMetrics]
    rows_total: int
    rows_train: int
    rows_val: int


def fit_quantile_gbm(
    conn: duckdb.DuckDBPyConnection,
    frame: TrainingFrame,
    *,
    predictor_version: str = GBM_VERSION,
    quantiles: tuple[float, ...] = GBM_QUANTILES,
    num_boost_round: int = 400,
    early_stopping_rounds: int = 30,
    per_line: bool = True,
    now: datetime | None = None,
) -> FitReport:
    """Train one residual-quantile model per (leg, line, quantile).

    With ``per_line=True`` each line gets its own boosters (richer
    structure per line, more models on disk). ``per_line=False`` trains
    a single global model with line as a categorical (cheaper, smaller).
    """
    try:
        import lightgbm as lgb  # type: ignore
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore  # noqa: F401  -- needed for to_pandas
    except ImportError as e:
        raise ImportError(
            "lightgbm + numpy required for training. "
            "uv sync --group learned"
        ) from e

    if now is None:
        now = datetime.now()

    train_frame, val_frame = time_based_split(frame, val_fraction=0.15, min_val_rows=200)
    n_train = len(train_frame)
    n_val = len(val_frame)
    if n_train < 200:
        raise RuntimeError(
            f"Too few training rows ({n_train}) — need ≥200. "
            "Wait for more outcomes to resolve."
        )

    df_train = to_pandas(train_frame)
    df_val = to_pandas(val_frame) if n_val > 0 else None

    lines = sorted(df_train["line"].dropna().unique().tolist()) if per_line else ["ALL"]
    if not lines:
        lines = ["ALL"]

    booster_metrics: list[BoosterMetrics] = []

    for leg in ("wait", "in_vehicle"):
        label_col = "residual_wait_s" if leg == "wait" else "residual_in_vehicle_s"
        if label_col not in df_train.columns:
            log.warning("fit.skip_leg", leg=leg, reason="label_missing")
            continue
        per_line_loop = lines if per_line else ["ALL"]
        for line in per_line_loop:
            if per_line and line != "ALL":
                df_tr = df_train[df_train["line"] == line]
                df_va = df_val[df_val["line"] == line] if df_val is not None else None
            else:
                df_tr = df_train
                df_va = df_val

            # Drop rows with missing label
            df_tr = df_tr.dropna(subset=[label_col])
            if df_va is not None:
                df_va = df_va.dropna(subset=[label_col])

            if len(df_tr) < 50:
                log.info(
                    "fit.skip_per_line", leg=leg, line=line,
                    reason="too_few_rows", n=len(df_tr),
                )
                continue

            X_tr = _prepare_X(df_tr)
            y_tr = df_tr[label_col].to_numpy()
            w_tr = df_tr["truth_confidence"].fillna(0.5).to_numpy()

            X_va = _prepare_X(df_va) if (df_va is not None and len(df_va) > 0) else None
            y_va = df_va[label_col].to_numpy() if X_va is not None else None
            w_va = df_va["truth_confidence"].fillna(0.5).to_numpy() if X_va is not None else None

            cat_features = [c for c in CATEGORICAL_FEATURES if c in X_tr.columns]

            per_q_artifact_paths: dict[float, Path] = {}
            per_q_val_pinball: dict[float, float] = {}
            per_q_val_preds: dict[float, Any] = {}
            for q in quantiles:
                params = {**LGBM_PARAMS_BASE, "alpha": q}
                train_ds = lgb.Dataset(
                    X_tr, label=y_tr, weight=w_tr,
                    categorical_feature=cat_features,
                    free_raw_data=False,
                )
                valid_sets = [train_ds]
                valid_names = ["train"]
                if X_va is not None and len(X_va) > 0:
                    val_ds = lgb.Dataset(
                        X_va, label=y_va, weight=w_va,
                        categorical_feature=cat_features,
                        reference=train_ds,
                        free_raw_data=False,
                    )
                    valid_sets.append(val_ds)
                    valid_names.append("val")

                callbacks = []
                if X_va is not None:
                    callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
                callbacks.append(lgb.log_evaluation(period=0))

                booster = lgb.train(
                    params,
                    train_ds,
                    num_boost_round=num_boost_round,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    callbacks=callbacks,
                )

                val_pinball = math.nan
                if X_va is not None and len(X_va) > 0:
                    preds = booster.predict(X_va)
                    # Vectorized pinball: alpha * err+ + (1-alpha) * err-
                    err = y_va - preds
                    val_pinball = float(np.mean(np.maximum(q * err, (q - 1.0) * err)))
                    per_q_val_preds[q] = preds

                feature_cols = list(X_tr.columns)
                path = art.save_booster(
                    booster,
                    predictor_version=predictor_version,
                    leg=leg, line=line, quantile=q,
                    feature_columns=feature_cols,
                )
                per_q_artifact_paths[q] = path
                per_q_val_pinball[q] = val_pinball

            # Vectorized CRPS across the three trained quantiles on the val
            # set — single BLAS-backed reduction instead of a per-row Python
            # loop. Uses the predictions we already computed for pinball.
            val_crps = math.nan
            if X_va is not None and len(X_va) > 0 and per_q_val_preds:
                from ..predictors.diagnostics import crps_from_quantiles_batch

                sorted_q = sorted(per_q_val_preds.keys())
                qvals = np.stack([per_q_val_preds[q] for q in sorted_q], axis=1)
                crps_arr = crps_from_quantiles_batch(
                    quantile_levels=np.asarray(sorted_q, dtype=np.float64),
                    quantile_values=qvals,
                    actuals=np.asarray(y_va, dtype=np.float64),
                )
                finite_crps = crps_arr[np.isfinite(crps_arr)]
                val_crps = float(finite_crps.mean()) if finite_crps.size else math.nan

            # Register all three quantiles for this (leg, line)
            for q, path in per_q_artifact_paths.items():
                art.register_artifact(
                    conn,
                    predictor_version=predictor_version,
                    leg=leg, line=line, quantile=q,
                    artifact_path=path,
                    n_train_rows=int(len(df_tr)),
                    n_val_rows=int(len(df_va)) if df_va is not None else 0,
                    val_pinball_loss=per_q_val_pinball.get(q, math.nan),
                    val_crps=val_crps,
                    feature_columns=feature_cols,
                    now=now,
                )
                booster_metrics.append(BoosterMetrics(
                    leg=leg, line=line, quantile=q,
                    n_train=int(len(df_tr)),
                    n_val=int(len(df_va)) if df_va is not None else 0,
                    val_pinball=per_q_val_pinball.get(q, math.nan),
                    val_crps=val_crps,
                    artifact_path=path,
                ))

    return FitReport(
        predictor_version=predictor_version,
        boosters=booster_metrics,
        rows_total=len(frame),
        rows_train=n_train,
        rows_val=n_val,
    )


def _prepare_X(df):
    """Project a DataFrame down to the model's feature columns + cast types.

    LightGBM accepts pandas categoricals natively, so we coerce string
    cols to ``category`` dtype.
    """
    import pandas as pd  # type: ignore
    keep = [c for c in L_FEATURE_NAMES if c in df.columns]
    X = df[keep].copy()
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def warmup_dtaci(
    conn: duckdb.DuckDBPyConnection,
    *,
    predictor_version: str = GBM_VERSION,
    n_warmup_rows: int = 500,
    since: datetime | None = None,
) -> dict[str, int]:
    """Replay recent resolved outcomes through DtACI to seed the offsets.

    Useful after a fresh training run: the conformal layer otherwise
    starts at offset=0 and takes hundreds of observations to settle.
    This replays the most recent ``n_warmup_rows`` outcomes per (line,
    direction) and writes the resulting offsets back to
    ``predictor_state``.

    Returns a {state_key: n_updates_applied} dict for the operator.
    """
    from ..predictors import conformal
    from datetime import datetime as _dt

    sql = """
        SELECT q.line, q.direction_code,
               q.predicted_wait_p80, q.predicted_wait_p90,
               q.predicted_total_p80, q.predicted_total_p90,
               o.actual_wait_seconds, o.actual_total_seconds, o.resolved_at
          FROM forecast_queue q
          JOIN forecast_outcomes o USING (forecast_id)
         WHERE q.mode = 'L' AND q.status = 'resolved'
           AND o.actual_wait_seconds IS NOT NULL
           AND COALESCE(o.truth_confidence, 0) >= 0.5
    """
    params: list[Any] = []
    if since is not None:
        sql += " AND o.resolved_at >= ?"
        params.append(since)
    sql += " ORDER BY o.resolved_at DESC LIMIT ?"
    params.append(n_warmup_rows)
    rows = conn.execute(sql, params).fetchall()
    now = _dt.now()
    n_updates = 0
    for line, direction, p80_w, p90_w, p80_t, p90_t, actual_w, actual_t, _resolved_at in rows:
        line = line or ""
        direction = direction or ""
        if actual_w is not None and p80_w is not None:
            conformal.update(
                conn,
                predictor_version=predictor_version,
                line=line, direction_code=direction,
                leg="wait", quantile=0.8,
                raw_quantile_seconds=float(p80_w),
                observed_seconds=float(actual_w),
                now=now,
            )
            n_updates += 1
        if actual_w is not None and p90_w is not None:
            conformal.update(
                conn,
                predictor_version=predictor_version,
                line=line, direction_code=direction,
                leg="wait", quantile=0.9,
                raw_quantile_seconds=float(p90_w),
                observed_seconds=float(actual_w),
                now=now,
            )
            n_updates += 1
        if actual_t is not None and p90_t is not None:
            conformal.update(
                conn,
                predictor_version=predictor_version,
                line=line, direction_code=direction,
                leg="total", quantile=0.9,
                raw_quantile_seconds=float(p90_t),
                observed_seconds=float(actual_t),
                now=now,
            )
            n_updates += 1
    return {"n_warmup_updates": n_updates, "n_warmup_rows": len(rows)}
