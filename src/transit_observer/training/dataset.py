"""Assemble a training frame from forecast_queue, forecast_outcomes, and
raw tables. Reads the read replica; never writes to the writable DB.

Returns one row per resolved L forecast for which:
  - status = 'resolved'
  - truth_confidence >= ``min_truth_confidence``
  - predictor_version was one of the kernel variants (we bootstrap the
    GBM from kernel-resolved outcomes; once gbm-v1 is live, additional
    outcomes get scored under gbm-v1 too but we still train on the
    kernel's outcomes to avoid a self-reinforcement loop).

The features mirror :mod:`predictors.features` exactly. Drift between
the two is how learned predictors silently degrade — keep them in sync.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator

import duckdb

from ..predictors.features import L_FEATURE_NAMES

log = logging.getLogger(__name__)


# Same kernel-output baseline column we want the model to predict residual on.
KERNEL_BASELINE_COLUMNS: tuple[str, ...] = (
    "kernel_wait_p50", "kernel_wait_p80", "kernel_wait_p90",
    "kernel_in_vehicle_mean",
)


# When set to True, the polars path is used; falls back to pandas if
# polars is missing.
USE_POLARS: bool = True


@dataclass(frozen=True)
class TrainingFrame:
    """Output of build_training_frame_l. Plain dict-of-lists keeps the
    module pandas/polars-agnostic at the boundary."""

    rows: list[dict[str, Any]]
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...] = ("residual_wait_s", "residual_in_vehicle_s")

    def __len__(self) -> int:
        return len(self.rows)


LABEL_TARGETS = ("residual_wait_s", "residual_in_vehicle_s")


_BASE_SQL = """
WITH base AS (
  SELECT
      q.forecast_id,
      q.snapshot_polled_at,
      q.leave_at,
      q.mode,
      q.line,
      q.direction_code,
      q.boarding_map_id,
      q.alighting_map_id,
      q.feature_json,
      q.predicted_wait_p50      AS kernel_wait_p50,
      q.predicted_wait_p80      AS kernel_wait_p80,
      q.predicted_wait_p90      AS kernel_wait_p90,
      q.predicted_in_vehicle_mean AS kernel_in_vehicle_mean,
      o.actual_wait_seconds,
      o.actual_in_vehicle_seconds,
      o.actual_total_seconds,
      o.truth_confidence,
      o.boarded_run_number,
      o.boarded_at
    FROM forecast_queue q
    JOIN forecast_outcomes o USING (forecast_id)
   WHERE q.status = 'resolved'
     AND q.mode = 'L'
     AND COALESCE(o.truth_confidence, 0) >= ?
     AND q.leave_at >= ?
     AND q.leave_at < ?
     AND q.predictor_version IN ('kernel-v1', 'kernel-v1+eb')
     AND o.actual_wait_seconds IS NOT NULL
)
SELECT * FROM base
"""


def build_training_frame_l(
    conn: duckdb.DuckDBPyConnection,
    *,
    since: datetime,
    until: datetime,
    min_truth_confidence: float = 0.5,
    min_horizon_seconds: float = 90.0,
) -> TrainingFrame:
    """One row per resolved L forecast in the window.

    ``min_horizon_seconds`` drops near-arrival rows (where
    is_approaching=True collapses residuals to ~0 by construction).
    Without this filter the GBM gets dominated by easy short-horizon
    cases (see CLAUDE.md gotchas).
    """
    base_rows = conn.execute(
        _BASE_SQL,
        [min_truth_confidence, since, until],
    ).fetchall()
    columns = [d[0] for d in conn.description]

    out_rows: list[dict[str, Any]] = []
    for raw in base_rows:
        row = dict(zip(columns, raw))
        # Parse the feature_json captured at predict time
        feat: dict[str, Any] = {}
        if row.get("feature_json"):
            try:
                feat = json.loads(row["feature_json"])
            except json.JSONDecodeError:
                feat = {}

        snapshot_polled_at: datetime | None = row["snapshot_polled_at"]
        leave_at: datetime | None = row["leave_at"]
        # Horizon: seconds_until_next_arrival if we have it; else leave_at - snapshot
        horizon = feat.get("seconds_until_next_arrival")
        if (horizon is None or not _isfinite(horizon)) and (
            snapshot_polled_at is not None and leave_at is not None
        ):
            horizon = (leave_at - snapshot_polled_at).total_seconds()
        if horizon is not None and _isfinite(horizon) and horizon < min_horizon_seconds:
            continue

        kernel_wait_p50 = float(row["kernel_wait_p50"] or 0.0)
        actual_wait = float(row["actual_wait_seconds"] or 0.0)
        actual_iv = (
            float(row["actual_in_vehicle_seconds"])
            if row["actual_in_vehicle_seconds"] is not None
            else None
        )
        kernel_iv_mean = float(row["kernel_in_vehicle_mean"] or 0.0)

        residual_wait = actual_wait - kernel_wait_p50
        residual_iv = (actual_iv - kernel_iv_mean) if actual_iv is not None else None

        out_row: dict[str, Any] = {
            "forecast_id": row["forecast_id"],
            "snapshot_polled_at": snapshot_polled_at,
            "leave_at": leave_at,
            "line": feat.get("line") or row.get("line") or "",
            "direction_code": feat.get("direction_code") or row.get("direction_code") or "",
            "boarding_map_id": str(feat.get("boarding_map_id") or row["boarding_map_id"]),
            "alighting_map_id": str(feat.get("alighting_map_id") or row["alighting_map_id"]),
            "hour_of_day": int(feat.get("hour_of_day") or (leave_at.hour if leave_at else 0)),
            "weekday_or_weekend": (
                feat.get("weekday_or_weekend")
                or ("weekday" if leave_at and leave_at.weekday() < 5 else "weekend")
            ),
            "mode": "L",
            "haversine_meters": float(feat.get("haversine_meters") or 0.0),
            # Dynamic features (NaN if not in snapshot)
            **{
                name: feat.get(name) if feat.get(name) is not None else math.nan
                for name in L_FEATURE_NAMES
                if name not in {
                    "line", "direction_code", "boarding_map_id", "alighting_map_id",
                    "hour_of_day", "weekday_or_weekend", "mode", "haversine_meters",
                }
            },
            # Labels (residual targets) and meta
            "kernel_wait_p50": kernel_wait_p50,
            "kernel_in_vehicle_mean": kernel_iv_mean,
            "actual_wait_seconds": actual_wait,
            "actual_in_vehicle_seconds": actual_iv,
            "actual_total_seconds": float(row["actual_total_seconds"]) if row["actual_total_seconds"] else None,
            "residual_wait_s": residual_wait,
            "residual_in_vehicle_s": residual_iv,
            "truth_confidence": float(row["truth_confidence"]) if row["truth_confidence"] else 0.5,
            "horizon_seconds": float(horizon) if horizon is not None and _isfinite(horizon) else math.nan,
        }
        out_rows.append(out_row)

    return TrainingFrame(rows=out_rows, feature_columns=L_FEATURE_NAMES)


def time_based_split(
    frame: TrainingFrame,
    *,
    val_fraction: float = 0.1,
    min_val_rows: int = 100,
) -> tuple[TrainingFrame, TrainingFrame]:
    """Split a TrainingFrame in time order — last ``val_fraction`` is val.

    Time-based split is mandatory for sequential data; random splits leak
    future state. Falls back to skipping the val if ``min_val_rows`` is
    unreachable.
    """
    rows = sorted(frame.rows, key=lambda r: r["leave_at"] or datetime.min)
    n = len(rows)
    n_val = max(int(n * val_fraction), min_val_rows) if n > min_val_rows else 0
    if n_val == 0 or n - n_val < min_val_rows:
        return TrainingFrame(rows=rows, feature_columns=frame.feature_columns), TrainingFrame(rows=[], feature_columns=frame.feature_columns)
    train = TrainingFrame(rows=rows[: n - n_val], feature_columns=frame.feature_columns)
    val = TrainingFrame(rows=rows[n - n_val:], feature_columns=frame.feature_columns)
    return train, val


def _isfinite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def to_pandas(frame: TrainingFrame):
    """Lazy pandas import — only loaded when the trainer actually runs."""
    try:
        import pandas as pd  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pandas is required for training; install the 'learned' dependency group:\n"
            "    uv sync --group learned\n"
        ) from e
    df = pd.DataFrame(frame.rows)
    return df


def cold_start_threshold(
    conn: duckdb.DuckDBPyConnection,
    *,
    global_min: int = 5000,
    per_line_dir_min: int = 200,
    min_distinct_buckets: int = 4,
    min_truth_confidence: float = 0.5,
) -> tuple[bool, dict[str, Any]]:
    """Check whether we have enough resolved L outcomes to fit the GBM.

    Returns ``(ready, diagnostics)``. ``diagnostics`` is a dict the CLI
    surfaces to the operator: total rows, per-(line, direction) counts,
    and whether each threshold passed.
    """
    total = conn.execute(
        """
        SELECT COUNT(*) FROM forecast_queue q
          JOIN forecast_outcomes o USING (forecast_id)
         WHERE q.status = 'resolved' AND q.mode = 'L'
           AND COALESCE(o.truth_confidence, 0) >= ?
        """,
        [min_truth_confidence],
    ).fetchone()
    total = int(total[0] if total else 0)

    rows = conn.execute(
        """
        SELECT q.line, q.direction_code, COUNT(*) AS n
          FROM forecast_queue q
          JOIN forecast_outcomes o USING (forecast_id)
         WHERE q.status = 'resolved' AND q.mode = 'L'
           AND COALESCE(o.truth_confidence, 0) >= ?
         GROUP BY q.line, q.direction_code
         ORDER BY n DESC
        """,
        [min_truth_confidence],
    ).fetchall()

    n_strong_buckets = sum(1 for _, _, n in rows if int(n) >= per_line_dir_min)
    diag = {
        "total_resolved": total,
        "global_threshold": global_min,
        "per_line_dir_threshold": per_line_dir_min,
        "min_distinct_buckets": min_distinct_buckets,
        "n_strong_buckets": n_strong_buckets,
        "buckets_top10": [
            {"line": ln, "direction_code": dc, "n": int(n)} for ln, dc, n in rows[:10]
        ],
    }
    ready = (
        total >= global_min
        and n_strong_buckets >= min_distinct_buckets
    )
    diag["ready"] = ready
    return ready, diag
