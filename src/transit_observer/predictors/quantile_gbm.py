"""Residual-target conformalized quantile gradient boosting predictor.

Architecture:
  1. Run the journey kernel to produce a baseline (mean + quantiles).
  2. Extract a feature vector via predictors.features.
  3. For each quantile q in {0.5, 0.8, 0.9} and each leg in {wait,
     in_vehicle}, call a LightGBM booster trained on the *residual*
     (actual − kernel_p50). Output: per-quantile residual prediction.
  4. Isotonic post-processing: sort residual predictions ascending so
     q0.5 ≤ q0.8 ≤ q0.9 (LightGBM trained independently can produce
     crossings).
  5. Add back the kernel's mean to recover absolute quantile predictions.
  6. Apply DtACI conformal offset per (line, direction, leg, quantile).

If LightGBM isn't installed, model files are missing for the corridor's
line, or feature_completeness is too low, the predictor returns ``None``
(the registry falls back to a kernel).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import structlog

from ..corridors import Corridor

from . import conformal
from . import features as feats
from .journey_kernel import _dispatch
from .protocol import SCHEMA_QUANTILES, Prediction, PredictionLeg

log = structlog.get_logger(__name__)


GBM_VERSION = "gbm-v1"

# Match to the schema-storable triple. The lower tail (0.1) isn't stored
# directly; if needed for diagnostics we recover it from a log-normal
# fit on (p50, p80), same as the kernel.
GBM_QUANTILES: tuple[float, ...] = SCHEMA_QUANTILES

# Schedule-fallback threshold: if fewer than this fraction of the
# expected dynamic features are populated, defer to the kernel.
MIN_FEATURE_COMPLETENESS: float = 0.7

LEGS: tuple[str, ...] = ("wait", "in_vehicle")


@dataclass
class _LineBoosters:
    """All boosters needed to predict one leg for one line: one Booster
    per quantile."""

    boosters: dict[float, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)


@dataclass
class QuantileGBMArtifact:
    """Disk-loaded ensemble for one predictor_version.

    Boosters are loaded lazily per (leg, line) on first use.
    """

    predictor_version: str
    artifacts_root: Path
    boosters_by_leg_line: dict[tuple[str, str], _LineBoosters] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)

    def _key(self, leg: str, line: str) -> tuple[str, str]:
        return (leg, line)

    def get(self, leg: str, line: str) -> _LineBoosters | None:
        key = self._key(leg, line)
        if key in self.boosters_by_leg_line:
            return self.boosters_by_leg_line[key]
        bundle = self._load_from_disk(leg, line)
        if bundle is None:
            return None
        self.boosters_by_leg_line[key] = bundle
        return bundle

    def _load_from_disk(self, leg: str, line: str) -> _LineBoosters | None:
        try:
            import joblib  # type: ignore
        except ImportError:
            log.warning("gbm.load_skip", reason="joblib_missing")
            return None
        line_dir = self.artifacts_root / self.predictor_version / leg
        if not line_dir.is_dir():
            return None
        bundle = _LineBoosters()
        for q in GBM_QUANTILES:
            path = line_dir / f"{line}_q{q:.2f}.joblib"
            if not path.exists():
                path = line_dir / f"ALL_q{q:.2f}.joblib"  # global fallback
            if not path.exists():
                return None
            obj = joblib.load(path)
            if isinstance(obj, dict):
                bundle.boosters[q] = obj.get("model")
                if not bundle.feature_columns:
                    bundle.feature_columns = list(obj.get("feature_columns", []))
            else:
                bundle.boosters[q] = obj
        if not bundle.boosters:
            return None
        return bundle


class QuantileGBMPredictor:
    """LightGBM residual-quantile predictor.

    Built around an ``QuantileGBMArtifact`` loaded once at process start
    and reused for all predictions. Each ``predict()`` call:
      1. dispatches the kernel to get a baseline and feature snapshot,
      2. validates feature_completeness,
      3. predicts residuals from each booster,
      4. fixes quantile crossings,
      5. applies the DtACI offset,
      6. returns a Prediction.
    """

    predictor_version = GBM_VERSION

    def __init__(self, artifact: QuantileGBMArtifact) -> None:
        self.artifact = artifact

    @classmethod
    def from_root(cls, root: Path | str, *, version: str = GBM_VERSION) -> "QuantileGBMPredictor":
        root_path = Path(root)
        return cls(QuantileGBMArtifact(predictor_version=version, artifacts_root=root_path))

    def predict(
        self,
        conn: duckdb.DuckDBPyConnection,
        corridor: Corridor,
        *,
        now: datetime,
    ) -> Prediction | None:
        if corridor.mode != "L":
            return None
        kd = _dispatch(conn, corridor, now=now)
        if kd is None or kd.feature_bundle is None:
            return None
        if kd.feature_bundle.completeness < MIN_FEATURE_COMPLETENESS:
            return None

        feature_row = feats.normalize_for_model(kd.feature_bundle.values)
        baseline_wait = kd.wait
        baseline_iv = kd.in_vehicle

        wait_quantiles = self._predict_leg(
            "wait", line=kd.line, feature_row=feature_row, baseline_p50=baseline_wait.p50,
        )
        if wait_quantiles is None:
            return None
        iv_quantiles = self._predict_leg(
            "in_vehicle", line=kd.line, feature_row=feature_row, baseline_p50=baseline_iv.p50,
        )
        if iv_quantiles is None:
            iv_quantiles = {q: baseline_iv.quantiles.get(q, baseline_iv.mean) for q in GBM_QUANTILES}

        # DtACI conformal adjustment per quantile
        wait_offsets = conformal.offsets_for(
            conn, predictor_version=self.predictor_version,
            line=kd.line, direction_code=kd.direction_code or "",
            leg="wait", quantiles=GBM_QUANTILES,
        )
        iv_offsets = conformal.offsets_for(
            conn, predictor_version=self.predictor_version,
            line=kd.line, direction_code=kd.direction_code or "",
            leg="in_vehicle", quantiles=GBM_QUANTILES,
        )
        wait_quantiles = {q: max(0.0, v + wait_offsets.get(q, 0.0)) for q, v in wait_quantiles.items()}
        iv_quantiles = {q: max(0.0, v + iv_offsets.get(q, 0.0)) for q, v in iv_quantiles.items()}
        # Re-enforce isotonic order after the conformal adjustment.
        wait_quantiles = _isotonize(wait_quantiles)
        iv_quantiles = _isotonize(iv_quantiles)

        # Treat the q0.5 as the mean for storage (the kernel does the same).
        wait_leg = PredictionLeg(
            quantiles=wait_quantiles,
            mean=wait_quantiles[0.5],
            confidence=kd.feature_bundle.completeness,
            sample_count=int(kd.feature_bundle.values.get("n_upcoming_arrivals_30m") or 0),
        )
        iv_leg = PredictionLeg(
            quantiles=iv_quantiles,
            mean=iv_quantiles[0.5],
            confidence=kd.feature_bundle.completeness,
            sample_count=wait_leg.sample_count,
        )

        snap = dict(kd.feature_bundle.values)
        snap["feature_completeness"] = kd.feature_bundle.completeness
        if kd.wait_forecast and kd.wait_forecast.next_departure_at:
            snap["next_departure_at"] = kd.wait_forecast.next_departure_at.isoformat()

        return Prediction(
            predictor_version=self.predictor_version,
            wait=wait_leg,
            in_vehicle=iv_leg,
            feature_snapshot=snap,
            feature_completeness=kd.feature_bundle.completeness,
            state_label=None,
            explanation=None,
            schedule_fallback=False,
        )

    def _predict_leg(
        self,
        leg: str,
        *,
        line: str,
        feature_row: dict[str, Any],
        baseline_p50: float,
    ) -> dict[float, float] | None:
        bundle = self.artifact.get(leg, line)
        if bundle is None:
            return None
        try:
            import numpy as np  # type: ignore
            import pandas as pd  # type: ignore
        except ImportError:
            log.warning("gbm.predict_skip", reason="numpy_or_pandas_missing")
            return None

        cols = bundle.feature_columns or list(feature_row.keys())
        df = pd.DataFrame([{c: feature_row.get(c, np.nan) for c in cols}])
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype("category")

        raw_residuals: dict[float, float] = {}
        for q in GBM_QUANTILES:
            booster = bundle.boosters.get(q)
            if booster is None:
                return None
            try:
                pred = booster.predict(df)
            except Exception:  # noqa: BLE001 — booster API is opaque
                log.exception("gbm.predict_error", leg=leg, line=line, quantile=q)
                return None
            raw_residuals[q] = float(pred[0])

        # Add back to kernel baseline and isotonize.
        absolute = {q: max(0.0, baseline_p50 + r) for q, r in raw_residuals.items()}
        return _isotonize(absolute)


def _isotonize(quantiles: dict[float, float]) -> dict[float, float]:
    """Pool adjacent violators isn't needed for three points — just sort.

    For more than three quantiles, a real PAV would be appropriate; for
    {0.5, 0.8, 0.9} sorting by ascending nominal quantile and
    monotonically clamping is exact.
    """
    if not quantiles:
        return {}
    items = sorted(quantiles.items(), key=lambda kv: kv[0])
    out: dict[float, float] = {}
    last = -math.inf
    for q, v in items:
        new = max(last, v)
        out[q] = new
        last = new
    return out
