"""Save/load LightGBM artifacts to disk and register them in the DB.

Layout::

    data/models/{predictor_version}/{leg}/{line}_q{q:.2f}.joblib

``line`` may be ``ALL`` for a global model. The companion row in
``model_artifacts`` ties (predictor_version, leg, line, quantile) to a
filesystem path and records training-time metrics.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from ..config import settings


log = logging.getLogger(__name__)


def models_root() -> Path:
    return settings.data_dir / "models"


def artifact_path(
    *, predictor_version: str, leg: str, line: str, quantile: float,
) -> Path:
    return models_root() / predictor_version / leg / f"{line}_q{quantile:.2f}.joblib"


def save_booster(
    booster: Any,
    *,
    predictor_version: str,
    leg: str,
    line: str,
    quantile: float,
    feature_columns: list[str],
) -> Path:
    """Persist a single Booster + the feature schema it was trained on."""
    try:
        import joblib  # type: ignore
    except ImportError as e:
        raise ImportError(
            "joblib is required to save model artifacts; install the 'learned' "
            "dependency group: uv sync --group learned"
        ) from e
    path = artifact_path(
        predictor_version=predictor_version, leg=leg,
        line=line, quantile=quantile,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": booster, "feature_columns": feature_columns}, path)
    return path


def register_artifact(
    conn: duckdb.DuckDBPyConnection,
    *,
    predictor_version: str,
    leg: str,
    line: str,
    quantile: float,
    artifact_path: Path,
    n_train_rows: int,
    n_val_rows: int,
    val_pinball_loss: float,
    val_crps: float,
    feature_columns: list[str],
    now: datetime,
) -> None:
    """Upsert a model_artifacts row pointing at the saved file."""
    conn.execute(
        """
        INSERT INTO model_artifacts
            (predictor_version, leg, line, quantile, artifact_path,
             trained_at, n_train_rows, n_val_rows,
             val_pinball_loss, val_crps, feature_columns)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (predictor_version, leg, line, quantile) DO UPDATE SET
            artifact_path    = excluded.artifact_path,
            trained_at       = excluded.trained_at,
            n_train_rows     = excluded.n_train_rows,
            n_val_rows       = excluded.n_val_rows,
            val_pinball_loss = excluded.val_pinball_loss,
            val_crps         = excluded.val_crps,
            feature_columns  = excluded.feature_columns
        """,
        [
            predictor_version, leg, line, quantile, str(artifact_path),
            now, n_train_rows, n_val_rows,
            val_pinball_loss, val_crps,
            json.dumps(feature_columns),
        ],
    )


def list_artifacts(
    conn: duckdb.DuckDBPyConnection,
    *,
    predictor_version: str | None = None,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM model_artifacts"
    params: list[Any] = []
    if predictor_version is not None:
        sql += " WHERE predictor_version = ?"
        params.append(predictor_version)
    sql += " ORDER BY trained_at DESC, leg, line, quantile"
    rows = conn.execute(sql, params).fetchall()
    cols = [d[0] for d in conn.description]
    return [dict(zip(cols, r)) for r in rows]
