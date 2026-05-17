"""Multi-predictor registry with anti-flap switch-margin logic.

Predictors live as in-memory singletons; ``predictor_active`` (a DB
table) maps each corridor to the predictor that should serve its next
forecast. Cold corridors fall back to ``KERNEL_VERSION``.

Promotion: a candidate replaces the incumbent only when it improves the
composite decision-loss (CRPS + α·coverage_gap) by ``SWITCH_MARGIN`` for
``MIN_CONSECUTIVE_WINS`` evaluation windows in a row. This mirrors
divvy-observer's active-switch policy and is enough to keep noisy
short-window metrics from flapping the active predictor.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import duckdb
import structlog

from ..config import settings
from ..corridors import Corridor

from .bus_telemetry import BUS_TELEMETRY_VERSION, BusTelemetryPredictor
from .journey_kernel import (
    KERNEL_EB_VERSION,
    KERNEL_VERSION,
    JourneyKernelEBPredictor,
    JourneyKernelPredictor,
)
from .protocol import Predictor
from .quantile_gbm import GBM_VERSION, QuantileGBMPredictor
from .train_telemetry import TRAIN_TELEMETRY_VERSION, TrainTelemetryPredictor


log = structlog.get_logger(__name__)


SWITCH_MARGIN = 0.005   # composite score must improve by 0.5% to promote
MIN_CONSECUTIVE_WINS = 2


# Module-level singleton cache: predictor_version -> Predictor instance.
_PREDICTORS: dict[str, Predictor] = {}


def _models_root() -> Path:
    return settings.data_dir / "models"


def _instantiate(version: str) -> Predictor | None:
    if version == KERNEL_VERSION:
        return JourneyKernelPredictor()
    if version == KERNEL_EB_VERSION:
        return JourneyKernelEBPredictor()
    if version == BUS_TELEMETRY_VERSION:
        return BusTelemetryPredictor()
    if version == TRAIN_TELEMETRY_VERSION:
        return TrainTelemetryPredictor()
    if version == GBM_VERSION or version.startswith("gbm-"):
        root = _models_root()
        if not (root / version).is_dir():
            log.info("registry.gbm_missing", version=version, root=str(root))
            return None
        return QuantileGBMPredictor.from_root(root, version=version)
    log.warning("registry.unknown_version", version=version)
    return None


def predictor(version: str) -> Predictor | None:
    """Return the singleton predictor for ``version`` (loading on first use)."""
    if version in _PREDICTORS:
        return _PREDICTORS[version]
    inst = _instantiate(version)
    if inst is None:
        return None
    _PREDICTORS[version] = inst
    return inst


def reset() -> None:
    """Test hook — drop the singleton cache."""
    _PREDICTORS.clear()


def active_version_for_corridor(
    conn: duckdb.DuckDBPyConnection,
    *,
    corridor_id: str,
) -> str:
    """Return which predictor_version is currently active for ``corridor_id``.

    Falls back to ``KERNEL_VERSION`` when no row exists.
    """
    row = conn.execute(
        "SELECT predictor_version FROM predictor_active WHERE corridor_id = ?",
        [corridor_id],
    ).fetchone()
    if row is None or row[0] is None:
        return KERNEL_VERSION
    return str(row[0])


def active_predictor_for(
    conn: duckdb.DuckDBPyConnection,
    *,
    corridor: Corridor,
) -> Predictor:
    """Return the Predictor that should serve ``corridor`` next.

    Always returns a usable predictor — if the configured one fails to
    load (e.g. GBM artifacts missing), falls back to the kernel.
    """
    version = active_version_for_corridor(conn, corridor_id=corridor.corridor_id)
    inst = predictor(version)
    if inst is None:
        log.info("registry.fallback_to_kernel", corridor=corridor.corridor_id, requested=version)
        inst = predictor(KERNEL_VERSION)
        assert inst is not None  # the kernel is always available
    return inst


def promote(
    conn: duckdb.DuckDBPyConnection,
    *,
    corridor_id: str,
    candidate_version: str,
    candidate_score: float,
    incumbent_score: float | None,
    now: datetime,
    margin: float = SWITCH_MARGIN,
    min_consecutive_wins: int = MIN_CONSECUTIVE_WINS,
) -> bool:
    """Consider promoting ``candidate_version`` for ``corridor_id``.

    The composite score is "lower is better" (CRPS-style). A candidate
    promotes only when (a) it beats the incumbent by at least ``margin``
    AND (b) it has done so for ``min_consecutive_wins`` consecutive
    evaluation windows. Streak state lives in ``predictor_active``'s
    ``pending_candidate`` / ``pending_wins`` columns.

    Returns ``True`` iff the active predictor changed.
    """
    cur = conn.execute(
        """
        SELECT predictor_version, decided_score,
               pending_candidate, pending_wins, pending_score
          FROM predictor_active
         WHERE corridor_id = ?
        """,
        [corridor_id],
    ).fetchone()

    if cur is None:
        # No row yet — seed with the kernel as the incumbent, then
        # immediately rerun the promotion logic so the candidate gets a
        # fair shot at its first win.
        conn.execute(
            """
            INSERT INTO predictor_active
                (corridor_id, predictor_version, decided_at,
                 decided_score, incumbent_score, margin, n_consecutive_wins,
                 pending_candidate, pending_wins, pending_score)
            VALUES (?, ?, ?, ?, NULL, NULL, 1, NULL, 0, NULL)
            """,
            [corridor_id, KERNEL_VERSION, now, incumbent_score],
        )
        cur = (KERNEL_VERSION, incumbent_score, None, 0, None)

    incumbent_version, incumbent_recorded, pending_cand, pending_wins, _pending_score = cur
    pending_wins = int(pending_wins or 0)
    incumbent_recorded = (
        float(incumbent_recorded) if incumbent_recorded is not None else (
            float(incumbent_score) if incumbent_score is not None else None
        )
    )

    if candidate_version == incumbent_version:
        # The candidate IS the incumbent — refresh its score, clear any
        # pending challenger that didn't make it through.
        conn.execute(
            """
            UPDATE predictor_active
               SET decided_score = ?, decided_at = ?,
                   pending_candidate = NULL, pending_wins = 0, pending_score = NULL
             WHERE corridor_id = ?
            """,
            [candidate_score, now, corridor_id],
        )
        return False

    beats = (
        incumbent_recorded is not None
        and candidate_score + margin < incumbent_recorded
    )

    if not beats:
        # The candidate failed this window — reset its streak if it had one.
        if pending_cand == candidate_version:
            conn.execute(
                """
                UPDATE predictor_active
                   SET pending_candidate = NULL, pending_wins = 0, pending_score = NULL
                 WHERE corridor_id = ?
                """,
                [corridor_id],
            )
        return False

    # Candidate beats incumbent this window. Is it on a streak?
    new_wins = (pending_wins + 1) if pending_cand == candidate_version else 1
    if new_wins >= min_consecutive_wins:
        delta = incumbent_recorded - candidate_score if incumbent_recorded is not None else 0.0
        conn.execute(
            """
            UPDATE predictor_active
               SET predictor_version = ?, decided_at = ?,
                   decided_score = ?, incumbent_score = ?, margin = ?,
                   n_consecutive_wins = 1,
                   pending_candidate = NULL, pending_wins = 0, pending_score = NULL
             WHERE corridor_id = ?
            """,
            [candidate_version, now, candidate_score, incumbent_recorded, delta, corridor_id],
        )
        log.info(
            "registry.promote",
            corridor=corridor_id,
            from_=incumbent_version, to=candidate_version,
            margin=delta,
        )
        return True

    # Not enough wins yet — record the pending streak and wait.
    conn.execute(
        """
        UPDATE predictor_active
           SET pending_candidate = ?, pending_wins = ?, pending_score = ?
         WHERE corridor_id = ?
        """,
        [candidate_version, new_wins, candidate_score, corridor_id],
    )
    return False


def list_active(
    conn: duckdb.DuckDBPyConnection,
) -> list[tuple[str, str, datetime | None, float | None]]:
    """Return [(corridor_id, predictor_version, decided_at, decided_score)] for every active row."""
    rows = conn.execute(
        """
        SELECT corridor_id, predictor_version, decided_at, decided_score
          FROM predictor_active
         ORDER BY corridor_id
        """
    ).fetchall()
    return [
        (str(c), str(v), d, (float(s) if s is not None else None))
        for c, v, d, s in rows
    ]


def set_active(
    conn: duckdb.DuckDBPyConnection,
    *,
    corridor_id: str,
    predictor_version: str,
    now: datetime,
) -> None:
    """Manual override — CLI ``transit predictors switch`` calls this."""
    conn.execute(
        """
        INSERT INTO predictor_active
            (corridor_id, predictor_version, decided_at, decided_score, incumbent_score, margin, n_consecutive_wins)
        VALUES (?, ?, ?, NULL, NULL, NULL, 1)
        ON CONFLICT (corridor_id) DO UPDATE SET
            predictor_version = excluded.predictor_version,
            decided_at = excluded.decided_at,
            n_consecutive_wins = 1
        """,
        [corridor_id, predictor_version, now],
    )


def candidate_versions(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Versions that have at least one resolved outcome (or for which we
    have a model artifact registered). Always includes ``KERNEL_VERSION``.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT predictor_version
          FROM forecast_queue
         WHERE predictor_version IS NOT NULL
        """
    ).fetchall()
    versions = {
        KERNEL_VERSION, KERNEL_EB_VERSION, BUS_TELEMETRY_VERSION, TRAIN_TELEMETRY_VERSION,
    }
    for (v,) in rows:
        if v:
            versions.add(str(v))
    art_rows = conn.execute(
        "SELECT DISTINCT predictor_version FROM model_artifacts"
    ).fetchall()
    for (v,) in art_rows:
        if v:
            versions.add(str(v))
    return sorted(versions)
