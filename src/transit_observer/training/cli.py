"""CLI for the learned predictor training pipeline.

Wired into the main click group as ``transit train …``. Subcommands:

- ``transit train check``     — diagnose readiness (cold-start gate)
- ``transit train fit``       — train all (leg, line, quantile) boosters
- ``transit train evaluate``  — replay last N resolved outcomes through
  the chosen predictor; report CRPS / pinball / coverage
- ``transit predictors list`` — show which predictor is active per corridor
- ``transit predictors switch <corridor_id> <version>`` — manual override
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import click
import structlog

from .. import db
from ..predictors import registry as pred_registry
from ..predictors.diagnostics import (
    aggregate_coverage,
    aggregate_crps,
    crps_from_quantiles,
    decision_score,
    pinball_loss,
)
from ..predictors.journey_kernel import KERNEL_EB_VERSION, KERNEL_VERSION
from ..predictors.quantile_gbm import GBM_VERSION
from . import dataset as train_dataset
from . import fit as train_fit

log = structlog.get_logger(__name__)


@click.group("train")
def train_group() -> None:
    """Offline training for the learned predictor."""


@train_group.command("check")
def train_check() -> None:
    """Check whether we have enough resolved outcomes to fit the GBM."""
    with db.reader() as conn:
        ready, diag = train_dataset.cold_start_threshold(conn)
    status = "READY" if ready else "NOT YET"
    click.echo(f"GBM training readiness: {status}")
    click.echo(
        f"  resolved_total = {diag['total_resolved']:>6}  "
        f"(threshold {diag['global_threshold']})"
    )
    click.echo(
        f"  strong_buckets  = {diag['n_strong_buckets']:>6}  "
        f"(need {diag['min_distinct_buckets']} × ≥{diag['per_line_dir_threshold']})"
    )
    click.echo("\n  top buckets by sample count:")
    for b in diag["buckets_top10"]:
        click.echo(
            f"    {b['line']:<6} {b['direction_code']:<8}  n={b['n']:>4}"
        )


@train_group.command("fit")
@click.option("--window-days", default=60, show_default=True, type=int)
@click.option("--predictor-version", default=GBM_VERSION, show_default=True)
@click.option("--per-line/--global-only", default=True, show_default=True)
@click.option("--allow-cold", is_flag=True,
              help="Bypass the cold-start gate (use for synthetic-fixture tests).")
def train_fit_cmd(window_days: int, predictor_version: str, per_line: bool, allow_cold: bool) -> None:
    """Train the residual-quantile GBM on rolling window of resolved outcomes."""
    with db.writer() as conn:
        if not allow_cold:
            ready, diag = train_dataset.cold_start_threshold(conn)
            if not ready:
                click.echo("Refusing to fit — cold-start thresholds not met:")
                click.echo(json.dumps(diag, indent=2, default=str))
                click.echo("\nPass --allow-cold to override (testing only).")
                raise click.Abort()
        now = datetime.now()
        since = now - timedelta(days=window_days)
        frame = train_dataset.build_training_frame_l(
            conn, since=since, until=now,
        )
        if len(frame) < 200:
            click.echo(f"Only {len(frame)} usable training rows in the last {window_days} days; aborting.")
            raise click.Abort()
        click.echo(f"Fitting on {len(frame)} rows (window={window_days}d, per_line={per_line})…")
        try:
            report = train_fit.fit_quantile_gbm(
                conn,
                frame,
                predictor_version=predictor_version,
                per_line=per_line,
                now=now,
            )
        except ImportError as e:
            click.echo(f"Missing dependency: {e}", err=True)
            click.echo("  uv sync --group learned", err=True)
            raise click.Abort()
        click.echo(
            f"Trained {len(report.boosters)} boosters. "
            f"train={report.rows_train} val={report.rows_val}"
        )
        for bm in report.boosters[:20]:
            click.echo(
                f"  {bm.leg:<11} {bm.line:<6} q={bm.quantile:.2f}  "
                f"pinball={bm.val_pinball:>7.2f}  crps={bm.val_crps:>7.2f}"
            )
        if len(report.boosters) > 20:
            click.echo(f"  … {len(report.boosters) - 20} more")
        click.echo("Warming up DtACI from the last 500 outcomes…")
        warm = train_fit.warmup_dtaci(conn, predictor_version=predictor_version, n_warmup_rows=500)
        click.echo(f"  conformal state seeded: {warm['n_warmup_updates']} updates from {warm['n_warmup_rows']} rows")


@train_group.command("evaluate")
@click.option("--predictor-version", default=GBM_VERSION, show_default=True)
@click.option("--window-days", default=7, show_default=True, type=int)
def train_evaluate(predictor_version: str, window_days: int) -> None:
    """Score recent resolved outcomes under a given predictor_version.

    Reports CRPS, mean pinball at each quantile, and p80/p90 coverage —
    aggregated globally and per (line, direction). Useful for sanity
    checking after a fresh fit; metrics.py is the canonical source.
    """
    with db.reader() as conn:
        now = datetime.now()
        since = now - timedelta(days=window_days)
        rows = conn.execute(
            """
            SELECT q.line, q.direction_code,
                   q.predicted_wait_p50, q.predicted_wait_p80, q.predicted_wait_p90,
                   o.actual_wait_seconds, o.truth_confidence
              FROM forecast_queue q
              JOIN forecast_outcomes o USING (forecast_id)
             WHERE q.predictor_version = ?
               AND q.mode = 'L' AND q.status = 'resolved'
               AND o.resolved_at >= ?
               AND COALESCE(o.truth_confidence, 0) >= 0.5
            """,
            [predictor_version, since],
        ).fetchall()
    if not rows:
        click.echo(f"No resolved {predictor_version} outcomes in the last {window_days} days.")
        return
    crps_vals = []
    pinball_50, pinball_80, pinball_90 = [], [], []
    p80_hits, p90_hits = [], []
    for ln, dc, p50, p80, p90, actual, _tc in rows:
        if actual is None:
            continue
        q = {0.5: float(p50 or 0), 0.8: float(p80 or 0), 0.9: float(p90 or 0)}
        a = float(actual)
        crps_vals.append(crps_from_quantiles(q, a))
        pinball_50.append(pinball_loss(a, q[0.5], 0.5))
        pinball_80.append(pinball_loss(a, q[0.8], 0.8))
        pinball_90.append(pinball_loss(a, q[0.9], 0.9))
        p80_hits.append(a <= q[0.8])
        p90_hits.append(a <= q[0.9])
    avg_crps = aggregate_crps(crps_vals)
    cov80 = aggregate_coverage(p80_hits)
    cov90 = aggregate_coverage(p90_hits)
    gap80 = abs(cov80 - 0.8) if cov80 == cov80 else float("nan")
    score = decision_score(avg_crps, gap80)
    click.echo(f"{predictor_version}  n={len(rows)}  window={window_days}d")
    click.echo(f"  CRPS = {avg_crps:.2f}")
    click.echo(f"  pinball  q50={_mean(pinball_50):.2f}  q80={_mean(pinball_80):.2f}  q90={_mean(pinball_90):.2f}")
    click.echo(f"  coverage p80={cov80:.1%}  p90={cov90:.1%}")
    click.echo(f"  decision_score = {score:.4f}")


def _mean(values):
    finite = [v for v in values if v == v]  # NaN filter
    return sum(finite) / len(finite) if finite else float("nan")


@click.group("predictors")
def predictors_group() -> None:
    """Inspect and switch active predictors per corridor."""


@predictors_group.command("list")
def predictors_list() -> None:
    """Show the active predictor for each corridor."""
    with db.reader() as conn:
        rows = pred_registry.list_active(conn)
        candidates = pred_registry.candidate_versions(conn)
    click.echo(f"candidates: {', '.join(candidates)}")
    if not rows:
        click.echo("no corridors have an explicit active predictor — all defaulting to kernel-v1")
        return
    click.echo(f"{'corridor_id':<40} {'predictor':<24} {'decided_at':<22} {'score':>8}")
    for corridor_id, version, decided_at, score in rows:
        s = "-" if score is None else f"{score:>8.4f}"
        at = decided_at.isoformat() if decided_at else "-"
        click.echo(f"{corridor_id:<40} {version:<24} {at:<22} {s:>8}")


@predictors_group.command("switch")
@click.argument("corridor_id", required=True)
@click.argument("predictor_version", required=True)
def predictors_switch(corridor_id: str, predictor_version: str) -> None:
    """Manually set the active predictor for a corridor (bypasses promote()).

    Use for hot-fixing a misbehaving predictor or kicking off A/B tests.
    Allowed versions: kernel-v1, kernel-v1+eb, gbm-v1, or any version
    that has at least one model_artifacts row.
    """
    with db.writer() as conn:
        candidates = set(pred_registry.candidate_versions(conn))
        if predictor_version not in candidates and predictor_version not in (
            KERNEL_VERSION, KERNEL_EB_VERSION, GBM_VERSION,
        ):
            click.echo(
                f"Unknown predictor_version '{predictor_version}'. Candidates: "
                f"{', '.join(sorted(candidates))}",
                err=True,
            )
            raise click.Abort()
        pred_registry.set_active(
            conn, corridor_id=corridor_id,
            predictor_version=predictor_version,
            now=datetime.now(),
        )
    click.echo(f"switched {corridor_id} -> {predictor_version}")
