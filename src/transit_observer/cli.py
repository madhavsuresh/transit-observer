"""`uv run transit ...` entry point."""

from __future__ import annotations

import click

from . import db
from .direction_audit import audit_summary
from .metrics import calibration_bins, corridor_coverage, status, uncovered_buckets


@click.group()
def cli() -> None:
    """transit-observer command surface."""


@cli.command()
def status_cmd() -> None:
    """One-line health check."""
    with db.reader() as conn:
        s = status(conn)
    click.echo(
        f"L      arrivals={s.raw_arrivals_count}  positions={s.positions_count}  "
        f"runs_observed={s.runs_observed_count}"
    )
    click.echo(f"bus    predictions={s.bus_predictions_count}")
    click.echo(f"metra  predictions={s.metra_arrivals_count}")
    click.echo(f"ic     predictions={s.intercampus_arrivals_count}")
    click.echo(
        f"forecasts pending/resolved/unresolvable="
        f"{s.forecasts_pending}/{s.forecasts_resolved}/{s.forecasts_unresolvable}"
    )
    if s.latest_poll:
        click.echo(f"latest L poll: {s.latest_poll.isoformat()}")
    if s.overall_p80_coverage is not None:
        click.echo(f"overall p80 coverage: {s.overall_p80_coverage:.1%}")


@cli.command()
@click.option("--min-samples", default=5, show_default=True, type=int)
def metrics(min_samples: int) -> None:
    """Coverage / calibration table."""
    with db.reader() as conn:
        rows = corridor_coverage(conn, min_samples=min_samples)
    if not rows:
        click.echo("no buckets with ≥{} samples yet".format(min_samples))
        return
    click.echo(f"{'line':<8}{'dir':<8}{'hour':>4}  {'wkd':>3}  {'n':>4}  {'p80':>6}  {'p90':>6}  {'sharp_s':>8}  {'p50_resid_s':>11}")
    for r in rows:
        click.echo(
            f"{r.line:<8}{r.direction_label:<8}{r.hour_of_day:>4}  {str(r.weekday):>3}  "
            f"{r.n_samples:>4}  {r.coverage_p80:>5.1%}  {r.coverage_p90:>5.1%}  "
            f"{r.median_sharpness_seconds:>8.0f}  {r.median_p50_residual_seconds:>11.0f}"
        )


@cli.command()
@click.option("--target", default=5, show_default=True, type=int)
def corridors(target: int) -> None:
    """Which (line, direction, hour, weekday) buckets need more samples."""
    with db.reader() as conn:
        rows = uncovered_buckets(conn, target_samples=target)
    if not rows:
        click.echo(f"all sampled buckets at >= {target} forecasts")
        return
    click.echo(f"{'line':<8}{'dir':<8}{'hour':>4}  {'wkd':>3}  {'n':>4}")
    for line, direction, hod, weekday, n in rows[:40]:
        click.echo(f"{line:<8}{direction:<8}{hod:>4}  {str(weekday):>3}  {n:>4}")
    if len(rows) > 40:
        click.echo(f"... and {len(rows) - 40} more")


@cli.command()
@click.option("--min-samples", default=5, show_default=True, type=int)
def audit(min_samples: int) -> None:
    """Direction-filter audit. Recall = did the filter keep the boarded
    train? Precision = of arrivals it kept, what fraction matched the
    boarded direction code?"""
    with db.reader() as conn:
        rows = audit_summary(conn, min_samples=min_samples)
    if not rows:
        click.echo(f"no audited lines with ≥{min_samples} samples yet")
        return
    click.echo(f"{'line':<8}{'n':>5}  {'recall':>8}  {'precision':>10}")
    for r in rows:
        click.echo(f"{r.line:<8}{r.n_audited:>5}  {r.recall_rate:>7.1%}  {r.avg_direction_precision:>9.1%}")


@cli.command()
@click.option("--bins", default=10, show_default=True, type=int)
def calibration(bins: int) -> None:
    """Reliability diagram bins for predicted failure probability."""
    with db.reader() as conn:
        items = calibration_bins(conn, n_bins=bins)
    click.echo(f"{'lower':>6}{'upper':>7}  {'n':>5}  {'actual':>7}")
    for b in items:
        click.echo(f"{b.predicted_lower:>6.2f}{b.predicted_upper:>7.2f}  {b.n:>5}  {b.actual_failure_rate:>6.1%}")


def main() -> None:
    cli.add_command(status_cmd, name="status")
    cli(standalone_mode=True)


if __name__ == "__main__":
    main()
