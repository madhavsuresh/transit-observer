"""`uv run transit ...` entry point."""

from __future__ import annotations

import click

from . import db
from .config import CONFIG_PATH
from .corridors import SEED_CORRIDORS, by_id
from .direction_audit import audit_summary
from .metrics import (
    calibration_bins,
    corpus_corridor_rows,
    corpus_summary,
    corridor_coverage,
    status,
    uncovered_buckets,
)
from .setup import write_config
from .training.cli import predictors_group, train_group


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
@click.option("--force", is_flag=True, help="Overwrite an existing config.toml without prompting.")
def setup(force: bool) -> None:
    """Create config.toml interactively (API keys)."""
    if CONFIG_PATH.exists() and not force:
        if not click.confirm(f"{CONFIG_PATH} already exists. Overwrite?", default=False):
            click.echo("Aborted.")
            return
    click.echo()
    click.echo("Set up transit-observer API keys.")
    click.echo("CTA Train Tracker key is required; others are optional (press Enter to skip).")
    click.echo()
    cta_train = click.prompt("CTA Train Tracker API key", type=str).strip()
    cta_bus = click.prompt(
        "CTA Bus Tracker API key (optional)", default="", show_default=False, type=str
    ).strip()
    metra = click.prompt(
        "Metra GTFS-RT API key (optional)", default="", show_default=False, type=str
    ).strip()
    if not cta_train:
        click.echo("error: CTA Train Tracker API key is required.", err=True)
        raise click.Abort()
    write_config(cta_train=cta_train, cta_bus=cta_bus, metra=metra)
    click.echo(f"wrote {CONFIG_PATH}")


@cli.group()
def corpus() -> None:
    """Synthetic-route corpus inspection."""


@corpus.command("list")
def corpus_list() -> None:
    """List seed corridors with last-prediction freshness."""
    with db.reader() as conn:
        summary = corpus_summary(conn)
    if not summary:
        click.echo("no corridors seeded yet -- start the collector to seed them")
        return
    click.echo(
        f"{'corridor_id':<40} {'mode':<12} {'line':<10} "
        f"{'dir':<10} {'n':>4}  {'res':>4}  {'cov80':>6}"
    )
    for r in summary:
        cov = "-" if r.coverage_p80 is None else f"{r.coverage_p80:.1%}"
        click.echo(
            f"{r.corridor_id:<40} {r.mode:<12} {r.line:<10} "
            f"{r.direction:<10} {r.n_predictions:>4}  {r.n_resolved:>4}  {cov:>6}"
        )


@corpus.command("query")
@click.argument("corridor_id", required=True)
@click.option("--limit", default=20, show_default=True, type=int)
def corpus_query(corridor_id: str, limit: int) -> None:
    """Show recent predictions + outcomes for one corridor."""
    seeds = by_id()
    if corridor_id not in seeds:
        click.echo(f"unknown corridor_id: {corridor_id}", err=True)
        click.echo("hint: `uv run transit corpus list` to see seeded corridors", err=True)
        raise click.Abort()
    corr = seeds[corridor_id]
    click.echo(
        f"{corridor_id}  {corr.mode}/{corr.line}/{corr.direction}\n"
        f"  {corr.origin_label}  ->  {corr.destination_label}\n"
    )
    with db.reader() as conn:
        rows = corpus_corridor_rows(conn, corridor_id=corridor_id, limit=limit)
    if not rows:
        click.echo("no predictions yet for this corridor")
        return
    click.echo(
        f"{'leave_at':<22} {'status':<12} {'p50':>7} {'p80':>7} {'p90':>7} "
        f"{'actual':>7} {'resid':>7} {'in_p80':>6} {'tconf':>5} {'pv':<12}"
    )
    for r in rows:
        actual = "-" if r.actual_total_seconds is None else f"{r.actual_total_seconds:>6.0f}s"
        resid = "-" if r.p50_residual_seconds is None else f"{r.p50_residual_seconds:>+6.0f}"
        in_p80 = "-" if r.in_p80_window is None else ("yes" if r.in_p80_window else "no")
        tconf = "-" if r.truth_confidence is None else f"{r.truth_confidence:.2f}"
        pv = (r.predictor_version or "-")[:12]
        click.echo(
            f"{r.leave_at.isoformat():<22} {r.status:<12} "
            f"{r.predicted_total_p50:>6.0f}s {r.predicted_total_p80:>6.0f}s {r.predicted_total_p90:>6.0f}s "
            f"{actual:>7} {resid:>7} {in_p80:>6} {tconf:>5} {pv:<12}"
        )


@cli.command()
@click.option("--bins", default=10, show_default=True, type=int)
def calibration(bins: int) -> None:
    """Reliability diagram bins for predicted failure probability."""
    with db.reader() as conn:
        items = calibration_bins(conn, n_bins=bins)
    click.echo(f"{'lower':>6}{'upper':>7}  {'n':>5}  {'actual':>7}")
    for b in items:
        click.echo(f"{b.predicted_lower:>6.2f}{b.predicted_upper:>7.2f}  {b.n:>5}  {b.actual_failure_rate:>6.1%}")


@cli.group("bus-v3")
def bus_v3_group() -> None:
    """CTA Bus Tracker v3 parallel pipeline commands."""


@bus_v3_group.command("infer-arrivals")
@click.option("--run-id", default=None, help="Restrict to a single v3 cycle run_id.")
@click.option("--route", default=None, help="Restrict to a single CTA route.")
@click.option("--stop", default=None, help="Restrict to a single stop ID.")
@click.option("--direction", default=None, help="Restrict to a single rtdir.")
@click.option("--replace", is_flag=True, default=False,
              help="Delete prior events for the run_id before re-inferring.")
def bus_v3_infer_arrivals(run_id, route, stop, direction, replace) -> None:
    """Run pdist-crossing arrival inference over bus_v3_* tables."""
    from .bus_v3.inference import infer_bus_arrivals

    with db.writer() as conn:
        n = infer_bus_arrivals(
            conn, run_id=run_id, route=route, stop_id=stop, direction=direction, replace=replace,
        )
    click.echo(f"inferred {n} arrival event(s)")


@bus_v3_group.command("calibrate")
@click.option("--min-n", default=20, show_default=True, type=int,
              help="Minimum sample count per (rt, stpid, rtdir, horizon_bin, quality_bin) cell.")
@click.option("--predictor-version", default="bus-telemetry-v1", show_default=True)
def bus_v3_calibrate(min_n, predictor_version) -> None:
    """Refresh the bus_v3_residual_quantile empirical calibration table."""
    from .bus_v3.calibration import refresh_bus_residual_quantiles

    with db.writer() as conn:
        n = refresh_bus_residual_quantiles(
            conn, min_n=min_n, predictor_version=predictor_version,
        )
    click.echo(f"wrote {n} calibration cell(s)")


@bus_v3_group.command("status")
def bus_v3_status() -> None:
    """Show v3 ingest, inference, and calibration counts."""
    with db.reader() as conn:
        rows = conn.execute(
            """
            SELECT endpoint, COUNT(*) AS n, MAX(local_response_end_ms) AS last_ms
              FROM bus_v3_api_poll
             GROUP BY endpoint
             ORDER BY endpoint
            """
        ).fetchall()
        events = conn.execute(
            """
            SELECT label,
                   SUM(CASE WHEN high_confidence THEN 1 ELSE 0 END) AS high,
                   COUNT(*) AS total
              FROM bus_v3_arrival_event
             GROUP BY label
             ORDER BY label
            """
        ).fetchall()
        cal = conn.execute("SELECT COUNT(*) FROM bus_v3_residual_quantile").fetchone()[0]
    click.echo("bus_v3_api_poll by endpoint:")
    for endpoint, n, _last in rows:
        click.echo(f"  {endpoint:<25} {n}")
    click.echo("\nbus_v3_arrival_event by label:")
    for label, high, total in events:
        click.echo(f"  {label:<40} high={high} total={total}")
    click.echo(f"\nbus_v3_residual_quantile rows: {cal}")


@cli.group("train-v2")
def train_v2_group() -> None:
    """CTA Train (L) v2 parallel pipeline commands."""


@train_v2_group.command("infer-arrivals")
@click.option("--run-id", default=None, help="Restrict to a single v2 cycle run_id.")
@click.option("--replace", is_flag=True, default=False,
              help="Delete prior events for the run_id before re-inferring.")
def train_v2_infer_arrivals(run_id, replace) -> None:
    """Run nextStaId-transition arrival inference over train_v2_* tables."""
    from .train_v2.inference import infer_train_arrivals

    with db.writer() as conn:
        n = infer_train_arrivals(conn, run_id=run_id, replace=replace)
    click.echo(f"inferred {n} arrival event(s)")


@train_v2_group.command("topology")
@click.option("--window-hours", default=168, show_default=True, type=int)
@click.option("--min-transitions", default=3, show_default=True, type=int)
def train_v2_topology(window_hours, min_transitions) -> None:
    """Rebuild train_v2_line_topology from observed run transitions."""
    from .train_v2.topology import refresh_line_topology

    with db.writer() as conn:
        n = refresh_line_topology(
            conn, window_hours=window_hours, min_transitions=min_transitions,
        )
    click.echo(f"wrote {n} topology row(s)")


@train_v2_group.command("calibrate")
@click.option("--min-n", default=20, show_default=True, type=int,
              help="Minimum sample count per (line, map_id, direction, horizon_bin, quality_bin) cell.")
@click.option("--predictor-version", default="train-telemetry-v1", show_default=True)
def train_v2_calibrate(min_n, predictor_version) -> None:
    """Refresh the train_v2_residual_quantile empirical calibration table."""
    from .train_v2.calibration import refresh_train_residual_quantiles

    with db.writer() as conn:
        n = refresh_train_residual_quantiles(
            conn, min_n=min_n, predictor_version=predictor_version,
        )
    click.echo(f"wrote {n} calibration cell(s)")


@train_v2_group.command("status")
def train_v2_status() -> None:
    """Show v2 train ingest, inference, and calibration counts."""
    with db.reader() as conn:
        rows = conn.execute(
            """
            SELECT source, endpoint, COUNT(*) AS n, MAX(local_response_end_ms) AS last_ms
              FROM train_v2_api_poll
             GROUP BY source, endpoint
             ORDER BY source, endpoint
            """
        ).fetchall()
        events = conn.execute(
            """
            SELECT label,
                   SUM(CASE WHEN high_confidence THEN 1 ELSE 0 END) AS high,
                   COUNT(*) AS total
              FROM train_v2_arrival_event
             GROUP BY label
             ORDER BY label
            """
        ).fetchall()
        cal = conn.execute("SELECT COUNT(*) FROM train_v2_residual_quantile").fetchone()[0]
        topo = conn.execute("SELECT COUNT(*) FROM train_v2_line_topology").fetchone()[0]
    click.echo("train_v2_api_poll by (source, endpoint):")
    for source, endpoint, n, _last in rows:
        click.echo(f"  {source:<14} {endpoint:<22} {n}")
    click.echo("\ntrain_v2_arrival_event by label:")
    for label, high, total in events:
        click.echo(f"  {label:<40} high={high} total={total}")
    click.echo(f"\ntrain_v2_residual_quantile rows: {cal}")
    click.echo(f"train_v2_line_topology rows: {topo}")


@cli.group("gtfs-static")
def gtfs_static_group() -> None:
    """GTFS-static archive + extraction commands."""


@gtfs_static_group.command("extract")
@click.option("--agency", required=True, help="Agency tag, e.g. 'cta', 'metra', 'pace'.")
@click.option("--archive", "archive_path", required=True, type=click.Path(exists=True),
              help="Path to the GTFS .zip on disk.")
@click.option("--replace/--no-replace", default=True,
              help="If a snapshot with the same sha256 exists, replace its rows.")
def gtfs_static_extract(agency, archive_path, replace) -> None:
    """Parse a GTFS zip into the gtfs_static_* tables."""
    from .gtfs_static_extract import extract_gtfs_archive

    with db.writer() as conn:
        snapshot_id = extract_gtfs_archive(
            conn, agency=agency, archive_path=archive_path,
            replace_existing=replace,
        )
    click.echo(f"snapshot_id = {snapshot_id}")


@gtfs_static_group.command("extract-latest")
@click.option("--agency", help="Restrict to one agency (default: all).")
def gtfs_static_extract_latest(agency) -> None:
    """Extract the latest archived GTFS zip per agency."""
    from .gtfs_static_extract import extract_gtfs_archive
    from pathlib import Path

    with db.writer() as conn:
        params: list = []
        where = ""
        if agency:
            where = "WHERE agency = ?"
            params.append(agency)
        rows = conn.execute(
            f"""
            SELECT agency, archive_path, downloaded_at
              FROM gtfs_feed_versions {where}
             QUALIFY ROW_NUMBER() OVER (PARTITION BY agency ORDER BY downloaded_at DESC) = 1
            """,
            params,
        ).fetchall()
        n = 0
        for agency_name, archive_path, downloaded_at in rows:
            sid = extract_gtfs_archive(
                conn,
                agency=str(agency_name),
                archive_path=Path(str(archive_path)),
                fetched_at_ms=int(downloaded_at.timestamp() * 1000) if downloaded_at else None,
            )
            click.echo(f"{agency_name}: snapshot_id={sid}")
            n += 1
    click.echo(f"extracted {n} archive(s)")


@gtfs_static_group.command("status")
def gtfs_static_status() -> None:
    """Show GTFS-static snapshot counts per agency."""
    with db.reader() as conn:
        if not conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'gtfs_static_snapshot' LIMIT 1"
        ).fetchone():
            click.echo("no gtfs_static_snapshot table yet")
            return
        rows = conn.execute(
            """
            SELECT agency,
                   COUNT(*) AS n_snapshots,
                   MAX(extracted_at_ms) AS last_extracted_ms,
                   SUM(n_stop_times) AS total_stop_times,
                   SUM(n_stops) AS total_stops
              FROM gtfs_static_snapshot
             GROUP BY agency
             ORDER BY agency
            """
        ).fetchall()
    click.echo("gtfs_static_snapshot by agency:")
    for agency, n, last_ms, stop_times, stops in rows:
        click.echo(f"  {agency:<10} snapshots={n} stops={stops} stop_times={stop_times}")


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8001, show_default=True, type=int,
              help="Port 8000 is reserved for the sister divvy-observer service.")
def api(host: str, port: int) -> None:
    """Run the HTTP prediction API. POST queries get logged for auto-promote."""
    import uvicorn
    uvicorn.run("transit_observer.api:app", host=host, port=port, reload=False)


def main() -> None:
    cli.add_command(status_cmd, name="status")
    cli.add_command(train_group, name="train")
    cli.add_command(predictors_group, name="predictors")
    cli(standalone_mode=True)


if __name__ == "__main__":
    main()
