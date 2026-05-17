"""Streamlit dashboard for transit-observer.

Run via `./run.sh` or directly:
    uv run streamlit run src/transit_observer/dashboard.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

from transit_observer import db
from transit_observer.config import CHICAGO, settings
from transit_observer.corpus import predict_for_od
from transit_observer.corridors import SEED_CORRIDORS
from transit_observer.direction_audit import audit_summary
from transit_observer.metrics import (
    active_predictor_rows,
    available_predictors,
    best_predictor_per_corridor,
    corpus_summary,
    corridor_coverage,
    crps_per_predictor,
    diagnose_pit_shape,
    historical_prediction,
    live_data_diagnostic,
    per_line_resolved_counts,
    pit_histogram,
    pit_histogram_aggregated,
    reliability_curve,
    reliability_curve_aggregated,
    status,
)
from transit_observer.viz import (
    coverage_heatmap_chart,
    pit_histogram_chart,
    quantile_dotplot_chart,
    reliability_diagram_chart,
    sharpness_coverage_chart,
)


st.set_page_config(page_title="transit-observer", layout="wide")


def _db_ready() -> bool:
    return settings.db_path.exists() or settings.read_replica_path.exists()


@st.cache_data(ttl=30)
def _status_dict() -> dict:
    if not _db_ready():
        return {}
    with db.reader() as conn:
        s = status(conn)
    return {
        "L raw arrivals": s.raw_arrivals_count,
        "L positions": s.positions_count,
        "L runs observed": s.runs_observed_count,
        "Bus predictions": s.bus_predictions_count,
        "Metra predictions": s.metra_arrivals_count,
        "Intercampus predictions": s.intercampus_arrivals_count,
        "Forecasts pending": s.forecasts_pending,
        "Forecasts resolved": s.forecasts_resolved,
        "Forecasts unresolvable": s.forecasts_unresolvable,
        "Overall p80 coverage": (
            f"{s.overall_p80_coverage:.1%}" if s.overall_p80_coverage is not None else "—"
        ),
        "Latest L poll": s.latest_poll.isoformat() if s.latest_poll else "—",
    }


@st.cache_data(ttl=30)
def _coverage_df(min_samples: int, predictor_version: str | None = None) -> pd.DataFrame:
    if not _db_ready():
        return pd.DataFrame()
    with db.reader() as conn:
        rows = corridor_coverage(
            conn, min_samples=min_samples, predictor_version=predictor_version,
        )
    return pd.DataFrame(
        [
            {
                "line": r.line,
                "direction": r.direction_label,
                "hour_of_day": r.hour_of_day,
                "weekday": "weekday" if r.weekday else "weekend",
                "n_samples": r.n_samples,
                "coverage_p80": r.coverage_p80,
                "coverage_p90": r.coverage_p90,
                "median_sharpness_s": r.median_sharpness_seconds,
                "median_p50_residual_s": r.median_p50_residual_seconds,
            }
            for r in rows
        ]
    )


@st.cache_data(ttl=30)
def _audit_df(min_samples: int) -> pd.DataFrame:
    if not _db_ready():
        return pd.DataFrame()
    with db.reader() as conn:
        rows = audit_summary(conn, min_samples=min_samples)
    return pd.DataFrame(
        [
            {
                "mode": r.mode,
                "line": r.line,
                "n_audited": r.n_audited,
                "recall": r.recall_rate,
                "direction_precision": r.avg_direction_precision,
            }
            for r in rows
        ]
    )


@st.cache_data(ttl=30)
def _corpus_df(high_confidence_only: bool) -> pd.DataFrame:
    if not _db_ready():
        return pd.DataFrame()
    with db.reader() as conn:
        rows = corpus_summary(conn, high_confidence_only=high_confidence_only)
    return pd.DataFrame(
        [
            {
                "corridor_id": r.corridor_id,
                "mode": r.mode,
                "line": r.line,
                "direction": r.direction,
                "origin": r.origin_label,
                "destination": r.destination_label,
                "n_predictions": r.n_predictions,
                "n_resolved": r.n_resolved,
                "n_unresolvable": r.n_unresolvable,
                "coverage_p80": r.coverage_p80,
                "median_p50_residual_s": r.median_p50_residual_seconds,
                "median_truth_confidence": r.median_truth_confidence,
                "last_predicted_at": r.last_predicted_at,
            }
            for r in rows
        ]
    )


@st.cache_data(ttl=30)
def _residual_df(window_hours: int) -> pd.DataFrame:
    if not _db_ready():
        return pd.DataFrame()
    with db.reader() as conn:
        cutoff = datetime.now(CHICAGO) - timedelta(hours=window_hours)
        rows = conn.execute(
            """
            SELECT q.mode, q.line, q.leave_at, q.predicted_total_p50, q.predicted_total_p80,
                   o.actual_total_seconds, o.in_p80_window
              FROM forecast_outcomes o
              JOIN forecast_queue q USING (forecast_id)
             WHERE q.leave_at >= ?
            """,
            [cutoff],
        ).fetchall()
    return pd.DataFrame(
        rows,
        columns=[
            "mode", "line", "leave_at",
            "predicted_p50", "predicted_p80",
            "actual_total", "in_p80",
        ],
    )


@st.cache_data(ttl=30)
def _reliability_df(
    min_samples: int, mode: str = "per_line", predictor_version: str | None = None,
) -> pd.DataFrame:
    """``mode`` = 'per_line' (split by line, min_samples gate applied) or
    'aggregated' (all data combined under line='ALL')."""
    if not _db_ready():
        return pd.DataFrame()
    with db.reader() as conn:
        if mode == "aggregated":
            rows = reliability_curve_aggregated(conn, predictor_version=predictor_version)
        else:
            rows = reliability_curve(conn, min_samples=min_samples, predictor_version=predictor_version)
    return pd.DataFrame(
        [
            {
                "line": r.line,
                "nominal_quantile": r.nominal_quantile,
                "empirical_coverage": r.empirical_coverage,
                "n": r.n,
            }
            for r in rows
        ]
    )


@st.cache_data(ttl=30)
def _pit_df(
    min_samples: int, n_bins: int, mode: str = "per_line",
    predictor_version: str | None = None,
) -> pd.DataFrame:
    if not _db_ready():
        return pd.DataFrame()
    with db.reader() as conn:
        if mode == "aggregated":
            rows = pit_histogram_aggregated(conn, n_bins=n_bins, predictor_version=predictor_version)
        else:
            rows = pit_histogram(
                conn, n_bins=n_bins, min_samples=min_samples, predictor_version=predictor_version,
            )
    return pd.DataFrame(
        [
            {
                "line": r.line,
                "bin_lower": r.bin_lower,
                "bin_upper": r.bin_upper,
                "count": r.count,
                "density": r.density,
            }
            for r in rows
        ]
    )


@st.cache_data(ttl=30)
def _line_counts_df(predictor_version: str | None = None) -> pd.DataFrame:
    """Per (mode, line) breakdown of resolved-forecast counts — lets the
    user see which lines have enough data to show up in PIT/reliability."""
    if not _db_ready():
        return pd.DataFrame()
    with db.reader() as conn:
        rows = per_line_resolved_counts(conn, predictor_version=predictor_version)
    return pd.DataFrame(
        [
            {
                "mode": r.mode,
                "line": r.line,
                "n_resolved": r.n_resolved,
                "n_high_conf": r.n_resolved_high_conf,
            }
            for r in rows
        ]
    )


@st.cache_data(ttl=30)
def _available_predictors() -> list[tuple[str, int]]:
    """Predictor_versions present in resolved forecasts (with row counts)."""
    if not _db_ready():
        return []
    with db.reader() as conn:
        return available_predictors(conn)


@st.cache_data(ttl=30)
def _active_predictor_df() -> pd.DataFrame:
    """Which predictor is active per corridor + the pending-challenger streak."""
    if not _db_ready():
        return pd.DataFrame()
    with db.reader() as conn:
        rows = active_predictor_rows(conn)
    return pd.DataFrame(
        [
            {
                "corridor_id": r.corridor_id,
                "active": r.predictor_version,
                "score": r.decided_score,
                "decided_at": r.decided_at,
                "challenger": r.pending_candidate or "—",
                "challenger_wins": r.pending_wins,
                "challenger_score": r.pending_score,
            }
            for r in rows
        ]
    )


@st.cache_data(ttl=30)
def _predictor_scoreboard_df(min_samples: int = 30) -> pd.DataFrame:
    """Long-format scoreboard: one row per (predictor, line, direction)."""
    if not _db_ready():
        return pd.DataFrame()
    with db.reader() as conn:
        scores = crps_per_predictor(conn, min_samples=min_samples)
    return pd.DataFrame(
        [
            {
                "predictor": s.predictor_version,
                "line": s.line,
                "direction": s.direction_code or "?",
                "n": s.n,
                "crps": s.crps,
                "pinball_q50": s.pinball_q50,
                "pinball_q80": s.pinball_q80,
                "pinball_q90": s.pinball_q90,
                "coverage_p80": s.coverage_p80,
                "coverage_p90": s.coverage_p90,
                "coverage_gap_p80": s.coverage_gap_p80,
                "tail_miss_p90": s.tail_miss_p90_rate,
                "interval_score_p50_p90": s.interval_score_p50_p90,
                "decision_loss": s.decision_loss,
            }
            for s in scores
        ]
    )


@st.cache_data(ttl=30)
def _bus_v3_status_df() -> pd.DataFrame:
    """Top-line v3 ingest counts per endpoint."""
    if not _db_ready():
        return pd.DataFrame()
    with db.reader() as conn:
        rows = conn.execute(
            """
            SELECT endpoint,
                   COUNT(*) AS polls,
                   SUM(CASE WHEN ok THEN 1 ELSE 0 END) AS ok_polls,
                   MAX(local_response_end_ms) AS last_response_ms
              FROM bus_v3_api_poll
             GROUP BY endpoint
             ORDER BY endpoint
            """
        ).fetchall()
    return pd.DataFrame(
        [
            {"endpoint": e, "polls": int(p), "ok_polls": int(o or 0), "last_response_ms": int(l or 0)}
            for e, p, o, l in rows
        ]
    )


@st.cache_data(ttl=30)
def _bus_v3_arrival_label_df() -> pd.DataFrame:
    """Counts of bus_v3_arrival_event rows broken out by label."""
    if not _db_ready():
        return pd.DataFrame()
    with db.reader() as conn:
        rows = conn.execute(
            """
            SELECT label,
                   SUM(CASE WHEN high_confidence THEN 1 ELSE 0 END) AS high_conf,
                   COUNT(*) AS total
              FROM bus_v3_arrival_event
             GROUP BY label
             ORDER BY label
            """
        ).fetchall()
    return pd.DataFrame(
        [{"label": l, "high_confidence": int(h or 0), "total": int(t or 0)} for l, h, t in rows]
    )


@st.cache_data(ttl=30)
def _bus_v3_residual_df() -> pd.DataFrame:
    """Latest residual quantile rows."""
    if not _db_ready():
        return pd.DataFrame()
    with db.reader() as conn:
        rows = conn.execute(
            """
            SELECT rt, stpid, rtdir, horizon_bin, quality_bin, n,
                   q10_s, q50_s, q90_s, mae_s, bias_s
              FROM bus_v3_residual_quantile
             ORDER BY created_at_ms DESC, rt, stpid, rtdir
             LIMIT 200
            """
        ).fetchall()
    return pd.DataFrame(
        [
            {
                "rt": rt, "stpid": stpid, "rtdir": rtdir,
                "horizon_bin": hbin, "quality_bin": qbin, "n": int(n),
                "q10_s": q10, "q50_s": q50, "q90_s": q90,
                "mae_s": mae, "bias_s": bias,
            }
            for rt, stpid, rtdir, hbin, qbin, n, q10, q50, q90, mae, bias in rows
        ]
    )


def _render_bus_v3_tab() -> None:
    """Read-only summary of the CTA Bus Tracker v3 pipeline."""
    st.subheader("CTA Bus Tracker v3 pipeline")
    st.caption(
        "Parallel to the legacy v2 path. Vehicle pdist-crossings give "
        "high-confidence ground truth; ``bus-telemetry-v1`` is the "
        "predictor that consumes it via the registry."
    )
    status = _bus_v3_status_df()
    if status.empty:
        st.info(
            "No v3 polls yet. Enable in settings and confirm "
            "``CTA_BUS_API_KEY`` is set."
        )
        return
    st.markdown("**Ingest counts (per endpoint)**")
    st.dataframe(status, use_container_width=True, hide_index=True)
    arrivals = _bus_v3_arrival_label_df()
    if not arrivals.empty:
        st.markdown("**Arrival events by label**")
        st.dataframe(arrivals, use_container_width=True, hide_index=True)
        st.caption(
            "``ARRIVED_CONFIRMED`` rows are the only ones used for "
            "accuracy metrics by default. Everything else is recorded "
            "for diagnostics."
        )
    residuals = _bus_v3_residual_df()
    if not residuals.empty:
        st.markdown("**Residual calibration (latest 200 cells)**")
        st.dataframe(residuals, use_container_width=True, hide_index=True)
        st.caption(
            "Empirical residual seconds (actual − predicted) per "
            "``(rt, stpid, rtdir, horizon_bin, quality_bin)`` stratum. "
            "Run ``transit bus-v3 calibrate`` to refresh."
        )


def _render_live_forecast_tab() -> None:
    """Interactive: pick a seeded corridor, predict, dotplot.

    Tries the live kernel first. Falls back to an *empirical* prediction
    drawn from past resolved forecasts for the same OD pair if the live
    feed lacks data. Always shows a diagnostic so the user understands
    which path was taken and why.
    """
    options = {
        f"{c.mode} · {c.line} · {c.origin_label} → {c.destination_label}": c
        for c in SEED_CORRIDORS
    }
    if not options:
        st.info("No seeded corridors available.")
        return

    label = st.selectbox("Corridor", list(options.keys()), key="live_forecast_corridor")
    corridor = options[label]
    source_pref = st.radio(
        "Source",
        ("auto: live → historical fallback", "live only", "historical only"),
        horizontal=True, key="live_forecast_source",
    )
    if not st.button("Predict", key="live_forecast_predict"):
        st.caption(
            "Each dot ≈ 2% probability. Read 'will I arrive in ≤ T?' by "
            "counting dots to the left of T. Auto mode tries the live "
            "kernel first; historical falls back to past resolved trips "
            "for this exact OD when the live feed is sparse."
        )
        return

    now = datetime.now(CHICAGO)
    p50, p80, p90, source_label, n_samples = (None, None, None, None, None)
    diagnostic = None

    if source_pref != "historical only":
        try:
            with db.reader() as conn:
                live = predict_for_od(
                    conn,
                    mode=corridor.mode, line=corridor.line,
                    boarding_int_id=corridor.boarding_int_id,
                    boarding_text_id=corridor.boarding_text_id,
                    alighting_int_id=corridor.alighting_int_id,
                    alighting_text_id=corridor.alighting_text_id,
                    now=now,
                )
                diagnostic = live_data_diagnostic(
                    conn,
                    mode=corridor.mode, line=corridor.line,
                    boarding_int_id=corridor.boarding_int_id,
                    boarding_text_id=corridor.boarding_text_id,
                    now=now,
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Live predictor raised: {type(exc).__name__}: {exc}")
            live = None
        if live is not None and live.predicted_total_p80 > live.predicted_total_p50 > 0:
            p50, p80, p90 = (
                live.predicted_total_p50,
                live.predicted_total_p80,
                live.predicted_total_p90,
            )
            source_label = "live"

    if (p50 is None or p80 is None) and source_pref != "live only":
        with db.reader() as conn:
            hist = historical_prediction(
                conn,
                mode=corridor.mode, line=corridor.line,
                boarding_int_id=corridor.boarding_int_id,
                boarding_text_id=corridor.boarding_text_id,
                alighting_int_id=corridor.alighting_int_id,
                alighting_text_id=corridor.alighting_text_id,
            )
        if hist is not None and hist.p80_seconds > hist.p50_seconds:
            p50, p80, p90 = hist.p50_seconds, hist.p80_seconds, hist.p90_seconds
            source_label = "historical (empirical)"
            n_samples = hist.n_samples

    if p50 is None or p80 is None:
        st.warning(
            "Couldn't produce a prediction from either source. "
            "Live feed has no upcoming arrivals at the boarding stop, and "
            "fewer than 5 past trips have been resolved for this OD pair."
        )
        if diagnostic is not None:
            cols = st.columns(3)
            cols[0].metric(
                "raw rows (live window)", str(diagnostic.raw_rows_in_window),
                help="arrivals at boarding stop seen in last 5 min + 30-min future window",
            )
            cols[1].metric(
                "future-window rows", str(diagnostic.future_rows),
                help="subset where arrival_at >= now",
            )
            last = diagnostic.last_raw_polled_at
            cols[2].metric(
                "last polled", last.strftime("%H:%M:%S") if last else "—",
                help="latest polled_at for this stop in the live window",
            )
        st.caption(
            "Try a different corridor, or wait for the collector to "
            "accumulate more arrivals at this boarding stop."
        )
        return

    cols = st.columns(4)
    cols[0].metric("p50 (min)", f"{p50 / 60:.1f}")
    cols[1].metric("p80 (min)", f"{p80 / 60:.1f}")
    cols[2].metric("p90 (min)", f"{(p90 or p80) / 60:.1f}")
    if n_samples is not None:
        cols[3].metric("n past trips", str(n_samples))
    else:
        cols[3].metric("source", source_label or "?")

    badge = "🟢 live forecast" if source_label == "live" else "🟡 historical fallback"
    st.markdown(f"**{badge}** · `{corridor.line}` · {corridor.origin_label} → {corridor.destination_label}")

    st.altair_chart(
        quantile_dotplot_chart(
            p50, p80,
            title=f"{source_label}: each dot ≈ 2% probability",
        ),
        use_container_width=True,
    )

    if source_label == "live":
        st.caption(
            "Live kernel fitted a log-normal to (p50, p80). 50 dots, "
            "each ≈ 2% probability mass. Count dots to the left of any "
            "target minute to read Pr(arrival within that time)."
        )
        if diagnostic is not None and diagnostic.raw_rows_in_window > 0:
            st.caption(
                f"Diagnostic: {diagnostic.raw_rows_in_window} raw rows at "
                f"boarding stop in live window ({diagnostic.future_rows} in future). "
                f"Last polled at "
                f"{diagnostic.last_raw_polled_at.strftime('%H:%M:%S') if diagnostic.last_raw_polled_at else '?'}."
            )
    else:
        st.caption(
            f"Historical fallback: empirical p50/p80/p90 from the most "
            f"recent {n_samples} resolved trips on this OD pair. The "
            f"log-normal fit and 50-dot encoding are identical to the "
            f"live case — but the underlying distribution is past outcomes, "
            f"not a live arrivals snapshot."
        )


_ALL_PREDICTORS = "all (compare in scoreboard)"


def _render_predictors_section() -> None:
    """Active assignments + head-to-head scoreboard.

    Always shows all predictors regardless of the sidebar filter — this
    section's whole point is the comparison.
    """
    st.subheader("Predictors")
    st.caption(
        "Which predictor is serving each corridor, and how every "
        "predictor scores against realized outcomes. The registry "
        "auto-promotes a candidate once it beats the incumbent on "
        "CRPS + 0.05·coverage_gap for two consecutive evaluation "
        "windows by ≥ 0.005."
    )

    active = _active_predictor_df()
    if active.empty:
        st.info(
            "No `predictor_active` rows yet. The registry seeds rows on "
            "first promotion attempt — `transit predictors switch <c> <v>` "
            "or the in-collector trainer will populate this once data "
            "accumulates."
        )
    else:
        # Headline: who's serving most corridors?
        version_counts = (
            active["active"].value_counts().rename_axis("predictor").reset_index(name="n_corridors")
        )
        cols = st.columns(min(4, len(version_counts)))
        for i, row in version_counts.iterrows():
            cols[i % len(cols)].metric(row["predictor"], f"{row['n_corridors']} corridors")
        st.markdown("**Active predictor per corridor** (with pending challenger streak):")
        st.dataframe(
            active,
            hide_index=True, use_container_width=True,
            column_config={
                "score": st.column_config.NumberColumn("score", format="%.4f"),
                "challenger_score": st.column_config.NumberColumn("challenger score", format="%.4f"),
            },
        )

    scoreboard_min = st.sidebar.slider(
        "Scoreboard: min samples per (predictor, line, dir)",
        min_value=10, max_value=200, value=30, step=10,
    )
    scores = _predictor_scoreboard_df(min_samples=scoreboard_min)
    if scores.empty:
        st.info(
            f"No (predictor, line, direction) buckets with ≥{scoreboard_min} "
            "resolved outcomes yet. Lower the threshold in the sidebar or "
            "wait for more forecasts to resolve."
        )
        return

    st.markdown("**Head-to-head scoreboard** (lower is better for CRPS, "
                "pinball, interval score, decision loss):")
    st.dataframe(
        scores.sort_values(["line", "direction", "decision_loss"]),
        hide_index=True, use_container_width=True,
        column_config={
            "crps": st.column_config.NumberColumn(format="%.2f"),
            "pinball_q50": st.column_config.NumberColumn(format="%.2f"),
            "pinball_q80": st.column_config.NumberColumn(format="%.2f"),
            "pinball_q90": st.column_config.NumberColumn(format="%.2f"),
            "coverage_p80": st.column_config.NumberColumn(format="%.1%%"),
            "coverage_p90": st.column_config.NumberColumn(format="%.1%%"),
            "coverage_gap_p80": st.column_config.NumberColumn(format="%.3f"),
            "tail_miss_p90": st.column_config.NumberColumn(format="%.1%%"),
            "interval_score_p50_p90": st.column_config.NumberColumn(format="%.1f"),
            "decision_loss": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    # Visual: decision loss per (line, direction), bar per predictor.
    # Predictors with the lowest bar in each group win.
    scores_long = scores.assign(
        bucket=lambda d: d["line"].astype(str) + " " + d["direction"].astype(str),
    )
    chart = (
        alt.Chart(scores_long)
        .mark_bar()
        .encode(
            x=alt.X("predictor:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("decision_loss:Q", title="decision loss (lower is better)"),
            color=alt.Color("predictor:N", legend=alt.Legend(orient="bottom")),
            tooltip=["predictor", "line", "direction", "n", "crps",
                     "coverage_p80", "coverage_gap_p80", "decision_loss"],
            column=alt.Column("bucket:N", title=None, header=alt.Header(labelOrient="bottom")),
        )
        .properties(width=70, height=200)
    )
    st.altair_chart(chart, use_container_width=False)
    st.caption(
        "Each cluster = one (line, direction) bucket; bars within a "
        "cluster compare predictors. Anywhere a non-kernel bar is lower "
        "than the kernel bar, that predictor is the current promotion "
        "candidate for that corridor."
    )

    # CRPS vs coverage gap scatter — clarity on the accuracy/calibration tradeoff
    st.markdown("**CRPS vs. coverage gap** "
                "(top-left quadrant is the goal: sharp + well-calibrated):")
    scatter = (
        alt.Chart(scores)
        .mark_circle(size=140, opacity=0.7)
        .encode(
            x=alt.X("coverage_gap_p80:Q", title="|coverage − 0.8|  (calibration gap)"),
            y=alt.Y("crps:Q", title="CRPS  (accuracy)"),
            color=alt.Color("predictor:N", legend=alt.Legend(orient="right")),
            tooltip=["predictor", "line", "direction", "n", "crps",
                     "coverage_p80", "coverage_gap_p80", "decision_loss"],
        )
        .properties(height=320)
    )
    st.altair_chart(scatter, use_container_width=True)


def _predictor_filter() -> str | None:
    """Sidebar widget: which predictor to filter every existing chart by.

    Returns ``None`` when "all" is selected so the downstream metrics
    aggregate across every predictor. Default highlights whichever
    predictor has the most resolved rows (typically kernel-v1 today,
    gbm-v1 later).
    """
    pairs = _available_predictors()
    if not pairs:
        return None
    labels = [_ALL_PREDICTORS] + [f"{pv}  (n={n:,})" for pv, n in pairs]
    choice = st.sidebar.selectbox(
        "Predictor (filters charts below)",
        labels,
        index=0,
        help=(
            "Filters every chart on this page to one predictor's forecasts. "
            "The 'Predictors' section at the bottom compares them all "
            "head-to-head regardless of this selection."
        ),
    )
    if choice == _ALL_PREDICTORS:
        return None
    # Strip the "  (n=...)" suffix
    return choice.split("  (n=")[0]


def _render() -> None:
    st.title("transit-observer")
    st.caption(
        "Long-running validator for the Cozy Fox journey kernel and the "
        "learned residual-quantile GBM. Sidebar predictor filter applies "
        "to every chart except the Predictors scoreboard at the bottom."
    )

    if not _db_ready():
        st.warning(
            f"No DuckDB at {settings.db_path}. The collector hasn't started yet — "
            "run `./run.sh` (or `uv run python -m transit_observer.collector`) and refresh."
        )
        return

    status_data = _status_dict()
    cols = st.columns(4)
    for i, (label, value) in enumerate(status_data.items()):
        cols[i % 4].metric(label, value)

    predictor_version = _predictor_filter()
    if predictor_version is not None:
        st.info(f"Charts below show forecasts from **{predictor_version}** only.")

    st.divider()

    st.subheader("Coverage by corridor")
    min_samples = st.sidebar.slider("Min samples per bucket", min_value=1, max_value=20, value=5)
    coverage = _coverage_df(min_samples, predictor_version=predictor_version)
    if coverage.empty:
        st.info(f"No buckets with ≥{min_samples} samples yet. Leave the collector running.")
    else:
        st.dataframe(coverage, hide_index=True, use_container_width=True)
        chart = (
            alt.Chart(coverage)
            .mark_circle(size=120)
            .encode(
                x=alt.X("hour_of_day:O", title="Hour of day"),
                y=alt.Y("coverage_p80:Q", title="p80 coverage", scale=alt.Scale(domain=[0, 1])),
                color="line:N",
                size=alt.Size("n_samples:Q", legend=None),
                tooltip=["line", "direction", "weekday", "n_samples", "coverage_p80", "coverage_p90"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    st.divider()
    st.subheader("Calibration & forecast displays")
    st.caption(
        "Diagnostic views recommended by the transit-uncertainty literature "
        "(Kay et al., CHI 2016 / 2018) and standard forecast-verification "
        "practice. PIT and reliability assume a log-normal fit through "
        "(p50, p80); see CALIBRATION_VIZ_DESIGN.md for rationale."
    )

    line_counts = _line_counts_df(predictor_version=predictor_version)
    if line_counts.empty:
        st.info(
            "No resolved forecasts yet. Let the collector run until trips "
            "are resolved (typically takes >30 min after the first prediction)."
        )
    else:
        with st.expander(
            f"Data inventory · {int(line_counts['n_resolved'].sum())} resolved forecasts "
            f"across {len(line_counts)} (mode, line) pairs",
            expanded=False,
        ):
            st.dataframe(
                line_counts.rename(columns={
                    "n_resolved": "resolved",
                    "n_high_conf": "high-conf (used by PIT/reliability)",
                }),
                hide_index=True, use_container_width=True,
            )
            st.caption(
                "PIT and reliability filter to truth_confidence ≥ 0.5. "
                "Per-line views require ≥30 high-confidence samples; "
                "aggregated views require ≥1."
            )

    tabs = st.tabs([
        "Reliability",
        "PIT shape",
        "Coverage map",
        "Sharpness ↔ coverage",
        "Live forecast",
        "Bus v3 telemetry",
    ])
    with tabs[0]:
        view = st.radio(
            "View",
            ("aggregated (all lines)", "per line (≥30 samples)"),
            horizontal=True, key="reliability_view",
        )
        if view.startswith("aggregated"):
            df = _reliability_df(min_samples=1, mode="aggregated", predictor_version=predictor_version)
        else:
            df = _reliability_df(min_samples=30, mode="per_line", predictor_version=predictor_version)
        if df.empty:
            st.info(
                "No data for this view yet. Try 'aggregated' for the "
                "broadest pool, or wait for more resolved forecasts."
            )
        else:
            st.altair_chart(
                reliability_diagram_chart(df, facet=view.startswith("per line")),
                use_container_width=True,
            )
            n_total = int(df.groupby("line")["n"].first().sum())
            st.caption(
                f"x = the *claimed* probability (e.g. p80 → 0.80). y = the "
                f"empirical fraction of actuals that landed below the fitted "
                f"q-quantile. Dashed line = perfect calibration. "
                f"Pool: {n_total} samples across "
                f"{df['line'].nunique()} line(s)."
            )
    with tabs[1]:
        view = st.radio(
            "View",
            ("aggregated (all lines)", "per line (≥30 samples)"),
            horizontal=True, key="pit_view",
        )
        n_bins = st.slider(
            "PIT bins", min_value=5, max_value=40, value=20, step=5, key="pit_bins",
        )
        if view.startswith("aggregated"):
            pit = _pit_df(
                min_samples=1, n_bins=n_bins, mode="aggregated",
                predictor_version=predictor_version,
            )
        else:
            pit = _pit_df(
                min_samples=30, n_bins=n_bins, mode="per_line",
                predictor_version=predictor_version,
            )
        if pit.empty:
            st.info(
                "No data for this view. Try 'aggregated' first; the "
                "per-line view needs ≥30 high-confidence samples per line."
            )
        else:
            from transit_observer.metrics import PitBin as _PitBin  # local import for type
            # Reconstruct PitBin records for the textual diagnosis.
            bins = [
                _PitBin(
                    line=r["line"], bin_lower=r["bin_lower"],
                    bin_upper=r["bin_upper"], count=int(r["count"]),
                    density=float(r["density"]),
                )
                for r in pit.to_dict("records")
            ]
            diagnosis = diagnose_pit_shape(bins)
            n_total = int(pit["count"].sum())
            st.markdown(f"**Diagnosis** — {diagnosis}")
            st.altair_chart(
                pit_histogram_chart(pit, facet=view.startswith("per line")),
                use_container_width=True,
            )
            st.caption(
                f"PIT = F(actual) under the fitted log-normal forecast. "
                f"Each bar's height is *density* — a calibrated kernel "
                f"produces density ≈ 1.0 (dashed line) across all bins. "
                f"Mass on the left ⇒ actuals were faster than predicted; "
                f"mass on the right ⇒ slower; piled in the middle ⇒ "
                f"intervals too wide; piled in both tails ⇒ too tight. "
                f"Pool: {n_total} samples across "
                f"{pit['line'].nunique()} line(s)."
            )
    with tabs[2]:
        if coverage.empty:
            st.info("No buckets with the current sample threshold.")
        else:
            st.altair_chart(coverage_heatmap_chart(coverage), use_container_width=True)
            st.caption(
                "Each cell: the bucket's empirical p80 coverage, labelled "
                "as a percentage. Color: deviation from the 80% target "
                "(white = on target). Red cells are overconfident "
                "(intervals too tight); blue are underconfident."
            )
    with tabs[3]:
        if coverage.empty:
            st.info("No buckets with the current sample threshold.")
        else:
            st.altair_chart(
                sharpness_coverage_chart(coverage),
                use_container_width=True,
            )
            st.caption(
                "Each dot is one (line, direction, hour, weekday|weekend) "
                "bucket. The top-left quadrant is the target zone: tight "
                "intervals (low sharpness, x-axis) that still hit 80% "
                "coverage (y-axis). The bottom half is biased; the right "
                "half is loose."
            )
    with tabs[4]:
        _render_live_forecast_tab()
    with tabs[5]:
        _render_bus_v3_tab()

    st.divider()
    _render_predictors_section()

    st.divider()
    st.subheader("Corpus corridors")
    hi_conf = st.sidebar.checkbox("Headline metrics: high-confidence truths only", value=True)
    corpus = _corpus_df(hi_conf)
    if corpus.empty:
        st.info("No corridors seeded yet -- start the collector.")
    else:
        st.dataframe(
            corpus,
            hide_index=True,
            use_container_width=True,
            column_config={
                "coverage_p80": st.column_config.NumberColumn("p80 coverage", format="%.1f%%"),
                "median_p50_residual_s": st.column_config.NumberColumn("median p50 residual (s)", format="%.0f"),
                "median_truth_confidence": st.column_config.NumberColumn("median truth conf", format="%.2f"),
            },
        )

    st.divider()
    st.subheader("Direction-filter audit (recall + direction-precision)")
    audit = _audit_df(min_samples)
    if audit.empty:
        st.info("No audited forecasts yet.")
    else:
        st.dataframe(audit, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Recent residuals")
    window_hours = st.sidebar.slider("Residual window (hours)", min_value=1, max_value=168, value=24)
    residuals = _residual_df(window_hours)
    if residuals.empty:
        st.info("No resolved forecasts in window.")
    else:
        residuals = residuals.assign(
            residual_s=lambda df: df["actual_total"] - df["predicted_p50"],
        )
        st.dataframe(residuals[["mode", "line", "leave_at", "predicted_p50", "actual_total", "residual_s", "in_p80"]], hide_index=True, use_container_width=True)
        chart = (
            alt.Chart(residuals)
            .mark_point(opacity=0.5)
            .encode(
                x=alt.X("predicted_p50:Q", title="Predicted p50 (s)"),
                y=alt.Y("residual_s:Q", title="Actual − Predicted p50 (s)"),
                color="mode:N",
                tooltip=["mode", "line", "leave_at", "predicted_p50", "actual_total", "residual_s"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)


_render()
