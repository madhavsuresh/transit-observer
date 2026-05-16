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
    corpus_summary,
    corridor_coverage,
    pit_histogram,
    reliability_curve,
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
def _coverage_df(min_samples: int) -> pd.DataFrame:
    if not _db_ready():
        return pd.DataFrame()
    with db.reader() as conn:
        rows = corridor_coverage(conn, min_samples=min_samples)
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
def _reliability_df(min_samples: int) -> pd.DataFrame:
    if not _db_ready():
        return pd.DataFrame()
    with db.reader() as conn:
        rows = reliability_curve(conn, min_samples=min_samples)
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
def _pit_df(min_samples: int, n_bins: int) -> pd.DataFrame:
    if not _db_ready():
        return pd.DataFrame()
    with db.reader() as conn:
        rows = pit_histogram(conn, n_bins=n_bins, min_samples=min_samples)
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


def _render_live_forecast_tab() -> None:
    """Interactive: pick a seeded corridor, run the predictor, dotplot it."""
    options = {
        f"{c.mode} · {c.line} · {c.origin_label} → {c.destination_label}": c
        for c in SEED_CORRIDORS
    }
    if not options:
        st.info("No seeded corridors available.")
        return
    label = st.selectbox("Corridor", list(options.keys()), key="live_forecast_corridor")
    corridor = options[label]
    if not st.button("Predict", key="live_forecast_predict"):
        st.caption(
            "Click Predict to run the live kernel against the read replica "
            "and render a 50-dot quantile dotplot. Each dot ≈ 2% probability."
        )
        return

    now = datetime.now(CHICAGO)
    try:
        with db.reader() as conn:
            prediction = predict_for_od(
                conn,
                mode=corridor.mode, line=corridor.line,
                boarding_int_id=corridor.boarding_int_id,
                boarding_text_id=corridor.boarding_text_id,
                alighting_int_id=corridor.alighting_int_id,
                alighting_text_id=corridor.alighting_text_id,
                now=now,
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Predictor failed: {type(exc).__name__}: {exc}")
        return

    if prediction is None:
        st.warning(
            "No prediction available right now — the per-mode predictor "
            "lacked recent data (e.g. no upcoming arrivals at the boarding stop)."
        )
        return

    p50 = prediction.predicted_total_p50
    p80 = prediction.predicted_total_p80
    p90 = prediction.predicted_total_p90
    if not (p50 > 0 and p80 > p50):
        st.warning(
            f"Predictor returned a degenerate quantile triple "
            f"(p50={p50:.0f}s, p80={p80:.0f}s) — cannot fit a distribution."
        )
        return

    cols = st.columns(3)
    cols[0].metric("p50 (minutes)", f"{p50 / 60:.1f}")
    cols[1].metric("p80 (minutes)", f"{p80 / 60:.1f}")
    cols[2].metric("p90 (minutes)", f"{p90 / 60:.1f}")
    st.altair_chart(
        quantile_dotplot_chart(
            p50, p80,
            title=f"{corridor.line}: {corridor.origin_label} → {corridor.destination_label}",
        ),
        use_container_width=True,
    )
    st.caption(
        "Fitted log-normal to (p50, p80). Each dot is one of 50 evenly-spaced "
        "quantiles ⇒ ≈ 2% probability mass per dot. Count dots ≤ a target "
        "duration to read off Pr(arrival within that time)."
    )


def _render() -> None:
    st.title("transit-observer")
    st.caption("Long-running validator for the Cozy Fox journey kernels.")

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

    st.divider()

    st.subheader("Coverage by corridor")
    min_samples = st.sidebar.slider("Min samples per bucket", min_value=1, max_value=20, value=5)
    coverage = _coverage_df(min_samples)
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
        "(p50, p80); see the project README for the rationale."
    )
    tabs = st.tabs([
        "Reliability",
        "PIT shape",
        "Coverage map",
        "Sharpness ↔ coverage",
        "Live forecast",
    ])
    with tabs[0]:
        reliability = _reliability_df(min_samples=max(min_samples, 30))
        if reliability.empty:
            st.info(
                "Need ≥30 resolved forecasts per line to draw a stable "
                "reliability curve. Keep the collector running."
            )
        else:
            st.altair_chart(
                reliability_diagram_chart(reliability),
                use_container_width=True,
            )
            st.caption(
                "Each point: at the nominal quantile q (x), the fraction of "
                "actuals that fell below the fitted q-quantile (y). Dashed "
                "y=x line is perfect calibration. Above ⇒ predictions too "
                "pessimistic; below ⇒ too optimistic."
            )
    with tabs[1]:
        pit = _pit_df(min_samples=max(min_samples, 30), n_bins=20)
        if pit.empty:
            st.info("Need ≥30 resolved forecasts per line to draw a PIT histogram.")
        else:
            st.altair_chart(pit_histogram_chart(pit), use_container_width=True)
            st.caption(
                "Histogram of PIT = F_predicted(actual). "
                "Flat ⇒ calibrated. U-shape ⇒ intervals too tight. "
                "∩-shape ⇒ too wide. Left-skew ⇒ actuals slower than predicted. "
                "Right-skew ⇒ actuals faster than predicted."
            )
    with tabs[2]:
        if coverage.empty:
            st.info("No buckets with the current sample threshold.")
        else:
            st.altair_chart(coverage_heatmap_chart(coverage), use_container_width=True)
            st.caption(
                "p80 coverage minus target (0.80). Red ⇒ overconfident "
                "(intervals too tight); blue ⇒ underconfident (intervals too "
                "wide); white ⇒ on target."
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
                "Lower-right is the worst quadrant: confident *and* wrong. "
                "Aim for points clustered near the dashed line at y=0.8 with "
                "low sharpness (tight intervals)."
            )
    with tabs[4]:
        _render_live_forecast_tab()

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
