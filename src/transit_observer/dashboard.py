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

from . import db
from .config import CHICAGO, settings
from .direction_audit import audit_summary
from .metrics import corridor_coverage, status


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
