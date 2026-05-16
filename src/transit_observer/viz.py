"""Altair chart constructors for the calibration dashboard.

Each function takes a pre-shaped DataFrame and returns an Altair chart
spec. The dashboard's cached helper functions provide the DataFrames;
this module is presentation-only and easy to test in isolation.

The five charts here correspond to the five jobs identified in the
viz plan:

1. ``coverage_heatmap_chart``     — "which buckets are broken?"
2. ``reliability_diagram_chart``  — "is the model calibrated overall?"
3. ``pit_histogram_chart``        — "what's the *shape* of the miscalibration?"
4. ``sharpness_coverage_chart``   — "are we sharp where we're accurate?"
5. ``quantile_dotplot_chart``     — user-facing: "when will *my* train come?"

The first four are diagnostic; the fifth is the product surface (the
Cozy Fox app will eventually render this for end users).
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from .journey.quantile_distribution import quantile_dotplot_positions


# --- Maintainer diagnostics ----------------------------------------------


def coverage_heatmap_chart(coverage_df: pd.DataFrame) -> alt.Chart:
    """Heatmap of `coverage_p80 − 0.80` faceted by (line, direction).

    Diverging colour: blue = underconfident (intervals too wide),
    red = overconfident (intervals too tight), white = on target.
    """
    if coverage_df.empty:
        return alt.Chart(pd.DataFrame({"msg": ["no data"]})).mark_text().encode()

    df = coverage_df.copy()
    df["coverage_delta"] = df["coverage_p80"] - 0.8

    cell = (
        alt.Chart()
        .mark_rect()
        .encode(
            x=alt.X("hour_of_day:O", title="hour of day"),
            y=alt.Y("weekday:N", title=None),
            color=alt.Color(
                "coverage_delta:Q",
                title="p80 − target",
                scale=alt.Scale(scheme="redblue", domain=[-0.3, 0.3], reverse=True),
            ),
            tooltip=[
                alt.Tooltip("line:N"),
                alt.Tooltip("direction:N"),
                alt.Tooltip("hour_of_day:O", title="hour"),
                alt.Tooltip("weekday:N"),
                alt.Tooltip("n_samples:Q", title="n"),
                alt.Tooltip("coverage_p80:Q", title="p80", format=".1%"),
                alt.Tooltip("coverage_p90:Q", title="p90", format=".1%"),
                alt.Tooltip("median_sharpness_s:Q", title="sharpness (s)", format=".0f"),
            ],
        )
        .properties(width=320, height=80)
    )
    return cell.facet(
        facet=alt.Facet("line:N", title=None),
        columns=2,
        data=df,
    )


def reliability_diagram_chart(reliability_df: pd.DataFrame) -> alt.Chart:
    """Reliability diagram: empirical coverage vs nominal quantile, per line.

    Perfect calibration plots on y=x (rendered as a dashed reference).
    Above diagonal = predictions too pessimistic.
    Below diagonal = predictions too optimistic.
    """
    if reliability_df.empty:
        return alt.Chart(pd.DataFrame({"msg": ["no data"]})).mark_text().encode()

    # Add the y=x reference to the dataframe as two extra rows per line so
    # the diagonal lives on the same data source as the points (faceting
    # layered charts requires a single shared dataset).
    df = reliability_df.copy()
    df["reference"] = df["nominal_quantile"]

    diag = (
        alt.Chart()
        .mark_line(strokeDash=[4, 4], color="gray")
        .encode(
            x=alt.X("nominal_quantile:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("reference:Q", scale=alt.Scale(domain=[0, 1])),
        )
    )

    points = (
        alt.Chart()
        .mark_circle(size=80)
        .encode(
            x=alt.X(
                "nominal_quantile:Q",
                title="nominal quantile",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "empirical_coverage:Q",
                title="empirical coverage",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("line:N", legend=None),
            tooltip=[
                alt.Tooltip("line:N"),
                alt.Tooltip("nominal_quantile:Q", format=".0%"),
                alt.Tooltip("empirical_coverage:Q", format=".1%"),
                alt.Tooltip("n:Q"),
            ],
        )
    )

    line = (
        alt.Chart()
        .mark_line()
        .encode(
            x="nominal_quantile:Q",
            y="empirical_coverage:Q",
            color=alt.Color("line:N", legend=None),
        )
    )

    layered = alt.layer(diag, line, points).properties(width=240, height=240)
    return layered.facet(
        facet=alt.Facet("line:N", title=None),
        columns=4,
        data=df,
    )


def pit_histogram_chart(pit_df: pd.DataFrame) -> alt.Chart:
    """Histogram of PIT values per line, with the uniform-density reference.

    Uniform shape ⇒ calibrated. U-shape ⇒ intervals too tight.
    ∩-shape ⇒ intervals too wide. Skew ⇒ central-tendency bias.
    """
    if pit_df.empty:
        return alt.Chart(pd.DataFrame({"msg": ["no data"]})).mark_text().encode()

    df = pit_df.copy()
    df["uniform"] = 1.0

    bars = (
        alt.Chart()
        .mark_bar()
        .encode(
            x=alt.X("bin_lower:Q", title="PIT bin", scale=alt.Scale(domain=[0, 1])),
            x2="bin_upper:Q",
            y=alt.Y("density:Q", title="density"),
            color=alt.Color("line:N", legend=None),
            tooltip=[
                alt.Tooltip("line:N"),
                alt.Tooltip("bin_lower:Q", format=".2f"),
                alt.Tooltip("bin_upper:Q", format=".2f"),
                alt.Tooltip("count:Q"),
                alt.Tooltip("density:Q", format=".2f"),
            ],
        )
    )

    uniform = (
        alt.Chart()
        .mark_rule(strokeDash=[4, 4], color="gray")
        .encode(y="uniform:Q")
    )

    layered = alt.layer(bars, uniform).properties(width=240, height=200)
    return layered.facet(
        facet=alt.Facet("line:N", title=None),
        columns=4,
        data=df,
    )


def sharpness_coverage_chart(coverage_df: pd.DataFrame) -> alt.Chart:
    """Scatter of (median sharpness, p80 coverage), one point per bucket.

    Target zone: low x, y ≈ 0.80. Lower-right (sharp but miscalibrated)
    is the worst quadrant: confident *and* wrong.
    """
    if coverage_df.empty:
        return alt.Chart(pd.DataFrame({"msg": ["no data"]})).mark_text().encode()

    reference = (
        alt.Chart(pd.DataFrame({"y": [0.8]}))
        .mark_rule(strokeDash=[4, 4], color="gray")
        .encode(y="y:Q")
    )

    points = (
        alt.Chart(coverage_df)
        .mark_circle(opacity=0.7)
        .encode(
            x=alt.X("median_sharpness_s:Q", title="median sharpness (s) — lower is tighter"),
            y=alt.Y(
                "coverage_p80:Q",
                title="p80 coverage",
                scale=alt.Scale(domain=[0, 1]),
            ),
            size=alt.Size("n_samples:Q", title="n samples"),
            color=alt.Color("line:N", title="line"),
            tooltip=[
                "line:N", "direction:N", "hour_of_day:O", "weekday:N",
                "n_samples:Q",
                alt.Tooltip("coverage_p80:Q", format=".1%"),
                alt.Tooltip("median_sharpness_s:Q", format=".0f"),
            ],
        )
    )

    return (reference + points).properties(height=360)


# --- User-facing single-prediction display -------------------------------


def quantile_dotplot_chart(
    p50: float,
    p80: float,
    *,
    n: int = 50,
    dot_radius_seconds: float | None = None,
    title: str | None = None,
) -> alt.Chart:
    """Quantile dotplot of the predicted distribution.

    Per Fernandes et al. (CHI 2018), 50 dots is the sweet spot — yielding
    97% of optimal payoff in transit decision tasks. Each dot represents
    ``1/n`` of the probability mass (so at n=50, each dot ≈ 2%).

    The y-axis is decorative: we stack dots Wilkinson-style by walking
    sorted x-positions and bumping the y-row whenever the next dot is
    within ``dot_radius_seconds`` of the previous one. This produces a
    histogram-shaped pile of countable dots.
    """
    positions = sorted(quantile_dotplot_positions(p50, p80, n=n))
    span = positions[-1] - positions[0]
    if dot_radius_seconds is None:
        dot_radius_seconds = max(span / n, 1.0)

    # Wilkinson stacking: y increases until a gap larger than the dot
    # width is found, then resets to 1.
    rows: list[dict] = []
    current_row = 1
    last_x: float | None = None
    for x in positions:
        if last_x is None or (x - last_x) > dot_radius_seconds:
            current_row = 1
        else:
            current_row += 1
        rows.append({"x_seconds": x, "x_minutes": x / 60.0, "y_row": current_row})
        last_x = x
    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_circle(size=120, color="#1f77b4", opacity=1.0)
        .encode(
            x=alt.X(
                "x_minutes:Q",
                title="Trip duration (minutes)",
                scale=alt.Scale(domain=[max(0.0, df["x_minutes"].min() - 1),
                                        df["x_minutes"].max() + 1]),
            ),
            y=alt.Y("y_row:Q", title=None, axis=None),
            tooltip=[
                alt.Tooltip("x_minutes:Q", title="minutes", format=".1f"),
            ],
        )
        .properties(height=200, title=title or f"each dot ≈ {100 / n:.0f}% probability")
    )
    return chart
