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
    """Heatmap of `coverage_p80 − 0.80` faceted by line.

    Diverging colour: blue = underconfident (intervals too wide),
    red = overconfident (intervals too tight), white = on target (0.80).
    Each cell carries its sample count as a text annotation.
    """
    if coverage_df.empty:
        return alt.Chart(pd.DataFrame({"msg": ["no data"]})).mark_text().encode()

    df = coverage_df.copy()
    df["coverage_delta"] = df["coverage_p80"] - 0.8
    df["coverage_pct"] = (df["coverage_p80"] * 100).round().astype(int).astype(str) + "%"

    rect = (
        alt.Chart()
        .mark_rect()
        .encode(
            x=alt.X("hour_of_day:O", title="hour of day"),
            y=alt.Y("weekday:N", title=None),
            color=alt.Color(
                "coverage_delta:Q",
                title="p80 coverage − 0.80",
                scale=alt.Scale(scheme="redblue", domain=[-0.3, 0.3], reverse=True),
                legend=alt.Legend(format=".0%"),
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
    )
    text = (
        alt.Chart()
        .mark_text(fontSize=10, color="black")
        .encode(
            x=alt.X("hour_of_day:O"),
            y=alt.Y("weekday:N"),
            text=alt.Text("coverage_pct:N"),
        )
    )
    layered = alt.layer(rect, text).properties(width=320, height=80)
    return layered.facet(
        facet=alt.Facet("line:N", title="Line — cells show p80 coverage; target is 80% (white)"),
        columns=2,
        data=df,
    )


def reliability_diagram_chart(
    reliability_df: pd.DataFrame, *, facet: bool = False,
) -> alt.Chart:
    """Reliability diagram: empirical coverage vs claimed probability.

    By default plots one chart with all lines overlaid (colored, with
    a legend) so the user can compare lines at a glance. Pass
    ``facet=True`` for one panel per line — useful when there are many
    lines.

    The diagonal is `y = x`: predictions claiming probability q match
    reality if exactly q-fraction of actuals fell at or below the
    claimed q-quantile. Above the diagonal = predictions too
    pessimistic (intervals wider than needed); below = too optimistic
    (intervals too tight).
    """
    if reliability_df.empty:
        return alt.Chart(pd.DataFrame({"msg": ["no data"]})).mark_text().encode()

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

    line_mark = (
        alt.Chart()
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(
                "nominal_quantile:Q",
                title="claimed probability (e.g. p80 → 0.80)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "empirical_coverage:Q",
                title="fraction of actuals that fell below the q-quantile",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "line:N",
                title="line",
                legend=alt.Legend(orient="right"),
            ),
        )
    )

    points = (
        alt.Chart()
        .mark_circle(size=80)
        .encode(
            x=alt.X("nominal_quantile:Q"),
            y=alt.Y("empirical_coverage:Q"),
            color=alt.Color("line:N", legend=None),
            tooltip=[
                alt.Tooltip("line:N"),
                alt.Tooltip("nominal_quantile:Q", title="claimed", format=".0%"),
                alt.Tooltip("empirical_coverage:Q", title="observed", format=".1%"),
                alt.Tooltip("n:Q", title="n samples"),
            ],
        )
    )

    # Annotations: "above diagonal = pessimistic", "below = optimistic"
    annotations = pd.DataFrame([
        {"x": 0.25, "y": 0.65, "label": "above: too pessimistic"},
        {"x": 0.7, "y": 0.3, "label": "below: too optimistic"},
    ])
    notes = (
        alt.Chart(annotations)
        .mark_text(color="#999", fontSize=10, fontStyle="italic")
        .encode(x="x:Q", y="y:Q", text="label:N")
    )

    if facet:
        layered = alt.layer(diag, line_mark, points).properties(width=240, height=240)
        return layered.facet(
            facet=alt.Facet("line:N", title=None),
            columns=4,
            data=df,
        )
    return (
        (alt.layer(diag, notes, line_mark, points, data=df))
        .properties(width=480, height=360)
    )


def pit_histogram_chart(pit_df: pd.DataFrame, *, facet: bool = False) -> alt.Chart:
    """PIT histogram with a uniform-density reference and shape annotations.

    PIT = F_predicted(actual). For a calibrated kernel, PITs are uniform
    on [0, 1] so the histogram is flat at density = 1. Deviations
    diagnose the kind of miscalibration:

    - Mass piled in the **middle** ⇒ intervals too wide (over-dispersed).
    - Mass piled in **both tails** (U-shape) ⇒ intervals too tight
      (under-dispersed).
    - Mass piled on the **left** (low PIT) ⇒ actuals faster than predicted.
    - Mass piled on the **right** (high PIT) ⇒ actuals slower than predicted.

    By default produces one chart across all lines (the simplest read);
    pass ``facet=True`` to break out one panel per line.
    """
    if pit_df.empty:
        return alt.Chart(pd.DataFrame({"msg": ["no data"]})).mark_text().encode()

    df = pit_df.copy()
    df["uniform"] = 1.0
    # Width in axis units; bars need this when binning is pre-computed.
    df["bin_mid"] = (df["bin_lower"] + df["bin_upper"]) / 2

    bars = (
        alt.Chart()
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X(
                "bin_lower:Q",
                title="PIT = F_forecast(actual)  →  0 = fast, 1 = slow",
                scale=alt.Scale(domain=[0, 1]),
            ),
            x2="bin_upper:Q",
            y=alt.Y(
                "density:Q",
                title="density (1.0 = uniform / calibrated)",
            ),
            color=alt.Color(
                "line:N", title="line",
                legend=alt.Legend(orient="right") if not facet else None,
            ),
            tooltip=[
                alt.Tooltip("line:N"),
                alt.Tooltip("bin_lower:Q", title="bin start", format=".2f"),
                alt.Tooltip("bin_upper:Q", title="bin end", format=".2f"),
                alt.Tooltip("count:Q"),
                alt.Tooltip("density:Q", title="density", format=".2f"),
            ],
        )
    )

    uniform = (
        alt.Chart()
        .mark_rule(strokeDash=[4, 4], color="black", strokeWidth=1.5)
        .encode(y="uniform:Q")
    )

    # Inline annotation marks calling out the diagnostic vocabulary.
    notes = pd.DataFrame([
        {"x": 0.08, "y": 1.6, "label": "← actuals faster"},
        {"x": 0.92, "y": 1.6, "label": "actuals slower →"},
    ])
    note_marks = (
        alt.Chart(notes)
        .mark_text(color="#888", fontSize=10, fontStyle="italic")
        .encode(x="x:Q", y="y:Q", text="label:N")
    )

    if facet:
        layered = alt.layer(bars, uniform).properties(width=240, height=200)
        return layered.facet(
            facet=alt.Facet("line:N", title="Per-line PIT shape — flat ribbon = calibrated"),
            columns=4,
            data=df,
        )

    return (
        alt.layer(bars, uniform, note_marks, data=df)
        .properties(width=480, height=320)
    )


def sharpness_coverage_chart(coverage_df: pd.DataFrame) -> alt.Chart:
    """Scatter of (median sharpness, p80 coverage), one point per bucket.

    Target zone: low x, y ≈ 0.80 (tight intervals that still hit the
    mark). The chart annotates each quadrant so the maintainer can
    interpret at a glance.
    """
    if coverage_df.empty:
        return alt.Chart(pd.DataFrame({"msg": ["no data"]})).mark_text().encode()

    df = coverage_df.copy()

    reference = (
        alt.Chart(pd.DataFrame({"y": [0.8]}))
        .mark_rule(strokeDash=[4, 4], color="gray")
        .encode(y="y:Q")
    )

    points = (
        alt.Chart(df)
        .mark_circle(opacity=0.65)
        .encode(
            x=alt.X(
                "median_sharpness_s:Q",
                title="median (p80 − p50) seconds — lower is tighter / more confident",
            ),
            y=alt.Y(
                "coverage_p80:Q",
                title="empirical p80 coverage (target = 0.80)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            size=alt.Size("n_samples:Q", title="n samples"),
            color=alt.Color("line:N", title="line"),
            tooltip=[
                "line:N", "direction:N", "hour_of_day:O", "weekday:N",
                "n_samples:Q",
                alt.Tooltip("coverage_p80:Q", title="p80", format=".1%"),
                alt.Tooltip("median_sharpness_s:Q", title="sharpness (s)", format=".0f"),
            ],
        )
    )

    # Quadrant labels — positioned in screen quadrants of the chart.
    x_max = max(df["median_sharpness_s"].max(), 1.0)
    annotations = pd.DataFrame([
        {"x": x_max * 0.08, "y": 0.83, "label": "✓ target: tight & calibrated"},
        {"x": x_max * 0.08, "y": 0.55, "label": "⚠ tight but biased: overconfident"},
        {"x": x_max * 0.75, "y": 0.83, "label": "ok: loose but calibrated"},
        {"x": x_max * 0.75, "y": 0.55, "label": "⚠ loose and biased"},
    ])
    notes = (
        alt.Chart(annotations)
        .mark_text(color="#666", fontSize=10, fontStyle="italic", align="left")
        .encode(x="x:Q", y="y:Q", text="label:N")
    )

    return (reference + notes + points).properties(height=420)


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
