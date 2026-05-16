"""Chart constructors should produce valid Altair specs without DB access."""

from __future__ import annotations

import altair as alt
import pandas as pd

from transit_observer.viz import (
    coverage_heatmap_chart,
    pit_histogram_chart,
    quantile_dotplot_chart,
    reliability_diagram_chart,
    sharpness_coverage_chart,
)


def _coverage_fixture() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "line": "Red", "direction": "1", "hour_of_day": 8, "weekday": "weekday",
            "n_samples": 25, "coverage_p80": 0.82, "coverage_p90": 0.91,
            "median_sharpness_s": 130.0, "median_p50_residual_s": -20.0,
        },
        {
            "line": "Red", "direction": "5", "hour_of_day": 17, "weekday": "weekday",
            "n_samples": 30, "coverage_p80": 0.65, "coverage_p90": 0.78,
            "median_sharpness_s": 80.0, "median_p50_residual_s": 50.0,
        },
        {
            "line": "Blue", "direction": "1", "hour_of_day": 8, "weekday": "weekday",
            "n_samples": 12, "coverage_p80": 0.95, "coverage_p90": 0.98,
            "median_sharpness_s": 240.0, "median_p50_residual_s": -10.0,
        },
    ])


def _reliability_fixture() -> pd.DataFrame:
    rows = []
    for line in ("Red", "Blue"):
        for q in (0.1, 0.3, 0.5, 0.7, 0.9):
            rows.append({"line": line, "nominal_quantile": q,
                         "empirical_coverage": q - 0.05, "n": 100})
    return pd.DataFrame(rows)


def _pit_fixture() -> pd.DataFrame:
    rows = []
    for line in ("Red", "Blue"):
        for i in range(10):
            lower = i / 10
            rows.append({
                "line": line, "bin_lower": lower, "bin_upper": lower + 0.1,
                "count": 50, "density": 1.0,
            })
    return pd.DataFrame(rows)


def test_coverage_heatmap_returns_chart():
    chart = coverage_heatmap_chart(_coverage_fixture())
    spec = chart.to_dict()
    assert "spec" in spec or "facet" in spec or "mark" in spec


def test_coverage_heatmap_handles_empty():
    chart = coverage_heatmap_chart(pd.DataFrame())
    spec = chart.to_dict()
    assert spec  # produces *some* valid spec, not a crash


def test_reliability_diagram_default_is_overlaid():
    """Default mode overlays lines into one chart for easy comparison."""
    chart = reliability_diagram_chart(_reliability_fixture())
    spec = chart.to_dict()
    # Default is not faceted; should be a layered chart.
    assert "facet" not in spec


def test_reliability_diagram_facet_mode_produces_facets():
    chart = reliability_diagram_chart(_reliability_fixture(), facet=True)
    spec = chart.to_dict()
    assert "facet" in spec


def test_reliability_diagram_handles_empty():
    chart = reliability_diagram_chart(pd.DataFrame())
    assert chart.to_dict() is not None


def test_pit_histogram_default_is_overlaid():
    chart = pit_histogram_chart(_pit_fixture())
    spec = chart.to_dict()
    # Default is not faceted; should be one overlaid chart.
    assert "facet" not in spec


def test_pit_histogram_facet_mode_produces_facets():
    chart = pit_histogram_chart(_pit_fixture(), facet=True)
    spec = chart.to_dict()
    assert "facet" in spec


def test_pit_histogram_handles_empty():
    chart = pit_histogram_chart(pd.DataFrame())
    assert chart.to_dict() is not None


def test_pit_histogram_uniform_reference_present():
    """Uniform density (=1.0) reference line should be in the spec."""
    chart = pit_histogram_chart(_pit_fixture())
    spec_str = str(chart.to_dict())
    assert "uniform" in spec_str.lower() or "1.0" in spec_str


def test_sharpness_coverage_returns_chart():
    chart = sharpness_coverage_chart(_coverage_fixture())
    spec = chart.to_dict()
    assert spec
    # Reference rule at y=0.8 should be present in serialised spec.
    assert "0.8" in str(spec)


def test_sharpness_coverage_handles_empty():
    chart = sharpness_coverage_chart(pd.DataFrame())
    assert chart.to_dict() is not None


def test_quantile_dotplot_produces_50_dots_by_default():
    """Per CHI 2018, 50 is the right default."""
    chart = quantile_dotplot_chart(p50=600.0, p80=900.0)
    assert len(chart.data) == 50


def test_quantile_dotplot_returns_chart():
    chart = quantile_dotplot_chart(p50=600.0, p80=900.0)
    assert isinstance(chart, alt.Chart)


def test_quantile_dotplot_minutes_axis():
    chart = quantile_dotplot_chart(p50=600.0, p80=900.0)
    spec = chart.to_dict()
    # x-axis title mentions minutes.
    assert "minutes" in str(spec).lower()


def test_quantile_dotplot_n_parameter_changes_dot_count():
    chart = quantile_dotplot_chart(p50=600.0, p80=900.0, n=20)
    assert len(chart.data) == 20


def test_quantile_dotplot_dots_are_wilkinson_stacked():
    """Adjacent dots should occupy increasing y_row, not all at y=1."""
    chart = quantile_dotplot_chart(p50=600.0, p80=900.0, n=50)
    max_y = chart.data["y_row"].max()
    assert max_y > 1, "expected Wilkinson stacking, got all dots on y=1"
