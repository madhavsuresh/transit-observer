"""Coverage / calibration / sharpness metrics, bucketed by corridor.

A "corridor bucket" is `(line, direction, hour_of_day, weekday|weekend)`.
Use this module to answer:

- Which buckets have ≥N samples? (corridor inventory)
- Per bucket: what fraction of actuals fell inside p80? (coverage)
- Per bucket: median(p80 − p50)? (sharpness)
- Across all samples, when we predict failure probability p, what's the
  realized failure rate? (calibration curve)

Reads from the read replica; doesn't write.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import duckdb

from .journey.quantile_distribution import (
    lognormal_quantile,
    pit_value,
    fit_lognormal_from_p50_p80,
)


@dataclass(frozen=True)
class CorridorCoverage:
    line: str
    direction_label: str
    hour_of_day: int
    weekday: bool
    n_samples: int
    coverage_p80: float
    coverage_p90: float
    median_sharpness_seconds: float
    median_p50_residual_seconds: float


@dataclass(frozen=True)
class CalibrationBin:
    predicted_lower: float
    predicted_upper: float
    n: int
    actual_failure_rate: float


@dataclass(frozen=True)
class ReliabilityPoint:
    """One point on a reliability diagram (Job A in the viz plan).

    The diagram plots ``empirical_coverage`` (y) against
    ``nominal_quantile`` (x). A perfectly-calibrated kernel produces
    points on the y=x diagonal.
    """
    line: str
    nominal_quantile: float
    empirical_coverage: float
    n: int


@dataclass(frozen=True)
class PitBin:
    """One bar of a PIT histogram.

    Uniform-looking histograms ⇒ calibrated kernel.
    U-shape ⇒ predictions too tight (under-dispersion).
    ∩-shape ⇒ predictions too wide (over-dispersion).
    Left-skew ⇒ actuals are systematically slower than predictions.
    """
    line: str
    bin_lower: float
    bin_upper: float
    count: int
    density: float        # count / (n_total * bin_width)


def corridor_coverage(
    conn: duckdb.DuckDBPyConnection,
    *,
    min_samples: int = 5,
) -> list[CorridorCoverage]:
    rows = conn.execute(
        """
        SELECT q.line, q.direction_code,
               EXTRACT(hour FROM q.leave_at)::INTEGER AS hod,
               (EXTRACT(dow FROM q.leave_at) BETWEEN 1 AND 5) AS weekday,
               COUNT(*) AS n,
               AVG(CASE WHEN o.in_p80_window THEN 1 ELSE 0 END) AS cov80,
               AVG(CASE WHEN o.in_p90_window THEN 1 ELSE 0 END) AS cov90,
               MEDIAN(q.predicted_total_p80 - q.predicted_total_p50) AS sharpness,
               MEDIAN(o.p50_residual_seconds) AS median_residual
          FROM forecast_outcomes o
          JOIN forecast_queue q USING (forecast_id)
         GROUP BY q.line, q.direction_code, hod, weekday
        HAVING COUNT(*) >= ?
         ORDER BY q.line, q.direction_code, hod, weekday
        """,
        [min_samples],
    ).fetchall()
    return [
        CorridorCoverage(
            line=line,
            direction_label=direction or "?",
            hour_of_day=hod,
            weekday=bool(weekday),
            n_samples=n,
            coverage_p80=cov80 or 0.0,
            coverage_p90=cov90 or 0.0,
            median_sharpness_seconds=sharp or 0.0,
            median_p50_residual_seconds=median_residual or 0.0,
        )
        for line, direction, hod, weekday, n, cov80, cov90, sharp, median_residual in rows
    ]


def uncovered_buckets(
    conn: duckdb.DuckDBPyConnection,
    *,
    target_samples: int = 5,
) -> list[tuple[str, str, int, bool, int]]:
    """Return (line, direction, hour_of_day, weekday, n) for buckets below target.

    Includes buckets with zero samples too, by joining against the
    cross-product of possible buckets implicit in the line catalog.
    """
    rows = conn.execute(
        """
        SELECT q.line, q.direction_code,
               EXTRACT(hour FROM q.leave_at)::INTEGER AS hod,
               (EXTRACT(dow FROM q.leave_at) BETWEEN 1 AND 5) AS weekday,
               COUNT(*) AS n
          FROM forecast_queue q
         GROUP BY q.line, q.direction_code, hod, weekday
        HAVING COUNT(*) < ?
         ORDER BY n ASC, q.line, hod
        """,
        [target_samples],
    ).fetchall()
    return [(line, direction or "?", hod, bool(weekday), n) for line, direction, hod, weekday, n in rows]


_DEFAULT_RELIABILITY_GRID: tuple[float, ...] = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
)


def _resolved_forecasts_for_calibration(
    conn: duckdb.DuckDBPyConnection,
    *,
    line: str | None,
    min_truth_confidence: float,
) -> list[tuple[str, float, float, float, float]]:
    """Pull (line, p50, p80, p90, actual_total_seconds) for resolved rows.

    Filters to ``status='resolved'`` and ``truth_confidence >= threshold``
    so that headline calibration metrics only count cleanly-bracketed
    outcomes (matches the corpus_summary policy).
    """
    sql = """
        SELECT q.line,
               q.predicted_total_p50,
               q.predicted_total_p80,
               q.predicted_total_p90,
               o.actual_total_seconds
          FROM forecast_outcomes o
          JOIN forecast_queue q USING (forecast_id)
         WHERE q.status = 'resolved'
           AND COALESCE(o.truth_confidence, 0) >= ?
           AND q.predicted_total_p50 > 0
           AND q.predicted_total_p80 > q.predicted_total_p50
           AND o.actual_total_seconds IS NOT NULL
    """
    params: list = [min_truth_confidence]
    if line is not None:
        sql += " AND q.line = ?"
        params.append(line)
    return conn.execute(sql, params).fetchall()


def reliability_curve(
    conn: duckdb.DuckDBPyConnection,
    *,
    line: str | None = None,
    quantile_grid: tuple[float, ...] = _DEFAULT_RELIABILITY_GRID,
    min_truth_confidence: float = 0.5,
    min_samples: int = 30,
) -> list[ReliabilityPoint]:
    """For each (line, q), empirical fraction with actual ≤ fitted q-quantile.

    Fits a log-normal to each forecast's (p50, p80) and asks "did the
    actual outcome fall below the q-th percentile of that fit?" — then
    averages those indicators per (line, q). Perfect calibration plots
    on the y=x diagonal.

    Lines with fewer than ``min_samples`` resolved forecasts are
    omitted; their reliability curve would be too noisy to read.
    """
    rows = _resolved_forecasts_for_calibration(
        conn, line=line, min_truth_confidence=min_truth_confidence,
    )
    by_line: dict[str, list[tuple[float, float, float, float]]] = {}
    for ln, p50, p80, p90, actual in rows:
        by_line.setdefault(ln, []).append((p50, p80, p90, actual))

    out: list[ReliabilityPoint] = []
    for ln, items in by_line.items():
        if len(items) < min_samples:
            continue
        for q in quantile_grid:
            hits = 0
            n = 0
            for p50, p80, _p90, actual in items:
                try:
                    mu, sigma = fit_lognormal_from_p50_p80(p50, p80)
                except ValueError:
                    continue
                threshold = lognormal_quantile(q, mu, sigma)
                if actual <= threshold:
                    hits += 1
                n += 1
            if n == 0:
                continue
            out.append(ReliabilityPoint(
                line=ln,
                nominal_quantile=q,
                empirical_coverage=hits / n,
                n=n,
            ))
    out.sort(key=lambda p: (p.line, p.nominal_quantile))
    return out


def pit_histogram(
    conn: duckdb.DuckDBPyConnection,
    *,
    line: str | None = None,
    n_bins: int = 20,
    min_truth_confidence: float = 0.5,
    min_samples: int = 30,
) -> list[PitBin]:
    """Binned PIT values per line.

    Probability Integral Transform: PIT_i = F_predicted(actual_i). If
    the kernel is calibrated, the PIT values are uniformly distributed
    on [0, 1]. Deviations diagnose the kind of miscalibration (see the
    PitBin docstring).
    """
    rows = _resolved_forecasts_for_calibration(
        conn, line=line, min_truth_confidence=min_truth_confidence,
    )
    by_line: dict[str, list[float]] = {}
    for ln, p50, p80, _p90, actual in rows:
        pit = pit_value(actual, p50, p80)
        if not math.isfinite(pit):
            continue
        by_line.setdefault(ln, []).append(pit)

    bin_width = 1.0 / n_bins
    out: list[PitBin] = []
    for ln, pits in by_line.items():
        if len(pits) < min_samples:
            continue
        counts = [0] * n_bins
        for pit in pits:
            idx = min(int(pit / bin_width), n_bins - 1)
            counts[idx] += 1
        total = sum(counts)
        for i in range(n_bins):
            lower = i * bin_width
            upper = lower + bin_width
            density = counts[i] / total / bin_width if total else 0.0
            out.append(PitBin(
                line=ln,
                bin_lower=lower,
                bin_upper=upper,
                count=counts[i],
                density=density,
            ))
    out.sort(key=lambda b: (b.line, b.bin_lower))
    return out


@dataclass(frozen=True)
class HistoricalPrediction:
    """Empirical (p50, p80, p90) for one OD pair from past resolved forecasts.

    Used by the dashboard as a fallback when the live predictor lacks
    data. Each quantile is computed via nearest-rank from the sample of
    actual_total_seconds values.
    """
    n_samples: int
    p50_seconds: float
    p80_seconds: float
    p90_seconds: float
    oldest_at: datetime | None
    newest_at: datetime | None


def historical_prediction(
    conn: duckdb.DuckDBPyConnection,
    *,
    mode: str,
    line: str,
    boarding_int_id: int = 0,
    boarding_text_id: str | None = None,
    alighting_int_id: int = 0,
    alighting_text_id: str | None = None,
    limit: int = 200,
) -> HistoricalPrediction | None:
    """Empirical quantiles from past resolved forecasts for this OD pair.

    Returns None when fewer than 5 resolved samples exist (anything less
    gives meaningless quantiles).
    """
    rows = conn.execute(
        """
        SELECT o.actual_total_seconds, o.resolved_at
          FROM forecast_outcomes o
          JOIN forecast_queue q USING (forecast_id)
         WHERE q.mode = ? AND q.line = ?
           AND q.boarding_map_id = ? AND q.alighting_map_id = ?
           AND COALESCE(q.boarding_text_id, '') = COALESCE(?, '')
           AND COALESCE(q.alighting_text_id, '') = COALESCE(?, '')
           AND q.status = 'resolved'
           AND o.actual_total_seconds IS NOT NULL
         ORDER BY o.resolved_at DESC
         LIMIT ?
        """,
        [mode, line, boarding_int_id, alighting_int_id,
         boarding_text_id, alighting_text_id, limit],
    ).fetchall()
    if len(rows) < 5:
        return None
    actuals = sorted(r[0] for r in rows)
    resolved_ats = [r[1] for r in rows if r[1] is not None]
    n = len(actuals)
    return HistoricalPrediction(
        n_samples=n,
        p50_seconds=_nearest_rank(actuals, 0.5),
        p80_seconds=_nearest_rank(actuals, 0.8),
        p90_seconds=_nearest_rank(actuals, 0.9),
        oldest_at=min(resolved_ats) if resolved_ats else None,
        newest_at=max(resolved_ats) if resolved_ats else None,
    )


def _nearest_rank(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    rank = max(1, math.ceil(p * n))
    return sorted_values[min(rank, n) - 1]


@dataclass(frozen=True)
class LiveDataDiagnostic:
    """What the live predictor sees when it tries to predict a corridor.

    Distinguishes "no raw rows at all" (collector hasn't covered this
    stop recently) from "rows exist but direction filter dropped them"
    (a coverage problem with the direction filter).
    """
    mode: str
    raw_rows_in_window: int        # arrivals at the boarding stop in the live window
    future_rows: int               # subset where arrival_at >= now
    last_raw_polled_at: datetime | None


def live_data_diagnostic(
    conn: duckdb.DuckDBPyConnection,
    *,
    mode: str,
    line: str,
    boarding_int_id: int = 0,
    boarding_text_id: str | None = None,
    now: datetime,
    window_minutes: float = 30.0,
) -> LiveDataDiagnostic:
    """Count raw-feed rows at the boarding stop in the live window.

    For L mode we count `train_arrivals_raw`; for bus, `bus_predictions_raw`;
    for metra, `metra_arrivals_raw`; for intercampus, `intercampus_arrivals_raw`.
    A zero count means the collector hasn't seen this stop recently —
    nothing the predictor can do with that.
    """
    from datetime import timedelta as _td
    cutoff = now - _td(minutes=5)
    horizon = now + _td(minutes=window_minutes)
    raw, future, last_at = 0, 0, None
    if mode == "L":
        row = conn.execute(
            """
            SELECT COUNT(*),
                   COUNT(*) FILTER (WHERE arrival_at >= ?),
                   MAX(polled_at)
              FROM train_arrivals_raw
             WHERE line = ? AND map_id = ?
               AND polled_at >= ? AND arrival_at <= ?
            """,
            [now, line, boarding_int_id, cutoff, horizon],
        ).fetchone()
        raw, future, last_at = row or (0, 0, None)
    elif mode == "bus":
        stop_id = boarding_int_id or (int(boarding_text_id) if boarding_text_id else 0)
        row = conn.execute(
            """
            SELECT COUNT(*),
                   COUNT(*) FILTER (WHERE arrival_at >= ?),
                   MAX(polled_at)
              FROM bus_predictions_raw
             WHERE route = ? AND stop_id = ?
               AND polled_at >= ? AND arrival_at <= ?
            """,
            [now, line, stop_id, cutoff, horizon],
        ).fetchone()
        raw, future, last_at = row or (0, 0, None)
    elif mode == "metra":
        row = conn.execute(
            """
            SELECT COUNT(*),
                   COUNT(*) FILTER (WHERE predicted_at >= ?),
                   MAX(polled_at)
              FROM metra_arrivals_raw
             WHERE route_id = ? AND station_id = ?
               AND polled_at >= ? AND predicted_at <= ?
            """,
            [now, line, boarding_text_id or "", cutoff, horizon],
        ).fetchone()
        raw, future, last_at = row or (0, 0, None)
    elif mode == "intercampus":
        row = conn.execute(
            """
            SELECT COUNT(*),
                   COUNT(*) FILTER (WHERE predicted_at >= ?),
                   MAX(polled_at)
              FROM intercampus_arrivals_raw
             WHERE stop_id = ?
               AND polled_at >= ? AND predicted_at <= ?
            """,
            [now, boarding_text_id or "", cutoff, horizon],
        ).fetchone()
        raw, future, last_at = row or (0, 0, None)
    return LiveDataDiagnostic(
        mode=mode,
        raw_rows_in_window=int(raw or 0),
        future_rows=int(future or 0),
        last_raw_polled_at=last_at,
    )


@dataclass(frozen=True)
class LineSampleCount:
    """Per-line breakdown of how many resolved forecasts back the calibration metrics."""
    line: str
    mode: str
    n_resolved: int
    n_resolved_high_conf: int


def per_line_resolved_counts(
    conn: duckdb.DuckDBPyConnection,
    *,
    min_truth_confidence: float = 0.5,
) -> list[LineSampleCount]:
    """How many resolved outcomes exist per (mode, line). Lets the dashboard
    explain which lines are present in the PIT / reliability charts."""
    rows = conn.execute(
        """
        SELECT q.mode, q.line,
               COUNT(*) AS n_resolved,
               COUNT(*) FILTER (WHERE COALESCE(o.truth_confidence, 0) >= ?) AS n_hi
          FROM forecast_outcomes o
          JOIN forecast_queue q USING (forecast_id)
         WHERE q.status = 'resolved'
           AND q.predicted_total_p50 > 0
           AND q.predicted_total_p80 > q.predicted_total_p50
           AND o.actual_total_seconds IS NOT NULL
         GROUP BY q.mode, q.line
         ORDER BY n_resolved DESC
        """,
        [min_truth_confidence],
    ).fetchall()
    return [
        LineSampleCount(line=ln, mode=mode, n_resolved=int(nr), n_resolved_high_conf=int(nhi))
        for mode, ln, nr, nhi in rows
    ]


def reliability_curve_aggregated(
    conn: duckdb.DuckDBPyConnection,
    *,
    quantile_grid: tuple[float, ...] = _DEFAULT_RELIABILITY_GRID,
    min_truth_confidence: float = 0.5,
) -> list[ReliabilityPoint]:
    """Reliability curve across all lines combined (line='ALL').

    Less informative per-line but always has more samples — useful when
    individual lines are below threshold.
    """
    rows = _resolved_forecasts_for_calibration(
        conn, line=None, min_truth_confidence=min_truth_confidence,
    )
    if not rows:
        return []
    out: list[ReliabilityPoint] = []
    for q in quantile_grid:
        hits, n = 0, 0
        for _ln, p50, p80, _p90, actual in rows:
            try:
                mu, sigma = fit_lognormal_from_p50_p80(p50, p80)
            except ValueError:
                continue
            if actual <= lognormal_quantile(q, mu, sigma):
                hits += 1
            n += 1
        if n:
            out.append(ReliabilityPoint(
                line="ALL", nominal_quantile=q, empirical_coverage=hits / n, n=n,
            ))
    return out


def pit_histogram_aggregated(
    conn: duckdb.DuckDBPyConnection,
    *,
    n_bins: int = 20,
    min_truth_confidence: float = 0.5,
) -> list[PitBin]:
    """PIT histogram across all lines combined (line='ALL')."""
    rows = _resolved_forecasts_for_calibration(
        conn, line=None, min_truth_confidence=min_truth_confidence,
    )
    pits: list[float] = []
    for _ln, p50, p80, _p90, actual in rows:
        pit = pit_value(actual, p50, p80)
        if math.isfinite(pit):
            pits.append(pit)
    if not pits:
        return []
    bin_width = 1.0 / n_bins
    counts = [0] * n_bins
    for pit in pits:
        idx = min(int(pit / bin_width), n_bins - 1)
        counts[idx] += 1
    total = sum(counts)
    return [
        PitBin(
            line="ALL",
            bin_lower=i * bin_width,
            bin_upper=(i + 1) * bin_width,
            count=counts[i],
            density=counts[i] / total / bin_width if total else 0.0,
        )
        for i in range(n_bins)
    ]


def diagnose_pit_shape(bins: list[PitBin]) -> str:
    """Return a one-line plain-language interpretation of a PIT histogram.

    Looks at the first and last 20% of bins vs the middle 60% to label
    one of: calibrated / U-shape (too tight) / hump (too wide) /
    left-skew (too slow) / right-skew (too fast).
    """
    if not bins:
        return "no data"
    by_line: dict[str, list[PitBin]] = {}
    for b in bins:
        by_line.setdefault(b.line, []).append(b)
    out_parts: list[str] = []
    for ln, bs in by_line.items():
        n = sum(b.count for b in bs)
        if n == 0:
            continue
        bs_sorted = sorted(bs, key=lambda b: b.bin_lower)
        nb = len(bs_sorted)
        cutoff_low = max(1, nb // 5)
        cutoff_high = nb - cutoff_low
        left = sum(b.count for b in bs_sorted[:cutoff_low])
        middle = sum(b.count for b in bs_sorted[cutoff_low:cutoff_high])
        right = sum(b.count for b in bs_sorted[cutoff_high:])
        left_share = left / n
        middle_share = middle / n
        right_share = right / n
        expected_tail = cutoff_low / nb  # what each tail share would be under uniform
        expected_mid = (cutoff_high - cutoff_low) / nb
        # Heuristics with a 20% slack around uniform expectation.
        if abs(left_share - expected_tail) < 0.05 and abs(right_share - expected_tail) < 0.05:
            label = "calibrated (flat)"
        elif left_share + right_share > expected_tail * 2 * 1.4:
            label = "U-shape — intervals too tight (under-dispersed)"
        elif middle_share > expected_mid * 1.4:
            label = "∩-shape — intervals too wide (over-dispersed)"
        elif right_share > expected_tail * 1.6 and left_share < expected_tail:
            label = "right-skew — actuals slower than predicted"
        elif left_share > expected_tail * 1.6 and right_share < expected_tail:
            label = "left-skew — actuals faster than predicted"
        else:
            label = "mixed shape"
        out_parts.append(f"{ln}: {label} (n={n})")
    return " · ".join(out_parts)


def calibration_bins(
    conn: duckdb.DuckDBPyConnection,
    *,
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Calibration of `predicted_failure_prob` vs actual unresolved/failed.

    With the v1 single-leg simulator, `predicted_failure_prob` is 0.0 for
    successful predictions, so this is mostly informational until the
    multi-leg phase. We still produce the bin shape so the dashboard can
    plot it once it's meaningful.
    """
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        row = conn.execute(
            """
            SELECT COUNT(*) AS n,
                   AVG(CASE WHEN o.failed OR o.actual_total_seconds > q.predicted_total_p90 THEN 1 ELSE 0 END) AS actual_rate
              FROM forecast_outcomes o
              JOIN forecast_queue q USING (forecast_id)
             WHERE q.predicted_failure_prob >= ?
               AND q.predicted_failure_prob < ?
            """,
            [lower, upper],
        ).fetchone()
        n, actual = row or (0, 0.0)
        bins.append(CalibrationBin(predicted_lower=lower, predicted_upper=upper, n=n or 0, actual_failure_rate=actual or 0.0))
    return bins


@dataclass(frozen=True)
class Status:
    raw_arrivals_count: int
    runs_observed_count: int
    positions_count: int
    bus_predictions_count: int
    metra_arrivals_count: int
    intercampus_arrivals_count: int
    forecasts_pending: int
    forecasts_resolved: int
    forecasts_unresolvable: int
    latest_poll: datetime | None
    oldest_pending: datetime | None
    overall_p80_coverage: float | None


def _has_column(conn: duckdb.DuckDBPyConnection, *, table: str, column: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM pragma_table_info(?) WHERE name = ? LIMIT 1",
        [table, column],
    ).fetchone()
    return bool(row)


@dataclass(frozen=True)
class CorpusRow:
    forecast_id: str
    leave_at: datetime
    predictor_version: str | None
    direction_code: str | None
    predicted_total_p50: float
    predicted_total_p80: float
    predicted_total_p90: float
    actual_total_seconds: float | None
    in_p80_window: bool | None
    p50_residual_seconds: float | None
    truth_confidence: float | None
    status: str


def corpus_corridor_rows(
    conn: duckdb.DuckDBPyConnection,
    *,
    corridor_id: str,
    limit: int = 50,
) -> list[CorpusRow]:
    """Recent forecasts for one corridor, joined with outcomes if resolved."""
    if not _has_column(conn, table="forecast_queue", column="corridor_id"):
        return []
    rows = conn.execute(
        """
        SELECT q.forecast_id, q.leave_at, q.predictor_version, q.direction_code,
               q.predicted_total_p50, q.predicted_total_p80, q.predicted_total_p90,
               o.actual_total_seconds, o.in_p80_window, o.p50_residual_seconds,
               o.truth_confidence, q.status
          FROM forecast_queue q
          LEFT JOIN forecast_outcomes o USING (forecast_id)
         WHERE q.corridor_id = ?
         ORDER BY q.leave_at DESC
         LIMIT ?
        """,
        [corridor_id, limit],
    ).fetchall()
    return [
        CorpusRow(
            forecast_id=fid, leave_at=leave_at, predictor_version=pv,
            direction_code=dir_code,
            predicted_total_p50=p50, predicted_total_p80=p80, predicted_total_p90=p90,
            actual_total_seconds=actual, in_p80_window=in_p80,
            p50_residual_seconds=resid, truth_confidence=trustconf, status=stat,
        )
        for (
            fid, leave_at, pv, dir_code, p50, p80, p90,
            actual, in_p80, resid, trustconf, stat,
        ) in rows
    ]


@dataclass(frozen=True)
class CorpusCorridorSummary:
    corridor_id: str
    mode: str
    line: str
    direction: str
    origin_label: str
    destination_label: str
    n_predictions: int
    n_resolved: int
    n_unresolvable: int
    coverage_p80: float | None
    median_p50_residual_seconds: float | None
    median_truth_confidence: float | None
    last_predicted_at: datetime | None


def corpus_summary(
    conn: duckdb.DuckDBPyConnection,
    *,
    high_confidence_only: bool = False,
    confidence_threshold: float = 0.5,
) -> list[CorpusCorridorSummary]:
    """Per-corridor coverage + residual stats, joining corridors -> outcomes.

    ``high_confidence_only`` excludes outcomes with ``truth_confidence``
    below ``confidence_threshold`` (spec: low-confidence truths should be
    excluded from headline metrics).
    """
    # Reader may be hitting an older DB whose schema predates the
    # corridors table; return empty rather than blowing up the dashboard.
    has_corridors = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'corridors'"
    ).fetchone()
    if not has_corridors or not has_corridors[0]:
        return []

    threshold = confidence_threshold if high_confidence_only else -1.0
    rows = conn.execute(
        """
        SELECT c.corridor_id, c.mode, c.line, c.direction,
               c.origin_label, c.destination_label,
               COUNT(q.forecast_id) AS n_predictions,
               SUM(CASE WHEN q.status = 'resolved' THEN 1 ELSE 0 END) AS n_resolved,
               SUM(CASE WHEN q.status = 'unresolvable' THEN 1 ELSE 0 END) AS n_unresolvable,
               AVG(CASE WHEN o.in_p80_window IS NOT NULL
                            AND COALESCE(o.truth_confidence, 0) >= ?
                       THEN (CASE WHEN o.in_p80_window THEN 1.0 ELSE 0.0 END) END) AS cov80,
               MEDIAN(CASE WHEN COALESCE(o.truth_confidence, 0) >= ?
                            THEN o.p50_residual_seconds END) AS median_p50_residual,
               MEDIAN(o.truth_confidence) AS median_tc,
               c.last_predicted_at
          FROM corridors c
          LEFT JOIN forecast_queue q ON q.corridor_id = c.corridor_id
          LEFT JOIN forecast_outcomes o ON o.forecast_id = q.forecast_id
         GROUP BY c.corridor_id, c.mode, c.line, c.direction,
                  c.origin_label, c.destination_label, c.last_predicted_at,
                  c.priority
         ORDER BY c.priority ASC, c.corridor_id
        """,
        [threshold, threshold],
    ).fetchall()
    return [
        CorpusCorridorSummary(
            corridor_id=cid, mode=mode, line=line, direction=direction,
            origin_label=origin, destination_label=destination,
            n_predictions=n_pred or 0,
            n_resolved=n_resolved or 0,
            n_unresolvable=n_unresolvable or 0,
            coverage_p80=cov80,
            median_p50_residual_seconds=median_resid,
            median_truth_confidence=median_tc,
            last_predicted_at=last_at,
        )
        for (
            cid, mode, line, direction, origin, destination,
            n_pred, n_resolved, n_unresolvable,
            cov80, median_resid, median_tc, last_at,
        ) in rows
    ]


def status(conn: duckdb.DuckDBPyConnection) -> Status:
    raw = conn.execute("SELECT COUNT(*), MAX(polled_at) FROM train_arrivals_raw").fetchone() or (0, None)
    runs = conn.execute("SELECT COUNT(*) FROM train_runs_observed").fetchone() or (0,)
    positions = conn.execute("SELECT COUNT(*) FROM train_positions_raw").fetchone() or (0,)
    bus = conn.execute("SELECT COUNT(*) FROM bus_predictions_raw").fetchone() or (0,)
    metra = conn.execute("SELECT COUNT(*) FROM metra_arrivals_raw").fetchone() or (0,)
    intercampus = conn.execute("SELECT COUNT(*) FROM intercampus_arrivals_raw").fetchone() or (0,)
    queue = conn.execute(
        """
        SELECT
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END),
            SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END),
            SUM(CASE WHEN status = 'unresolvable' THEN 1 ELSE 0 END),
            MIN(CASE WHEN status = 'pending' THEN enqueued_at END)
          FROM forecast_queue
        """
    ).fetchone() or (0, 0, 0, None)
    coverage = conn.execute(
        "SELECT AVG(CASE WHEN in_p80_window THEN 1 ELSE 0 END) FROM forecast_outcomes"
    ).fetchone()
    return Status(
        raw_arrivals_count=raw[0] or 0,
        runs_observed_count=runs[0] or 0,
        positions_count=positions[0] or 0,
        bus_predictions_count=bus[0] or 0,
        metra_arrivals_count=metra[0] or 0,
        intercampus_arrivals_count=intercampus[0] or 0,
        forecasts_pending=queue[0] or 0,
        forecasts_resolved=queue[1] or 0,
        forecasts_unresolvable=queue[2] or 0,
        latest_poll=raw[1],
        oldest_pending=queue[3],
        overall_p80_coverage=(coverage[0] if coverage else None),
    )
