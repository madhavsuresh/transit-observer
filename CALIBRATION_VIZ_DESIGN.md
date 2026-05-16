# Calibration & forecast-display design

Design document for the `Calibration & forecast displays` section of the
transit-observer dashboard ([src/transit_observer/dashboard.py](src/transit_observer/dashboard.py))
and its supporting modules. Captures *why* each choice was made, with
citations.

## Why this exists

transit-observer produces probabilistic forecasts of trip duration —
stored as `(p50, p80, p90)` triples in `forecast_queue` — and resolves
them against observed outcomes in `forecast_outcomes`. Before this
work, the dashboard showed tables and a residual scatter that answered
"how big is the gap?" but not the two questions that actually matter:

1. **Is the kernel calibrated?** Across all forecasts, when we say "80%
   chance the trip takes ≤ X seconds," does it actually happen 80% of
   the time?
2. **Would a user understand the forecast?** If the Cozy Fox app showed
   the predicted distribution to a rider, could they make a good
   decision from it?

These are two different problems with two different literatures. We
adopted the standards from each.

## Two audiences, two chart families

The single most-load-bearing decision: **the maintainer view and the
user view use different chart types and should not be conflated.**
Padilla, Kay, and Hullman's
[*Uncertainty Visualization* handbook chapter (2022)](http://space.ucmerced.edu/Downloads/publications/Uncertainty_Visualization_Padilla_Kay_Hullman_2022.pdf)
emphasizes this: diagnostic charts (calibration, reliability) are for
people who already understand probabilistic forecasts; user-facing
charts must support frequency-counting cognition, which is how
non-statisticians reason about probability.

| Audience | Question they ask | Chart we use |
|---|---|---|
| Maintainer | "Is the model calibrated overall?" | Reliability diagram |
| Maintainer | "What *shape* is the miscalibration?" | PIT histogram |
| Maintainer | "*Which* buckets are broken?" | Faceted coverage heatmap |
| Maintainer | "Are we sharp where we're accurate?" | Sharpness ↔ coverage scatter |
| End user | "When will *my* train come?" | Quantile dotplot (50 dots) |

The first four go in maintainer tabs. The fifth is the product
surface — the same chart shape Cozy Fox should render in-app.

## Choice 1: Quantile dotplots for end-user displays

### What we chose

50 dots, each representing 2% probability mass, placed at evenly-spaced
quantiles of the predicted distribution and Wilkinson-stacked
vertically when adjacent. Built in
[src/transit_observer/viz.py:quantile_dotplot_chart](src/transit_observer/viz.py),
backed by
[journey/quantile_distribution.py:quantile_dotplot_positions](src/transit_observer/journey/quantile_distribution.py).

### Why

Two CHI papers settled this. Kay, Kola, Hullman, and Munson's
[*When (ish) is My Bus?* (CHI 2016)](https://dl.acm.org/doi/10.1145/2858036.2858558)
identified the design space and proposed quantile dotplots specifically
for mobile transit-uncertainty contexts. Fernandes, Walls, Munson,
Hullman, and Kay's follow-up
[*Uncertainty Displays Using Quantile Dotplots or CDFs Improve Transit
Decision-Making* (CHI 2018, Best Paper Honourable Mention)](https://idl.uw.edu/papers/uncertainty-bus)
ran the head-to-head study against textual probabilities, error bars,
and density plots. The 50-dot quantile dotplot produced decisions at
**97% of optimal expected payoff** with **within-subject standard
deviation of 3 percentage points** — beating every alternative they
tested.

### Why dots specifically, not density or error bars

Frequency framing. Per
[Hullman, Resnick, and Adar — *Hypothetical Outcome Plots Outperform
Error Bars and Violin Plots*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0142444),
the cognitive load of mapping a continuous-area chart back to a
probability is high; people consistently misread error bars as hard
bounds and density curves as discrete probability ridges. Dots are
countable: "what fraction will arrive by 8 minutes?" reduces to "count
the dots ≤ 8 min and divide by 50."

### Why 50, not 20 or 100

Fernandes et al. tested both 20 and 50; 50 wins on both accuracy and
consistency. Above 50 the per-dot probability becomes too granular to
count reliably on a phone screen. The dot-count knob is exposed
through the `n` parameter so the iOS port can tune for screen size if
needed.

### Why we did *not* use the NYT-style needle

[FlowingData's "Needle of uncertainty"](https://flowingdata.com/2018/03/14/needle-of-uncertainty/)
documents the NYT Upshot's election-night gauge that jittered between
the 25th and 75th percentiles of simulated outcomes.
[Fast Company called it "the most hated data visualization in
politics"](https://www.fastcompany.com/90459366/the-most-hated-data-visualization-in-politics-is-back-to-spike-your-blood-pressure):
the live randomness raised anxiety without improving understanding.
That failure mode applies directly to a transit context: a jittering
arrival-time gauge would unsettle users without communicating useful
information.

## Choice 2: Log-normal at viz time, no schema change

### What we chose

Fit a 2-parameter log-normal to each stored `(p50, p80)` pair at viz
time, in closed form (`mu = log(p50)`; `sigma = (log(p80) − mu) /
Φ⁻¹(0.8)`). Use the fitted distribution for the reliability diagram,
PIT histogram, and the quantile dotplot. Use the stored `p90` as a
free fit-check (its deviation from the fit's `p90` shows when the
kernel's tail is heavier or lighter than log-normal).

Implementation: [journey/quantile_distribution.py:fit_lognormal_from_p50_p80](src/transit_observer/journey/quantile_distribution.py).

### Why not store extra quantile booleans

Earlier drafts proposed adding `in_p10_window`, `in_p25_window`, etc.
columns to `forecast_outcomes`. Rejected for three reasons:

1. Three stored quantiles overdetermine a 2-parameter family, so all
   five (or any number) of extra-percentile-coverage booleans are
   redundantly derivable.
2. No migration, no backfill, no risk of stored values going stale if
   the fitting policy changes.
3. Streamlit's `@st.cache_data` makes viz-time computation cheap; the
   cost is dominated by the SQL pull, not the fit.

### Why log-normal

Travel-time distributions are positive and right-skewed; log-normal is
the textbook fit. Two practical justifications:

- **Positive support**: a Gaussian fitted to `(p50, p80)` would assign
  non-zero probability to negative trip durations, which a PIT
  histogram would then report as miscalibration that isn't really the
  kernel's fault.
- **Cozy Fox parity**: the Swift `StopArrivalProcess` kernel and the
  `TimeDistributionSummary` port already shape travel-time
  distributions as positive, right-skewed quantities. Log-normal makes
  the Python diagnostic layer's assumptions consistent with the
  underlying model.

If the kernel's true output isn't log-normal, the PIT histogram itself
will reveal that — bimodality or skew shows up as histogram shape. The
`p90` fit-check residual exposed by
[`p90_fit_residual()`](src/transit_observer/journey/quantile_distribution.py)
gives an explicit numeric handle on how much the kernel deviates from
the log-normal assumption.

### Why fit from (p50, p80), not (p50, p90)

`p80` is closer to the median and therefore less influenced by tail
noise; the fit is more stable. `p90` then becomes a visible
fit-quality check on the reliability diagram rather than being baked
into the fit.

## Choice 3: Reliability diagram + PIT histogram for calibration

### What we chose

Two complementary diagnostics, both filtered to resolved forecasts
with `truth_confidence >= 0.5` (matching the corpus headline-metrics
policy).

- **Reliability diagram**: for each nominal quantile `q ∈ {0.1, 0.2,
  …, 0.9}`, count the fraction of actuals that fell ≤ the fitted
  q-quantile. Plot against `y = x`. Implemented in
  [metrics.py:reliability_curve](src/transit_observer/metrics.py) and
  [viz.py:reliability_diagram_chart](src/transit_observer/viz.py).
- **PIT histogram**: for each resolved forecast, compute
  `PIT = F_predicted(actual)`. Histogram those values per line.
  Implemented in
  [metrics.py:pit_histogram](src/transit_observer/metrics.py) and
  [viz.py:pit_histogram_chart](src/transit_observer/viz.py).

### Why these two together

Reliability diagrams give the *overall* calibration story; PIT
histograms diagnose the *shape* of the miscalibration. Both are
standard in forecast verification (weather, climate, ML), and they
complement each other — the reliability diagram tells you "you're
off" and the PIT histogram tells you "this is *how* you're off."

The reliability-diagram convention is described in
[Dimitriadis, Gneiting, and Jordan — *Stable Reliability Diagrams for
Probabilistic Classifiers* (PNAS 2021)](https://www.pnas.org/doi/10.1073/pnas.2016191118).
We use the classical binning-and-counting form rather than the CORP
(isotonic-regression) form to keep the dashboard scipy-free; CORP is
listed as a possible upgrade in the deferred section.

PIT histograms come from
[Gneiting, Balabdaoui, and Raftery, *Probabilistic forecasts,
calibration and sharpness* (JRSS B 2007)](https://doi.org/10.1111/j.1467-9868.2007.00587.x).
The diagnostic vocabulary is well established and surfaced as a
caption in the dashboard tab:

| Histogram shape | Diagnosis |
|---|---|
| Flat / uniform | Calibrated |
| U-shape (peaks at 0 and 1) | Underdispersion — intervals too tight |
| ∩-shape (peak in middle) | Overdispersion — intervals too wide |
| Left-skewed (peak near 0) | Predictions systematically too slow |
| Right-skewed (peak near 1) | Predictions systematically too fast |

See [scores documentation — PIT](https://scores.readthedocs.io/en/stable/tutorials/PIT.html)
for the canonical reference implementation of these tests.

## Choice 4: Coverage heatmap, faceted by line

### What we chose

A small-multiples grid: one panel per line, each panel a 2-row × 24-column
heatmap where rows are `weekday | weekend`, columns are `hour_of_day`,
and cell colour is `coverage_p80 − 0.80` on a diverging red↔blue scale.
Implemented in
[viz.py:coverage_heatmap_chart](src/transit_observer/viz.py).

### Why

This is the single highest signal-density chart for "where is the
kernel broken?" Each cell encodes an exact coverage deviation for a
specific time-of-week. The diverging palette lets the eye locate
outliers (deep red = overconfident; deep blue = underconfident; white
= on target) without having to read numbers.

The small-multiples convention is from Tufte's *Envisioning
Information*; it's also how
[*The Economist* presents per-state election forecasts](https://statmodeling.stat.columbia.edu/2021/08/11/forecast-displays-that-emphasize-uncertainty/),
which is the closest political-forecasting analogue to our per-line
breakdown.

## Choice 5: Sharpness ↔ coverage scatter

### What we chose

One point per `(line, direction, hour, weekday|weekend)` bucket. x =
median sharpness in seconds (`p80 − p50`), y = empirical coverage at
p80, point size scaled to sample count, colour to line. Reference rule
at y = 0.80. Implemented in
[viz.py:sharpness_coverage_chart](src/transit_observer/viz.py).

### Why

The
[Gneiting et al. (2007) paper](https://doi.org/10.1111/j.1467-9868.2007.00587.x)
articulates the **sharpness principle**: among all calibrated
forecasts, prefer the sharpest. A model with perfect coverage and
huge intervals (`[0, ∞)`) is calibrated but useless. The trade-off is
made visible by plotting both axes together: the target zone is
low-x, y ≈ 0.80; the worst quadrant is high-coverage but high-sharpness
(loose hedging) — and the truly worst is low-x, low-y (confident *and*
wrong).

## Library choice: Altair

Altair was already a project dependency, ships declarative chart specs
that round-trip to/from JSON (easy to snapshot-test), and the
small-multiples grids and faceted layered charts are ~15 lines each.
We avoided Plotly to keep the dependency footprint lean.

Constraint encountered during implementation: Altair's faceted layered
charts must declare the shared dataset at the facet level rather than
per-layer (otherwise `to_dict()` raises a schema-validation error).
The chart constructors in
[viz.py](src/transit_observer/viz.py) handle this by passing
`data=df` at the `.facet()` call.

## What's deliberately out of scope

- **Schema migration for stored PIT / extra-quantile booleans.** Viz-time
  fit is fast enough; no migration debt.
- **CDF fallback chart.** Fernandes et al. show CDFs perform "nearly as
  well" as dotplots. Worth adding if Cozy Fox needs a small-screen
  fallback; not needed for the dashboard.
- **Hypothetical Outcome Plots (HOPs).** Animated outcome plots help
  with multi-variable trend inference (Hullman et al.) but for
  single-prediction probability reading the static quantile dotplot
  wins the head-to-head.
- **CORP / isotonic-regression confidence bands** on the reliability
  diagram. Would tighten the diagnostic but requires scipy; the
  simpler binning form is already informative.
- **Geographic map view** of corridors. Belongs to a separate product
  surface, not the calibration concern.
- **Animated jittery needles.** Per the Fast Company / FlowingData
  critique, we explicitly do not want this for a transit context.

## Verification

The new section is covered by tests at three layers:

- [tests/test_quantile_distribution.py](tests/test_quantile_distribution.py)
  — 18 tests on the pure math (round-trip fit, CDF↔quantile inverse,
  PIT uniformity on synthetic calibrated samples, dot endpoint
  positions match Kay's quantile spec).
- [tests/test_viz.py](tests/test_viz.py)
  — 13 tests on the chart constructors (valid Altair specs, empty
  inputs handled, Wilkinson stacking observed in the dotplot).
- [tests/test_calibration_metrics.py](tests/test_calibration_metrics.py)
  — 6 tests against an in-memory DB seeded with synthetic
  log-normally-calibrated outcomes; asserts the reliability curve
  hugs the diagonal and the PIT histogram is approximately flat
  (the two-tail end-to-end calibration sanity check).

To smoke-test the dashboard live: `./run.sh`, visit
`http://127.0.0.1:8502`, and confirm:

1. The new **Calibration & forecast displays** section renders between
   *Coverage by corridor* and *Corpus corridors*.
2. All five tabs show a helpful "no data yet" message when the
   collector hasn't run long enough.
3. The Live forecast tab's dropdown enumerates the seeded corridors;
   clicking *Predict* on a Red Line OD pair renders a ~50-dot dotplot.

## Citation index

- Kay, Kola, Hullman, Munson — *When (ish) is My Bus?* — [CHI 2016](https://dl.acm.org/doi/10.1145/2858036.2858558) — [code/data](https://github.com/mjskay/when-ish-is-my-bus) — [project page](https://mucollective.northwestern.edu/project/when-ish-is-my-bus)
- Fernandes, Walls, Munson, Hullman, Kay — *Uncertainty Displays Using Quantile Dotplots or CDFs Improve Transit Decision-Making* — [CHI 2018](https://idl.uw.edu/papers/uncertainty-bus)
- Hullman, Resnick, Adar — *Hypothetical Outcome Plots Outperform Error Bars and Violin Plots for Inferences about Reliability of Variable Ordering* — [PLOS ONE 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0142444)
- Padilla, Kay, Hullman — *Uncertainty Visualization* — [Handbook chapter, 2022](http://space.ucmerced.edu/Downloads/publications/Uncertainty_Visualization_Padilla_Kay_Hullman_2022.pdf)
- Gneiting, Balabdaoui, Raftery — *Probabilistic forecasts, calibration and sharpness* — [JRSS Series B 2007](https://doi.org/10.1111/j.1467-9868.2007.00587.x)
- Dimitriadis, Gneiting, Jordan — *Stable Reliability Diagrams for Probabilistic Classifiers* — [PNAS 2021](https://www.pnas.org/doi/10.1073/pnas.2016191118)
- Kay — [Quantile dotplots construction guide](https://github.com/mjskay/when-ish-is-my-bus/blob/master/quantile-dotplots.md)
- Carroll — [Python `quantile_dotplot` reference implementation](https://github.com/ColCarroll/quantile_dotplot)
- [Vega-Lite quantile dot plot example](https://vega.github.io/vega/examples/quantile-dot-plot/)
- [scores library — Probability Integral Transform tutorial](https://scores.readthedocs.io/en/stable/tutorials/PIT.html)
- Gelman/Morris — [Forecast displays that emphasize uncertainty (Economist 2020)](https://statmodeling.stat.columbia.edu/2021/08/11/forecast-displays-that-emphasize-uncertainty/)
- Yau — [Needle of uncertainty (FlowingData, 2018)](https://flowingdata.com/2018/03/14/needle-of-uncertainty/)
- Schiller — [The most hated data visualization in politics is back to spike your blood pressure (Fast Company, 2020)](https://www.fastcompany.com/90459366/the-most-hated-data-visualization-in-politics-is-back-to-spike-your-blood-pressure)
