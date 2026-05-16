# AGENTS.md

Guidance for Codex when working in this repo.

## What this is

A long-running Python observatory for CTA L trip predictions. Companion to [Cozy Fox](../transit/) (the iOS app, Swift) — Cozy Fox's `TransitCore/Journey` kernels are the canonical implementation; this project re-implements them in Python to validate predictions against realized outcomes. Drift between Swift and Python is the main maintenance cost; a single golden-file test pins behavioral parity.

Modeled after [divvy-observer](../divvy-observer/): single-writer DuckDB, forecast queue, separate read replica, `uv`-managed.

## Commands

```bash
uv sync
./run.sh                            # collector + trip generator + resolver, foreground
uv run transit status               # health
uv run transit metrics              # coverage / calibration
uv run transit corridors            # bucket inventory
uv run pytest                       # all tests
uv run pytest -m 'not live'         # skip live-API tests
```

## Architecture

Four layers, all in `src/transit_observer/`:

1. **Collection** — `collector.py` (main loop), `cta_train_client.py`. Polls `ttarrivals.aspx` round-robin across the L catalog (~145 stations) under CTA's 100-req-per-5-min limit. Writes raw arrivals to `train_arrivals_raw`. Each tick also kicks off trajectory build + forecast generation + resolution.

2. **State** — `db.py`. Single DuckDB writer = the collector. Schema in `db.py:SCHEMA_SQL`. Tables:
   - `train_arrivals_raw` — every prediction we saw (one row per (poll, run, station))
   - `train_runs_observed` — one row per (run, station) with the inferred actual arrival time
   - `forecast_queue` — predictions waiting to be resolved
   - `forecast_outcomes` — resolved predictions with the realized actual

3. **Prediction** — `journey/` mirrors the Swift kernel layout. `stop_arrival.py` is the Python port of `StopArrivalProcess`. Don't rewrite features per-kernel — share helpers in `journey/`. The shared port is the API surface, not internal model implementations.

4. **Validation** — `trip_generator.py` picks random (line, boarding, alighting) trips. `resolver.py` finds the realized run for each. `metrics.py` reports coverage / calibration / sharpness per corridor bucket.

## Coverage definition

A "corridor bucket" is `(line, direction, hour_of_day, weekday|weekend)`. ~8 lines × 2 dirs × 24 hours × 2 = ~768 buckets. We target ≥5 samples per bucket per week before the coverage estimate is trusted.

## Drift policy

Whenever the Swift `StopArrivalProcess` changes substantively:
- Update `journey/stop_arrival.py` to match
- Update `tests/test_kernel_parity.py` (golden-file: a fixed snapshot → identical p50/p80/p90 across Swift and Python)
- If parity can't be preserved, document why in the test and tag the divergence

## Gotchas

- **Rate limit**: 100 req / 5 min. The round-robin is the only way to cover the catalog. Don't add per-station polls outside of it.
- **DST**: CTA times are local Chicago. Resolver compares timestamps; use timezone-aware datetimes everywhere.
- **Trajectory inference**: We don't have a real "train arrived" signal. We approximate observed arrival as the moment a prediction with `isApproaching=true` first appeared, falling back to the last predicted `arrivalAt` before the run disappeared from feeds at that station. Document any algorithm change in `trajectory.py`.
- **DuckDB writers**: Only the collector writes. API/dashboard (when they exist) read the replica.
- **Adding modes**: Bus, Metra, Intercampus arrive in later phases. Don't pre-emptively schema-fy.

## What's NOT here

- Bus / Metra / Intercampus / Divvy
- Multi-leg trips (transfers)
- Walking time validation
- FastAPI / Streamlit dashboard (planned)
- ML models — the only "model" is the journey kernel; we're testing that, not competing predictors
