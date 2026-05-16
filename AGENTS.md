# AGENTS.md

Guidance for Codex when working in this repo.

## What this is

A long-running Python observatory for Chicago transit prediction quality. Companion to [Cozy Fox](../transit/) (the iOS app, Swift) — Cozy Fox's `TransitCore/Journey` kernels are the canonical implementation; this project re-implements them in Python to validate predictions against realized outcomes. Drift between Swift and Python is the main maintenance cost.

Modeled after [divvy-observer](../divvy-observer/): single-writer DuckDB, forecast queue, separate read replica, `uv`-managed.

Four modes are live: **CTA L, CTA bus, Metra, Northwestern Intercampus**.

## Commands

```bash
uv sync
./run.sh                            # collector + API + dashboard, foreground
./run.sh setup                      # re-run the interactive API-key setup

uv run transit status               # one-line health check
uv run transit metrics              # coverage / calibration table
uv run transit corridors            # which buckets are under-sampled
uv run transit audit                # direction-filter recall/precision
uv run transit calibration          # reliability bins for failure prob
uv run transit corpus list          # all 60 seeded corridors
uv run transit corpus query <id>    # predicted/actual/residual for one corridor
uv run transit api --port 8001      # standalone API (also auto-started by run.sh)

uv run pytest                       # all tests (~90 fast)
uv run pytest -m 'not live'         # skip live-API tests
```

URLs when `./run.sh` is up:

- Dashboard: <http://127.0.0.1:8502>
- API: <http://127.0.0.1:8001> (Swagger at `/docs`)
- Port **8000** is reserved for the sister divvy-observer service.

## Architecture

All Python under `src/transit_observer/`. Loosely layered:

1. **Collection** — `collector.py` is the main loop. Per-mode HTTP clients in `cta_train_client.py`, `bus_client.py`, `metra_client.py`, `intercampus_client.py`. Polls each upstream on its own cadence inside one asyncio loop. Writes raw rows to `*_raw` tables.

2. **State** — `db.py`. Single-writer DuckDB at `data/transit_observer.duckdb`; the collector is the only writer. A read replica at `data/transit_observer_readonly.duckdb` refreshes every 60s for the API + dashboard. Schema lives in `SCHEMA_SQL`; in-place migrations for older DBs are in `_MIGRATIONS`.

   Tables (one per mode where it matters):
   - `train_arrivals_raw`, `train_positions_raw`, `train_runs_observed`
   - `bus_predictions_raw`, `bus_runs_observed`
   - `metra_arrivals_raw`, `metra_trips_observed`
   - `intercampus_arrivals_raw`, `intercampus_trips_observed`
   - `corridors` — the canonical synthetic-route corpus seed set + auto-upgraded entries
   - `forecast_queue` — predictions waiting to resolve, tagged with `corridor_id` / `predictor_version` / `feature_json`
   - `forecast_outcomes` — resolved predictions w/ residuals + `truth_confidence`
   - `direction_audit` — per-resolved-forecast recall/precision of the direction filter
   - `query_log` — every API `/predict` hit, ingested from `data/queries.ndjson`

3. **Kernels** — `journey/` mirrors the Swift kernel layout. `stop_arrival.py` is the Python port of `StopArrivalProcess`. Share helpers in `journey/`; don't re-implement features per kernel.

4. **Corpus + prediction** — `corridors.py` defines `SEED_CORRIDORS` (60 hand-seeded). `corpus.py` runs `predict_and_enqueue_corridor` on a fixed cadence per corridor; `predict_for_od` does on-demand predictions for the API. Per-mode predictors are `trip_generator.py` (L), `bus_predictor.py`, `metra_predictor.py`, `intercampus_predictor.py`. `resolver.py` finds the realized run + computes residuals + writes `direction_audit`. `metrics.py` aggregates coverage/calibration/sharpness.

5. **Auto-upgrade** — `query_log.py` appends to `data/queries.ndjson` from the API and imports into `query_log` from the collector. `auto_upgrade.py` promotes any (mode, line, boarding, alighting) queried ≥50 times in a 7-day window into a real corridor (`source='auto_upgraded'`).

6. **Surface** — `cli.py` (Click), `api.py` (FastAPI on 8001), `dashboard.py` (Streamlit on 8502). All read the replica; only `collector.py` writes.

## Corridor design

Each `Corridor` is one direction of one (mode, line, boarding, alighting) pair. Two rows per OD (inbound + outbound) is the rule; bidirectional records are not allowed. Priority 1–6: hand-seeded endpoints get 1–4, hand-seeded intermediates get 5, auto-upgraded gets 6. The collector predicts higher-priority corridors first when the per-tick budget runs out.

When adding a corridor, also check `monitored_bus_stops` in `config.py` — bus corridors only get data if both endpoints are in the monitored set, because the bus poller only hits stops in that list.

## Coverage definition

A "corridor bucket" is `(line, direction, hour_of_day, weekday|weekend)`. We target ≥5 samples per bucket per week before the coverage estimate is trusted. `transit corridors` lists buckets below threshold.

## Drift policy

Whenever the Swift `StopArrivalProcess` (or any other kernel) changes substantively:

1. Update the matching file in `journey/`.
2. Re-run the parity tests (`tests/test_stop_arrival.py`, `tests/test_time_distribution.py`) — they pin expected p50/p80/p90 against fixed inputs.
3. If parity can't be preserved, document the divergence in the test docstring.

## Gotchas

- **Rate limits.** CTA Train: 100 req / 5 min — round-robin polling at 18 stations / 30 s sits well under. CTA Bus: ~10k req / day budget shared across all keys — bus poll runs at 6 stops / 30 s = 12/min independent of how many stops are monitored (more stops just means each polled less often).
- **Intercampus is M–F only.** `_intercampus_service_active(now)` in `collector.py` gates both the raw poll and corpus prediction on weekends to avoid hammering an empty feed.
- **DST and timezones.** All domain code uses `America/Chicago`. Use timezone-aware datetimes everywhere; the resolver compares timestamps directly.
- **Trajectory inference.** No real "train arrived" signal exists. Priority is `positions:isApproaching` > `arrivals:isApproaching` > `arrivals:dropoff` (last predicted arrival before the run disappears). Document any algorithm change in `trajectory.py`.
- **DuckDB single-writer.** The collector holds the writer. API + dashboard read the replica. Cross-process write requires NDJSON-bridging (see `query_log.py`).
- **DuckDB ALTER limits.** `ALTER TABLE ADD COLUMN` rejects `NOT NULL`/`DEFAULT` constraints; migrations add the column unconstrained then `UPDATE` to backfill. See `_MIGRATIONS` in `db.py`.
- **`source='seed'` vs `source='auto_upgraded'`.** Both are real corridors with graded forecasts. Don't filter to `source='seed'` unless you have a specific reason.
- **Schema parsing.** `SCHEMA_SQL` is split on `;`; don't put `;` inside SQL comments.

## What's NOT here

- Divvy bike — divvy-observer is its own project; this repo never polls Divvy.
- Multi-leg trips / transfers — each corridor is one mode end-to-end.
- Walking-time estimates — kernels predict transit legs, not walks.
- ML models — the "model" is the journey kernel. We test the kernel, not competing predictors.
