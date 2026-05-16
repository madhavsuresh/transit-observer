# transit-observer

A long-running observatory for the **CTA L network** that empirically validates the journey-prediction kernels in [Cozy Fox](../transit/). Mirrors [divvy-observer](../divvy-observer/)'s shape: a collector polls the CTA Train Tracker, a DuckDB writer owns state, a Python implementation of the same kernels predicts random trips, a resolver pairs predictions with realized outcomes, and metrics report coverage / calibration / sharpness — bucketed so we can tell which corridors are under-validated.

The Cozy Fox Swift kernels remain canonical; the Python ones here are a research/validation port. A golden-file test asserts the two produce equivalent output for a fixed snapshot.

## What this validates

For each random sampled trip on a CTA L line:

- **Wait kernel** — given the user arrives at the boarding platform at time `t`, what's the distribution of wait time until the next train? Did the realized wait fall inside our p80?
- **In-vehicle kernel** — once boarded, how long does the train take to reach the alighting station?
- **Composed total** — wait + in-vehicle as a single distribution. Coverage = P(actual ≤ predicted p80).

v1 scope:
- **CTA L** — prediction + resolution + direction-label audit (the full loop)
- **CTA Bus** — raw collection only (monitored-stop set; prediction/resolution to come)
- **Metra** — raw GTFS-RT collection only
- **Intercampus** (Northwestern shuttle) — raw GTFS-RT collection only

No walking, no transfers, no Divvy yet. Multi-mode prediction + resolution come in v2.

## What "good coverage" means

For each (line, direction, hour-of-day) bucket we want **at least 5 realized samples per week** so the coverage estimate stabilizes. The corridor inventory dashboard surfaces buckets that fall short.

## Quick start

```bash
brew install uv
uv sync
./run.sh                       # collector + dashboard, foreground
```

On first run, `./run.sh` notices there's no `config.toml` and walks you through an
interactive prompt for the CTA Train (required), CTA Bus, and Metra API keys.
The file ends up gitignored at `config.toml`. To re-run the prompt later:
`./run.sh setup` or `uv run transit setup`.

If you'd rather skip the file entirely, set the env vars instead — they win over
`config.toml` whenever both are present:

```bash
export CTA_TRAIN_API_KEY=<your-cta-key>
export CTA_BUS_API_KEY=<your-bus-key>    # optional
export METRA_API_KEY=<your-metra-key>    # optional
```

Dashboard at http://127.0.0.1:8501. The single command starts:

- the **L collector** — round-robin polls `ttarrivals.aspx` across the L catalog (under CTA's 100-req-per-5-min limit) and writes raw arrivals to `train_arrivals_raw`. Also pulls `ttpositions.aspx` once per tick (1 req per line, 8 reqs/30s) into `train_positions_raw` as a stronger trajectory signal.
- the **trajectory builders** — per mode, derive observed run/trip arrivals from raw data.
- the **bus collector** — round-robin polls `getpredictions` for a configurable monitored-stop set (see `config.py`).
- the **Metra collector** — pulls Metra GTFS-RT tripUpdates every 60s into `metra_arrivals_raw`.
- the **Intercampus collector** — pulls Northwestern TripShot GTFS-RT every 60s into `intercampus_arrivals_raw` (no auth).
- the **multimodal trip generator** — each tick samples random (mode, route, boarding, alighting) trips. L picks any line+station pair; bus is constrained to the monitored set; Metra uses the station catalog; Intercampus uses the two-direction shuttle.
- the **resolver** — dispatches per mode. Finds the realized run/trip/vehicle for each due forecast, writes the outcome, and runs the per-mode direction audit.
- the **dashboard** — Streamlit app showing per-mode counts, coverage by corridor, residual scatterplot, and direction-filter audit table.

DB lives at `data/transit_observer.duckdb` (writer) with `data/transit_observer_readonly.duckdb` as the 60s-refreshed read replica.

```bash
uv run transit status         # one-line health check
uv run transit metrics        # coverage / calibration table
uv run transit corridors      # which (line, hour) buckets need more samples
```

## Layout

```
src/transit_observer/
├── cli.py                  uv run transit ...
├── config.py               env vars + defaults
├── db.py                   schema, connections, read replica
├── cta_train_client.py     httpx client for ttarrivals / ttpositions
├── collector.py            main poll loop
├── trajectory.py           arrivals_raw -> runs_observed
├── trip_generator.py       sample random (line, boarding, alighting)
├── forecast_queue.py       enqueue + drain pending forecasts
├── resolver.py             find realized train per forecast
├── metrics.py              coverage / calibration / sharpness
└── journey/                Python port of Cozy Fox kernels
    ├── time_distribution.py
    ├── stop_arrival.py     port of StopArrivalProcess
    ├── kernels.py          PreparedTransitLeg analog
    └── composer.py         JourneyComposer analog
```

## Rate limits

CTA Train Tracker allows **100 requests per 5-minute window per key**. We poll `ttarrivals.aspx` for each L station in a round-robin so every station refreshes every ~7 minutes. For more granular wait predictions we'll later layer on `ttpositions.aspx?rt=<line>` (one request per line covers all in-flight runs); for now the round-robin is enough to validate that the kernel produces sensible p80 windows.

## Privacy

Local-only. No personal data, no destinations, no addresses. The simulator's "random trip" is a synthetic rider — there's no real user attached.

## Status

v1: under construction. The Python kernel port matches Swift for the wait kernel; in-vehicle is a hand-fit Gaussian for now. No dashboard yet. No multi-leg. No Divvy.
