#!/usr/bin/env bash
# One-command local runner. Starts the collector (which also trains the
# learned GBM in-process), the API, and the Streamlit dashboard in the
# foreground so any failure is visible in this terminal.
#
# The collector owns the single DuckDB writer, so training MUST run
# inside it (see _maybe_train in collector.py). A separate trainer
# process would deadlock on the writer lock.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required. install with: brew install uv" >&2
  exit 1
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  ./run.sh                Sync deps, start collector + API + dashboard.
                          The collector runs the learned-predictor fit
                          on its own writable connection every
                          Settings.train_interval_seconds (12h default).
  ./run.sh setup          Re-run the interactive API-key setup
  ./run.sh check          Print the learned-predictor readiness gate
                          and exit. Useful for diagnosing why the GBM
                          hasn't started training yet.

Config:
  config.toml             Persistent API keys (created on first run).
                          Edit by hand or via `uv run transit setup`.

Environment overrides (highest precedence):
  CTA_TRAIN_API_KEY, CTA_BUS_API_KEY, METRA_API_KEY

Learned-predictor cadence is set in config.py (Settings.train_interval_seconds,
default 12h). Edit there to tune.

URLs:
  Dashboard:  http://127.0.0.1:8502
  API:        http://127.0.0.1:8001      (swagger at /docs)
              (8000 is reserved for the sister divvy-observer service)
EOF
  exit 0
fi

if [[ "${1:-}" == "setup" ]]; then
  exec uv run transit setup
fi

if [[ "${1:-}" == "check" ]]; then
  exec uv run transit train check
fi

# First-run setup: if neither config.toml nor an env var is set, prompt.
if [[ ! -f config.toml && -z "${CTA_TRAIN_API_KEY:-}" ]]; then
  echo "==> no config.toml found — running interactive setup"
  uv run transit setup
fi

mkdir -p data logs

# Sync both default and `learned` groups so LightGBM / polars / joblib
# are present before the collector tries to import them. `uv sync` is
# idempotent and fast (<1s) when everything is already installed.
echo "==> syncing dependencies (base + learned)"
uv sync --group learned --quiet

# Surface the training readiness gate at boot so the operator can tell
# whether the GBM is about to start fitting. Always non-fatal.
if [[ -f data/transit_observer.duckdb ]]; then
  echo "==> learned-predictor readiness:"
  uv run transit train check 2>&1 | sed 's/^/    /'
else
  echo "==> learned-predictor readiness: no DB yet (collector will create on first tick)"
fi

echo "==> starting collector (with in-process learned-predictor training)"
uv run python -m transit_observer.collector &
COLLECTOR_PID=$!

echo "==> starting API on http://127.0.0.1:8001 (docs at /docs)"
uv run transit api --host 127.0.0.1 --port 8001 &
API_PID=$!

echo "==> starting dashboard on http://127.0.0.1:8502"
uv run streamlit run src/transit_observer/dashboard.py \
  --server.address=127.0.0.1 \
  --server.port=8502 \
  --server.headless=true \
  --browser.gatherUsageStats=false &
DASHBOARD_PID=$!

cleanup() {
  echo
  echo "==> stopping transit-observer"
  kill -TERM "$COLLECTOR_PID" "$API_PID" "$DASHBOARD_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    if ! kill -0 "$COLLECTOR_PID" 2>/dev/null \
       && ! kill -0 "$API_PID" 2>/dev/null \
       && ! kill -0 "$DASHBOARD_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  kill -KILL "$COLLECTOR_PID" "$API_PID" "$DASHBOARD_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait "$COLLECTOR_PID" "$API_PID" "$DASHBOARD_PID"
