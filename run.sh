#!/usr/bin/env bash
# One-command local runner. Starts the collector and the Streamlit
# dashboard in the foreground so any failure is visible in this
# terminal. Modeled after divvy-observer's run.sh.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required. install with: brew install uv" >&2
  exit 1
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  ./run.sh                Start collector + dashboard in this terminal
  ./run.sh setup          Re-run the interactive API-key setup

Config:
  config.toml             Persistent API keys (created on first run).
                          Edit by hand or via `uv run transit setup`.

Environment overrides (highest precedence):
  CTA_TRAIN_API_KEY, CTA_BUS_API_KEY, METRA_API_KEY

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

# First-run setup: if neither config.toml nor an env var is set, prompt.
if [[ ! -f config.toml && -z "${CTA_TRAIN_API_KEY:-}" ]]; then
  echo "==> no config.toml found — running interactive setup"
  uv run transit setup
fi

mkdir -p data logs

echo "==> starting collector"
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
