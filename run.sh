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
  Dashboard:  http://127.0.0.1:8501
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

echo "==> starting dashboard on http://127.0.0.1:8501"
uv run streamlit run src/transit_observer/dashboard.py \
  --server.address=127.0.0.1 \
  --server.port=8501 \
  --server.headless=true \
  --browser.gatherUsageStats=false &
DASHBOARD_PID=$!

cleanup() {
  echo
  echo "==> stopping transit-observer"
  kill -TERM "$COLLECTOR_PID" "$DASHBOARD_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    if ! kill -0 "$COLLECTOR_PID" 2>/dev/null && ! kill -0 "$DASHBOARD_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  kill -KILL "$COLLECTOR_PID" "$DASHBOARD_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait "$COLLECTOR_PID" "$DASHBOARD_PID"
