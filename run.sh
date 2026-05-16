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

Environment:
  CTA_TRAIN_API_KEY       Required for the L collector
  CTA_BUS_API_KEY         Optional; enables CTA Bus polling
  METRA_API_KEY           Optional; enables Metra polling

URLs:
  Dashboard:  http://127.0.0.1:8501
EOF
  exit 0
fi

if [[ -z "${CTA_TRAIN_API_KEY:-}" ]]; then
  echo "error: CTA_TRAIN_API_KEY env var is required." >&2
  echo "       (CTA_BUS_API_KEY and METRA_API_KEY are optional.)" >&2
  exit 1
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
