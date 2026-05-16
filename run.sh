#!/usr/bin/env bash
# Foreground runner. Starts collector (which internally drives trajectory build,
# trip generation, and resolution). Logs go to logs/collector.log + stdout.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required. install with: brew install uv" >&2
  exit 1
fi

if [[ -z "${CTA_TRAIN_API_KEY:-}" ]]; then
  echo "error: CTA_TRAIN_API_KEY env var is required. obtain from https://www.transitchicago.com/developers/traintrackerapply/" >&2
  exit 1
fi

mkdir -p data logs

echo "==> starting transit-observer collector"
exec uv run python -m transit_observer.collector
