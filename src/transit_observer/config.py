"""Configuration loaded from environment variables.

Single source of truth for paths, intervals, rate limits, and tunables.
Modules import `settings` rather than reading env vars directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo


CHICAGO = ZoneInfo("America/Chicago")


@dataclass(frozen=True)
class Settings:
    cta_train_api_key: str | None
    data_dir: Path
    logs_dir: Path
    db_path: Path
    read_replica_path: Path

    poll_interval_seconds: float = 30.0
    station_round_robin_batch: int = 18  # ~18 stations per 30s = 36/min < 100/5min
    trip_generation_interval_seconds: float = 60.0
    trips_per_generation_tick: int = 3
    resolver_interval_seconds: float = 30.0
    forecast_resolution_buffer_seconds: float = 300.0  # wait this long after p90 before declaring unresolvable
    read_replica_refresh_seconds: float = 60.0

    line_codes: tuple[str, ...] = field(
        default_factory=lambda: ("red", "blue", "brn", "g", "org", "p", "pink", "y")
    )


def load() -> Settings:
    """Read env + filesystem defaults."""
    root = Path(__file__).resolve().parents[2]
    data = root / "data"
    logs = root / "logs"
    data.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return Settings(
        cta_train_api_key=os.environ.get("CTA_TRAIN_API_KEY"),
        data_dir=data,
        logs_dir=logs,
        db_path=data / "transit_observer.duckdb",
        read_replica_path=data / "transit_observer_readonly.duckdb",
    )


settings = load()
