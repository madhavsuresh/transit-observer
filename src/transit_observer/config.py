"""Configuration loaded from ``config.toml`` with environment-variable override.

Resolution order (highest precedence first):
1. ``CTA_TRAIN_API_KEY`` / ``CTA_BUS_API_KEY`` / ``METRA_API_KEY`` env vars.
2. ``[api_keys]`` section in ``config.toml`` at the project root.

Use ``transit setup`` to create or refresh ``config.toml`` interactively.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo


CHICAGO = ZoneInfo("America/Chicago")


@dataclass(frozen=True)
class Settings:
    cta_train_api_key: str | None
    cta_bus_api_key: str | None
    metra_api_key: str | None
    data_dir: Path
    logs_dir: Path
    db_path: Path
    read_replica_path: Path

    poll_interval_seconds: float = 30.0
    station_round_robin_batch: int = 18  # ~18 L stations per 30s = 36/min < 100/5min
    bus_round_robin_batch: int = 6        # 12/min — gentle on the 10,000/day bus budget
    metra_poll_interval_seconds: float = 60.0
    intercampus_poll_interval_seconds: float = 60.0
    trip_generation_interval_seconds: float = 60.0
    trips_per_generation_tick: int = 3
    resolver_interval_seconds: float = 30.0
    forecast_resolution_buffer_seconds: float = 300.0  # wait this long after p90 before declaring unresolvable
    read_replica_refresh_seconds: float = 60.0

    query_import_interval_seconds: float = 60.0   # how often to import API queries.ndjson into query_log
    promotion_interval_seconds: float = 600.0     # how often to scan for popular ODs to auto-upgrade
    promotion_min_count: int = 50                 # queries-in-7-days threshold for auto-upgrade

    # Learned predictor: how often to retry the GBM fit. The first
    # firing happens one interval after collector startup. The fit
    # itself refuses to run until the cold-start thresholds are met
    # (see training.dataset.cold_start_threshold), so this is safe to
    # tick frequently early on.
    train_interval_seconds: float = 43200.0       # 12 h — nightly-ish
    train_window_days: int = 60
    # Set to 0 to disable in-collector training (e.g., when running a
    # separate cron-driven trainer).
    train_enabled: bool = True

    line_codes: tuple[str, ...] = field(
        default_factory=lambda: ("red", "blue", "brn", "g", "org", "p", "pink", "y")
    )
    monitored_bus_stops: tuple[tuple[str, int], ...] = field(
        default_factory=lambda: (
            # General-coverage stops (not tied to any corridor; kept for
            # raw data collection across high-ridership routes).
            ("147", 12550),  # Outer Drive Express
            ("151", 1928),   # Sheridan & Belmont
            ("9", 8089),     # Ashland & Madison
            ("X9", 8089),
            ("J14", 17131),  # Jeffery Jump
            ("3", 8056),     # King Drive
            ("66", 4519),    # Chicago & State (EB)
            ("66", 4540),    # Chicago & State (WB)

            # Route 22 (Clark) corridor stops -- both directions, both ends.
            ("22", 1828),    # Clark & Belmont (SB)
            ("22", 1921),    # Clark & Belmont (NB)
            ("22", 1869),    # Clark & Adams (SB)
            ("22", 14767),   # Dearborn & Grand (NB)

            # Route 66 (Chicago Ave) corridor: Michigan <-> Western.
            ("66", 599),     # Chicago & Michigan (WB)
            ("66", 580),     # Chicago & Michigan (EB)
            ("66", 15203),   # Chicago & Western (WB)
            ("66", 548),     # Chicago & Western (EB)

            # Route 9 (Ashland) corridor: Belmont <-> 87th.
            ("9", 6003),     # Ashland & Belmont (SB)
            ("9", 6272),     # Ashland & Belmont (NB)
            ("9", 6155),     # Ashland & 87th (NB)
            ("9", 15249),    # Ashland & 87th (SB)

            # Route 79 (79th St) corridor: Halsted <-> Stony Island.
            ("79", 2762),    # 79th & Halsted (EB)
            ("79", 17349),   # 79th & Halsted (WB)
            ("79", 2795),    # 79th & Stony Island (EB)
            ("79", 2621),    # 79th & Stony Island (WB)
        )
    )


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.toml"


def _read_config_file(path: Path | None = None) -> dict:
    target = path if path is not None else CONFIG_PATH
    if not target.exists():
        return {}
    try:
        with target.open("rb") as f:
            return tomllib.load(f) or {}
    except (tomllib.TOMLDecodeError, OSError):
        return {}


def _resolve(env_key: str, file_dict: dict, file_key: str) -> str | None:
    env_value = os.environ.get(env_key)
    if env_value:
        return env_value
    file_value = file_dict.get(file_key)
    if isinstance(file_value, str) and file_value:
        return file_value
    return None


def load() -> Settings:
    """Read config + env defaults. Env vars override the TOML file."""
    data = ROOT / "data"
    logs = ROOT / "logs"
    data.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    file_config = _read_config_file()
    api_keys = file_config.get("api_keys", {}) if isinstance(file_config, dict) else {}

    return Settings(
        cta_train_api_key=_resolve("CTA_TRAIN_API_KEY", api_keys, "cta_train"),
        cta_bus_api_key=_resolve("CTA_BUS_API_KEY", api_keys, "cta_bus"),
        metra_api_key=_resolve("METRA_API_KEY", api_keys, "metra"),
        data_dir=data,
        logs_dir=logs,
        db_path=data / "transit_observer.duckdb",
        read_replica_path=data / "transit_observer_readonly.duckdb",
    )


settings = load()
