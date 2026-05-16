"""Write a config.toml. Pure function so tests can call it directly."""

from __future__ import annotations

from pathlib import Path

from .config import CONFIG_PATH


def _toml_escape(value: str) -> str:
    """Escape backslashes and double quotes for a TOML basic string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_config(*, cta_train: str, cta_bus: str = "", metra: str = "") -> str:
    return (
        '# transit-observer config — gitignored; keep this file private.\n'
        '# Environment variables (CTA_TRAIN_API_KEY, CTA_BUS_API_KEY, METRA_API_KEY)\n'
        '# override these values when set.\n'
        '#\n'
        '# Regenerate any time with: uv run transit setup\n'
        '\n'
        '[api_keys]\n'
        f'cta_train = "{_toml_escape(cta_train)}"\n'
        f'cta_bus = "{_toml_escape(cta_bus)}"\n'
        f'metra = "{_toml_escape(metra)}"\n'
    )


def write_config(*, cta_train: str, cta_bus: str = "", metra: str = "", path: Path = CONFIG_PATH) -> Path:
    body = render_config(cta_train=cta_train, cta_bus=cta_bus, metra=metra)
    path.write_text(body)
    return path
