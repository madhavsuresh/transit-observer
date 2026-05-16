"""Config rendering + load round-trip."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest

from transit_observer import config as config_module
from transit_observer.setup import render_config, write_config


def test_render_includes_all_keys():
    body = render_config(cta_train="train-abc", cta_bus="bus-def", metra="metra-ghi")
    parsed = tomllib.loads(body)
    assert parsed["api_keys"]["cta_train"] == "train-abc"
    assert parsed["api_keys"]["cta_bus"] == "bus-def"
    assert parsed["api_keys"]["metra"] == "metra-ghi"


def test_render_with_optional_blank_keys():
    body = render_config(cta_train="train-abc")
    parsed = tomllib.loads(body)
    assert parsed["api_keys"]["cta_train"] == "train-abc"
    assert parsed["api_keys"]["cta_bus"] == ""
    assert parsed["api_keys"]["metra"] == ""


def test_render_escapes_quotes_and_backslashes():
    body = render_config(cta_train='abc"def\\ghi')
    parsed = tomllib.loads(body)
    assert parsed["api_keys"]["cta_train"] == 'abc"def\\ghi'


def test_write_round_trips_through_loader(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.toml"
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg)
    monkeypatch.delenv("CTA_TRAIN_API_KEY", raising=False)
    monkeypatch.delenv("CTA_BUS_API_KEY", raising=False)
    monkeypatch.delenv("METRA_API_KEY", raising=False)

    write_config(cta_train="trainK", cta_bus="busK", metra="", path=cfg)
    loaded = config_module.load()
    assert loaded.cta_train_api_key == "trainK"
    assert loaded.cta_bus_api_key == "busK"
    assert loaded.metra_api_key is None  # blank → None


def test_env_var_overrides_config_file(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.toml"
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg)
    write_config(cta_train="from-file", path=cfg)
    monkeypatch.setenv("CTA_TRAIN_API_KEY", "from-env")
    loaded = config_module.load()
    assert loaded.cta_train_api_key == "from-env"


def test_load_with_no_config_returns_none_keys(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.toml"
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg)
    monkeypatch.delenv("CTA_TRAIN_API_KEY", raising=False)
    monkeypatch.delenv("CTA_BUS_API_KEY", raising=False)
    monkeypatch.delenv("METRA_API_KEY", raising=False)
    loaded = config_module.load()
    assert loaded.cta_train_api_key is None
    assert loaded.cta_bus_api_key is None
    assert loaded.metra_api_key is None
