"""CTABusClient JSON parsing."""

from __future__ import annotations

import pytest

from transit_observer.bus_client import CTABusClient, _from_prediction


def test_parses_prediction_payload():
    raw = {
        "tmstmp": "20260513 08:23",
        "prdtm": "20260513 08:28",
        "rt": "22",
        "rtdir": "Northbound",
        "vid": "1234",
        "stpid": "1106",
        "stpnm": "Clark & Belmont",
        "des": "Howard",
        "prdctdn": "5",
        "dly": False,
    }
    pred = _from_prediction(raw)
    assert pred is not None
    assert pred.route == "22"
    assert pred.stop_id == 1106
    assert pred.destination_name == "Howard"
    assert pred.direction_name == "Northbound"
    assert pred.is_approaching is False
    assert not pred.is_delayed


def test_marks_approaching_when_countdown_low():
    raw = {
        "tmstmp": "20260513 08:23",
        "prdtm": "20260513 08:24",
        "rt": "22",
        "rtdir": "Northbound",
        "vid": "1",
        "stpid": "1106",
        "stpnm": "x",
        "des": "y",
        "prdctdn": "DUE",
    }
    pred = _from_prediction(raw)
    assert pred is not None
    assert pred.is_approaching is True


def test_skips_malformed_timestamp():
    raw = {
        "tmstmp": "not-a-date",
        "prdtm": "20260513 08:28",
        "rt": "22",
        "stpid": "1106",
        "vid": "1",
        "stpnm": "x",
        "des": "y",
        "rtdir": "z",
        "prdctdn": "5",
    }
    assert _from_prediction(raw) is None


def test_constructor_rejects_missing_key():
    with pytest.raises(ValueError):
        CTABusClient("")
