"""Smoke test for the bundled CTAStations.json catalog."""

from __future__ import annotations

from transit_observer.catalog import by_line, by_map_id, by_name, load_catalog


def test_catalog_loads():
    catalog = load_catalog()
    assert len(catalog) >= 140
    assert all(s.map_id > 0 for s in catalog)
    assert all(s.served_lines for s in catalog)


def test_known_stations_present():
    cat = load_catalog()
    ids = by_map_id(cat)
    # spot-check a few well-known map_ids
    howard = ids.get(40900)
    davis = ids.get(40050)
    chicago_state = ids.get(41450)
    assert howard is not None and howard.name == "Howard"
    assert davis is not None and davis.name == "Davis"
    assert chicago_state is not None and chicago_state.name == "Chicago"


def test_by_line_buckets_have_minimum_stations():
    cat = load_catalog()
    buckets = by_line(cat)
    # All eight L lines should have at least two stations
    expected = {"red", "blue", "brown", "green", "orange", "purple", "pink", "yellow"}
    assert expected.issubset(buckets.keys())
    for line in expected:
        assert len(buckets[line]) >= 2, line


def test_name_lookup_is_case_insensitive():
    cat = load_catalog()
    lookup = by_name(cat)
    assert "howard" in lookup
    assert "davis" in lookup
    # Multiple stations share names — at least one should resolve
    assert "western" in lookup
