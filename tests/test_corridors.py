"""Corridor seeding + due-corridor cadence selection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from transit_observer import db
from transit_observer.corridors import (
    SEED_CORRIDORS,
    by_id,
    by_mode,
    due_corridors,
    mark_predicted,
    seed_corridors,
)


T0 = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    db.init_schema(c)
    try:
        yield c
    finally:
        c.close()


def test_seed_corridors_inserts_all(conn: duckdb.DuckDBPyConnection):
    n = seed_corridors(conn, now=T0)
    assert n == len(SEED_CORRIDORS)
    rows = conn.execute("SELECT COUNT(*) FROM corridors").fetchone()
    assert rows[0] == len(SEED_CORRIDORS)


def test_seed_corridors_is_idempotent(conn: duckdb.DuckDBPyConnection):
    seed_corridors(conn, now=T0)
    seed_corridors(conn, now=T0 + timedelta(hours=1))
    rows = conn.execute("SELECT COUNT(*) FROM corridors").fetchone()
    assert rows[0] == len(SEED_CORRIDORS)


def test_every_line_has_both_directions():
    """Spec: never a single bidirectional record. We assert the load-bearing
    invariant: every (mode, line) appearing in the seed list has at least
    two distinct directions.

    Endpoint labels may differ between the two directions (e.g. NB Route 22
    downtown stops are on Dearborn, SB on Clark; Intercampus IB/OB stops
    are physically distinct). What matters is that we never have a line
    with a single direction collapsed across both ways of travel.
    """
    by_line: dict[tuple, set[str]] = {}
    for c in SEED_CORRIDORS:
        by_line.setdefault((c.mode, c.line), set()).add(c.direction)
    for key, dirs in by_line.items():
        assert len(dirs) >= 2, f"{key} has only direction(s) {dirs}; spec wants both ways"


def test_by_id_and_by_mode_lookups():
    seeds = by_id()
    assert "metra-upn-evanston-otc-ib" in seeds
    assert seeds["metra-upn-evanston-otc-ib"].mode == "metra"
    metra = by_mode("metra")
    assert all(c.mode == "metra" for c in metra)
    assert len(metra) >= 4


def test_due_corridors_returns_unpredicted_first(conn: duckdb.DuckDBPyConnection):
    seed_corridors(conn, now=T0)
    due = due_corridors(conn, now=T0, enabled_modes=["L", "bus", "metra", "intercampus"])
    # Every corridor is "due" on first tick since none have last_predicted_at.
    assert len(due) == len(SEED_CORRIDORS)


def test_due_corridors_respects_cadence(conn: duckdb.DuckDBPyConnection):
    seed_corridors(conn, now=T0)
    # Predict one corridor; with cadence 300s, it should NOT be due 100s later.
    target = SEED_CORRIDORS[0].corridor_id
    mark_predicted(conn, corridor_id=target, at=T0)

    soon = T0 + timedelta(seconds=100)
    due_ids = {c.corridor_id for c in due_corridors(conn, now=soon, enabled_modes=["L", "bus", "metra", "intercampus"])}
    assert target not in due_ids

    later = T0 + timedelta(seconds=400)
    due_ids = {c.corridor_id for c in due_corridors(conn, now=later, enabled_modes=["L", "bus", "metra", "intercampus"])}
    assert target in due_ids


def test_due_corridors_filtered_by_mode(conn: duckdb.DuckDBPyConnection):
    seed_corridors(conn, now=T0)
    due = due_corridors(conn, now=T0, enabled_modes=["intercampus"])
    assert all(c.mode == "intercampus" for c in due)
    assert len(due) == len(by_mode("intercampus"))


def test_due_corridors_sorted_by_priority(conn: duckdb.DuckDBPyConnection):
    seed_corridors(conn, now=T0)
    due = due_corridors(conn, now=T0, enabled_modes=["L", "bus", "metra", "intercampus"])
    priorities = [c.priority for c in due]
    assert priorities == sorted(priorities), "due_corridors should be priority-ascending"


def test_re_seed_updates_metadata(conn: duckdb.DuckDBPyConnection):
    seed_corridors(conn, now=T0)
    # Simulate a manual mutation to verify upsert restores authoritative values.
    conn.execute("UPDATE corridors SET cadence_seconds = 9999 WHERE corridor_id = ?",
                 [SEED_CORRIDORS[0].corridor_id])
    seed_corridors(conn, now=T0 + timedelta(hours=1))
    cadence = conn.execute("SELECT cadence_seconds FROM corridors WHERE corridor_id = ?",
                           [SEED_CORRIDORS[0].corridor_id]).fetchone()[0]
    assert cadence == SEED_CORRIDORS[0].cadence_seconds
