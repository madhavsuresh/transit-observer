"""Direction-filter audit at resolve time."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from transit_observer import db
from transit_observer.direction_audit import audit_resolved_forecast, audit_summary


T0 = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)

# Real CTA map_ids — direction_audit looks the boarding + alighting up in
# the catalog so they have to exist.
CHICAGO_STATE = 41450
HOWARD = 40900
NINETY_FIFTH = 40450


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    db.init_schema(c)
    try:
        yield c
    finally:
        c.close()


def _insert_arrival(
    conn: duckdb.DuckDBPyConnection,
    *,
    polled_at: datetime,
    line: str,
    run: str,
    map_id: int,
    arrival_at: datetime,
    destination: str,
    direction: str,
) -> None:
    conn.execute(
        """
        INSERT INTO train_arrivals_raw (
            polled_at, line, run_number, map_id, stop_id, station_name,
            direction_code, destination_name, predicted_at, arrival_at,
            is_approaching, is_delayed, is_fault, is_scheduled
        ) VALUES (?, ?, ?, ?, 0, 'X', ?, ?, ?, ?, FALSE, FALSE, FALSE, FALSE)
        """,
        [polled_at, line, run, map_id, direction, destination, polled_at, arrival_at],
    )


def _enqueue_and_resolve(
    conn: duckdb.DuckDBPyConnection,
    *,
    leave_at: datetime,
    boarding: int,
    alighting: int,
    line: str,
    boarded_run: str,
    boarded_at: datetime,
    alighted_at: datetime,
) -> str:
    fid = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO forecast_queue (
            forecast_id, enqueued_at, snapshot_polled_at, leave_at,
            line, direction_code,
            boarding_map_id, boarding_station_name,
            alighting_map_id, alighting_station_name,
            predicted_wait_mean, predicted_wait_p50, predicted_wait_p80, predicted_wait_p90,
            predicted_in_vehicle_mean,
            predicted_total_mean, predicted_total_p50, predicted_total_p80, predicted_total_p90,
            predicted_failure_prob, resolve_after, status
        ) VALUES (?, ?, ?, ?, ?, NULL, ?, 'A', ?, 'B', 60, 60, 90, 120, 600, 660, 660, 780, 840, 0.0, ?, 'resolved')
        """,
        [fid, leave_at, leave_at, leave_at, line, boarding, alighting,
         leave_at + timedelta(seconds=900)],
    )
    conn.execute(
        """
        INSERT INTO forecast_outcomes (
            forecast_id, resolved_at, boarded_run_number,
            boarded_at, alighted_at,
            actual_wait_seconds, actual_in_vehicle_seconds, actual_total_seconds,
            in_p80_window, in_p90_window,
            p50_residual_seconds, p80_residual_seconds, failed, notes
        ) VALUES (?, ?, ?, ?, ?, 0, 0, 0, TRUE, TRUE, 0, 0, FALSE, NULL)
        """,
        [fid, leave_at + timedelta(seconds=900), boarded_run, boarded_at, alighted_at],
    )
    return fid


def test_filter_keeps_correct_direction_train(conn: duckdb.DuckDBPyConnection):
    leave = T0
    # 2 northbound (Howard), 2 southbound (95th).
    for i, m in enumerate([3, 11]):
        _insert_arrival(
            conn, polled_at=leave, line="Red", run=f"NB{i}", map_id=CHICAGO_STATE,
            arrival_at=leave + timedelta(minutes=m), destination="Howard", direction="1",
        )
    for i, m in enumerate([5, 13]):
        _insert_arrival(
            conn, polled_at=leave, line="Red", run=f"SB{i}", map_id=CHICAGO_STATE,
            arrival_at=leave + timedelta(minutes=m), destination="95th/Dan Ryan", direction="5",
        )
    fid = _enqueue_and_resolve(
        conn, leave_at=leave, boarding=CHICAGO_STATE, alighting=HOWARD,
        line="Red", boarded_run="NB0",
        boarded_at=leave + timedelta(minutes=3), alighted_at=leave + timedelta(minutes=33),
    )
    result = audit_resolved_forecast(conn, forecast_id=fid, now=leave + timedelta(minutes=40))
    assert result is not None
    assert result.boarded_was_kept
    # Two Howard-bound arrivals matched the boarded direction.
    assert result.kept_matching_boarded_direction == 2
    # The filter kept only Howard-bound arrivals.
    assert "Howard" in result.kept_destination_names
    assert "95th/Dan Ryan" not in result.kept_destination_names


def test_filter_keeps_unknown_destination_conservatively(conn: duckdb.DuckDBPyConnection):
    leave = T0
    _insert_arrival(
        conn, polled_at=leave, line="Brn", run="LOOP", map_id=CHICAGO_STATE,
        arrival_at=leave + timedelta(minutes=4), destination="Loop", direction="5",
    )
    _insert_arrival(
        conn, polled_at=leave, line="Brn", run="KMBL", map_id=CHICAGO_STATE,
        arrival_at=leave + timedelta(minutes=6), destination="Kimball", direction="1",
    )
    fid = _enqueue_and_resolve(
        conn, leave_at=leave, boarding=CHICAGO_STATE, alighting=HOWARD,
        line="Brn", boarded_run="KMBL",
        boarded_at=leave + timedelta(minutes=6), alighted_at=leave + timedelta(minutes=36),
    )
    result = audit_resolved_forecast(conn, forecast_id=fid, now=leave + timedelta(minutes=40))
    assert result is not None
    # 'Loop' isn't in the catalog so it's kept conservatively. Both stay.
    assert result.kept_arrivals_count == 2


def test_audit_summary_aggregates_per_line(conn: duckdb.DuckDBPyConnection):
    leave = T0
    for offset_minutes in range(10):
        per_leave = leave + timedelta(minutes=offset_minutes * 30)
        _insert_arrival(
            conn, polled_at=per_leave, line="Red", run=f"R{offset_minutes}",
            map_id=CHICAGO_STATE, arrival_at=per_leave + timedelta(minutes=3),
            destination="Howard", direction="1",
        )
        fid = _enqueue_and_resolve(
            conn, leave_at=per_leave, boarding=CHICAGO_STATE, alighting=HOWARD,
            line="Red", boarded_run=f"R{offset_minutes}",
            boarded_at=per_leave + timedelta(minutes=3),
            alighted_at=per_leave + timedelta(minutes=33),
        )
        audit_resolved_forecast(conn, forecast_id=fid, now=per_leave + timedelta(minutes=40))
    summary = audit_summary(conn, min_samples=5)
    assert len(summary) == 1
    red = summary[0]
    assert red.line == "Red"
    assert red.n_audited == 10
    assert red.recall_rate == 1.0
    assert red.avg_direction_precision == 1.0
