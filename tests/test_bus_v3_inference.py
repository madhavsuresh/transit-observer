"""Synthetic-fixture tests for bus_v3 ground-truth inference.

These fixtures seed `bus_v3_*` rows directly (skipping the API client +
normalizer) so they can exercise the inference algorithm at the unit
level. The port of the validator's ``synthetic_smoke.py`` lives in
:func:`test_high_confidence_crossing`.
"""

from __future__ import annotations

import os
import tempfile

import duckdb
import pytest

from transit_observer.bus_v3.inference import (
    GROUND_TRUTH_MAX_VEHICLE_AGE_S,
    infer_bus_arrivals,
)
from transit_observer.db import init_schema


BASE_MS = 1_700_000_000_000


@pytest.fixture()
def conn():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test.duckdb")
    conn = duckdb.connect(path)
    init_schema(conn)
    try:
        yield conn
    finally:
        conn.close()
        import shutil

        shutil.rmtree(tmpdir)


def _seed_pattern(conn: duckdb.DuckDBPyConnection) -> None:
    """A 3-point pattern with the target stop at pdist 1000."""
    conn.execute(
        """
        INSERT INTO bus_v3_pattern(pid, rt, rtdir, length_ft, dtrid, raw_json)
        VALUES (1, '20', 'Westbound', 10000, NULL, '{}')
        """
    )
    points = [
        (1, 1, "W", None, None, 41.0, -87.0, 0.0, 0, "{}"),
        (1, 2, "S", "456", "Test Stop", 41.0, -87.001, 1000.0, 0, "{}"),
        (1, 3, "W", None, None, 41.0, -87.002, 2000.0, 0, "{}"),
    ]
    conn.executemany(
        """
        INSERT INTO bus_v3_pattern_point(pid, seq, typ, stpid, stpnm, lat, lon, pdist_ft,
                                         is_detour_original_point, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        points,
    )
    conn.execute(
        """
        INSERT INTO bus_v3_stop(stpid, stpnm, lat, lon, rt, rtdir, raw_json)
        VALUES ('456', 'Test Stop', 41.0, -87.001, '20', 'Westbound', '{}')
        """
    )


def _seed_poll(conn: duckdb.DuckDBPyConnection, poll_id: int, endpoint: str, t_ms: int) -> None:
    conn.execute(
        """
        INSERT INTO bus_v3_api_poll(poll_id, run_id, endpoint, params_json_redacted,
                                    local_request_start_ms, local_response_end_ms,
                                    cta_server_time_ms, ok, created_at_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [poll_id, "r1", endpoint, "{}", t_ms, t_ms + 500, t_ms, True, t_ms],
    )


def _seed_prediction(
    conn: duckdb.DuckDBPyConnection,
    *,
    vid: str = "V1",
    dyn: int | None = 0,
    flagstop: int | None = 0,
) -> None:
    _seed_poll(conn, 1, "getpredictions", BASE_MS)
    conn.execute(
        """
        INSERT INTO bus_v3_prediction_observation(
            poll_id, run_id, cta_server_time_ms, local_response_end_ms, query_kind,
            tmstmp_ms, prediction_age_s, typ, stpid, stpnm, vid, dstp_ft, rt, rtdir, des,
            prdtm_ms, eta_s, prdctdn_raw, prdctdn_min, dly, dyn, tablockid, tatripid,
            origtatripno, stst, stsd, flagstop, raw_json
        ) VALUES (1, 'r1', ?, ?, 'predictions_by_stop', ?, 0, 'A', '456', 'Test Stop',
                  ?, 600, '20', 'Westbound', 'Austin', ?, 60, '1', 1, 0, ?, 'b', 't',
                  'o', 0, '2026-01-01', ?, '{}')
        """,
        [BASE_MS, BASE_MS, BASE_MS, vid, BASE_MS + 60_000, dyn, flagstop],
    )


def _seed_vehicle(
    conn: duckdb.DuckDBPyConnection,
    *,
    poll_id: int,
    tmstmp_ms: int,
    pdist_ft: float,
    lon: float,
    vehicle_age_s: float = 1.0,
    vid: str = "V1",
) -> None:
    _seed_poll(conn, poll_id, "getvehicles", tmstmp_ms)
    conn.execute(
        """
        INSERT INTO bus_v3_vehicle_observation(
            poll_id, run_id, cta_server_time_ms, local_response_end_ms, vid, tmstmp_ms,
            vehicle_age_s, lat, lon, hdg, pid, pdist_ft, rt, des, dly, tablockid, tatripid,
            origtatripno, stst, stsd, raw_json
        ) VALUES (?, 'r1', ?, ?, ?, ?, ?, 41.0, ?, 270, 1, ?, '20', 'Austin', 0, 'b',
                  't', 'o', 0, '2026-01-01', '{}')
        """,
        [
            poll_id,
            tmstmp_ms + 1000,
            tmstmp_ms + 1000,
            vid,
            tmstmp_ms,
            vehicle_age_s,
            lon,
            pdist_ft,
        ],
    )


def test_high_confidence_crossing(conn):
    """Two observations bracket the stop's pdist → one ARRIVED_CONFIRMED row."""
    _seed_pattern(conn)
    _seed_prediction(conn)
    _seed_vehicle(conn, poll_id=2, tmstmp_ms=BASE_MS + 10_000, pdist_ft=700.0, lon=-87.0007)
    _seed_vehicle(conn, poll_id=3, tmstmp_ms=BASE_MS + 40_000, pdist_ft=1200.0, lon=-87.0012)
    n = infer_bus_arrivals(conn, run_id="r1", replace=True)
    assert n == 1
    rows = conn.execute(
        "SELECT label, high_confidence, actual_arrival_ms, confidence FROM bus_v3_arrival_event"
    ).fetchall()
    assert len(rows) == 1
    label, high, actual, conf = rows[0]
    assert label == "ARRIVED_CONFIRMED"
    assert high is True
    assert conf == pytest.approx(0.97)
    # Interpolated arrival is at fraction (1000-700)/(1200-700) = 0.6 of
    # (40s - 10s), so actual = base+10s + 0.6 * 30s = base+28s.
    assert actual == BASE_MS + 28_000


def test_ghost_candidate_when_no_vehicle(conn):
    """Prediction exists, no matching vehicle obs → NO_EVIDENCE_GHOST_CANDIDATE."""
    _seed_pattern(conn)
    _seed_prediction(conn)
    n = infer_bus_arrivals(conn, run_id="r1", replace=True)
    assert n == 1
    rows = conn.execute(
        "SELECT label, high_confidence FROM bus_v3_arrival_event"
    ).fetchall()
    assert rows == [("NO_EVIDENCE_GHOST_CANDIDATE", False)]


def test_stale_data_label(conn):
    """All vehicle obs older than 90s → STALE_DATA, not ARRIVED_CONFIRMED."""
    _seed_pattern(conn)
    _seed_prediction(conn)
    # Both observations have vehicle_age_s = 200, exceeding the 90s freshness threshold.
    _seed_vehicle(
        conn, poll_id=2, tmstmp_ms=BASE_MS + 10_000, pdist_ft=700.0, lon=-87.0007,
        vehicle_age_s=200.0,
    )
    _seed_vehicle(
        conn, poll_id=3, tmstmp_ms=BASE_MS + 40_000, pdist_ft=1200.0, lon=-87.0012,
        vehicle_age_s=200.0,
    )
    n = infer_bus_arrivals(conn, run_id="r1", replace=True)
    assert n == 1
    label, high = conn.execute(
        "SELECT label, high_confidence FROM bus_v3_arrival_event"
    ).fetchone()
    assert label == "STALE_DATA"
    assert high is False


def test_dyn_canceled_does_not_emit_high_conf(conn):
    """dyn=1 (CANCELED) → never high_confidence, label = CANCELED_OR_INVALIDATED."""
    _seed_pattern(conn)
    _seed_prediction(conn, dyn=1)
    _seed_vehicle(conn, poll_id=2, tmstmp_ms=BASE_MS + 10_000, pdist_ft=700.0, lon=-87.0007)
    _seed_vehicle(conn, poll_id=3, tmstmp_ms=BASE_MS + 40_000, pdist_ft=1200.0, lon=-87.0012)
    n = infer_bus_arrivals(conn, run_id="r1", replace=True)
    assert n == 1
    label, high = conn.execute(
        "SELECT label, high_confidence FROM bus_v3_arrival_event"
    ).fetchone()
    assert label == "CANCELED_OR_INVALIDATED"
    assert high is False


def test_flagstop_only_discharge(conn):
    """flagstop=2 (PICK_UP_AND_DISCHARGE_ONLY) → PASSED_WITHOUT_PICKUP_OR_EXPRESSED."""
    _seed_pattern(conn)
    _seed_prediction(conn, flagstop=2)
    _seed_vehicle(conn, poll_id=2, tmstmp_ms=BASE_MS + 10_000, pdist_ft=700.0, lon=-87.0007)
    _seed_vehicle(conn, poll_id=3, tmstmp_ms=BASE_MS + 40_000, pdist_ft=1200.0, lon=-87.0012)
    n = infer_bus_arrivals(conn, run_id="r1", replace=True)
    label, high = conn.execute(
        "SELECT label, high_confidence FROM bus_v3_arrival_event"
    ).fetchone()
    assert label == "PASSED_WITHOUT_PICKUP_OR_EXPRESSED"
    assert high is False


def test_detour_removes_stop(conn):
    """Active detour removes the stop → DETOUR_AMBIGUOUS, no high_confidence."""
    _seed_pattern(conn)
    _seed_prediction(conn)
    _seed_vehicle(conn, poll_id=2, tmstmp_ms=BASE_MS + 10_000, pdist_ft=700.0, lon=-87.0007)
    _seed_vehicle(conn, poll_id=3, tmstmp_ms=BASE_MS + 40_000, pdist_ft=1200.0, lon=-87.0012)
    # Insert an active detour referencing this stop.
    conn.execute(
        """
        INSERT INTO bus_v3_detour(detour_pk, id, state, route_dirs_json, raw_json)
        VALUES ('D1:1', 'D1', 1, '[]', '{}')
        """
    )
    conn.execute(
        "UPDATE bus_v3_stop SET dtrrem_json = '[\"D1\"]' WHERE stpid = '456'"
    )
    n = infer_bus_arrivals(conn, run_id="r1", replace=True)
    label, high = conn.execute(
        "SELECT label, high_confidence FROM bus_v3_arrival_event"
    ).fetchone()
    assert label == "DETOUR_AMBIGUOUS"
    assert high is False


def test_idempotent_inference(conn):
    """Running infer_bus_arrivals twice without replace=True is a no-op on the second pass."""
    _seed_pattern(conn)
    _seed_prediction(conn)
    _seed_vehicle(conn, poll_id=2, tmstmp_ms=BASE_MS + 10_000, pdist_ft=700.0, lon=-87.0007)
    _seed_vehicle(conn, poll_id=3, tmstmp_ms=BASE_MS + 40_000, pdist_ft=1200.0, lon=-87.0012)
    n1 = infer_bus_arrivals(conn, run_id="r1", replace=False)
    n2 = infer_bus_arrivals(conn, run_id="r1", replace=False)
    assert n1 == 1
    assert n2 == 0
    total = conn.execute("SELECT COUNT(*) FROM bus_v3_arrival_event").fetchone()[0]
    assert total == 1
