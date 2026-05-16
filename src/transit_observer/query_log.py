"""Log API predict queries to NDJSON, then import into DuckDB.

The API runs in a separate process from the collector. DuckDB is a
single-writer database, so we don't let the API write to it directly.
Instead the API appends to ``data/queries.ndjson``; the collector
imports new lines into the ``query_log`` table on each tick.

Once imported, a separate function (``find_popular_ods``) ranks pending
OD pairs by query count over a recent window so the auto-upgrade path
can promote frequently-queried pairs into seeded corridors.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

from .config import settings
from .corpus import AdHocPrediction


log = logging.getLogger(__name__)


QUERIES_PATH = settings.data_dir / "queries.ndjson"
QUERIES_CURSOR_PATH = settings.data_dir / "queries.ndjson.cursor"


def append_query(
    *,
    queried_at: datetime,
    client_id: str | None,
    mode: str,
    line: str,
    boarding_int_id: int,
    boarding_text_id: str | None,
    alighting_int_id: int,
    alighting_text_id: str | None,
    prediction: AdHocPrediction | None,
    error_reason: str | None,
) -> str:
    """Append one query record to the NDJSON log. Returns the query_id.

    Called by the API on every predict request. Cheap: a single fsync at
    file close, no DB involvement.
    """
    query_id = str(uuid.uuid4())
    record: dict = {
        "query_id": query_id,
        "queried_at": queried_at.isoformat(),
        "client_id": client_id,
        "mode": mode,
        "line": line,
        "boarding_int_id": boarding_int_id,
        "boarding_text_id": boarding_text_id,
        "alighting_int_id": alighting_int_id,
        "alighting_text_id": alighting_text_id,
        "success": prediction is not None,
        "error_reason": error_reason,
    }
    if prediction is not None:
        record.update({
            "direction_code": prediction.direction_code,
            "boarding_station_name": prediction.boarding_label,
            "alighting_station_name": prediction.alighting_label,
            "predicted_wait_mean": prediction.predicted_wait_mean,
            "predicted_wait_p50": prediction.predicted_wait_p50,
            "predicted_wait_p80": prediction.predicted_wait_p80,
            "predicted_wait_p90": prediction.predicted_wait_p90,
            "predicted_in_vehicle_mean": prediction.predicted_in_vehicle_mean,
            "predicted_total_p50": prediction.predicted_total_p50,
            "predicted_total_p80": prediction.predicted_total_p80,
            "predicted_total_p90": prediction.predicted_total_p90,
            "predictor_version": prediction.predictor_version,
        })
    QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUERIES_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return query_id


def import_pending(conn: duckdb.DuckDBPyConnection) -> int:
    """Import every NDJSON record after the last cursor position.

    Idempotent: the cursor file tracks the byte offset already imported.
    If the NDJSON is rotated (size shrinks), we reset to 0.
    """
    if not QUERIES_PATH.exists():
        return 0
    cursor = _read_cursor()
    size = QUERIES_PATH.stat().st_size
    if size < cursor:
        # File was rotated/truncated -- reset.
        cursor = 0
    if size == cursor:
        return 0

    rows: list[tuple] = []
    new_cursor = cursor
    with QUERIES_PATH.open("rb") as f:
        f.seek(cursor)
        for raw_line in f:
            new_cursor += len(raw_line)
            try:
                rec = json.loads(raw_line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                log.warning("query_log.parse_error", err=str(exc))
                continue
            rows.append((
                rec.get("query_id"),
                rec.get("queried_at"),
                rec.get("client_id"),
                rec.get("mode"),
                rec.get("line"),
                rec.get("direction_code"),
                rec.get("boarding_int_id") or 0,
                rec.get("boarding_text_id"),
                rec.get("boarding_station_name"),
                rec.get("alighting_int_id") or 0,
                rec.get("alighting_text_id"),
                rec.get("alighting_station_name"),
                rec.get("predicted_wait_mean"),
                rec.get("predicted_wait_p50"),
                rec.get("predicted_wait_p80"),
                rec.get("predicted_wait_p90"),
                rec.get("predicted_in_vehicle_mean"),
                rec.get("predicted_total_p50"),
                rec.get("predicted_total_p80"),
                rec.get("predicted_total_p90"),
                rec.get("predictor_version"),
                bool(rec.get("success")),
                rec.get("error_reason"),
            ))

    if rows:
        conn.executemany(
            """
            INSERT INTO query_log (
                query_id, queried_at, client_id, mode, line, direction_code,
                boarding_int_id, boarding_text_id, boarding_station_name,
                alighting_int_id, alighting_text_id, alighting_station_name,
                predicted_wait_mean, predicted_wait_p50, predicted_wait_p80, predicted_wait_p90,
                predicted_in_vehicle_mean,
                predicted_total_p50, predicted_total_p80, predicted_total_p90,
                predictor_version, success, error_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (query_id) DO NOTHING
            """,
            rows,
        )
    _write_cursor(new_cursor)
    return len(rows)


def _read_cursor() -> int:
    if not QUERIES_CURSOR_PATH.exists():
        return 0
    try:
        return int(QUERIES_CURSOR_PATH.read_text().strip() or "0")
    except (ValueError, OSError):
        return 0


def _write_cursor(offset: int) -> None:
    QUERIES_CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = QUERIES_CURSOR_PATH.with_suffix(".tmp")
    tmp.write_text(str(offset))
    tmp.replace(QUERIES_CURSOR_PATH)


def find_popular_ods(
    conn: duckdb.DuckDBPyConnection,
    *,
    now: datetime,
    window: timedelta = timedelta(days=7),
    min_count: int = 50,
) -> list[dict]:
    """Return OD pairs that have been queried >= ``min_count`` times in the
    rolling window, **excluding** OD pairs that already match a corridor.

    Returns a list of dicts with the fields needed to build a new corridor.
    """
    cutoff = now - window
    rows = conn.execute(
        """
        SELECT q.mode, q.line, q.direction_code,
               q.boarding_int_id, q.boarding_text_id, q.boarding_station_name,
               q.alighting_int_id, q.alighting_text_id, q.alighting_station_name,
               COUNT(*) AS n
          FROM query_log q
         WHERE q.queried_at >= ?
           AND q.success = TRUE
           AND NOT EXISTS (
               SELECT 1 FROM corridors c
                WHERE c.mode = q.mode
                  AND c.line = q.line
                  AND c.boarding_int_id = q.boarding_int_id
                  AND COALESCE(c.boarding_text_id, '') = COALESCE(q.boarding_text_id, '')
                  AND c.alighting_int_id = q.alighting_int_id
                  AND COALESCE(c.alighting_text_id, '') = COALESCE(q.alighting_text_id, '')
           )
         GROUP BY q.mode, q.line, q.direction_code,
                  q.boarding_int_id, q.boarding_text_id, q.boarding_station_name,
                  q.alighting_int_id, q.alighting_text_id, q.alighting_station_name
        HAVING COUNT(*) >= ?
         ORDER BY n DESC
        """,
        [cutoff, min_count],
    ).fetchall()
    return [
        {
            "mode": mode, "line": line, "direction_code": direction,
            "boarding_int_id": b_int, "boarding_text_id": b_text, "boarding_station_name": b_name,
            "alighting_int_id": a_int, "alighting_text_id": a_text, "alighting_station_name": a_name,
            "count": n,
        }
        for (
            mode, line, direction,
            b_int, b_text, b_name,
            a_int, a_text, a_name,
            n,
        ) in rows
    ]
