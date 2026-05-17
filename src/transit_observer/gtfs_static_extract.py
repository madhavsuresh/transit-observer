"""Extract a GTFS-static zip into the ``gtfs_static_*`` tables.

Companion to :mod:`gtfs_static_archive`: that module downloads + hashes
zips into ``data/gtfs_snapshots/<agency>/<sha256>.zip``; this module
parses one of those zips and projects it into normalized tables keyed
by ``snapshot_id``. The schedule (``stop_times``), official stop
coordinates (``stops``), line geometry (``shapes``), and service
calendars are all joinable from there.

We use DuckDB's native CSV reader for speed (``stop_times.txt`` is
typically multi-million rows on CTA). Each known table has an explicit
column list; missing columns in the source CSV become NULL, so the
parser is forward-compatible with optional GTFS fields. Type coercion
uses ``TRY_CAST`` so a malformed row doesn't kill the whole load.
"""

from __future__ import annotations

import hashlib
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Optional

import duckdb
import structlog


log = structlog.get_logger(__name__)


# (gtfs_column_name, duckdb_cast_type)
_STOPS_COLS = [
    ("stop_id", "TEXT"),
    ("stop_code", "TEXT"),
    ("stop_name", "TEXT"),
    ("stop_desc", "TEXT"),
    ("stop_lat", "DOUBLE"),
    ("stop_lon", "DOUBLE"),
    ("zone_id", "TEXT"),
    ("stop_url", "TEXT"),
    ("location_type", "INTEGER"),
    ("parent_station", "TEXT"),
    ("stop_timezone", "TEXT"),
    ("wheelchair_boarding", "INTEGER"),
    ("platform_code", "TEXT"),
]

_ROUTES_COLS = [
    ("route_id", "TEXT"),
    ("agency_id", "TEXT"),
    ("route_short_name", "TEXT"),
    ("route_long_name", "TEXT"),
    ("route_desc", "TEXT"),
    ("route_type", "INTEGER"),
    ("route_url", "TEXT"),
    ("route_color", "TEXT"),
    ("route_text_color", "TEXT"),
]

_TRIPS_COLS = [
    ("route_id", "TEXT"),
    ("service_id", "TEXT"),
    ("trip_id", "TEXT"),
    ("trip_headsign", "TEXT"),
    ("trip_short_name", "TEXT"),
    ("direction_id", "INTEGER"),
    ("block_id", "TEXT"),
    ("shape_id", "TEXT"),
    ("wheelchair_accessible", "INTEGER"),
    ("bikes_allowed", "INTEGER"),
]

_STOP_TIMES_COLS = [
    ("trip_id", "TEXT"),
    ("stop_sequence", "INTEGER"),
    ("arrival_time", "TEXT"),
    ("departure_time", "TEXT"),
    ("stop_id", "TEXT"),
    ("stop_headsign", "TEXT"),
    ("pickup_type", "INTEGER"),
    ("drop_off_type", "INTEGER"),
    ("continuous_pickup", "INTEGER"),
    ("continuous_drop_off", "INTEGER"),
    ("shape_dist_traveled", "DOUBLE"),
    ("timepoint", "INTEGER"),
]

_SHAPES_COLS = [
    ("shape_id", "TEXT"),
    ("shape_pt_sequence", "INTEGER"),
    ("shape_pt_lat", "DOUBLE"),
    ("shape_pt_lon", "DOUBLE"),
    ("shape_dist_traveled", "DOUBLE"),
]

_CALENDAR_COLS = [
    ("service_id", "TEXT"),
    ("monday", "INTEGER"),
    ("tuesday", "INTEGER"),
    ("wednesday", "INTEGER"),
    ("thursday", "INTEGER"),
    ("friday", "INTEGER"),
    ("saturday", "INTEGER"),
    ("sunday", "INTEGER"),
    ("start_date", "TEXT"),
    ("end_date", "TEXT"),
]

_CALENDAR_DATES_COLS = [
    ("service_id", "TEXT"),
    ("date", "TEXT"),
    ("exception_type", "INTEGER"),
]

_AGENCY_COLS = [
    ("agency_id", "TEXT"),
    ("agency_name", "TEXT"),
    ("agency_url", "TEXT"),
    ("agency_timezone", "TEXT"),
    ("agency_lang", "TEXT"),
    ("agency_phone", "TEXT"),
    ("agency_fare_url", "TEXT"),
    ("agency_email", "TEXT"),
]


def extract_gtfs_archive(
    conn: duckdb.DuckDBPyConnection,
    *,
    agency: str,
    archive_path: Path | str,
    fetched_at_ms: Optional[int] = None,
    replace_existing: bool = True,
) -> int:
    """Parse one GTFS zip into the ``gtfs_static_*`` tables.

    Args:
        agency: tag for this snapshot (``'cta'``, ``'metra'``, ``'pace'``).
        archive_path: path to the downloaded GTFS .zip.
        fetched_at_ms: when the zip was downloaded (defaults to now).
        replace_existing: if a snapshot with the same sha256 already
            exists, optionally delete its rows and re-extract.

    Returns the new ``snapshot_id`` (or the existing one if we re-used it).
    """
    path = Path(archive_path)
    if not path.exists():
        raise FileNotFoundError(f"GTFS archive not found: {path}")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    fetched_ms = fetched_at_ms or int(time.time() * 1000)

    existing = conn.execute(
        """
        SELECT snapshot_id FROM gtfs_static_snapshot
         WHERE agency = ? AND archive_sha256 = ?
         LIMIT 1
        """,
        [agency, sha],
    ).fetchone()
    if existing and not replace_existing:
        log.info("gtfs_static.skip_existing", agency=agency, sha256=sha)
        return int(existing[0])
    if existing and replace_existing:
        snapshot_id = int(existing[0])
        _purge_snapshot_rows(conn, snapshot_id)
    else:
        snapshot_id = _insert_snapshot(conn, agency, str(path), fetched_ms, sha)

    with _extract_to_tempdir(path) as csv_dir:
        feed_meta = _read_feed_info(csv_dir / "feed_info.txt")
        n_agency = _load_csv(conn, csv_dir / "agency.txt", snapshot_id, "gtfs_static_agency", _AGENCY_COLS)
        n_stops = _load_csv(conn, csv_dir / "stops.txt", snapshot_id, "gtfs_static_stops", _STOPS_COLS)
        n_routes = _load_csv(conn, csv_dir / "routes.txt", snapshot_id, "gtfs_static_routes", _ROUTES_COLS)
        n_trips = _load_csv(conn, csv_dir / "trips.txt", snapshot_id, "gtfs_static_trips", _TRIPS_COLS)
        n_stop_times = _load_csv(
            conn, csv_dir / "stop_times.txt", snapshot_id, "gtfs_static_stop_times", _STOP_TIMES_COLS,
        )
        n_shapes = _load_csv(conn, csv_dir / "shapes.txt", snapshot_id, "gtfs_static_shapes", _SHAPES_COLS)
        n_calendar = _load_csv(conn, csv_dir / "calendar.txt", snapshot_id, "gtfs_static_calendar", _CALENDAR_COLS)
        n_calendar_dates = _load_csv(
            conn, csv_dir / "calendar_dates.txt", snapshot_id, "gtfs_static_calendar_dates", _CALENDAR_DATES_COLS,
        )

    conn.execute(
        """
        UPDATE gtfs_static_snapshot
           SET extracted_at_ms = ?,
               feed_start_date = ?, feed_end_date = ?,
               feed_publisher_name = ?, feed_version = ?,
               n_stops = ?, n_routes = ?, n_trips = ?,
               n_stop_times = ?, n_shapes = ?, n_calendar = ?, n_calendar_dates = ?
         WHERE snapshot_id = ?
        """,
        [
            int(time.time() * 1000),
            feed_meta.get("feed_start_date"),
            feed_meta.get("feed_end_date"),
            feed_meta.get("feed_publisher_name"),
            feed_meta.get("feed_version"),
            n_stops, n_routes, n_trips, n_stop_times, n_shapes,
            n_calendar, n_calendar_dates,
            snapshot_id,
        ],
    )
    log.info(
        "gtfs_static.extracted",
        agency=agency, snapshot_id=snapshot_id, sha256=sha[:12],
        stops=n_stops, routes=n_routes, trips=n_trips,
        stop_times=n_stop_times, shapes=n_shapes,
        calendar=n_calendar, calendar_dates=n_calendar_dates,
        agency_rows=n_agency,
    )
    return snapshot_id


def _insert_snapshot(
    conn: duckdb.DuckDBPyConnection,
    agency: str,
    archive_path: str,
    fetched_at_ms: int,
    sha256: str,
) -> int:
    row = conn.execute(
        """
        INSERT INTO gtfs_static_snapshot(agency, archive_path, fetched_at_ms, archive_sha256)
        VALUES (?, ?, ?, ?)
        RETURNING snapshot_id
        """,
        [agency, archive_path, fetched_at_ms, sha256],
    ).fetchone()
    if row is None or row[0] is None:
        raise RuntimeError("Failed to insert gtfs_static_snapshot row")
    return int(row[0])


def _purge_snapshot_rows(conn: duckdb.DuckDBPyConnection, snapshot_id: int) -> None:
    for table in (
        "gtfs_static_agency",
        "gtfs_static_stops",
        "gtfs_static_routes",
        "gtfs_static_trips",
        "gtfs_static_stop_times",
        "gtfs_static_shapes",
        "gtfs_static_calendar",
        "gtfs_static_calendar_dates",
    ):
        conn.execute(f"DELETE FROM {table} WHERE snapshot_id = ?", [snapshot_id])


@contextmanager
def _extract_to_tempdir(zip_path: Path) -> Iterator[Path]:
    with TemporaryDirectory(prefix="gtfs_static_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)
        yield tmp_path


def _csv_header(csv_path: Path) -> set[str]:
    with csv_path.open("r", encoding="utf-8-sig") as f:
        line = f.readline().strip()
    return {col.strip().strip('"') for col in line.split(",") if col.strip()}


def _project(col: str, header: set[str], cast_type: str) -> str:
    """Build a SELECT expression that:
    - reads ``col`` from the CSV when present, else NULL
    - turns the empty string into NULL (GTFS uses '' for missing)
    - try-casts to the target type so a single bad row can't fail the load
    """
    quoted = f'"{col}"'
    if col not in header:
        return f"CAST(NULL AS {cast_type}) AS {col}"
    if cast_type == "TEXT":
        return f"NULLIF({quoted}, '') AS {col}"
    return f"TRY_CAST(NULLIF({quoted}, '') AS {cast_type}) AS {col}"


def _load_csv(
    conn: duckdb.DuckDBPyConnection,
    csv_path: Path,
    snapshot_id: int,
    target_table: str,
    columns: list[tuple[str, str]],
) -> int:
    """Bulk-insert one CSV into one ``gtfs_static_*`` table."""
    if not csv_path.exists():
        return 0
    header = _csv_header(csv_path)
    if not header:
        return 0
    select_exprs = ["?"] + [_project(c, header, t) for c, t in columns]
    insert_cols = ["snapshot_id"] + [c for c, _ in columns]
    sql = f"""
        INSERT INTO {target_table} ({', '.join(insert_cols)})
        SELECT {', '.join(select_exprs)}
          FROM read_csv(?, header=true, all_varchar=true, ignore_errors=true)
    """
    conn.execute(sql, [snapshot_id, str(csv_path)])
    n_row = conn.execute(
        f"SELECT COUNT(*) FROM {target_table} WHERE snapshot_id = ?",
        [snapshot_id],
    ).fetchone()
    return int(n_row[0]) if n_row else 0


def _read_feed_info(path: Path) -> dict[str, str | None]:
    """Pull a few fields from ``feed_info.txt`` if present (optional file)."""
    out: dict[str, str | None] = {
        "feed_publisher_name": None,
        "feed_version": None,
        "feed_start_date": None,
        "feed_end_date": None,
    }
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            header = [c.strip().strip('"') for c in f.readline().strip().split(",")]
            row = f.readline().strip()
        if not row:
            return out
        # Naive CSV parse (feed_info rows don't contain commas in practice).
        values = [c.strip().strip('"') for c in row.split(",")]
        zipped = dict(zip(header, values))
        for key in out:
            v = zipped.get(key)
            out[key] = v if v else None
    except OSError:
        return out
    return out
