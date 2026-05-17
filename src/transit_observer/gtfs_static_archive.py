"""Versioned archive of GTFS-static feeds.

Why this lives outside ``api_payloads_raw``: GTFS-static zips are
megabytes each; we don't want them blowing up the payload table. We
hash the zip and keep one copy per content hash on disk, plus a row in
``gtfs_feed_versions`` per (agency, hash). Schedules change quarterly
and once an agency replaces a feed, the prior version is gone — so
even a single live snapshot has long-term value.

The poll cadence is weekly; the operator can re-run on demand.

Tier-1 widening: each newly archived zip is now also extracted into
the ``gtfs_static_*`` tables (stops/routes/trips/stop_times/shapes/
calendar). Set ``extract=False`` to keep the archive but skip the
extraction.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import datetime
from pathlib import Path

import duckdb
import httpx
import structlog

from .config import CHICAGO
from .gtfs_static_extract import extract_gtfs_archive


log = structlog.get_logger(__name__)


async def snapshot_gtfs_feeds(
    conn: duckdb.DuckDBPyConnection,
    *,
    feeds: tuple[tuple[str, str], ...],
    archive_dir: Path,
    extract: bool = True,
) -> int:
    """Download each (agency, url). Save only if the content hash is new.

    If ``extract=True`` (default) the new zip is also parsed into the
    ``gtfs_static_*`` tables on success. Extraction failures are logged
    but never abort the archive — we always want the bytes on disk so
    we can re-extract later.

    Returns the number of *new* versions archived.
    """
    if not feeds:
        return 0
    n_new = 0
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http:
        for agency, url in feeds:
            try:
                resp = await http.get(url)
                resp.raise_for_status()
            except Exception:  # noqa: BLE001
                continue
            blob = resp.content
            if not blob:
                continue
            sha256 = hashlib.sha256(blob).hexdigest()
            exists = conn.execute(
                "SELECT 1 FROM gtfs_feed_versions WHERE agency = ? AND sha256 = ? LIMIT 1",
                [agency, sha256],
            ).fetchone()
            if exists:
                continue
            agency_dir = archive_dir / agency
            agency_dir.mkdir(parents=True, exist_ok=True)
            target = agency_dir / f"{sha256}.zip"
            try:
                target.write_bytes(blob)
            except OSError:
                continue
            feed_version = _read_feed_version(blob)
            fetched_at = datetime.now(CHICAGO)
            conn.execute(
                """
                INSERT INTO gtfs_feed_versions (
                    agency, sha256, downloaded_at, file_size, feed_version,
                    source_url, archive_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    agency,
                    sha256,
                    fetched_at,
                    len(blob),
                    feed_version,
                    url,
                    str(target),
                ],
            )
            n_new += 1
            if extract:
                try:
                    extract_gtfs_archive(
                        conn,
                        agency=agency,
                        archive_path=target,
                        fetched_at_ms=int(fetched_at.timestamp() * 1000),
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "gtfs_static.extract_failed",
                        agency=agency, archive=str(target), err=str(exc),
                    )
    return n_new


def _read_feed_version(blob: bytes) -> str | None:
    """Pull ``feed_version`` from feed_info.txt if present in the GTFS zip."""
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            if "feed_info.txt" not in zf.namelist():
                return None
            with zf.open("feed_info.txt") as fp:
                header = fp.readline().decode("utf-8", errors="replace").strip().split(",")
                first = fp.readline().decode("utf-8", errors="replace").strip().split(",")
        if "feed_version" not in header:
            return None
        idx = header.index("feed_version")
        if idx >= len(first):
            return None
        return first[idx].strip().strip('"')
    except (zipfile.BadZipFile, KeyError, OSError):
        return None
