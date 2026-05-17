"""Parse CTA slow-zone HTML snapshots into ``train_v2_slow_zone`` rows.

The CTA publishes its slow-zone list at
``transitchicago.com/yourtimedeservesatrain/``. We snapshot the raw
HTML into ``api_payloads_raw`` on the daily cadence; this module reads
the most recent snapshot and projects what it can into structured rows.

The parser is **deliberately tolerant**: the CTA page layout shifts
periodically, and a brittle parser would silently drop new rows. Every
detected table row is stored — fields we can identify (line, direction,
max_mph, posted/clear dates) go in the structured columns; the
remaining cell text always lands in ``raw_payload_json`` so a future
pass can re-extract anything we missed.

Run with ``transit train-v2 parse-slow-zones`` (CLI) or call
``parse_slow_zones`` directly. The snapshot row is keyed on a stable
``slow_zone_id`` derived from line + segment, so rerunning the parser
upserts in place.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from typing import Any, Optional

import duckdb
import structlog


log = structlog.get_logger(__name__)


_LINE_NAMES = {
    "red": "Red", "blue": "Blue", "brown": "Brn", "green": "G",
    "orange": "Org", "purple": "P", "pink": "Pink", "yellow": "Y",
}

_DIRECTION_WORDS = {
    "northbound": "NB", "n/b": "NB", "northbnd": "NB",
    "southbound": "SB", "s/b": "SB", "southbnd": "SB",
    "eastbound": "EB", "e/b": "EB",
    "westbound": "WB", "w/b": "WB",
    "inbound": "IB", "outbound": "OB",
    "both": "BOTH", "both directions": "BOTH",
}

_MPH_RE = re.compile(r"(\d+)\s*mph", re.IGNORECASE)
_DATE_PATTERNS = [
    ("%m/%d/%Y", re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")),
    ("%m/%d/%y", re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2})\b")),
    ("%Y-%m-%d", re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")),
    ("%B %d, %Y", re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b")),
]

_BETWEEN_RE = re.compile(r"\bbetween\s+(.+?)\s+(?:and|&|to)\s+(.+)", re.IGNORECASE)


class _TableRowParser(HTMLParser):
    """Walks an HTML document and accumulates the text of every
    ``<tr>``-bounded row as a list of cell strings. Robust against
    missing closing tags and arbitrary nested formatting inside cells.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: Optional[list[str]] = None
        self._in_cell = False
        self._cell_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t == "tr":
            self._flush_row()
            self._row = []
        elif t in ("td", "th") and self._row is not None:
            self._flush_cell()
            self._in_cell = True
            self._cell_buf = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "tr":
            self._flush_row()
        elif t in ("td", "th"):
            self._flush_cell()

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_buf.append(data)

    def _flush_cell(self) -> None:
        if not self._in_cell or self._row is None:
            self._in_cell = False
            self._cell_buf = []
            return
        text = unescape(" ".join(self._cell_buf))
        text = re.sub(r"\s+", " ", text).strip()
        self._row.append(text)
        self._in_cell = False
        self._cell_buf = []

    def _flush_row(self) -> None:
        if self._row is not None and self._row:
            # Drop pure-header / pure-blank rows.
            if any(c.strip() for c in self._row):
                self.rows.append(self._row)
        self._row = None


def extract_rows(html: str) -> list[list[str]]:
    """Return every ``<tr>``-bounded row in ``html`` as a list of cells."""
    parser = _TableRowParser()
    parser.feed(html)
    parser._flush_cell()
    parser._flush_row()
    return parser.rows


def _classify_line(text: str) -> Optional[str]:
    low = text.lower()
    for name, code in _LINE_NAMES.items():
        if name in low:
            return code
    return None


def _classify_direction(text: str) -> Optional[str]:
    low = text.lower()
    for word, code in _DIRECTION_WORDS.items():
        if word in low:
            return code
    return None


def _extract_mph(text: str) -> Optional[float]:
    m = _MPH_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _extract_dates(text: str) -> list[int]:
    """Return list of ms-epoch timestamps found in ``text``."""
    out: list[int] = []
    for fmt, pattern in _DATE_PATTERNS:
        for match in pattern.findall(text):
            value = match[0] if isinstance(match, tuple) else match
            try:
                dt = datetime.strptime(value, fmt)
                out.append(int(dt.timestamp() * 1000))
            except ValueError:
                continue
    return out


def _extract_between(text: str) -> tuple[Optional[str], Optional[str]]:
    """Pull (from_station, to_station) out of "between X and Y" phrases."""
    m = _BETWEEN_RE.search(text)
    if not m:
        return None, None
    a = m.group(1).strip().rstrip(",")
    b = m.group(2).strip().rstrip(".").rstrip(",")
    return a, b


def classify_row(cells: list[str]) -> dict[str, Any]:
    """Best-effort extraction of slow-zone fields from one row's cells.

    Returns a dict with the structured fields we found plus the raw
    cells. The caller always writes ``raw_payload_json`` so a future
    pass can re-parse cells we couldn't classify.
    """
    joined = " | ".join(c for c in cells if c)
    line = None
    direction = None
    max_mph = None
    posted_ms = None
    clear_ms = None
    from_station = None
    to_station = None
    description = joined

    for cell in cells:
        if line is None:
            line = _classify_line(cell)
        if direction is None:
            direction = _classify_direction(cell)
        if max_mph is None:
            max_mph = _extract_mph(cell)
        if from_station is None or to_station is None:
            a, b = _extract_between(cell)
            from_station = from_station or a
            to_station = to_station or b

    dates_ms = _extract_dates(joined)
    if dates_ms:
        posted_ms = min(dates_ms)
        if len(dates_ms) > 1:
            clear_ms = max(dates_ms)

    return {
        "line": line,
        "direction_code": direction,
        "max_mph": max_mph,
        "posted_at_ms": posted_ms,
        "expected_clear_at_ms": clear_ms,
        "from_station": from_station,
        "to_station": to_station,
        "description": description,
        "raw_cells": cells,
    }


def _slow_zone_id(record: dict[str, Any]) -> str:
    """Deterministic id so re-runs upsert instead of duplicating."""
    parts = [
        str(record.get("line") or ""),
        str(record.get("direction_code") or ""),
        str(record.get("from_station") or ""),
        str(record.get("to_station") or ""),
        f"{record.get('max_mph')}" if record.get("max_mph") is not None else "",
        str(record.get("description") or "")[:120],
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:24]


def _latest_snapshot(conn: duckdb.DuckDBPyConnection) -> Optional[tuple[int, str]]:
    """Return (polled_at_ms, html_body) of the most recent slow-zone snapshot,
    or None if none has been captured yet."""
    row = conn.execute(
        """
        SELECT polled_at, response_body
          FROM api_payloads_raw
         WHERE source = 'cta_slow_zones_page'
           AND response_body IS NOT NULL
         ORDER BY polled_at DESC
         LIMIT 1
        """
    ).fetchone()
    if row is None or row[1] is None:
        return None
    polled_at_ms = int(row[0].timestamp() * 1000) if row[0] else int(time.time() * 1000)
    return polled_at_ms, str(row[1])


def parse_slow_zones(
    conn: duckdb.DuckDBPyConnection,
    *,
    html: Optional[str] = None,
    polled_at_ms: Optional[int] = None,
    poll_id: Optional[int] = None,
) -> int:
    """Parse a slow-zone HTML snapshot and upsert rows into
    ``train_v2_slow_zone``.

    If ``html`` is None, reads the most recent
    ``api_payloads_raw.source='cta_slow_zones_page'`` row. Returns the
    number of rows we considered slow-zone records (some rows in the
    table may be headers we couldn't classify — those land with a
    NULL ``line`` and only the ``description`` populated).
    """
    if html is None:
        latest = _latest_snapshot(conn)
        if latest is None:
            log.info("train_v2.slow_zones.no_snapshot")
            return 0
        polled_at_ms, html = latest
    polled_at_ms = polled_at_ms or int(time.time() * 1000)
    rows = extract_rows(html)
    n_kept = 0
    for cells in rows:
        record = classify_row(cells)
        # Skip pure header / nav noise: keep only rows that look like
        # they have at least one signal (line, mph, or between-stations).
        if not (record["line"] or record["max_mph"] or record["from_station"]):
            continue
        slow_zone_id = _slow_zone_id(record)
        conn.execute(
            """
            INSERT INTO train_v2_slow_zone(
                slow_zone_id, line, direction_code, from_station, to_station,
                max_mph, posted_at_ms, expected_clear_at_ms, description,
                raw_payload_json, first_seen_poll_id, last_seen_poll_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (slow_zone_id) DO UPDATE SET
                direction_code = COALESCE(excluded.direction_code, train_v2_slow_zone.direction_code),
                from_station = COALESCE(excluded.from_station, train_v2_slow_zone.from_station),
                to_station = COALESCE(excluded.to_station, train_v2_slow_zone.to_station),
                max_mph = COALESCE(excluded.max_mph, train_v2_slow_zone.max_mph),
                posted_at_ms = COALESCE(excluded.posted_at_ms, train_v2_slow_zone.posted_at_ms),
                expected_clear_at_ms = COALESCE(excluded.expected_clear_at_ms, train_v2_slow_zone.expected_clear_at_ms),
                description = excluded.description,
                raw_payload_json = excluded.raw_payload_json,
                last_seen_poll_id = excluded.last_seen_poll_id
            """,
            [
                slow_zone_id,
                record["line"] or "?",  # NOT NULL column; "?" is the unknown sentinel
                record["direction_code"],
                record["from_station"],
                record["to_station"],
                record["max_mph"],
                record["posted_at_ms"],
                record["expected_clear_at_ms"],
                record["description"][:500],
                json.dumps(record["raw_cells"], ensure_ascii=False),
                poll_id, poll_id,
            ],
        )
        n_kept += 1
    log.info(
        "train_v2.slow_zones.parsed",
        rows_seen=len(rows),
        rows_kept=n_kept,
        polled_at_ms=polled_at_ms,
    )
    return n_kept
