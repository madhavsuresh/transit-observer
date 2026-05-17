"""Tiny helper for ``GET <url>`` → ``api_payloads_raw`` snapshots.

Used by collection sources where there's no structured API to call but
the raw HTML/ICS/CSV is itself the signal (slow-zone map page, McCormick
calendar, university academic calendars, etc.). The response body lands
in ``api_payloads_raw`` via the payload recorder — future processing
can parse it offline without re-fetching.
"""

from __future__ import annotations

from typing import Iterable

import duckdb
import httpx

from .payload_archive import make_response_recorder


async def snapshot_urls(
    conn: duckdb.DuckDBPyConnection,
    *,
    source: str,
    urls: Iterable[str],
    timeout: float = 30.0,
    follow_redirects: bool = True,
) -> int:
    """Fetch each URL once. Body lands in api_payloads_raw via the recorder.

    Returns the count of successful (2xx/3xx) responses.
    """
    recorder = make_response_recorder(conn, source=source)
    n = 0
    async with httpx.AsyncClient(
        timeout=timeout,
        event_hooks={"response": [recorder]},
        follow_redirects=follow_redirects,
    ) as http:
        for url in urls:
            try:
                resp = await http.get(url)
            except Exception:  # noqa: BLE001
                continue
            if resp.status_code < 400:
                n += 1
    return n
