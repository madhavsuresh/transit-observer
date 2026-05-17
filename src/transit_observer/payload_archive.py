"""Archive every external HTTP response into ``api_payloads_raw``.

Wired in via ``httpx`` response event hooks at client init. One row per
HTTP call. Lets future feature extraction re-parse fields we currently
drop without needing to re-poll (impossible for time-of-day signal).

API keys (URL ``key`` param and ``Authorization`` header) are scrubbed
before storage. Non-text bodies (protobuf) are base64-encoded with a
``BASE64:`` prefix.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Awaitable, Callable

import duckdb
import httpx

from .config import CHICAGO


PayloadRecorder = Callable[[httpx.Response], Awaitable[None]]


_REDACT = {"key", "apikey", "api_key", "token", "access_token"}


def _scrub_params(params: httpx.QueryParams) -> str:
    safe: dict[str, str] = {}
    for k, v in params.multi_items():
        safe[k] = "<redacted>" if k.lower() in _REDACT else v
    return json.dumps(safe)


def _body_to_text(response: httpx.Response) -> str | None:
    content_type = response.headers.get("content-type", "").lower()
    if any(t in content_type for t in ("text/", "json", "xml", "html", "javascript")):
        try:
            return response.text
        except (UnicodeDecodeError, LookupError):
            pass
    raw = response.content
    if not raw:
        return None
    return "BASE64:" + base64.b64encode(raw).decode("ascii")


def make_response_recorder(
    conn: duckdb.DuckDBPyConnection,
    *,
    source: str,
) -> PayloadRecorder:
    """Build an httpx response hook that inserts into ``api_payloads_raw``.

    The hook never raises; archive failures shouldn't break the underlying
    poll. The caller's DuckDB connection is used directly (single-writer
    pattern; safe to share within the asyncio collector loop).
    """

    async def _hook(response: httpx.Response) -> None:
        try:
            request = response.request
            try:
                await response.aread()
            except Exception:  # noqa: BLE001
                pass
            try:
                latency_ms = response.elapsed.total_seconds() * 1000.0
            except RuntimeError:
                latency_ms = None
            body = _body_to_text(response)
            conn.execute(
                """
                INSERT INTO api_payloads_raw (
                    polled_at, source, endpoint, request_params_json,
                    response_body, http_status, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    datetime.now(CHICAGO),
                    source,
                    str(request.url.path),
                    _scrub_params(request.url.params),
                    body,
                    response.status_code,
                    latency_ms,
                ],
            )
        except Exception:  # noqa: BLE001
            pass

    return _hook
