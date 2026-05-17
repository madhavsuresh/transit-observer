"""Transit-account social media capture.

Supports two free, no-auth APIs:
- Bluesky (``public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed``)
- Mastodon (any instance, ``/api/v1/accounts/{id}/statuses``)

Why both: Twitter historical search is paywalled; once an alert tweet
scrolls past, it's painful to backfill. Capture forward-only into
``transit_social_raw``. Operators who haven't curated a list of
accounts will see this poll no-op cleanly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import httpx

from .config import CHICAGO


BLUESKY_BASE = "https://public.api.bsky.app/xrpc"


@dataclass(frozen=True)
class SocialAccount:
    """One configured account to poll. ``identifier`` is platform-specific:
    Bluesky handle (e.g. ``cta.bsky.social``) or Mastodon ``user@instance``
    (e.g. ``cta@mastodon.social``).
    """

    platform: str   # 'bluesky' | 'mastodon'
    identifier: str


@dataclass(frozen=True)
class SocialPost:
    platform: str
    handle: str
    post_id: str
    posted_at: datetime | None
    body: str | None
    url: str | None
    in_reply_to: str | None
    media_urls_json: str | None
    raw_payload_json: str


def _parse_iso_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # Bluesky uses RFC3339 with Z. Mastodon uses ISO8601 with Z or offset.
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(CHICAGO)
    except (ValueError, AttributeError):
        return None


class SocialClient:
    def __init__(
        self,
        *,
        http: httpx.AsyncClient | None = None,
        payload_recorder=None,
    ) -> None:
        if http is not None:
            self._http = http
        else:
            event_hooks: dict = {}
            if payload_recorder is not None:
                event_hooks["response"] = [payload_recorder]
            self._http = httpx.AsyncClient(timeout=15.0, event_hooks=event_hooks or None)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def fetch_posts(self, accounts: Iterable[SocialAccount], *, limit: int = 20) -> list[SocialPost]:
        out: list[SocialPost] = []
        for account in accounts:
            try:
                if account.platform == "bluesky":
                    out.extend(await self._fetch_bluesky(account.identifier, limit=limit))
                elif account.platform == "mastodon":
                    out.extend(await self._fetch_mastodon(account.identifier, limit=limit))
            except Exception:  # noqa: BLE001
                # individual account failures are routine; keep going
                continue
        return out

    async def _fetch_bluesky(self, handle: str, *, limit: int) -> list[SocialPost]:
        params = {"actor": handle, "limit": str(limit)}
        resp = await self._http.get(f"{BLUESKY_BASE}/app.bsky.feed.getAuthorFeed", params=params)
        resp.raise_for_status()
        payload = resp.json()
        out: list[SocialPost] = []
        for entry in payload.get("feed") or []:
            post = entry.get("post") or {}
            record = post.get("record") or {}
            uri = post.get("uri") or ""
            post_id = uri.rsplit("/", 1)[-1] if uri else ""
            if not post_id:
                continue
            media_urls: list[str] = []
            embed = post.get("embed") or {}
            for img in embed.get("images") or []:
                fullsize = img.get("fullsize")
                if fullsize:
                    media_urls.append(fullsize)
            in_reply_to = None
            reply = record.get("reply")
            if reply:
                parent_uri = (reply.get("parent") or {}).get("uri")
                if parent_uri:
                    in_reply_to = parent_uri
            out.append(
                SocialPost(
                    platform="bluesky",
                    handle=handle,
                    post_id=post_id,
                    posted_at=_parse_iso_dt(record.get("createdAt") or post.get("indexedAt")),
                    body=record.get("text"),
                    url=f"https://bsky.app/profile/{handle}/post/{post_id}" if post_id else None,
                    in_reply_to=in_reply_to,
                    media_urls_json=json.dumps(media_urls) if media_urls else None,
                    raw_payload_json=json.dumps(entry),
                )
            )
        return out

    async def _fetch_mastodon(self, user_at_instance: str, *, limit: int) -> list[SocialPost]:
        # 'user@instance' — split into the parts.
        if "@" not in user_at_instance:
            return []
        user, instance = user_at_instance.split("@", 1)
        base = f"https://{instance}"
        # Look up account id (no auth).
        lookup = await self._http.get(
            f"{base}/api/v1/accounts/lookup", params={"acct": user}
        )
        lookup.raise_for_status()
        account = lookup.json()
        account_id = account.get("id")
        if not account_id:
            return []
        statuses = await self._http.get(
            f"{base}/api/v1/accounts/{account_id}/statuses",
            params={"limit": str(limit), "exclude_replies": "false"},
        )
        statuses.raise_for_status()
        out: list[SocialPost] = []
        for status in statuses.json() or []:
            post_id = str(status.get("id") or "")
            if not post_id:
                continue
            media_urls = [m.get("url") for m in (status.get("media_attachments") or []) if m.get("url")]
            out.append(
                SocialPost(
                    platform="mastodon",
                    handle=user_at_instance,
                    post_id=post_id,
                    posted_at=_parse_iso_dt(status.get("created_at")),
                    body=status.get("content"),  # HTML body
                    url=status.get("url"),
                    in_reply_to=status.get("in_reply_to_id"),
                    media_urls_json=json.dumps(media_urls) if media_urls else None,
                    raw_payload_json=json.dumps(status),
                )
            )
        return out
