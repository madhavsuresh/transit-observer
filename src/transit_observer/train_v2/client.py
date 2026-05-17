"""Async httpx client for the three CTA Train Tracker endpoints.

The legacy ``cta_train_client.CTATrainClient`` covers ttarrivals and
ttpositions but not ``ttfollow.aspx`` — the per-run trajectory endpoint
that returns the train's predicted arrivals at *all* upcoming stations
on its current trip. That's the train analog of bus_v3's by-vid
predictions and is the single biggest gap in the legacy pipeline.

This client returns ``ApiCallResult`` records the way bus_v3 does, so
the normalizer can write the raw JSON to ``train_v2_api_poll`` and the
estimator can replay polls against a snapshotted database.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable, Optional
from urllib.parse import urlencode

import httpx

from .models import ApiCallResult
from .util import json_dumps, now_ms, redact_params, redact_url


BASE_URL = "http://lapi.transitchicago.com/api/1.0"


class CTATrainV2Error(RuntimeError):
    pass


class CTATrainV2Client:
    """Async client for ``ttarrivals.aspx``, ``ttfollow.aspx``, and
    ``ttpositions.aspx``.

    Rate-limited at 100 requests / 5 min per key. The collector is
    responsible for staying under the budget; this client just makes
    the request, retries 3x with exponential backoff, and packages the
    result.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        timeout_s: float = 15.0,
        user_agent: str = "transit-observer-train-v2/0.1",
        http: Optional[httpx.AsyncClient] = None,
        retries: int = 2,
        backoff_s: float = 0.5,
    ) -> None:
        if not api_key:
            raise CTATrainV2Error("CTA_TRAIN_API_KEY is required")
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s
        self._retries = retries
        self._backoff = backoff_s
        self._owned_http = http is None
        self._http = http or httpx.AsyncClient(
            timeout=timeout_s, headers={"User-Agent": user_agent}
        )

    async def __aenter__(self) -> "CTATrainV2Client":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owned_http:
            await self._http.aclose()

    def _url(self, endpoint: str) -> str:
        return f"{self._base}/{endpoint.lstrip('/')}"

    async def _request(
        self,
        endpoint: str,
        params: dict[str, Any],
        *,
        query_kind: Optional[str],
    ) -> ApiCallResult:
        params = dict(params)
        params.setdefault("outputType", "JSON")
        params["key"] = self._key

        url = self._url(endpoint)
        redacted_params = redact_params(params)
        redacted_url = redact_url(f"{url}?{urlencode(params, doseq=True)}")

        start = now_ms()
        status: Optional[int] = None
        data: Optional[dict[str, Any]] = None
        err: Optional[str] = None
        ok = False
        for attempt in range(self._retries + 1):
            try:
                resp = await self._http.get(url, params=params, timeout=self._timeout)
                status = resp.status_code
                resp.raise_for_status()
                data = resp.json()
                ok = True
                err = None
                break
            except Exception as exc:  # noqa: BLE001 — deliberate API boundary
                err = str(exc)
                ok = False
                if attempt < self._retries:
                    await asyncio.sleep(self._backoff * (2**attempt))
                else:
                    break
        end = now_ms()
        return ApiCallResult(
            endpoint=endpoint,
            source="train_tracker",
            params_redacted=redacted_params,
            query_kind=query_kind,
            request_url_redacted=redacted_url,
            local_request_start_ms=start,
            local_response_end_ms=end,
            cta_server_time_ms=None,  # ttarrivals/ttpositions include ``tmst`` in the body;
                                       # the normalizer fills this in after parsing.
            http_status=status,
            latency_ms=float(end - start),
            ok=ok,
            json_data=data,
            raw_bytes=None,
            error_message=err,
        )

    async def ttarrivals(
        self,
        *,
        map_id: int,
        max_predictions: int = 12,
    ) -> ApiCallResult:
        """Predictions for a station (all platforms)."""
        return await self._request(
            "ttarrivals.aspx",
            {"mapid": str(map_id), "max": str(max_predictions)},
            query_kind="arrivals_by_station",
        )

    async def ttarrivals_by_stop(
        self,
        *,
        stop_id: int,
        max_predictions: int = 12,
    ) -> ApiCallResult:
        """Predictions for a single platform (one direction)."""
        return await self._request(
            "ttarrivals.aspx",
            {"stpid": str(stop_id), "max": str(max_predictions)},
            query_kind="arrivals_by_stop",
        )

    async def ttfollow(self, *, run_number: str) -> ApiCallResult:
        """Per-run trajectory: predicted arrivals at *all* upcoming stops.

        This is the train analog of bus_v3's ``getpredictions(vid=…)``.
        Captures the train's full intended path so the estimator can
        cross-validate the by-station prediction against the by-run
        prediction.
        """
        return await self._request(
            "ttfollow.aspx",
            {"runnumber": str(run_number)},
            query_kind="follow_by_run",
        )

    async def ttpositions(self, *, line_codes: Iterable[str]) -> ApiCallResult:
        """All trains on the listed lines with lat/lon + ``nextStaId``."""
        return await self._request(
            "ttpositions.aspx",
            {"rt": ",".join(line_codes)},
            query_kind="positions_by_line",
        )
