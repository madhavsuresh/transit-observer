"""Async httpx client for CTA Bus Tracker v3.

API key resolution: ``CTA_BUS_API_KEY`` env var (preferred; never logged)
or ``Settings.cta_bus_api_key`` from config.toml. The raw key never
leaves the client — ``ApiCallResult`` stores only redacted forms.

The client is endpoint-thin: each method packs query params, calls the
shared ``_request`` with a ``query_kind`` discriminator, and returns an
``ApiCallResult``. Parsing is deferred to the normalizer.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

from .models import ApiCallResult
from .util import json_dumps, now_ms, redact_params, redact_url, root_of, safe_int


class CTABusV3Error(RuntimeError):
    pass


class CTABusV3Client:
    """CTA Bus Tracker v3 JSON client (async).

    Use as: ``async with CTABusV3Client(api_key) as client: ...``
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://www.ctabustracker.com/bustime/api/v3",
        timeout_s: float = 15.0,
        user_agent: str = "transit-observer-bus-v3/0.1",
        http: Optional[httpx.AsyncClient] = None,
        retries: int = 2,
        backoff_s: float = 0.8,
    ) -> None:
        if not api_key:
            raise CTABusV3Error("CTA_BUS_API_KEY is required")
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s
        self._retries = retries
        self._backoff = backoff_s
        self._owned_http = http is None
        self._http = http or httpx.AsyncClient(
            timeout=timeout_s, headers={"User-Agent": user_agent}
        )

    async def __aenter__(self) -> "CTABusV3Client":
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
        cta_server_time_ms: Optional[int],
    ) -> ApiCallResult:
        params = dict(params)
        params.setdefault("format", "json")
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
            params_redacted=redacted_params,
            query_kind=query_kind,
            request_url_redacted=redacted_url,
            local_request_start_ms=start,
            local_response_end_ms=end,
            cta_server_time_ms=cta_server_time_ms,
            http_status=status,
            latency_ms=float(end - start),
            ok=ok,
            json_data=data,
            error_message=err,
        )

    async def gettime(self) -> ApiCallResult:
        return await self._request(
            "gettime",
            {"unixTime": "true"},
            query_kind="server_time",
            cta_server_time_ms=None,
        )

    @staticmethod
    def extract_server_time_ms(result: ApiCallResult) -> Optional[int]:
        root = root_of(result.json_data)
        return safe_int(root.get("tm"))

    async def getroutes(self, *, cta_server_time_ms: Optional[int] = None) -> ApiCallResult:
        return await self._request("getroutes", {}, query_kind="routes", cta_server_time_ms=cta_server_time_ms)

    async def getdirections(
        self,
        rt: str,
        *,
        cta_server_time_ms: Optional[int] = None,
    ) -> ApiCallResult:
        return await self._request(
            "getdirections",
            {"rt": rt},
            query_kind="directions",
            cta_server_time_ms=cta_server_time_ms,
        )

    async def getstops(
        self,
        *,
        rt: Optional[str] = None,
        direction: Optional[str] = None,
        stpids: Optional[list[str]] = None,
        cta_server_time_ms: Optional[int] = None,
    ) -> ApiCallResult:
        params: dict[str, Any] = {}
        if stpids:
            params["stpid"] = ",".join(map(str, stpids))
            qk = "stops_by_id"
        else:
            if not rt or not direction:
                raise ValueError("getstops requires either stpids or rt+direction")
            params["rt"] = rt
            params["dir"] = direction
            qk = "stops_by_route_direction"
        return await self._request("getstops", params, query_kind=qk, cta_server_time_ms=cta_server_time_ms)

    async def getpatterns(
        self,
        *,
        rt: Optional[str] = None,
        pids: Optional[list[int | str]] = None,
        cta_server_time_ms: Optional[int] = None,
    ) -> ApiCallResult:
        params: dict[str, Any] = {}
        if pids:
            params["pid"] = ",".join(map(str, pids))
            qk = "patterns_by_id"
        else:
            if not rt:
                raise ValueError("getpatterns requires rt or pids")
            params["rt"] = rt
            qk = "patterns_by_route"
        return await self._request("getpatterns", params, query_kind=qk, cta_server_time_ms=cta_server_time_ms)

    async def getvehicles(
        self,
        *,
        routes: Optional[list[str]] = None,
        vids: Optional[list[str]] = None,
        cta_server_time_ms: Optional[int] = None,
    ) -> ApiCallResult:
        params: dict[str, Any] = {"tmres": "s"}
        if vids:
            params["vid"] = ",".join(map(str, vids))
            qk = "vehicles_by_id"
        else:
            if not routes:
                raise ValueError("getvehicles requires routes or vids")
            params["rt"] = ",".join(map(str, routes))
            qk = "vehicles_by_route"
        return await self._request("getvehicles", params, query_kind=qk, cta_server_time_ms=cta_server_time_ms)

    async def getpredictions(
        self,
        *,
        stpids: Optional[list[str]] = None,
        routes: Optional[list[str]] = None,
        vids: Optional[list[str]] = None,
        top: Optional[int] = None,
        cta_server_time_ms: Optional[int] = None,
    ) -> ApiCallResult:
        params: dict[str, Any] = {"tmres": "s", "unixTime": "true"}
        if vids:
            params["vid"] = ",".join(map(str, vids))
            qk = "predictions_by_vehicle"
        else:
            if not stpids:
                raise ValueError("getpredictions requires stpids or vids")
            params["stpid"] = ",".join(map(str, stpids))
            if routes:
                params["rt"] = ",".join(map(str, routes))
            qk = "predictions_by_stop"
        if top is not None:
            params["top"] = int(top)
        return await self._request("getpredictions", params, query_kind=qk, cta_server_time_ms=cta_server_time_ms)

    async def getdetours(
        self,
        *,
        rt: Optional[str] = None,
        direction: Optional[str] = None,
        cta_server_time_ms: Optional[int] = None,
    ) -> ApiCallResult:
        params: dict[str, Any] = {}
        if rt:
            params["rt"] = rt
        if direction:
            if not rt:
                raise ValueError("getdetours direction requires rt")
            params["rtdir"] = direction
        return await self._request("getdetours", params, query_kind="detours", cta_server_time_ms=cta_server_time_ms)

    async def getenhanceddetours(self, *, cta_server_time_ms: Optional[int] = None) -> ApiCallResult:
        return await self._request("getenhanceddetours", {}, query_kind="enhanced_detours", cta_server_time_ms=cta_server_time_ms)


def serialize_params_for_storage(params: dict[str, Any]) -> str:
    """Stable JSON encoding used when writing api_poll rows."""
    return json_dumps(redact_params(params))
