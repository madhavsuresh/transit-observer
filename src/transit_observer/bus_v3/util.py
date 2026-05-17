"""Pure-Python utilities for the v3 bus pipeline.

Ports the validator's ``util.py`` verbatim except for the request budget
helper, which is bus-specific. No DB coupling here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Optional
from zoneinfo import ZoneInfo

CHICAGO_TZ = ZoneInfo("America/Chicago")


def now_ms() -> int:
    return int(time.time() * 1000)


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def json_sha256(obj: Any) -> str:
    return hashlib.sha256(json_dumps(obj).encode("utf-8")).hexdigest()


def redact_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: ("<redacted>" if k.lower() == "key" else v) for k, v in params.items()}


def redact_url(url: str) -> str:
    return re.sub(r"([?&]key=)[^&]+", r"\1<redacted>", url)


def as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def chunked(xs: Iterable[Any], n: int) -> Iterator[list[Any]]:
    buf: list[Any] = []
    for x in xs:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def safe_int(x: Any) -> Optional[int]:
    if x is None or x == "":
        return None
    try:
        if isinstance(x, bool):
            return int(x)
        return int(float(str(x)))
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def safe_bool_int(x: Any) -> Optional[int]:
    if x is None or x == "":
        return None
    if isinstance(x, bool):
        return int(x)
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return 1
    if s in {"false", "0", "no", "n"}:
        return 0
    return None


def parse_cta_timestamp_ms(value: Any, assume_tz: ZoneInfo = CHICAGO_TZ) -> Optional[int]:
    """Parse CTA timestamp fields.

    With ``unixTime=true`` prediction fields are epoch milliseconds.
    Vehicle timestamp fields usually remain local strings with
    ``tmres=s``. This function accepts both.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = int(value)
        return v * 1000 if v < 10_000_000_000 else v
    s = str(value).strip()
    if not s or s.lower() == "null":
        return None
    if s.isdigit():
        v = int(s)
        return v * 1000 if v < 10_000_000_000 else v
    for fmt in ("%Y%m%d %H:%M:%S", "%Y%m%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=assume_tz)
            return int(dt.astimezone(timezone.utc).timestamp() * 1000)
        except ValueError:
            pass
    return None


def format_iso_ms(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def parse_prdctdn_minutes(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    s = str(x).strip().upper()
    if s == "DUE":
        return 0.0
    if s in {"DLY", "DELAYED"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    xs = sorted(v for v in values if v is not None and math.isfinite(v))
    if not xs:
        return None
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def quantile(values: list[float], q: float) -> Optional[float]:
    xs = sorted(v for v in values if v is not None and math.isfinite(v))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def root_of(response_json: dict[str, Any] | None) -> dict[str, Any]:
    if not response_json:
        return {}
    return response_json.get("bustime-response", response_json)


def pick_first(root: dict[str, Any], *names: str) -> list[Any]:
    for name in names:
        if name in root:
            return as_list(root.get(name))
    return []


def horizon_bin(horizon_s: Optional[float]) -> str:
    if horizon_s is None:
        return "unknown"
    m = horizon_s / 60.0
    if m < 0:
        return "past"
    if m <= 2:
        return "0_2m"
    if m <= 5:
        return "2_5m"
    if m <= 10:
        return "5_10m"
    if m <= 20:
        return "10_20m"
    return "20m_plus"


def request_budget_per_day(
    routes: int,
    stops: int,
    interval_s: float,
    detour_interval_s: float = 300.0,
) -> int:
    cycles = 86400 / interval_s
    stop_calls = math.ceil(stops / 10) * cycles
    vehicle_calls = math.ceil(routes / 10) * cycles
    detour_calls = (86400 / detour_interval_s) * 2
    return int(stop_calls + vehicle_calls + cycles + detour_calls)
