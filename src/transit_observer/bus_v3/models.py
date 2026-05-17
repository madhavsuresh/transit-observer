"""Enums, constants, and dataclasses ported from the validator package.

These are pure value types with no DB coupling. The estimator and
inference modules use these to express reliability scores, display
states, and disruption labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class ArrivalLabel(str, Enum):
    ARRIVED_CONFIRMED = "ARRIVED_CONFIRMED"
    ARRIVED_INFERRED = "ARRIVED_INFERRED"
    PASSED_WITHOUT_PICKUP_OR_EXPRESSED = "PASSED_WITHOUT_PICKUP_OR_EXPRESSED"
    CANCELED_OR_INVALIDATED = "CANCELED_OR_INVALIDATED"
    VEHICLE_DISAPPEARED = "VEHICLE_DISAPPEARED"
    STALE_DATA = "STALE_DATA"
    DETOUR_AMBIGUOUS = "DETOUR_AMBIGUOUS"
    NO_EVIDENCE_GHOST_CANDIDATE = "NO_EVIDENCE_GHOST_CANDIDATE"
    CENSORED_UNKNOWN = "CENSORED_UNKNOWN"


class DataQuality(str, Enum):
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    CONTRADICTORY = "CONTRADICTORY"
    INSUFFICIENT = "INSUFFICIENT"
    DETOUR_AMBIGUOUS = "DETOUR_AMBIGUOUS"
    API_ERROR = "API_ERROR"


class DisplayState(str, Enum):
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    MEDIUM_CONFIDENCE = "MEDIUM_CONFIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    UNRELIABLE = "UNRELIABLE"
    DO_NOT_DISPLAY_AS_ARRIVING = "DO_NOT_DISPLAY_AS_ARRIVING"


# CTA BusTime v3 dyn codes from the official Developer Guide.
DYN_ACTIONS: dict[int, str] = {
    0: "NONE",
    1: "CANCELED",
    2: "REASSIGNED",
    3: "SHIFTED",
    4: "EXPRESSED",
    6: "STOPS_AFFECTED",
    8: "NEW_TRIP",
    9: "PARTIAL_TRIP",
    10: "PARTIAL_TRIP_NEW",
    12: "DELAYED_CANCEL",
    13: "ADDED_STOP",
    14: "UNKNOWN_DELAY",
    15: "UNKNOWN_DELAY_NEW",
    16: "INVALIDATED_TRIP",
    17: "INVALIDATED_TRIP_NEW",
    18: "CANCELLED_TRIP_NEW",
    19: "STOPS_AFFECTED_NEW",
}

DYN_SEVERE_ABSTAIN = {1, 4, 12, 16, 17, 18}
DYN_WARN = {2, 3, 6, 8, 9, 10, 13, 14, 15, 19}

FLAGSTOP_UNDEFINED = -1
FLAGSTOP_NORMAL = 0
FLAGSTOP_PICKUP_AND_DISCHARGE = 1
FLAGSTOP_ONLY_DISCHARGE = 2


class ReasonCode(str, Enum):
    VEHICLE_POSITION_FRESH = "VEHICLE_POSITION_FRESH"
    VEHICLE_POSITION_STALE = "VEHICLE_POSITION_STALE"
    VEHICLE_POSITION_FROZEN = "VEHICLE_POSITION_FROZEN"
    VEHICLE_NOT_FOUND = "VEHICLE_NOT_FOUND"
    VEHICLE_FOUND = "VEHICLE_FOUND"
    VEHICLE_DISAPPEARED = "VEHICLE_DISAPPEARED"
    PREDICTION_FRESH = "PREDICTION_FRESH"
    PREDICTION_STALE = "PREDICTION_STALE"
    DSTP_DECREASING = "DSTP_DECREASING"
    DSTP_STALLED = "DSTP_STALLED"
    DSTP_INCREASING = "DSTP_INCREASING"
    PDIST_INCREASING = "PDIST_INCREASING"
    PDIST_STALLED = "PDIST_STALLED"
    PDIST_CROSSED_STOP = "PDIST_CROSSED_STOP"
    PATTERN_MATCH = "PATTERN_MATCH"
    PATTERN_MISMATCH = "PATTERN_MISMATCH"
    GPS_ON_EXPECTED_PATTERN = "GPS_ON_EXPECTED_PATTERN"
    GPS_OFF_EXPECTED_PATTERN = "GPS_OFF_EXPECTED_PATTERN"
    GPS_NEAR_STOP = "GPS_NEAR_STOP"
    GPS_NEAR_STOP_WITHOUT_PDIST_CROSSING = "GPS_NEAR_STOP_WITHOUT_PDIST_CROSSING"
    GPS_PDIST_INCONSISTENT = "GPS_PDIST_INCONSISTENT"
    ROUTE_DIRECTION_MATCH = "ROUTE_DIRECTION_MATCH"
    ROUTE_DIRECTION_MISMATCH = "ROUTE_DIRECTION_MISMATCH"
    CTA_GEOMETRY_AGREE = "CTA_GEOMETRY_AGREE"
    CTA_GEOMETRY_DISAGREE = "CTA_GEOMETRY_DISAGREE"
    PREDICTION_VOLATILE = "PREDICTION_VOLATILE"
    PREDICTION_STABLE = "PREDICTION_STABLE"
    DYN_CANCELED = "DYN_CANCELED"
    DYN_EXPRESSED = "DYN_EXPRESSED"
    DYN_REASSIGNED = "DYN_REASSIGNED"
    DYN_INVALIDATED = "DYN_INVALIDATED"
    DYN_PARTIAL_TRIP = "DYN_PARTIAL_TRIP"
    DYN_DELAYED_OR_SHIFTED = "DYN_DELAYED_OR_SHIFTED"
    DLY_TRUE = "DLY_TRUE"
    DETOUR_ACTIVE = "DETOUR_ACTIVE"
    STOP_REMOVED_BY_DETOUR = "STOP_REMOVED_BY_DETOUR"
    STOP_ADDED_BY_DETOUR = "STOP_ADDED_BY_DETOUR"
    FLAGSTOP_ONLY_DISCHARGE = "FLAGSTOP_ONLY_DISCHARGE"
    API_ERROR = "API_ERROR"
    ARRIVAL_ESTIMATE_ABSTAINED = "ARRIVAL_ESTIMATE_ABSTAINED"
    GROUND_TRUTH_HIGH_CONFIDENCE = "GROUND_TRUTH_HIGH_CONFIDENCE"
    GROUND_TRUTH_LOW_CONFIDENCE = "GROUND_TRUTH_LOW_CONFIDENCE"
    DUE_BUT_VEHICLE_NOT_NEAR_STOP = "DUE_BUT_VEHICLE_NOT_NEAR_STOP"


@dataclass
class EstimateResult:
    generated_at_ms: int
    stpid: str
    rt: Optional[str]
    rtdir: Optional[str]
    vid: Optional[str]
    destination: Optional[str]
    tatripid: Optional[str]
    tablockid: Optional[str]
    predicted_arrival_ms: Optional[int]
    interval80_low_ms: Optional[int]
    interval80_high_ms: Optional[int]
    interval90_low_ms: Optional[int]
    interval90_high_ms: Optional[int]
    interval95_low_ms: Optional[int]
    interval95_high_ms: Optional[int]
    reliability: float
    display_state: DisplayState
    data_quality: DataQuality
    rider_message: str
    reason_codes: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["display_state"] = str(self.display_state.value)
        d["data_quality"] = str(self.data_quality.value)
        return d


@dataclass
class ApiCallResult:
    endpoint: str
    params_redacted: dict[str, Any]
    query_kind: Optional[str]
    request_url_redacted: str
    local_request_start_ms: int
    local_response_end_ms: int
    cta_server_time_ms: Optional[int]
    http_status: Optional[int]
    latency_ms: float
    ok: bool
    json_data: Optional[dict[str, Any]]
    error_message: Optional[str]
