"""Enums, constants, and dataclasses for the v2 train pipeline.

Mirrors the bus_v3 ``models.py`` layout — same display states + data
quality + reason code surface so the dashboard can render both
predictors with one set of components.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class ArrivalLabel(str, Enum):
    ARRIVED_CONFIRMED = "ARRIVED_CONFIRMED"           # nextStaId transition + corroboration
    ARRIVED_INFERRED = "ARRIVED_INFERRED"             # is_approaching disappearance only
    PASSED_WITHOUT_PICKUP = "PASSED_WITHOUT_PICKUP"   # express, run-as-directed, etc.
    FAULTED_TRACKING = "FAULTED_TRACKING"             # ttarrivals isFlt=1
    STALE_DATA = "STALE_DATA"                         # predictions older than freshness threshold
    NO_EVIDENCE_GHOST = "NO_EVIDENCE_GHOST"           # prediction present but no run trajectory match
    CENSORED_UNKNOWN = "CENSORED_UNKNOWN"             # too little info to label
    REROUTED_OR_SHORT_TURN = "REROUTED_OR_SHORT_TURN" # destination changed mid-run


class DataQuality(str, Enum):
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    CONTRADICTORY = "CONTRADICTORY"
    INSUFFICIENT = "INSUFFICIENT"
    API_ERROR = "API_ERROR"


class DisplayState(str, Enum):
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    MEDIUM_CONFIDENCE = "MEDIUM_CONFIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    UNRELIABLE = "UNRELIABLE"
    DO_NOT_DISPLAY_AS_ARRIVING = "DO_NOT_DISPLAY_AS_ARRIVING"


class ReasonCode(str, Enum):
    # Train Tracker arrival signals.
    PREDICTION_FRESH = "PREDICTION_FRESH"
    PREDICTION_STALE = "PREDICTION_STALE"
    IS_APPROACHING_TRUE = "IS_APPROACHING_TRUE"
    IS_DELAYED_FLAG = "IS_DELAYED_FLAG"
    IS_FAULT_FLAG = "IS_FAULT_FLAG"
    IS_SCHEDULED_FALLBACK = "IS_SCHEDULED_FALLBACK"
    # Trajectory and topology signals.
    NEXT_STATION_MATCH = "NEXT_STATION_MATCH"
    NEXT_STATION_MISMATCH = "NEXT_STATION_MISMATCH"
    NEXT_STATION_ADVANCED = "NEXT_STATION_ADVANCED"   # train passed the queried stop
    POSITION_FOUND = "POSITION_FOUND"
    POSITION_STALE = "POSITION_STALE"
    POSITION_NOT_FOUND = "POSITION_NOT_FOUND"
    GPS_ON_EXPECTED_LINE = "GPS_ON_EXPECTED_LINE"
    GPS_OFF_EXPECTED_LINE = "GPS_OFF_EXPECTED_LINE"
    # Follow (per-run trajectory) signals.
    FOLLOW_TRAJECTORY_CONSISTENT = "FOLLOW_TRAJECTORY_CONSISTENT"
    FOLLOW_TRAJECTORY_DIVERGENT = "FOLLOW_TRAJECTORY_DIVERGENT"
    FOLLOW_MISSING = "FOLLOW_MISSING"
    # GTFS-RT cross-validation.
    GTFSRT_AGREE = "GTFSRT_AGREE"
    GTFSRT_DISAGREE = "GTFSRT_DISAGREE"
    GTFSRT_MISSING = "GTFSRT_MISSING"
    GTFSRT_DELAY_REPORTED = "GTFSRT_DELAY_REPORTED"
    GTFSRT_STOPPED_AT = "GTFSRT_STOPPED_AT"
    GTFSRT_INCOMING_AT = "GTFSRT_INCOMING_AT"
    # Cross-stream tri-validation.
    THREE_WAY_AGREE = "THREE_WAY_AGREE"
    TWO_WAY_AGREE = "TWO_WAY_AGREE"
    # Slow zone evidence.
    SLOW_ZONE_AHEAD = "SLOW_ZONE_AHEAD"
    # Prediction volatility / monotonicity.
    PREDICTION_STABLE = "PREDICTION_STABLE"
    PREDICTION_VOLATILE = "PREDICTION_VOLATILE"
    PREDICTION_DECREASING = "PREDICTION_DECREASING"
    PREDICTION_INCREASING = "PREDICTION_INCREASING"
    # Ground truth flags.
    GROUND_TRUTH_HIGH_CONFIDENCE = "GROUND_TRUTH_HIGH_CONFIDENCE"
    GROUND_TRUTH_LOW_CONFIDENCE = "GROUND_TRUTH_LOW_CONFIDENCE"
    # Aborts.
    ARRIVAL_ESTIMATE_ABSTAINED = "ARRIVAL_ESTIMATE_ABSTAINED"
    API_ERROR = "API_ERROR"


@dataclass
class EstimateResult:
    generated_at_ms: int
    map_id: int
    line: Optional[str]
    direction_code: Optional[str]
    run_number: Optional[str]
    destination: Optional[str]
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
    endpoint: str                              # 'ttarrivals' | 'ttfollow' | 'ttpositions' | 'gtfsrt'
    source: str                                # 'train_tracker' | 'gtfsrt_train'
    params_redacted: dict[str, Any]
    query_kind: Optional[str]
    request_url_redacted: str
    local_request_start_ms: int
    local_response_end_ms: int
    cta_server_time_ms: Optional[int]
    http_status: Optional[int]
    latency_ms: float
    ok: bool
    json_data: Optional[dict[str, Any]]        # JSON for tt*.aspx; None for protobuf
    raw_bytes: Optional[bytes]                 # protobuf for GTFS-RT (we don't serialize back to JSON)
    error_message: Optional[str]
