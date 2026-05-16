"""HTTP API for on-demand transit predictions.

Runs as a separate process from the collector. Reads the DuckDB read
replica (so it never blocks the writer), runs the per-mode predictor
against the live raw feed for an arbitrary OD pair, and appends a row
to ``data/queries.ndjson`` so the collector can ingest it for the
auto-upgrade path.

Endpoints
---------
``GET /predict``
    Run a prediction for one OD pair. Returns the wait + in-vehicle
    distribution. Logs the query.

``GET /corridors``
    List seeded corridors (mirrors ``transit corpus list``).

``GET /healthz``
    Liveness probe.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from . import db
from .config import CHICAGO
from .corpus import PREDICTOR_VERSION, AdHocPrediction, predict_for_od
from .corridors import SEED_CORRIDORS
from .query_log import append_query


log = logging.getLogger(__name__)


app = FastAPI(
    title="transit-observer",
    description="On-demand predictions for arbitrary (mode, line, boarding, alighting) tuples.",
    version="0.1.0",
)


class PredictionResponse(BaseModel):
    query_id: str
    mode: str
    line: str
    direction_code: str | None
    boarding_label: str
    alighting_label: str
    predicted_wait_seconds: dict
    predicted_in_vehicle_seconds: dict
    predicted_total_seconds: dict
    predictor_version: str
    queried_at: str


class CorridorSummary(BaseModel):
    corridor_id: str
    mode: str
    line: str
    direction: str
    origin_label: str
    destination_label: str
    priority: int


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "predictor_version": PREDICTOR_VERSION}


@app.get("/corridors", response_model=list[CorridorSummary])
async def list_corridors() -> list[CorridorSummary]:
    return [
        CorridorSummary(
            corridor_id=c.corridor_id,
            mode=c.mode,
            line=c.line,
            direction=c.direction,
            origin_label=c.origin_label,
            destination_label=c.destination_label,
            priority=c.priority,
        )
        for c in SEED_CORRIDORS
    ]


@app.get("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    mode: str = Query(..., description="'L' | 'bus' | 'metra' | 'intercampus'"),
    line: str = Query(..., description="Line code: e.g. 'Red', 'Blue', '22', 'UP-N', 'intercampus'"),
    boarding: str = Query(..., description="Boarding station ID (int for L/bus, text for metra/intercampus)"),
    alighting: str = Query(..., description="Alighting station ID"),
    client_id: str | None = Query(None, description="Optional caller identifier for log correlation"),
) -> PredictionResponse:
    """Run a prediction for an arbitrary OD pair using the live feed.

    The query is logged so frequently-queried pairs get promoted into
    seeded corridors by the collector's auto-upgrade pass.
    """
    boarding_int_id, boarding_text_id = _split_station_id(mode, boarding)
    alighting_int_id, alighting_text_id = _split_station_id(mode, alighting)
    queried_at = datetime.now(CHICAGO)

    prediction: AdHocPrediction | None = None
    error_reason: str | None = None
    try:
        with db.reader() as conn:
            prediction = predict_for_od(
                conn,
                mode=mode, line=line,
                boarding_int_id=boarding_int_id,
                boarding_text_id=boarding_text_id,
                alighting_int_id=alighting_int_id,
                alighting_text_id=alighting_text_id,
                now=queried_at,
            )
    except Exception as exc:  # noqa: BLE001
        error_reason = f"{type(exc).__name__}: {exc}"
        log.warning("predict.exception", err=error_reason)

    query_id = append_query(
        queried_at=queried_at, client_id=client_id,
        mode=mode, line=line,
        boarding_int_id=boarding_int_id, boarding_text_id=boarding_text_id,
        alighting_int_id=alighting_int_id, alighting_text_id=alighting_text_id,
        prediction=prediction, error_reason=error_reason,
    )

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail=error_reason or "No data available for this OD pair right now.",
        )

    return PredictionResponse(
        query_id=query_id,
        mode=prediction.mode,
        line=prediction.line,
        direction_code=prediction.direction_code,
        boarding_label=prediction.boarding_label,
        alighting_label=prediction.alighting_label,
        predicted_wait_seconds={
            "mean": prediction.predicted_wait_mean,
            "p50": prediction.predicted_wait_p50,
            "p80": prediction.predicted_wait_p80,
            "p90": prediction.predicted_wait_p90,
        },
        predicted_in_vehicle_seconds={
            "mean": prediction.predicted_in_vehicle_mean,
        },
        predicted_total_seconds={
            "p50": prediction.predicted_total_p50,
            "p80": prediction.predicted_total_p80,
            "p90": prediction.predicted_total_p90,
        },
        predictor_version=prediction.predictor_version,
        queried_at=queried_at.isoformat(),
    )


def _split_station_id(mode: str, value: str) -> tuple[int, str | None]:
    """L and bus use integer IDs; Metra and Intercampus use text IDs."""
    if mode in ("L", "bus"):
        try:
            return int(value), None
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail=f"mode '{mode}' expects integer station IDs, got {value!r}",
            )
    return 0, value


def main() -> None:
    """Entry point: launches uvicorn. Used by ``transit api`` CLI."""
    import uvicorn
    uvicorn.run(
        "transit_observer.api:app",
        host="127.0.0.1", port=8000, reload=False,
    )


if __name__ == "__main__":
    main()
