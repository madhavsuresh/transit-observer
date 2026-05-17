"""CTA L (train) richer ingest + telemetry-based estimator.

Parallel to the legacy ``cta_train_client`` / ``train_arrivals_raw``
pipeline (which keeps running untouched). The v2 pipeline pulls from
three independent CTA sources and stores them as immutable raw + tagged
observations, so the predictor can cross-validate:

1. **Train Tracker ``ttarrivals.aspx``** — by-station predictions.
2. **Train Tracker ``ttfollow.aspx``** — per-run trajectory (NEW —
   not previously captured). For each run the API returns predicted
   arrivals at *all* upcoming stops, not just the queried one. This is
   the train analog of bus_v3's by-vid prediction queries.
3. **Train Tracker ``ttpositions.aspx``** — per-line vehicle positions
   with ``nextStaId`` (track topology).
4. **CTA GTFS-RT TripUpdates** — independent prediction stream with
   schedule-adherence ``delay`` field that the proprietary feeds don't
   expose for L.
5. **CTA GTFS-RT VehiclePositions** — vehicle lat/lon + bearing + speed
   + ``current_status`` (``incoming_at`` / ``stopped_at`` /
   ``in_transit_to``) + ``current_stop_sequence``.

High-confidence ground truth comes from same-run ``nextStaId`` transitions
(train left station A on its way to B → A arrival inferred) plus
GTFS-RT ``current_status='stopped_at'`` corroboration. The estimator
combines all three streams with reliability scoring and reason codes.
"""
