"""CTA Bus Tracker v3 ingest + telemetry-based estimator.

Runs alongside the legacy v2 bus pipeline (``bus_client.py`` /
``bus_predictor.py``). Both pipelines write to their own tables; the
multi-predictor registry decides which estimator serves each corridor.
"""
