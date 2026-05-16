"""Predictor package: pluggable forecasters that share a Protocol.

The journey kernel (Swift port) is one predictor. The learned residual-GBM
is another. The registry decides which is active per corridor. All write
to ``forecast_queue`` with a distinct ``predictor_version`` so
``metrics.py`` can rank them on real outcomes.
"""

from .protocol import (
    Prediction,
    PredictionLeg,
    Predictor,
    quantiles_to_summary,
)

__all__ = [
    "Prediction",
    "PredictionLeg",
    "Predictor",
    "quantiles_to_summary",
]
