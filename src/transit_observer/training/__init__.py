"""Offline training package for the learned predictors.

This package is NEVER imported by the collector / resolver / API hot
paths. It writes only to ``settings.data_dir / "models"`` and the
``model_artifacts`` registry table. Lives behind the ``learned``
optional dependency group.
"""
