"""TrendLab Prediction Engine V1 — research ranking stack (Phases A–E).

See docs/PREDICTION_ENGINE_V1.md. Separate from legacy ternary ml_train.py.
"""

from __future__ import annotations

DISCLAIMER = "EXPLORATORY — SURVIVORSHIP BIAS NOT FULLY CONTROLLED"
DATASET_VERSION = "pe_dataset_v1"
HORIZON_DAYS = 5
PURGE_GAP_DAYS = 5
VOL_WINDOW = 20

__all__ = [
    "DISCLAIMER",
    "DATASET_VERSION",
    "HORIZON_DAYS",
    "PURGE_GAP_DAYS",
    "VOL_WINDOW",
]
