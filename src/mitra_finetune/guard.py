"""Prediction-shape helpers for bagged children.

Test predictions are the plain mean over all bag children -- the exact
behavior of the reference pipeline's ``predictor.predict_proba``
(AutoGluon's bagged-ensemble average). No child is ever dropped.
"""
from __future__ import annotations

import numpy as np


def as_proba_matrix(proba: np.ndarray) -> np.ndarray:
    """Normalize a child's predictions to a 2-D probability matrix.

    AutoGluon returns binary children as 1-D positive-class vectors;
    rebuild the two-class matrix [1-p, p] so shapes are uniform.
    """
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return np.stack([1.0 - proba, proba], axis=1)
    return proba
