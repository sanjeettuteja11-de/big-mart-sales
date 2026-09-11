"""Blend model predictions with non-negative weights that sum to one."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from src.config import SEED
from src.cv import rmse


def fit_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weights for the columns of P (rows x models) that minimise RMSE.

    Restricting weights to the simplex keeps the blend an average of models,
    which cannot overshoot the way unconstrained stacking can on noisy data.
    """
    scale = y.std()
    P, y = P / scale, y / scale  # keeps SLSQP's tolerances meaningful
    k = P.shape[1]
    result = minimize(
        fun=lambda w: np.mean((P @ w - y) ** 2),
        jac=lambda w: 2 * P.T @ (P @ w - y) / len(y),
        x0=np.full(k, 1 / k),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    w = np.clip(result.x, 0, None)
    return w / w.sum()


def blend_cv_rmse(P: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = SEED) -> float:
    """Second-stage OOF-row crossfit diagnostic, NOT a fully nested score.

    The base predictions were generated globally, and their training labels
    can cross this meta split. Model/hyperparameter selection also used the
    full data. For an isolated blending evaluation, regenerate base OOF
    predictions inside each outer training fold (src.nested_study).
    """
    pred = np.zeros(len(y))
    for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(P):
        pred[va] = P[va] @ fit_weights(P[tr], y[tr])
    return rmse(y, pred)
