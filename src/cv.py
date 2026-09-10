"""Repeated stratified cross-validation producing out-of-fold and test predictions."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import N_REPEATS, N_SPLITS, OUTLET_ID, SEED, TARGET
from src.features import catboost_matrix, item_popularity, linear_matrix, tree_matrix
from src.models import ModelSpec
from src.targets import TRANSFORMS

Matrices = dict[str, tuple[pd.DataFrame, pd.DataFrame]]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def build_matrices(train: pd.DataFrame, test: pd.DataFrame) -> Matrices:
    return {
        "linear": linear_matrix(train, test),
        "glm": linear_matrix(train, test, log_link=True),
        "tree": tree_matrix(train, test),
        "codes": tree_matrix(train, test, as_codes=True),
        "catboost": catboost_matrix(train, test),
    }


@dataclass
class CVResult:
    name: str
    oof: np.ndarray
    test: np.ndarray
    fold_rmse: list[float]
    oof_rmse: float
    seconds: float

    def summary(self) -> dict:
        return {
            "model": self.name,
            "oof_rmse": round(self.oof_rmse, 2),
            "fold_rmse_mean": round(float(np.mean(self.fold_rmse)), 2),
            "fold_rmse_std": round(float(np.std(self.fold_rmse)), 2),
            "seconds": round(self.seconds, 1),
        }


def _fit(model, X, z, weight) -> None:
    if weight is None:
        model.fit(X, z)
    elif isinstance(model, Pipeline):
        model.fit(X, z, **{f"{model.steps[-1][0]}__sample_weight": weight})
    else:
        model.fit(X, z, sample_weight=weight)


def run_cv(
    spec: ModelSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    matrices: Matrices,
    params: dict | None = None,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    seed: int = SEED,
) -> CVResult:
    """Out-of-fold predictions are averaged over repeats; test predictions are
    averaged over every fold model. Folds are stratified by store so each one
    sees all ten stores in the same proportions as the full data."""
    start = time.perf_counter()
    X, X_test = matrices[spec.matrix]
    y = train[TARGET].to_numpy(dtype=float)
    transform = TRANSFORMS[spec.target]

    oof_sum, oof_count = np.zeros(len(train)), np.zeros(len(train))
    test_sum = np.zeros(len(test))
    fold_scores: list[float] = []

    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for fold, (tr, va) in enumerate(splitter.split(X, train[OUTLET_ID])):
        rows_tr, rows_va = train.iloc[tr], train.iloc[va]
        X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X_test

        if spec.item_popularity:
            pop_tr, (pop_va, pop_te) = item_popularity(
                rows_tr, y[tr], [rows_va, test], seed=seed + fold
            )
            X_tr = X_tr.assign(Item_Popularity=pop_tr)
            X_va = X_va.assign(Item_Popularity=pop_va)
            X_te = X_te.assign(Item_Popularity=pop_te)

        model = spec.make(params, seed + fold)
        _fit(model, X_tr, transform.forward(y[tr], rows_tr), transform.weight(rows_tr))

        # Sales cannot be negative; clipping only ever removes error.
        pred_va = np.clip(transform.inverse(model.predict(X_va), rows_va), 0, None)
        pred_te = np.clip(transform.inverse(model.predict(X_te), test), 0, None)

        oof_sum[va] += pred_va
        oof_count[va] += 1
        test_sum += pred_te
        fold_scores.append(rmse(y[va], pred_va))

    oof = oof_sum / oof_count
    return CVResult(
        name=spec.name,
        oof=oof,
        test=test_sum / (n_splits * n_repeats),
        fold_rmse=fold_scores,
        oof_rmse=rmse(y, oof),
        seconds=time.perf_counter() - start,
    )
