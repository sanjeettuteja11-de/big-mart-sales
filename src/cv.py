"""Cross-validation producing out-of-fold and test predictions.

Three ways to split, chosen with `scheme`:

- "stratified": repeated K-fold stratified by store. Every test product also
  appears in train, so this mirrors the competition: known products, new
  product-store rows. It is the primary scheme.
- "holdout": one fixed 80/20 split for quick experiments.
- "grouped": K-fold grouped by product, so validation products are unseen.
  The competition never asks this, but it shows how much a model leans on
  product identity.

Preprocessing is fit either once on train and test features together
(`prepared`, the default) or inside every fold on the training rows only
(`fold_fit=True`). Neither uses sales; the two are compared in src/validate.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit, train_test_split,
)
from sklearn.pipeline import Pipeline

from src.config import ITEM_ID, N_REPEATS, N_SPLITS, OUTLET_ID, SEED, TARGET
from src.data import Cleaner
from src.features import (
    CATEGORICAL, FeatureBuilder, catboost_matrix, item_popularity, linear_matrix, tree_matrix,
)
from src.models import ModelSpec
from src.targets import TRANSFORMS

EARLY_STOPPING_ROUNDS = 50
EARLY_STOPPING_FRACTION = 0.15


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


MATRIX_BUILDERS = {
    "linear": lambda tr, others, drop: linear_matrix(tr, others, drop=drop),
    "glm": lambda tr, others, drop: linear_matrix(tr, others, log_link=True, drop=drop),
    "tree": lambda tr, others, drop: tree_matrix(tr, others, drop=drop),
    "codes": lambda tr, others, drop: tree_matrix(tr, others, as_codes=True, drop=drop),
    "catboost": lambda tr, others, drop: catboost_matrix(tr, others, drop=drop),
}


@dataclass
class Prepared:
    """Cleaned, feature-engineered frames and the matrices built from them."""

    train: pd.DataFrame
    test: pd.DataFrame
    matrices: dict[str, tuple[pd.DataFrame, pd.DataFrame]]
    drop: set[str] | None = None


def prepare(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    use_test_features: bool = True,
    drop: set[str] | None = None,
    matrices: list[str] | None = None,
) -> Prepared:
    frames = [train_raw, test_raw] if use_test_features else [train_raw]
    cleaner = Cleaner().fit(frames)
    train, test = cleaner.transform(train_raw), cleaner.transform(test_raw)
    builder = FeatureBuilder().fit([train, test] if use_test_features else [train])
    train, test = builder.transform(train), builder.transform(test)
    built = {}
    for name in matrices or MATRIX_BUILDERS:
        X_tr, (X_te,) = MATRIX_BUILDERS[name](train, [test], drop)
        built[name] = (X_tr, X_te)
    return Prepared(train, test, built, drop)


def folds(scheme: str, train: pd.DataFrame, n_splits: int, n_repeats: int, seed: int):
    """Yield (train_idx, val_idx) pairs for the chosen scheme."""
    if scheme == "stratified":
        splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        yield from splitter.split(train, train[OUTLET_ID])
    elif scheme == "holdout":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        yield from splitter.split(train, train[OUTLET_ID])
    elif scheme == "grouped":
        for repeat in range(n_repeats):
            splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
            yield from splitter.split(train, groups=train[ITEM_ID])
    else:
        raise ValueError(f"unknown scheme {scheme!r}")


@dataclass
class CVResult:
    name: str
    oof: np.ndarray
    test: np.ndarray
    fold_rmse: list[float]
    oof_rmse: float
    seconds: float
    scheme: str = "stratified"
    best_iterations: list[int] = field(default_factory=list)
    negative_share: float = 0.0

    def summary(self) -> dict:
        out = {
            "model": self.name,
            "scheme": self.scheme,
            "oof_rmse": round(self.oof_rmse, 2),
            "fold_rmse_mean": round(float(np.mean(self.fold_rmse)), 2),
            "fold_rmse_std": round(float(np.std(self.fold_rmse)), 2),
            "n_folds": len(self.fold_rmse),
            "seconds": round(self.seconds, 1),
        }
        if self.best_iterations:
            out["best_iterations_mean"] = int(round(float(np.mean(self.best_iterations))))
        if self.negative_share:
            out["negative_prediction_share"] = round(self.negative_share, 4)
        return out


def _catboost_cat_features(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c in CATEGORICAL or c == ITEM_ID]


def _fit_kwargs(spec: ModelSpec, model, X, weight) -> dict:
    kwargs = {}
    if spec.library == "catboost":
        kwargs["cat_features"] = _catboost_cat_features(X)
    if weight is not None:
        if isinstance(model, Pipeline):
            kwargs[f"{model.steps[-1][0]}__sample_weight"] = weight
        else:
            kwargs["sample_weight"] = weight
    return kwargs


def _early_stopping_iterations(spec: ModelSpec, params: dict, seed: int, X, z, w, stores) -> int:
    """Best iteration count from a split of the training fold, on the weighted
    objective the model is trained with."""
    idx_a, idx_b = train_test_split(
        np.arange(len(z)), test_size=EARLY_STOPPING_FRACTION, random_state=seed, stratify=stores
    )
    X_a, X_b, z_a, z_b = X.iloc[idx_a], X.iloc[idx_b], z[idx_a], z[idx_b]
    w_a = None if w is None else w[idx_a]
    w_b = None if w is None else w[idx_b]

    if spec.library == "lightgbm":
        import lightgbm as lgb

        model = spec.make(params, seed)
        model.fit(X_a, z_a, sample_weight=w_a, eval_X=X_b, eval_y=z_b,
                  eval_sample_weight=None if w_b is None else [w_b],
                  callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        return int(model.best_iteration_)
    if spec.library == "xgboost":
        model = spec.make({**params, "early_stopping_rounds": EARLY_STOPPING_ROUNDS}, seed)
        model.fit(X_a, z_a, sample_weight=w_a, eval_set=[(X_b, z_b)],
                  sample_weight_eval_set=None if w_b is None else [w_b], verbose=False)
        return int(model.best_iteration) + 1
    if spec.library == "catboost":
        from catboost import Pool

        model = spec.make(params, seed)
        cat = _catboost_cat_features(X)
        model.fit(X_a, z_a, sample_weight=w_a, cat_features=cat,
                  eval_set=Pool(X_b, z_b, weight=w_b, cat_features=cat),
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        return int(model.get_best_iteration()) + 1
    raise ValueError(f"early stopping is not implemented for {spec.library}")


def run_cv(
    spec: ModelSpec,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    prepared: Prepared | None = None,
    params: dict | None = None,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    seed: int = SEED,
    scheme: str = "stratified",
    fold_fit: bool = False,
    use_test_features: bool = True,
    drop: set[str] | None = None,
    early_stopping: bool | None = None,
) -> CVResult:
    """Out-of-fold predictions are averaged over repeats; test predictions are
    averaged over every fold model.

    `prepared` is reused across models when preprocessing is fit once. With
    `fold_fit`, cleaning and feature statistics are refit on each training
    fold and applied to the validation rows and test.
    """
    start = time.perf_counter()
    params = {**(params or {})}
    if early_stopping is None:
        early_stopping = spec.iterations_param is not None
    if not fold_fit and prepared is None:
        prepared = prepare(train_raw, test_raw, use_test_features, drop, [spec.matrix])
    train_frame = train_raw if fold_fit else prepared.train
    y = train_raw[TARGET].to_numpy(dtype=float)
    stores = train_raw[OUTLET_ID].to_numpy()
    transform = TRANSFORMS[spec.target]

    oof_sum, oof_count = np.zeros(len(train_raw)), np.zeros(len(train_raw))
    test_sum, n_models = np.zeros(len(test_raw)), 0
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    negatives = 0

    for fold, (tr, va) in enumerate(folds(scheme, train_frame, n_splits, n_repeats, seed)):
        if fold_fit:
            cleaner = Cleaner().fit([train_raw.iloc[tr]])
            rows_tr, rows_va, rows_te = (cleaner.transform(f) for f in (train_raw.iloc[tr], train_raw.iloc[va], test_raw))
            builder = FeatureBuilder().fit([rows_tr])
            rows_tr, rows_va, rows_te = (builder.transform(f) for f in (rows_tr, rows_va, rows_te))
            X_tr, (X_va, X_te) = MATRIX_BUILDERS[spec.matrix](rows_tr, [rows_va, rows_te], drop)
        else:
            rows_tr, rows_va, rows_te = prepared.train.iloc[tr], prepared.train.iloc[va], prepared.test
            X, X_te = prepared.matrices[spec.matrix]
            X_tr, X_va = X.iloc[tr], X.iloc[va]

        if spec.item_popularity:
            pop_tr, (pop_va, pop_te) = item_popularity(rows_tr, y[tr], [rows_va, rows_te], seed=seed + fold)
            X_tr, X_va, X_te = (f.assign(Item_Popularity=p) for f, p in ((X_tr, pop_tr), (X_va, pop_va), (X_te, pop_te)))

        z_tr, w_tr = transform.forward(y[tr], rows_tr), transform.weight(rows_tr)
        fold_params = dict(params)
        if early_stopping:
            best = _early_stopping_iterations(spec, params, seed + fold, X_tr, z_tr, w_tr, stores[tr])
            best_iterations.append(best)
            fold_params[spec.iterations_param] = best

        model = spec.make(fold_params, seed + fold)
        model.fit(X_tr, z_tr, **_fit_kwargs(spec, model, X_tr, w_tr))

        raw_va = transform.inverse(model.predict(X_va), rows_va)
        raw_te = transform.inverse(model.predict(X_te), rows_te)
        negatives += int((raw_va < 0).sum())
        # Sales cannot be negative; clipping only ever removes error (checked in src/validate.py).
        pred_va, pred_te = np.clip(raw_va, 0, None), np.clip(raw_te, 0, None)

        oof_sum[va] += pred_va
        oof_count[va] += 1
        test_sum += pred_te
        n_models += 1
        fold_scores.append(rmse(y[va], pred_va))

    covered = oof_count > 0
    oof = np.full(len(y), np.nan)
    oof[covered] = oof_sum[covered] / oof_count[covered]
    return CVResult(
        name=spec.name,
        oof=oof,
        test=test_sum / n_models,
        fold_rmse=fold_scores,
        oof_rmse=rmse(y[covered], oof[covered]),
        seconds=time.perf_counter() - start,
        scheme=scheme,
        best_iterations=best_iterations,
        negative_share=negatives / int(oof_count.sum()),
    )
