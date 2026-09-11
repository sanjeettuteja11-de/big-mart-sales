"""Cross-validation producing out-of-fold and test predictions.

Three ways to split, chosen with `scheme`:

- "stratified": repeated K-fold stratified by store. Every test product also
  appears in train, so this mirrors the competition: known products, new
  product-store rows. It is the primary scheme.
- "holdout": one fixed 80/20 split for quick experiments.
- "grouped": K-fold grouped by product, so validation products are unseen.
  The competition never asks this, but it shows how much a model leans on
  product identity.

Preprocessing is fitted inside every training fold, including the inner
early-stopping split. Transductive fitting requires explicit opt-in and is
retained only for historical comparisons.
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
from src.data import Cleaner, FEATURE_COLUMNS
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
    "raw": lambda tr, others, drop: (tr[FEATURE_COLUMNS], [f[FEATURE_COLUMNS] for f in others]),
    "structural": lambda tr, others, drop: (
        tr[[c for c in [ITEM_ID, OUTLET_ID, "Outlet_Type", "Item_MRP", "MRP_Band"] if not drop or c not in drop]],
        [f[[c for c in [ITEM_ID, OUTLET_ID, "Outlet_Type", "Item_MRP", "MRP_Band"] if not drop or c not in drop]] for f in others]
    ),
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
    use_test_features: bool = False,
    drop: set[str] | None = None,
    matrices: list[str] | None = None,
) -> Prepared:
    frames = [train_raw, test_raw] if use_test_features else [train_raw]
    cleaner = Cleaner().fit(frames)
    train, test = cleaner.transform(train_raw), cleaner.transform(test_raw)
    builder = FeatureBuilder().fit([train, test] if use_test_features else [train])
    train, test = builder.transform(train), builder.transform(test)
    built = {}
    for name in matrices or [n for n in MATRIX_BUILDERS if n != "raw"]:
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
    raw_oof_rmse: float = 0.0
    split_indices: list = field(default_factory=list)
    fold_predictions: list = field(default_factory=list)
    raw_oof: np.ndarray | None = None

    def summary(self) -> dict:
        out = {
            "model": self.name,
            "scheme": self.scheme,
            "oof_rmse": round(self.oof_rmse, 2),
            "fold_rmse_mean": round(float(np.mean(self.fold_rmse)), 2),
            "fold_rmse_std": round(float(np.std(self.fold_rmse)), 2),
            "n_folds": len(self.fold_rmse),
            "seconds": round(self.seconds, 1),
            "fold_rmse": self.fold_rmse,
            "oof_rmse_full_precision": self.oof_rmse,
            "raw_oof_rmse": self.raw_oof_rmse,
            "best_iterations": self.best_iterations,
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


def fold_matrices(spec, train_raw, others, drop=None, seed=SEED):
    """Fit every learned transform on exactly these training rows."""
    if spec.matrix == "raw":
        X_tr, X_others = MATRIX_BUILDERS["raw"](train_raw, others, drop)
        return train_raw, others, X_tr, X_others
    if spec.matrix == "structural":
        # No learned preprocessing is consumed by this regressor.
        from src.features import MRP_BAND_EDGES
        frames = [f.assign(MRP_Band=pd.cut(f.Item_MRP, MRP_BAND_EDGES, labels=False))
                  for f in [train_raw, *others]]
        X_tr, X_others = MATRIX_BUILDERS["structural"](frames[0], frames[1:], drop)
        return frames[0], frames[1:], X_tr, X_others
    cleaner = Cleaner().fit([train_raw])
    rows_tr = cleaner.transform(train_raw)
    rows_others = [cleaner.transform(f) for f in others]
    builder = FeatureBuilder().fit([rows_tr])
    rows_tr = builder.transform(rows_tr)
    rows_others = [builder.transform(f) for f in rows_others]
    X_tr, X_others = MATRIX_BUILDERS[spec.matrix](rows_tr, rows_others, drop)
    if spec.item_popularity:
        pop_tr, pop_others = item_popularity(rows_tr, train_raw[TARGET].to_numpy(), rows_others, seed=seed)
        X_tr = X_tr.assign(Item_Popularity=pop_tr)
        X_others = [f.assign(Item_Popularity=p) for f, p in zip(X_others, pop_others)]
    return rows_tr, rows_others, X_tr, X_others


def _early_stopping_iterations(spec: ModelSpec, params: dict, seed: int, train_raw, drop=None) -> int:
    """Best iteration count from a split of the training fold, on the weighted
    objective the model is trained with."""
    idx_a, idx_b = train_test_split(
        np.arange(len(train_raw)), test_size=EARLY_STOPPING_FRACTION,
        random_state=seed, stratify=train_raw[OUTLET_ID]
    )
    rows_a, (rows_b,), X_a, (X_b,) = fold_matrices(
        spec, train_raw.iloc[idx_a], [train_raw.iloc[idx_b]], drop, seed
    )
    transform = TRANSFORMS[spec.target]
    z_a = transform.forward(rows_a[TARGET].to_numpy(), rows_a)
    z_b = transform.forward(rows_b[TARGET].to_numpy(), rows_b)
    w_a, w_b = transform.weight(rows_a), transform.weight(rows_b)

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
        cat = _catboost_cat_features(X_a)
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
    fold_fit: bool = True,
    use_test_features: bool = False,
    drop: set[str] | None = None,
    early_stopping: bool | None = None,
    split_indices: list | None = None,
) -> CVResult:
    """Out-of-fold predictions are averaged over repeats; test predictions are
    averaged over every fold model.

    `prepared` is reused across models when preprocessing is fit once. With
    `fold_fit`, cleaning and feature statistics are refit on each training
    fold and applied to the validation rows and test.
    """
    start = time.perf_counter()
    if drop is None and prepared is not None:
        drop = prepared.drop
    params = {**(params or {})}
    if early_stopping is None:
        early_stopping = spec.iterations_param is not None
    if not fold_fit and prepared is None:
        prepared = prepare(train_raw, test_raw, use_test_features, drop, [spec.matrix])
    train_frame = train_raw if fold_fit else prepared.train
    y = train_raw[TARGET].to_numpy(dtype=float)
    transform = TRANSFORMS[spec.target]

    oof_sum, oof_count = np.zeros(len(train_raw)), np.zeros(len(train_raw))
    test_sum, n_models = np.zeros(len(test_raw)), 0
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    negatives = 0
    raw_sum = np.zeros(len(y))
    saved_splits, fold_predictions = [], []

    splits = split_indices if split_indices is not None else folds(scheme, train_frame, n_splits, n_repeats, seed)
    for fold, (tr, va) in enumerate(splits):
        tr, va = np.asarray(tr), np.asarray(va)
        if np.intersect1d(tr, va).size or len(np.unique(va)) != len(va):
            raise ValueError("Train/validation indices must be disjoint and unique")
        saved_splits.append((tr, va))
        if fold_fit:
            rows_tr, (rows_va, rows_te), X_tr, (X_va, X_te) = fold_matrices(
                spec, train_raw.iloc[tr], [train_raw.iloc[va], test_raw], drop, seed + fold
            )
        else:
            rows_tr, rows_va, rows_te = prepared.train.iloc[tr], prepared.train.iloc[va], prepared.test
            X, X_te = prepared.matrices[spec.matrix]
            X_tr, X_va = X.iloc[tr], X.iloc[va]

        if spec.item_popularity and not fold_fit:
            pop_tr, (pop_va, pop_te) = item_popularity(rows_tr, y[tr], [rows_va, rows_te], seed=seed + fold)
            X_tr, X_va, X_te = (f.assign(Item_Popularity=p) for f, p in ((X_tr, pop_tr), (X_va, pop_va), (X_te, pop_te)))

        z_tr, w_tr = transform.forward(y[tr], rows_tr), transform.weight(rows_tr)
        fold_params = dict(params)
        if early_stopping:
            best = _early_stopping_iterations(spec, params, seed + fold, train_raw.iloc[tr], drop)
            best_iterations.append(best)
            fold_params[spec.iterations_param] = best

        model = spec.make(fold_params, seed + fold)
        model.fit(X_tr, z_tr, **_fit_kwargs(spec, model, X_tr, w_tr))
        if not early_stopping and hasattr(model, "best_iteration_count_"):
            best_iterations.append(model.best_iteration_count_)

        raw_va = transform.inverse(model.predict(X_va), rows_va)
        raw_te = transform.inverse(model.predict(X_te), rows_te)
        negatives += int((raw_va < 0).sum())
        # Sales cannot be negative; clipping only ever removes error (checked in src/validate.py).
        pred_va, pred_te = np.clip(raw_va, 0, None), np.clip(raw_te, 0, None)

        oof_sum[va] += pred_va
        raw_sum[va] += raw_va
        fold_predictions.append(pred_va)
        oof_count[va] += 1
        test_sum += pred_te
        n_models += 1
        fold_scores.append(rmse(y[va], pred_va))

    covered = oof_count > 0
    oof = np.full(len(y), np.nan)
    oof[covered] = oof_sum[covered] / oof_count[covered]
    raw_oof = np.full(len(y), np.nan)
    raw_oof[covered] = raw_sum[covered] / oof_count[covered]
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
        raw_oof_rmse=rmse(y[covered], raw_oof[covered]),
        raw_oof=raw_oof,
        split_indices=saved_splits,
        fold_predictions=fold_predictions,
    )
