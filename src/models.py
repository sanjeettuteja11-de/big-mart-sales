"""The model zoo: every candidate model, the matrix it reads and the target it learns.

`matrix` names an entry of `cv.build_matrices`; `target` names an entry of
`targets.TRANSFORMS`. The boosting libraries are optional: a spec whose library
cannot be imported is reported and skipped rather than stopping the run.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ITEM_ID, OUTPUTS, SEED
from src.features import CATEGORICAL

PARAMS_DIR = OUTPUTS / "params"


def library_status(module: str | None) -> str | None:
    """None when `module` imports cleanly, otherwise the reason it does not."""
    if module is None:
        return None
    try:
        importlib.import_module(module)
    except Exception as exc:  # XGBoost raises its own error type when libomp is missing
        return f"{type(exc).__name__}: {str(exc).strip().splitlines()[0]}"
    return None


def load_tuned(name: str, params_dir: Path = PARAMS_DIR) -> dict:
    path = Path(params_dir) / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    matrix: str
    target: str
    build: Callable[[dict, int], object]
    params: dict
    library: str | None = None
    item_popularity: bool = False
    # Name of the boosting-rounds parameter; set on models that support
    # early stopping, which src/cv.py then uses to pick the round count.
    iterations_param: str | None = None

    def make(self, overrides: dict | None = None, seed: int = SEED):
        return self.build({**self.params, **(overrides or {})}, seed)


def _ridge(p, seed):
    return make_pipeline(StandardScaler(), Ridge(**p))


def _poisson(p, seed):
    return make_pipeline(StandardScaler(), PoissonRegressor(**p))


def _hist_gb(p, seed):
    return HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=seed, **p)


def _random_forest(p, seed):
    return RandomForestRegressor(n_jobs=-1, random_state=seed, **p)


def _extra_trees(p, seed):
    return ExtraTreesRegressor(n_jobs=-1, random_state=seed, **p)


def _lightgbm(p, seed):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(random_state=seed, verbose=-1, n_jobs=-1, **p)


def _xgboost(p, seed):
    from xgboost import XGBRegressor

    return XGBRegressor(
        random_state=seed, tree_method="hist", enable_categorical=True, n_jobs=-1, **p
    )


def _catboost(p, seed):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
        thread_count=-1,
        cat_features=CATEGORICAL + [ITEM_ID],
        **p,
    )


# Defaults lean shallow and heavily regularised: 8.5k rows of very noisy sales
# reward caution far more than capacity. The round counts are ceilings; early
# stopping picks the actual count per fold. `src/tune.py` refines the rest.
LIGHTGBM = dict(
    n_estimators=2000, learning_rate=0.02, num_leaves=12, min_child_samples=60,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
)
XGBOOST = dict(
    n_estimators=2000, learning_rate=0.02, max_depth=4, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
)
CATBOOST = dict(iterations=3000, learning_rate=0.03, depth=5, l2_leaf_reg=6.0)

_LGB = dict(library="lightgbm", iterations_param="n_estimators")
_XGB = dict(library="xgboost", iterations_param="n_estimators")
_CAT = dict(library="catboost", iterations_param="iterations")

def _store_rate(p, seed):
    from src.structure import StoreRatePopularity

    return StoreRatePopularity(**p)


SPECS = [
    # Sales = whole units x price, and units depend on little but the store
    # (src/structure.py). Units are independent of price, so the plain mean
    # ("units_unweighted") estimates each store's rate more efficiently than
    # an MRP^2-weighted one.
    ModelSpec("store_rate_popularity", "catboost", "units_unweighted", _store_rate, dict(smoothing=50.0)),
    # Variants that test which part of it carries over to the test set.
    ModelSpec("store_rate_plain", "catboost", "units_unweighted", _store_rate,
              dict(smoothing=float("inf"))),
    ModelSpec("store_type_rate_popularity", "catboost", "units_unweighted", _store_rate,
              dict(smoothing=55.0, rate_level="type")),
    ModelSpec("ridge_interactions", "linear", "raw", _ridge, dict(alpha=3.0)),
    ModelSpec("poisson_glm", "glm", "raw", _poisson, dict(alpha=1e-4, max_iter=3000)),
    ModelSpec(
        "hist_gb_units", "tree", "units", _hist_gb,
        dict(learning_rate=0.03, max_iter=400, max_leaf_nodes=15, min_samples_leaf=60,
             l2_regularization=1.0),
    ),
    ModelSpec(
        "random_forest_units", "codes", "units", _random_forest,
        dict(n_estimators=400, max_depth=8, min_samples_leaf=30, max_features=0.5),
    ),
    ModelSpec(
        "extra_trees_units", "codes", "units", _extra_trees,
        dict(n_estimators=400, max_depth=10, min_samples_leaf=20, max_features=0.6),
    ),
    ModelSpec("lightgbm_raw", "tree", "raw", _lightgbm, LIGHTGBM, **_LGB),
    ModelSpec("lightgbm_log1p", "tree", "log1p", _lightgbm, LIGHTGBM, **_LGB),
    ModelSpec("lightgbm_units", "tree", "units", _lightgbm, LIGHTGBM, **_LGB),
    ModelSpec("lightgbm_units_popularity", "tree", "units", _lightgbm, LIGHTGBM, item_popularity=True, **_LGB),
    ModelSpec("xgboost_raw", "tree", "raw", _xgboost, XGBOOST, **_XGB),
    ModelSpec("xgboost_units", "tree", "units", _xgboost, XGBOOST, **_XGB),
    ModelSpec("catboost_raw", "catboost", "raw", _catboost, CATBOOST, **_CAT),
    ModelSpec("catboost_log1p", "catboost", "log1p", _catboost, CATBOOST, **_CAT),
    ModelSpec("catboost_units", "catboost", "units", _catboost, CATBOOST, **_CAT),
]

SPECS_BY_NAME = {spec.name: spec for spec in SPECS}
