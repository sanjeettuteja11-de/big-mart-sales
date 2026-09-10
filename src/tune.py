"""Tune one model's hyperparameters with Optuna, scored by cross-validated RMSE.

    python -m src.tune --model lightgbm_units --trials 60
    python -m src.tune --model catboost_units --trials 40 --repeats 2

The best parameters go to outputs/params/<model>.json, which src/train.py
picks up automatically. Every trial uses the same folds, so trials are compared
on identical data; the hand-picked defaults run first, so tuning can only
improve on them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna

from src.config import DATA_RAW, N_SPLITS, SEED
from src.cv import build_matrices, run_cv
from src.data import clean, load_raw
from src.features import build_features
from src.models import PARAMS_DIR, SPECS_BY_NAME


def lightgbm_space(t: optuna.Trial) -> dict:
    return dict(
        n_estimators=t.suggest_int("n_estimators", 200, 2000, log=True),
        learning_rate=t.suggest_float("learning_rate", 0.005, 0.1, log=True),
        num_leaves=t.suggest_int("num_leaves", 4, 64, log=True),
        min_child_samples=t.suggest_int("min_child_samples", 10, 300, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 100, log=True),
    )


def xgboost_space(t: optuna.Trial) -> dict:
    return dict(
        n_estimators=t.suggest_int("n_estimators", 200, 2000, log=True),
        learning_rate=t.suggest_float("learning_rate", 0.005, 0.1, log=True),
        max_depth=t.suggest_int("max_depth", 2, 8),
        min_child_weight=t.suggest_float("min_child_weight", 1, 200, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 100, log=True),
    )


def catboost_space(t: optuna.Trial) -> dict:
    return dict(
        iterations=t.suggest_int("iterations", 300, 3000, log=True),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        depth=t.suggest_int("depth", 3, 8),
        l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 50, log=True),
        random_strength=t.suggest_float("random_strength", 0.1, 10, log=True),
        bagging_temperature=t.suggest_float("bagging_temperature", 0.0, 2.0),
    )


def hist_gb_space(t: optuna.Trial) -> dict:
    return dict(
        max_iter=t.suggest_int("max_iter", 100, 1500, log=True),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_leaf_nodes=t.suggest_int("max_leaf_nodes", 4, 64, log=True),
        min_samples_leaf=t.suggest_int("min_samples_leaf", 10, 300, log=True),
        l2_regularization=t.suggest_float("l2_regularization", 1e-3, 100, log=True),
    )


def forest_space(t: optuna.Trial) -> dict:
    return dict(
        n_estimators=300,
        max_depth=t.suggest_int("max_depth", 4, 16),
        min_samples_leaf=t.suggest_int("min_samples_leaf", 5, 100, log=True),
        max_features=t.suggest_float("max_features", 0.2, 1.0),
    )


def ridge_space(t: optuna.Trial) -> dict:
    return dict(alpha=t.suggest_float("alpha", 1e-3, 1e3, log=True))


def poisson_space(t: optuna.Trial) -> dict:
    return dict(alpha=t.suggest_float("alpha", 1e-6, 1.0, log=True), max_iter=3000)


def store_rate_space(t: optuna.Trial) -> dict:
    return dict(smoothing=t.suggest_float("smoothing", 1, 2000, log=True))


SPACES = {
    "store_rate": store_rate_space,
    "lightgbm": lightgbm_space, "xgboost": xgboost_space, "catboost": catboost_space,
    "hist_gb": hist_gb_space, "random_forest": forest_space, "extra_trees": forest_space,
    "ridge": ridge_space, "poisson": poisson_space,
}


def space_for(model_name: str):
    return next(fn for prefix, fn in SPACES.items() if model_name.startswith(prefix))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, choices=sorted(SPECS_BY_NAME))
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None, help="seconds")
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data", type=Path, default=DATA_RAW)
    args = parser.parse_args()

    train, test = build_features(*clean(*load_raw(args.data)))
    matrices = build_matrices(train, test)
    spec = SPECS_BY_NAME[args.model]
    space = space_for(spec.name)

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        return run_cv(spec, train, test, matrices, params, args.splits, args.repeats, args.seed).oof_rmse

    def report(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(f"  trial {trial.number:3d}   RMSE {trial.value:8.2f}   best so far {study.best_value:8.2f}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.enqueue_trial(spec.params, skip_if_exists=True)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, callbacks=[report])

    best = space(optuna.trial.FixedTrial(study.best_params))
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    (PARAMS_DIR / f"{spec.name}.json").write_text(json.dumps(best, indent=2))
    (PARAMS_DIR / f"{spec.name}.study.json").write_text(json.dumps({
        "cv_rmse": round(study.best_value, 2),
        "default_params_rmse": round(study.trials[0].value, 2),
        "trials": len(study.trials),
        "splits": args.splits,
        "repeats": args.repeats,
    }, indent=2))
    print(f"\n{spec.name}: {study.trials[0].value:.2f} with defaults -> {study.best_value:.2f} tuned")
    print(f"Saved {PARAMS_DIR / (spec.name + '.json')}")


if __name__ == "__main__":
    main()
