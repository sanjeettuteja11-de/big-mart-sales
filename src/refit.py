"""Refit models on all training rows and write a checked submission.

    python -m src.refit store_rate_popularity
    python -m src.refit catboost_units --seeds 5
    python -m src.refit store_rate_popularity catboost_units --weights 0.7 0.3 --name final_blend

Cross-validated test predictions (src/train.py) average 15 models that each
saw 80% of the rows. A refit trains on all 8,523 rows instead. A boosting
model has no validation set left to stop on, so its round count is the mean
early-stopping round recorded in outputs/cv_results.json, scaled by
--rounds-scale (default 1.0). Scaling rounds is an optional heuristic,
not a validated improvement. Several seeds are averaged
to reduce the boosting models' run-to-run variance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_RAW, OUTPUTS, SEED, SUBMISSIONS, TARGET
from src.cv import _fit_kwargs, prepare, rmse
from src.data import load_raw
from src.models import SPECS_BY_NAME, load_tuned
from src.submission import check_submission
from src.targets import TRANSFORMS
from src.train import write_submission


def cv_record(name: str, outputs: Path = OUTPUTS) -> dict:
    path = outputs / "cv_results.json"
    if path.exists():
        for m in json.loads(path.read_text())["models"]:
            if m["model"] == name:
                return m
    return {}


def refit_predict(
    name: str,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    seeds: int = 1,
    rounds: int | None = None,
    rounds_scale: float = 1.0,
    outputs: Path = OUTPUTS,
) -> np.ndarray:
    spec = SPECS_BY_NAME[name]
    if spec.item_popularity:
        raise NotImplementedError(f"{name} needs out-of-fold encodings; refit is not supported for it")
    prepared = prepare(train_raw, test_raw, matrices=[spec.matrix]) if spec.matrix != "raw" else None
    X, X_test = prepared.matrices[spec.matrix] if prepared is not None else (train_raw.drop(columns=TARGET), test_raw)
    rows_train = prepared.train if prepared is not None else train_raw
    rows_test = prepared.test if prepared is not None else test_raw

    params = dict(load_tuned(name, outputs / "params"))
    if spec.iterations_param:
        rounds = rounds or cv_record(name, outputs).get("best_iterations_mean")
        if rounds is None:
            raise ValueError(f"No early-stopping rounds recorded for {name}: run src.train first or pass --rounds")
        params[spec.iterations_param] = max(1, round(rounds * rounds_scale))

    transform = TRANSFORMS[spec.target]
    y = rows_train[TARGET].to_numpy(dtype=float)
    z, w = transform.forward(y, rows_train), transform.weight(rows_train)
    predictions = []
    for s in range(seeds):
        model = spec.make(params, SEED + s)
        model.fit(X, z, **_fit_kwargs(spec, model, X, w))
        predictions.append(np.clip(transform.inverse(model.predict(X_test), rows_test), 0, None))
    return np.mean(predictions, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", choices=sorted(SPECS_BY_NAME))
    parser.add_argument("--weights", nargs="+", type=float, help="blend weights, one per model (normalised to sum to 1)")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--rounds-scale", type=float, default=1.0)
    parser.add_argument("--name", help="submission file name without .csv")
    parser.add_argument("--out-dir", type=Path, default=SUBMISSIONS)
    parser.add_argument("--data", type=Path, default=DATA_RAW)
    args = parser.parse_args()

    weights = np.array(args.weights if args.weights else [1.0] * len(args.models), dtype=float)
    if len(weights) != len(args.models) or (weights < 0).any() or weights.sum() == 0:
        parser.error("--weights needs one non-negative weight per model")
    weights /= weights.sum()

    train_raw, test_raw = load_raw(args.data)
    prediction = np.zeros(len(test_raw))
    for name, weight in zip(args.models, weights):
        pred = refit_predict(name, train_raw, test_raw, args.seeds, rounds_scale=args.rounds_scale)
        cv_test = OUTPUTS / "test_preds" / f"{name}.npy"
        if cv_test.exists():
            diff = rmse(np.load(cv_test), pred)
            print(f"  {name}: refit on all rows; RMS difference from the cross-validated test predictions {diff:.1f}")
        prediction += weight * pred

    if args.name:
        stem = args.name
    elif len(args.models) == 1:
        score = cv_record(args.models[0]).get("oof_rmse")
        stem = f"{args.models[0]}_refit" + (f"_cv{score:.0f}" if score else "")
    else:
        stem = "blend_refit"
    path = write_submission(test_raw, prediction, args.out_dir / f"{stem}.csv")

    problems = check_submission(path, test_raw, float(train_raw[TARGET].max()))
    if problems:
        raise SystemExit(f"{path} FAILED checks:\n  " + "\n  ".join(problems))
    summary = pd.Series(prediction).describe()
    print(f"Wrote {path}: {len(prediction)} rows, all checks passed; predictions "
          f"{summary['min']:.0f} to {summary['max']:.0f}, mean {summary['mean']:.0f}")


if __name__ == "__main__":
    main()
