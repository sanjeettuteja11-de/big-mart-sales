"""Cross-validate every model, blend them, and write leaderboard-ready submissions.

    python -m src.train                                   # all models, 5-fold x 3 repeats
    python -m src.train --repeats 1                       # about 3x faster
    python -m src.train --models lightgbm_units catboost_units
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.blend import blend_cv_rmse, fit_weights
from src.config import (
    DATA_RAW, ITEM_ID, N_REPEATS, N_SPLITS, OUTLET_ID, OUTPUTS, SEED, SUBMISSIONS, TARGET,
)
from src.cv import prepare, rmse, run_cv
from src.data import load_raw
from src.models import SPECS, SPECS_BY_NAME, library_status, load_tuned


def write_submission(test: pd.DataFrame, pred: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        ITEM_ID: test[ITEM_ID],
        OUTLET_ID: test[OUTLET_ID],
        TARGET: np.clip(pred, 0, None),
    }).to_csv(path, index=False)
    return path


def run(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    model_names: list[str] | None = None,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    seed: int = SEED,
    outputs: Path = OUTPUTS,
    submissions: Path = SUBMISSIONS,
) -> dict:
    prepared = prepare(train_raw, test_raw)
    train, test = prepared.train, prepared.test
    y = train[TARGET].to_numpy(dtype=float)
    for folder in (outputs / "oof", outputs / "test_preds"):
        folder.mkdir(parents=True, exist_ok=True)

    baseline = rmse(y, np.full_like(y, y.mean()))
    print(f"{len(train):,} train rows, {len(test):,} test rows. "
          f"Baseline (always predict the mean): RMSE {baseline:,.1f}\n")

    results, skipped = [], {}
    for spec in [SPECS_BY_NAME[n] for n in model_names] if model_names else SPECS:
        problem = library_status(spec.library)
        if problem:
            skipped[spec.name] = problem
            print(f"  skipped {spec.name}: {problem}")
            continue
        tuned = load_tuned(spec.name, outputs / "params")
        result = run_cv(spec, train_raw, test_raw, prepared, tuned, n_splits, n_repeats, seed)
        np.save(outputs / "oof" / f"{spec.name}.npy", result.oof)
        np.save(outputs / "test_preds" / f"{spec.name}.npy", result.test)
        results.append(result)
        s = result.summary()
        rounds = f", {s['best_iterations_mean']} rounds" if "best_iterations_mean" in s else ""
        print(f"  {spec.name:27s} OOF RMSE {s['oof_rmse']:8.2f}   "
              f"folds {s['fold_rmse_mean']:.1f} ± {s['fold_rmse_std']:.1f}   "
              f"{'tuned' if tuned else 'default'} params{rounds}, {s['seconds']}s")
    if not results:
        raise RuntimeError("No model could run; see the skipped reasons above.")

    results.sort(key=lambda r: r.oof_rmse)
    oof_matrix = np.column_stack([r.oof for r in results])
    test_matrix = np.column_stack([r.test for r in results])
    weights = fit_weights(oof_matrix, y)
    blend_honest = blend_cv_rmse(oof_matrix, y, seed=seed)
    best = results[0]

    best_path = write_submission(test, best.test, submissions / f"{best.name}_cv{best.oof_rmse:.0f}.csv")
    blend_path = write_submission(test, test_matrix @ weights, submissions / f"blend_cv{blend_honest:.0f}.csv")
    recommended = blend_path if blend_honest < best.oof_rmse else best_path

    report = {
        "rows": {"train": len(train), "test": len(test)},
        "cv": {"n_splits": n_splits, "n_repeats": n_repeats, "seed": seed},
        "baseline_rmse": round(baseline, 2),
        "models": [r.summary() for r in results],
        "skipped": skipped,
        "blend": {
            "weights": {r.name: round(float(w), 4) for r, w in zip(results, weights) if w > 1e-4},
            "oof_rmse": round(rmse(y, oof_matrix @ weights), 2),
            "honest_cv_rmse": round(blend_honest, 2),
        },
        "submissions": {
            "best_single": best_path.name,
            "blend": blend_path.name,
            "recommended": recommended.name,
        },
    }
    (outputs / "cv_results.json").write_text(json.dumps(report, indent=2))

    print(f"\n  best single: {best.name} ({best.oof_rmse:.2f})")
    print(f"  blend:       {blend_honest:.2f} honest CV   weights {report['blend']['weights']}")
    print(f"\nUpload this file: {recommended}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", choices=sorted(SPECS_BY_NAME), metavar="MODEL")
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data", type=Path, default=DATA_RAW, help="folder with train/test CSVs")
    args = parser.parse_args()
    run(*load_raw(args.data), args.models, args.splits, args.repeats, args.seed)


if __name__ == "__main__":
    main()
