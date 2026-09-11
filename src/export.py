"""Write submission files from saved cross-validation predictions.

    python -m src.export store_rate_plain catboost_raw                 # one file per model
    python -m src.export --average --name top5 store_rate_popularity ridge_interactions poisson_glm
    python -m src.export store_rate_plain --out-dir submissions/round1

Reads outputs/oof/<model>.npy and outputs/test_preds/<model>.npy, which
src/train.py writes, so run that first. The CV score in each file name comes
from the same out-of-fold predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import DATA_RAW, OUTPUTS, SUBMISSIONS, TARGET
from src.cv import rmse
from src.data import load_raw
from src.train import write_submission


def load_predictions(names: list[str], outputs: Path = OUTPUTS) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold and test predictions, averaged over `names`."""
    oof = np.mean([np.load(outputs / "oof" / f"{n}.npy") for n in names], axis=0)
    test = np.mean([np.load(outputs / "test_preds" / f"{n}.npy") for n in names], axis=0)
    return oof, test


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("models", nargs="+")
    parser.add_argument("--average", action="store_true", help="write one file averaging all models")
    parser.add_argument("--name", help="file name for --average (default: average_<n>_models)")
    parser.add_argument("--out-dir", type=Path, default=SUBMISSIONS)
    parser.add_argument("--data", type=Path, default=DATA_RAW)
    args = parser.parse_args()

    train, test = load_raw(args.data)
    y = train[TARGET].to_numpy(dtype=float)
    groups = [args.models] if args.average else [[m] for m in args.models]
    for members in groups:
        oof, pred = load_predictions(members)
        if (len(oof), len(pred)) != (len(train), len(test)):
            raise ValueError(f"Saved predictions for {members} don't match the data; rerun src.train")
        stem = (args.name or f"average_{len(members)}_models") if args.average else members[0]
        score = rmse(y, oof)
        path = write_submission(test, pred, args.out_dir / f"{stem}_cv{score:.0f}.csv")
        print(f"CV {score:8.2f}  {path}")


if __name__ == "__main__":
    main()
