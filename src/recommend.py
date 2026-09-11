"""Choose the final model from saved strict-CV runs, refit it, and record the evidence.

    python -m src.recommend
    python -m src.recommend --candidate lattice_store_rate --baseline store_rate_popularity

Reads the strict-CV runs listed in RUNS (written by src.price_study and
src.continue_study under outputs/strict_study/), compares candidate and
baseline seed by seed, refits the candidate on all training rows with
src.refit, checks the submission file, and writes
outputs/strict_study/recommendation.json.

A fold-level paired comparison is reported only when both runs of a seed
saved identical fold indices; otherwise only the two overall scores are
listed. The CV numbers say nothing about a leaderboard score: the file has
none until it is uploaded and evaluated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.artifacts import provenance, sha256
from src.config import OUTPUTS, SUBMISSIONS, TARGET
from src.data import load_raw
from src.refit import refit_predict
from src.submission import check_submission
from src.train import write_submission

STUDY = OUTPUTS / "strict_study"

# Strict-CV runs (5-fold x 3 repeats, preprocessing fitted inside folds) per model, one per seed.
RUNS = {
    "lattice_store_rate": ["price/candidate_0_seed_42", "price/candidate_0_seed_137", "price/candidate_0_seed_2026"],
    "store_rate_popularity": ["baseline/store_rate_popularity", "confirm/baseline_seed_137", "confirm/baseline_seed_2026"],
}


def _same_folds(a: Path, b: Path) -> bool:
    """True when both runs saved the same train/validation indices.

    folds.npz also stores each fold's predictions (pred_*), which differ
    between models by design, so those arrays are not compared.
    """
    fa, fb = (np.load(p / "folds.npz", allow_pickle=True) for p in (a, b))
    keys_a = sorted(k for k in fa.files if not k.startswith("pred_"))
    keys_b = sorted(k for k in fb.files if not k.startswith("pred_"))
    return bool(keys_a) and keys_a == keys_b and all(np.array_equal(fa[k], fb[k]) for k in keys_a)


def evidence(candidate: str, baseline: str) -> dict:
    per_seed = []
    for cand_path, base_path in zip(RUNS[candidate], RUNS[baseline]):
        cand_dir, base_dir = STUDY / cand_path, STUDY / base_path
        cand, base = (json.loads((d / "result.json").read_text()) for d in (cand_dir, base_dir))
        row = {
            "seed": cand.get("config", {}).get("seed"),
            "candidate_run": cand_path, "baseline_run": base_path,
            "candidate_rmse": cand["oof_rmse_full_precision"],
            "baseline_rmse": base["oof_rmse_full_precision"],
            "delta": cand["oof_rmse_full_precision"] - base["oof_rmse_full_precision"],
            "identical_folds": _same_folds(cand_dir, base_dir),
        }
        if row["identical_folds"] and len(cand["fold_rmse"]) == len(base["fold_rmse"]):
            d = np.subtract(cand["fold_rmse"], base["fold_rmse"])
            row["paired_fold_delta_mean"] = float(d.mean())
            row["paired_fold_delta_std"] = float(d.std(ddof=1))
            row["folds_candidate_better"] = f"{int((d < 0).sum())}/{len(d)}"
        per_seed.append(row)
    return {
        "per_seed": per_seed,
        "candidate_mean_rmse": float(np.mean([r["candidate_rmse"] for r in per_seed])),
        "baseline_mean_rmse": float(np.mean([r["baseline_rmse"] for r in per_seed])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", default="lattice_store_rate", choices=sorted(RUNS))
    parser.add_argument("--baseline", default="store_rate_popularity", choices=sorted(RUNS))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    train, test = load_raw()
    ev = evidence(args.candidate, args.baseline)
    for r in ev["per_seed"]:
        paired = (f"; paired folds: mean {r['paired_fold_delta_mean']:+.3f} ± {r['paired_fold_delta_std']:.3f}, "
                  f"candidate better in {r['folds_candidate_better']}") if "paired_fold_delta_mean" in r else "; folds differ, unpaired"
        print(f"  seed {r['seed']}: {args.candidate} {r['candidate_rmse']:.3f} vs {args.baseline} {r['baseline_rmse']:.3f} "
              f"({r['delta']:+.3f}){paired}")

    out = args.out or SUBMISSIONS / f"final_{args.candidate}_refit.csv"
    prediction = refit_predict(args.candidate, train, test)
    write_submission(test, prediction, out)
    problems = check_submission(out, test, float(train[TARGET].max()))
    if problems:
        raise SystemExit(f"{out} FAILED checks:\n  " + "\n  ".join(problems))

    record = {
        "candidate": args.candidate,
        "baseline": args.baseline,
        "evidence": ev,
        "selected_because": "lowest mean strict-CV RMSE over the seeds above; blends with boosting models did not improve it",
        "submission": {"path": str(out.relative_to(out.parents[1])) if len(out.parents) > 1 else str(out),
                       "sha256": sha256(out), "rows": len(prediction), "checks_passed": True,
                       "refit": "all training rows"},
        "leaderboard_score": None,
        "note": "Not a leaderboard result. The file has no score until it is uploaded and evaluated.",
        "provenance": provenance(train, test),
    }
    STUDY.mkdir(parents=True, exist_ok=True)
    (STUDY / "recommendation.json").write_text(json.dumps(record, indent=2))
    print(f"\nWrote {out} (checks passed) and {STUDY / 'recommendation.json'}")


if __name__ == "__main__":
    main()
