"""Create the leaderboard-anchored BigMart v24 submission.

This final-stage script combines leakage-free OOF/test prediction artifacts made
by bigmart_v5_reproduce.py and the component models evaluated in the v22-v23
pipeline. The repeated Ridge meta-model is deliberately blended back toward v5,
which is the best submission with a measured public-leaderboard result.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path("/Users/abhimanyunain/Desktop/CODEX")
DOWNLOADS = Path("/Users/abhimanyunain/Downloads")
TRAIN_PATH = DOWNLOADS / "train_v9rqX0R.csv"
TEST_PATH = DOWNLOADS / "test_AbJTz2l.csv"
OUTPUT_PATH = DOWNLOADS / "bigmart_submission_v24_lb_anchored.csv"

ARTIFACTS = [
    "v5_reproduced_preds",
    "outlet_mrp",
    "cat_True",
    "cb_unw",
    "ridge_ratio_100",
    "lgb7",
    "gbr2",
]
META_SEEDS = [1, 42, 123, 456, 999]


def load_predictions() -> tuple[np.ndarray, np.ndarray]:
    oof_columns = []
    test_columns = []
    for name in ARTIFACTS:
        artifact = np.load(BASE / f"{name}.npz")
        oof_columns.append(artifact["oof"])
        test_columns.append(artifact["test"])
    return np.column_stack(oof_columns), np.column_stack(test_columns)


def meta_features(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    outlets: list[str],
    weight_median: float,
) -> np.ndarray:
    outlet_matrix = np.column_stack(
        [frame["Outlet_Identifier"].eq(outlet) for outlet in outlets]
    )
    mrp = frame["Item_MRP"].to_numpy()[:, None]
    return np.column_stack(
        [
            predictions,
            predictions.mean(axis=1),
            predictions.std(axis=1),
            predictions.max(axis=1) - predictions.min(axis=1),
            mrp,
            outlet_matrix,
            outlet_matrix * mrp,
            outlet_matrix * predictions[:, [0]],
            frame["Item_Visibility"].to_numpy(),
            frame["Item_Weight"].fillna(weight_median).to_numpy(),
        ]
    )


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    target = train["Item_Outlet_Sales"].to_numpy()
    oof_base, test_base = load_predictions()

    outlets = sorted(train["Outlet_Identifier"].unique())
    weight_median = pd.concat(
        [train["Item_Weight"], test["Item_Weight"]], ignore_index=True
    ).median()
    x_train = meta_features(train, oof_base, outlets, weight_median)
    x_test = meta_features(test, test_base, outlets, weight_median)

    repeated_oof = []
    repeated_test = []
    for seed in META_SEEDS:
        splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
        fold_oof = np.zeros(len(train))
        fold_test = np.zeros(len(test))
        for fit_index, validation_index in splitter.split(x_train):
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(x_train[fit_index], target[fit_index])
            fold_oof[validation_index] = model.predict(x_train[validation_index])
            fold_test += model.predict(x_test) / splitter.n_splits
        repeated_oof.append(fold_oof)
        repeated_test.append(fold_test)

    meta_oof = np.mean(repeated_oof, axis=0)
    meta_test = np.mean(repeated_test, axis=0)
    v5_oof = oof_base[:, 0]
    v5_test = test_base[:, 0]

    # Public-LB anchor: retain 37.5% of the best measured leaderboard model.
    final_oof = 0.625 * meta_oof + 0.375 * v5_oof
    final_test = 0.625 * meta_test + 0.375 * v5_test
    rmse = np.sqrt(np.mean((target - final_oof) ** 2))

    submission = test[["Item_Identifier", "Outlet_Identifier"]].copy()
    submission["Item_Outlet_Sales"] = np.clip(final_test, 33.29, 13087.0)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Repeated-meta OOF RMSE: {np.sqrt(np.mean((target-meta_oof)**2)):.4f}")
    print(f"Anchored v24 OOF RMSE: {rmse:.4f}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
