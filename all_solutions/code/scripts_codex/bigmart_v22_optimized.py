"""Reproducible BigMart Sales III solution.

The final prediction blends:
  * a low-variance outlet-specific sales-per-MRP model;
  * a conservative CatBoost sales model; and
  * a CatBoost model for the inferred integer unit count.

Usage:
  python bigmart_optimized_solution.py \
      --train train_v9rqX0R.csv \
      --test test_AbJTz2l.csv \
      --output bigmart_submission_optimized.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold


TARGET = "Item_Outlet_Sales"
ID_COLUMNS = ["Item_Identifier", "Outlet_Identifier"]
CATEGORICAL_FEATURES = [
    "Item_Identifier",
    "Item_Fat_Content",
    "Item_Type",
    "Outlet_Identifier",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type",
    "Item_Category",
    "MRP_Bin",
    "Visibility_Zero",
]
NUMERIC_FEATURES = [
    "Item_Weight",
    "Item_Visibility",
    "Item_MRP",
    "Outlet_Age",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
SALES_FEATURES = [
    "Item_Identifier",
    "Item_Weight",
    "Item_Fat_Content",
    "Item_Visibility",
    "Item_Type",
    "Item_MRP",
    "Outlet_Identifier",
    "Outlet_Establishment_Year",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type",
    "Item_Category",
    "Outlet_Age",
    "MRP_Bin",
    "Visibility_Zero",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train_v9rqX0R.csv")
    parser.add_argument("--test", default="test_AbJTz2l.csv")
    parser.add_argument("--output", default="bigmart_submission_optimized.csv")
    parser.add_argument(
        "--baseline-output", default="bigmart_submission_outlet_mrp.csv"
    )
    return parser.parse_args()


def prepare_data(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    combined = pd.concat(
        [train.drop(columns=TARGET), test], ignore_index=True, sort=False
    )

    weight_by_item = combined.groupby("Item_Identifier")["Item_Weight"].median()
    global_weight = combined["Item_Weight"].median()

    for frame in (train, test):
        frame["Item_Fat_Content"] = frame["Item_Fat_Content"].replace(
            {"low fat": "Low Fat", "LF": "Low Fat", "reg": "Regular"}
        )
        frame["Item_Category"] = frame["Item_Identifier"].str[:2]
        frame["Outlet_Age"] = 2013 - frame["Outlet_Establishment_Year"]
        frame["MRP_Bin"] = pd.cut(
            frame["Item_MRP"],
            bins=[0, 70, 140, 200, 300],
            labels=False,
            include_lowest=True,
        ).astype(str)
        frame["Visibility_Zero"] = (frame["Item_Visibility"] == 0).astype(str)
        frame["Outlet_Size"] = frame["Outlet_Size"].fillna("Missing")
        frame["Item_Weight"] = (
            frame["Item_Weight"]
            .fillna(frame["Item_Identifier"].map(weight_by_item))
            .fillna(global_weight)
        )
        for column in CATEGORICAL_FEATURES:
            frame[column] = frame[column].fillna("Missing").astype(str)

    return train, test


def new_catboost(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=10,
        random_strength=0.5,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        # Fixing the worker count keeps CatBoost's fold results reproducible.
        thread_count=8,
    )


def outlet_mrp_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    folds: KFold,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(train))
    test_prediction = np.zeros(len(test))

    for train_index, validation_index in folds.split(train):
        train_fold = train.iloc[train_index]
        validation_fold = train.iloc[validation_index]
        for outlet in train["Outlet_Identifier"].unique():
            fit_mask = train_fold["Outlet_Identifier"].eq(outlet).to_numpy()
            validation_mask = validation_fold["Outlet_Identifier"].eq(outlet).to_numpy()
            model = LinearRegression(fit_intercept=False)
            model.fit(
                train_fold.loc[fit_mask, ["Item_MRP"]],
                target[train_index][fit_mask],
            )
            target_rows = validation_index[validation_mask]
            oof[target_rows] = model.predict(
                train.loc[target_rows, ["Item_MRP"]]
            )

    for outlet in train["Outlet_Identifier"].unique():
        fit_mask = train["Outlet_Identifier"].eq(outlet)
        test_mask = test["Outlet_Identifier"].eq(outlet)
        model = LinearRegression(fit_intercept=False)
        model.fit(train.loc[fit_mask, ["Item_MRP"]], target[fit_mask])
        test_prediction[test_mask] = model.predict(
            test.loc[test_mask, ["Item_MRP"]]
        )

    return oof, test_prediction


def catboost_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    folds: KFold,
    fit_target: np.ndarray,
    sales_target: np.ndarray,
    reconstruct_with_mrp: bool,
    seed_offset: int,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    oof_sales = np.zeros(len(train))
    test_sales = np.zeros(len(test))
    categorical_indices = [
        feature_columns.index(column) for column in CATEGORICAL_FEATURES
    ]

    for fold_number, (train_index, validation_index) in enumerate(
        folds.split(train), start=1
    ):
        model = new_catboost(seed_offset + fold_number)
        model.fit(
            train.iloc[train_index][feature_columns],
            fit_target[train_index],
            cat_features=categorical_indices,
            eval_set=(
                train.iloc[validation_index][feature_columns],
                fit_target[validation_index],
            ),
            early_stopping_rounds=200,
            verbose=False,
        )

        validation_prediction = model.predict(
            train.iloc[validation_index][feature_columns]
        )
        fold_test_prediction = model.predict(test[feature_columns])
        if reconstruct_with_mrp:
            validation_prediction *= train.iloc[validation_index]["Item_MRP"].to_numpy()
            fold_test_prediction *= test["Item_MRP"].to_numpy()

        oof_sales[validation_index] = validation_prediction
        test_sales += fold_test_prediction / folds.n_splits

    return oof_sales, test_sales


def write_submission(
    test: pd.DataFrame, prediction: np.ndarray, output_path: Path
) -> None:
    submission = test[ID_COLUMNS].copy()
    submission[TARGET] = np.clip(prediction, 0, None)
    submission.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    train_raw = pd.read_csv(args.train)
    test_raw = pd.read_csv(args.test)
    train, test = prepare_data(train_raw, test_raw)

    sales = train[TARGET].to_numpy()
    # The target is generated as an integer unit count times a unit price that is
    # within +/- 2 of MRP. Rounding Sales/MRP recovers that count very accurately.
    units = np.rint(sales / train["Item_MRP"].to_numpy())
    reconstructed_sales = units * train["Item_MRP"].to_numpy()
    print(
        "Integer-unit reconstruction RMSE:",
        f"{root_mean_squared_error(sales, reconstructed_sales):.4f}",
    )

    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    outlet_oof, outlet_test = outlet_mrp_predictions(
        train, test, folds, sales
    )
    sales_oof, sales_test = catboost_predictions(
        train,
        test,
        folds,
        sales,
        sales,
        reconstruct_with_mrp=False,
        seed_offset=100,
        feature_columns=SALES_FEATURES,
    )
    units_oof, units_test = catboost_predictions(
        train,
        test,
        folds,
        units,
        sales,
        reconstruct_with_mrp=True,
        seed_offset=0,
        feature_columns=FEATURES,
    )

    # Rounded, stable weights chosen from leakage-free OOF predictions.
    blended_oof = 0.60 * outlet_oof + 0.25 * sales_oof + 0.15 * units_oof
    blended_test = 0.60 * outlet_test + 0.25 * sales_test + 0.15 * units_test

    print("Outlet-MRP OOF RMSE:", f"{root_mean_squared_error(sales, outlet_oof):.4f}")
    print("CatBoost sales OOF RMSE:", f"{root_mean_squared_error(sales, sales_oof):.4f}")
    print("CatBoost units OOF RMSE:", f"{root_mean_squared_error(sales, units_oof):.4f}")
    print("Final blended OOF RMSE:", f"{root_mean_squared_error(sales, blended_oof):.4f}")

    output_path = Path(args.output).resolve()
    baseline_path = Path(args.baseline_output).resolve()
    write_submission(test_raw, blended_test, output_path)
    write_submission(test_raw, outlet_test, baseline_path)
    print("Recommended submission:", output_path)
    print("Conservative fallback:", baseline_path)


if __name__ == "__main__":
    main()
