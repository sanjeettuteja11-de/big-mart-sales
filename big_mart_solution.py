"""Big Mart Sales III: single-file solution (CV RMSE 1071.33).

Every sale in the data is exactly whole units x (Item_MRP + an offset of up to
±2 on a 0.1 grid). The offset is noise. Units depend on the store and, weakly,
on the product, but not on price, category, fat content or visibility. So each
sale is predicted as

    Item_MRP x store rate x product popularity

A store's rate is its mean units per row (sales / MRP). A product's popularity
is how far it beat its stores' rates, shrunk toward 1 by SMOOTHING pseudo-rows.

The test predictions are the average of 15 models (5 folds x 3 repeats,
stratified by store), exactly as in the full pipeline:
https://github.com/sanjeettuteja11-de/big-mart-sales

Needs numpy, pandas and scikit-learn.

    python big_mart_solution.py --train train_v9rqX0R.csv --test test_AbJTz2l.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

SMOOTHING = 55.234948951954145  # tuned with Optuna on the same 5x3 folds
SEED, N_SPLITS, N_REPEATS = 42, 5, 3


def fit(train: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    units = train["Item_Outlet_Sales"] / train["Item_MRP"]
    rates = units.groupby(train["Outlet_Identifier"]).mean()
    relative = units / train["Outlet_Identifier"].map(rates)
    stats = relative.groupby(train["Item_Identifier"]).agg(["sum", "count"])
    popularity = (stats["sum"] + SMOOTHING) / (stats["count"] + SMOOTHING)
    return rates, popularity, float(units.mean())


def predict(model: tuple[pd.Series, pd.Series, float], rows: pd.DataFrame) -> np.ndarray:
    rates, popularity, default_rate = model
    rate = rows["Outlet_Identifier"].map(rates).fillna(default_rate)
    product = rows["Item_Identifier"].map(popularity).fillna(1.0)
    return np.clip((rows["Item_MRP"] * rate * product).to_numpy(dtype=float), 0, None)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=here / "data" / "raw" / "train_v9rqX0R.csv")
    parser.add_argument("--test", type=Path, default=here / "data" / "raw" / "test_AbJTz2l.csv")
    parser.add_argument("--out", type=Path, default=here / "submissions" / "big_mart_solution.csv")
    args = parser.parse_args()

    train, test = pd.read_csv(args.train), pd.read_csv(args.test)
    y = train["Item_Outlet_Sales"].to_numpy()

    oof_sum, oof_count = np.zeros(len(train)), np.zeros(len(train))
    test_sum = np.zeros(len(test))
    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    for fit_idx, val_idx in splitter.split(train, train["Outlet_Identifier"]):
        model = fit(train.iloc[fit_idx])
        oof_sum[val_idx] += predict(model, train.iloc[val_idx])
        oof_count[val_idx] += 1
        test_sum += predict(model, test)

    oof = oof_sum / oof_count
    print(f"CV RMSE: {np.sqrt(np.mean((y - oof) ** 2)):.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    test[["Item_Identifier", "Outlet_Identifier"]].assign(
        Item_Outlet_Sales=test_sum / (N_SPLITS * N_REPEATS)
    ).to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
