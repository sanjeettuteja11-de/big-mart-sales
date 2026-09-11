"""Big Mart Sales III: single-file version of the final model (leaderboard 1147.83).

Every training sale is a whole number of units times a unit price, and every
sale is a multiple of one currency step: the greatest common divisor of the
training sales (0.6658). For each product, exactly one price within ±2.0 of
its MRP, in steps of 0.1, lies on that step. That is its unit price. Each
sale is then predicted as

    unit price x store rate x product popularity

A store's rate is its mean units per row. A product's popularity is how far
its units beat its stores' rates, shrunk toward 1 by SMOOTHING pseudo-rows.
Everything is learned from the training file only.

Reproduces submissions/final_lattice_store_rate_refit.csv from
`python -m src.recommend`. Full pipeline and evidence:
https://github.com/sanjeettuteja11-de/big-mart-sales

    python big_mart_solution.py --train train_v9rqX0R.csv --test test_AbJTz2l.csv
    python big_mart_solution.py --cv     # also print 5-fold x 3 cross-validated RMSE

Needs numpy and pandas (plus scikit-learn for --cv).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SMOOTHING = 55.234948951954145  # tuned by cross-validation
OFFSETS = np.arange(-20, 21) / 10  # candidate unit price = MRP + offset


def currency_step(sales) -> float:
    return float(np.gcd.reduce(np.rint(np.asarray(sales, float) * 10000).astype(np.int64))) / 10000


def unit_price(mrp, step: float) -> np.ndarray:
    """The one candidate price on the currency step; MRP if there isn't exactly one."""
    mrp = np.asarray(mrp, float)
    if not 0.1 <= step <= 5.0:
        return mrp
    candidates = mrp[:, None] + OFFSETS[None, :]
    on_step = np.isclose(candidates / step, np.round(candidates / step), rtol=0, atol=1e-7)
    unique = on_step.sum(axis=1) == 1
    return np.where(unique, candidates[np.arange(len(mrp)), on_step.argmax(axis=1)], mrp)


def fit(train: pd.DataFrame) -> tuple[float, pd.Series, pd.Series, float]:
    step = currency_step(train["Item_Outlet_Sales"])
    units = train["Item_Outlet_Sales"].to_numpy(float) / unit_price(train["Item_MRP"], step)
    stores = train["Outlet_Identifier"].to_numpy()
    rates = pd.Series(units).groupby(stores).mean()
    relative = units / pd.Series(stores).map(rates).to_numpy()
    stats = pd.Series(relative).groupby(train["Item_Identifier"].to_numpy()).agg(["sum", "count"])
    popularity = (stats["sum"] + SMOOTHING) / (stats["count"] + SMOOTHING)
    return step, rates, popularity, float(units.mean())


def predict(model: tuple[float, pd.Series, pd.Series, float], rows: pd.DataFrame) -> np.ndarray:
    step, rates, popularity, default_rate = model
    price = unit_price(rows["Item_MRP"], step)
    rate = rows["Outlet_Identifier"].map(rates).fillna(default_rate).to_numpy(float)
    product = rows["Item_Identifier"].map(popularity).fillna(1.0).to_numpy(float)
    return np.clip(price * (rate * product), 0, None)


def cross_validated_rmse(train: pd.DataFrame, seed: int = 42) -> float:
    from sklearn.model_selection import RepeatedStratifiedKFold

    y = train["Item_Outlet_Sales"].to_numpy(float)
    oof_sum, oof_count = np.zeros(len(train)), np.zeros(len(train))
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    for fit_idx, val_idx in splitter.split(train, train["Outlet_Identifier"]):
        oof_sum[val_idx] += predict(fit(train.iloc[fit_idx]), train.iloc[val_idx])
        oof_count[val_idx] += 1
    return float(np.sqrt(np.mean((y - oof_sum / oof_count) ** 2)))


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=here / "data" / "raw" / "train_v9rqX0R.csv")
    parser.add_argument("--test", type=Path, default=here / "data" / "raw" / "test_AbJTz2l.csv")
    parser.add_argument("--out", type=Path, default=here / "submissions" / "big_mart_solution.csv")
    parser.add_argument("--cv", action="store_true", help="also print cross-validated RMSE")
    args = parser.parse_args()

    train, test = pd.read_csv(args.train), pd.read_csv(args.test)
    if args.cv:
        print(f"CV RMSE (5-fold x 3, stratified by store, seed 42): {cross_validated_rmse(train):.2f}")

    model = fit(train)
    print(f"Currency step learned from training sales: {model[0]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    test[["Item_Identifier", "Outlet_Identifier"]].assign(Item_Outlet_Sales=predict(model, test)).to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
