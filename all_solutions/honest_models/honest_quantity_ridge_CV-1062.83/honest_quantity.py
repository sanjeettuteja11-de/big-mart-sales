"""Standalone honest quantity model (no leaderboard values used at any stage).

Every training sale is a whole number of units times a unit price that lies on
the 0.6658 currency step, and the unit price decodes from MRP alone. This model
predicts the *units* directly:

    units = outlet mean units + ridge(spline(item id number), family, fat, Item_Type) * scale
    sales = units * decoded unit price

The per-outlet mean carries the store-format effect; the ridge term carries the
product effect, as a scale-free deviation weighted by (outlet scale)^2 so that
high-volume outlets do not dominate the fit.

Selected purely on fold-local cross-validation: n_knots=30, alpha=3000
(CV 1062.83 +/- 0.30 over seeds 42/137/2026/7/99, versus 1071.42 for the
store-rate x popularity model in big_mart_solution.py).
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

TARGET = "Item_Outlet_Sales"
KEYS = ["Item_Identifier", "Outlet_Identifier"]
CATEGORICAL = ["family", "fat", "Item_Type"]
N_KNOTS, ALPHA, SCALE_UNIT = 30, 3000, 16.0
SEEDS = (42, 137, 2026, 7, 99)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def currency_step(sales) -> float:
    """Greatest common divisor of the sales, in units of 0.0001."""
    return float(np.gcd.reduce(np.rint(np.asarray(sales, float) * 10000).astype(np.int64))) / 10000


def unit_price(mrp, step: float) -> np.ndarray:
    """The one candidate price within +/-2.0 of MRP that lies on the currency step."""
    mrp = np.asarray(mrp, float)
    candidates = mrp[:, None] + (np.arange(-20, 21) / 10)[None, :]
    on_step = np.isclose(candidates / step, np.round(candidates / step), rtol=0, atol=1e-7)
    if not np.all(on_step.sum(axis=1) == 1):
        raise ValueError("unit price is not uniquely identified for every row")
    return np.rint(candidates[np.arange(len(mrp)), on_step.argmax(axis=1)] / step) * step


def features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[["Item_Type"]].copy()
    x["id_number"] = frame.Item_Identifier.str[3:].astype(int)
    x["family"] = frame.Item_Identifier.str[:2]
    x["fat"] = frame.Item_Fat_Content.replace(
        {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"})
    x.loc[x.family.eq("NC"), "fat"] = "Non-Edible"
    return x


def _pipeline():
    transform = ColumnTransformer([
        ("id", SplineTransformer(n_knots=N_KNOTS, degree=3, include_bias=False), ["id_number"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return make_pipeline(transform, StandardScaler(), Ridge(alpha=ALPHA))


def fit_predict(x_fit, y_fit, price_fit, outlets_fit, x_apply, price_apply, outlets_apply):
    """Fit on the given rows only and predict sales for the applied rows."""
    units = y_fit / price_fit
    rates = pd.Series(units).groupby(outlets_fit).mean()
    mean_fit = pd.Series(outlets_fit).map(rates).to_numpy(float)
    mean_apply = pd.Series(outlets_apply).map(rates).fillna(units.mean()).to_numpy(float)
    scale_fit, scale_apply = mean_fit / SCALE_UNIT, mean_apply / SCALE_UNIT
    model = _pipeline()
    model.fit(x_fit, (units - mean_fit) / scale_fit, ridge__sample_weight=scale_fit ** 2)
    predicted_units = mean_apply + model.predict(x_apply) * scale_apply
    return np.maximum(0.0, predicted_units * price_apply)


def cross_validate(train: pd.DataFrame, step: float) -> dict:
    price = unit_price(train.Item_MRP, step)
    y = train[TARGET].to_numpy(float)
    outlets = train.Outlet_Identifier.to_numpy()
    x = features(train)
    per_seed = []
    for seed in SEEDS:
        oof = np.zeros(len(train))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, outlets):
            oof[va] = fit_predict(x.iloc[tr], y[tr], price[tr], outlets[tr],
                                  x.iloc[va], price[va], outlets[va])
        per_seed.append(rmse(y, oof))
    return {"per_seed": per_seed, "mean": float(np.mean(per_seed)), "std": float(np.std(per_seed))}


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--train", type=Path, default=root / "data/raw/train_v9rqX0R.csv")
    parser.add_argument("--test", type=Path, default=root / "data/raw/test_AbJTz2l.csv")
    parser.add_argument("--out", type=Path, default=root / "submissions/honest_quantity_ridge.csv")
    parser.add_argument("--cv", action="store_true", help="also print cross-validated RMSE")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    step = currency_step(train[TARGET])
    print(f"currency step: {step}")

    if args.cv:
        cv = cross_validate(train, step)
        print("cross-validated RMSE (5 folds x 5 seeds, fold-local fits):")
        for seed, value in zip(SEEDS, cv["per_seed"]):
            print(f"  seed {seed:5d}  {value:8.2f}")
        print(f"  mean {cv['mean']:.2f}  std {cv['std']:.2f}")

    price_train = unit_price(train.Item_MRP, step)
    price_test = unit_price(test.Item_MRP, step)
    units = train[TARGET].to_numpy(float) / price_train
    if abs(units - np.rint(units)).max() > 1e-8:
        raise ValueError("training sales are not whole units of the decoded price")

    predictions = fit_predict(
        features(train), train[TARGET].to_numpy(float), price_train,
        train.Outlet_Identifier.to_numpy(),
        features(test), price_test, test.Outlet_Identifier.to_numpy())

    out = test[KEYS].copy()
    out[TARGET] = predictions
    assert out.shape == (len(test), 3) and np.isfinite(predictions).all() and (predictions >= 0).all()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(f"  rows {len(out)}  min {predictions.min():.2f}  max {predictions.max():.2f} "
          f"mean {predictions.mean():.2f}")


if __name__ == "__main__":
    main()
