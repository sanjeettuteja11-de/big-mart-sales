"""Feature engineering and the model-ready matrices built from it.

Everything except `item_popularity` uses product and store columns only, so it
can be computed on train and test together. `item_popularity` uses sales and is
therefore only ever called inside a cross-validation fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.config import DATA_YEAR, ITEM_ID, OUTLET_ID, SEED, TARGET

# Item_MRP sits in four price bands with empty gaps between them (see EDA).
MRP_BAND_EDGES = [-np.inf, 69, 136, 203, np.inf]

ITEM_CATEGORY = {"FD": "Food", "DR": "Drinks", "NC": "Non-Consumable"}

CATEGORICAL = [
    "Item_Fat_Content", "Item_Type", "Item_Category", "Outlet_Identifier",
    "Outlet_Size", "Outlet_Location_Type", "Outlet_Type",
]
NUMERIC = [
    "Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Age", "MRP_Band",
    "Visibility_vs_Item_Mean", "Item_Store_Count", "MRP_per_Weight", "MRP_vs_Type_Median",
]


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.concat([train, test], keys=["train", "test"])

    df["Item_Category"] = df[ITEM_ID].str[:2].map(ITEM_CATEGORY).fillna("Other")
    df["Outlet_Age"] = DATA_YEAR - df["Outlet_Establishment_Year"]
    df["MRP_Band"] = pd.cut(df["Item_MRP"], MRP_BAND_EDGES, labels=False)
    # Shelf share relative to what the same product gets in other stores.
    df["Visibility_vs_Item_Mean"] = (
        df["Item_Visibility"] / df.groupby(ITEM_ID)["Item_Visibility"].transform("mean")
    )
    df["Item_Store_Count"] = df.groupby(ITEM_ID)[OUTLET_ID].transform("nunique")
    df["MRP_per_Weight"] = df["Item_MRP"] / df["Item_Weight"]
    df["MRP_vs_Type_Median"] = (
        df["Item_MRP"] / df.groupby("Item_Type")["Item_MRP"].transform("median")
    )

    return df.loc["train"].copy(), df.loc["test"].drop(columns=TARGET, errors="ignore").copy()


def tree_matrix(
    train: pd.DataFrame, test: pd.DataFrame, as_codes: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Categoricals as pandas `category` dtype (LightGBM, XGBoost, HistGB) or
    integer codes (random forest, extra trees). Categories come from train and
    test together so both sides share one encoding."""
    columns = NUMERIC + CATEGORICAL
    X_tr, X_te = train[columns].copy(), test[columns].copy()
    for col in CATEGORICAL:
        dtype = pd.CategoricalDtype(sorted(set(X_tr[col]) | set(X_te[col])))
        X_tr[col], X_te[col] = X_tr[col].astype(dtype), X_te[col].astype(dtype)
        if as_codes:
            X_tr[col], X_te[col] = X_tr[col].cat.codes, X_te[col].cat.codes
    return X_tr, X_te


def catboost_matrix(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CatBoost encodes categoricals itself, including the 1,559 product codes."""
    columns = NUMERIC + CATEGORICAL + [ITEM_ID]
    return train[columns].copy(), test[columns].copy()


def linear_matrix(
    train: pd.DataFrame, test: pd.DataFrame, log_link: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot categoricals plus price x store interactions.

    Each store turns price into sales at its own rate, so a linear model needs a
    separate price slope per store. With `log_link` (Poisson GLM) the slopes are
    on log price, which makes the model multiplicative: sales = price^b * store effect.
    """
    both = pd.concat([train, test], keys=["train", "test"])
    price = np.log(both["Item_MRP"]) if log_link else both["Item_MRP"]

    cats = CATEGORICAL + ["MRP_Band"]
    dummies = pd.get_dummies(both[cats].astype(str), prefix=cats, dtype=float)
    numeric = both[[c for c in NUMERIC if c not in ("Item_MRP", "MRP_Band")]].astype(float)
    outlets = dummies.filter(like=f"{OUTLET_ID}_")
    slopes = outlets.mul(price, axis=0).add_prefix("Price_x_")

    X = pd.concat([numeric, price.rename("Price"), dummies, slopes], axis=1)
    return X.loc["train"], X.loc["test"]


def item_popularity(
    train_rows: pd.DataFrame,
    y: np.ndarray,
    others: list[pd.DataFrame],
    n_splits: int = 5,
    smoothing: float = 5.0,
    seed: int = SEED,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """How well each product sells compared with its store's average.

    Sales become units (sales / MRP), then are divided by the store's mean
    units, so a popular product scores above 1 in any store. Each product's
    mean is shrunk toward 1 when it rests on few rows.

    Training rows get out-of-fold values, so no row sees its own sales. Each
    frame in `others` gets values fitted on all training rows.
    """

    def fit(rows: pd.DataFrame, target: np.ndarray) -> pd.Series:
        units = target / rows["Item_MRP"].to_numpy()
        store_mean = pd.Series(units).groupby(rows[OUTLET_ID].to_numpy()).transform("mean")
        relative = units / store_mean.to_numpy()
        stats = pd.Series(relative).groupby(rows[ITEM_ID].to_numpy()).agg(["sum", "count"])
        return (stats["sum"] + smoothing) / (stats["count"] + smoothing)

    def apply(mapping: pd.Series, rows: pd.DataFrame) -> np.ndarray:
        return rows[ITEM_ID].map(mapping).fillna(1.0).to_numpy(dtype=float)

    train_values = np.ones(len(train_rows))
    for fit_idx, enc_idx in KFold(n_splits, shuffle=True, random_state=seed).split(train_rows):
        mapping = fit(train_rows.iloc[fit_idx], y[fit_idx])
        train_values[enc_idx] = apply(mapping, train_rows.iloc[enc_idx])

    full = fit(train_rows, y)
    return train_values, [apply(full, rows) for rows in others]
