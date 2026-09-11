"""Feature engineering and the model-ready matrices built from it.

Row-wise features (store age, product family, price band) need nothing but
the row. Learned features (a product's mean visibility, how many stores carry
it, the category's median price) come from `FeatureBuilder.fit`, which reads
feature columns only. `item_popularity` is the one feature that uses sales,
so it is only ever computed inside a cross-validation fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.config import DATA_YEAR, ITEM_ID, OUTLET_ID, SEED, TARGET
from src.data import INDICATOR_COLUMNS

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
INDICATORS = list(INDICATOR_COLUMNS)

# Named groups for ablations: dropping a group removes these columns.
FEATURE_GROUPS = {
    "product_family": ["Item_Category"],
    "store_age": ["Outlet_Age"],
    "indicators": INDICATORS,
    "weight": ["Item_Weight", "MRP_per_Weight"],
    "visibility": ["Item_Visibility", "Visibility_vs_Item_Mean"],
    "aggregates": ["Item_Store_Count", "MRP_vs_Type_Median"],
    "mrp_band": ["MRP_Band"],
    "outlet_id": ["Outlet_Identifier"],
    "store_attributes": ["Outlet_Size", "Outlet_Location_Type", "Outlet_Type"],
    "item_attributes": ["Item_Fat_Content", "Item_Type"],
}


def row_wise_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Item_Category"] = df[ITEM_ID].str[:2].map(ITEM_CATEGORY).fillna("Other")
    df["Outlet_Age"] = DATA_YEAR - df["Outlet_Establishment_Year"]
    df["MRP_Band"] = pd.cut(df["Item_MRP"], MRP_BAND_EDGES, labels=False)
    df["MRP_per_Weight"] = df["Item_MRP"] / df["Item_Weight"]
    return df


class FeatureBuilder:
    """Learned, label-free aggregates. Expects cleaned frames."""

    def fit(self, frames: list[pd.DataFrame]) -> "FeatureBuilder":
        df = pd.concat(frames, ignore_index=True)
        self.item_visibility_ = df.groupby(ITEM_ID)["Item_Visibility"].mean()
        self.item_store_count_ = df.groupby(ITEM_ID)[OUTLET_ID].nunique()
        self.type_mrp_median_ = df.groupby("Item_Type")["Item_MRP"].median()
        self.global_mrp_median_ = float(df["Item_MRP"].median())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = row_wise_features(df)
        # Shelf share relative to what the same product gets in other stores.
        item_mean = out[ITEM_ID].map(self.item_visibility_).fillna(out["Item_Visibility"])
        out["Visibility_vs_Item_Mean"] = out["Item_Visibility"] / item_mean
        out["Item_Store_Count"] = out[ITEM_ID].map(self.item_store_count_).fillna(1).astype(int)
        type_median = out["Item_Type"].map(self.type_mrp_median_).fillna(self.global_mrp_median_)
        out["MRP_vs_Type_Median"] = out["Item_MRP"] / type_median
        return out


def build_features(
    train: pd.DataFrame, test: pd.DataFrame, use_test_features: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    builder = FeatureBuilder().fit([train, test] if use_test_features else [train])
    return builder.transform(train), builder.transform(test).drop(columns=TARGET, errors="ignore")


def _select(columns: list[str], drop: set[str] | None) -> list[str]:
    return [c for c in columns if not drop or c not in drop]


def tree_matrix(
    train: pd.DataFrame, others: list[pd.DataFrame], as_codes: bool = False, drop: set[str] | None = None
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Categoricals as pandas `category` dtype (LightGBM, XGBoost, HistGB) or
    integer codes (random forest, extra trees). Every frame shares one
    encoding, whose levels are the union of all frames' levels: a label-free
    vocabulary, needed so a model can score a frame it never trained on."""
    columns = _select(NUMERIC + INDICATORS + CATEGORICAL, drop)
    frames = [f[columns].copy() for f in [train, *others]]
    for col in _select(CATEGORICAL, drop):
        dtype = pd.CategoricalDtype(sorted(set().union(*(set(f[col]) for f in frames))))
        for f in frames:
            f[col] = f[col].astype(dtype)
            if as_codes:
                f[col] = f[col].cat.codes
    return frames[0], frames[1:]


def catboost_matrix(
    train: pd.DataFrame, others: list[pd.DataFrame], drop: set[str] | None = None
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """CatBoost encodes categoricals itself, including the 1,559 product codes."""
    columns = _select(NUMERIC + INDICATORS + CATEGORICAL + [ITEM_ID], drop)
    frames = [f[columns].copy() for f in [train, *others]]
    return frames[0], frames[1:]


def linear_matrix(
    train: pd.DataFrame, others: list[pd.DataFrame], log_link: bool = False, drop: set[str] | None = None
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """One-hot categoricals plus price x store interactions.

    Each store turns price into sales at its own rate, so a linear model needs a
    separate price slope per store. With `log_link` (Poisson GLM) the slopes are
    on log price, which makes the model multiplicative: sales = price^b * store effect.
    """
    both = pd.concat([train, *others], keys=range(1 + len(others)))
    price = np.log(both["Item_MRP"]) if log_link else both["Item_MRP"]

    cats = _select(CATEGORICAL + ["MRP_Band"], drop)
    dummies = pd.get_dummies(both[cats].astype(str), prefix=cats, dtype=float)
    numeric = both[_select([c for c in NUMERIC + INDICATORS if c not in ("Item_MRP", "MRP_Band")], drop)].astype(float)
    parts = [numeric, price.rename("Price"), dummies]
    if not drop or OUTLET_ID not in drop:
        outlets = dummies.filter(like=f"{OUTLET_ID}_")
        parts.append(outlets.mul(price, axis=0).add_prefix("Price_x_"))

    X = pd.concat(parts, axis=1)
    return X.loc[0], [X.loc[i] for i in range(1, 1 + len(others))]


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
