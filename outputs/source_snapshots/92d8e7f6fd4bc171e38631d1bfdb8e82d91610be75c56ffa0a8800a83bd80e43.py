"""Load the Analytics Vidhya Big Mart files and fix the known data-quality issues.

Two kinds of cleaning happen here:

- Row-wise fixes need no statistics: the fat-content spellings, "Non-Edible"
  for non-consumables, and indicator columns recording what was missing.
- Learned fixes need statistics from other rows: a product's weight from the
  stores that recorded it, a product's visibility elsewhere, the usual size
  of a store type. `Cleaner.fit` learns these from whichever rows it is
  given, never from sales, so it can be fit on a training fold only, or on
  train and test features together only with explicit opt-in.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_RAW, ITEM_ID, OUTLET_ID, TARGET

FEATURE_COLUMNS = [
    "Item_Identifier", "Item_Weight", "Item_Fat_Content", "Item_Visibility",
    "Item_Type", "Item_MRP", "Outlet_Identifier", "Outlet_Establishment_Year",
    "Outlet_Size", "Outlet_Location_Type", "Outlet_Type",
]
INDICATOR_COLUMNS = ["Item_Weight_Missing", "Outlet_Size_Missing", "Visibility_Was_Zero"]

FAT_CONTENT_FIXES = {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}


def _find_csv(raw_dir: Path, keyword: str) -> Path:
    matches = sorted(p for p in Path(raw_dir).glob("*.csv") if keyword in p.name.lower())
    if len(matches) != 1:
        found = ", ".join(p.name for p in matches) or "none"
        raise FileNotFoundError(
            f"Expected exactly one CSV with '{keyword}' in its name in {raw_dir} (found: {found}). "
            "Download train and test from the Analytics Vidhya Big Mart Sales III page into that folder."
        )
    return matches[0]


def load_raw(raw_dir: Path = DATA_RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(_find_csv(raw_dir, "train"))
    test = pd.read_csv(_find_csv(raw_dir, "test"))
    for name, df, expected in (
        ("train", train, FEATURE_COLUMNS + [TARGET]),
        ("test", test, FEATURE_COLUMNS),
    ):
        missing = sorted(set(expected) - set(df.columns))
        if missing:
            raise ValueError(f"{name} file is missing columns: {missing}")
    return train, test


def row_wise_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """Fixes that need no statistics from other rows."""
    df = df.copy()
    df["Item_Weight_Missing"] = df["Item_Weight"].isna().astype(int)
    df["Outlet_Size_Missing"] = df["Outlet_Size"].isna().astype(int)
    df["Visibility_Was_Zero"] = df["Item_Visibility"].eq(0).astype(int)
    # "LF", "low fat" and "Low Fat" are one label spelled three ways.
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(FAT_CONTENT_FIXES)
    # Codes starting "NC" are soaps, detergents and the like: they have no fat content.
    df.loc[df[ITEM_ID].str.startswith("NC"), "Item_Fat_Content"] = "Non-Edible"
    return df


class Cleaner:
    """Learned imputations. `fit` reads feature columns only, never sales."""

    def fit(self, frames: list[pd.DataFrame]) -> "Cleaner":
        df = pd.concat([row_wise_fixes(f[FEATURE_COLUMNS]) for f in frames], ignore_index=True)

        # A product weighs the same in every store, so borrow its weight from
        # the stores that recorded it (median, not mean: the mean of identical
        # floats can drift in the last bit). Category median covers the rest.
        self.item_weight_ = df.groupby(ITEM_ID)["Item_Weight"].median().dropna()
        self.type_weight_ = df.groupby("Item_Type")["Item_Weight"].median()
        self.global_weight_ = float(df["Item_Weight"].median())

        # Treat zero visibility as missing, a modeling assumption tested through
        # visibility/indicator ablations, not a verified operational explanation.
        visibility = df["Item_Visibility"].replace(0, np.nan)
        self.item_visibility_ = visibility.groupby(df[ITEM_ID]).mean().dropna()
        self.global_visibility_ = float(visibility.median())

        # Outlet_Size is blank for entire stores. Use the most common size among
        # stores of the same type, counting each store once rather than each row.
        stores = df.drop_duplicates(OUTLET_ID).dropna(subset=["Outlet_Size"])
        self.size_by_type_ = stores.groupby("Outlet_Type")["Outlet_Size"].agg(lambda s: s.mode().iloc[0])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = row_wise_fixes(df)
        out["Item_Weight"] = (
            out["Item_Weight"]
            .fillna(out[ITEM_ID].map(self.item_weight_))
            .fillna(out["Item_Type"].map(self.type_weight_))
            .fillna(self.global_weight_)
        )
        visibility = out["Item_Visibility"].replace(0, np.nan)
        out["Item_Visibility"] = (
            visibility.fillna(out[ITEM_ID].map(self.item_visibility_)).fillna(self.global_visibility_)
        )
        out["Outlet_Size"] = (
            out["Outlet_Size"].fillna(out["Outlet_Type"].map(self.size_by_type_)).fillna("Unknown")
        )
        still_missing = out[FEATURE_COLUMNS].isna().sum()
        if still_missing.any():
            raise ValueError(f"Cleaning left gaps: {still_missing[still_missing > 0].to_dict()}")
        return out


def clean(
    train: pd.DataFrame, test: pd.DataFrame, use_test_features: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cleaned copies of train and test, keeping row order.

    By default, statistics come from training features only. Production CV
    additionally fits these statistics within each training fold. Passing
    use_test_features=True is an explicit transductive experiment.
    """
    cleaner = Cleaner().fit([train, test] if use_test_features else [train])
    return cleaner.transform(train), cleaner.transform(test)
