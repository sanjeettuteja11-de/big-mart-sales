"""Load the Analytics Vidhya Big Mart files and fix the known data-quality issues.

Cleaning looks at train and test together. That is safe: it only uses product
and store columns, never sales, and it lets a weight recorded for a product in
one store fill the gap for the same product in another store.
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


def clean(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cleaned copies of train and test, keeping row order."""
    df = pd.concat(
        [train.assign(_is_train=True), test.assign(_is_train=False)], ignore_index=True
    )

    # "LF", "low fat" and "Low Fat" are one label spelled three ways.
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(FAT_CONTENT_FIXES)
    # Codes starting "NC" are soaps, detergents and the like: they have no fat content.
    df.loc[df[ITEM_ID].str.startswith("NC"), "Item_Fat_Content"] = "Non-Edible"

    # A product weighs the same in every store, so borrow its weight from the
    # stores that recorded it (median, not mean: the mean of identical floats
    # can drift in the last bit). Category median covers products never weighed.
    df["Item_Weight"] = df["Item_Weight"].fillna(
        df.groupby(ITEM_ID)["Item_Weight"].transform("median")
    )
    df["Item_Weight"] = df["Item_Weight"].fillna(
        df.groupby("Item_Type")["Item_Weight"].transform("median")
    )

    # A product that sold was on a shelf, so zero visibility means "not recorded".
    visibility = df["Item_Visibility"].replace(0, np.nan)
    visibility = visibility.fillna(visibility.groupby(df[ITEM_ID]).transform("mean"))
    df["Item_Visibility"] = visibility.fillna(visibility.median())

    # Outlet_Size is blank for entire stores. Use the most common size among
    # stores of the same type, counting each store once rather than each row.
    stores = df.drop_duplicates(OUTLET_ID).dropna(subset=["Outlet_Size"])
    size_by_type = stores.groupby("Outlet_Type")["Outlet_Size"].agg(lambda s: s.mode().iloc[0])
    df["Outlet_Size"] = (
        df["Outlet_Size"].fillna(df["Outlet_Type"].map(size_by_type)).fillna("Unknown")
    )

    still_missing = df[FEATURE_COLUMNS].isna().sum()
    if still_missing.any():
        raise ValueError(f"Cleaning left gaps: {still_missing[still_missing > 0].to_dict()}")

    train_out = df[df["_is_train"]].drop(columns="_is_train").reset_index(drop=True)
    test_out = (
        df[~df["_is_train"]].drop(columns=["_is_train", TARGET]).reset_index(drop=True)
    )
    return train_out, test_out
