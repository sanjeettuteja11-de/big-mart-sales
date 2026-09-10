"""Synthetic data with the Big Mart schema and its data-quality quirks.

The real files need an Analytics Vidhya login, so tests run on this instead.
Sales follow the structure of the real data: price x a store-specific rate x
product popularity x heavy noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# id, opened, size, tier, type, units-per-rupee rate
OUTLETS = [
    ("OUT010", 1998, None, "Tier 3", "Grocery Store", 2.4),
    ("OUT013", 1987, "High", "Tier 3", "Supermarket Type1", 16.0),
    ("OUT017", 2007, None, "Tier 2", "Supermarket Type1", 16.0),
    ("OUT018", 2009, "Medium", "Tier 3", "Supermarket Type2", 14.0),
    ("OUT019", 1985, "Small", "Tier 1", "Grocery Store", 2.4),
    ("OUT027", 1985, "Medium", "Tier 3", "Supermarket Type3", 26.0),
    ("OUT035", 2004, "Small", "Tier 2", "Supermarket Type1", 16.0),
    ("OUT045", 2002, None, "Tier 2", "Supermarket Type1", 16.0),
    ("OUT046", 1997, "Small", "Tier 1", "Supermarket Type1", 16.0),
    ("OUT049", 1999, "Medium", "Tier 1", "Supermarket Type1", 16.0),
]
ITEM_TYPES = {
    "FD": ["Snack Foods", "Fruits and Vegetables", "Frozen Foods", "Canned", "Baking Goods",
           "Breads", "Meat", "Breakfast", "Seafood", "Starchy Foods", "Dairy"],
    "DR": ["Soft Drinks", "Hard Drinks", "Dairy"],
    "NC": ["Household", "Health and Hygiene", "Others"],
}
MRP_BANDS = [(31, 69), (69, 136), (136, 203), (203, 267)]


def make_synthetic(
    n_items: int = 200, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return (train, test, true test sales)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        prefix = rng.choice(["FD", "DR", "NC"], p=[0.72, 0.1, 0.18])
        item_id = f"{prefix}{chr(65 + i // 100 % 26)}{i % 100:02d}"
        lo, hi = MRP_BANDS[rng.integers(len(MRP_BANDS))]
        item = {
            "Item_Identifier": item_id,
            "Item_Weight": rng.uniform(4.5, 21.5),
            "Item_Fat_Content": rng.choice(["Low Fat", "Regular"], p=[0.65, 0.35]),
            "Item_Type": rng.choice(ITEM_TYPES[prefix]),
            "Item_MRP": rng.uniform(lo + 2, hi - 2),
        }
        popularity = rng.lognormal(0, 0.25)
        base_visibility = rng.uniform(0.01, 0.2)
        n_stores = rng.integers(5, 10)
        for outlet_idx in rng.choice(len(OUTLETS), size=n_stores, replace=False):
            oid, year, size, tier, otype, rate = OUTLETS[outlet_idx]
            rows.append({
                **item,
                "Item_Visibility": base_visibility * rng.uniform(0.8, 1.2),
                "Outlet_Identifier": oid,
                "Outlet_Establishment_Year": year,
                "Outlet_Size": size,
                "Outlet_Location_Type": tier,
                "Outlet_Type": otype,
                "Item_Outlet_Sales": item["Item_MRP"] * rate * popularity * rng.lognormal(-0.1, 0.45),
            })

    df = pd.DataFrame(rows)
    # Reproduce the real files' quirks.
    df.loc[rng.random(len(df)) < 0.06, "Item_Visibility"] = 0.0
    df.loc[df["Outlet_Identifier"].isin(["OUT019", "OUT027"]), "Item_Weight"] = np.nan
    low_fat = df["Item_Fat_Content"].eq("Low Fat")
    df.loc[low_fat & (rng.random(len(df)) < 0.1), "Item_Fat_Content"] = "LF"
    df.loc[low_fat & (rng.random(len(df)) < 0.05), "Item_Fat_Content"] = "low fat"
    df.loc[df["Item_Fat_Content"].eq("Regular") & (rng.random(len(df)) < 0.05), "Item_Fat_Content"] = "reg"

    is_train = rng.random(len(df)) < 0.6
    train = df[is_train].reset_index(drop=True)
    test = df[~is_train].reset_index(drop=True)
    truth = test.pop("Item_Outlet_Sales").to_numpy()
    return train, test, truth
