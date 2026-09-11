"""The hidden structure of the target, and the model built on it.

Every training row splits exactly as

    Item_Outlet_Sales = units x (Item_MRP + offset)

where units is a whole number and offset is a multiple of 0.1 between -2 and
+2. The offset behaves like random noise, unrelated to product, store or
price. Units depend on the store and, very weakly, on the product. Nothing
else (category, fat content, visibility, price band) moves them. So the best
prediction is Item_MRP x expected units, which is what `StoreRatePopularity`
estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin

from src.config import ITEM_ID, OUTLET_ID, TARGET


def recover_units(sales, mrp, max_offset: float = 2.0, step: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Split each sale into whole units and a unit-price offset from MRP.

    Returns (units, offset), NaN where no split exists. When several splits
    fit, the one with the smallest offset wins.
    """
    sales, mrp = np.asarray(sales, dtype=float), np.asarray(mrp, dtype=float)
    lo = np.maximum(1, np.floor(sales / (mrp + max_offset + step / 2)))
    hi = np.ceil(sales / np.maximum(mrp - max_offset - step / 2, 1e-9))
    width = int((hi - lo).max()) + 1
    ks = lo[:, None] + np.arange(width)[None, :]
    offsets = sales[:, None] / ks - mrp[:, None]
    on_grid = np.abs(offsets / step - np.round(offsets / step)) < 2e-3
    valid = (ks <= hi[:, None]) & (np.abs(offsets) <= max_offset + 1e-4) & on_grid
    score = np.where(valid, np.abs(offsets), np.inf)
    rows, best = np.arange(len(sales)), score.argmin(axis=1)
    found = np.isfinite(score[rows, best])
    units = np.where(found, ks[rows, best], np.nan)
    offset = np.where(found, np.round(offsets[rows, best] / step) * step, np.nan)
    return units, offset


def units_findings(train: pd.DataFrame) -> list[str]:
    """Plain-language checks of what drives units sold, for the EDA summary.

    Expects the output of `features.build_features`.
    """
    units, offset = recover_units(train[TARGET], train["Item_MRP"])
    found = ~np.isnan(units)
    df = train.loc[found].assign(units=units[found], offset=offset[found])
    df["relative"] = df["units"] / df.groupby(OUTLET_ID)["units"].transform("mean")

    findings = [
        f"{found.mean():.1%} of train rows are exactly whole units x (MRP + offset), with the offset "
        f"between {df.offset.min():+.1f} and {df.offset.max():+.1f} in steps of 0.1 (mean {df.offset.mean():+.3f})",
        f"Correlation of units with MRP {np.corrcoef(df.units, df.Item_MRP)[0, 1]:+.3f}; "
        f"of the price offset with MRP {np.corrcoef(df.offset, df.Item_MRP)[0, 1]:+.3f} "
        f"and with units {np.corrcoef(df.offset, df.units)[0, 1]:+.3f}",
        "Units per row by store type (mean / variance): " + ", ".join(
            f"{t} {r['mean']:.1f} / {r['var']:.1f}"
            for t, r in df.groupby("Outlet_Type")["units"].agg(["mean", "var"]).iterrows()),
    ]
    for col in ("Item_Type", "Item_Fat_Content", "MRP_Band"):
        groups = [g.to_numpy() for _, g in df.groupby(col)["relative"]]
        findings.append(f"Units relative to the store's average vs {col}: ANOVA p = {stats.f_oneway(*groups).pvalue:.2f}")
    within = np.mean([np.corrcoef(g.Item_Visibility, g.relative)[0, 1] for _, g in df.groupby(OUTLET_ID)])
    findings.append(f"Within-store correlation of units with visibility: {within:+.3f}")
    return findings


class StoreRatePopularity(RegressorMixin, BaseEstimator):
    """Units sold = store rate x product popularity.

    Each store sells at its own average rate; with rate_level="type", all
    stores of one type share a rate. A product that beat its stores' rates
    elsewhere gets a factor above 1, shrunk toward 1 by `smoothing`
    pseudo-rows, because the product effect is small next to the noise;
    smoothing=inf drops the product factor. Expects a units target
    (sales / MRP) and a frame with the store, store type and product codes.
    """

    RATE_COLUMNS = {"store": OUTLET_ID, "type": "Outlet_Type"}

    def __init__(self, smoothing: float = 50.0, rate_level: str = "store"):
        self.smoothing = smoothing
        self.rate_level = rate_level

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        y = np.asarray(y, dtype=float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        keys = X[self.RATE_COLUMNS[self.rate_level]].to_numpy()

        self.rates_ = pd.Series(y * w).groupby(keys).sum() / pd.Series(w).groupby(keys).sum()
        self.default_rate_ = float(np.average(y, weights=w))
        if np.isinf(self.smoothing):
            self.popularity_ = pd.Series(dtype=float)
            return self

        relative = y / pd.Series(keys).map(self.rates_).to_numpy()
        prior = self.smoothing * w.mean()
        sums = pd.DataFrame({"wr": relative * w, "w": w}).groupby(X[ITEM_ID].to_numpy()).sum()
        self.popularity_ = (sums["wr"] + prior) / (sums["w"] + prior)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        rate = X[self.RATE_COLUMNS[self.rate_level]].map(self.rates_).fillna(self.default_rate_)
        popularity = X[ITEM_ID].map(self.popularity_).fillna(1.0)
        return (rate * popularity).to_numpy(dtype=float)
