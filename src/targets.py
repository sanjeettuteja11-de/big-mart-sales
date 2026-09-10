"""Target transforms: train on an easier target, then convert back to sales.

The leaderboard scores RMSE on raw sales, so every transform is judged only
after its predictions are converted back.

`units` matters most. Sales are roughly price x units sold, and trees are good
at splitting on price but poor at multiplying by it. Training on units with
sample weight MRP^2 minimises exactly the same squared error as raw sales,
because (sales - p)^2 = MRP^2 * (units - p / MRP)^2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetTransform:
    name: str
    forward: Callable[[np.ndarray, pd.DataFrame], np.ndarray]
    inverse: Callable[[np.ndarray, pd.DataFrame], np.ndarray]
    weight: Callable[[pd.DataFrame], np.ndarray | None]


def _mrp(rows: pd.DataFrame) -> np.ndarray:
    return rows["Item_MRP"].to_numpy(dtype=float)


def _no_weight(rows: pd.DataFrame) -> None:
    return None


def _mrp_squared(rows: pd.DataFrame) -> np.ndarray:
    w = _mrp(rows) ** 2
    return w / w.mean()  # mean 1 keeps min_child_weight-style limits meaningful


TRANSFORMS = {
    "raw": TargetTransform("raw", lambda y, r: y, lambda z, r: z, _no_weight),
    "sqrt": TargetTransform(
        "sqrt", lambda y, r: np.sqrt(y), lambda z, r: np.square(np.clip(z, 0, None)), _no_weight
    ),
    "units": TargetTransform(
        "units", lambda y, r: y / _mrp(r), lambda z, r: z * _mrp(r), _mrp_squared
    ),
    "units_unweighted": TargetTransform(
        "units_unweighted", lambda y, r: y / _mrp(r), lambda z, r: z * _mrp(r), _no_weight
    ),
}
