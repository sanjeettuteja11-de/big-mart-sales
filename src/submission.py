"""Checks every submission file must pass before upload."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ITEM_ID, OUTLET_ID, TARGET

COLUMNS = [ITEM_ID, OUTLET_ID, TARGET]


def check_submission(path: Path, test: pd.DataFrame, max_train_sales: float) -> list[str]:
    """Problems with the file at `path`; an empty list means it is valid.

    `test` is the raw test file, whose row order the submission must follow.
    """
    problems = []
    sub = pd.read_csv(path)
    if list(sub.columns) != COLUMNS:
        problems.append(f"columns are {list(sub.columns)}, expected exactly {COLUMNS} (no index column)")
    if len(sub) != len(test):
        problems.append(f"{len(sub)} rows, expected {len(test)}")
    elif {ITEM_ID, OUTLET_ID} <= set(sub.columns):
        mismatched = (sub[[ITEM_ID, OUTLET_ID]].to_numpy() != test[[ITEM_ID, OUTLET_ID]].to_numpy()).any(axis=1)
        if mismatched.any():
            problems.append(f"{int(mismatched.sum())} rows whose identifiers differ from the test file at the same position")
    if TARGET in sub.columns:
        pred = pd.to_numeric(sub[TARGET], errors="coerce")
        if pred.isna().any():
            problems.append(f"{int(pred.isna().sum())} missing or non-numeric predictions")
        finite = pred.dropna()
        if np.isinf(finite).any():
            problems.append("infinite predictions")
        if (finite < 0).any():
            problems.append(f"{int((finite < 0).sum())} negative predictions")
        if len(finite) and finite.max() > 1.5 * max_train_sales:
            problems.append(f"largest prediction {finite.max():,.0f} is far above the largest training sale {max_train_sales:,.0f}")
    return problems
