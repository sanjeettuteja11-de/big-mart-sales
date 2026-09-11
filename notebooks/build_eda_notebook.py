"""Write notebooks/01_eda.ipynb and, with --execute, run it so the outputs are saved.

    python notebooks/build_eda_notebook.py --execute

The notebook is generated from code so it stays in step with src/. It reads
data/raw/ through the same functions the pipeline uses.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent

CELLS = [
    ("md", """# Big Mart Sales III: exploratory analysis

Sales of 1,559 products across 10 BigMart stores in 2013. The task is to predict
`Item_Outlet_Sales` for product-store pairs in the test file; the leaderboard
scores RMSE on the sales scale.

This notebook walks through the raw files, the data-quality issues and how they
are treated, and the structure that the modelling relies on. Every function
comes from `src/`, so the pipeline and the notebook cannot disagree."""),
    ("code", """import sys; sys.path.insert(0, "..")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from src.data import load_raw, clean, FEATURE_COLUMNS
from src.features import build_features, MRP_BAND_EDGES
from src.structure import recover_units
sns.set_theme(style="whitegrid"); pd.set_option("display.width", 140)

train_raw, test_raw = load_raw()
print(train_raw.shape, test_raw.shape)
train_raw.head()"""),
    ("md", """## 1. What is in the files

Each row is one product in one store. The test file has the same columns
without `Item_Outlet_Sales`."""),
    ("code", """print("train columns:", list(train_raw.columns))
print("test columns: ", list(test_raw.columns))
print()
print("products: train", train_raw.Item_Identifier.nunique(), "| test", test_raw.Item_Identifier.nunique(),
      "| test products unseen in train:", len(set(test_raw.Item_Identifier) - set(train_raw.Item_Identifier)))
print("stores:", sorted(train_raw.Outlet_Identifier.unique()))
both = pd.concat([train_raw, test_raw])
print("duplicate product-store pairs across train+test:", both.duplicated(["Item_Identifier", "Outlet_Identifier"]).sum())
print("rows per product (train):", train_raw.Item_Identifier.value_counts().describe()[["mean", "min", "max"]].round(2).to_dict())"""),
    ("code", """train_raw.describe(include="all").T"""),
    ("md", """## 2. Data-quality issues

Four things are wrong or odd in the raw data. None of them is an error to
delete; each is handled in `src/data.py` and the choice is tested by
validation (`reports/validation.md`, `reports/ablations.md`)."""),
    ("code", """print("Missing values (%):")
print(both[FEATURE_COLUMNS].isna().mean().mul(100).round(1)[lambda s: s > 0])
print()
print("Outlet_Size missing in stores:", sorted(both.loc[both.Outlet_Size.isna(), "Outlet_Identifier"].unique()))
print("Item_Weight missing in stores:", sorted(both.loc[both.Item_Weight.isna(), "Outlet_Identifier"].unique()))
print("Item_Visibility == 0 share:", f"{(both.Item_Visibility == 0).mean():.1%}")
print()
print("Item_Fat_Content spellings:", both.Item_Fat_Content.value_counts().to_dict())"""),
    ("md", """**Treatments:**

- *Fat content*: `LF`, `low fat` → `Low Fat`; `reg` → `Regular`. Products whose code starts `NC` (household, health & hygiene) get `Non-Edible`.
- *Missing weight*: a product weighs the same in every store, so the weight is borrowed from the stores that recorded it. Two stores (OUT019, OUT027) recorded no weights at all.
- *Zero visibility*: a product that sold was on a shelf, so 0 means "not recorded". It is replaced by the product's mean visibility elsewhere, and a `Visibility_Was_Zero` flag is kept.
- *Missing store size*: three whole stores have no size. They get the most common size of stores of the same type, plus an `Outlet_Size_Missing` flag.

Imputation statistics come from product and store columns only, never from sales."""),
    ("code", """train, test = build_features(*clean(train_raw, test_raw))
train[["Item_Identifier", "Item_Weight", "Item_Weight_Missing", "Item_Fat_Content", "Item_Visibility",
       "Visibility_Was_Zero", "Outlet_Size", "Outlet_Size_Missing", "Outlet_Age", "Item_Category", "MRP_Band"]].head(8)"""),
    ("md", """## 3. The target"""),
    ("code", """y = train["Item_Outlet_Sales"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(y, bins=60, ax=axes[0]); axes[0].set(title="Item_Outlet_Sales", xlabel="Sales")
sns.histplot(np.log1p(y), bins=60, ax=axes[1]); axes[1].set(title="log1p(sales)", xlabel="log1p(Sales)")
plt.tight_layout()
print(y.describe().round(1).to_dict(), "| skew", round(y.skew(), 2))"""),
    ("md", """The target is right-skewed, which usually invites a log transform. The competition metric is RMSE on the raw scale, though, so any transform has to earn its place there (it does not; see `reports/ablations.md`)."""),
    ("md", """## 4. Price sits in four bands"""),
    ("code", """mrp = pd.concat([train.Item_MRP, test.Item_MRP])
fig, ax = plt.subplots(figsize=(10, 3.5))
sns.histplot(mrp, bins=150, ax=ax)
for edge in MRP_BAND_EDGES[1:-1]: ax.axvline(edge, color="crimson", ls="--", lw=1)
ax.set(title="Item_MRP with the band edges used for MRP_Band", xlabel="MRP")
v = np.sort(mrp.to_numpy()); gaps = np.diff(v)
print("largest gaps between consecutive prices:", [(round(v[i], 2), round(v[i+1], 2)) for i in np.argsort(gaps)[::-1][:3]])"""),
    ("md", """## 5. What drives sales: the store, through price"""),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=train, x="Item_MRP", y="Item_Outlet_Sales", hue="Outlet_Type", s=8, alpha=0.5, ax=axes[0])
axes[0].set(title="Sales vs price, by store type")
order = train.groupby("Outlet_Identifier").Item_Outlet_Sales.median().sort_values().index
sns.boxplot(data=train, x="Outlet_Identifier", y="Item_Outlet_Sales", hue="Outlet_Type", order=order, dodge=False, fliersize=1, ax=axes[1])
axes[1].tick_params(axis="x", rotation=45); axes[1].set(title="Sales by store", xlabel="")
plt.tight_layout()
print("correlation with sales:", train[["Item_MRP", "Item_Visibility", "Item_Weight", "Outlet_Age"]].corrwith(y).round(3).to_dict())"""),
    ("code", """print("mean sales by product category (range):", train.groupby("Item_Type").Item_Outlet_Sales.mean().agg(["min", "max"]).round(0).to_dict())
print("mean sales by fat content:", train.groupby("Item_Fat_Content").Item_Outlet_Sales.mean().round(0).to_dict())
print("mean sales by store type:", train.groupby("Outlet_Type").Item_Outlet_Sales.mean().round(0).to_dict())"""),
    ("md", """Price and store type dominate. Category and fat content barely move the mean.

## 6. The hidden structure: whole units × price

Every training sale splits exactly as `units × (Item_MRP + offset)`, with
`units` a whole number and `offset` a multiple of 0.1 between −2 and +2
(`src/structure.py`)."""),
    ("code", """units, offset = recover_units(train.Item_Outlet_Sales, train.Item_MRP)
found = ~np.isnan(units)
print(f"rows that split exactly: {found.mean():.1%}")
example = train.iloc[1]
print(f"example: {example.Item_Outlet_Sales} = {units[1]:.0f} x ({example.Item_MRP} + {offset[1]:+.1f})")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(offset[found], binwidth=0.1, binrange=(-2.05, 2.05), ax=axes[0]); axes[0].set(title="Unit price minus MRP", xlabel="offset")
sns.boxplot(x=train.Outlet_Type, y=units, fliersize=1, ax=axes[1]); axes[1].set(title="Units per row by store type", xlabel="", ylabel="units")
axes[1].tick_params(axis="x", rotation=15); plt.tight_layout()
df = train[found].assign(units=units[found], offset=offset[found])
df["relative"] = df.units / df.groupby("Outlet_Identifier").units.transform("mean")
print("corr(units, MRP):", round(np.corrcoef(df.units, df.Item_MRP)[0, 1], 3), "| corr(offset, MRP):", round(np.corrcoef(df.offset, df.Item_MRP)[0, 1], 3))
print("mean units by store type:", df.groupby("Outlet_Type").units.mean().round(1).to_dict())
print("units relative to store mean, by category (range):", df.groupby("Item_Type").relative.mean().agg(["min", "max"]).round(3).to_dict())"""),
    ("md", """The offset is centred on zero and unrelated to anything; it behaves as noise.
Units depend on the store format and not on price. Within a store, product
attributes barely move units. So the best prediction is
**price × the store's expected units**, with a small, heavily shrunk product
adjustment. That is the `StoreRatePopularity` model in `src/structure.py`, and
`reports/experiments.md` shows how it compares with gradient boosting.

Two cautions for the write-up: the association between store format and units
is descriptive, not causal (stores were not randomised into formats), and the
train-to-leaderboard gap (`reports/leaderboard_log.md`) shows the test sales
are noisier than the training sales for reasons the features cannot explain."""),
]


def build() -> nbformat.NotebookNode:
    nb = new_notebook(metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    nb.cells = [new_markdown_cell(src) if kind == "md" else new_code_cell(src) for kind, src in CELLS]
    return nb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the notebook and save the outputs")
    args = parser.parse_args()
    path = HERE / "01_eda.ipynb"
    nbformat.write(build(), path)
    print(f"wrote {path}")
    if args.execute:
        subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--inplace", "--ExecutePreprocessor.timeout=600", str(path)], check=True, cwd=HERE)
        print("executed")


if __name__ == "__main__":
    main()
