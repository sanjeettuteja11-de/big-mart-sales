"""Exploratory analysis: figures and the numbers behind them, for the report.

    python -m src.eda

Writes PNGs to reports/figures/ and every computed finding to
reports/eda_summary.md. Figure titles describe what is plotted; the summary
states what the data actually shows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import DATA_RAW, FIGURES, OUTLET_ID, REPORTS, TARGET  # noqa: E402
from src.data import FEATURE_COLUMNS, clean, load_raw  # noqa: E402
from src.features import MRP_BAND_EDGES, build_features  # noqa: E402


def _save(fig: plt.Figure, figures: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(figures / f"{name}.png", dpi=150)
    plt.close(fig)


def run(train_raw: pd.DataFrame, test_raw: pd.DataFrame,
        figures: Path = FIGURES, reports: Path = REPORTS) -> list[str]:
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    findings: list[str] = []

    # Data quality, before cleaning.
    raw = pd.concat([train_raw, test_raw])
    missing = raw[FEATURE_COLUMNS].isna().mean().mul(100)
    findings.append("Missing values: " + ", ".join(
        f"{col} {pct:.1f}%" for col, pct in missing[missing > 0].items()))
    for col in ("Outlet_Size", "Item_Weight"):
        stores = sorted(raw.loc[raw[col].isna(), OUTLET_ID].unique())
        findings.append(f"{col} is blank only in stores: {', '.join(stores)}")
    findings.append(f"Item_Visibility is exactly 0 in {(raw['Item_Visibility'] == 0).mean():.1%} of rows")
    findings.append(f"Item_Fat_Content as recorded: {raw['Item_Fat_Content'].value_counts().to_dict()}")

    train, test = build_features(*clean(train_raw, test_raw))
    y = train[TARGET]
    units = y / train["Item_MRP"]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(y, bins=60, ax=ax)
    ax.set(title="Distribution of Item_Outlet_Sales", xlabel="Sales")
    _save(fig, figures, "01_sales_distribution")
    findings.append(f"Sales: mean {y.mean():,.0f}, median {y.median():,.0f}, "
                    f"std {y.std():,.0f}, max {y.max():,.0f}, skew {y.skew():.2f}")

    mrp = pd.concat([train["Item_MRP"], test["Item_MRP"]])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(mrp, bins=120, ax=ax)
    for edge in MRP_BAND_EDGES[1:-1]:
        ax.axvline(edge, color="crimson", linestyle="--", linewidth=1)
    ax.set(title="Item_MRP with the assumed price-band edges", xlabel="MRP")
    _save(fig, figures, "02_mrp_bands")
    near = {e: int((mrp - e).abs().lt(1).sum()) for e in MRP_BAND_EDGES[1:-1]}
    findings.append(f"Rows within ±1 of each assumed MRP band edge (should be ~0 if the gaps are real): {near}")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=train, x="Item_MRP", y=TARGET, hue="Outlet_Type", s=8, alpha=0.5, ax=ax)
    ax.set(title="Sales vs price, coloured by store type")
    _save(fig, figures, "03_sales_vs_mrp_by_outlet_type")
    findings.append(f"Correlation of sales with Item_MRP {np.corrcoef(train['Item_MRP'], y)[0, 1]:.2f}, "
                    f"with Item_Visibility {np.corrcoef(train['Item_Visibility'], y)[0, 1]:.2f}, "
                    f"with Item_Weight {np.corrcoef(train['Item_Weight'], y)[0, 1]:.2f}")

    order = train.groupby(OUTLET_ID)[TARGET].median().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=train, x=OUTLET_ID, y=TARGET, hue="Outlet_Type", order=order,
                dodge=False, fliersize=1, ax=ax)
    ax.tick_params(axis="x", rotation=45)
    ax.set(title="Sales per store, sorted by median", xlabel="")
    _save(fig, figures, "04_sales_by_store")
    by_store = train.groupby([OUTLET_ID, "Outlet_Type"])[TARGET].mean().sort_values()
    findings.append("Mean sales by store: " + ", ".join(
        f"{oid} ({otype}) {v:,.0f}" for (oid, otype), v in by_store.items()))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(x=train["Outlet_Type"], y=units, fliersize=1, ax=ax)
    ax.set(title="Units sold (sales / MRP) by store type", xlabel="", ylabel="Sales / MRP")
    _save(fig, figures, "05_units_by_store_type")
    findings.append("Median units (sales / MRP) by store type: " + ", ".join(
        f"{k} {v:.1f}" for k, v in units.groupby(train["Outlet_Type"]).median().sort_values().items()))

    by_type = train.groupby("Item_Type")[TARGET].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    by_type.plot.barh(ax=ax)
    ax.set(title="Mean sales by product category", xlabel="Mean sales", ylabel="")
    _save(fig, figures, "06_sales_by_item_type")
    findings.append(f"Product category spread in mean sales: {by_type.index[0]} {by_type.iloc[0]:,.0f} "
                    f"to {by_type.index[-1]} {by_type.iloc[-1]:,.0f}")

    numeric = train[["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Age",
                     "Visibility_vs_Item_Mean", "Item_Store_Count", TARGET]]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set(title="Correlation between numeric columns")
    _save(fig, figures, "07_correlation")

    reports.mkdir(parents=True, exist_ok=True)
    (reports / "eda_summary.md").write_text(
        "# EDA summary\n\nGenerated by `python -m src.eda`; figures are in `reports/figures/`.\n\n"
        + "\n".join(f"- {f}" for f in findings) + "\n"
    )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_RAW)
    args = parser.parse_args()
    for finding in run(*load_raw(args.data)):
        print(f"- {finding}")


if __name__ == "__main__":
    main()
