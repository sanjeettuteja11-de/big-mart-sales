"""Project paths and constants shared by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
SUBMISSIONS = ROOT / "submissions"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

TARGET = "Item_Outlet_Sales"
ITEM_ID = "Item_Identifier"
OUTLET_ID = "Outlet_Identifier"

# The sales were recorded in 2013, so store age is measured against that year.
DATA_YEAR = 2013

SEED = 42
N_SPLITS = 5
# The target is very noisy, so a single 5-fold split can rank two close models
# the wrong way round. Three repeats average that out.
N_REPEATS = 3
