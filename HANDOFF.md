# Handoff: Big Mart Sales III project

Paste this into a new chat to continue where the previous one stopped.

## Project

- Competition: https://www.analyticsvidhya.com/datahack/contest/practice-problem-big-mart-sales-iii/ (RMSE on `Item_Outlet_Sales`)
- Purpose: AMBA course project. Goal: lowest *legitimate* leaderboard RMSE with reproducible code and a defensible explanation.
- Code: `~/Library/Mobile Documents/com~apple~CloudDocs/Code/big-mart-sales` (its own git repo, branch `main`)
- GitHub: https://github.com/sanjeettuteja11-de/big-mart-sales (public; all work pushed, latest commit `23e18dc`)
- Python env: `~/.venvs/big-mart-sales` (kept outside iCloud on purpose). `./setup.sh` rebuilds it.
- Data: `data/raw/train_v9rqX0R.csv`, `data/raw/test_AbJTz2l.csv` (gitignored; download from the competition page after logging in)

## Where things stand

- Best leaderboard score: **1147.83** (rank ~560s; #1 is 1126.03). 12 submissions made, all logged in `reports/leaderboard_log.md`.
- Final model: `LatticeStoreRate`: decoded unit price × store rate × shrunk product factor. Strict CV 1070.85 (seed 42), 1070.95 mean over 3 seeds. Single-file version: `big_mart_solution.py`.
- Key finding: every sale is whole units × a unit price on a 0.6658 currency step; the unit price decodes from MRP alone. Units depend on the store format; nothing else in the data explains them.
- Everything on the 14-point competition checklist has been tried and documented in `reports/final_investigation.md`. Nothing beats the final model. The remaining gap to #1 is not reachable by honest modelling; the user was told this several times.

## Rules the user set and the assistant held

- No leaderboard probing, no reverse-engineering the scorer, no multi-account submissions. The user asked for this repeatedly; the assistant declined each time. Keep declining.
- Validation is independent of the leaderboard. Preprocessing is fitted inside each training fold. Tests enforce this (`tests/test_validation_integrity.py`).
- The user approves uploads to the leaderboard themselves; the assistant prepares checked files (`submissions/`, copies in `~/Downloads/big-mart-round*`).

## Commands

```bash
PY=~/.venvs/big-mart-sales/bin/python
$PY -m pytest                                  # 60 tests on synthetic data
$PY big_mart_solution.py --cv                  # final model: CV + submission CSV
$PY -m src.recommend                           # per-seed evidence + refit + checked CSV
$PY -m src.train                               # 5x3 CV for every model
$PY -m src.validate / src.ablate / src.report  # studies and the experiments table
$PY notebooks/build_eda_notebook.py --execute  # EDA notebook
```

## Files that matter for the report

- `README.md`: approach, results, final model, leaderboard table
- `reports/amba_summary.md`: the course write-up (business problem, cleaning, validation, model, limitations, implications)
- `reports/final_investigation.md`, `experiments.md`, `validation.md`, `ablations.md`, `eda_summary.md`, `leaderboard_log.md`
- `notebooks/01_eda.ipynb` (executed), `reports/figures/`

## Suggested next work

The modelling is finished. Remaining value is in the deliverable: the report, the presentation, and the defence (the currency-step discovery, the validation story, the business implications).
