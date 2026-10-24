# honest_quantity — `honest_quantity_ridge.csv`

**Leaderboard RMSE: not recorded**

Cross-validation on training data: CV 1062.83 +/- 0.30 (5 folds x 5 seeds) - best CV in the project (not comparable with leaderboard scores; LB runs ~70-80 points higher).

| | |
|---|---|
| Phase | Phase 5 - clean rebuilds / later files |
| Approach | Decode exact unit price from MRP; predict units = fold-local outlet mean units + Ridge(alpha=3000) on spline(product number, 30 knots) + one-hot family/fat/Item_Type, outlet-scaled; sales = units x price. Selected on training CV only. |
| Uses leaderboard feedback? | No - chosen with training-data cross-validation only |
| Prediction file | [`honest_quantity_ridge.csv`](honest_quantity_ridge.csv) (5,681 rows) |
| Python | [`honest_quantity.py`](honest_quantity.py) |
| Note | BEST HONEST MODEL. Fully reproducible from the raw CSVs in ~4 seconds. Same CV folds, spline-only variant: 1061.41. |

## Run it

From the repository root, with the competition CSVs in `data/raw/`:

```bash
python all_solutions/honest_models/honest_quantity_ridge_CV-1062.83/honest_quantity.py --train data/raw/train_v9rqX0R.csv --test data/raw/test_AbJTz2l.csv --out submissions/honest_quantity_ridge.csv --cv
```

Prints the 5-fold x 5-seed CV (mean 1062.83, std 0.30) and writes the same file as the one in this folder.
