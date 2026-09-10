# Big Mart Sales Prediction

Predicting how much each product sells in each store, for the
[Analytics Vidhya Big Mart Sales III](https://www.analyticsvidhya.com/datahack/contest/practice-problem-big-mart-sales-iii/)
practice problem. Submissions are scored by RMSE on `Item_Outlet_Sales`.

## The problem

BigMart recorded 2013 sales for 1,559 products across 10 stores. For each
product-store pair we know the product (weight, fat content, shelf visibility,
category, price) and the store (size, city tier, type, opening year). The train
file has 8,523 rows with sales; the test file has 5,681 rows to predict.
Besides the prediction, the business question is **which product and store
properties drive sales**.

## Key finding: every sale is whole units × a price near MRP

Every one of the 8,523 training sales splits exactly as

```
Item_Outlet_Sales = units × (Item_MRP + offset)
```

where **units** is a whole number and **offset** is a multiple of 0.1 between
−2 and +2 ([src/structure.py](src/structure.py)). For example,
443.4228 = 9 × (48.2692 + 1.0).

What that shows ([reports/eda_summary.md](reports/eda_summary.md)):

- **The offset is noise.** Its mean is −0.015, and it is unrelated to store,
  category, price band, visibility or units.
- **Units don't depend on price** (correlation +0.01). Grocery stores sell
  about 2.4 units per product, Supermarket Type1 about 16, Type2 about 14 and
  Type3 about 27.
- **Within a store, units don't depend on the product's attributes.**
  Category, fat content, price band, visibility and weight all show no effect.
  Only the product's identity has a small effect, and it has to be shrunk hard
  to help.

So the best prediction is **MRP × the store's expected units × a heavily shrunk
product factor**. That is `StoreRatePopularity`, and it beats every
general-purpose model below. What's left is randomness in units that no column
explains: at a Type1 supermarket, units have a variance of about 50.

**Business reading:** price sets how much revenue each unit brings in, and the
store format sets how many units sell. Shelf visibility, category and fat
content don't change units sold. Revenue growth comes from store format
(supermarkets sell 6–11× more units per product than grocery stores) and from
product price.

## Approach

### 1. Cleaning ([src/data.py](src/data.py))

| Issue in the raw data | Fix |
|---|---|
| `Item_Fat_Content` spelled five ways (`LF`, `low fat`, `reg`, ...) | Mapped to `Low Fat` / `Regular`; non-consumables (codes starting `NC`) become `Non-Edible` |
| `Item_Weight` blank for some stores | A product weighs the same everywhere, so its weight is taken from other stores |
| `Item_Visibility` is 0 for some rows | A product that sold was on a shelf, so 0 means "not recorded"; replaced by the product's visibility elsewhere |
| `Outlet_Size` blank for three whole stores | Most common size among stores of the same type |

Cleaning looks at train and test together, but never at sales.

### 2. Features ([src/features.py](src/features.py))

- `Outlet_Age` (2013 minus opening year) and `Item_Category` (Food / Drinks / Non-Consumable, from the product code)
- `MRP_Band`: prices fall into four bands with gaps between them
- `Visibility_vs_Item_Mean`: shelf share compared with the same product in other stores
- `Item_Store_Count`, `MRP_per_Weight`, `MRP_vs_Type_Median`
- `Item_Popularity`: how well a product sells compared with its store's average. It uses sales, so it is computed inside each cross-validation fold, out of fold, and never sees the row it describes.

### 3. The key idea: predict units, not sales

Sales are roughly *price × units sold × a store-specific rate*. Tree models
are good at splitting on price but poor at multiplying by it. So the tree
models learn **units = sales / MRP** and their predictions are multiplied back
by price.

Training on units with sample weight MRP² minimises *exactly* the same squared
error as training on sales, because
`(sales − p)² = MRP² · (units − p / MRP)²`.
The target changes; the objective does not. This is checked by a test.

### 4. Models ([src/models.py](src/models.py))

| Model | Input | Target |
|---|---|---|
| **StoreRatePopularity**: MRP × store rate × shrunk product factor ([src/structure.py](src/structure.py)) | store and product codes | units |
| Ridge with a price slope per store | one-hot | sales |
| Poisson GLM (log link: sales = price^b × store effect) | one-hot | sales |
| HistGradientBoosting, Random Forest, Extra Trees | native / codes | units |
| LightGBM (with and without `Item_Popularity`) | native categoricals | sales, units |
| XGBoost | native categoricals | units |
| CatBoost (also encodes all 1,559 product codes itself) | raw strings | sales, units |

Defaults are shallow and heavily regularised, because the target is very
noisy. [src/tune.py](src/tune.py) refines them with Optuna.

### 5. Validation ([src/cv.py](src/cv.py))

Five folds × three repeats, stratified by store, all with the same seed so
every model is scored on identical splits. Each training row gets an
out-of-fold prediction averaged over the repeats. Test predictions are the
average of all 15 fold models.

### 6. Blending ([src/blend.py](src/blend.py))

The blend uses non-negative weights that sum to one, fitted on the
out-of-fold predictions. The score it reports is *honest*: weights are fitted
on some rows and scored on others, so the blend doesn't get credit for fitting
the rows it is scored on. `src/train.py` recommends whichever of the blend
and the best single model scores better.

## Results

Five folds × three repeats on the 8,523 training rows, RMSE on sales (lower
is better). Always predicting the mean scores 1,706.4. Full details are in
[outputs/cv_results.json](outputs/cv_results.json).

| Model | CV RMSE |
|---|---|
| **StoreRatePopularity** (tuned smoothing = 55) | **1,071.33** |
| Ridge, price slope per store | 1,075.10 |
| Poisson GLM | 1,075.98 |
| CatBoost (sales) | 1,078.64 |
| CatBoost (units) | 1,078.94 |
| Extra Trees (units) | 1,079.07 |
| Random Forest (units) | 1,079.19 |
| LightGBM (sales) | 1,083.87 |
| LightGBM (units + popularity) | 1,086.12 |
| LightGBM (units) | 1,087.27 |
| XGBoost (units) | 1,088.14 |
| HistGradientBoosting (units) | 1,091.81 |
| Blend of all, honest CV | 1,071.46 |

The structural model wins, and its whole fit takes about 0.1 seconds: 10 store
rates plus 1,559 shrunk product factors. Without the product factor (smoothing
→ ∞) it scores 1,071.95. The general-purpose models lose because their extra
flexibility mostly fits noise. The blend puts 96% of its weight on
StoreRatePopularity and scores no better, so the recommended submission is
`submissions/store_rate_popularity_cv1071.csv`.

### Leaderboard

`store_rate_popularity_cv1071.csv` scored **1148.28** on the Analytics Vidhya
leaderboard (rank 566 of all submissions on 2026-09-11; rank 1 was 1126.03,
rank 40 was 1138.81).

The leaderboard score is 77 points worse than the cross-validated one. That
is not a modelling problem: train and test features are indistinguishable
(adversarial validation AUC 0.495, i.e. chance), and a random subset of
training rows the size of the test set scores 1071 ± 9 with this model. The
scored test rows are simply noisier than the training rows, which hits every
competitor equally. Published write-ups of this problem report the same gap
(CV around 1080–1090, leaderboard around 1150–1155).

## How to run

```bash
./setup.sh
```

Download `train` and `test` from the competition page into `data/raw/`
(any filenames containing "train" and "test").

```bash
~/.venvs/big-mart-sales/bin/python -m src.eda                                        # figures + reports/eda_summary.md
~/.venvs/big-mart-sales/bin/python -m src.train                                      # CV every model, blend, write submissions/
~/.venvs/big-mart-sales/bin/python -m src.tune --model catboost_units --trials 50    # optional: tune, then re-run train
~/.venvs/big-mart-sales/bin/python -m pytest                                         # runs on synthetic data, no download needed
```

## Project layout

```
src/
  config.py      paths and constants
  data.py        load + clean
  features.py    feature engineering and model matrices
  targets.py     sales / sqrt / units target transforms
  models.py      the model zoo
  cv.py          repeated stratified CV, out-of-fold predictions
  blend.py       simplex-weighted blending with an honest score
  train.py       end-to-end: CV -> blend -> submission CSVs
  tune.py        Optuna hyperparameter search
  eda.py         figures and findings for the report
tests/
  synthetic.py   fake data with the real schema and its quirks
  test_pipeline.py
```

## Notes

- The competition data isn't included in this repo: download it yourself.
- The virtualenv lives in `~/.venvs/`, not in the project, because the project folder syncs to iCloud.
- On macOS without Homebrew, LightGBM and XGBoost can't find `libomp`. `setup.sh` points them at the copy that scikit-learn already ships.
