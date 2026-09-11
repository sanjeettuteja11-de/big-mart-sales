# Big Mart Sales Prediction: summary for the AMBA course

## 1. The business problem

BigMart runs 10 stores of four formats (grocery stores and three supermarket
types) in three city tiers. For 2013 it has per-product sales for each store:
8,523 product–store rows with sales, and 5,681 rows where the sales figure is
to be predicted. Two questions matter to the business:

1. **Forecasting:** how much of product X will store Y sell? Accurate
   forecasts feed stocking, allocation across stores and shelf planning.
2. **Understanding:** which product and store properties are associated with
   higher sales, and by how much?

The competition scores forecasts by root-mean-square error (RMSE) on the
sales figure, so large misses are penalised most.

## 2. The data and how it was cleaned

Each row has 11 attributes: six about the product (code, weight, fat content,
shelf visibility, category, maximum retail price) and five about the store
(code, opening year, size, city tier, format). Every product in the test file
also appears in the training file, in other stores.

| Issue in the raw file | Decision | Why |
|---|---|---|
| Fat content spelled five ways (`LF`, `low fat`, `reg`, …) | Mapped to two labels; non-consumables (soap, household) marked `Non-Edible` | Same fact, different spellings; a fat label on soap is meaningless |
| Product weight blank in 17% of rows, all in two stores | Copied from the same product in other stores | A product weighs the same everywhere |
| Shelf visibility exactly 0 in 6% of rows | Treated as "not recorded", replaced by the product's visibility elsewhere, with a flag kept | A product that sold was on a shelf |
| Store size blank for three whole stores | Most common size of the same store format, with a flag kept | Best available guess; the flag lets a model use "unknown" if that matters |

All imputation statistics come from product and store columns, never from the
sales figure, and they were checked by refitting them inside each validation
fold: the result is unchanged (`validation.md`). Store age uses 2013 as the
reference year, the year the sales were recorded.

## 3. How the models were validated

Predictions were scored on rows the model had not seen, with the scoring
repeated: five folds, three times, always keeping every store in proportion.
The same splits were used for every model so comparisons are paired.

Two further checks: a single fixed 80/20 split (for quick experiments; its
score is noisier), and folds that hold out whole products. The product-held-out
check is not the competition's situation, since every test product is known,
but it shows how much each model depends on product identity.

Anything learned from sales figures, such as a product's popularity, was
computed strictly out-of-fold, so no row ever informed its own prediction.

## 4. What the data showed

**Every sale is a whole number of units times a price within ±2 of the
listed MRP.** For example, 443.42 = 9 × (48.27 + 1.00). This holds for all
8,523 training rows. The ±2 price offset is centred on zero and unrelated to
anything else: it is noise.

That turns the question "what drives sales?" into "what drives units sold?",
and the answer is the store format:

| Store format | Units sold per product (mean) |
|---|---|
| Grocery store | 2.4 |
| Supermarket Type 2 | 13.9 |
| Supermarket Type 1 | 16.3 |
| Supermarket Type 3 | 26.6 |

Within a store, units do not depend on price, product category, fat content,
shelf visibility or weight (statistical tests in `eda_summary.md`). Product
identity has a small effect: some products sell a few percent above or below
their store's norm, and even that has to be shrunk heavily to be useful.

## 5. The models

Fourteen models were compared, from linear regression to gradient boosting
(CatBoost, LightGBM, XGBoost) and random forests. The best is the simplest one
that matches the structure above:

> predicted sales = MRP × the store's average units per product × a small
> product adjustment

| Model | CV RMSE |
|---|---|
| Always predict the average | 1,706 |
| Best gradient boosting (CatBoost, early-stopped) | 1,074 |
| Ridge regression with a price slope per store | 1,075 |
| **Structural model (above)** | **1,071** |

The boosting models can see every column, yet they score slightly worse:
the extra columns carry no signal about units, so the flexibility only fits
noise. Predicting log(sales), a common default for skewed targets, scored
about 40 points worse because the competition measures error on the raw
scale. Combining models did not beat the best single model.

The leaderboard score of the structural model was **1,148.28** (rank 566 at
submission time), against 1,071 in cross-validation. The gap is far larger
than sampling noise, while train and test features are statistically
indistinguishable, so the test sales are noisier than the training sales in
a way the features cannot explain. The gap affects every competitor; the
follow-up submissions in `leaderboard_log.md` test which model handles it best.

## 6. Limitations

- The RMSE floor is high. At a Type 1 supermarket, units per product vary
  with a standard deviation of about 7 around a mean of 16, and nothing in
  the data explains that variation. No model can forecast what the data does
  not contain.
- One year, ten stores. Store-format effects are estimated from two to six
  stores per format, so they describe these stores rather than a law.
- The findings are associations. Stores were not randomly assigned formats,
  and products were not randomly priced, so "Type 3 supermarkets sell 11×
  more units than grocery stores" describes the observed stores, not what
  would happen if a grocery store were converted.
- The leaderboard test set behaves differently from the training set for
  reasons the data does not reveal.

## 7. Business implications

- **Revenue per product is price × store-format units.** Product-level
  attributes that are usually assumed to matter (visibility, category, fat
  content) show no association with units sold once the store is known. Spending
  on shelf placement should be justified by evidence beyond this dataset.
- **Store format is the lever.** Supermarkets sell 6–11× the units per product
  of grocery stores. Expansion and conversion decisions dominate everything
  else in this data.
- **Pricing changes revenue, not units.** Within the observed range, units
  sold do not fall with price. That is worth testing deliberately before
  acting on it, since the data never varied one product's price within a
  store.
- **Forecast at the store level.** For stocking and allocation, a simple
  store-rate model is as accurate as any machine-learning model here, far
  cheaper to run, and easy to explain to store managers.

## Reproducibility

Code, tests and exact commands: https://github.com/sanjeettuteja11-de/big-mart-sales
(see the README). Experiment tables: `experiments.md`, `validation.md`,
`ablations.md`. Submission history: `leaderboard_log.md`.
