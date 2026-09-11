# Leaderboard log

Every file submitted to the Analytics Vidhya leaderboard, what it tests, and
how it scored. Lower is better throughout.

- **CV**: 5-fold × 3-repeat cross-validation on train.
- **Expected**: the score a file should get if the test set behaves like
  train, meaning the first submission's leaderboard score plus the CV
  difference in squared error. A score clearly off from expected is the signal;
  luck between two models on the same test rows is only about 1–3 points.

Files are built with `python -m src.export` from the predictions that
`python -m src.train` saves.

## Round 0

| File | Model | CV | Leaderboard |
|---|---|---|---|
| `store_rate_popularity_cv1071.csv` | MRP × store rate × shrunk product factor | 1071.33 | **1148.28** (rank 566) |

Gap to CV: 77 points, about 8× the sampling noise of a test-sized sample of
train rows (1071 ± 9). Train and test features are indistinguishable
(adversarial validation AUC 0.495), so the test sales themselves behave
differently.

## Round 1: which part of the model carries over to test?

| File | Model | Tests | CV | Expected | Leaderboard |
|---|---|---|---|---|---|
| `store_rate_plain_cv1072.csv` | MRP × store rate, no product factor | Does the product factor carry over? | 1071.95 | 1148.9 | |
| `store_type_rate_popularity_cv1072.csv` | Rates pooled by store type × product factor | Do each store's own rates carry over? | 1072.32 | 1149.2 | |
| `ridge_interactions_cv1075.csv` | Ridge with a price slope per store, all columns | Do the other columns help on test? | 1075.10 | 1151.8 | |
| `catboost_raw_cv1079.csv` | CatBoost on sales, all columns | Same, for a tree model | 1078.64 | 1155.1 | |
| `average_top5_cv1074.csv` | Average of structural, Ridge, Poisson GLM, CatBoost, Extra Trees | Hedge | 1073.96 | 1150.7 | |

How to read the results:

- **No product factor clearly beats expected**: the product effect doesn't carry over. Drop it.
- **Pooled rates clearly beat expected**: individual store rates don't carry over. Next, shrink each store's rate toward its store type.
- **Ridge or CatBoost clearly beat expected**: the other columns carry signal on test that they don't on train. Next, tune the machine-learning models and blend them.
- **Everything lands near expected**: the test sales are just noisier. The structural model is the honest ceiling, and the remaining submissions go to small refinements.
