> Historical artifact from commit 6720e0a. Superseded by the current reports. Scores use the old protocol unless explicitly stated; causal, ceiling, leaderboard-offset and “honest nested” claims below are not endorsed.

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

| File | Model | Tests | CV | Expected | Leaderboard | vs expected |
|---|---|---|---|---|---|---|
| `store_rate_plain_cv1072.csv` | MRP × store rate, no product factor | Does the product factor carry over? | 1071.95 | 1148.86 | **1148.74** | −0.1 |
| `store_type_rate_popularity_cv1072.csv` | Rates pooled by store type × product factor | Do each store's own rates carry over? | 1072.32 | 1149.20 | **1150.20** | +1.0 |
| `ridge_interactions_cv1075.csv` | Ridge with a price slope per store, all columns | Do the other columns help on test? | 1075.10 | 1151.80 | **1150.79** | −1.0 |
| `catboost_raw_cv1079.csv` | CatBoost on sales, all columns | Same, for a tree model | 1078.64 | 1155.10 | **1149.96** | −5.1 |
| `average_top5_cv1074.csv` | Average of structural, Ridge, Poisson GLM, CatBoost, Extra Trees | Hedge | 1073.96 | 1150.74 | **1148.60** | −2.1 |

Submitted 11 Sep 2026, 10:50–10:51, in alphabetical file order; scores
matched to files by submission time (the four 10:50 entries are assumed to
be listed in upload order).

How to read the results:

- **No product factor clearly beats expected**: the product effect doesn't carry over. Drop it.
- **Pooled rates clearly beat expected**: individual store rates don't carry over. Next, shrink each store's rate toward its store type.
- **Ridge or CatBoost clearly beat expected**: the other columns carry signal on test that they don't on train. Next, tune the machine-learning models and blend them.
- **Everything lands near expected**: the test sales are just noisier. The structural model is the honest ceiling, and the remaining submissions go to small refinements.

### Round 1 results

No file beat the original 1148.28. The two structural variants landed on
their expected scores, so neither the product factor nor store-specific rates
behave differently on test.

The models that use every column did better than expected: Ridge by 1.0, the
average by 2.1, CatBoost by 5.1. How much of that could be luck? On random
subsets of training rows, the gap between CatBoost and the structural model
varies with a standard deviation of about 3.2 points if the public
leaderboard scores 30% of the test rows, or about 1.5 points if it scores all
of them. The share isn't published. CatBoost's −5.1 is therefore 1.6 to 3.5
standard deviations and the average's −2.1 is 1.4 to 3.0. Suggestive, not
conclusive, and consistent across all three feature-based files.

Round 2 follows that lead without tuning to the leaderboard: the improved
boosting models (early stopping and tuning moved CatBoost from 1078.64 to
1073.51 CV) and blends of boosting with the structural model, with blend
weights chosen by nested cross-validation.

## Round 2: do blends with the stronger boosting models carry over?

All four files come from the final cross-validation run (early stopping,
tuned parameters). The boosting average is the mean of CatBoost (units
unweighted, units, sales), XGBoost (units unweighted, sales) and LightGBM
(sales). Under cross-validation the structural model and both blends are
tied (within 0.3), yet their test predictions differ, so the leaderboard can
separate them.

| File | Model | CV | Expected | Leaderboard | vs expected |
|---|---|---|---|---|---|
| `blend_structural75_boosting25_cv1071.csv` | 75% structural + 25% boosting average (weight from nested CV: 0.24 ± 0.09) | 1071.10 | 1148.07 | **1147.85** | −0.2 |
| `blend_structural50_boosting50_cv1071.csv` | 50% structural + 50% boosting average | 1071.36 | 1148.31 | **1147.90** | −0.4 |
| `boosting_avg6_cv1073.csv` | Boosting average | 1073.32 | 1150.14 | **1149.49** | −0.7 |
| `catboost_units_unweighted_cv1074.csv` | Best single boosting model | 1073.51 | 1150.32 | **1150.90** | +0.6 |

Submitted 11 Sep 2026, 11:00–11:01, in the order 50/50 blend, 75/25 blend,
boosting average, CatBoost; scores matched to files by submission time.

### Round 2 results

Every file landed within 0.7 of its expected score, in the order
cross-validation predicted. Round 1's CatBoost result (5.1 better than
expected) did not repeat: the stronger CatBoost scored 0.6 worse than
expected. The likeliest reading is that round 1's gap was luck and that
cross-validation ranks these models correctly.

The 75/25 blend, whose weight was chosen by nested cross-validation before
any leaderboard feedback, scored **1147.85**: the best so far, 0.43 better
than round 0. It is the final model.
