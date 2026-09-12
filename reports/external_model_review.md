# Review of an externally generated candidate submission (2026-09-12)

A prediction file produced in a separate ChatGPT session
(`us_bigmart_TwzTdzM.csv`, 5,681 rows) was offered as a possible submission.
It has no accompanying code, so it cannot be cross-validated here. This is
what can be established from the predictions themselves.

## What the file is

| Check | Result |
|---|---|
| Format, row count, row order | Valid: 5,681 rows in test order, correct columns |
| Reconstructed test labels? | **No.** Every real sale is a whole number of units at the decoded unit price; only 1.9% of these predictions come within 0.01 of that. It is a model's output |
| One of ours? | No. The closest of our twelve submissions differs by an RMS of 226 |
| Store structure | Same as ours: mean predicted units per store within a few percent of our model |
| Spread | Wider than any of our models (sd 1,366 vs 1,283–1,339) |

## What it does differently

Comparing it with our final model in log space (sd of the log ratio: 0.156):

| Source of the difference | Share |
|---|---|
| Per-product tilt | R² 0.554 |
| Price band | R² 0.054 |
| Store identity | R² 0.024 |
| Product category, fat content | R² 0.021, 0.017 |
| Shelf visibility | correlation −0.170 |

So it applies product-level adjustments of roughly ±12%, plus visibility and
price-band effects.

**The decisive check:** its per-product tilt correlates only **+0.12** with
what the training sales say about those same products. Most of its product
adjustment is not supported by the training evidence.

That matches our measurements. The product effect in this data has a
split-half reliability of 0.065, which is why our model shrinks it hard
(smoothing 55; weaker shrinkage of 25 costs +0.83 RMSE in paired CV). Shelf
visibility and price band show no effect on units sold once the store is
known (ablations: within ±1.1 RMSE, ANOVA p ≥ 0.25).

## Expected leaderboard score

Across our twelve submissions, a file's leaderboard score rises with its
distance from the plain structural model:

| Deviation from the structural model | Leaderboard |
|---|---|
| 0–30 | 1147.83 – 1148.28 |
| 58–79 | 1150.20 – 1150.90 |
| 95–121 | 1149.49 – 1149.96 |
| **228 (this file)** | **1157 – 1170 (estimated)** |

Two estimates agree: a noise model (LB² = best² + deviation², mean error
+0.46 on our twelve files) gives 1170, and a least-squares fit to our own
scores gives 1157. Both extrapolate beyond the measured range, so they
indicate direction rather than an exact figure.

## Conclusion

The file is legitimate to submit (it is the user's own, from their own
session) but is expected to score worse than the current best of 1147.83,
because it applies product, visibility and price-band adjustments that this
dataset does not support. Nothing in it was adopted. Its approach is already
covered by our experiments: weaker product shrinkage and feature-based
boosting both validated worse.
