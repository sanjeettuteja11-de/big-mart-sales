# Final systematic investigation (2026-09-11)

Checklist requested for the AMBA project, with where each item's evidence lives.
Validation throughout: 5-fold × 3 repeats stratified by store, preprocessing
fitted inside each training fold, three seeds where a comparison is close.
No leaderboard feedback was used to build or select any model.

| # | Item | Finding | Evidence |
|---|---|---|---|
| 1 | Leakage, train/test differences | No unseen products or stores, no duplicate pairs, no train/test overlap. Adversarial validation AUC 0.495. Regression tests prove validation labels and test features cannot reach out-of-fold predictions. | `error_analysis.json`, `tests/test_validation_integrity.py` |
| 2 | Target definition and transforms | Target is sales in currency; metric is RMSE on that scale. Every sale is whole units × a unit price on a 0.6658 currency step. log1p costs 35–40 RMSE; sqrt costs 6–9; unweighted units is best for boosting. | `ablations.md`, `price_model.py` |
| 3 | Outliers and robust losses | Units per store are mildly right-skewed (skew 0.25–1.2). Every robust store-rate estimator is worse than the mean (median +4.14, 5% trim +0.68, Huber c=3 +0.04). CatBoost with Huber or MAE loss is 1.9–4.2 worse. Under RMSE the mean is the right target. | this file |
| 4 | Time, seasonality, trends | The data has no dates: one year of totals. The only time-like field is the store's opening year (store age), which is used. | data audit |
| 5 | Store / product / category effects | Store format sets units sold (grocery 2.4, supermarkets 14–27 per product). Product identity: small, shrunk factor (split-half reliability 0.065). Category: none once the store is known. | `eda_summary.md`, `ablations.md` |
| 6 | Promotions, holidays, dates | Not in the data. The one price-related effect, each product's fixed offset from MRP, does not move units (p = 0.42). | this file |
| 7 | Lag and rolling features | Not applicable: no time axis. | |
| 8 | Group-based and time-aware CV | Grouped-by-product CV changes scores by under 1 point (models do not depend on product identity). Time-aware CV is not applicable. | `validation.md` |
| 9 | CatBoost, LightGBM, XGBoost, ensembles | All tuned with early stopping; best is CatBoost on unweighted units, 1073.25. Blends with the structural model add nothing under nested evaluation. | `experiments.md`, `nested_study.py` |
| 10 | Hyperparameter optimisation | Optuna, 25 trials per boosting model, bounded; gains 0–2 points. | `params/*.study.json` |
| 11 | Residual analysis | No segment has a significant bias: largest |t| is 1.9 (Bonferroni threshold 3.23); ANOVA p ≥ 0.44 for store, type, category, fat content, price band and prediction decile; residual–prediction correlation −0.003. | this file |
| 12 | Segment-specific models | The structural model is already per store. Store-type-specific product factors, grocery-specific factors and store × category factors change the score by −0.00 to +2.43. | this file |
| 13 | Blending / stacking on OOF | Simplex blends, fixed 75/25 mixes and second-stage crossfit all land within 0.1 of the single structural model. | `experiments.md` |
| 14 | Calibration / post-processing | Nested calibration (global scale, linear, per-store scale) makes the score 0.08–0.30 worse. Clipping at zero never triggers for the structural model. | this file |

## Best validated model

`LatticeStoreRate` (single file: `big_mart_solution.py`):

    sales = decoded unit price × store rate × shrunk product factor

- Validation RMSE 1070.85 (seed 42), 1070.95 mean over seeds 42 / 137 / 2026; the MRP version scores 1071.44 on the same folds.
- Features: Item_MRP (decoded to the unit price), Outlet_Identifier, Item_Identifier. Nothing else survived validation.
- Hyperparameters: product smoothing 55.23 (Optuna); price offsets −2.0…+2.0 in steps of 0.1; currency step learned from training sales inside each fold.
- Leaderboard 1147.83, the best of twelve submissions; the leaderboard tracked cross-validation on 9 of 11 follow-up files.
