# v22 — `bigmart_submission_optimized.csv`

**Leaderboard RMSE: 1147.89** (rank 40 of 41 scored files in this archive; lower is better)

Score source: Analytics Vidhya leaderboard (reported in the project chat log).

Cross-validation on training data: CV 1071.25 (not comparable with leaderboard scores; LB runs ~70-80 points higher).

| | |
|---|---|
| Phase | Phase 1 - classical ML |
| Approach | Optimised feature engineering + target encoding + LightGBM/XGBoost/CatBoost (bigmart_v22_optimized.py, run with --output submission_v22.csv). |
| Uses leaderboard feedback? | No - chosen with training-data cross-validation only |
| Prediction file | [`bigmart_submission_optimized.csv`](bigmart_submission_optimized.csv) (5,681 rows) |
| Python | [`scripts_codex/bigmart_v22_optimized.py`](../../code/scripts_codex/bigmart_v22_optimized.py)<br>[`scripts_codex/bigmart_optimized_solution.py`](../../code/scripts_codex/bigmart_optimized_solution.py) |
| Identical files | submission_v22.csv<br>cd173aef-..._us_submission_v22.csv (Analytics Vidhya copy of the upload) |
| Note | Better CV than v5 but worse LB: a good example of the CV/LB gap. |
