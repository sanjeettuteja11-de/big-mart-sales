# v5 — `bigmart_submission_v5.csv`

**Leaderboard RMSE: 1145.78** (rank 40 of 42 scored files in this archive; lower is better)

Score source: Analytics Vidhya leaderboard (reported in the project chat log).

Cross-validation on training data: CV 1073.5 (not comparable with leaderboard scores; LB runs ~70-80 points higher).

| | |
|---|---|
| Phase | Phase 1 - classical ML |
| Approach | 7-model stack (Ridge, GBR, RF, ExtraTrees, XGBoost, LightGBM, CatBoost) on 24 engineered features + fold-safe smoothed target encodings; RidgeCV meta-learner. |
| Uses leaderboard feedback? | No - chosen with training-data cross-validation only |
| Prediction file | [`bigmart_submission_v5.csv`](bigmart_submission_v5.csv) (5,681 rows) |
| Python | [`scripts_codex/bigmart_v5.py`](../../code/scripts_codex/bigmart_v5.py) |
| Identical files | bigmart_submission_v5_reproduced.csv |
