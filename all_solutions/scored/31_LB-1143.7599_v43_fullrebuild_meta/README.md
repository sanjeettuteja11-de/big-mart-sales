# v43 — `bigmart_submission_v43_fullrebuild_meta.csv`

**Leaderboard RMSE: 1143.7599** (rank 31 of 42 scored files in this archive; lower is better)

Score source: Analytics Vidhya leaderboard (reported in the project chat log).

Cross-validation on training data: OOF 1066.45 (not comparable with leaderboard scores; LB runs ~70-80 points higher).

| | |
|---|---|
| Phase | Phase 2 - large ensembles |
| Approach | Independent from-scratch 4-model pipeline (LightGBM/XGBoost/CatBoost/GBR) merged into the meta-stack. |
| Uses leaderboard feedback? | No - chosen with training-data cross-validation only |
| Prediction file | [`bigmart_submission_v43_fullrebuild_meta.csv`](bigmart_submission_v43_fullrebuild_meta.csv) (5,681 rows) |
| Python | No saved generator script (built inline in an interactive session) |
| Note | No saved generator. |
