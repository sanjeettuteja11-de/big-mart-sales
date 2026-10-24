# v55 — `bigmart_submission_v55_genie_strategy.csv`

**Leaderboard RMSE: 1142.1520** (rank 26 of 42 scored files in this archive; lower is better)

Score source: Analytics Vidhya leaderboard (recorded in submission_inventory.csv / SCORE_RANKING.md).

Cross-validation on training data: OOF 1066.16 (not comparable with leaderboard scores; LB runs ~70-80 points higher).

| | |
|---|---|
| Phase | Phase 2 - large ensembles |
| Approach | Databricks-Genie strategy: Optuna tuning with stratified, repeated CV (bigmart_v54_optuna_stratified_repeated.py, not in the folder). |
| Uses leaderboard feedback? | No - chosen with training-data cross-validation only |
| Prediction file | [`bigmart_submission_v55_genie_strategy.csv`](bigmart_submission_v55_genie_strategy.csv) (5,681 rows) |
| Python | No saved generator script (built inline in an interactive session) |
| Note | Best pure-ensemble leaderboard score. No saved generator. |
