# v9 — `bigmart_submission_v9.csv`

**Leaderboard RMSE: not recorded**

Cross-validation on training data: CV 1082.4 (not comparable with leaderboard scores; LB runs ~70-80 points higher).

| | |
|---|---|
| Phase | Phase 1 - classical ML |
| Approach | Full-data target encoding and refit, 5-seed bagging. |
| Uses leaderboard feedback? | No - chosen with training-data cross-validation only |
| Prediction file | [`bigmart_submission_v9.csv`](bigmart_submission_v9.csv) (5,681 rows) |
| Python | [`scripts_codex/bigmart_v9_fulldata.py`](../../code/scripts_codex/bigmart_v9_fulldata.py) |
