# v23 — `bigmart_submission_v23_meta.csv`

**Leaderboard RMSE: not recorded**

| | |
|---|---|
| Phase | Phase 1 - classical ML |
| Approach | Meta-model over v22/v23 artifacts (LightGBM/XGBoost/CatBoost + RandomizedSearchCV experiments in v23 script). |
| Uses leaderboard feedback? | No - chosen with training-data cross-validation only |
| Prediction file | [`bigmart_submission_v23_meta.csv`](bigmart_submission_v23_meta.csv) (5,681 rows) |
| Python | [`scripts_codex/bigmart_v23_three_techniques.py`](../../code/scripts_codex/bigmart_v23_three_techniques.py) |
| Note | The v23 script only runs the CV experiments; the CSV itself was built inline. |
