# v6 — `bigmart_submission_v6.csv`

**Leaderboard RMSE: not recorded**

Cross-validation on training data: CV 1078.9 (not comparable with leaderboard scores; LB runs ~70-80 points higher).

| | |
|---|---|
| Phase | Phase 1 - classical ML |
| Approach | Optuna-tuned individual models (params in code/model_artifacts/bigmart_best_params.json) + RidgeCV stack. |
| Uses leaderboard feedback? | No - chosen with training-data cross-validation only |
| Prediction file | [`bigmart_submission_v6.csv`](bigmart_submission_v6.csv) (5,681 rows) |
| Python | [`scripts_codex/bigmart_tune_v2.py`](../../code/scripts_codex/bigmart_tune_v2.py)<br>[`scripts_codex/bigmart_final_v6.py`](../../code/scripts_codex/bigmart_final_v6.py) |
| Note | Chat log: "v6 and v7 scored the worst" (no number recorded). |
