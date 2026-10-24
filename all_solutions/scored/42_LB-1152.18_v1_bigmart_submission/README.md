# v1 — `bigmart_submission.csv`

**Leaderboard RMSE: 1152.18** (rank 42 of 42 scored files in this archive; lower is better)

Score source: Analytics Vidhya leaderboard (reported in the project chat log).

Cross-validation on training data: CV ~1079.9 (not comparable with leaderboard scores; LB runs ~70-80 points higher).

| | |
|---|---|
| Phase | Phase 1 - classical ML |
| Approach | First model: cleaning + one-hot features, GridSearch-tuned GradientBoostingRegressor refit on all rows. |
| Uses leaderboard feedback? | No - chosen with training-data cross-validation only |
| Prediction file | [`bigmart_submission.csv`](bigmart_submission.csv) (5,681 rows) |
| Python | [`scripts_codex/bigmart_prep.py`](../../code/scripts_codex/bigmart_prep.py)<br>[`scripts_codex/bigmart_model.py`](../../code/scripts_codex/bigmart_model.py)<br>[`scripts_codex/bigmart_tune.py`](../../code/scripts_codex/bigmart_tune.py)<br>[`scripts_codex/bigmart_final.py`](../../code/scripts_codex/bigmart_final.py) |
