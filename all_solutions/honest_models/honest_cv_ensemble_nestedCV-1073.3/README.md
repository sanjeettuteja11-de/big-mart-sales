# honest CV ensemble — `bigmart_submission_honest_cv_ensemble.csv`

**Leaderboard RMSE: not recorded**

Cross-validation on training data: nested CV 1073.3 +/- 22.3 (not comparable with leaderboard scores; LB runs ~70-80 points higher).

| | |
|---|---|
| Phase | Phase 5 - clean rebuilds / later files |
| Approach | CatBoost with interaction features + XGBoost on decoded units + the project NN and FLAML OOF predictions; blend weights by Nelder-Mead on OOF RMSE only. |
| Uses leaderboard feedback? | No - chosen with training-data cross-validation only |
| Prediction file | [`bigmart_submission_honest_cv_ensemble.csv`](bigmart_submission_honest_cv_ensemble.csv) (5,681 rows) |
| Python | No saved generator script (built inline in an interactive session) |
| Note | Generator build_honest_model.py was made in a separate chat (14 Sep) and is not in the project folder. |
