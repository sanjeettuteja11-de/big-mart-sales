# v12/v13 — `bigmart_submission_v12_withNN.csv`

**Leaderboard RMSE: not recorded**

Cross-validation on training data: CV 1078.8 (stack); NN alone 1122.9 (not comparable with leaderboard scores; LB runs ~70-80 points higher).

| | |
|---|---|
| Phase | Phase 1 - classical ML |
| Approach | PyTorch embedding MLP (v12) blended into the 7-model stack (v13). |
| Uses leaderboard feedback? | No - chosen with training-data cross-validation only |
| Prediction file | [`bigmart_submission_v12_withNN.csv`](bigmart_submission_v12_withNN.csv) (5,681 rows) |
| Python | [`scripts_codex/bigmart_nn_v12.py`](../../code/scripts_codex/bigmart_nn_v12.py)<br>[`scripts_codex/bigmart_stack_with_nn_v13.py`](../../code/scripts_codex/bigmart_stack_with_nn_v13.py) |
