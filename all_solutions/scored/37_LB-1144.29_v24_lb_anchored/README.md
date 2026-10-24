# v24 — `bigmart_submission_v24_lb_anchored.csv`

**Leaderboard RMSE: 1144.29** (rank 37 of 42 scored files in this archive; lower is better)

Score source: Analytics Vidhya leaderboard (reported in the project chat log).

Cross-validation on training data: OOF 1068.98 (not comparable with leaderboard scores; LB runs ~70-80 points higher).

| | |
|---|---|
| Phase | Phase 2 - large ensembles |
| Approach | Repeated Ridge meta-model over 7 OOF artifacts, blended 62.5% / 37.5% with v5 ("leaderboard-anchored"). |
| Uses leaderboard feedback? | Partly - mixes trained models with choices or blends steered by leaderboard scores |
| Prediction file | [`bigmart_submission_v24_lb_anchored.csv`](bigmart_submission_v24_lb_anchored.csv) (5,681 rows) |
| Python | [`scripts_codex/bigmart_v24_lb_anchored.py`](../../code/scripts_codex/bigmart_v24_lb_anchored.py) |
| Note | Needs the Codex .npz artifacts, which are not in the project folder. |
