# v31 — `bigmart_submission_v31_meta_of_metas.csv`

**Leaderboard RMSE: 1143.9253** (rank 34 of 42 scored files in this archive; lower is better)

Score source: Analytics Vidhya leaderboard (recorded in submission_inventory.csv / SCORE_RANKING.md).

Cross-validation on training data: OOF 1067.22 (not comparable with leaderboard scores; LB runs ~70-80 points higher).

| | |
|---|---|
| Phase | Phase 2 - large ensembles |
| Approach | Meta-of-metas: v24-meta + v30-meta + v5 three-way blend. |
| Uses leaderboard feedback? | Partly - mixes trained models with choices or blends steered by leaderboard scores |
| Prediction file | [`bigmart_submission_v31_meta_of_metas.csv`](bigmart_submission_v31_meta_of_metas.csv) (5,681 rows) |
| Python | No saved generator script (built inline in an interactive session) |
| Note | Contains the LB-anchored v24. No saved generator. |
