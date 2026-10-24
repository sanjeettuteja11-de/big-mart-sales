# v69q — `bigmart_submission_v69_denoised_quantity.csv`

**Leaderboard RMSE: 1122.9470** (rank 3 of 42 scored files in this archive; lower is better)

Score source: Analytics Vidhya leaderboard (recorded in submission_inventory.csv / SCORE_RANKING.md).

Cross-validation on training data: fresh-model proxy checks 1061.99 / 1062.00 (not comparable with leaderboard scores; LB runs ~70-80 points higher).

| | |
|---|---|
| Phase | Phase 4 - units x price correction |
| Approach | Same as v68q with delta applied at 1.5x. |
| Uses leaderboard feedback? | Partly - mixes trained models with choices or blends steered by leaderboard scores |
| Prediction file | [`bigmart_submission_v69_denoised_quantity.csv`](bigmart_submission_v69_denoised_quantity.csv) (5,681 rows) |
| Python | [`v67_v69_bundle/bigmart_v67_research/refinement.py`](../../code/v67_v69_bundle/bigmart_v67_research/refinement.py)<br>[`v67_v69_bundle/bigmart_v67_research/make_submission.py`](../../code/v67_v69_bundle/bigmart_v67_research/make_submission.py)<br>[`v67_v69_bundle/bigmart_v67_research/validate.py`](../../code/v67_v69_bundle/bigmart_v67_research/validate.py)<br>[`v67_v69_bundle/bigmart_v67_research/README.md`](../../code/v67_v69_bundle/bigmart_v67_research/README.md) |
