# v66q — `bigmart_submission_v66_quantity_correction.csv`

**Leaderboard RMSE: 1124.4059** (rank 4 of 41 scored files in this archive; lower is better)

Score source: Analytics Vidhya leaderboard (recorded in submission_inventory.csv / SCORE_RANKING.md).

| | |
|---|---|
| Phase | Phase 4 - units x price correction |
| Approach | Same correction as v65q added at 1.0x to v63. |
| Uses leaderboard feedback? | Partly - mixes trained models with choices or blends steered by leaderboard scores |
| Prediction file | [`bigmart_submission_v66_quantity_correction.csv`](bigmart_submission_v66_quantity_correction.csv) (5,681 rows) |
| Python | [`v65_bundle/make_submission.py`](../../code/v65_bundle/make_submission.py)<br>[`v65_bundle/experiment.py`](../../code/v65_bundle/experiment.py)<br>[`v65_bundle/nested_check.py`](../../code/v65_bundle/nested_check.py)<br>[`v65_bundle/README.md`](../../code/v65_bundle/README.md) |
| Note | The correction is trained on data; the base (v63) came from leaderboard probing. |
