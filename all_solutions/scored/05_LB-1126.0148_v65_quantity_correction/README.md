# v65q — `bigmart_submission_v65_quantity_correction.csv`

**Leaderboard RMSE: 1126.0148** (rank 5 of 41 scored files in this archive; lower is better)

Score source: Analytics Vidhya leaderboard (recorded in submission_inventory.csv / SCORE_RANKING.md).

Cross-validation on training data: fresh-base checks 1064.49 / 1063.29 / 1062.41 (not comparable with leaderboard scores; LB runs ~70-80 points higher).

| | |
|---|---|
| Phase | Phase 4 - units x price correction |
| Approach | Decode exact unit price from MRP (0.6658 step), learn a units correction (fold-local outlet mean units + cubic spline on product number + one-hot family/fat/type, Ridge alpha=300), add 0.75x to v63. |
| Uses leaderboard feedback? | Partly - mixes trained models with choices or blends steered by leaderboard scores |
| Prediction file | [`bigmart_submission_v65_quantity_correction.csv`](bigmart_submission_v65_quantity_correction.csv) (5,681 rows) |
| Python | [`v65_bundle/make_submission.py`](../../code/v65_bundle/make_submission.py)<br>[`v65_bundle/experiment.py`](../../code/v65_bundle/experiment.py)<br>[`v65_bundle/nested_check.py`](../../code/v65_bundle/nested_check.py)<br>[`v65_bundle/README.md`](../../code/v65_bundle/README.md) |
| Note | The correction is trained on data; the base (v63) came from leaderboard probing. |
