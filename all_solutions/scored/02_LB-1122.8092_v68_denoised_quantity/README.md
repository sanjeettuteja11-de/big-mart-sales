# v68q — `bigmart_submission_v68_denoised_quantity.csv`

**Leaderboard RMSE: 1122.8092** (rank 2 of 42 scored files in this archive; lower is better)

Score source: Analytics Vidhya leaderboard (recorded in submission_inventory.csv / SCORE_RANKING.md).

Cross-validation on training data: fresh-model proxy checks 1062.72 / 1062.29 (not comparable with leaderboard scores; LB runs ~70-80 points higher).

| | |
|---|---|
| Phase | Phase 4 - units x price correction |
| Approach | v67q + delta, where delta = 1.1351 x (spline-only units model - original units model): the category terms are removed from the correction. |
| Uses leaderboard feedback? | Partly - mixes trained models with choices or blends steered by leaderboard scores |
| Prediction file | [`bigmart_submission_v68_denoised_quantity.csv`](bigmart_submission_v68_denoised_quantity.csv) (5,681 rows) |
| Python | [`v67_v69_bundle/bigmart_v67_research/refinement.py`](../../code/v67_v69_bundle/bigmart_v67_research/refinement.py)<br>[`v67_v69_bundle/bigmart_v67_research/make_submission.py`](../../code/v67_v69_bundle/bigmart_v67_research/make_submission.py)<br>[`v67_v69_bundle/bigmart_v67_research/validate.py`](../../code/v67_v69_bundle/bigmart_v67_research/validate.py)<br>[`v67_v69_bundle/bigmart_v67_research/research_variants.py`](../../code/v67_v69_bundle/bigmart_v67_research/research_variants.py)<br>[`v67_v69_bundle/bigmart_v67_research/README.md`](../../code/v67_v69_bundle/bigmart_v67_research/README.md) |
| Note | Second-best leaderboard score and the best file that contains a trained model change. Combines an LB-tuned base (v63), an LB-calibrated strength and a CV-validated units correction. Reproduces only from the frozen v63/v65/v66 CSVs. |
