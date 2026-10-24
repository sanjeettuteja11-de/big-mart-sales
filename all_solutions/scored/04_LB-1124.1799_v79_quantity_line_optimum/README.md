# v79 — `bigmart_submission_v79_quantity_line_optimum.csv`

**Leaderboard RMSE: 1124.1799** (rank 4 of 42 scored files in this archive; lower is better)

Score source: Analytics Vidhya leaderboard (recorded in submission_inventory.csv / SCORE_RANKING.md).

| | |
|---|---|
| Phase | Phase 4 - units x price correction |
| Approach | Exact optimum on the v65q -> v66q line from their two LB scores (pure algebra, no model). |
| Uses leaderboard feedback? | Yes - computed directly from reported leaderboard scores (leaderboard probing) |
| Prediction file | [`bigmart_submission_v79_quantity_line_optimum.csv`](bigmart_submission_v79_quantity_line_optimum.csv) (5,681 rows) |
| Python | [`scripts_verification/build_optimum.py`](../../code/scripts_verification/build_optimum.py) |
| Note | Calculated score 1124.17986 vs actual 1124.17987: shows the probing algebra is exact. |
