# Leaderboard log

These are **historical recorded scores**, recovered from commit `6720e0a` on
2026-09-11. This continuation did not upload a submission or independently
verify the authenticated leaderboard. The original log matched files by
submission time; four round-one files shared the same minute, so those
file-to-score associations remain provisional. Copies of the original log
are retained in Git history and `reports/history/`.

| Round | File | Historical leaderboard RMSE | Evidence status |
|---|---|---:|---|
| 0 | `store_rate_popularity_cv1071.csv` | 1148.28 | Recorded; upload receipt unavailable |
| 1 | `store_rate_plain_cv1072.csv` | 1148.74 | Provisional time-based association |
| 1 | `store_type_rate_popularity_cv1072.csv` | 1150.20 | Provisional time-based association |
| 1 | `ridge_interactions_cv1075.csv` | 1150.79 | Provisional time-based association |
| 1 | `catboost_raw_cv1079.csv` | 1149.96 | Provisional time-based association |
| 1 | `average_top5_cv1074.csv` | 1148.60 | Provisional time-based association |
| 2 | `blend_structural75_boosting25_cv1071.csv` | 1147.85 | Recorded; upload receipt unavailable |
| 2 | `blend_structural50_boosting50_cv1071.csv` | 1147.90 | Recorded; upload receipt unavailable |
| 2 | `boosting_avg6_cv1073.csv` | 1149.49 | Recorded; upload receipt unavailable |
| 2 | `catboost_units_unweighted_cv1074.csv` | 1150.90 | Recorded; upload receipt unavailable |
| Continuation | See `outputs/strict_study/recommendation.json` | **Not submitted** | Locally validated CSV; no leaderboard score |

Historical CV values were generated using preprocessing fitted on train plus
test features. Structural predictions have been reproduced exactly with
fold-local fitting; boosting results require the strict reruns in the current
experiment report. The later full-data refit file differs from the historical
fold-averaged submission and does **not** inherit its leaderboard score.

The best recorded historical score is 1147.85, but it is not a verified score
for the current recommended CSV. There is no justified fixed offset mapping
CV to the leaderboard. Public evaluation size, sampled rows, hidden targets,
and submission provenance are not available here. Model dependence, finite
samples, selected experiments, different preprocessing or refitting, and
possible distribution differences can all contribute to a CV–leaderboard gap.
A near-chance adversarial classifier only says that the tested classifier did
not distinguish observed features; it cannot prove identical distributions or
establish target shift. Similar model scores do not establish an achievable
minimum RMSE. Bootstrap diagnostics in the current report are conditional
training-sample analyses, not predictions of leaderboard error.

Ask for score receipts only when deciding which locally justified candidate
to upload next. Uploading requires the user's approval; missing scores do not
block local validation or reproducibility work.
