# All BigMart solutions (v1 → v81)

Every prediction file made during the project, one folder each. Each folder holds the **CSV** that was (or could be) uploaded to
[Analytics Vidhya Big Mart Sales III](https://www.analyticsvidhya.com/datahack/contest/practice-problem-big-mart-sales-iii/),
a **README with its score and approach**, and a link to the **Python** that produced it. Everything is also in one table:
[`scores.csv`](scores.csv).

74 distinct solutions: 41 with a leaderboard score, 2 honest cross-validated models, 31 never scored.

## The two files that matter

| | Solution | Score | Approach |
|---|---|---|---|
| Best leaderboard score | [`v68_denoised_quantity`](scored/01_LB-1122.8092_v68_denoised_quantity/) | **LB 1122.8092** | Leaderboard-tuned blend (v63) + a trained units × price correction, strength calibrated on leaderboard scores |
| Best honest model | [`honest_quantity.py`](honest_models/honest_quantity_ridge_CV-1062.83/) | **CV 1062.83 ± 0.30** (best CV in the project) | Decode the exact unit price from MRP, predict units = store mean + spline/Ridge product effect, sales = units × price. Chosen on training CV only; reproducible from the raw CSVs in ~4 s |

## How to read the scores

- **LB** = Analytics Vidhya leaderboard RMSE on the hidden test rows. **Lower is better.** The #1 entry was 1126.03 when the project started.
- **CV / OOF** = RMSE from cross-validation on the 8,523 training rows. In this project LB runs about 70–80 points above CV, so only compare CV with CV and LB with LB.
- **LB feedback** says whether the file was built using leaderboard scores as an input:
  - **No**: trained and chosen with cross-validation only.
  - **Partly**: trained models, but a blend weight, anchor or base file was picked because of its leaderboard score.
  - **Yes**: calculated from reported leaderboard scores (*leaderboard probing*: for a fixed scoring set, squared RMSE along a blend line is an exact quadratic, so 2–3 scores pin down its minimum). These files fit the scored rows, not the data.
- Score sources: most come from `docs/submission_inventory.csv`; the rest are marked in each folder (chat log or the v65 bundle score ledger).

## Leaderboard ranking

| # | Solution | LB RMSE | Train CV / OOF | Phase | LB feedback | Python |
|---|---|---|---|---|---|---|
| 1 | [v68q](scored/01_LB-1122.8092_v68_denoised_quantity/) | 1122.8092 | fresh-model proxy checks 1062.72 / 1062.29 | Phase 4 | Partly | [`refinement.py`](code/v67_v69_bundle/bigmart_v67_research/refinement.py) +4 |
| 2 | [v69q](scored/02_LB-1122.9470_v69_denoised_quantity/) | 1122.9470 | fresh-model proxy checks 1061.99 / 1062.00 | Phase 4 | Partly | [`refinement.py`](code/v67_v69_bundle/bigmart_v67_research/refinement.py) +3 |
| 3 | [v79](scored/03_LB-1124.1799_v79_quantity_line_optimum/) | 1124.1799 | — | Phase 4 | **Yes** | [`build_optimum.py`](code/scripts_verification/build_optimum.py) |
| 4 | [v66q](scored/04_LB-1124.4059_v66_quantity_correction/) | 1124.4059 | — | Phase 4 | Partly | [`make_submission.py`](code/v65_bundle/make_submission.py) +3 |
| 5 | [v65q](scored/05_LB-1126.0148_v65_quantity_correction/) | 1126.0148 | fresh-base checks 1064.49 / 1063.29 / 1062.41 | Phase 4 | Partly | [`make_submission.py`](code/v65_bundle/make_submission.py) +3 |
| 6 | [v76](scored/06_LB-1139.5953_v76_qp_optimal/) | 1139.5953 | — | Phase 3 | **Yes** | [`geometry_audit.py`](code/scripts_verification/geometry_audit.py) |
| 7 | [v72](scored/07_LB-1139.7129_v72_extrapolated_optimum/) | 1139.7129 | — | Phase 3 | **Yes** | [`geometry_audit.py`](code/scripts_verification/geometry_audit.py) |
| 8 | [v74 (v31)](scored/08_LB-1139.8102_v74_probe_away_v31/) | 1139.8102 | — | Phase 3 | **Yes** | [`geometry_audit.py`](code/scripts_verification/geometry_audit.py) |
| 9 | [v71](scored/09_LB-1139.8765_v71_probe_v67_away_v60/) | 1139.8765 | — | Phase 3 | **Yes** | [`geometry_audit.py`](code/scripts_verification/geometry_audit.py) |
| 10 | [v75 (p15)](scored/10_LB-1139.9001_v75_toward_v31_p15/) | 1139.9001 | — | Phase 3 | **Yes** | [`geometry_audit.py`](code/scripts_verification/geometry_audit.py) |
| 11 | [v67 probe](scored/11_LB-1139.9383_v67_exact_optimum_v57line/) | 1139.9383 | — | Phase 3 | **Yes** | [`geometry_audit.py`](code/scripts_verification/geometry_audit.py) |
| 12 | [v73](scored/12_LB-1139.9837_v73_probe_v71_away_v43/) | 1139.9837 | — | Phase 3 | **Yes** | [`geometry_audit.py`](code/scripts_verification/geometry_audit.py) |
| 13 | [v63](scored/13_LB-1140.0229_v63_exact_v60_optimum/) | 1140.0229 | — | Phase 3 | **Yes** | — |
| 14 | [v61](scored/14_LB-1140.0242_v61_v60_curve75/) | 1140.0242 | — | Phase 3 | **Yes** | — |
| 15 | [v74 (v5)](scored/15_LB-1140.0386_v74_probe_away_v5/) | 1140.0386 | — | Phase 3 | **Yes** | [`geometry_audit.py`](code/scripts_verification/geometry_audit.py) |
| 16 | [v74 (v24)](scored/16_LB-1140.0592_v74_probe_away_v24/) | 1140.0592 | — | Phase 3 | **Yes** | [`geometry_audit.py`](code/scripts_verification/geometry_audit.py) |
| 17 | [v70](scored/17_LB-1140.0632_v70_probe_v67_to_v60_half/) | 1140.0632 | — | Phase 3 | **Yes** | [`geometry_audit.py`](code/scripts_verification/geometry_audit.py) |
| 18 | [v64](scored/18_LB-1140.0961_v64_psd_high_upside/) | 1140.0961 | — | Phase 3 | **Yes** | — |
| 19 | [v60](scored/19_LB-1140.2156_v60_regularized_score_stack/) | 1140.2156 | — | Phase 3 | **Yes** | — |
| 20 | [v75 (p3)](scored/20_LB-1140.2446_v75_toward_v31_p3/) | 1140.2446 | — | Phase 3 | **Yes** | [`geometry_audit.py`](code/scripts_verification/geometry_audit.py) |
| 21 | [v62](scored/21_LB-1140.3943_v62_updated_response_surface/) | 1140.3943 | — | Phase 3 | **Yes** | — |
| 22 | [v68 probe](scored/22_LB-1141.1136_v68_probe_half_v30/) | 1141.1136 | — | Phase 3 | **Yes** | [`geometry_audit.py`](code/scripts_verification/geometry_audit.py) |
| 23 | [v65 probe](scored/23_LB-1141.2372_v65_probe_half/) | 1141.2372 | — | Phase 3 | **Yes** | [`geometry_audit.py`](code/scripts_verification/geometry_audit.py) |
| 24 | [v59](scored/24_LB-1141.7706_v59_exact_curve/) | 1141.7706 | — | Phase 3 | **Yes** | — |
| 25 | [v55](scored/25_LB-1142.1520_v55_genie_strategy/) | 1142.1520 | OOF 1066.16 | Phase 2 | No | — |
| 26 | [v58 half](scored/26_LB-1142.2866_v58_item_history_anchor50/) | 1142.2866 | — | Phase 2 | Partly | — |
| 27 | [v58](scored/27_LB-1142.4413_v58_item_history/) | 1142.4413 | OOF 1065.33 | Phase 2 | No | — |
| 28 | [v44](scored/28_LB-1142.4430_v44_with_nn/) | 1142.4430 | OOF 1066.17 | Phase 2 | No | — |
| 29 | [v56](scored/29_LB-1143.3407_v56_widedeep_ordered/) | 1143.3407 | OOF 1065.96 | Phase 2 | No | — |
| 30 | [v43](scored/30_LB-1143.7599_v43_fullrebuild_meta/) | 1143.7599 | OOF 1066.45 | Phase 2 | No | — |
| 31 | [v57](scored/31_LB-1143.8883_v57_kernel_bayesian/) | 1143.8883 | OOF 1065.26 | Phase 2 | No | — |
| 32 | [v33](scored/32_LB-1143.9054_v33_variance_bias/) | 1143.9054 | OOF 1068.14 | Phase 2 | Partly | — |
| 33 | [v31](scored/33_LB-1143.9253_v31_meta_of_metas/) | 1143.9253 | OOF 1067.22 | Phase 2 | Partly | — |
| 34 | [v35](scored/34_LB-1143.9273_v35_bagged/) | 1143.9273 | — | Phase 2 | Partly | — |
| 35 | [v30](scored/35_LB-1143.9866_v30_allartifacts_meta/) | 1143.9866 | OOF 1067.33 | Phase 2 | No | — |
| 36 | [v24](scored/36_LB-1144.29_v24_lb_anchored/) | 1144.29 | OOF 1068.98 | Phase 2 | Partly | [`bigmart_v24_lb_anchored.py`](code/scripts_codex/bigmart_v24_lb_anchored.py) |
| 37 | [v36](scored/37_LB-1144.2918_v36_with_quantile/) | 1144.2918 | OOF 1067.06 | Phase 2 | No | — |
| 38 | [v46](scored/38_LB-1144.6392_v46_secondrebuild/) | 1144.6392 | OOF 1064.51 | Phase 2 | No | — |
| 39 | [v5](scored/39_LB-1145.78_v5/) | 1145.78 | CV 1073.5 | Phase 1 | No | [`bigmart_v5.py`](code/scripts_codex/bigmart_v5.py) |
| 40 | [v22](scored/40_LB-1147.89_v22_optimized/) | 1147.89 | CV 1071.25 | Phase 1 | No | [`bigmart_v22_optimized.py`](code/scripts_codex/bigmart_v22_optimized.py) +1 |
| 41 | [v1](scored/41_LB-1152.18_v1_bigmart_submission/) | 1152.18 | CV ~1079.9 | Phase 1 | No | [`bigmart_prep.py`](code/scripts_codex/bigmart_prep.py) +3 |

## Honest models (no leaderboard input)

| Solution | Train CV | Python | Note |
|---|---|---|---|
| [honest_quantity](honest_models/honest_quantity_ridge_CV-1062.83/) | CV 1062.83 +/- 0.30 (5 folds x 5 seeds) - best CV in the project | [`honest_quantity.py`](honest_models/honest_quantity_ridge_CV-1062.83/honest_quantity.py) | BEST HONEST MODEL. Fully reproducible from the raw CSVs in ~4 seconds. Same CV folds, spline-only variant: 1061.41. |
| [honest CV ensemble](honest_models/honest_cv_ensemble_nestedCV-1073.3/) | nested CV 1073.3 +/- 22.3 | — | Generator build_honest_model.py was made in a separate chat (14 Sep) and is not in the project folder. |

## Never scored

| Solution | Train CV / OOF | Phase | LB feedback | Python | Note |
|---|---|---|---|---|---|
| [v80 smooth](unscored/bigmart_v80_smooth_correction_UNSCORED/) | — | Phase 5 | ? | — | Unscored. No saved generator. |
| [v81](unscored/bigmart_v81_fresh_quantity_UNSCORED/) | — | Phase 5 | ? | — | Unscored. No saved generator. |
| [DeepSeek](unscored/deepseek_submission_final/) | — | Phase 1 | No | [`deepseek_python_20260912_839cf1.py`](code/scripts_codex/deepseek_python_20260912_839cf1.py) |  |
| [v20](unscored/submission_final_v20/) | — | Phase 1 | No | [`bigmart_v20_userscript.py`](code/scripts_codex/bigmart_v20_userscript.py) |  |
| [v21](unscored/submission_v21/) | — | Phase 1 | No | [`bigmart_v21_userscript.py`](code/scripts_codex/bigmart_v21_userscript.py) |  |
| [(unknown)](unscored/unknown_submission/) | — | Phase 1 | No | — |  |
| [v12/v13](unscored/v12_withNN/) | CV 1078.8 (stack); NN alone 1122.9 | Phase 1 | No | [`bigmart_nn_v12.py`](code/scripts_codex/bigmart_nn_v12.py) +1 |  |
| [v14](unscored/v14_onehot/) | — | Phase 1 | No | [`bigmart_v14_onehot.py`](code/scripts_codex/bigmart_v14_onehot.py) |  |
| [v19](unscored/v19_pseudolabel/) | — | Phase 1 | No | [`bigmart_v19_pseudolabel.py`](code/scripts_codex/bigmart_v19_pseudolabel.py) |  |
| [v2](unscored/v2/) | — | Phase 1 | No | [`bigmart_v2.py`](code/scripts_codex/bigmart_v2.py) | The chat log reports "v2/v3 (blend, submitted) = 1146.90" without saying which file; not assigned. |
| [v22 baseline](unscored/v22_baseline_outlet_mrp/) | — | Phase 1 | No | [`bigmart_v22_optimized.py`](code/scripts_codex/bigmart_v22_optimized.py) |  |
| [v23](unscored/v23_meta/) | — | Phase 1 | No | [`bigmart_v23_three_techniques.py`](code/scripts_codex/bigmart_v23_three_techniques.py) | The v23 script only runs the CV experiments; the CSV itself was built inline. |
| [v23 + v5](unscored/v23_meta_v5_blend/) | — | Phase 1 | Partly | — | Blended toward v5 because v5 had the best leaderboard score at the time. No saved generator. |
| [v25](unscored/v25/) | — | Phase 2 | Partly | — | No saved generator. |
| [v25 extrapolated](unscored/v25_extrapolated/) | OOF 1068.74 | Phase 2 | Partly | — | No saved generator. |
| [v3](unscored/v3/) | — | Phase 1 | No | [`bigmart_v3.py`](code/scripts_codex/bigmart_v3.py) | The chat log reports "v2/v3 (blend, submitted) = 1146.90" without saying which file; not assigned. |
| [v4](unscored/v4/) | — | Phase 1 | No | [`bigmart_v4.py`](code/scripts_codex/bigmart_v4.py) |  |
| [v47](unscored/v47_nn_ensemble_replace/) | — | Phase 2 | No | — | No saved generator. |
| [v48](unscored/v48_both_nns/) | — | Phase 2 | No | — | No saved generator. |
| [v49](unscored/v49_variance_reduced/) | OOF 1066.85 | Phase 2 | No | — | The chat log reports a v49 score of 1142.4518 without naming which v49 file. No saved generator. |
| [v49b](unscored/v49b_bag_v44_v49/) | — | Phase 2 | No | — | See v49 note. No saved generator. |
| [v52](unscored/v52_svr_deepcat/) | — | Phase 2 | No | — | No saved generator. |
| [v6](unscored/v6/) | CV 1078.9 | Phase 1 | No | [`bigmart_tune_v2.py`](code/scripts_codex/bigmart_tune_v2.py) +1 | Chat log: "v6 and v7 scored the worst" (no number recorded). |
| [v67q](unscored/v67_exact_quantity_optimum/) | calculated LB 1124.1799 (not submitted) | Phase 4 | **Yes** | [`make_submission.py`](code/v67_v69_bundle/bigmart_v67_research/make_submission.py) +1 |  |
| [v7](unscored/v7/) | CV 1076.6 | Phase 1 | No | [`bigmart_mega_v7.py`](code/scripts_codex/bigmart_mega_v7.py) | Chat log: "v6 and v7 scored the worst" (no number recorded). |
| [v76 (v31 axis)](unscored/v76_v31_axis_optimum/) | — | Phase 3 | **Yes** | — | Built inline; no saved generator. |
| [v77](unscored/v77_double_extrapolation/) | — | Phase 3 | **Yes** | — | Built inline; no saved generator. |
| [v78](unscored/v78_compound_corrections/) | — | Phase 3 | **Yes** | — | Built inline; no saved generator. |
| [v8](unscored/v8/) | — | Phase 1 | No | [`bigmart_v8_itemTE.py`](code/scripts_codex/bigmart_v8_itemTE.py) |  |
| [v80 hull](unscored/v80_hull_optimum/) | — | Phase 5 | **Yes** | — | Appears to have been uploaded, but no score is recorded in any project file. No saved generator. |
| [v9](unscored/v9/) | CV 1082.4 | Phase 1 | No | [`bigmart_v9_fulldata.py`](code/scripts_codex/bigmart_v9_fulldata.py) |  |

## The five phases

1. **Classical ML (v1–v23)**: cleaning, feature engineering, fold-safe target encoding, 7-model stacking. Best: v5, CV 1073.5, LB 1145.78.
2. **Large ensembles (v24–v58)**: out-of-fold artifact meta-stacks, neural networks, Optuna, AutoML. Best: v55, LB 1142.15.
3. **Leaderboard probing (v59–v78)**: blends solved from reported LB scores. Best: v76, LB 1139.60.
4. **Units × price correction (v65q–v69q, v79)**: every training sale is a whole number of units × a unit price on a 0.6658 step, decodable from `Item_MRP`. A trained units correction on top of v63 gave the big jump. Best: v68q, LB 1122.81.
5. **Clean rebuilds (Sep–Oct)**: `honest_quantity.py` (CV 1062.83) and the honest CV ensemble (nested CV 1073.3), plus the later v80/v81 files.

## Folder layout

```
all_solutions/
  scores.csv              every solution: score, source, CV, phase, LB feedback, approach, Python file
  scored/NN_LB-<score>_<name>/     CSV + README (ranked, best first)
  honest_models/<name>_CV-<score>/ CSV + README (+ honest_quantity.py)
  unscored/<name>/                 CSV + README
  code/
    scripts_codex/        v1–v24 generator scripts
    scripts_verification/ score analysis + v79 build_optimum.py
    v65_bundle/           v65q/v66q quantity-correction research and generator
    v67_v69_bundle/       v67q–v69q refinement, validation, audit notebook
    model_artifacts/      Optuna best parameters used by v6/v7
  docs/                   original SCORE_RANKING.md, submission_inventory.csv, Databricks Genie solution PDF
```

## Reproducing

- **honest_quantity** (from the repo root, CSVs in `data/raw/`):

  ```bash
  python all_solutions/honest_models/honest_quantity_ridge_CV-1062.83/honest_quantity.py --train data/raw/train_v9rqX0R.csv --test data/raw/test_AbJTz2l.csv --out submissions/honest_quantity_ridge.csv --cv
  ```

- **v68q / v69q**: see [`code/v67_v69_bundle/bigmart_v67_research/README.md`](code/v67_v69_bundle/bigmart_v67_research/README.md). The scripts retrain the correction from raw data but add it to the frozen v63/v65/v66 CSVs (copy them into `code/v67_v69_bundle/data/`). The v63 ensemble itself cannot be retrained: its code was never saved.
- **v1–v22**: the scripts in `code/scripts_codex/` read `train_v9rqX0R.csv` / `test_AbJTz2l.csv` from the working directory (some use `~/Downloads`).
- Files marked "No saved generator" were produced inline in interactive sessions; only the CSV survives.

## Not included

- **Competition data** (`train`, `test`, `sample_submission`): needs an Analytics Vidhya login and is not redistributed (see the root `.gitignore`). Download it into `data/raw/`.
- Pickled feature tables and `.npy` prediction caches (they embed the data or are regenerated by the scripts), CatBoost logs, macOS metadata.
- The exported AI chat log of the working session.
- Third-party reference solutions consulted during the project: [karinsasaki/Big-Mart-Sales](https://github.com/karinsasaki/Big-Mart-Sales), [nguneonard/Big-Mart-Sales-Prediction](https://github.com/nguneonard/Big-Mart-Sales-Prediction), [SahilCharan/BigMart-Sales-Prediction](https://github.com/SahilCharan/BigMart-Sales-Prediction).
