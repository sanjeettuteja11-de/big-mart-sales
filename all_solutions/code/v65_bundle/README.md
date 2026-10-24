# BigMart v65: quantity correction from product-ID structure

Recommended submission: `bigmart_submission_v65_quantity_correction.csv`.
Alternative: `bigmart_submission_v66_quantity_correction.csv`.
**Neither file has a measured or estimated public RMSE. The 1120 target has not been achieved.**

## What changed

The earlier v64 estimate of 1128.21 was unreliable; the reported actual result
was 1140.0961134823. Approximating public-set quadratic statistics from nearly
dependent prior predictions produced false optimism. Those estimates are
withdrawn and are not used to fit this correction.

This experiment instead learns a correction from the supplied training labels.
Every known sale is an integer quantity times a product price. That price is
uniquely recovered from the MRP alone: enumerate tenths-valued adjustments in
[-2, 2] and identify the candidate on the 0.6658 price lattice. All 14,204 rows
have one candidate. All 8,523 known sales become integer quantities within
7.11e-15 numerical tolerance. This identifies the price component; it does not
reveal test quantities or test labels.

The numeric product-ID suffix contains a smooth demand pattern still present
in the legacy model's residuals. A regularized cubic spline learns that pattern,
with item type, family and cleaned fat-content controls. Outlet means are
computed within each training fold. Quantity effects are scaled to outlet demand
and converted back into sales using the recovered price.

The settings are fixed: 20 spline knots, ridge penalty 300, 5 outlet-stratified
folds at seeds 713, 2718, and 9317. v65 adds 75% of the averaged correction to
the exact previously scored v63 file; v66 adds 100%. Neither required clipping
negative predictions. The old v63 file is an explicit, checksummed model input.

## Validation evidence

46 candidate configurations were screened using training-data CV. The selected
recipe was then checked on fresh fold partitions with models retrained inside
each outer training fold. These are repeated validations on the available
training dataset, not an untouched external dataset.

| Validation | Fresh base RMSE | Corrected RMSE | Reduction |
|---|---:|---:|---:|
| Outlet-stratified, seed 2718 | 1073.2729 | 1064.4902 | 8.7826 |
| Outlet-stratified, seed 9317 | 1071.9576 | 1063.2923 | 8.6653 |
| Whole products held out, seed 4171 | 1071.6742 | 1062.4081 | 9.2661 |

Every one of these 15 folds improved. Whole-product holdout prevents any rows
of a held-out product entering the fit. A descriptive paired bootstrap grouped
by product gave a 95% improvement interval of about 5.96–12.71 RMSE for that
check; this interval does not adjust for earlier model selection.

The repeated quantity model itself had OOF RMSE 1062.5681, versus 1071.4433
for its neutral outlet-quantity baseline. Adding the same correction to the
legacy v55 OOF archive gave 1057.2952 versus 1066.1586. **That last comparison
is diagnostic only**: the legacy stack was not retrained in the new outer
folds and its historical selection used the same training data. The fresh-base
results above are the stronger evidence for the correction's contribution.

The final anchored submissions cannot be assigned honest local CV scores:
v63 was calibrated using earlier public feedback and has no corresponding
strictly held-out training prediction vector. Local reductions must not be
subtracted from the public 1140 score to promise another public result.

## Reproduce the submissions

The minimal generator uses Python 3.11, numpy, pandas, scipy and scikit-learn.
It trains the correction directly from raw data, without prediction caches.
It does **not** retrain the historical v63 ensemble. Required files in `--data-dir`:

- `train_v9rqX0R.csv`
- `test_AbJTz2l.csv`
- `sample_submission_8RXa3c6.csv`
- `bigmart_submission_v63_exact_v60_optimum.csv`

```sh
/Users/abhimanyunain/Desktop/CODEX/.venv-bigmart-v65/bin/python \
  /Users/abhimanyunain/Desktop/CODEX/bigmart_v65_research/make_submission.py
```

For another computer, install `requirements-submission.txt`, then run:

```sh
python make_submission.py --data-dir /path/to/inputs --output-dir /path/to/outputs
```

`submission_manifest.json` records input/output hashes, exact versions, settings,
prediction ranges, row checks and limitations. Both outputs have 5,681 rows in
the sample's exact identifier order, finite nonnegative values and no extra
columns. The independent experimental implementation and the minimal generator
agreed within 3.4e-12 on the learned corrections.

## Reproduce research checks

The local isolated environment also contains LightGBM, CatBoost and InterpretML.
See `requirements-research.txt`. The scripts run as follows:

```sh
python experiment.py --phase screen
python experiment.py --phase verify --seed 2718 --names clean_outlet_units,suffix_spline_k20_a300_p0
python experiment.py --phase verify2 --seed 9317 --names clean_outlet_units,suffix_spline_k20_a300_p0
python nested_check.py --seed 2718
python nested_check.py --seed 9317
python nested_check.py --seed 4171 --grouped
```

`nested_check.py` names the outer validation check; it evaluates a fixed recipe,
not an inner hyperparameter search. Saved research caches are local conveniences.
For a completely fresh rerun, copy the scripts into a new empty directory so no
old cache is reused. The final `make_submission.py` never uses those caches.

The Explainable Boosting Machine screening configurations follow the
[official InterpretML API](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html).
They were screened but are not needed by the selected correction generator.

## Current score ledger (user reported)

| Version | Public RMSE |
|---|---:|
| v55 | 1142.1520494780 |
| v58 half | 1142.2865500942 |
| v58 | 1142.4413084420 |
| v59 | 1141.7706040151 |
| v60 | 1140.2156006225 |
| v61 | 1140.0241654640 |
| v62 | 1140.3942899653 |
| v63 | 1140.0228905024 |
| v64 | 1140.0961134823 |
| v65 | Unscored |
| v66 | Unscored |

Only AV evaluation can establish whether the new correction closes the public
gap. No concealed test labels or public-row indicators were accessed or inferred.
