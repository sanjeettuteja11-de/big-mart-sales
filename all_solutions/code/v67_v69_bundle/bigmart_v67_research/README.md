# BigMart v67–v69: calibrated and denoised quantity correction

## What to submit

Try **v69** first, followed by **v68**. Both are new, unscored candidates.

| File | Change | Public RMSE status |
|---|---|---|
| v67_exact_quantity_optimum | Exact one-dimensional strength calibration of v66 | Calculated 1124.1798633738; not submitted here |
| v68_denoised_quantity | Remove the category-containing component of the new correction | Unknown |
| v69_denoised_quantity | Apply the same removal at 1.5 times strength | Unknown |

These are submission filenames prefixed by `bigmart_submission_` and suffixed by `.csv`.
No file is claimed to have achieved 1120. Only AV can establish the new public scores.

## Measured versus calculated

The user reported v63 = 1140.0228905024, v65 = 1126.0148395277 and
v66 = 1124.4058880453. Their exact local CSV hashes are recorded in `v67_manifest.json`.
For every row, v65 = v63 + 0.75 × (v66 − v63), within 1.37e-12.
On a fixed scored population, squared RMSE on that affine prediction line is
`27840.424981198972 * alpha² - 63204.01477970679 * alpha + 1299652.1908694473`.
Its minimum is at alpha = 1.135112248149756, giving the calculated v67 value.
This calculation assumes those reported scores match the exact files and the
public scoring rows and metric remain unchanged. It says nothing about a global
optimum, the private split, or the previously unscored v68/v69 direction.

## New trained improvement

v66's correction fitted a smooth product-number effect together with family,
fat-content and item-type adjustments. Training-validation tests supported
removing those category adjustments while keeping the smooth number effect.
This changes only the NEW correction; the historical base still has its features.
Noise fitting and overlap with the baseline are possible explanations, not
established causal facts. More elaborate category interactions performed worse.

The revised direction is trained on the same 15 folds as v66:

`delta = 1.135112248149756 × (suffix_only_model − original_quantity_model)`

`v68 = v67 + delta`; `v69 = v67 + 1.5 × delta`.

Both models recover the same exact product price, use fold-local outlet quantity
means, a 20-knot cubic spline, Ridge penalty 300, and outlet-scaled weights.
No hidden test labels, public-row membership, or external answer files are used.
The wider research explored 45 configurations, including the reference, with
additional fold checks for selected candidates. Reusing training data for
selection creates selection bias; these are sensitivity checks, not untouched
external tests or statistical guarantees.

## Validation evidence

The frozen v67 ensemble has no honest complete OOF vector. Therefore, the checks
below rebuild a separate CatBoost/LightGBM baseline in each outer fold, add the
original quantity correction at the calibrated strength, then compare the new
direction. These are fresh-model proxy checks, NOT v67/v68/v69 public scores.

| Check | Proxy baseline RMSE | v68-recipe RMSE | v69-recipe RMSE |
|---|---:|---:|---:|
| Fresh outlet-stratified five-fold, seed 11827 | 1066.5368 | 1062.7165 | 1061.9852 |
| Entire-product holdout five-fold, seed 8461 | 1065.3137 | 1062.2886 | 1062.0038 |

Every fold improved for both recipes in these two checks. Earlier fold checks
also favored category removal, with smaller overall gains. New seeds were
chosen after freezing the final two strengths; they still reuse the same dataset.
Test-set/public improvements can differ, including worsening. Public feedback
already influenced the historical baseline and v67 strength; private performance
is unknown.

## Reproduction

The bundle includes raw data, frozen scored CSV inputs, new submissions, source,
checksums, research results, a saved executed audit notebook and its text/table
preview. NumPy, pandas, SciPy and scikit-learn alone reproduce all three outputs.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r bigmart_v65_research/requirements-submission.txt
.venv/bin/python bigmart_v67_research/make_submission.py --data-dir data --output-dir data
.venv/bin/python bigmart_v67_research/refinement.py --data-dir data --output-dir data
```

This retrains the new correction and reproduces the supplied files from the
frozen v63/v65/v66 predictions. It does **not** retrain the historical v63
ensemble from scratch. The earlier v65 bundle contains its correction generator.
This limitation matters if the contest requires a complete end-to-end baseline
training source rather than reproducing from frozen prediction inputs.

To repeat the research and fresh proxy validation, install
`bigmart_v65_research/requirements-research.txt`. Set `DATA` in the preserved
`bigmart_v65_research/experiment.py` to your local `data` directory. Then run:

```sh
.venv/bin/python bigmart_v67_research/research_variants.py
.venv/bin/python bigmart_v67_research/validate.py --seed 11827
.venv/bin/python bigmart_v67_research/validate.py --seed 8461 --grouped
```

Research NPZ caches are intentionally omitted from the bundle and regenerate
from those commands. JSON/CSV measurements and the final delta cache are included.

## Expected versus observed checks

- 5681 rows and exactly three submission columns: passed for all outputs.
- Exact test identifier order and no duplicate item/outlet keys: passed.
- Finite, nonnegative predictions: passed. No rounding or clipping was applied.
- Original train/test key overlap: zero.
- Recovered training quantities are integers: maximum numerical error < 1e-8.
- Independent correction implementations agree: maximum test error < 3e-11,
  maximum OOF error < 9e-11.
- v69 − v67 = 1.5 × (v68 − v67): checked to floating-point tolerance.
- Actual v68/v69 public score and the 1120 target: **not yet verified**.

## Execution notes and blocker

All work ran locally on 2026-09-12 using Python 3.11.9. Screening runtimes and
fresh validation results are retained with the result files. No background training
or automatic submissions remain running. The remaining blocker is AV's unseen
evaluation: upload the new CSVs and report both full-precision scores.
The score curve then permits a new one-dimensional calibration, conditional
on the same scoring population; it cannot guarantee that direction reaches 1120.
