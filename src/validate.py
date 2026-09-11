"""Reproducible validation study; preprocessing fitted inside folds by default."""
import argparse,json
from pathlib import Path
from src.config import DATA_RAW,OUTPUTS,REPORTS,SEED
from src.data import load_raw
from src.models import SPECS_BY_NAME,load_tuned
from src.cv import run_cv,prepare
from src.artifacts import save_result,provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',default=['store_rate_popularity','lattice_store_rate','ridge_interactions','catboost_units_unweighted'])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--include-transductive',action='store_true')
    p.add_argument('--data',type=Path,default=DATA_RAW)
    args=p.parse_args();tr,te=load_raw(args.data);prov=provenance(tr,te);rows=[]
    for name in args.models:
        spec=SPECS_BY_NAME[name]
        runs=[('holdout',dict(scheme='holdout')),('stratified',dict(scheme='stratified')),('grouped',dict(scheme='grouped'))]
        if args.include_transductive:
            runs.append(('explicit_transductive',dict(fold_fit=False,prepared=prepare(tr,te,use_test_features=True))))
        for label,kw in runs:
            r=run_cv(spec,tr,te,params=load_tuned(name),n_repeats=args.repeats,**kw)
            rec=save_result(r,OUTPUTS/'validation_runs'/name/label,
                            {'label':label,'seed':SEED,'repeats':args.repeats,'params':{**spec.params,**load_tuned(name)},'fold_fit':label!='explicit_transductive'},prov)
            rows.append(rec);print(name,label,r.oof_rmse,flush=True)
    (OUTPUTS/'validation.json').write_text(json.dumps({'runs':rows},indent=2))
    lines=['# Validation study','','All scores are original-scale RMSE. Main evaluation uses shuffled outlet-stratified folds. Grouped validation holds out whole products and tests unseen-product robustness; it is a different prediction task. Fold variability is not a confidence interval.','','| Model | Scheme | RMSE | Raw RMSE | Fold mean ± SD |','|---|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['model']} | {r['config']['label']} | {r['oof_rmse']:.2f} | {r['raw_oof_rmse']:.2f} | {r['fold_rmse_mean']:.2f} ± {r['fold_rmse_std']:.2f} |")
    lines+=['','See `outputs/validation_runs/` for exact folds, predictions, configurations and fingerprints. Negative clipping is measured before and after; all learned preprocessing and early-stopping transforms are fitted within training splits.']
    (REPORTS/'validation.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
