"""Outer-fold evaluation of structural search and blend-weight selection.

Inside each outer training fold, regenerate all base OOF predictions. Select
structural regularization on those inner folds and learn a nonnegative blend
with a prespecified CatBoost units model. Outer validation labels are used
only for scoring. The search menu and model family were informed by earlier
whole-dataset exploration, so this is not a fresh untouched test dataset.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from src.artifacts import provenance, sha256
from src.config import TARGET, OUTPUTS
from src.continue_study import structural_grid
from src.cv import folds, run_cv, rmse
from src.data import load_raw
from src.models import SPECS_BY_NAME


def main():
    tr, te = load_raw()
    out = OUTPUTS / 'strict_study' / 'nested'; out.mkdir(parents=True, exist_ok=True)
    prov = provenance(tr,te)
    outer = list(folds('stratified',tr,5,1,314159))
    y = tr[TARGET].to_numpy()
    preds = {n: np.zeros(len(tr)) for n in ['baseline','selected_structural','blend']}
    test_preds=[]; records=[]; split_arrays={}
    for k,(a,b) in enumerate(outer):
        folder=out/str(k); folder.mkdir(exist_ok=True)
        if (folder/'result.json').exists():
            r=json.loads((folder/'result.json').read_text())
            if r['provenance'] != prov:
                raise ValueError('Nested cache mismatch; choose a new output directory')
            arr=np.load(folder/'predictions.npz')
        else:
            fit=tr.iloc[a].reset_index(drop=True); val=tr.iloc[b].reset_index(drop=True)
            inner=list(folds('stratified',fit,5,1,271828+k))
            iy=fit[TARGET].to_numpy()
            candidates=[]
            for p in structural_grid():
                z=run_cv(SPECS_BY_NAME['hierarchical_store_rate'],fit,val,params=p,n_repeats=1,
                         seed=271828+k,split_indices=inner)
                candidates.append((z.oof_rmse,p,z.oof))
            _,best_p,ip=min(candidates,key=lambda x:x[0])
            # Prespecified default, no parameters selected from outer labels.
            spec=SPECS_BY_NAME['catboost_units_unweighted']
            cb=run_cv(spec,fit,val,params={},n_repeats=1,seed=271828+k,split_indices=inner)
            d=cb.oof-ip
            w=float(np.clip(np.dot(d,iy-ip)/np.dot(d,d),0,1))
            # Full outer-training fit for both bases. Boosting early stopping
            # is entirely within that outer-training set.
            full=[(np.arange(len(a)),np.arange(len(a),len(a)+len(b)))]
            import pandas as pd
            joined=pd.concat([fit,val],ignore_index=True)
            st=run_cv(SPECS_BY_NAME['hierarchical_store_rate'],joined,te,params=best_p,
                      seed=314159+k,split_indices=full)
            boost=run_cv(spec,joined,te,params={},seed=314159+k,split_indices=full)
            baseline=run_cv(SPECS_BY_NAME['store_rate_popularity'],joined,te,params={'smoothing':55.234948951954145},
                            seed=314159+k,split_indices=full)
            arr={'baseline':baseline.oof[len(a):], 'selected_structural':st.oof[len(a):],
                 'blend':(1-w)*st.oof[len(a):]+w*boost.oof[len(a):],
                 'test_blend':(1-w)*st.test+w*boost.test}
            np.savez_compressed(folder/'predictions.npz',**arr)
            np.savez_compressed(folder/'inner_folds.npz',**{f'{kind}_{i}':ix for i,ab in enumerate(inner)
                                                         for kind,ix in zip(['train','valid'],ab)})
            np.savez_compressed(folder/'inner_predictions.npz',structural=ip,catboost=cb.oof)
            r={'outer_fold':k,'params':best_p,'boosting_weight':w,'catboost_params':spec.params,
               'inner_candidates':[{'rmse':c[0],'params':c[1]} for c in candidates],
               'inner_boosting_rmse':cb.oof_rmse,'boosting_rounds':boost.best_iterations,
               'provenance':prov}
            (folder/'result.json').write_text(json.dumps(r,indent=2))
        for n in preds: preds[n][b]=arr[n]
        test_preds.append(arr['test_blend']); records.append(r)
        split_arrays.update({f'train_{k}':a,f'valid_{k}':b})
        print(f"Outer {k}: structural {rmse(y[b],arr['selected_structural']):.2f}; "
              f"blend {rmse(y[b],arr['blend']):.2f}; booster weight {r['boosting_weight']:.3f}",flush=True)
    np.savez_compressed(out/'folds.npz',**split_arrays)
    np.savez_compressed(out/'predictions.npz',**preds,test_blend=np.mean(test_preds,axis=0))
    results={'scheme':'5 outer folds, seed 314159; 5 inner folds, seed 271828 + outer_fold',
             'rmse':{n:rmse(y,p) for n,p in preds.items()},
             'fold_rmse':{n:[rmse(y[b],p[b]) for _,b in outer] for n,p in preds.items()},
             'boosting_weights':[r['boosting_weight'] for r in records],
             'caveat':'Selection is isolated within each outer fold; prior exploration influenced the fixed model families/search menu. Fixed historical baseline smoothing was previously tuned on all rows.',
             'provenance':prov}
    (out/'summary.json').write_text(json.dumps(results,indent=2))
    print(results['rmse'],flush=True)

if __name__=='__main__': main()
