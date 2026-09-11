"""Conditional product-cluster bootstrap and segment diagnostics from saved OOF."""
import json
import numpy as np
import pandas as pd
from src.continue_study import DEST,BOOSTERS
from src.data import load_raw
from src.config import TARGET,ITEM_ID,OUTLET_ID
from src.cv import rmse


def cluster_bootstrap(y,base,candidate,groups,reps=2000,seed=918):
    codes,levels=pd.factorize(groups); n=len(levels)
    counts=np.bincount(codes)
    sb=np.bincount(codes,weights=(y-base)**2)
    sc=np.bincount(codes,weights=(y-candidate)**2)
    rng=np.random.default_rng(seed); diffs=[]; levels_rmse=[]
    for _ in range(reps):
        ix=rng.integers(n,size=n); denom=counts[ix].sum()
        rb=np.sqrt(sb[ix].sum()/denom); rc=np.sqrt(sc[ix].sum()/denom)
        diffs.append(rc-rb); levels_rmse.append(rb)
    return {'delta_rmse':rmse(y,candidate)-rmse(y,base),
            'delta_95pct_percentile_interval':np.quantile(diffs,[.025,.975]).tolist(),
            'baseline_95pct_percentile_interval':np.quantile(levels_rmse,[.025,.975]).tolist(),
            'bootstrap_replicates':reps,'seed':seed,
            'caveat':'Conditional on fixed OOF predictions, resampling entire products. Excludes retraining/selection uncertainty and new-outlet variation; not a leaderboard interval.'}


def main():
    tr,te=load_raw(); y=tr[TARGET].to_numpy()
    b=np.load(DEST/'baseline/store_rate_popularity/oof.npy')
    boost=np.mean([np.load(DEST/'baseline'/n/'oof.npy') for n in BOOSTERS],axis=0)
    blend=.75*b+.25*boost
    comparisons={'fixed_75_25_vs_structural':cluster_bootstrap(y,b,blend,tr[ITEM_ID]),
                 'catboost_vs_structural':cluster_bootstrap(y,b,np.load(DEST/'baseline/catboost_units_unweighted/oof.npy'),tr[ITEM_ID])}
    frame=tr.assign(residual=y-blend,sqerr=(y-blend)**2,mrp_band=pd.cut(tr.Item_MRP,[-np.inf,69,136,203,np.inf]).astype(str))
    segments={}
    for col in [OUTLET_ID,'Outlet_Type','Item_Type','mrp_band']:
        stats=frame.groupby(col).agg(rows=('sqerr','size'),mse=('sqerr','mean'),bias=('residual','mean'))
        stats['rmse']=np.sqrt(stats.pop('mse')); segments[col]=stats.reset_index().to_dict('records')
    audit={'train_rows':len(tr),'test_rows':len(te),'products_train':tr[ITEM_ID].nunique(),
           'outlets_train':tr[OUTLET_ID].nunique(),'unseen_test_products':len(set(te[ITEM_ID])-set(tr[ITEM_ID])),
           'unseen_test_outlets':len(set(te[OUTLET_ID])-set(tr[OUTLET_ID])),
           'duplicate_train_pairs':int(tr.duplicated([ITEM_ID,OUTLET_ID]).sum()),
           'duplicate_test_pairs':int(te.duplicated([ITEM_ID,OUTLET_ID]).sum()),
           'train_test_pair_overlap':len(set(zip(tr[ITEM_ID],tr[OUTLET_ID])) & set(zip(te[ITEM_ID],te[OUTLET_ID]))),
           'target_min':float(y.min()),'missing_train':tr.isna().sum().to_dict(),
           'missing_test':te.isna().sum().to_dict()}
    out={'data_audit':audit,'fixed_blend_rmse':rmse(y,blend),'boosting_average_rmse':rmse(y,boost),
         'comparisons':comparisons,'segments':segments}
    (DEST/'error_analysis.json').write_text(json.dumps(out,indent=2)); print(json.dumps(comparisons,indent=2))

if __name__=='__main__': main()
