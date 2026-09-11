"""Grouped-product checks, focused feature ablations and uncertainty diagnostics."""
from __future__ import annotations
import json
from dataclasses import replace
import numpy as np
import pandas as pd
from src.continue_study import DEST,evaluate,BOOSTERS
from src.data import load_raw
from src.cv import run_cv,rmse
from src.config import TARGET,ITEM_ID,OUTLET_ID
from src.features import FEATURE_GROUPS
from src.models import SPECS_BY_NAME,load_tuned
from src.artifacts import provenance,save_result


def main():
    train,test=load_raw()
    rows=[]
    for name in ['store_rate_popularity','store_rate_plain','ridge_interactions','catboost_units_unweighted']:
        r,_,_=evaluate(name,train,test,'grouped/'+name,scheme='grouped')
        rows.append(r)
    for name in ['catboost_units_unweighted','lightgbm_raw']:
        evaluate(name,train,test,'ablation/'+name+'/full',repeats=1)
        groups={**FEATURE_GROUPS,'product_id':[ITEM_ID]}
        for group,columns in groups.items():
            if group=='product_id' and name.startswith('lightgbm'): continue
            evaluate(name,train,test,f'ablation/{name}/{group}',repeats=1,drop=set(columns))
    # Raw/log1p comparison on identical strict folds; no search across transforms.
    for target in ['raw','log1p']:
        spec=replace(SPECS_BY_NAME['catboost_units_unweighted'],target=target)
        r=run_cv(spec,train,test,params={},n_repeats=1)
        save_result(r,DEST/'targets'/target,{'target':target,'params':spec.params,'seed':42,'repeats':1},provenance(train,test))
        print('target',target,r.oof_rmse,flush=True)
    (DEST/'grouped.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__': main()
