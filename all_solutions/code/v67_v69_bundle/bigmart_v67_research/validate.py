"""Fold-safe fresh baseline tests for the fixed v68/v69 refinement recipe.

New seeds are additional sensitivity checks, not new independent datasets.
Neither validation labels nor public scores fit any within-fold model.
"""
from pathlib import Path
import sys,argparse,json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,GroupKFold
from refinement import fit_predict as quantity_predict,features,ALPHA

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'bigmart_v65_research'))
from experiment import load_data,fit_predict,baseline,rmse


def run(seed,grouped=False):
    train,test,x,xt,y,price,cats,nums,_=load_data();n=len(y);p=price[:n];pt=price[n:]
    bc=[c for c in cats if c!='id_letter'];bn=[c for c in nums if c!='id_number']
    folds=(GroupKFold(5,shuffle=True,random_state=seed).split(x,groups=train.Item_Identifier)
           if grouped else StratifiedKFold(5,shuffle=True,random_state=seed).split(x,train.Outlet_Identifier))
    flags=np.zeros(n,int);old_oof=np.zeros(n);delta=np.zeros(n)
    configs=[{'kind':'cat','depth':2,'power':0},{'kind':'lgb','leaves':4,'power':2}]
    for k,(fi,va) in enumerate(folds):
        apply=x.iloc[va]
        base=np.mean([fit_predict(c,x.iloc[fi][bc+bn],apply[bc+bn],y[fi],p[fi],p[va],bc,bn,seed+k)
                      for c in configs],axis=0)
        neutral=baseline(x.iloc[fi],apply,y[fi],p[fi],0)*p[va]
        old=quantity_predict(x.iloc[fi],apply,y[fi],p[fi],p[va],True)
        new=quantity_predict(x.iloc[fi],apply,y[fi],p[fi],p[va],False)
        old_oof[va]=base+ALPHA*(old-neutral);delta[va]=ALPHA*(new-old);flags[va]=k
    result={'seed':seed,'grouped_by_item':grouped,'baseline_rmse':rmse(y,old_oof),'variants':{}}
    for strength in [1.,1.5]:
        pred=old_oof+strength*delta
        gains=[rmse(y[flags==k],old_oof[flags==k])-rmse(y[flags==k],pred[flags==k]) for k in range(5)]
        result['variants'][str(strength)]={'rmse':rmse(y,pred),'gain':rmse(y,old_oof)-rmse(y,pred),
                                         'fold_gains':gains}
    name=f'validation_{seed}'+('_grouped' if grouped else '')
    np.savez(ROOT/f'{name}.npz',baseline=old_oof,delta=delta,fold_id=flags)
    (ROOT/f'{name}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=11827)
    ap.add_argument('--grouped',action='store_true');args=ap.parse_args();run(args.seed,args.grouped)
