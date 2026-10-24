"""Independent fold-safe check that product suffix helps a fresh base model.

Unlike the legacy prediction archive comparison, every baseline prediction and
correction here is rebuilt inside the same outer fold. This evaluates a fixed
recipe, including the 0.75 correction weight selected during screening.
"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from experiment import ROOT, load_data, baseline, fit_predict, rmse


def run(seed, grouped=False):
    train,test,X,Xt,y,p,cats,nums,_=load_data()
    n=len(y);ptr=p[:n];pte=p[n:]
    # Base excludes both identifier components, to test the incremental signal.
    bc=[c for c in cats if c!='id_letter'];bn=[c for c in nums if c!='id_number']
    boof=np.zeros(n);coof=np.zeros(n);bt=np.zeros(len(test));dt=np.zeros(len(test))
    flags=np.zeros(n,dtype=int)
    correction={'kind':'suffix_spline','knots':20,'alpha':300,'power':0}
    base_configs=[{'kind':'cat','depth':2,'power':0}, {'kind':'lgb','leaves':4,'power':2}]
    splits=(GroupKFold(5,shuffle=True,random_state=seed).split(X,groups=train.Item_Identifier)
            if grouped else StratifiedKFold(5,shuffle=True,random_state=seed).split(X,train.Outlet_Identifier))
    for k,(fi,va) in enumerate(splits):
        ax=pd.concat([X.iloc[va],Xt],ignore_index=True);ap=np.r_[ptr[va],pte]
        bp=np.mean([fit_predict(c,X.iloc[fi][bc+bn],ax[bc+bn],y[fi],ptr[fi],ap,bc,bn,seed+k)
                     for c in base_configs],axis=0)
        sp=fit_predict(correction,X.iloc[fi],ax,y[fi],ptr[fi],ap,cats,nums,seed+k)
        neutral=baseline(X.iloc[fi],ax,y[fi],ptr[fi],0)*ap
        delta=sp-neutral
        boof[va]=bp[:len(va)];coof[va]=bp[:len(va)]+.75*delta[:len(va)]
        bt+=bp[len(va):]/5;dt+=delta[len(va):]/5;flags[va]=k
    result={'seed':seed,'grouped_by_product':grouped,'fresh_base_rmse':rmse(y,boof),'corrected_rmse':rmse(y,coof),
            'gain':rmse(y,boof)-rmse(y,coof),
            'fold_gains':[rmse(y[flags==k],boof[flags==k])-rmse(y[flags==k],coof[flags==k]) for k in range(5)]}
    label=f'fresh_check_{seed}'+('_grouped' if grouped else '')
    np.savez(ROOT/f'{label}.npz',oof_base=boof,oof=coof,test_base=bt,test=bt+.75*dt,delta_test=dt,fold_id=flags)
    (ROOT/f'{label}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=2718)
    parser.add_argument('--grouped',action='store_true');args=parser.parse_args()
    run(args.seed,args.grouped)
