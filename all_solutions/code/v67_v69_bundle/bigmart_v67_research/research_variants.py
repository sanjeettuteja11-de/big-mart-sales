"""Cross-fitted incremental quantity models. All target fitting is fold-local.

This screens training-data RMSE, never unseen public-test scores. Prior v66
correction is the comparator. Covariate-only summaries reuse the audited input.
"""
from pathlib import Path
import sys, argparse, json, time
import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.preprocessing import SplineTransformer, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'bigmart_v65_research'
sys.path.insert(0,str(PREVIOUS))
from experiment import load_data, rmse


def block(fit, apply, columns, knots=None):
    enc=(SplineTransformer(n_knots=knots,degree=3,include_bias=False)
         if knots else OneHotEncoder(handle_unknown='ignore',sparse_output=False))
    a=enc.fit_transform(fit[columns]);b=enc.transform(apply[columns])
    scaler=StandardScaler()
    return scaler.fit_transform(a),scaler.transform(b)


def predict(config,fit,apply,y,p,pt):
    q=y/p
    means=pd.DataFrame({'o':fit.Outlet_Identifier.to_numpy(),'q':q}).groupby('o').q.mean()
    m=fit.Outlet_Identifier.map(means).to_numpy(float)
    mt=apply.Outlet_Identifier.map(means).to_numpy(float)
    scale=(m/16.)**config.get('scale_power',1.)
    scalet=(mt/16.)**config.get('scale_power',1.)
    z=(q-m)/scale
    w=scale**2*(p/150.)**config.get('price_power',0)
    if 'period' in config:
        def fourier(frame):
            angle=2*np.pi*frame.id_number.to_numpy()[:,None]/config['period']
            angle=angle*np.arange(1,config.get('harmonics',2)+1)[None,:]
            return np.c_[np.sin(angle),np.cos(angle)]
        scaler=StandardScaler();a=scaler.fit_transform(fourier(fit));b=scaler.transform(fourier(apply))
    else:
        a,b=block(fit,apply,['id_number'],config.get('knots',20))
    blocks=[(a,b,config.get('alpha',300))]
    if config.get('controls',True):
        c,d=block(fit,apply,['family','fat','Item_Type'])
        blocks.append((c,d,config.get('control_alpha',300)))
    for col,kind,penalty in config.get('extra',[]):
        c,d=block(fit,apply,[col],None if kind=='cat' else kind)
        blocks.append((c,d,penalty))
    if config.get('interaction'):
        col,penalty=config['interaction']
        c,d=block(fit,apply,[col])
        interaction=np.einsum('ij,ik->ijk',a,c).reshape(len(fit),-1)
        interactiont=np.einsum('ij,ik->ijk',b,d).reshape(len(apply),-1)
        blocks.append((interaction,interactiont,penalty))
    xx=np.c_[np.ones(len(fit)),*[v[0] for v in blocks]]
    xt=np.c_[np.ones(len(apply)),*[v[1] for v in blocks]]
    penalties=np.r_[1e-10,*[np.full(v[0].shape[1],v[2]) for v in blocks]]
    beta=solve(xx.T@(w[:,None]*xx)+np.diag(penalties),xx.T@(w*z),assume_a='pos')
    return np.maximum(0,(mt+xt@beta*scalet)*pt)


CONFIGS=[
    {'name':'reference'},
    *[{'name':f'suffix_k{k}_a{a}','knots':k,'alpha':a}
      for k,a in [(15,300),(20,1000),(20,100),(25,300),(25,1000),(30,1000)]],
    *[{'name':f'controls_{a}','control_alpha':a} for a in [1000,10000,1000000]],
    {'name':'suffix_only','controls':False},
    *[{'name':f'period_{period}_h{h}','controls':False,'period':period,'harmonics':h}
      for period,h in [(18,2),(19,2),(20,1),(20,2),(20,3),(20,4),(20,6),(21,2),(22,2)]],
    *[{'name':f'pure_k{k}_a{a}','controls':False,'knots':k,'alpha':a}
      for k,a in [(15,300),(20,100),(20,1000),(25,300),(25,1000)]],
    *[{'name':f'scale_{s}','scale_power':s} for s in [0,.5,1.5]],
    *[{'name':f'price_power_{p}','price_power':p} for p in [.5,1]],
    *[{'name':f'letter_{k}','extra':[('id_letter','cat',k)]} for k in [1000,10000]],
    *[{'name':f'extra_{col}','extra':[(col,5,3000)]}
      for col in ['price','weight','item_visibility','mrp_noise','assortment_size']],
    *[{'name':f'interaction_{col}_{a}','interaction':(col,a)}
      for col in ['Outlet_Identifier','family','Item_Type','id_letter'] for a in [10000,100000]],
]


def run(seed=713,phase='screen',names='',grouped=False):
    train,test,x,xt,y,price,cats,nums,audit=load_data()
    p=price[:len(y)];pt=price[len(y):]
    folds=(GroupKFold(5,shuffle=True,random_state=seed).split(x,groups=train.Item_Identifier)
           if grouped else StratifiedKFold(5,shuffle=True,random_state=seed).split(x,train.Outlet_Identifier))
    folds=list(folds);fid=np.zeros(len(y),int)
    for k,(_,va) in enumerate(folds): fid[va]=k
    configs=[c for c in CONFIGS if not names or c['name'] in names.split(',')]
    rows=[]
    for cfg in configs:
        name=cfg['name'];path=ROOT/f'{phase}_{name}.npz';start=time.monotonic()
        if path.exists():
            cache=np.load(path);oof=cache['oof'];tp=cache['test']
        else:
            oof=np.zeros(len(y));tp=np.zeros(len(xt))
            for fi,va in folds:
                apply=pd.concat([x.iloc[va],xt],ignore_index=True)
                prediction=predict(cfg,x.iloc[fi],apply,y[fi],p[fi],np.r_[p[va],pt])
                oof[va]=prediction[:len(va)];tp+=prediction[len(va):]/5
            np.savez(path,oof=oof,test=tp,fold_id=fid)
        row={'name':name,'rmse':rmse(y,oof),'seconds':time.monotonic()-start,
             **{f'fold_{k}':rmse(y[fid==k],oof[fid==k]) for k in range(5)}}
        print(json.dumps(row),flush=True);rows.append(row)
        pd.DataFrame(rows).sort_values('rmse').to_csv(ROOT/f'{phase}_results.csv',index=False)
    return rows


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=713)
    ap.add_argument('--phase',default='screen');ap.add_argument('--names',default='')
    ap.add_argument('--grouped',action='store_true');args=ap.parse_args()
    run(**vars(args))
