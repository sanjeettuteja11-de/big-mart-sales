"""Fresh, reproducible quantity/price experiments; no leaderboard fitting.

Run with ../.venv-bigmart-v65/bin/python experiment.py --phase screen.
Covariate-only item summaries use supplied train+test features (transductive).
Every fitted target statistic uses only the training portion of the fold.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from interpret.glassbox import ExplainableBoostingRegressor

ROOT = Path(__file__).resolve().parent
DATA = Path('/Users/abhimanyunain/Downloads')
TARGET = 'Item_Outlet_Sales'
KEYS = ['Item_Identifier', 'Outlet_Identifier']


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def recover_price(mrp):
    candidates = np.asarray(mrp)[:, None] + .1*np.arange(-20, 21)[None, :]
    errors = abs(candidates/.6658 - np.rint(candidates/.6658))
    if not np.all((errors < 1e-7).sum(axis=1) == 1):
        raise ValueError('Price lattice is not uniquely identified for every row')
    # Round lattice index to avoid carrying floating point MRP noise.
    price = candidates[np.arange(len(candidates)), errors.argmin(axis=1)]
    return np.rint(price/.6658)*.6658


def load_data():
    train = pd.read_csv(DATA/'train_v9rqX0R.csv')
    test = pd.read_csv(DATA/'test_AbJTz2l.csv')
    allx = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    allx['price'] = recover_price(allx.Item_MRP)
    allx['mrp_noise'] = allx.Item_MRP-allx.price
    allx['weight'] = allx.Item_Weight.fillna(
        allx.groupby('Item_Identifier').Item_Weight.transform('median'))
    allx['weight'] = allx.weight.fillna(allx.weight.median())
    allx['fat'] = allx.Item_Fat_Content.replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})
    allx['family'] = allx.Item_Identifier.str[:2]
    allx.loc[allx.family.eq('NC'), 'fat'] = 'Non-Edible'
    allx['id_letter'] = allx.Item_Identifier.str[2]
    allx['id_number'] = allx.Item_Identifier.str[3:].astype(int)
    allx['visibility_zero'] = allx.Item_Visibility.eq(0).astype(float)
    positive = allx.Item_Visibility.replace(0, np.nan)
    allx['item_visibility'] = positive.groupby(allx.Item_Identifier).transform('median')
    allx['item_visibility'] = allx.item_visibility.fillna(positive.median())
    allx['visibility_ratio'] = allx.Item_Visibility/allx.item_visibility
    allx['price_band'] = np.digitize(allx.price, [69,136,203]).astype(str)
    allx['assortment_size'] = allx.groupby('Item_Identifier').Outlet_Identifier.transform('count')
    cats = ['Outlet_Identifier','Outlet_Type','Item_Type','family','fat','id_letter','price_band']
    nums = ['price','weight','Item_Visibility','item_visibility','visibility_ratio',
            'visibility_zero','mrp_noise','id_number','assortment_size']
    features = allx[cats+nums].copy()
    for c in cats:
        features[c] = features[c].astype(str)
    y = train[TARGET].to_numpy(float)
    price = allx.price.to_numpy(float)
    units = y/price[:len(train)]
    audit = {
        'train_rows':len(train), 'test_rows':len(test),
        'overlapping_item_outlet_pairs':len(train[KEYS].merge(test[KEYS], on=KEYS)),
        'maximum_integer_quantity_error':float(abs(units-np.rint(units)).max()),
        'all_price_candidates_unique':True,
        'feature_policy':'Covariate-only item summaries use supplied train/test features. No test targets.',
        'input_sha256':{f:hashlib.sha256((DATA/f).read_bytes()).hexdigest() for f in
                        ['train_v9rqX0R.csv','test_AbJTz2l.csv','sample_submission_8RXa3c6.csv']},
    }
    return train,test,features.iloc[:len(train)].copy(),features.iloc[len(train):].copy(),y,price,cats,nums,audit


def baseline(fit, apply, y, price, power=2):
    outlet=fit.Outlet_Identifier.to_numpy()
    weight=(price/150.)**power
    quantity=y/price
    means={o:np.average(quantity[outlet==o], weights=weight[outlet==o]) for o in np.unique(outlet)}
    return apply.Outlet_Identifier.map(means).to_numpy(float)


CONFIGS = [
    {'name':'raw_outlet_mrp','kind':'baseline_raw','power':2},
    {'name':'clean_outlet_price','kind':'baseline','power':2},
    {'name':'clean_outlet_units','kind':'baseline','power':0},
    *[{'name':f'spline_{a}_p{p}','kind':'spline','alpha':a,'power':p}
      for a in [30,300,3000] for p in [0,2]],
    *[{'name':f'cat_units_d{d}_p{p}','kind':'cat','depth':d,'power':p}
      for d in [2,4] for p in [0,2]],
    *[{'name':f'lgb_units_l{leaf}_p{p}','kind':'lgb','leaves':leaf,'power':p}
      for leaf in [4,10] for p in [0,2]],
    {'name':'ebm_units_additive','kind':'ebm','power':2,'interactions':0},
    {'name':'ebm_units_interactions','kind':'ebm','power':2,'interactions':5},
    {'name':'ebm_units_unweighted','kind':'ebm','power':0,'interactions':0},
    *[{'name':f'suffix_spline_k{k}_a{a}_p{p}','kind':'suffix_spline','knots':k,'alpha':a,'power':p}
      for k in [10,20,35] for a in [30,300] for p in [0,2]],
    *[{'name':f'suffix_table_a{a}_p{p}','kind':'suffix_table','alpha':a,'power':p}
      for a in [10,50,200] for p in [0,2]],
    *[{'name':f'scaled_lgb_l{leaf}_p{p}','kind':'lgb','leaves':leaf,'power':p,'scaled':True}
      for leaf in [4,10] for p in [0,2]],
    *[{'name':f'scaled_cat_d{d}_p{p}','kind':'cat','depth':d,'power':p,'scaled':True}
      for d in [2,4] for p in [0,2]],
]


def fit_predict(config, Xfit, Xapply, yfit, pfit, papply, cats, nums, seed):
    kind=config['kind']; power=config['power']
    if kind=='baseline_raw':
        mfit=pfit+Xfit.mrp_noise.to_numpy()
        mapply=papply+Xapply.mrp_noise.to_numpy()
        return baseline(Xfit,Xapply,yfit,mfit,power)*mapply
    if kind=='baseline':
        return baseline(Xfit,Xapply,yfit,pfit,power)*papply
    # Center at fold-specific outlet demand. Model only its residual.
    fit_mean=baseline(Xfit,Xfit,yfit,pfit,power)
    apply_mean=baseline(Xfit,Xapply,yfit,pfit,power)
    quantity=yfit/pfit
    weights=(pfit/150.)**power
    z=quantity-fit_mean
    apply_scale=np.ones(len(Xapply))
    if config.get('scaled',False) or kind.startswith('suffix'):
        fit_scale=fit_mean/16
        apply_scale=apply_mean/16
        z=z/fit_scale
        weights=weights*fit_scale**2
    if kind.startswith('suffix'):
        if kind=='suffix_spline':
            id_transform=SplineTransformer(n_knots=config['knots'],degree=3,include_bias=False)
        else:
            id_transform=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
        ct=ColumnTransformer([
            ('id',id_transform,['id_number']),
            ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['family','fat','Item_Type']),
        ])
        model=make_pipeline(ct,StandardScaler(),Ridge(alpha=config['alpha']))
        model.fit(Xfit,z,ridge__sample_weight=weights)
    elif kind=='spline':
        ct=ColumnTransformer([
            ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats),
            ('spline',SplineTransformer(n_knots=5,degree=3,include_bias=False,extrapolation='linear'),nums),
        ])
        model=make_pipeline(ct,StandardScaler(),Ridge(alpha=config['alpha']))
        model.fit(Xfit,z,ridge__sample_weight=weights)
    elif kind=='cat':
        model=CatBoostRegressor(iterations=500,depth=config['depth'],learning_rate=.025,
            l2_leaf_reg=30,loss_function='RMSE',cat_features=cats,
            random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False)
        model.fit(Xfit,z,sample_weight=weights)
    elif kind=='lgb':
        ct=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)],remainder='passthrough')
        model=make_pipeline(ct,LGBMRegressor(n_estimators=250,learning_rate=.02,num_leaves=config['leaves'],
                min_child_samples=100,reg_lambda=100,verbosity=-1,n_jobs=4,random_state=seed))
        model.fit(Xfit,z,lgbmregressor__sample_weight=weights)
    elif kind=='ebm':
        model=ExplainableBoostingRegressor(feature_types=['nominal']*len(cats)+['continuous']*len(nums),
            interactions=config['interactions'],max_bins=64,max_interaction_bins=16,
            outer_bags=4,max_rounds=1800,smoothing_rounds=500,early_stopping_rounds=80,
            min_samples_leaf=40,reg_lambda=50,n_jobs=4,random_state=seed)
        model.fit(Xfit,z,sample_weight=weights)
    else:
        raise ValueError(kind)
    return np.maximum(0,(apply_mean+model.predict(Xapply)*apply_scale)*papply)


def run(configs, seed, phase):
    train,test,X,Xtest,y,price,cats,nums,audit=load_data()
    p=price[:len(train)];pt=price[len(train):]
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
    folds=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=seed).split(X,train.Outlet_Identifier))
    fold_id=np.empty(len(train),dtype=int)
    for k,(_,va) in enumerate(folds): fold_id[va]=k
    rows=[]
    for config in configs:
        start=time.monotonic();name=config['name'];path=ROOT/f'{phase}_{name}.npz'
        if path.exists():
            z=np.load(path);oof=z['oof'];tp=z['test']
        else:
            oof=np.zeros(len(train));tp=np.zeros(len(test))
            for k,(fi,va) in enumerate(folds):
                apply=pd.concat([X.iloc[va],Xtest],ignore_index=True)
                pred=fit_predict(config,X.iloc[fi],apply,y[fi],p[fi],np.r_[p[va],pt],cats,nums,seed+k)
                oof[va]=pred[:len(va)];tp+=pred[len(va):]/len(folds)
            np.savez(path,oof=oof,test=tp,fold_id=fold_id)
        result={'name':name,'oof_rmse':rmse(y,oof),'seconds':time.monotonic()-start,
                **{f'fold_{k}':rmse(y[fold_id==k],oof[fold_id==k]) for k in range(5)}}
        rows.append(result)
        print(json.dumps(result),flush=True)
        pd.DataFrame(rows).sort_values('oof_rmse').to_csv(ROOT/f'{phase}_results.csv',index=False)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',default='screen')
    parser.add_argument('--names',default='')
    parser.add_argument('--seed',type=int,default=713)
    args=parser.parse_args()
    configs=[c for c in CONFIGS if not args.names or c['name'] in args.names.split(',')]
    run(configs,args.seed,args.phase)
