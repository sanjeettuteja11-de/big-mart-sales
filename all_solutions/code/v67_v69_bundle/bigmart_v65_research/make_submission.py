"""Train a quantity correction and apply it to the frozen, scored v63 baseline.

Reproduction inputs: raw train/test/sample CSVs and the exact v63 prediction
file identified by SHA-256 below. This reproduces the NEW correction from raw
training data; it does not retrain the historical v63 ensemble.

No leaderboard values are used to fit the correction or select its magnitude.
Settings were chosen on training-data CV and frozen before verification runs.
The generated files have no claimed or estimated public RMSE.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

ROOT=Path(__file__).resolve().parent
DEFAULT_DATA=Path('/Users/abhimanyunain/Downloads')
EXPECTED_BASE_SHA256='9341aaf1f70110d7759c35598932b5cb15c0c4bfec7bd7a48d98b6658ad438cd'
TARGET='Item_Outlet_Sales'
KEYS=['Item_Identifier','Outlet_Identifier']
SEEDS=(713,2718,9317)
WEIGHTS={'v65':0.75,'v66':1.0}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rmse(a,b):
    return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))


def price_and_features(frame):
    # MRP equals a product-specific multiple of .6658 plus a tenths-valued
    # perturbation in [-2,2]. Exactly one candidate exists for every supplied row.
    candidates=frame.Item_MRP.to_numpy()[:,None]+.1*np.arange(-20,21)[None,:]
    errors=abs(candidates/.6658-np.rint(candidates/.6658))
    if not np.all((errors<1e-7).sum(axis=1)==1):
        raise ValueError('Non-unique price recovery; dataset differs from audited input')
    price=np.rint(candidates[np.arange(len(frame)),errors.argmin(axis=1)]/.6658)*.6658
    x=frame[['Outlet_Identifier','Item_Type']].copy()
    x['id_number']=frame.Item_Identifier.str[3:].astype(int)
    x['family']=frame.Item_Identifier.str[:2]
    x['fat']=frame.Item_Fat_Content.replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})
    x.loc[x.family.eq('NC'),'fat']='Non-Edible'
    return price,x


def fit_correction(xfit,xapply,yfit,pfit,papply):
    quantity=yfit/pfit
    means=pd.DataFrame({'outlet':xfit.Outlet_Identifier.to_numpy(),'q':quantity}).groupby('outlet').q.mean()
    fit_mean=xfit.Outlet_Identifier.map(means).to_numpy(float)
    apply_mean=xapply.Outlet_Identifier.map(means).to_numpy(float)
    fit_scale=fit_mean/16.
    apply_scale=apply_mean/16.
    centered=(quantity-fit_mean)/fit_scale
    features=ColumnTransformer([
        ('id',SplineTransformer(n_knots=20,degree=3,include_bias=False),['id_number']),
        ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['family','fat','Item_Type']),
    ])
    model=make_pipeline(features,StandardScaler(),Ridge(alpha=300))
    model.fit(xfit,centered,ridge__sample_weight=fit_scale**2)
    neutral=apply_mean*papply
    adjusted=np.maximum(0,(apply_mean+model.predict(xapply)*apply_scale)*papply)
    return adjusted-neutral,neutral


def main(data_dir, output_dir):
    paths={
        'train':data_dir/'train_v9rqX0R.csv',
        'test':data_dir/'test_AbJTz2l.csv',
        'sample':data_dir/'sample_submission_8RXa3c6.csv',
        'baseline':data_dir/'bigmart_submission_v63_exact_v60_optimum.csv',
    }
    if digest(paths['baseline'])!=EXPECTED_BASE_SHA256:
        raise ValueError('v63 baseline changed: expected the exact previously scored file')
    train=pd.read_csv(paths['train']);test=pd.read_csv(paths['test'])
    sample=pd.read_csv(paths['sample']);base=pd.read_csv(paths['baseline'])
    assert len(train)==8523 and len(test)==5681
    assert sample.columns.tolist()==KEYS+[TARGET]
    assert base.columns.tolist()==sample.columns.tolist()
    assert test[KEYS].equals(sample[KEYS]) and base[KEYS].equals(sample[KEYS])
    assert not train[KEYS].duplicated().any() and not test[KEYS].duplicated().any()
    assert len(train[KEYS].merge(test[KEYS],on=KEYS))==0
    p,x=price_and_features(train);pt,xt=price_and_features(test)
    y=train[TARGET].to_numpy(float)
    assert abs(y/p-np.rint(y/p)).max()<1e-8
    test_delta=[];oof_delta=[];oof_neutral=[]
    for seed in SEEDS:
        vd=np.zeros(len(train));vn=np.zeros(len(train));td=np.zeros(len(test))
        for fi,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(x,train.Outlet_Identifier):
            xa=pd.concat([x.iloc[va],xt],ignore_index=True)
            delta,neutral=fit_correction(x.iloc[fi],xa,y[fi],p[fi],np.r_[p[va],pt])
            vd[va]=delta[:len(va)];vn[va]=neutral[:len(va)];td+=delta[len(va):]/5
        oof_delta.append(vd);oof_neutral.append(vn);test_delta.append(td)
    correction=np.mean(test_delta,axis=0)
    cv_delta=np.mean(oof_delta,axis=0);cv_neutral=np.mean(oof_neutral,axis=0)
    output_dir.mkdir(parents=True,exist_ok=True)
    manifest={
        'inputs':{name:{'path':str(path),'sha256':digest(path)} for name,path in paths.items()},
        'settings':{'seeds':SEEDS,'folds':5,'spline_knots':20,'ridge_alpha':300,'weights':WEIGHTS},
        'validation':{'quantity_integer_error':float(abs(y/p-np.rint(y/p)).max()),
            'repeated_oof_neutral_rmse':rmse(y,cv_neutral),
            'repeated_oof_quantity_model_rmse':rmse(y,cv_neutral+cv_delta)},
        'public_score':None,
        'limitations':[
            'No hidden test labels are available. Public RMSE is unknown until evaluated by AV.',
            'The frozen baseline was tuned on earlier public feedback; private-set effects are unknown.',
            'CV model selection uses the available training data; this is not an untouched external test.',
            'Only the new correction is retrained from raw data here. Historical baseline is a versioned input.',
        ],
        'versions':{'python':platform.python_version(),'numpy':np.__version__,
            'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'outputs':{},
    }
    for label,weight in WEIGHTS.items():
        pred=np.maximum(0,base[TARGET].to_numpy(float)+weight*correction)
        out=sample.copy();out[TARGET]=pred
        assert out.shape==(5681,3) and np.isfinite(pred).all() and (pred>=0).all()
        path=output_dir/f'bigmart_submission_{label}_quantity_correction.csv'
        out.to_csv(path,index=False)
        saved=pd.read_csv(path)
        assert saved[KEYS].equals(test[KEYS]) and saved.columns.tolist()==sample.columns.tolist()
        assert np.allclose(saved[TARGET],pred,atol=1e-9,rtol=0)
        manifest['outputs'][label]={'path':str(path),'sha256':digest(path),'weight':weight,
            'rms_change_from_v63':rmse(pred,base[TARGET]),'min':float(pred.min()),'max':float(pred.max()),
            'nonnegative_clipped_rows':int((base[TARGET].to_numpy()+weight*correction<0).sum())}
    np.savez(ROOT/'quantity_correction.npz',oof=cv_delta,test=correction,neutral_oof=cv_neutral)
    (ROOT/'submission_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_DATA)
    args=parser.parse_args();main(args.data_dir,args.output_dir)
