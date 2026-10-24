"""Minimal, reproducible v68/v69 correction from raw training data.

The old category-containing model and new suffix-only model are fitted on the
same 15 training folds. Only their difference is applied to frozen v67. No new
public-score forecast is computed for these previously unscored directions.
"""
from pathlib import Path
import argparse, hashlib, json, platform
import numpy as np
import pandas as pd
import scipy, sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parent
DATA=Path('/Users/abhimanyunain/Downloads')
SEEDS=(713,2718,9317);ALPHA=1.135112248149756
TARGET='Item_Outlet_Sales';KEYS=['Item_Identifier','Outlet_Identifier']
STRENGTHS={'v68':1.0,'v69':1.5}
BASE_SHA='f1052d3aa5f63bef2972f729fe8c0bb7c8529a251bcb5897c50cb1630649f298'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(frame):
    candidates=frame.Item_MRP.to_numpy()[:,None]+.1*np.arange(-20,21)[None,:]
    errors=abs(candidates/.6658-np.rint(candidates/.6658))
    assert np.all((errors<1e-7).sum(axis=1)==1)
    p=np.rint(candidates[np.arange(len(frame)),errors.argmin(axis=1)]/.6658)*.6658
    x=frame[['Outlet_Identifier','Item_Type']].copy()
    x['id_number']=frame.Item_Identifier.str[3:].astype(int)
    x['family']=frame.Item_Identifier.str[:2]
    x['fat']=frame.Item_Fat_Content.replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})
    x.loc[x.family.eq('NC'),'fat']='Non-Edible'
    return p,x


def fit_predict(fit,apply,y,p,pt,controls=True):
    quantity=y/p
    means=pd.DataFrame({'o':fit.Outlet_Identifier.to_numpy(),'q':quantity}).groupby('o').q.mean()
    m=fit.Outlet_Identifier.map(means).to_numpy(float)
    mt=apply.Outlet_Identifier.map(means).to_numpy(float)
    s=m/16.;st=mt/16.
    transformations=[('id',SplineTransformer(n_knots=20,degree=3,include_bias=False),['id_number'])]
    if controls:
        transformations.append(('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),
                                 ['family','fat','Item_Type']))
    model=make_pipeline(ColumnTransformer(transformations),StandardScaler(),Ridge(alpha=300))
    model.fit(fit,(quantity-m)/s,ridge__sample_weight=s**2)
    return np.maximum(0,(mt+model.predict(apply)*st)*pt)


def main(data_dir=DATA,output_dir=DATA):
    paths={k:data_dir/f for k,f in {
        'train':'train_v9rqX0R.csv','test':'test_AbJTz2l.csv',
        'sample':'sample_submission_8RXa3c6.csv',
        'baseline':'bigmart_submission_v67_exact_quantity_optimum.csv'}.items()}
    assert digest(paths['baseline'])==BASE_SHA
    train=pd.read_csv(paths['train']);test=pd.read_csv(paths['test'])
    sample=pd.read_csv(paths['sample']);base=pd.read_csv(paths['baseline'])
    assert train.shape==(8523,12) and test.shape==(5681,11)
    assert sample.columns.tolist()==base.columns.tolist()==KEYS+[TARGET]
    assert base[KEYS].equals(sample[KEYS]) and test[KEYS].equals(sample[KEYS])
    assert not train[KEYS].duplicated().any() and not test[KEYS].duplicated().any()
    assert train[KEYS].merge(test[KEYS],on=KEYS).empty
    p,x=features(train);pt,xt=features(test);y=train[TARGET].to_numpy(float)
    assert abs(y/p-np.rint(y/p)).max()<1e-8
    deltas=[];oofs=[]
    for seed in SEEDS:
        td=np.zeros(len(test));vd=np.zeros(len(train))
        for fi,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(x,train.Outlet_Identifier):
            apply=pd.concat([x.iloc[va],xt],ignore_index=True);prices=np.r_[p[va],pt]
            old=fit_predict(x.iloc[fi],apply,y[fi],p[fi],prices,controls=True)
            new=fit_predict(x.iloc[fi],apply,y[fi],p[fi],prices,controls=False)
            d=ALPHA*(new-old);vd[va]=d[:len(va)];td+=d[len(va):]/5
        deltas.append(td);oofs.append(vd)
    delta=np.mean(deltas,axis=0)
    output_dir.mkdir(parents=True,exist_ok=True)
    manifest={'inputs':{k:{'path':str(v),'sha256':digest(v)} for k,v in paths.items()},
        'recipe':{'seeds':SEEDS,'folds':5,'spline_knots':20,'ridge_alpha':300,
                  'previous_correction_alpha':ALPHA,'removal_strengths':STRENGTHS,
                  'formula':'v67 + strength * alpha * (suffix_only - suffix_with_category_controls)'},
        'public_rmse':None,'output':{},
        'limitations':['Public scores cannot be determined before AV evaluates this new direction.',
            'Training CV was used for model selection and is not an untouched external test.',
            'Baseline uses previous public feedback; private score remains unknown.',
            'Historical v63 ensemble is a frozen input, not retrained by this source.'],
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                    'scipy':scipy.__version__,'sklearn':sklearn.__version__}}
    for version,strength in STRENGTHS.items():
        prediction=base[TARGET].to_numpy()+strength*delta
        # Preserve the affine line for a future exact score calibration.
        assert np.isfinite(prediction).all() and (prediction>=0).all()
        out=sample.copy();out[TARGET]=prediction
        path=output_dir/f'bigmart_submission_{version}_denoised_quantity.csv'
        out.to_csv(path,index=False);saved=pd.read_csv(path)
        assert saved.shape==(5681,3) and saved[KEYS].equals(test[KEYS])
        assert np.allclose(saved[TARGET],prediction,atol=1e-9,rtol=0)
        manifest['output'][version]={'path':str(path),'sha256':digest(path),'rows':len(out),
            'strength':strength,'min':float(prediction.min()),'max':float(prediction.max()),
            'rms_change_from_v67':float(np.sqrt(np.mean((strength*delta)**2)))}
    np.savez(ROOT/'refinement_delta.npz',test=delta,oof=np.mean(oofs,axis=0))
    (ROOT/'refinement_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2));return manifest


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,default=DATA)
    ap.add_argument('--output-dir',type=Path,default=DATA);args=ap.parse_args()
    main(args.data_dir,args.output_dir)
