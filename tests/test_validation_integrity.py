"""Regression tests for fold boundaries and auditable predictions."""
import numpy as np
import pandas as pd
import pytest
from src.cv import run_cv, folds, prepare
from src.models import SPECS_BY_NAME
from src.structure import HierarchicalStoreRate, StoreRatePopularity
from src.config import TARGET
from tests.synthetic import make_synthetic


@pytest.mark.parametrize('name',['store_rate_popularity','ridge_interactions','catboost_raw','lightgbm_units_popularity'])
def test_outer_validation_labels_cannot_influence_predictions(name):
    train,test,_=make_synthetic(n_items=80,seed=18)
    indices=[next(folds('stratified',train,3,1,42))]
    a,b=indices[0]
    altered=train.copy()
    altered.loc[altered.index[b],TARGET]=1e8
    params={'iterations':50} if name.startswith('catboost') else {'n_estimators':50} if name.startswith('lightgbm') else {}
    spec=SPECS_BY_NAME[name]
    r1=run_cv(spec,train,test,params=params,split_indices=indices)
    r2=run_cv(spec,altered,test,params=params,split_indices=indices)
    np.testing.assert_array_equal(r1.oof[b],r2.oof[b])
    np.testing.assert_array_equal(r1.test,r2.test)


def test_test_features_cannot_change_oof_or_training_vocabulary():
    train,test,_=make_synthetic(n_items=80,seed=5)
    other=test.assign(Item_MRP=1e7,Item_Weight=1e5,Item_Visibility=1e4,Outlet_Size='UNSEEN')
    spec=SPECS_BY_NAME['ridge_interactions']
    a=run_cv(spec,train,test,n_splits=2,n_repeats=1)
    b=run_cv(spec,train,other,n_splits=2,n_repeats=1)
    np.testing.assert_array_equal(a.oof,b.oof)
    p=prepare(train,other,matrices=['tree','linear'])
    assert 'UNSEEN' not in p.matrices['tree'][0].Outlet_Size.cat.categories
    assert 'Outlet_Size_UNSEEN' not in p.matrices['linear'][0].columns


def test_saved_folds_reconstruct_oof_and_clipping_metric():
    train,test,_=make_synthetic(n_items=60,seed=3)
    r=run_cv(SPECS_BY_NAME['ridge_interactions'],train,test,n_splits=3,n_repeats=2)
    sums=np.zeros(len(train)); counts=np.zeros(len(train))
    for (_,va),pred in zip(r.split_indices,r.fold_predictions):
        sums[va]+=pred; counts[va]+=1
    np.testing.assert_array_equal(counts,np.full(len(train),2))
    np.testing.assert_allclose(r.oof,sums/counts)
    assert r.oof_rmse <= r.raw_oof_rmse+1e-10


def test_hierarchical_rate_limits_and_unseen_store_fallback():
    X=pd.DataFrame({'Outlet_Identifier':['A','A','B','B'], 'Outlet_Type':['T']*4,
                    'Item_Identifier':['x','y','x','y'],'Item_MRP':[10.]*4,'MRP_Band':[0]*4})
    y=np.array([12,8,6,4])
    a=HierarchicalStoreRate(smoothing=55,rate_smoothing=0).fit(X,y)
    b=StoreRatePopularity(smoothing=55).fit(X,y)
    np.testing.assert_allclose(a.predict(X),b.predict(X))
    pooled=HierarchicalStoreRate(smoothing=np.inf,rate_smoothing=np.inf).fit(X,y)
    np.testing.assert_allclose(pooled.predict(X),np.full(4,7.5))
    np.testing.assert_allclose(pooled.predict(X.assign(Outlet_Identifier='unseen')),np.full(4,7.5))


def test_price_lattice_recovers_unique_feature_price_and_ignores_heldout_labels():
    from src.price_model import LatticeStoreRate
    from src.features import MRP_BAND_EDGES
    train,test,_=make_synthetic(n_items=80,seed=4)
    rng=np.random.default_rng(4)
    prices=rng.integers(50,380,len(train))*.6658
    units=rng.integers(1,40,len(train))
    frame=train.assign(Item_MRP=prices+rng.integers(-20,21,len(train))*.1,
                       MRP_Band=0,Item_Outlet_Sales=units*prices)
    m=LatticeStoreRate().fit(frame,frame[TARGET])
    assert m.price_increment_ == pytest.approx(.6658)
    np.testing.assert_allclose(m.decode_price(frame),prices,atol=1e-8)
    assert np.isfinite(m.predict(frame)).all()
    # Predictions use features only.
    np.testing.assert_array_equal(m.predict(frame),m.predict(frame.assign(**{TARGET:1e8})))


def test_booster_price_feature_is_crossfit():
    from src.lattice_boost import LatticeBoostRegressor
    train,_,_=make_synthetic(n_items=60,seed=5)
    prices=np.round(train.Item_MRP/.6658)*.6658
    train=train.assign(Item_MRP=prices,Item_Outlet_Sales=prices*10)
    model=LatticeBoostRegressor()
    a,_,_,_=model._prepare(train,[])
    changed=train.copy();changed.loc[changed.index[0],TARGET]+=0.0001
    b,_,_,_=model._prepare(changed,[])
    assert a.Item_MRP.iloc[0] == b.Item_MRP.iloc[0]
