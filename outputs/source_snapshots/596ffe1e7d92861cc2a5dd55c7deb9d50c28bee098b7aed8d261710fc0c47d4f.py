"""Reusable boosting estimator on decoded unit prices, with internal validation.

All cleaning and price-lattice learning is repeated within the early-stopping
training split. The outer CV passes raw features to this regressor.
"""
import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.model_selection import train_test_split
from src.config import TARGET,OUTLET_ID
from src.price_model import LatticeStoreRate


class LatticeBoostRegressor(RegressorMixin,BaseEstimator):
    def __init__(self,library='catboost',params=None,weighted_training=False,
                 sales_early_stopping=True,seed=42):
        self.library=library; self.params=params; self.weighted_training=weighted_training
        self.sales_early_stopping=sales_early_stopping; self.seed=seed

    def _prepare(self,train,others):
        from src.cv import fold_matrices
        from src.models import SPECS_BY_NAME
        decoder=LatticeStoreRate()
        decoder.price_increment_=float(np.gcd.reduce(np.rint(train[TARGET].to_numpy()*10000).astype(np.int64)))/10000
        decoded=[f.assign(Item_MRP=decoder.decode_price(f)) for f in [train,*others]]
        spec=SPECS_BY_NAME[self.library+'_units_unweighted']
        return fold_matrices(spec,decoded[0],decoded[1:],seed=self.seed)

    def _model(self,rounds=None):
        from src.models import SPECS_BY_NAME
        spec=SPECS_BY_NAME[self.library+'_units_unweighted']
        p=dict(self.params or {})
        if rounds is not None: p[spec.iterations_param]=rounds
        model=spec.make(p,self.seed)
        model.set_params(**({'thread_count':2} if self.library=='catboost' else {'n_jobs':2}))
        return model

    def fit(self,X,y,sample_weight=None):
        from src.cv import _catboost_cat_features
        raw=X.assign(**{TARGET:np.asarray(y,dtype=float)})
        a,b=train_test_split(np.arange(len(raw)),test_size=.15,random_state=self.seed,stratify=raw[OUTLET_ID])
        ra,(rb,),xa,(xb,)=self._prepare(raw.iloc[a],[raw.iloc[b]])
        za=ra[TARGET].to_numpy()/ra.Item_MRP.to_numpy(); zb=rb[TARGET].to_numpy()/rb.Item_MRP.to_numpy()
        wa=ra.Item_MRP.to_numpy()**2 if self.weighted_training else None
        if wa is not None:wa/=wa.mean()
        wb=rb.Item_MRP.to_numpy()**2 if self.sales_early_stopping else None
        if wb is not None:wb/=wb.mean()
        m=self._model()
        if self.library=='catboost':
            from catboost import Pool
            cat=_catboost_cat_features(xa)
            m.fit(xa,za,sample_weight=wa,cat_features=cat,eval_set=Pool(xb,zb,weight=wb,cat_features=cat),
                  early_stopping_rounds=50,verbose=False)
            rounds=int(m.get_best_iteration())+1
        elif self.library=='lightgbm':
            import lightgbm as lgb
            m.fit(xa,za,sample_weight=wa,eval_X=xb,eval_y=zb,eval_sample_weight=None if wb is None else [wb],
                  callbacks=[lgb.early_stopping(50,verbose=False)])
            rounds=int(m.best_iteration_)
        else:
            m.set_params(early_stopping_rounds=50)
            m.fit(xa,za,sample_weight=wa,eval_set=[(xb,zb)],sample_weight_eval_set=None if wb is None else [wb],verbose=False)
            rounds=int(m.best_iteration)+1
        self.best_iteration_count_=rounds
        self.raw_train_=raw
        r,_,xt,_=self._prepare(raw,[])
        w=r.Item_MRP.to_numpy()**2 if self.weighted_training else None
        if w is not None:w/=w.mean()
        self.model_=self._model(rounds)
        kwargs={'sample_weight':w}
        if self.library=='catboost':kwargs['cat_features']=_catboost_cat_features(xt)
        self.model_.fit(xt,r[TARGET].to_numpy()/r.Item_MRP.to_numpy(),**kwargs)
        return self

    def predict(self,X):
        _,(r,),_,(xt,)=self._prepare(self.raw_train_,[X])
        return self.model_.predict(xt)*r.Item_MRP.to_numpy()
