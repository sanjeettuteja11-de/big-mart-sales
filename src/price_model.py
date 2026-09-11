"""Decode the consistent unit-price lattice using training sales only.

The fold's sales, rounded to 1e-4, have a common currency increment. For each
MRP, test the 41 offsets on [-2,2] at steps of 0.1. If exactly one adjusted
price lies on that lattice, use it; otherwise retain MRP. The increment is
learned inside fit, so no held-out sales enter prediction. The bounded offset
menu is a data-inspired hypothesis and is judged on paired CV, not assumed
universal retail behavior.
"""
import numpy as np
from src.structure import HierarchicalStoreRate


class LatticeStoreRate(HierarchicalStoreRate):
    def __init__(self, smoothing=55.234948951954145, rate_smoothing=0.0,
                 weight_power=0.0, integer_units=False, interaction_smoothing=float('inf')):
        super().__init__(smoothing, rate_smoothing, weight_power, integer_units, interaction_smoothing)

    def decode_price(self, X):
        mrp=X.Item_MRP.to_numpy(dtype=float)
        candidates=mrp[:,None]+np.arange(-20,21)[None,:]/10
        if not 0.1 <= self.price_increment_ <= 5.0:
            return mrp
        compatible=np.isclose(candidates/self.price_increment_,np.round(candidates/self.price_increment_),rtol=0,atol=1e-7)
        unique=compatible.sum(axis=1)==1
        return np.where(unique,candidates[np.arange(len(mrp)),compatible.argmax(axis=1)],mrp)

    def fit(self,X,y,sample_weight=None):
        y=np.asarray(y,dtype=float)
        self.price_increment_=float(np.gcd.reduce(np.rint(y*10000).astype(np.int64)))/10000
        price=self.decode_price(X)
        super().fit(X.assign(Item_MRP=price),y/price,sample_weight)
        return self

    def predict(self,X):
        price=self.decode_price(X)
        return price*super().predict(X.assign(Item_MRP=price))
