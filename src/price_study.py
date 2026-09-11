"""Paired check of price decoding, without rounding predicted units."""
import json
from pathlib import Path
import numpy as np
from src.models import ModelSpec,SPECS_BY_NAME,load_tuned
from src.price_model import LatticeStoreRate
from src.data import load_raw
from src.artifacts import provenance,save_result,sha256
from src.cv import run_cv,rmse
from src.continue_study import DEST
from src.config import TARGET


def main():
    tr,te=load_raw(); y=tr[TARGET].to_numpy()
    spec=ModelSpec('lattice_store_rate','structural','raw',lambda p,s:LatticeStoreRate(**p),{})
    out=DEST/'price'; out.mkdir(exist_ok=True)
    prov=provenance(tr,te); prov['price_model_sha256']=sha256(Path(__file__).with_name('price_model.py'))
    rows=[]
    for i,p in enumerate([{}, {'smoothing':float('inf')}, {'rate_smoothing':500},
                          {'smoothing':150}, {'weight_power':1}, {'weight_power':2}]):
        for seed in ([42,137,2026] if i==0 else [42]):
            r=run_cv(spec,tr,te,params=p,seed=seed)
            rows.append(save_result(r,out/f'candidate_{i}_seed_{seed}',
                                    {'params':{**LatticeStoreRate().get_params(),**p},'seed':seed,'repeats':3},prov))
            print('price',i,seed,r.oof_rmse,flush=True)
    model=LatticeStoreRate().fit(tr.assign(MRP_Band=0),y)
    price=model.decode_price(tr)
    audit={'increment':model.price_increment_,'integer_units_fraction':float(np.isclose(y/price,np.round(y/price)).mean()),
           'train_unique_prices_per_product_max':int(tr.assign(decoded_price=price.round(4)).groupby('Item_Identifier').decoded_price.nunique().max()),
           'runs':rows}
    (out/'summary.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':main()
