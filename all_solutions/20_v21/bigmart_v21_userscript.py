"""
BigMart v21 — CatBoost + Stacking + Interaction TE + Multi-seed
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

SEED = 42
NF = 5

# ---------- 1. Load ----------
train = pd.read_csv('train_v9rqX0R.csv')
test  = pd.read_csv('test_AbJTz2l.csv')

# ---------- 2. Clean ----------
fat_map = {'low fat':'Low Fat','LF':'Low Fat','reg':'Regular',
           'Low Fat':'Low Fat','Regular':'Regular'}
for df in (train, test):
    df['Item_Fat_Content'] = df['Item_Fat_Content'].map(fat_map)

size_map = {'Grocery Store':'Small','Supermarket Type1':'Small',
            'Supermarket Type2':'Medium','Supermarket Type3':'Medium'}
for df in (train, test):
    df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Type'].map(size_map))

iw_med  = train.groupby('Item_Identifier')['Item_Weight'].median()
iw_glob = train['Item_Weight'].median()
for df in (train, test):
    df['Item_Weight'] = (df['Item_Weight']
                         .fillna(df['Item_Identifier'].map(iw_med))
                         .fillna(iw_glob))

vis_med  = train.loc[train['Item_Visibility']>0].groupby('Item_Type')['Item_Visibility'].median()
vis_glob = train.loc[train['Item_Visibility']>0, 'Item_Visibility'].median()
for df in (train, test):
    mask = df['Item_Visibility'] == 0
    df.loc[mask, 'Item_Visibility'] = (df.loc[mask, 'Item_Type']
                                       .map(vis_med).fillna(vis_glob))

# ---------- 3. Feature engineering ----------
for df in (train, test):
    df['Outlet_Age']       = 2013 - df['Outlet_Establishment_Year']
    df['Item_Category']    = df['Item_Identifier'].str[:2]
    df['Item_Num']         = df['Item_Identifier'].str.extract(r'(\d+)').astype(int)
    df['is_OUT027']        = (df['Outlet_Identifier']=='OUT027').astype(int)
    df['is_Grocery']       = (df['Outlet_Type']=='Grocery Store').astype(int)
    df['MRP_Bin']          = pd.cut(df['Item_MRP'],
                                    bins=[0,70,140,200,270,10000],
                                    labels=[0,1,2,3,4]).astype(int)
    df['Visibility_MRP']   = df['Item_Visibility'] * df['Item_MRP']
    df['MRP_per_Weight']   = df['Item_MRP'] / df['Item_Weight']
    # interaction keys for TE
    df['OT_Cat']           = df['Outlet_Type'].astype(str) + '_' + df['Item_Category'].astype(str)
    df['OT_Size']          = df['Outlet_Type'].astype(str) + '_' + df['Outlet_Size'].astype(str)
    df['OId_ICat']         = df['Outlet_Identifier'].astype(str) + '_' + df['Item_Category'].astype(str)

# ---------- 4. Target encoding with counts ----------
def target_encode(tr, te, col, target, n_splits=NF, smoothing=10):
    gmean = tr[target].mean()
    agg = tr.groupby(col)[target].agg(['mean','count'])
    sm  = (agg['mean']*agg['count'] + gmean*smoothing) / (agg['count'] + smoothing)
    te[f'{col}_TE']  = te[col].map(sm).fillna(gmean)
    te[f'{col}_CNT'] = te[col].map(agg['count']).fillna(0)
    tr[f'{col}_CNT'] = tr[col].map(agg['count']).fillna(0)

    tr[f'{col}_TE'] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr_idx, val_idx in kf.split(tr):
        fold = tr.iloc[tr_idx]
        a = fold.groupby(col)[target].agg(['mean','count'])
        s = (a['mean']*a['count'] + gmean*smoothing) / (a['count'] + smoothing)
        tr.loc[tr.index[val_idx], f'{col}_TE'] = tr.iloc[val_idx][col].map(s)
    tr[f'{col}_TE'] = tr[f'{col}_TE'].fillna(gmean)
    return tr, te

te_cols = ['Outlet_Identifier','Item_Type','Outlet_Type','Item_Category',
           'Outlet_Size','Outlet_Location_Type','OT_Cat','OT_Size','OId_ICat','Outlet_Age']
for c in te_cols:
    train, test = target_encode(train, test, c, 'Item_Outlet_Sales')

# ---------- 5. Encode ----------
cat_cols = ['Item_Fat_Content','Outlet_Size','Outlet_Location_Type',
            'Outlet_Type','Item_Type','Item_Category','Outlet_Identifier',
            'OT_Cat','OT_Size','OId_ICat']
y = train['Item_Outlet_Sales'].astype(float).values

combined = pd.concat([train.drop(columns=['Item_Outlet_Sales']), test], axis=0)
combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

n_tr = len(train)
tr_e = combined.iloc[:n_tr].reset_index(drop=True)
te_e = combined.iloc[n_tr:].reset_index(drop=True)

drop_ids = ['Item_Identifier','Outlet_Identifier']
feat = [c for c in tr_e.columns if c not in drop_ids and tr_e[c].dtype != 'object']
X      = tr_e[feat].astype(float).values
X_test = te_e[feat].astype(float).values
print(f"Features: {len(feat)}")

# ---------- 6. Models ----------
kf = KFold(n_splits=NF, shuffle=True, random_state=SEED)
oofs, tests = {}, {}

def run(name, model_factory, fit_kwargs=None):
    oof  = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    for tr, val in kf.split(X):
        m = model_factory()
        if fit_kwargs:
            m.fit(X[tr], y[tr], **{**fit_kwargs, 'eval_set':[(X[val], y[val])]})
        else:
            m.fit(X[tr], y[tr])
        oof[val] = m.predict(X[val])
        pred    += m.predict(X_test) / NF
    r = np.sqrt(mean_squared_error(y, oof))
    print(f"{name:10s} OOF RMSE: {r:.4f}")
    oofs[name]  = oof
    tests[name] = pred

# LGB seed A
run('lgbA', lambda: lgb.LGBMRegressor(
        objective='regression', metric='rmse', learning_rate=0.02,
        num_leaves=63, max_depth=8, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        n_estimators=3000, random_state=SEED, verbose=-1),
    fit_kwargs=dict(callbacks=[lgb.early_stopping(200, verbose=False)]))

# LGB seed B
run('lgbB', lambda: lgb.LGBMRegressor(
        objective='regression', metric='rmse', learning_rate=0.03,
        num_leaves=127, max_depth=10, min_child_samples=40,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
        n_estimators=3000, random_state=SEED+1, verbose=-1),
    fit_kwargs=dict(callbacks=[lgb.early_stopping(200, verbose=False)]))

# XGB
run('xgb', lambda: xgb.XGBRegressor(
        objective='reg:squarederror', learning_rate=0.02, max_depth=7,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, n_estimators=3000,
        random_state=SEED, verbosity=0, early_stopping_rounds=200),
    fit_kwargs=dict(verbose=False))

# CatBoost — the missing piece
run('cb', lambda: CatBoostRegressor(
        iterations=3000, learning_rate=0.03, depth=6,
        l2_leaf_reg=3, random_seed=SEED, verbose=0,
        early_stopping_rounds=200),
    fit_kwargs=dict())

# ---------- 7. Stack ----------
names = ['lgbA','lgbB','xgb','cb']
S  = np.column_stack([oofs[n]  for n in names])
ST = np.column_stack([tests[n] for n in names])

meta_oof = np.zeros(len(X))
for tr, val in kf.split(S):
    ridge = Ridge(alpha=1.0).fit(S[tr], y[tr])
    meta_oof[val] = ridge.predict(S[val])
print(f"STACKED    OOF RMSE: {np.sqrt(mean_squared_error(y, meta_oof)):.4f}")

ridge = Ridge(alpha=1.0).fit(S, y)
final = np.clip(ridge.predict(ST), 33.29, 13087)

# ---------- 8. Save ----------
sub = pd.DataFrame({
    'Item_Identifier':   test['Item_Identifier'].values,
    'Outlet_Identifier': test['Outlet_Identifier'].values,
    'Item_Outlet_Sales': final
})
sub.to_csv('submission_v21.csv', index=False)
print("Saved submission_v21.csv")
print(sub['Item_Outlet_Sales'].describe())
