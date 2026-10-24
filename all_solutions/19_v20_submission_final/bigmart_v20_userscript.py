"""
BigMart Sales Prediction - Corrected
No leakage (AV split prevents duplicate item-outlet pairs).
Focus: robust FE + target encoding + LGB/XGB ensemble.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ---------- 1. Load ----------
train = pd.read_csv('train_v9rqX0R.csv')
test  = pd.read_csv('test_AbJTz2l.csv')
print("Train:", train.shape, "Test:", test.shape)

# ---------- 2. Clean categoricals ----------
fat_map = {'low fat':'Low Fat','LF':'Low Fat','reg':'Regular',
           'Low Fat':'Low Fat','Regular':'Regular'}
for df in (train, test):
    df['Item_Fat_Content'] = df['Item_Fat_Content'].map(fat_map)

# Outlet_Size imputation from Outlet_Type (test has many NaNs)
size_map = {'Grocery Store':'Small','Supermarket Type1':'Small',
            'Supermarket Type2':'Medium','Supermarket Type3':'Medium'}
train['Outlet_Size'] = train['Outlet_Size'].fillna(train['Outlet_Type'].map(size_map))
test['Outlet_Size']  = test['Outlet_Size'].fillna(test['Outlet_Type'].map(size_map))

# ---------- 3. Item_Weight imputation by Item_Identifier ----------
iw_med = train.groupby('Item_Identifier')['Item_Weight'].median()
iw_glob = train['Item_Weight'].median()
train['Item_Weight'] = np.where(train['Item_Weight'].isna(),
                                train['Item_Identifier'].map(iw_med).fillna(iw_glob),
                                train['Item_Weight'])
test['Item_Weight']  = np.where(test['Item_Weight'].isna(),
                                test['Item_Identifier'].map(iw_med).fillna(iw_glob),
                                test['Item_Weight'])

# ---------- 4. Visibility correction ----------
vis_med = train.loc[train['Item_Visibility']>0].groupby('Item_Type')['Item_Visibility'].median()
vis_glob = train.loc[train['Item_Visibility']>0,'Item_Visibility'].median()
for df in (train, test):
    mask = df['Item_Visibility'] == 0
    df.loc[mask,'Item_Visibility'] = df.loc[mask,'Item_Type'].map(vis_med).fillna(vis_glob)

# ---------- 5. Feature engineering ----------
for df in (train, test):
    df['Outlet_Age']       = 2013 - df['Outlet_Establishment_Year']  # per official problem statement
    df['Item_Category']    = df['Item_Identifier'].str[:2]           # FD, DR, NC
    df['is_OUT027']        = (df['Outlet_Identifier']=='OUT027').astype(int)
    df['is_Grocery']       = (df['Outlet_Type']=='Grocery Store').astype(int)
    df['MRP_Bin']          = pd.cut(df['Item_MRP'],
                                    bins=[0,70,140,200,270,10000],
                                    labels=[0,1,2,3,4]).astype(int)
    df['Visibility_MRP']   = df['Item_Visibility'] * df['Item_MRP']
    df['MRP_per_Weight']   = df['Item_MRP'] / df['Item_Weight']

# ---------- 6. K-fold target encoding (leak-safe) ----------
def target_encode(tr, te, col, target, n_splits=5, smoothing=10):
    gmean = tr[target].mean()
    agg = tr.groupby(col)[target].agg(['mean','count'])
    sm  = (agg['mean']*agg['count'] + gmean*smoothing) / (agg['count']+smoothing)
    te[f'{col}_TE'] = te[col].map(sm).fillna(gmean)

    tr[f'{col}_TE'] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(tr):
        fold = tr.iloc[tr_idx]
        a = fold.groupby(col)[target].agg(['mean','count'])
        s = (a['mean']*a['count'] + gmean*smoothing) / (a['count']+smoothing)
        tr.loc[tr.index[val_idx], f'{col}_TE'] = tr.iloc[val_idx][col].map(s)
    tr[f'{col}_TE'] = tr[f'{col}_TE'].fillna(gmean)
    return tr, te

for col in ['Outlet_Identifier','Item_Type','Outlet_Type','Item_Category',
            'Outlet_Size','Outlet_Location_Type']:
    train, test = target_encode(train, test, col, 'Item_Outlet_Sales')

# ---------- 7. One-hot + align columns ----------
cat_cols = ['Item_Fat_Content','Outlet_Size','Outlet_Location_Type',
            'Outlet_Type','Item_Type','Item_Category','Outlet_Identifier']

y = train['Item_Outlet_Sales'].astype(float).values

combined = pd.concat([train.drop(columns=['Item_Outlet_Sales']), test], axis=0)
combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

n_tr = len(train)
train_enc = combined.iloc[:n_tr].reset_index(drop=True)
test_enc  = combined.iloc[n_tr:].reset_index(drop=True)

drop_ids = ['Item_Identifier','Outlet_Identifier']
feat_cols = [c for c in train_enc.columns
             if c not in drop_ids and train_enc[c].dtype != 'object']

X      = train_enc[feat_cols].astype(float)
X_test = test_enc[feat_cols].astype(float)
print("Features:", len(feat_cols))

# ---------- 8. Train ensemble ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_p = dict(objective='regression', metric='rmse', learning_rate=0.02,
             num_leaves=63, max_depth=8, min_child_samples=20,
             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
             reg_lambda=0.1, n_estimators=3000, random_state=42, verbose=-1)
xgb_p = dict(objective='reg:squarederror', learning_rate=0.02, max_depth=7,
             min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
             reg_alpha=0.1, reg_lambda=1.0, n_estimators=3000,
             random_state=42, verbosity=0, early_stopping_rounds=200)

oof_lgb = np.zeros(len(X))
pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))

for fold, (tr, val) in enumerate(kf.split(X)):
    Xtr, Xval = X.iloc[tr], X.iloc[val]
    ytr, yval = y[tr], y[val]

    m1 = lgb.LGBMRegressor(**lgb_p)
    m1.fit(Xtr, ytr, eval_set=[(Xval,yval)],
           callbacks=[lgb.early_stopping(200, verbose=False)])
    oof_lgb[val] = m1.predict(Xval)
    pred_lgb    += m1.predict(X_test)/5

    m2 = xgb.XGBRegressor(**xgb_p)
    m2.fit(Xtr, ytr, eval_set=[(Xval,yval)], verbose=False)
    pred_xgb += m2.predict(X_test)/5

    print(f"Fold {fold+1}  LGB RMSE: {np.sqrt(mean_squared_error(yval, oof_lgb[val])):.4f}")

print(f"\nOOF LGB RMSE: {np.sqrt(mean_squared_error(y, oof_lgb)):.4f}")

# ---------- 9. Blend + save ----------
final = 0.55*pred_lgb + 0.45*pred_xgb
final = np.clip(final, 33.29, 13087)

sub = pd.DataFrame({
    'Item_Identifier':    test['Item_Identifier'].values,
    'Outlet_Identifier':  test['Outlet_Identifier'].values,
    'Item_Outlet_Sales':  final
})
sub.to_csv('submission_final_v20.csv', index=False)
print("Saved submission_final_v20.csv — rows:", len(sub))
print(sub['Item_Outlet_Sales'].describe())
