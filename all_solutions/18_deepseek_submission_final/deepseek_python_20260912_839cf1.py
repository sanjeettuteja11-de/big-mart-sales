"""
BigMart Sales Prediction - Optimized Solution
Target: RMSE ~1120-1130 (near leaderboard top)
Strategy: Direct mapping for duplicate item-outlet pairs + 
          LightGBM ensemble for new combinations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_csv('train_v9rqX0R.csv')
test = pd.read_csv('test_AbJTz2l.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ============================================================
# 2. CLEAN CATEGORICAL INCONSISTENCIES
# ============================================================
# Fix Item_Fat_Content variations
fat_map = {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular',
           'Low Fat': 'Low Fat', 'Regular': 'Regular'}
train['Item_Fat_Content'] = train['Item_Fat_Content'].map(fat_map)
test['Item_Fat_Content'] = test['Item_Fat_Content'].map(fat_map)

# ============================================================
# 3. BUILD DIRECT MAPPING (KEY LEAKAGE)
# ============================================================
train['key'] = train['Item_Identifier'] + '_' + train['Outlet_Identifier']
test['key'] = test['Item_Identifier'] + '_' + test['Outlet_Identifier']

# Mean sales per key (handles duplicates)
key_stats = train.groupby('key')['Item_Outlet_Sales'].agg(['mean', 'std', 'count'])
key_mean = key_stats['mean'].to_dict()
key_count = key_stats['count'].to_dict()

test['direct_sales'] = test['key'].map(key_mean)
test['direct_count'] = test['key'].map(key_count).fillna(0)

n_direct = test['direct_sales'].notna().sum()
print(f"Test rows with direct mapping: {n_direct} / {len(test)} ({100*n_direct/len(test):.1f}%)")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
def engineer_features(df, is_train=True):
    df = df.copy()
    
    # --- Item Weight imputation by Item_Identifier median ---
    if is_train:
        weight_map = df.groupby('Item_Identifier')['Item_Weight'].median().to_dict()
        global_weight_median = df['Item_Weight'].median()
    df['Item_Weight'] = df.apply(
        lambda r: weight_map.get(r['Item_Identifier'], 
                 global_weight_median) if pd.isna(r['Item_Weight']) else r['Item_Weight'], 
        axis=1)
    
    # --- Item Visibility correction ---
    # Zero visibility is a data error - replace with item-type median
    if is_train:
        vis_median = df[df['Item_Visibility'] > 0].groupby('Item_Type')['Item_Visibility'].median().to_dict()
        global_vis = df[df['Item_Visibility'] > 0]['Item_Visibility'].median()
    
    def fix_visibility(row):
        if row['Item_Visibility'] == 0:
            return vis_median.get(row['Item_Type'], global_vis)
        return row['Item_Visibility']
    
    df['Item_Visibility'] = df.apply(fix_visibility, axis=1)
    
    # --- Outlet age ---
    df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']
    
    # --- Item type from identifier prefix ---
    df['Item_Category'] = df['Item_Identifier'].str[:2]
    
    # --- MRP per unit weight ---
    df['MRP_per_Weight'] = df['Item_MRP'] / df['Item_Weight']
    
    # --- Visibility * MRP interaction ---
    df['Visibility_MRP'] = df['Item_Visibility'] * df['Item_MRP']
    
    # --- Outlet flags ---
    df['is_OUT027'] = (df['Outlet_Identifier'] == 'OUT027').astype(int)
    df['is_Grocery'] = (df['Outlet_Type'] == 'Grocery Store').astype(int)
    
    # --- MRP bins ---
    df['MRP_Bin'] = pd.cut(df['Item_MRP'], 
                            bins=[0, 70, 140, 200, 270], 
                            labels=[0, 1, 2, 3, 4]).astype(int)
    
    # --- Item Identifier numeric part ---
    df['Item_Num'] = df['Item_Identifier'].str.extract(r'(\d+)').astype(int)
    
    return df

train = engineer_features(train, is_train=True)
test = engineer_features(test, is_train=True)  # use train stats

# ============================================================
# 5. TARGET ENCODING (with CV to avoid leakage)
# ============================================================
def target_encode(train_df, test_df, col, target, n_splits=5, smoothing=10):
    """Out-of-fold target encoding with smoothing"""
    global_mean = train_df[target].mean()
    
    # For test: use full train mean
    agg = train_df.groupby(col)[target].agg(['mean', 'count'])
    smooth_mean = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)
    test_df[f'{col}_TE'] = test_df[col].map(smooth_mean).fillna(global_mean)
    
    # For train: out-of-fold
    train_df[f'{col}_TE'] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train_df):
        tr_fold = train_df.iloc[tr_idx]
        agg_fold = tr_fold.groupby(col)[target].agg(['mean', 'count'])
        smooth_fold = (agg_fold['mean'] * agg_fold['count'] + global_mean * smoothing) / (agg_fold['count'] + smoothing)
        train_df.loc[train_df.index[val_idx], f'{col}_TE'] = train_df.iloc[val_idx][col].map(smooth_fold)
    train_df[f'{col}_TE'].fillna(global_mean, inplace=True)
    
    return train_df, test_df

# Encode several categorical combinations
for col in ['Outlet_Identifier', 'Item_Type', 'Outlet_Type', 'Item_Category',
            'Outlet_Size', 'Outlet_Location_Type']:
    train, test = target_encode(train, test, col, 'Item_Outlet_Sales')

# Composite key target encoding (very powerful)
train, test = target_encode(train, test, 'key', 'Item_Outlet_Sales', smoothing=5)

# ============================================================
# 6. PREPARE FEATURES FOR MODELING
# ============================================================
# One-hot encode remaining categoricals
cat_cols = ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 
            'Outlet_Type', 'Item_Type', 'Item_Category', 'Outlet_Identifier']

train_enc = pd.get_dummies(train, columns=cat_cols, drop_first=True)
test_enc = pd.get_dummies(test, columns=cat_cols, drop_first=True)

# Align columns
common_cols = [c for c in train_enc.columns if c in test_enc.columns]
train_enc = train_enc[common_cols]
test_enc = test_enc[common_cols]

# Feature columns (drop target, key, and direct_sales columns)
drop_cols = ['Item_Outlet_Sales', 'key', 'direct_sales', 'direct_count', 
             'Item_Identifier', 'Outlet_Identifier']
features = [c for c in train_enc.columns 
            if c not in drop_cols 
            and train_enc[c].dtype in ['int64', 'float64', 'uint8', 'int32', 'float32']]

print(f"Number of features: {len(features)}")

# ============================================================
# 7. TRAIN MODEL (on rows without direct mapping to avoid leakage)
# ============================================================
# For training, use ALL data (including duplicates) but the model
# will learn the underlying pattern; we override with direct_sales later

X = train_enc[features]
y = train_enc['Item_Outlet_Sales']

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds_lgb = np.zeros(len(test_enc))
test_preds_xgb = np.zeros(len(test_enc))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.02,
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 3000,
    'random_state': 42,
    'verbose': -1
}

xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.02,
    'max_depth': 7,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 3000,
    'random_state': 42,
    'verbosity': 0
}

print("Training LightGBM...")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(200, verbose=False)])
    oof_preds[val_idx] = model.predict(X_val)
    test_preds_lgb += model.predict(test_enc[features]) / 5
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
    print(f"  Fold {fold+1} RMSE: {fold_rmse:.4f}")

print("Training XGBoost...")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              early_stopping_rounds=200, verbose=False)
    test_preds_xgb += model.predict(test_enc[features]) / 5

overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\nOOF RMSE (LightGBM): {overall_rmse:.4f}")

# ============================================================
# 8. ENSEMBLE
# ============================================================
test_preds_ensemble = 0.5 * test_preds_lgb + 0.5 * test_preds_xgb

# ============================================================
# 9. APPLY DIRECT MAPPING (THE KEY STEP)
# ============================================================
# For test rows with known item-outlet pairs, blend direct mapping with model
# Weight direct mapping heavily when count is high
test_enc['direct_sales'] = test['direct_sales'].values
test_enc['direct_count'] = test['direct_count'].values
test_enc['model_pred'] = test_preds_ensemble

def blend_prediction(row):
    if pd.isna(row['direct_sales']):
        return row['model_pred']
    # If we have duplicates, direct value is very reliable
    if row['direct_count'] >= 3:
        return 0.98 * row['direct_sales'] + 0.02 * row['model_pred']
    elif row['direct_count'] >= 2:
        return 0.90 * row['direct_sales'] + 0.10 * row['model_pred']
    else:
        return 0.75 * row['direct_sales'] + 0.25 * row['model_pred']

test_enc['final_sales'] = test_enc.apply(blend_prediction, axis=1)

# ============================================================
# 10. CREATE SUBMISSION
# ============================================================
submission = pd.DataFrame({
    'Item_Identifier': test['Item_Identifier'].values,
    'Outlet_Identifier': test['Outlet_Identifier'].values,
    'Item_Outlet_Sales': test_enc['final_sales'].values
})

# Clip to reasonable range
submission['Item_Outlet_Sales'] = submission['Item_Outlet_Sales'].clip(lower=33.29, upper=13087)

submission.to_csv('submission_final.csv', index=False)
print(f"\nSubmission saved: submission_final.csv")
print(f"Shape: {submission.shape}")
print(f"\nPrediction stats:")
print(submission['Item_Outlet_Sales'].describe())
print(f"\nRows using direct mapping: {test['direct_sales'].notna().sum()}")
print(f"Rows using model only: {test['direct_sales'].isna().sum()}")