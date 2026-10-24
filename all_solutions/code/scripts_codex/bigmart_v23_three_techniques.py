import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

SEED = 42

train_raw = pd.read_csv("train_v9rqX0R.csv")
test_raw = pd.read_csv("test_AbJTz2l.csv")

def prepare_data(train, test):
    train = train.copy()
    test = test.copy()
    combined = pd.concat([train.drop(columns="Item_Outlet_Sales"), test], ignore_index=True, sort=False)
    weight_by_item = combined.groupby("Item_Identifier")["Item_Weight"].median()
    global_weight = combined["Item_Weight"].median()
    for frame in (train, test):
        frame["Item_Fat_Content"] = frame["Item_Fat_Content"].replace(
            {"low fat": "Low Fat", "LF": "Low Fat", "reg": "Regular"})
        frame["Item_Category"] = frame["Item_Identifier"].str[:2]
        frame["Outlet_Age"] = 2013 - frame["Outlet_Establishment_Year"]
        frame["MRP_Bin"] = pd.cut(frame["Item_MRP"], bins=[0,70,140,200,300], labels=False, include_lowest=True)
        frame["Visibility_Zero"] = (frame["Item_Visibility"] == 0).astype(int)
        frame["Outlet_Size"] = frame["Outlet_Size"].fillna("Missing")
        frame["Item_Weight"] = frame["Item_Weight"].fillna(frame["Item_Identifier"].map(weight_by_item)).fillna(global_weight)
    return train, test

train, test = prepare_data(train_raw, test_raw)
y = train["Item_Outlet_Sales"].to_numpy()

from sklearn.preprocessing import LabelEncoder
cat_cols = ["Item_Identifier","Item_Fat_Content","Item_Type","Outlet_Identifier",
            "Outlet_Size","Outlet_Location_Type","Outlet_Type","Item_Category"]
le_dict = {}
combined_full = pd.concat([train, test], ignore_index=True, sort=False)
for c in cat_cols:
    le = LabelEncoder()
    le.fit(combined_full[c].astype(str))
    train[c + "_enc"] = le.transform(train[c].astype(str))
    test[c + "_enc"] = le.transform(test[c].astype(str))

feature_cols = [c + "_enc" for c in cat_cols] + [
    "Item_Weight","Item_Visibility","Item_MRP","Outlet_Age","MRP_Bin","Visibility_Zero"]

X = train[feature_cols].astype(float).values
X_test = test[feature_cols].astype(float).values

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

print("=" * 60)
print("TECHNIQUE 1: XGBoost + RandomizedSearchCV")
print("=" * 60)
param_dist = {
    'n_estimators': [500, 1000, 2000],
    'max_depth': [4, 5, 6, 7],
    'learning_rate': [0.01, 0.02, 0.03],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
    'min_child_weight': [1, 3, 5, 7],
    'reg_alpha': [0, 0.1, 1, 5],
    'reg_lambda': [0.5, 1, 2, 5],
    'gamma': [0, 0.1, 0.3, 0.5]
}
xgb_base = XGBRegressor(objective='reg:squarederror', random_state=SEED, n_jobs=-1)
search = RandomizedSearchCV(xgb_base, param_dist, n_iter=40,
                             scoring='neg_root_mean_squared_error',
                             cv=kf, random_state=SEED, n_jobs=-1, verbose=0)
search.fit(X, y)
print("Best XGB params:", search.best_params_)
print("Best XGB CV RMSE:", -search.best_score_)

print()
print("=" * 60)
print("TECHNIQUE 2: LightGBM with linear_tree=True")
print("=" * 60)
lgb_oof = np.zeros(len(X))
for tr, val in kf.split(X):
    m = lgb.LGBMRegressor(
        objective='regression', linear_tree=True, linear_lambda=0.1,
        num_leaves=31, learning_rate=0.02, n_estimators=1000,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, verbose=-1)
    m.fit(X[tr], y[tr])
    lgb_oof[val] = m.predict(X[val])
lgb_rmse = np.sqrt(mean_squared_error(y, lgb_oof))
print("LightGBM linear_tree CV RMSE:", lgb_rmse)

print()
print("=" * 60)
print("TECHNIQUE 3: CatBoost with native categoricals (NOT one-hot)")
print("=" * 60)
cat_feature_cols = cat_cols + ["MRP_Bin", "Visibility_Zero"]
numeric_cols = ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Age"]
cb_feature_cols = cat_feature_cols + numeric_cols

train_cb = train.copy()
test_cb = test.copy()
for c in cat_feature_cols:
    train_cb[c] = train_cb[c].astype(str)
    test_cb[c] = test_cb[c].astype(str)

X_cb = train_cb[cb_feature_cols]
X_test_cb = test_cb[cb_feature_cols]
cat_indices = [cb_feature_cols.index(c) for c in cat_feature_cols]

cb_oof = np.zeros(len(X_cb))
for tr, val in kf.split(X_cb):
    m = CatBoostRegressor(
        iterations=2000, learning_rate=0.02, depth=6, l2_leaf_reg=3,
        random_seed=SEED, verbose=0, early_stopping_rounds=200,
        cat_features=cat_indices, one_hot_max_size=2, max_ctr_complexity=4)
    m.fit(X_cb.iloc[tr], y[tr], eval_set=(X_cb.iloc[val], y[val]))
    cb_oof[val] = m.predict(X_cb.iloc[val])
cb_rmse = np.sqrt(mean_squared_error(y, cb_oof))
print("CatBoost native-cat CV RMSE:", cb_rmse)

print()
print("=" * 60)
print("SUMMARY (real, verified numbers)")
print("=" * 60)
print(f"XGBoost RandomizedSearchCV: {-search.best_score_:.4f}")
print(f"LightGBM linear_tree:       {lgb_rmse:.4f}")
print(f"CatBoost native categoricals: {cb_rmse:.4f}")
print(f"v22 (current best):         1071.2544")
print(f"v5 (previously submitted):   1073.5")
