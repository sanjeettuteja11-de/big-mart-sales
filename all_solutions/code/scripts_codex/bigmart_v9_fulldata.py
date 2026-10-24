import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

# =====================================================================
# FINAL PUSH: use full train data for target-encoding + final model fit
# (more data for the real predictions), multi-seed bagging to cut variance,
# while CV (fold-safe TE) is still used ONLY to pick honest stacking weights.
# =====================================================================

SEEDS = [42, 7, 123, 2024, 99]

train = pd.read_csv("train_v9rqX0R.csv")
test = pd.read_csv("test_AbJTz2l.csv")
train["source"] = "train"
test["source"] = "test"
test["Item_Outlet_Sales"] = np.nan
data = pd.concat([train, test], ignore_index=True, sort=False)

data["Item_Fat_Content"] = data["Item_Fat_Content"].replace(
    {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
)
item_avg_weight = data.groupby("Item_Identifier")["Item_Weight"].transform("mean")
data["Item_Weight"] = data["Item_Weight"].fillna(item_avg_weight)
type_avg_weight = data.groupby("Item_Type")["Item_Weight"].transform("mean")
data["Item_Weight"] = data["Item_Weight"].fillna(type_avg_weight)

outlet_size_mode = data.groupby("Outlet_Type")["Outlet_Size"].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else "Medium"
)
missing_size = data["Outlet_Size"].isnull()
data.loc[missing_size, "Outlet_Size"] = data.loc[missing_size, "Outlet_Type"].map(outlet_size_mode)

data["Item_Visibility"] = data["Item_Visibility"].replace(0, np.nan)
item_avg_vis = data.groupby("Item_Identifier")["Item_Visibility"].transform("mean")
data["Item_Visibility"] = data["Item_Visibility"].fillna(item_avg_vis)
data["Item_Visibility"] = data["Item_Visibility"].fillna(data["Item_Visibility"].mean())

data["Item_Type_Combined"] = data["Item_Identifier"].apply(lambda x: x[0:2])
data["Item_Type_Combined"] = data["Item_Type_Combined"].map(
    {"FD": "Food", "NC": "Non-Consumable", "DR": "Drinks"}
)
data.loc[data["Item_Type_Combined"] == "Non-Consumable", "Item_Fat_Content"] = "Non-Edible"
data["Outlet_Years"] = 2013 - data["Outlet_Establishment_Year"]

vis_avg_by_item = data.groupby("Item_Identifier")["Item_Visibility"].transform("mean")
data["Item_Visibility_MeanRatio"] = data["Item_Visibility"] / vis_avg_by_item

def mrp_band(mrp):
    if mrp < 69: return 0
    elif mrp < 136: return 1
    elif mrp < 203: return 2
    else: return 3
data["Item_MRP_Band"] = data["Item_MRP"].apply(mrp_band)
data["Item_Type_MRP_Mean"] = data.groupby("Item_Type")["Item_MRP"].transform("mean")
data["Outlet_Combo"] = data["Outlet_Type"] + "_" + data["Outlet_Size"] + "_" + data["Outlet_Location_Type"]
data["MRPBand_Outlet"] = data["Item_MRP_Band"].astype(str) + "_" + data["Outlet_Identifier"]

data["Item_Code3"] = data["Item_Identifier"].str[0:3]
data["Outlet_Item_Count"] = data.groupby("Outlet_Identifier")["Item_Identifier"].transform("count")
data["Outlet_Avg_MRP"] = data.groupby("Outlet_Identifier")["Item_MRP"].transform("mean")
data["Item_MRP_to_OutletAvg"] = data["Item_MRP"] / data["Outlet_Avg_MRP"]
data["Outlet_Avg_Visibility"] = data.groupby("Outlet_Identifier")["Item_Visibility"].transform("mean")
data["Item_Visibility_to_OutletAvg"] = data["Item_Visibility"] / data["Outlet_Avg_Visibility"]
data["ItemType_Avg_Visibility"] = data.groupby("Item_Type")["Item_Visibility"].transform("mean")
data["Item_MRP_sq"] = data["Item_MRP"] ** 2
data["Item_MRP_log"] = np.log1p(data["Item_MRP"])
data["Price_per_Weight"] = data["Item_MRP"] / data["Item_Weight"].replace(0, np.nan)
data["Price_per_Weight"] = data["Price_per_Weight"].fillna(data["Price_per_Weight"].mean())
data["ItemType_Outlet"] = data["Item_Type"] + "_" + data["Outlet_Identifier"]
data["ItemCode3_Outlet"] = data["Item_Code3"] + "_" + data["Outlet_Identifier"]
data["Item_Count_ItemType_Outlet"] = data.groupby(["Item_Type", "Outlet_Identifier"])["Item_Identifier"].transform("count")

le = LabelEncoder()
cat_cols = [
    "Item_Fat_Content", "Outlet_Identifier", "Outlet_Size",
    "Outlet_Location_Type", "Outlet_Type", "Item_Type_Combined",
    "Item_Type", "Outlet_Combo", "Item_Code3",
]
for col in cat_cols:
    data[col + "_Enc"] = le.fit_transform(data[col].astype(str))

train_data = data[data["source"] == "train"].reset_index(drop=True).copy()
test_data = data[data["source"] == "test"].reset_index(drop=True).copy()
y = train_data["Item_Outlet_Sales"]

target_enc_cols = ["Outlet_Identifier", "Item_Type", "Outlet_Combo", "MRPBand_Outlet",
                    "ItemType_Outlet", "ItemCode3_Outlet"]
smooth_m = {"Outlet_Identifier": 10, "Item_Type": 20, "Outlet_Combo": 10,
            "MRPBand_Outlet": 10, "ItemType_Outlet": 15, "ItemCode3_Outlet": 15}

def fit_target_enc(tr_df, tr_y, col, m):
    global_mean = tr_y.mean()
    tmp = pd.DataFrame({"key": tr_df[col].values, "target": tr_y.values})
    agg = tmp.groupby("key")["target"].agg(["mean", "count"])
    smooth = (agg["mean"] * agg["count"] + global_mean * m) / (agg["count"] + m)
    return smooth, global_mean

def apply_target_enc(df, col, smooth_map, global_mean):
    return df[col].map(smooth_map).fillna(global_mean)

base_feature_cols = [
    "Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Years",
    "Item_Visibility_MeanRatio", "Item_MRP_Band", "Item_Type_MRP_Mean",
    "Item_Fat_Content_Enc", "Outlet_Identifier_Enc", "Outlet_Size_Enc",
    "Outlet_Location_Type_Enc", "Outlet_Type_Enc", "Item_Type_Combined_Enc",
    "Item_Type_Enc", "Outlet_Combo_Enc", "Item_Code3_Enc",
    "Outlet_Item_Count", "Outlet_Avg_MRP", "Item_MRP_to_OutletAvg",
    "Outlet_Avg_Visibility", "Item_Visibility_to_OutletAvg", "ItemType_Avg_Visibility",
    "Item_MRP_sq", "Item_MRP_log", "Price_per_Weight", "Item_Count_ItemType_Outlet",
]

X_base = train_data[base_feature_cols].reset_index(drop=True)
X_test_base = test_data[base_feature_cols].reset_index(drop=True)

# -------------------------------------------------------------
# STEP 1: honest CV to get stacking weights (fold-safe TE, single seed)
# -------------------------------------------------------------
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_indices = list(kf.split(X_base))

def build_fold_features(tr_idx):
    tr_full_df = train_data.iloc[tr_idx]
    y_tr_fold = y.iloc[tr_idx]
    te_train_all = pd.DataFrame(index=range(len(train_data)))
    te_test_all = pd.DataFrame(index=range(len(test_data)))
    for col in target_enc_cols:
        smooth_map, global_mean = fit_target_enc(tr_full_df, y_tr_fold, col, smooth_m[col])
        te_train_all[col + "_TargetEnc"] = apply_target_enc(train_data, col, smooth_map, global_mean)
        te_test_all[col + "_TargetEnc"] = apply_target_enc(test_data, col, smooth_map, global_mean)
    X_full = pd.concat([X_base, te_train_all], axis=1)
    X_test_full = pd.concat([X_test_base, te_test_all], axis=1)
    return X_full, X_test_full

fold_features = [build_fold_features(tr_idx) for tr_idx, _ in fold_indices]

def make_models(seed):
    return {
        "ridge": (Ridge(alpha=3.0), True),
        "et": (ExtraTreesRegressor(n_estimators=500, max_depth=7, min_samples_leaf=10, random_state=seed, n_jobs=-1), False),
        "rf": (RandomForestRegressor(n_estimators=500, max_depth=6, min_samples_leaf=10, random_state=seed, n_jobs=-1), False),
        "gbm": (GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.02, subsample=0.8, random_state=seed), True),
        "cat": (CatBoostRegressor(iterations=500, depth=5, learning_rate=0.03, loss_function="RMSE", random_state=seed, verbose=False), True),
        "lgb": (lgb.LGBMRegressor(n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, random_state=seed, verbosity=-1), True),
        "xgb": (xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, random_state=seed, n_jobs=-1), True),
    }

model_names = list(make_models(42).keys())
oof_preds = {name: np.zeros(len(X_base)) for name in model_names}

for (tr_idx, val_idx), (X_full, _) in zip(fold_indices, fold_features):
    X_tr, X_val = X_full.iloc[tr_idx], X_full.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    models = make_models(42)
    for name, (model, use_log) in models.items():
        target_tr = np.log1p(y_tr) if use_log else y_tr
        model.fit(X_tr, target_tr)
        pred = model.predict(X_val)
        if use_log:
            pred = np.expm1(pred)
        oof_preds[name][val_idx] = np.clip(pred, 0, None)

y_true = y.values
for name in model_names:
    rmse = np.sqrt(np.mean((y_true - oof_preds[name]) ** 2))
    print(f"{name}: CV RMSE (fold-safe) = {rmse:.4f}")

meta_X = np.column_stack([oof_preds[name] for name in model_names])
best_rmse, best_alpha = 1e18, None
for alpha in [0.001,0.01,0.1,0.5,1,2,5,10,20,50,100]:
    mm = Ridge(alpha=alpha)
    mm.fit(meta_X, y_true)
    p = np.clip(mm.predict(meta_X), 0, None)
    r = np.sqrt(np.mean((y_true - p) ** 2))
    if r < best_rmse:
        best_rmse, best_alpha = r, alpha

meta_model = Ridge(alpha=best_alpha)
meta_model.fit(meta_X, y_true)
print("Honest stacking CV RMSE:", best_rmse, "alpha:", best_alpha)
print("Meta weights:", dict(zip(model_names, meta_model.coef_)), "intercept:", meta_model.intercept_)

# -------------------------------------------------------------
# STEP 2: refit on FULL train data (TE computed on full train, since test
# labels are genuinely unknown -- this is standard practice, not leakage)
# with multi-seed bagging for variance reduction, then apply meta weights.
# -------------------------------------------------------------
te_train_full = pd.DataFrame(index=range(len(train_data)))
te_test_full = pd.DataFrame(index=range(len(test_data)))
for col in target_enc_cols:
    smooth_map, global_mean = fit_target_enc(train_data, y, col, smooth_m[col])
    te_train_full[col + "_TargetEnc"] = apply_target_enc(train_data, col, smooth_map, global_mean)
    te_test_full[col + "_TargetEnc"] = apply_target_enc(test_data, col, smooth_map, global_mean)

X_full_all = pd.concat([X_base, te_train_full], axis=1)
X_test_full_all = pd.concat([X_test_base, te_test_full], axis=1)

test_pred_by_model_seed = {name: [] for name in model_names}
for seed in SEEDS:
    models = make_models(seed)
    for name, (model, use_log) in models.items():
        target_full = np.log1p(y) if use_log else y
        model.fit(X_full_all, target_full)
        tpred = model.predict(X_test_full_all)
        if use_log:
            tpred = np.expm1(tpred)
        test_pred_by_model_seed[name].append(np.clip(tpred, 0, None))

test_pred_final = {name: np.mean(test_pred_by_model_seed[name], axis=0) for name in model_names}

meta_test_X = np.column_stack([test_pred_final[name] for name in model_names])
final_test_pred = np.clip(meta_model.predict(meta_test_X), 0, None)

submission = pd.DataFrame({
    "Item_Identifier": test_data["Item_Identifier"],
    "Outlet_Identifier": test_data["Outlet_Identifier"],
    "Item_Outlet_Sales": final_test_pred,
})
submission.to_csv("bigmart_submission_v9.csv", index=False)
print("Saved bigmart_submission_v9.csv, shape:", submission.shape)
print(submission.head())
print("Sales stats:", submission["Item_Outlet_Sales"].describe())
