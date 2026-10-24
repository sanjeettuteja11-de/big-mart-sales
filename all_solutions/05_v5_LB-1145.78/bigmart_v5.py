import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
SEED = 42

train = pd.read_csv("train_v9rqX0R.csv")
test = pd.read_csv("test_AbJTz2l.csv")

train["source"] = "train"
test["source"] = "test"
test["Item_Outlet_Sales"] = np.nan
data = pd.concat([train, test], ignore_index=True, sort=False)

# ---- Cleaning ----
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
    if mrp < 69:
        return 0
    elif mrp < 136:
        return 1
    elif mrp < 203:
        return 2
    else:
        return 3

data["Item_MRP_Band"] = data["Item_MRP"].apply(mrp_band)
data["Item_Type_MRP_Mean"] = data.groupby("Item_Type")["Item_MRP"].transform("mean")
data["Outlet_Combo"] = (
    data["Outlet_Type"] + "_" + data["Outlet_Size"] + "_" + data["Outlet_Location_Type"]
)
data["MRPBand_Outlet"] = data["Item_MRP_Band"].astype(str) + "_" + data["Outlet_Identifier"]

# ---- Additional engineered features ----
data["Outlet_Item_Count"] = data.groupby("Outlet_Identifier")["Item_Identifier"].transform("count")
data["Outlet_Avg_MRP"] = data.groupby("Outlet_Identifier")["Item_MRP"].transform("mean")
data["Item_MRP_to_OutletAvg"] = data["Item_MRP"] / data["Outlet_Avg_MRP"]
data["Outlet_Avg_Visibility"] = data.groupby("Outlet_Identifier")["Item_Visibility"].transform("mean")
data["Item_Visibility_to_OutletAvg"] = data["Item_Visibility"] / data["Outlet_Avg_Visibility"]
data["ItemType_Avg_Visibility"] = data.groupby("Item_Type")["Item_Visibility"].transform("mean")
data["Item_MRP_sq"] = data["Item_MRP"] ** 2
data["Item_MRP_log"] = np.log1p(data["Item_MRP"])
# Price per weight
data["Price_per_Weight"] = data["Item_MRP"] / data["Item_Weight"].replace(0, np.nan)
data["Price_per_Weight"] = data["Price_per_Weight"].fillna(data["Price_per_Weight"].mean())

le = LabelEncoder()
cat_cols = [
    "Item_Fat_Content", "Outlet_Identifier", "Outlet_Size",
    "Outlet_Location_Type", "Outlet_Type", "Item_Type_Combined",
    "Item_Type", "Outlet_Combo",
]
for col in cat_cols:
    data[col + "_Enc"] = le.fit_transform(data[col].astype(str))

train_data = data[data["source"] == "train"].reset_index(drop=True).copy()
test_data = data[data["source"] == "test"].reset_index(drop=True).copy()

y = train_data["Item_Outlet_Sales"]

target_enc_cols = ["Outlet_Identifier", "Item_Type", "Outlet_Combo", "MRPBand_Outlet"]
smooth_m = {"Outlet_Identifier": 10, "Item_Type": 20, "Outlet_Combo": 10, "MRPBand_Outlet": 10}

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
    "Item_Type_Enc", "Outlet_Combo_Enc",
    "Outlet_Item_Count", "Outlet_Avg_MRP", "Item_MRP_to_OutletAvg",
    "Outlet_Avg_Visibility", "Item_Visibility_to_OutletAvg", "ItemType_Avg_Visibility",
    "Item_MRP_sq", "Item_MRP_log", "Price_per_Weight",
]
te_feature_names = [c + "_TargetEnc" for c in target_enc_cols]
feature_cols = base_feature_cols + te_feature_names

X_base = train_data[base_feature_cols].reset_index(drop=True)
X_test_base = test_data[base_feature_cols].reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

model_specs = {
    "ridge": (Ridge(alpha=3.0), True),
    "gbm": (GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.02,
                                       subsample=0.8, random_state=SEED), True),
    "rf": (RandomForestRegressor(n_estimators=500, max_depth=6, min_samples_leaf=10,
                                  random_state=SEED, n_jobs=-1), False),
    "extratrees": (ExtraTreesRegressor(n_estimators=500, max_depth=7, min_samples_leaf=10,
                                        random_state=SEED, n_jobs=-1), False),
    "xgb": (xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.8,
                              colsample_bytree=0.7, reg_lambda=2.0, random_state=SEED, n_jobs=-1), True),
    "lgb": (lgb.LGBMRegressor(n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.8,
                               colsample_bytree=0.7, reg_lambda=2.0, random_state=SEED, verbosity=-1), True),
    "cat": (CatBoostRegressor(iterations=500, depth=5, learning_rate=0.03, loss_function="RMSE",
                               random_state=SEED, verbose=False), True),
}

oof_preds = {name: np.zeros(len(X_base)) for name in model_specs}
test_pred_accum = {name: np.zeros(len(X_test_base)) for name in model_specs}
fold_rmse_log = {name: [] for name in model_specs}

for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_base)):
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

    X_tr, X_val = X_full.iloc[tr_idx], X_full.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    for name, (model, use_log) in model_specs.items():
        target_tr = np.log1p(y_tr) if use_log else y_tr
        model.fit(X_tr, target_tr)
        pred = model.predict(X_val)
        if use_log:
            pred = np.expm1(pred)
        pred = np.clip(pred, 0, None)
        oof_preds[name][val_idx] = pred
        rmse = np.sqrt(np.mean((y_val.values - pred) ** 2))
        fold_rmse_log[name].append(rmse)

        tpred = model.predict(X_test_full)
        if use_log:
            tpred = np.expm1(tpred)
        test_pred_accum[name] += np.clip(tpred, 0, None) / 5.0

for name in model_specs:
    print(f"{name}: CV RMSE = {np.mean(fold_rmse_log[name]):.4f} (+/- {np.std(fold_rmse_log[name]):.4f})")

y_true = y.values

# Stacking: meta-learner (Ridge, non-negative-ish via careful alpha) on OOF preds
from sklearn.linear_model import RidgeCV
meta_X = np.column_stack([oof_preds[name] for name in model_specs])
meta_test_X = np.column_stack([test_pred_accum[name] for name in model_specs])

meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 5.0, 10.0, 20.0])
meta_model.fit(meta_X, y_true)
meta_oof_pred = meta_model.predict(meta_X)
meta_oof_pred = np.clip(meta_oof_pred, 0, None)
stack_rmse = np.sqrt(np.mean((y_true - meta_oof_pred) ** 2))
print("Stacking (RidgeCV meta) RMSE:", stack_rmse)
print("Meta coefs:", dict(zip(model_specs.keys(), meta_model.coef_)), "intercept:", meta_model.intercept_)

# Also try simple average and weighted search for comparison
best_rmse = 1e18
best_weights = None
names = list(model_specs.keys())
import itertools
# random search over simplex weights for speed given 7 models
rng = np.random.default_rng(42)
for _ in range(20000):
    w = rng.dirichlet(np.ones(len(names)))
    blend = sum(w[i] * oof_preds[names[i]] for i in range(len(names)))
    rmse = np.sqrt(np.mean((y_true - blend) ** 2))
    if rmse < best_rmse:
        best_rmse = rmse
        best_weights = w

print("Best random-search blend RMSE:", best_rmse)
print("Best weights:", dict(zip(names, best_weights)))

# choose whichever method is better between stacking and best random blend
if stack_rmse <= best_rmse:
    print("Using stacking meta-model for final prediction")
    final_test_pred = meta_model.predict(meta_test_X)
else:
    print("Using weighted blend for final prediction")
    final_test_pred = sum(best_weights[i] * test_pred_accum[names[i]] for i in range(len(names)))

final_test_pred = np.clip(final_test_pred, 0, None)

submission = pd.DataFrame({
    "Item_Identifier": test_data["Item_Identifier"],
    "Outlet_Identifier": test_data["Outlet_Identifier"],
    "Item_Outlet_Sales": final_test_pred,
})
submission.to_csv("bigmart_submission_v5.csv", index=False)
print("Saved bigmart_submission_v5.csv, shape:", submission.shape)
print(submission.head())
