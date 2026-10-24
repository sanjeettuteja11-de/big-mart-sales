import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
import warnings, time
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

t0 = time.time()
SEED = 42
np.random.seed(SEED)

train = pd.read_csv("train_v9rqX0R.csv")
test = pd.read_csv("test_AbJTz2l.csv")
train["source"] = "train"
test["source"] = "test"
test["Item_Outlet_Sales"] = np.nan
data = pd.concat([train, test], ignore_index=True, sort=False)

# ---------------- Cleaning ----------------
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

# ---------------- Extra features ----------------
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
data["Outlet_Type_Years"] = data["Outlet_Type"] + "_" + data["Outlet_Years"].astype(str)
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
te_feature_names = [c + "_TargetEnc" for c in target_enc_cols]
feature_cols = base_feature_cols + te_feature_names

X_base = train_data[base_feature_cols].reset_index(drop=True)
X_test_base = test_data[base_feature_cols].reset_index(drop=True)

print("n_features (base+TE):", len(feature_cols))
print(f"[{time.time()-t0:.0f}s] Feature engineering done")

# Build fold-safe TE using single 5-fold split (used both for tuning and final)
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
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
print(f"[{time.time()-t0:.0f}s] Fold-safe target encodings built")

def cv_rmse_for_params(model_ctor, use_log=True):
    rmses = []
    for (tr_idx, val_idx), (X_full, _) in zip(fold_indices, fold_features):
        X_tr, X_val = X_full.iloc[tr_idx], X_full.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        target_tr = np.log1p(y_tr) if use_log else y_tr
        m = model_ctor()
        m.fit(X_tr, target_tr)
        pred = m.predict(X_val)
        if use_log:
            pred = np.expm1(pred)
        pred = np.clip(pred, 0, None)
        rmses.append(np.sqrt(np.mean((y_val.values - pred) ** 2)))
    return np.mean(rmses)

N_TRIALS = 25

def lgb_objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 150, 600),
        max_depth=trial.suggest_int("max_depth", 3, 6),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 40),
        subsample=trial.suggest_float("subsample", 0.6, 0.95),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.95),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
        random_state=SEED, verbosity=-1,
    )
    return cv_rmse_for_params(lambda: lgb.LGBMRegressor(**params), use_log=True)

def xgb_objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 150, 600),
        max_depth=trial.suggest_int("max_depth", 3, 6),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 0.95),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.95),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        random_state=SEED, n_jobs=-1,
    )
    return cv_rmse_for_params(lambda: xgb.XGBRegressor(**params), use_log=True)

def cat_objective(trial):
    params = dict(
        iterations=trial.suggest_int("iterations", 200, 600),
        depth=trial.suggest_int("depth", 3, 7),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        random_state=SEED, verbose=False, loss_function="RMSE",
    )
    return cv_rmse_for_params(lambda: CatBoostRegressor(**params), use_log=True)

def rf_objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 700),
        max_depth=trial.suggest_int("max_depth", 4, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 4, 30),
        random_state=SEED, n_jobs=-1,
    )
    return cv_rmse_for_params(lambda: RandomForestRegressor(**params), use_log=False)

def et_objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 700),
        max_depth=trial.suggest_int("max_depth", 4, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 4, 30),
        random_state=SEED, n_jobs=-1,
    )
    return cv_rmse_for_params(lambda: ExtraTreesRegressor(**params), use_log=False)

def ridge_objective(trial):
    alpha = trial.suggest_float("alpha", 0.1, 50.0, log=True)
    return cv_rmse_for_params(lambda: Ridge(alpha=alpha), use_log=True)

studies = {}
best_params = {}
for name, obj_fn in [
    ("lgb", lgb_objective), ("xgb", xgb_objective), ("cat", cat_objective),
    ("rf", rf_objective), ("et", et_objective), ("ridge", ridge_objective),
]:
    study = optuna.create_study(direction="minimize")
    study.optimize(obj_fn, n_trials=N_TRIALS, show_progress_bar=False)
    studies[name] = study
    best_params[name] = study.best_params
    print(f"[{time.time()-t0:.0f}s] {name} best CV RMSE = {study.best_value:.4f} params={study.best_params}")

import json
with open("bigmart_best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
print(f"[{time.time()-t0:.0f}s] Saved best_params to bigmart_best_params.json")
