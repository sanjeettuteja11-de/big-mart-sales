import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

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

# Known BigMart MRP gap-based price cluster (4 bands with clear gaps in histogram)
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

item_type_mrp_mean = data.groupby("Item_Type")["Item_MRP"].transform("mean")
data["Item_Type_MRP_Mean"] = item_type_mrp_mean

data["Outlet_Combo"] = (
    data["Outlet_Type"] + "_" + data["Outlet_Size"] + "_" + data["Outlet_Location_Type"]
)

train_mask = data["source"] == "train"

def smoothed_mean_encode(data, train_mask, col, target_col, m=20):
    # col can be a column name (str) or a Series aligned with data's index
    key_series = data[col] if isinstance(col, str) else col
    global_mean = data.loc[train_mask, target_col].mean()
    tmp = pd.DataFrame({"key": key_series, "target": data[target_col]})
    agg = tmp.loc[train_mask].groupby("key")["target"].agg(["mean", "count"])
    smooth = (agg["mean"] * agg["count"] + global_mean * m) / (agg["count"] + m)
    return key_series.map(smooth).fillna(global_mean)

data["Outlet_TargetEnc"] = smoothed_mean_encode(data, train_mask, "Outlet_Identifier", "Item_Outlet_Sales", m=10)
data["ItemType_TargetEnc"] = smoothed_mean_encode(data, train_mask, "Item_Type", "Item_Outlet_Sales", m=20)
data["OutletCombo_TargetEnc"] = smoothed_mean_encode(data, train_mask, "Outlet_Combo", "Item_Outlet_Sales", m=10)
data["MRPBand_Outlet_TargetEnc"] = smoothed_mean_encode(
    data, train_mask, data["Item_MRP_Band"].astype(str) + "_" + data["Outlet_Identifier"], "Item_Outlet_Sales", m=10
)

le = LabelEncoder()
cat_cols = [
    "Item_Fat_Content", "Outlet_Identifier", "Outlet_Size",
    "Outlet_Location_Type", "Outlet_Type", "Item_Type_Combined",
    "Item_Type", "Outlet_Combo",
]
for col in cat_cols:
    data[col + "_Enc"] = le.fit_transform(data[col].astype(str))

drop_cols = [
    "Item_Identifier", "Outlet_Establishment_Year", "Item_Fat_Content",
    "Outlet_Identifier", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type",
    "Item_Type_Combined", "Item_Type", "Outlet_Combo",
]

feature_cols = [c for c in data.columns if c not in drop_cols + ["source", "Item_Outlet_Sales"]]

train_data = data[data["source"] == "train"].copy()
test_data = data[data["source"] == "test"].copy()

X = train_data[feature_cols]
y = train_data["Item_Outlet_Sales"]
X_test = test_data[feature_cols]

print("Feature columns:", feature_cols)
print("Train shape:", X.shape, "Test shape:", X_test.shape)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Try both raw and log target for each model, pick per-model best via CV
model_specs = {
    "ridge": (Ridge(alpha=3.0), False),
    "gbm": (GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.85, random_state=42), False),
    "rf": (RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=8, random_state=42, n_jobs=-1), False),
    "xgb": (xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.85,
                              colsample_bytree=0.7, reg_lambda=1.5, random_state=42, n_jobs=-1), False),
}

oof_preds = {}
test_preds = {}

for name, (model, use_log) in model_specs.items():
    oof = np.zeros(len(X))
    test_fold_preds = np.zeros((5, len(X_test)))
    fold_rmses = []
    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        target_tr = np.log1p(y_tr) if use_log else y_tr
        model.fit(X_tr, target_tr)
        pred = model.predict(X_val)
        if use_log:
            pred = np.expm1(pred)
        pred = np.clip(pred, 0, None)
        oof[val_idx] = pred
        rmse = np.sqrt(np.mean((y_val.values - pred) ** 2))
        fold_rmses.append(rmse)
        tpred = model.predict(X_test)
        if use_log:
            tpred = np.expm1(tpred)
        test_fold_preds[fold_i] = np.clip(tpred, 0, None)
    oof_preds[name] = oof
    test_preds[name] = test_fold_preds.mean(axis=0)
    print(f"{name} (log={use_log}): CV RMSE = {np.mean(fold_rmses):.4f} (+/- {np.std(fold_rmses):.4f})")

y_true = y.values

best_rmse = 1e18
best_weights = None
weight_options = np.arange(0.0, 1.01, 0.1)
for w_ridge in weight_options:
    for w_gbm in weight_options:
        for w_rf in weight_options:
            w_xgb = 1.0 - w_ridge - w_gbm - w_rf
            if w_xgb < -1e-9 or w_xgb > 1 + 1e-9:
                continue
            w_xgb = max(w_xgb, 0.0)
            blend = (
                w_ridge * oof_preds["ridge"]
                + w_gbm * oof_preds["gbm"]
                + w_rf * oof_preds["rf"]
                + w_xgb * oof_preds["xgb"]
            )
            rmse = np.sqrt(np.mean((y_true - blend) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = (w_ridge, w_gbm, w_rf, w_xgb)

print("Best blend weights (ridge, gbm, rf, xgb):", best_weights)
print("Best blended CV RMSE:", best_rmse)

w_ridge, w_gbm, w_rf, w_xgb = best_weights
final_test_pred = (
    w_ridge * test_preds["ridge"]
    + w_gbm * test_preds["gbm"]
    + w_rf * test_preds["rf"]
    + w_xgb * test_preds["xgb"]
)
final_test_pred = np.clip(final_test_pred, 0, None)

submission = pd.DataFrame({
    "Item_Identifier": test_data["Item_Identifier"],
    "Outlet_Identifier": test_data["Outlet_Identifier"],
    "Item_Outlet_Sales": final_test_pred,
})
submission.to_csv("bigmart_submission_v3.csv", index=False)
print("Saved bigmart_submission_v3.csv, shape:", submission.shape)
print(submission.head())
