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

# ---- Clean Item_Fat_Content ----
data["Item_Fat_Content"] = data["Item_Fat_Content"].replace(
    {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
)

# ---- Item_Weight: impute by Item_Identifier mean, then Item_Type mean ----
item_avg_weight = data.groupby("Item_Identifier")["Item_Weight"].transform("mean")
data["Item_Weight"] = data["Item_Weight"].fillna(item_avg_weight)
type_avg_weight = data.groupby("Item_Type")["Item_Weight"].transform("mean")
data["Item_Weight"] = data["Item_Weight"].fillna(type_avg_weight)

# ---- Outlet_Size: impute by Outlet_Type mode ----
outlet_size_mode = data.groupby("Outlet_Type")["Outlet_Size"].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else "Medium"
)
missing_size = data["Outlet_Size"].isnull()
data.loc[missing_size, "Outlet_Size"] = data.loc[missing_size, "Outlet_Type"].map(outlet_size_mode)

# ---- Item_Visibility: 0 -> mean by Item_Identifier ----
data["Item_Visibility"] = data["Item_Visibility"].replace(0, np.nan)
item_avg_vis = data.groupby("Item_Identifier")["Item_Visibility"].transform("mean")
data["Item_Visibility"] = data["Item_Visibility"].fillna(item_avg_vis)
data["Item_Visibility"] = data["Item_Visibility"].fillna(data["Item_Visibility"].mean())

# ---- Item_Type_Combined ----
data["Item_Type_Combined"] = data["Item_Identifier"].apply(lambda x: x[0:2])
data["Item_Type_Combined"] = data["Item_Type_Combined"].map(
    {"FD": "Food", "NC": "Non-Consumable", "DR": "Drinks"}
)
data.loc[data["Item_Type_Combined"] == "Non-Consumable", "Item_Fat_Content"] = "Non-Edible"

# ---- Outlet_Years ----
data["Outlet_Years"] = 2013 - data["Outlet_Establishment_Year"]

# ---- Visibility ratio: item visibility relative to mean visibility of that item across outlets ----
vis_avg_by_item = data.groupby("Item_Identifier")["Item_Visibility"].transform("mean")
data["Item_Visibility_MeanRatio"] = data["Item_Visibility"] / vis_avg_by_item

# ---- Price-based features ----
data["Item_MRP_Bin"] = pd.qcut(data["Item_MRP"], 4, labels=False)

# Mean-encode Item_Type by average MRP rank (proxy for category price tier)
item_type_mrp_mean = data.groupby("Item_Type")["Item_MRP"].transform("mean")
data["Item_Type_MRP_Mean"] = item_type_mrp_mean

# Outlet identifier combined feature (captures store identity strongly)
data["Outlet_Combo"] = (
    data["Outlet_Type"] + "_" + data["Outlet_Size"] + "_" + data["Outlet_Location_Type"]
)

# Item_Identifier without number suffix (category code) already in Item_Type_Combined

# ---- Target (smoothed) mean-encoding for Outlet_Identifier & Item_Type using train rows only ----
train_mask = data["source"] == "train"

def smoothed_mean_encode(data, train_mask, col, target_col, m=20):
    global_mean = data.loc[train_mask, target_col].mean()
    agg = data.loc[train_mask].groupby(col)[target_col].agg(["mean", "count"])
    smooth = (agg["mean"] * agg["count"] + global_mean * m) / (agg["count"] + m)
    return data[col].map(smooth).fillna(global_mean)

data["Outlet_TargetEnc"] = smoothed_mean_encode(data, train_mask, "Outlet_Identifier", "Item_Outlet_Sales", m=10)
data["ItemType_TargetEnc"] = smoothed_mean_encode(data, train_mask, "Item_Type", "Item_Outlet_Sales", m=20)
data["OutletCombo_TargetEnc"] = smoothed_mean_encode(data, train_mask, "Outlet_Combo", "Item_Outlet_Sales", m=10)

# ---- Label encode categoricals ----
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
y_log = np.log1p(train_data["Item_Outlet_Sales"])
X_test = test_data[feature_cols]

print("Feature columns:", feature_cols)
print("Train shape:", X.shape, "Test shape:", X_test.shape)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "ridge": Ridge(alpha=5.0),
    "gbm": GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.85, random_state=42
    ),
    "rf": RandomForestRegressor(
        n_estimators=400, max_depth=6, min_samples_leaf=8, random_state=42, n_jobs=-1
    ),
    "xgb": xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.85,
        colsample_bytree=0.7, reg_lambda=1.5, random_state=42, n_jobs=-1
    ),
}

oof_preds = {name: np.zeros(len(X)) for name in models}
test_preds = {name: np.zeros(len(X_test)) for name in models}

for name, model in models.items():
    fold_rmses = []
    test_pred_folds = np.zeros((5, len(X_test)))
    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]
        m = model
        m.fit(X_tr, y_tr)
        val_pred_log = m.predict(X_val)
        val_pred = np.expm1(val_pred_log)
        val_pred = np.clip(val_pred, 0, None)
        oof_preds[name][val_idx] = val_pred
        true_val = np.expm1(y_val)
        rmse = np.sqrt(np.mean((true_val - val_pred) ** 2))
        fold_rmses.append(rmse)
        test_pred_folds[fold_i] = np.clip(np.expm1(m.predict(X_test)), 0, None)
    test_preds[name] = test_pred_folds.mean(axis=0)
    print(f"{name}: CV RMSE = {np.mean(fold_rmses):.4f} (+/- {np.std(fold_rmses):.4f})")

y_true = train_data["Item_Outlet_Sales"].values

# Try weighted blend, search a few weight combos
from itertools import product
best_rmse = 1e18
best_weights = None
weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
names = list(models.keys())
for w_ridge in weight_options:
    for w_gbm in weight_options:
        for w_rf in weight_options:
            w_xgb = 1.0 - w_ridge - w_gbm - w_rf
            if w_xgb < 0 or w_xgb > 1:
                continue
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
submission.to_csv("bigmart_submission_v2.csv", index=False)
print("Saved bigmart_submission_v2.csv, shape:", submission.shape)
print(submission.head())
