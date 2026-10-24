import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

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
cat_cols = ["Item_Fat_Content", "Outlet_Identifier", "Outlet_Size", "Outlet_Location_Type",
            "Outlet_Type", "Item_Type_Combined", "Item_Type", "Outlet_Combo", "Item_Code3"]
for col in cat_cols:
    data[col + "_Enc"] = le.fit_transform(data[col].astype(str))

train_data = data[data["source"] == "train"].reset_index(drop=True).copy()
test_data = data[data["source"] == "test"].reset_index(drop=True).copy()
y = train_data["Item_Outlet_Sales"]

target_enc_cols = ["Outlet_Identifier", "Item_Type", "Outlet_Combo", "MRPBand_Outlet", "ItemType_Outlet", "ItemCode3_Outlet"]
smooth_m = {"Outlet_Identifier": 10, "Item_Type": 20, "Outlet_Combo": 10, "MRPBand_Outlet": 10, "ItemType_Outlet": 15, "ItemCode3_Outlet": 15}

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

# Best single model so far: ExtraTrees (1089), test variance recalibration on it
oof = np.zeros(len(X_base))
test_pred_accum = np.zeros(len(X_test_base))
for (tr_idx, val_idx), (X_full, X_test_full) in zip(fold_indices, fold_features):
    X_tr, X_val = X_full.iloc[tr_idx], X_full.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    m = ExtraTreesRegressor(n_estimators=500, max_depth=7, min_samples_leaf=10, random_state=SEED, n_jobs=-1)
    m.fit(X_tr, y_tr)
    pred = np.clip(m.predict(X_val), 0, None)
    oof[val_idx] = pred
    test_pred_accum += np.clip(m.predict(X_test_full), 0, None) / N_SPLITS

y_true = y.values
base_rmse = np.sqrt(np.mean((y_true - oof) ** 2))
print("ET baseline RMSE:", base_rmse)

# Try variance-inflation recalibration: pred_new = mean + k*(pred - mean), search k
best_rmse, best_k = base_rmse, 1.0
for k in np.arange(1.0, 1.6, 0.02):
    mean_pred = oof.mean()
    recal = mean_pred + k * (oof - mean_pred)
    recal = np.clip(recal, 0, None)
    r = np.sqrt(np.mean((y_true - recal) ** 2))
    if r < best_rmse:
        best_rmse, best_k = r, k

print(f"Best variance-inflation k={best_k:.2f}, RMSE={best_rmse:.4f} (vs base {base_rmse:.4f})")

# Try isotonic-style: fit a simple linear recalibration pred -> true (this is basically Ridge(pred) but 1D)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(oof.reshape(-1,1), y_true)
recal2 = np.clip(lr.predict(oof.reshape(-1,1)), 0, None)
r2 = np.sqrt(np.mean((y_true - recal2) ** 2))
print(f"Linear recalibration: slope={lr.coef_[0]:.4f}, intercept={lr.intercept_:.2f}, RMSE={r2:.4f}")
