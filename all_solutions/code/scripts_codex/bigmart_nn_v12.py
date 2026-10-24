import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

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
data["Outlet_Item_Count"] = data.groupby("Outlet_Identifier")["Item_Identifier"].transform("count")
data["Outlet_Avg_MRP"] = data.groupby("Outlet_Identifier")["Item_MRP"].transform("mean")
data["Item_MRP_to_OutletAvg"] = data["Item_MRP"] / data["Outlet_Avg_MRP"]
data["Item_MRP_log"] = np.log1p(data["Item_MRP"])
data["Price_per_Weight"] = data["Item_MRP"] / data["Item_Weight"].replace(0, np.nan)
data["Price_per_Weight"] = data["Price_per_Weight"].fillna(data["Price_per_Weight"].mean())

# Categorical columns for embeddings
cat_cols_emb = ["Outlet_Identifier", "Item_Type", "Item_Fat_Content", "Outlet_Size",
                "Outlet_Location_Type", "Outlet_Type", "Item_Type_Combined"]
le_dict = {}
for col in cat_cols_emb:
    le = LabelEncoder()
    data[col + "_idx"] = le.fit_transform(data[col].astype(str))
    le_dict[col] = le

numeric_cols = [
    "Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Years",
    "Item_Visibility_MeanRatio", "Item_MRP_Band", "Item_Type_MRP_Mean",
    "Outlet_Item_Count", "Outlet_Avg_MRP", "Item_MRP_to_OutletAvg",
    "Item_MRP_log", "Price_per_Weight",
]

train_data = data[data["source"] == "train"].reset_index(drop=True).copy()
test_data = data[data["source"] == "test"].reset_index(drop=True).copy()
y = train_data["Item_Outlet_Sales"].values
y_log = np.log1p(y)

scaler = StandardScaler()
train_num = scaler.fit_transform(train_data[numeric_cols])
test_num = scaler.transform(test_data[numeric_cols])

cat_cardinalities = {col: data[col + "_idx"].nunique() for col in cat_cols_emb}
print("Cardinalities:", cat_cardinalities)

device = torch.device("cpu")

class EmbedMLP(nn.Module):
    def __init__(self, cat_cards, n_numeric):
        super().__init__()
        self.embeds = nn.ModuleDict()
        emb_total = 0
        for col, card in cat_cards.items():
            dim = min(20, (card + 1) // 2 + 1)
            self.embeds[col] = nn.Embedding(card, dim)
            emb_total += dim
        in_dim = emb_total + n_numeric
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, cat_inputs, num_input):
        embs = [self.embeds[col](cat_inputs[col]) for col in self.embeds]
        x = torch.cat(embs + [num_input], dim=1)
        return self.mlp(x).squeeze(-1)

def make_tensors(df, num_arr):
    cat_tensors = {col: torch.tensor(df[col + "_idx"].values, dtype=torch.long) for col in cat_cols_emb}
    num_tensor = torch.tensor(num_arr, dtype=torch.float32)
    return cat_tensors, num_tensor

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train_data))
test_pred_accum = np.zeros(len(test_data))

train_cat_all, train_num_all = make_tensors(train_data, train_num)
test_cat_all, test_num_all = make_tensors(test_data, test_num)

for fold_i, (tr_idx, val_idx) in enumerate(kf.split(train_num)):
    model = EmbedMLP(cat_cardinalities, len(numeric_cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    tr_cat = {col: train_cat_all[col][tr_idx] for col in cat_cols_emb}
    val_cat = {col: train_cat_all[col][val_idx] for col in cat_cols_emb}
    tr_num = train_num_all[tr_idx]
    val_num = train_num_all[val_idx]
    tr_y = torch.tensor(y_log[tr_idx], dtype=torch.float32)
    val_y_true = y[val_idx]

    n = len(tr_idx)
    batch_size = 256
    best_val_rmse = 1e18
    best_state = None
    patience, bad_epochs = 15, 0

    for epoch in range(200):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            batch_cat = {col: tr_cat[col][idx] for col in cat_cols_emb}
            batch_num = tr_num[idx]
            batch_y = tr_y[idx]
            opt.zero_grad()
            pred = model(batch_cat, batch_num)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_pred_log = model(val_cat, val_num).numpy()
            val_pred = np.clip(np.expm1(val_pred_log), 0, None)
            val_rmse = np.sqrt(np.mean((val_y_true - val_pred) ** 2))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred_log = model(val_cat, val_num).numpy()
        val_pred = np.clip(np.expm1(val_pred_log), 0, None)
        oof[val_idx] = val_pred
        test_pred_log = model(test_cat_all, test_num_all).numpy()
        test_pred = np.clip(np.expm1(test_pred_log), 0, None)
        test_pred_accum += test_pred / N_SPLITS

    fold_rmse = np.sqrt(np.mean((val_y_true - val_pred) ** 2))
    print(f"Fold {fold_i}: best val RMSE = {fold_rmse:.4f}")

overall_rmse = np.sqrt(np.mean((y - oof) ** 2))
print(f"\nEmbedding NN overall CV RMSE: {overall_rmse:.4f}")

np.save("nn_oof_preds.npy", oof)
np.save("nn_test_preds.npy", test_pred_accum)
print("Saved NN OOF and test predictions for potential stacking")
