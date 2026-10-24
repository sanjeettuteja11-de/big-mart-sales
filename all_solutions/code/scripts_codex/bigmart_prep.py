import pandas as pd, numpy as np, os
os.chdir(os.path.expanduser("~/Downloads"))
train = pd.read_csv("train_v9rqX0R.csv")
test = pd.read_csv("test_AbJTz2l.csv")

train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], ignore_index=True, sort=False)
print(data.shape)

# --- Fix Item_Fat_Content typos ---
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
    'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'
})
print(data['Item_Fat_Content'].value_counts())

# --- Item_Identifier prefix -> category (FD/DR/NC) ---
data['Item_Type_Combined'] = data['Item_Identifier'].str[:2]
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({
    'FD':'Food','DR':'Drinks','NC':'Non-Consumable'
})
print(data['Item_Type_Combined'].value_counts())

# Non-consumables shouldn't have fat content -> set to 'Non-Edible'
data.loc[data['Item_Type_Combined']=='Non-Consumable','Item_Fat_Content'] = 'Non-Edible'
print(data['Item_Fat_Content'].value_counts())

# --- Item_Weight: impute by mean per Item_Identifier, fallback overall mean ---
item_weight_avg = data.pivot_table(values='Item_Weight', index='Item_Identifier', aggfunc='mean')
def impute_weight(row):
    if pd.isnull(row['Item_Weight']):
        idn = row['Item_Identifier']
        if idn in item_weight_avg.index and not pd.isnull(item_weight_avg.loc[idn,'Item_Weight']):
            return item_weight_avg.loc[idn,'Item_Weight']
        return data['Item_Weight'].mean()
    return row['Item_Weight']
data['Item_Weight'] = data.apply(impute_weight, axis=1)
print("Item_Weight nulls after impute:", data['Item_Weight'].isnull().sum())

# --- Outlet_Size: impute by mode per Outlet_Type ---
outlet_size_mode = data.dropna(subset=['Outlet_Size']).groupby('Outlet_Type')['Outlet_Size'].agg(lambda x: x.mode()[0])
print(outlet_size_mode)
def impute_size(row):
    if pd.isnull(row['Outlet_Size']):
        return outlet_size_mode.get(row['Outlet_Type'], 'Small')
    return row['Outlet_Size']
data['Outlet_Size'] = data.apply(impute_size, axis=1)
print("Outlet_Size nulls after impute:", data['Outlet_Size'].isnull().sum())
print(data['Outlet_Size'].value_counts())

# --- Item_Visibility: zeros are implausible -> replace with mean visibility per item ---
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier', aggfunc='mean')
def impute_vis(row):
    if row['Item_Visibility'] == 0:
        idn = row['Item_Identifier']
        v = visibility_avg.loc[idn,'Item_Visibility'] if idn in visibility_avg.index else np.nan
        return v if v>0 else data['Item_Visibility'].mean()
    return row['Item_Visibility']
data['Item_Visibility'] = data.apply(impute_vis, axis=1)
print("zero visibility remaining:", (data['Item_Visibility']==0).sum())

# Visibility relative to item's mean visibility (Mean_Ratio)
data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg.loc[x['Item_Identifier'],'Item_Visibility'] if visibility_avg.loc[x['Item_Identifier'],'Item_Visibility']>0 else 1, axis=1)

# --- Outlet_Years ---
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']  # dataset known to be as-of 2013

print(data[['Item_Weight','Outlet_Size','Item_Visibility']].isnull().sum())

data.to_pickle('bigmart_prepped.pkl')
print("saved prepped data")
print(data.head())
