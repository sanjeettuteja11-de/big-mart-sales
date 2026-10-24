import pandas as pd, numpy as np, os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

os.chdir(os.path.expanduser("~/Downloads"))
data = pd.read_pickle('bigmart_prepped.pkl')

le = LabelEncoder()
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])

cat_cols = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type']
data_dum = pd.get_dummies(data, columns=cat_cols)

drop_cols = ['Item_Type','Outlet_Establishment_Year']
data_dum = data_dum.drop(columns=drop_cols)

train = data_dum[data_dum['source']=='train'].copy()
test = data_dum[data_dum['source']=='test'].copy()
train = train.drop(columns=['source'])
test = test.drop(columns=['source','Item_Outlet_Sales'])

target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
predictors = [c for c in train.columns if c not in [target]+IDcol]
print(len(predictors), predictors)

X = train[predictors].values
y = train[target].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def rmse_cv(model):
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    return -scores.mean(), scores.std()

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.05, random_state=42, max_iter=5000),
    'DecisionTree(depth=8,minleaf=100)': DecisionTreeRegressor(max_depth=8, min_samples_leaf=100, random_state=42),
    'RandomForest(200,d6,minleaf100)': RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, min_samples_leaf=50, random_state=42),
}

results = {}
for name, m in models.items():
    rmse, std = rmse_cv(m)
    results[name] = rmse
    print(f"{name}: RMSE={rmse:.2f} (+/- {std:.2f})")

data_dum.to_pickle('bigmart_encoded.pkl')
print("saved encoded")
