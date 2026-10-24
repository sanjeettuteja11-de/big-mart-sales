import pandas as pd, numpy as np, os
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

os.chdir(os.path.expanduser("~/Downloads"))
data_dum = pd.read_pickle('bigmart_encoded.pkl')

train = data_dum[data_dum['source']=='train'].copy().drop(columns=['source'])
test = data_dum[data_dum['source']=='test'].copy().drop(columns=['source','Item_Outlet_Sales'])

target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
predictors = [c for c in train.columns if c not in [target]+IDcol]

X = train[predictors].values
y = train[target].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def rmse_cv(model):
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -scores.mean(), scores.std()

# Tuned GBM
gbm = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                 min_samples_leaf=50, subsample=0.8, random_state=42)
print("GBM tuned:", rmse_cv(gbm))

# XGBoost
xgbm = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
                         random_state=42, n_jobs=-1)
print("XGB:", rmse_cv(xgbm))

xgbm2 = xgb.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.7, min_child_weight=30,
                         reg_lambda=2, random_state=42, n_jobs=-1)
print("XGB2:", rmse_cv(xgbm2))

rf = RandomForestRegressor(n_estimators=400, max_depth=5, min_samples_leaf=100, random_state=42, n_jobs=-1)
print("RF tuned:", rmse_cv(rf))
