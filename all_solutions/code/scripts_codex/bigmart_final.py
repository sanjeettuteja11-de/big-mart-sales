import pandas as pd, numpy as np, os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

os.chdir(os.path.expanduser("~/Downloads"))
data_dum = pd.read_pickle('bigmart_encoded.pkl')

train = data_dum[data_dum['source']=='train'].copy().drop(columns=['source'])
test = data_dum[data_dum['source']=='test'].copy().drop(columns=['source','Item_Outlet_Sales'])

target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
predictors = [c for c in train.columns if c not in [target]+IDcol]

X = train[predictors].values
y = train[target].values
Xtest = test[predictors].values

gbm = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                 min_samples_leaf=50, subsample=0.8, random_state=42)
gbm.fit(X, y)

preds = gbm.predict(Xtest)
preds = np.clip(preds, 0, None)  # sales can't be negative

submission = test[['Item_Identifier','Outlet_Identifier']].copy()
submission['Item_Outlet_Sales'] = preds
submission.to_csv('bigmart_submission.csv', index=False)
print(submission.head(10))
print(submission.shape)

# feature importances
imp = pd.Series(gbm.feature_importances_, index=predictors).sort_values(ascending=False)
print(imp)
