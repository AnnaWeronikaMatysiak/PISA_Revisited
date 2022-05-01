#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:31:29 2022

@author: Jo&Anna
"""

"""
TO DO - for both files 4 and 5:
Max/together:
- performance plots
- evaluate on the test
- present the 10 best predictors of the best model
- if we have time: combine models in an ensemble (like in the lab)
"""

#%% necessary packages

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error

#%% call setup file
import runpy
runpy.run_path(path_name = '/0_setup.py')

# imports sys, sklearn, numpy, os, matplotlib, pathlib
# checks versions, sets wd, sets random.seed 42, specifies plots
# defines function save_fig()

#%% read in data
X_test = pd.read_csv("/My Drive/PISA_Revisited/data/X_test.csv") 
#y_test = pd.read_csv("/My Drive/PISA_Revisited/data/y_test.csv")

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

X_val_1 = pd.read_csv("data/X_val_1.csv") 
y_val_1 = pd.read_csv("data/y_val_1.csv")

X_val_2 = pd.read_csv("data/X_val_2.csv") 
y_val_2 = pd.read_csv("data/y_val_2.csv")

# if needed, drop first column that was automatically generated 
# -> dimensions should be 205 columns in X and 1 in y
def drop_first_entry(df):
    df.drop(df.columns[[0]], axis = 1, inplace = True)

drop_first_entry(X_train)
drop_first_entry(X_val_1)
drop_first_entry(X_test)
drop_first_entry(X_val_2)


#%% initial ridge regression - mid term report
#training
ridge_reg = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10], normalize=True)
ridge_reg_model=ridge_reg.fit(X_train, y_train)

#evaluation
predicted_ridge=ridge_reg.predict(X_val_1)

#to check which alpha was used
ridge_reg_model.alpha_
rmse_ridge= np.sqrt(mean_squared_error(y_val_1, predicted_ridge))
mae_ridge=mean_absolute_error(y_val_1, predicted_ridge)
r_2_ridge=r2_score(y_val_1, predicted_ridge)

print('RMSE_ridge: ',rmse_ridge) #result: RMSE=68.5100950967124
print('MAE_linear:', mae_ridge) # result: MAE=54.39827907320501
print('R_2_linear:', r_2_ridge) #result: R_2=0.5922063884528036

#%% simple ridge regression
#trianing
ridge= Ridge()
ridge_model=ridge.fit(X_train, y_train)

#predicting
predicted_ridge=ridge_model.predict(X_val_1)

#evaluation
rmse_ridge= np.sqrt(mean_squared_error(y_val_1, predicted_ridge))
print('RMSE_ridge: ',rmse_ridge) #result: 68.51986262316217

#saving model
joblib.dump(ridge_model, "models/ridge.pkl")

#cross-validation on the ridge
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(ridge, X_val_1, y_val_1, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

print('RMSE_ridge_after_cross_validation: ',np.mean(np.sqrt(np.abs(scores))))
#result: 69.15260004052853

#%% fine-tuning

model=make_pipeline(Ridge())

#putting together a parameter grid to search over using grid search
#alphas based on the first tuning when the alpha was 6
params={
    'ridge__fit_intercept':[True,False],
    'ridge__alpha':[4,5,6,7,8],
    'ridge__solver':[ 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag',
'saga']
}
#grid search
gs=GridSearchCV(model,params,n_jobs=-1,cv=cv, scoring="neg_mean_squared_error")
#fitting gs to data
gs.fit(X_val_2, y_val_2)

#print(gs.best_estimator_.get_params()["ridge"])
print('Best Score: %s' % np.sqrt(-gs.best_score_))
print('Best Hyperparameters: %s' % gs.best_params_)

#results:
#this was on the train data, so probaby overfitted
#Best Score: 48.089040334701856
#Best Hyperparameters: {'ridge__alpha': 5, 'ridge__fit_intercept': True, 'ridge__solver': 'cholesky'}

#this is a result on the correct val_2 data
#Ridge(alpha=5, solver='sag')
#Best Score: 67.9992874603911
#Best Hyperparameters: {'ridge__alpha': 5, 'ridge__fit_intercept': True, 'ridge__solver': 'sag'}

#%%save the model
final_ridge_model= gs.best_estimator_
joblib.dump(final_ridge_model, "models/final_ridge_model.pkl")

#%% feature transformation to degree=2
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly_2 = poly_features.fit_transform(X_train)
X_poly_2=pd.DataFrame(X_poly_2)
X_val_1_poly=poly_features.fit_transform(X_val_1)
X_val_1_poly=pd.DataFrame(X_val_1_poly)
X_val_2_poly=poly_features.fit_transform(X_val_2)
X_val_2_poly=pd.DataFrame(X_val_2_poly)
#%% polynomial regression with ridge regularisation - degree 2
#training model
poly_reg_w_ridge = Ridge()
poly_reg_w_ridge.fit(X_poly_2, y_train) 

#predicting outcome
y_predicted_poly_2 = poly_reg_w_ridge.predict(X_val_1_poly)

# evaluation
rmse_poly2= np.sqrt(mean_squared_error(y_val_1, y_predicted_poly_2))
r2_poly2= r2_score(y_val_1,y_predicted_poly_2)

print('RMSE_ridge_poly_2: ',rmse_poly2)
print ('R2_ridge_poly2:', r2_poly2)

# saving the model
joblib.dump(poly_reg_w_ridge, "models/poly_reg_w_ridge.pkl")

#%% fine-tuning polynomial model with ridge 

model2=make_pipeline(Ridge())

#putting together a parameter grid to search over using grid search
params={
    'ridge__fit_intercept':[True,False],
    'ridge__alpha':[4,5,6,7,8],
    'ridge__solver':[ 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag',
'saga']
}
#setting up the grid search
gs2=GridSearchCV(model2,params,n_jobs=-1,cv=cv, scoring="neg_mean_squared_error")
#fitting gs to data
gs2.fit(X_val_2_poly, y_val_2)

print(gs2.best_estimator_.get_params()["ridge"])
print('Best Score: %s' % np.sqrt(-gs2.best_score_))
print('Best Hyperparameters: %s' % gs2.best_params_)

#%%save the model
final_poly_ridge= gs2.best_estimator_
joblib.dump(final_poly_ridge, "models/final_poly_ridge_tuned.pkl")
