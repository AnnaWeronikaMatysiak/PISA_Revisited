#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:31:29 2022

@author: Jo&Anna
"""

"""
TO DO - for both files 4 and 5: (different order)
Ania:
- polynomial regression with ridge penalty  - add evaluations
- apply evaluation structure from the lab 8., cross-validation (p.73)
- research pipeline and gride search
- saving models
- add rmse to all

Johanna:
- all done so far

Max/together:
- performance plots
- evaluate on the test
- present the 10 best predictors of the best model
"""

#%% necessary packages

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV

#%% call setup file
import runpy
runpy.run_path(path_name = '/0_setup.py')

# imports sys, sklearn, numpy, os, matplotlib, pathlib
# checks versions, sets wd, sets random.seed 42, specifies plots
# defines function save_fig()

#%% read in data
#X_test = pd.read_csv("/My Drive/PISA_Revisited/data/X_test.csv") 
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
drop_first_entry(y_train)
drop_first_entry(X_val_1)
drop_first_entry(y_val_1)
drop_first_entry(X_val_2)
drop_first_entry(y_val_2)

#%% ridge regression
# in case we need to scale it:
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X_std = scaler.fit_transform(X_train)

#training
ridge_reg = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10], normalize=True)
ridge_reg_model=ridge_reg.fit(X_train, y_train)

#evaluation
predicted_ridge=ridge_reg.predict(X_val_1)

#to check which alpha was used
ridge_reg_model.alpha_
rmse_ridge= np.sqrt(mean_squared_error(y_val_1, predicted_ridge))
mae_ridge=mean_absolute_error(y_val_1, predicted_ridge)
r2_ridge=r2_score(y_val_1, predicted_ridge)

print('RMSE_ridge: ',rmse_ridge)
print('MAE_ridge: ', mae_ridge)

#adding values to table
d2 = {'Model': ['Baseline: Linear Regression', 'Ridge (0.01)'], 'RMSE': [round(rmse, 4), round(rmse_ridge, 4)], 'MAE': [round(mae, 4), round(mae_ridge, 4)], 'R2': [round(r2, 4), round(r2_ridge, 4)]}
table_baseline_ridge = pd.DataFrame(data=d2)
table_baseline_ridge
table_baseline.to_latex("baseline_ridge_table.tex", index=False, caption="Comparing Model Performance")


#%% polynomial transformation of independent variables

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly_2 = poly_features.fit_transform(X_train)
X_poly_2=pd.DataFrame(X_poly_2)
X_val_1_poly=poly_features.fit_transform(X_val_1)



#%% polynomial regression with ridge regularisation - degree 2

#training model
poly_reg_w_ridge = Ridge()
poly_reg_w_ridge.fit(X_poly_2, y_train) 

#predicting outcome
y_predicted_poly_2 = poly_reg_w_ridge.predict(X_val_1_poly)

#change type
y_predicted_poly_2=pd.DataFrame(y_predicted_poly_2)
y_predicted_poly_2.head()

# evaluation
rmse_poly2= np.sqrt(mean_squared_error(y_val_1, y_predicted_poly_2))
mae_poly2= mean_absolute_error(y_val_1, y_predicted_poly_2)
r2_poly2= r2_score(y_val_1,y_predicted_poly_2)

print('RMSE_ridge_poly_2: ',rmse_poly2)
print('MAE_ridge_poly_2: ', mae_poly2)
print ('R2_ridge_poly2:', r2_poly2)

# saving the baseline model
joblib.dump(poly_reg_w_ridge, "/models/poly_reg_w_ridge.pkl")

#saving in table
d3 = {'Model': ['Baseline: Linear Regression', 'Ridge (0.01)', 'Polynomial Ridge'],
'RMSE': [round(rmse, 4), round(rmse_ridge, 4), round(rmse_poly2, 4)], 
'MAE': [round(mae, 4), round(mae_ridge, 4), round(mae_poly2, 4)], 
'R2': [round(r2, 4), round(r2_ridge, 4), round(r2_poly2, 4)]}
table_base_ridge_poly = pd.DataFrame(data=d3)
table_base_ridge_poly
table_base_ridge_poly.to_latex("base_ridge_poly_table.tex", index=False, caption="Comparing Model Performance")

#loading if needed
#poly_reg_w_ridge_loaded=joblib.load("/models/poly_reg_w_ridge.pkl")

# comparing parameters - definitions
param = {
    'alpha':[.0001, 0.001,0.01, 0.01,1],
    'fit_intercept':[True,False],
    'normalize':[True,False],
'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
       }

# search
search = GridSearchCV(poly_reg_w_ridge, param, scoring='neg_mean_squared_error', n_jobs=-1, cv=X_val_1)
result = search.fit(X_poly_2, y_train)

# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)


#%% Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

forest_reg =  RandomForestRegressor()

# fit the model to our training data
forest_reg.fit(X_train, y_train)

# prediction for X_val_1
y_pred = forest_reg.predict(X_val_1)

# evaluate
forest_mse = mean_squared_error(y_val_1, y_pred)
forest_r2 = r2_score(y_val_1, y_pred)
forest_mae = mean_absolute_error(y_val_1, y_pred)
print(forest_r2)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)


forest_rmse # result: 71.15293129026736

# alternative: compute cross validation scores
#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(forest_reg, X_train, y_train, scoring = "neg_root_mean_squared_error", cv = 5) 

# saves the model 
import joblib
joblib.dump(forest_reg, "models/RandomForests.pkl")

#save to evaluation table
d4 = {'Model': ['Baseline: Linear Regression', 'Ridge (0.01)', 'Polynomial Ridge', 'Random Forest'],
'RMSE': [round(rmse, 4), round(rmse_ridge, 4), round(rmse_poly2, 4), round(forest_rmse, 4)], 
'MAE': [round(mae, 4), round(mae_ridge, 4), round(mae_poly2, 4), round(forest_mae, 4)], 
'R2': [round(r2, 4), round(r2_ridge, 4), round(r2_poly2, 4), round(forest_r2, 4)]}
table_base_ridge_poly_forest = pd.DataFrame(data=d4)
table_base_ridge_poly_forest
table_base_ridge_poly_forest.to_latex("base_ridge_poly_forest.tex", index=False, caption="Comparing Model Performance")

# load the model if needed
forest_reg = joblib.load("models/RandomForests.pkl")


#%% ExtraTrees (Extremely Randomized Trees)

# trades more bias for a lower variance, much faster to train than RandomForests

from sklearn.ensemble import ExtraTreesRegressor

extra_reg =  ExtraTreesRegressor()

# fit the model to our training data
extra_reg.fit(X_train, y_train)

# prediction for X_val_1
y_pred = extra_reg.predict(X_val_1)

# evaluate
extra_mse = mean_squared_error(y_val_1, y_pred)
extra_r2 = r2_score(y_val_1, y_pred)
extra_mae = mean_absolute_error(y_val_1, y_pred)
print(extra_mse)
extra_rmse = np.sqrt(extra_mse)

extra_rmse # result: 70.29067166539122

# saves the model 
import joblib
joblib.dump(extra_reg, "models/ExtraTrees.pkl")

#saving in table

d5 = {'Model': ['Baseline: Linear Regression', 'Ridge (0.01)', 'Polynomial Ridge', 'Random Forest', 'Extra Trees'],
'RMSE': [round(rmse, 4), round(rmse_ridge, 4), round(rmse_poly2, 4), round(forest_rmse, 4), round(extra_rmse, 4)], 
'MAE': [round(mae, 4), round(mae_ridge, 4), round(mae_poly2, 4), round(forest_mae, 4), round(extra_mae, 4)], 
'R2': [round(r2, 4), round(r2_ridge, 4), round(r2_poly2, 4), round(forest_r2, 4), round(extra_r2, 4)]}
table_base_ridge_poly_forest_extra = pd.DataFrame(data=d5)
table_base_ridge_poly_forest_extra
table_base_ridge_poly_forest_extra.to_latex("base_ridge_poly_forest_extra.tex", index=False, caption="Comparing Model Performance")

# load the model if needed
extra_reg = joblib.load("models/ExtraTrees.pkl")

# %%
