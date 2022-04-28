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
- add extra trees with evalautions
- work on the random forest
- saving models
-grid search for the trees (with val_2, CV is automaticallz implemented)

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

print('RMSE_ridge: ',rmse_ridge)
print('MAE_ridge: ', mae_ridge)


#%% polynomial transformation of independent variables

# degree 2
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly_2 = poly_features.fit_transform(X_train)

#degree 3
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly_3 = poly_features.fit_transform(X_train)


#%% polynomial regression with ridge regularisation - degree 2

#training model
poly_reg_w_ridge = Ridge()
poly_reg_w_ridge.fit(X_poly_2, y_train) 

#predicting outcome
y_predicted_poly_2 = poly_reg_w_ridge.predict(X_val_1)

# evaluation
rmse_poly2= np.sqrt(mean_squared_error(y_val_1, y_predicted_poly_2))
mae_poly2= mean_absolute_error(y_val_1, y_predicted_poly_2)


r2_poly2= r2_score(y_val_1,y_predicted_poly_2)

print('RMSE_ridge_poly_2: ',rmse_poly2)
print('MAE_ridge_poly_2: ', mae_poly2)
print ('R2_ridge_poly2:', r2_poly2)

# saving the baseline model
joblib.dump(poly_reg_w_ridge, "/models/poly_reg_w_ridge.pkl")

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
search = GridSearchCV(poly_reg_w_ridge, param, scoring='rmse', n_jobs=-1, cv=X_val_1)
result = search.fit(X_poly_2, y_train)

# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

#%% polynomial regression with ridge regularisation - degree 3

poly_reg_w_ridge_3= Ridge()
poly_reg_w_ridge_3.fit(X_poly_3, y_train)

y_predicted_poly_3 = poly_reg_w_ridge_3.predict(X_val_1)

# evaluation
rmse_poly3= np.sqrt(mean_squared_error(y_val_1, y_predicted_poly_3))
mae_poly3= mean_absolute_error(y_val_1, y_predicted_poly_3)


r2_poly3= r2_score(y_val_1,y_predicted_poly_3)

print('RMSE_ridge_poly_2: ',rmse_poly3)
print('MAE_ridge_poly_2: ', mae_poly3)
print ('R2_ridge_poly2:', r2_poly3)

# saving the model
joblib.dump(poly_reg_w_ridge_3, "/models/poly_reg_w_ridge_3.pkl")

#loading if needed
#poly_reg_w_ridge_3_loaded=joblib.load("/models/poly_reg_w_ridge_3.pkl")

# comparing parameters - definitions

param = {
    'alpha':[.0001, 0.001,0.01, 0.01,1],
    'fit_intercept':[True,False],
    'normalize':[True,False],
'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
       }
# search
search = GridSearchCV(poly_reg_w_ridge_3, param, scoring='rmse', n_jobs=-1, cv=X_val_1)
result = search.fit(X_poly_3, y_train)

# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

#%% Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

forest_reg =  RandomForestRegressor()

# fit the model to our training data
forest_reg.fit(X_train, y_train)

# prediction for X_val_1 (not needed because this is done during cross validation)
# predicted_forest = forest_reg.predict(X_val_1)

# compute cross validation scores
from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest_reg, X_val_1, y_val_1, scoring = "neg_root_mean_squared_error", cv = 5) 
forest_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# forest_mse = mean_squared_error(y_val_1, y_pred)
# print(forest_reg_mse)

display_scores(forest_rmse_scores)

# saves the model 
import joblib
joblib.dump(forest_reg, "models/RandomForests.pkl")

# load the model if needed
RandomForest_loaded=joblib.load("models/RandomForests.pkl")


#%% ExtraTrees (Extremely Randomized Trees)

# trades more bias for a lower variance, much faster to train than RandomForests

from sklearn.ensemble import ExtraTreesRegressor

extra_reg =  ExtraTreesRegressor()

# fit the model to our training data
extra_reg.fit(X_train, y_train)

# prediction for X_val_1 (not needed because this is done during cross validation)
# predicted_forest = forest_reg.predict(X_val_1)

# compute cross validation scores
from sklearn.model_selection import cross_val_score
scores = cross_val_score(extra_reg, X_val_1, y_val_1, scoring = "neg_root_mean_squared_error", cv = 5) # scoring = ???
extra_rmse_scores = np.sqrt(-scores)

# forest_mse = mean_squared_error(y_val_1, y_pred)
# print(forest_reg_mse)

display_scores(extra_rmse_scores)

# saves the model 
import joblib
joblib.dump(extra_reg, "models/ExtraTrees.pkl")

# load the model if needed
# ExtraTreers_loaded=joblib.load("models/ExtraTrees.pkl")
