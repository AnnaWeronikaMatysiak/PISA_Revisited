#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:31:29 2022

@author: Jo&Anna
"""

"""
TO DO:
- polynomial regression with ridge penalty  - add evaluations
- add extra trees with evalautions
- work on the decision trees
- apply evaluation structure from the lab 8.
- performance plots
- add hyperparamether tuning
- research pipeline and gride search
"""

#%% necessary packages

import pandas as pd
#from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np

#%% call setup file
import runpy
runpy.run_path(path_name = '/0_setup.py')

# imports sys, sklearn, numpy, os, matplotlib, pathlib
# checks versions, sets wd, sets random.seed 42, specifies plots
# defines function save_fig()

#%% read in data
#X_test = pd.read_csv("/My Drive/PISA_Revisited/data/X_test.csv") 
#y_test = pd.read_csv("/My Drive/PISA_Revisited/data/y_test.csv")

X_train = pd.read_csv("/My Drive/PISA_Revisited/data/X_train.csv")
y_train = pd.read_csv("/My Drive/PISA_Revisited/data/y_train.csv")

X_val_1 = pd.read_csv("/My Drive/PISA_Revisited/data/X_val_1.csv") 
y_val_1 = pd.read_csv("/My Drive/PISA_Revisited/data/y_val_1.csv")

X_val_2 = pd.read_csv("/My Drive/PISA_Revisited/data/X_val_2.csv") 
y_val_2 = pd.read_csv("/My Drive/PISA_Revisited/data/y_val_2.csv")

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
mse_ridge = mean_squared_error(y_val_1, predicted_ridge)
rmse_ridge= np.sqrt(mean_squared_error(y_val_1, predicted_ridge))
mae_ridge=mean_absolute_error(y_val_1, predicted_ridge)

print('MSE_ridge: ',mse_ridge)
print('RMSE_ridge: ',rmse_ridge)
print('MAE_ridge: ', mae_ridge)


#%% polynomial regressions degree=2

#train
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X_train)

lin_reg_pol = LinearRegression()
lin_reg_pol.fit(X_poly, y_train)
lin_reg_pol.intercept_, lin_reg_pol.coef_

#test
validation_X_poly = poly_features.fit_transform(X_val_1)
validation_y_poly = lin_reg_pol.predict(validation_X_poly)

#evaluation
mse_poly = mean_squared_error(y_val_1,validation_y_poly)
rmse_poly= np.sqrt(mean_squared_error(y_val_1, validation_y_poly))
mae_poly = mean_absolute_error(y_val_1,validation_y_poly)
r2_poly=r2_score(y_val_1,validation_y_poly)

print('MSE_polynomial: ',mse_poly)
print('RMSE_polynomial: ',rmse_poly)
print('MAE_polynomial: ', mae_poly)
print('R2_polynomial: ',r2_poly)


#%% polynomial regression with ridge regularisation - to be continued
from sklearn.linear_model import Ridge
poly_reg_w_ridge = Ridge()
poly_reg_w_ridge.fit(X_poly, y_train)   

poly_reg_w_ridge.coef_
poly_reg_w_ridge.intercept_

#grid search - alphas fro ridge and degrees fro polynomial

#%% Random Forest Regressor - to be continued

from sklearn.ensemble import RandomForestRegressor

rnd_rgr =  RandomForestRegressor(n_estimators = 100, max_leaf_nodes = 10, max_features = 1.0, max_samples = 1.0, n_jobs = -1)

# for the whole training set, only using 10% of features and 10% of observations for each try
# rnd_rgr =  RandomForestRegressor(n_estimators = 100, max_leaf_nodes = 10, max_features = 0.2, max_samples = 1.0, n_jobs = -1)

# fit the model to our training data
rnd_rgr.fit(X_train, y_train)

#%% extra trees

