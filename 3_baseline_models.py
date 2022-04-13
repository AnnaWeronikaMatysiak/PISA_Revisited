# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:14:00 2022

@author: Anna
"""

"""
TO DO:
- next stage: plots
- next stage: visualisation of error
- next stage: use whole data set
- next stage:predictions' description an ordering them
"""

#%% import packages
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

#%% call setup file
#import runpy
#runpy.run_path(path_name = '/My Drive/PISA_Revisited/0_setup.py')

#%% read in data

#becuase y is an array, code to transform it to dataframe:
#y_train=y_train.to_frame()
#code to check if any of the columns have NAs:
#y_train.isnull().any()


### MID-TERM:
midterm_train = pd.read_csv("/My Drive/PISA_Revisited/data/midterm_train.csv") 
midterm_validation=pd.read_csv("/My Drive/PISA_Revisited/data/midterm_val.csv")

X_train=midterm_train.drop(columns=["read_score"])
y_train=midterm_train["read_score"]

X_validation=midterm_validation.drop(columns=["read_score"])
y_validation=midterm_validation["read_score"]

#%% linear regression

#training
lin_reg= LinearRegression()
lin_reg.fit(X_train, y_train)   

lin_reg.coef_
lin_reg.intercept_

#evaluation
y_predicted=lin_reg.predict(X_validation)

mse = mean_squared_error(y_validation, y_predicted)
rmse= np.sqrt(mean_squared_error(y_validation, y_predicted))
mae= mean_absolute_error(y_validation, y_predicted)
print('MSE_linear: ',mse)
print('RMSE_linear: ',rmse)
print('MSE_linear: ', mae)


#%% ridge regression
# in case we need to scale it:
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X_std = scaler.fit_transform(X_train)

#training
ridge_reg = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10], normalize=True)
ridge_reg_model=ridge_reg.fit(X_train, y_train)

#evaluation
predicted_ridge=ridge_reg.predict(X_validation)

#to check which alpha was used
ridge_reg_model.alpha_
mse_ridge = mean_squared_error(y_validation, predicted_ridge)
rmse_ridge= np.sqrt(mean_squared_error(y_validation, predicted_ridge))
mae_ridge=mean_absolute_error(y_validation, predicted_ridge)

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
validation_X_poly = poly_features.fit_transform(X_validation)
validation_y_poly = lin_reg_pol.predict(validation_X_poly)

mse_poly = mean_squared_error(y_validation,validation_y_poly)
rmse_poly= np.sqrt(mean_squared_error(y_validation, validation_y_poly))
mae_poly = mean_absolute_error(y_validation,validation_y_poly)
r2_poly=r2_score(y_validation,validation_y_poly)
print('MSE_polynomial: ',mse_poly)
print('RMSE_polynomial: ',rmse_poly)
print('MAE_polynomial: ', mae_poly)
print('R2_polynomial: ',r2_poly)

#%% code is ready but needs a lot of space and power - possibly to run in colab
"""poly_features_3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly_3 = poly_features_3.fit_transform(X_train)

lin_reg_pol_3 = LinearRegression()
lin_reg_pol_3.fit(X_poly_3, y_train)
lin_reg_pol_3.intercept_, lin_reg_pol_3.coef_

# The coefficients
print ('Coefficients: ', lin_reg_pol_3.coef_)
print ('Intercept: ',lin_reg_pol_3.intercept_)

#y_predicted_poly=lin_reg_pol.predict(X_validation) 
validation_X_poly_3 = poly_features_3.fit_transform(X_validation)
validation_y_poly_3 = lin_reg_pol_3.predict(validation_X_poly_3)

mse_poly_3 = mean_squared_error(y_validation,validation_y_poly_3)
rmse_poly_3= np.sqrt(mean_squared_error(y_validation, validation_y_poly_3))
r2_poly_3=r2_score(y_validation,validation_y_poly_3)
print(mse_poly_3)
print(rmse_poly_3)
print(r2_poly_3)"""

#%% polynomial with ridge - still working on it
"""
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X_train)

poly_ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10], normalize=True)
poly_ridge.fit(X_poly, y_train)
poly_ridge.intercept_, poly_ridge.coef_



# The coefficients
print ('Coefficients: ', poly_ridge.coef_)
print ('Intercept: ',poly_ridge.intercept_)

#y_predicted_poly=lin_reg_pol.predict(X_validation) 
validation_poly_ridge = poly_features_3.fit_transform(X_validation)
validation_y_poly_ridge = lin_reg_pol_3.predict(validation_poly_ridge)

mse_poly_ridge = mean_squared_error(y_validation,validation_poly_ridge)
rmse_poly_ridge= np.sqrt(mean_squared_error(y_validation, validation_poly_ridge))

print(mse_poly_ridge)
print(rmse_poly_ridge)"""






