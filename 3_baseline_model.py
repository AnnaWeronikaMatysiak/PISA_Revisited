# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:14:00 2022

@author: Anna
"""

"""
TO DO:
- check the evaluation metrics, 
- check if that's what Eric recommended 
- add plots
- find out at which point we need to use test and when shall we compare between the models
- are we able/allowed to compare them now already?
- at the end, change PISA_sample_100 to the final sample we are using'
"""

#%% import packages
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge

#%% call setup file
#import runpy
#runpy.run_path(path_name = '/My Drive/PISA_Revisited/0_setup.py')
PISA_sample_100 = pd.read_csv("/My Drive/PISA_Revisited/data/PISA_sample_100.csv")
PISA_sample_100 = PISA_sample_100.drop(columns=["VER_DAT", "CNT", "CYC", "STRATUM"])


#%%
PISA_sample_100 = pd.read_csv("/My Drive/PISA_Revisited/data/PISA_sample_100.csv")
#utill we drop these variables from the whole data set,
# I manualy exclude them from sample
PISA_sample_100 = PISA_sample_100.drop(columns=["VER_DAT", "CNT", "CYC", "STRATUM"])

X_train=PISA_sample_100.drop(columns=["read_score"])
y_train=PISA_sample_100["read_score"]

#becuase y is an array, I change it back to data frame
y_train=y_train.to_frame()


#for test untill we have clean data I replace the data columns
# with NAs witht the colum means. This does not work for "test" column, so I drop it. 
#By the way, what is that column about?
y_train=y_train.apply(lambda x: x.fillna(x.mean()))
X_train=X_train.apply(lambda x: x.fillna(x.mean())) 
X_train=X_train.drop(columns=["test"])

#code to check if any of the columns have NAs:
#y_train.isnull().any()

#%% linear regression
lin_reg= LinearRegression()
lin_reg.fit(X_train, y_train)   

lin_reg.coef_
lin_reg.intercept_
predicted=lin_reg.predict(X_train)
predicted.heac()

#%% evaluation 
"""questions to Eric 
- are we supposed to calculate it now (at this stage) on train or test tests?
- why our mse and rmse are the same?"""
mse = mean_squared_error(y_train, predicted, squared=False)
rmse= np.sqrt(mean_squared_error(y_train, predicted))
mae=mean_absolute_error(y_train, predicted)

print(mse)
print(rmse)
print(mae)

#%%plots
plt.plot(X_train, predicted, color="blue", linewidth=3)

#%% polynomial regression

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X_train)
X_train[0]

lin_reg_pol = LinearRegression()
lin_reg_pol.fit(X_poly, y_train)
lin_reg_pol.intercept_, lin_reg.coef_
predicted_poly=lin_reg_pol.predict(X_poly)
predicted_poly.head()

#%% evaluation
#ask Eric as above
mse_poly = mean_squared_error(y_train, predicted_poly, squared=False)
rmse_poly= np.sqrt(mean_squared_error(y_train, predicted_poly))
mae_poly=mean_absolute_error(y_train, predicted_poly)

print(mse_poly)
print(rmse_poly)
print(mae_poly)

#%% ridge regression
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X_train, y_train)
predicted_ridge=ridge_reg.predict(X_train)

#%% evaluation 
#ask Eric - do we need more or only these measures

mse_ridge = mean_squared_error(y_train, predicted_ridge, squared=False)
rmse_ridge= np.sqrt(mean_squared_error(y_train, predicted_ridge))
mae_ridge=mean_absolute_error(y_train, predicted_ridge)

print(mse_ridge)
print(rmse_ridge)
print(mae_ridge)

