# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:14:00 2022

@author: Anna
"""

"""
TO DO:
- add the validation sets and test models on them
- add plots
- add models with changes parameters like alpha, degree of polymomials etc. 
- at the end, change PISA_sample_100 to the final sample we are using'
- check the metrics for each model
- check aplication for polynomial regression if ours is correct
"""

#%% import packages
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
#from sklearn.preprocessing import PolynomialFeatures
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge

#%% call setup file
#import runpy
#runpy.run_path(path_name = '/My Drive/PISA_Revisited/0_setup.py')

#%%
PISA_sample_100 = pd.read_csv("/My Drive/PISA_Revisited/data/PISA_sample_100.csv")
#utill we drop these variables from the whole data set, I manualy exclude them from sample
PISA_sample_100 = PISA_sample_100.drop(columns=["VER_DAT", "CNT", "CYC", "STRATUM"])

X_train=PISA_sample_100.drop(columns=["read_score"])
y_train=PISA_sample_100["read_score"]

#becuase y is an array, I change it back to data frame
y_train=y_train.to_frame()


#for current tests untill we have clean data I replaced the data columns
# with NAs witht the colum means. This does not work for "test" column, so I drop it. 
#By the way, what is that column about?
y_train=y_train.apply(lambda x: x.fillna(x.mean()))
X_train=X_train.apply(lambda x: x.fillna(x.mean())) 
X_train=X_train.drop(columns=["test"])

#code to check if any of the columns have NAs:
#y_train.isnull().any()

# repeat that for validation sets
y_validation=...
X_validation=...

#%% linear regression
lin_reg= LinearRegression()
lin_reg.fit(X_train, y_train)   

lin_reg.coef_
lin_reg.intercept_

y_predicted=lin_reg.predict(X_validation)
y_predicted.head()

#%% evaluation 
mse = mean_squared_error(y_validation, y_predicted)
rmse= np.sqrt(mean_squared_error(y_validation, y_predicted))
print(mse)
print(rmse)

#%%plots
#plt.plot(X_train, y_predicted, color="blue", linewidth=3)

#%% ridge regression
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X_train, y_train)
predicted_ridge=ridge_reg.predict(X_validation)

#%% evaluation 

mse_ridge = mean_squared_error(y_validation, predicted_ridge)
rmse_ridge= np.sqrt(mean_squared_error(y_validation, predicted_ridge))

print(mse_ridge)
print(rmse_ridge)

#%% polynomial regression - NOT READY, LEARNING ABOUT IT NOW, 
#is this appropirate for the data we have already to we 
#have to apply poly measures to train and test?

"""poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X_train)
X_train[0]

lin_reg_pol = LinearRegression()
lin_reg_pol.fit(X_poly, y_train)
lin_reg_pol.intercept_, lin_reg.coef_
y_predicted_poly=lin_reg_pol.predict(#X_poly or X_evaluation) #check
y_predicted_poly.head()"""

#%% evaluation 
"""mse_poly = mean_squared_error(y_validation, y_predicted_poly)
rmse_poly= np.sqrt(mean_squared_error(y_validation, y_predicted_poly))
mae_poly=mean_absolute_error(y_validation, y_predicted_poly)

print(mse_poly)
print(rmse_poly)
print(mae_poly)"""


