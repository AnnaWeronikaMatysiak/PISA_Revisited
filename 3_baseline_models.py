# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:14:00 2022

@author: Anna
"""

"""
TO DO:
- change datasets to the "preprocessed 1000"
- visualisation of error
- predictions' description an ordering them
"""

#%% call setup file
import runpy
runpy.run_path(path_name = '/0_setup.py')

#%% necessary packages
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

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

