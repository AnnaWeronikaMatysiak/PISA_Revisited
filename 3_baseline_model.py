# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:14:00 2022

@author: Anna
"""

"""
TO DO:
- visualisation of error measures (look at session 5)
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

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

X_val_1 = pd.read_csv("data/X_val_1.csv") 
y_val_1 = pd.read_csv("data/y_val_1.csv")

X_val_2 = pd.read_csv("data/X_val_2.csv") 
y_val_2 = pd.read_csv("data/y_val_2.csv")


#%% linear regression

#training
lin_reg= LinearRegression()
lin_reg.fit(X_train, y_train)   

lin_reg.coef_
lin_reg.intercept_

#predicting
y_predicted=lin_reg.predict(X_val_1)

#evaluation
rmse= np.sqrt(mean_squared_error(y_val_1, y_predicted))
mae= mean_absolute_error(y_val_1, y_predicted)

print('RMSE_linear: ',rmse) # result: 68.5229128573592
# print('MAE_linear: ', mae)

# saves the model 
import joblib
joblib.dump(lin_reg, "models/LinearRegression.pkl")

# load the model if needed
# lin_reg = joblib.load("models/LinearRegression.pkl")

#%%Saving Baseline Metrics in Table
d = {'Model': ['Baseline: Linear Regression'], 'RMSE': [round(rmse, 4)], 'MAE': [round(mae, 4)]}
table_baseline = pd.DataFrame(data=d)
table_baseline

# saving
import joblib

joblib.dump(lin_reg, "/models/baseline.pkl")

#loading
#baseline_loaded=joblib.load("/models/baseline.pkl")



# %%
