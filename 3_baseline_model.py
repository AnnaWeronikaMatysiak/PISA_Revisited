# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:14:00 2022

@author: Anna
"""

#%% import packages
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#%% call setup file
#import runpy
#runpy.run_path(path_name = '/My Drive/PISA_Revisited/0_setup.py')
PISA_sample_100=pd.read_csv("/My Drive/PISA_Revisited/data/PISA_sample_100.csv/")
PISA_sample_100 = PISA_sample_100.columns.drop(["VER_DAT", "CNT"])

# imports sys, sklearn, numpy, os, matplotlib, pathlib
# checks versions, sets wd, sets random.seed 42, specifies plots
# defines function save_fig()


#%% preceding steps to enable us run these models (work in progress)

#we should make secision on NAs
#PISA_sample_100=PISA_sample_100.fillna(value=PISA_sample_100.mean())

#we need to use either LabelEncoder or OneHot thingy
#OneHotEncoder().fit_transform(PISA_sample_10)


#from sklearn.preprocessing import LabelEncoder
#def Encoder(PISA_sample_100):
#          columnsToEncode = list(PISA_sample_100.select_dtypes(include=['category','object']))
#          le = LabelEncoder()
#          for feature in columnsToEncode:
#              try:
#                  PISA_sample_100[feature] = le.fit_transform(PISA_sample_100[feature])
#              except:
#                  print('Error encoding '+feature)
#          return PISA_sample_100

#Encoder(PISA_sample_100)
      

#%%

X=PISA_sample_100.loc[:, PISA_sample_100.columns.drop(['read_score'])]
y = PISA_sample_100[["read_score"]]

#%% linear regression
lin_reg= LinearRegression()
lin_reg.fit(X, y)   

lin_reg.coef_
lin_reg.intercept_
lin_reg.predict(X)

#%% RMSE

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

mse = mean_squared_error(y_true, y_predicted, squared=False)
rmse= np.sqrt(metrics.mean_squared_error(y_true, y_predicted)

#%% polynomial regression
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]

lin_reg_pol = LinearRegression()
lin_reg_pol.fit(X_poly, y)
lin_reg_pol.intercept_, lin_reg.coef_
lin_reg_pol.predict(X)

#%% cost funtions

#%% plots

#%% validation and evaluation measures


