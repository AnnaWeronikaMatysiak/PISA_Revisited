# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:14:00 2022

@author: Anna
"""

#%% import packages
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#%% call setup file
import runpy
runpy.run_path(path_name = '/Volumes/GoogleDrive/My Drive/PISA_Revisited/0_setup.py')
PISA_sample_10=pd.read_csv("/My Drive/PISA_Revisited/data/PISA_sample_10.csv/")

# imports sys, sklearn, numpy, os, matplotlib, pathlib
# checks versions, sets wd, sets random.seed 42, specifies plots
# defines function save_fig()

#%% dependent and independent variables
X=PISA_sample_10.loc[:, PISA_sample_10.columns.drop(['read_score'])]
y = PISA_sample_10[["read_score"]]

#%% version with normal equation - add predictions
X_b = np.c_[np.ones((100, 1)), X] 
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

theta_best

#%% linear regression
lin_reg= LinearRegression()
lin_reg.fit(X, y)   

lin_reg.coef_
lin_reg.intercept_

#%% polynomial regression
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_

#%% plots

#%% validation and evaluation measures

