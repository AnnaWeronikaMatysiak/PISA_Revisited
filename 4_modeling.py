#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:31:29 2022

@author: Jo
"""

#%% call setup file

import runpy
runpy.run_path(path_name = '/Volumes/GoogleDrive/My Drive/PISA_Revisited/0_setup.py')

# imports sys, sklearn, numpy, os, matplotlib, pathlib
# checks versions, sets wd, sets random.seed 42, specifies plots
# defines function save_fig()

#%% additional packages

import pandas as pd

#%% read in data

PISA_sample_10 = pd.read_csv("data/PISA_sample_10.csv")
PISA_sample_100 = pd.read_csv("data/PISA_sample_100.csv")
PISA_sample_1000 = pd.read_csv("data/PISA_sample_1000.csv")
midterm_train = pd.read_csv("/My Drive/PISA_Revisited/data/midterm_train.csv") 
midterm_validation=pd.read_csv("/My Drive/PISA_Revisited/data/midterm_val.csv")

#%% define dependent and independent variables

#X_=PISA_sample_10.drop(columns=["read_score"])
#y_sample=PISA_sample_10["read_score"]

#becuase y is an array, I change it back to data frame
#y_sample=y_sample.to_frame()

#code to check if any of the columns still have NAs:
#y_train.isnull().any()

X_train=midterm_train.drop(columns=["read_score"])
y_train=midterm_train["read_score"]

y_train=y_train.to_frame()


X_validation=midterm_validation.drop(columns=["read_score"])
y_validation=midterm_validation["read_score"]

y_validation=y_train.to_frame()
#%% Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

rnd_rgr =  RandomForestRegressor(n_estimators = 100, max_leaf_nodes = 10, max_features = 1.0, max_samples = 1.0, n_jobs = -1)

# for the whole training set, only using 10% of features and 10% of observations for each try
# rnd_rgr =  RandomForestRegressor(n_estimators = 100, max_leaf_nodes = 10, max_features = 0.2, max_samples = 1.0, n_jobs = -1)

# fit the model to our training data
rnd_rgr.fit(X_train, y_train)

#%% Linear SVM base line 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


polynomial_svm_clf = Pipeline([("poly_features", PolynomialFeatures(degree=3)),
                               ("scaler", StandardScaler()),("svm_clf", 
                                LinearSVC(C=10, loss="hinge")) ])
polynomial_svm_clf.fit(X_train, y_train)


