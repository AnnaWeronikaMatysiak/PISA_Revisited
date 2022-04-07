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

#%% define dependent and independent variables

X_sample = PISA_sample_10.loc[:, PISA_sample_10.columns.drop(['read_score'])]
y_sample = PISA_sample_10[["read_score"]]


#%% Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

rnd_rgr =  RandomForestRegressor(n_estimators = 100, max_leaf_nodes = 10, max_features = 1.0, max_samples = 1.0, n_jobs = -1)

# for the whole training set, only using 10% of features and 10% of observations for each try
# rnd_rgr =  RandomForestRegressor(n_estimators = 100, max_leaf_nodes = 10, max_features = 0.2, max_samples = 1.0, n_jobs = -1)

# fit the model to our training data
rnd_rgr.fit(X_sample, y_sample)
