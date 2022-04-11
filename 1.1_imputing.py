#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:01:27 2022

@author: Jo
"""

#%% imputing for NaN's

# One type of imputation algorithm is univariate, which imputes values in the i-th feature dimension 
# using only non-missing values in that feature dimension (e.g. impute.SimpleImputer). By contrast, 
# multivariate imputation algorithms use the entire set of available feature dimensions to estimate 
# the missing values (e.g. impute.IterativeImputer).

# for now, to save runtime, we will use a small sample created above
PISA_reduced_sample = pd.read_csv("/Volumes/GoogleDrive/My Drive/PISA_Revisited/data/PISA_reduced.csv")

#%% multivariate imputation

# itearative imputation (k nearest neighbor or random forest - where is linear?) check out these links
# https://scikit-learn.org/stable/modules/impute.html
# https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#sphx-glr-auto-examples-impute-plot-missing-values-py

# to compare different imputers, we create a function which will score the results on the differently 
# imputed data

rng = np.random.RandomState(42)

from sklearn.ensemble import RandomForestRegressor

# To use the experimental IterativeImputer, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline


N_SPLITS = 5
regressor = RandomForestRegressor(random_state=0)

#%% simple and fast version (univariate)

# simple imputer, works only with numerical data
#from sklearn.impute import SimpleImputer
#imputer = SimpleImputer(strategy = "median")
#imputer.fit(PISA_reduced_sample)

imputer.statistics_
PISA_reduced_sample.median().values

X = imputer.transform(PISA_sample_100)

# This doesn't work yet but I don't know why
# PISA_sample_transformed = pd.DataFrame(X, columns = PISA_sample_100.comlumns, index = PISA_sample_100.index)

#%% any other preprocessing (pattern missingness? etc.)



