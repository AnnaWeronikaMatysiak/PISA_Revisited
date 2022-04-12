#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:01:27 2022

@author: Jo
"""
"""

To Do's'
- create imputing and compare different approaches (on different variables?)

Problems
- we should maybe not decide for imputer for everything - how do we decide this?

Endproduct should be a dataframe saved to csv without NA's. 
The file takes the PISA_reduced.csv file as a starting point 

"""


#%% call setup file

import runpy
runpy.run_path(path_name = '/Volumes/GoogleDrive/My Drive/PISA_Revisited/0_setup.py')

# imports sys, sklearn, numpy, os, matplotlib, pathlib
# checks versions, sets wd, sets random.seed 42, specifies plots
# defines function save_fig()

#%% additional packages

import pandas as pd
import numpy as np

#%% imputing for NaN's

# One type of imputation algorithm is univariate, which imputes values in the i-th feature dimension 
# using only non-missing values in that feature dimension (e.g. impute.SimpleImputer). By contrast, 
# multivariate imputation algorithms use the entire set of available feature dimensions to estimate 
# the missing values (e.g. impute.IterativeImputer).

# for now, to save runtime, we will use a small sample (10 observations) created in file 1
PISA_reduced_sample = pd.read_csv("/Volumes/GoogleDrive/My Drive/PISA_Revisited/data/PISA_reduced_sample.csv")

# create bigger reduced sample
PISA_raw_1000 = pd.read_csv("/Volumes/GoogleDrive/My Drive/PISA_Revisited/data/PISA_raw_1000.csv")
rename_read_score_female(PISA_raw_1000)
remove_string_columns(PISA_raw_1000)
PISA_1000 = drop_columns_with_missingness(PISA_raw_1000, 5)



#%% MissForest simple try

# description: https://towardsdatascience.com/missforest-the-best-missing-data-imputation-algorithm-4d01182aed3
# and official page: https://pypi.org/project/missingpy/

# type "pip install missingpy" in the console for installation
from missingpy import MissForest

X = PISA_reduced_sample

imputer = MissForest()
X_imputed = imputer.fit_transform(X)

# missing: 
    # - specified categorical variables
    # - restrictions on interations (to reduce runtime)
    
#%% MissForest more advanced, indicating factor variables and restricting the algorithm

# documentation https://pypi.org/project/missingpy/

# max_iter: default= 10. The maximum of iterations.
# n_estimators : integer, optional (default=100). The number of trees in the forest.
# max_features : The number of features to consider when looking for the best split
# n_jobs : int or None, optional (default=None) The number of jobs to run in parallel 
        # for both `fit` and `predict`. ``None`` means 1 
        # ``-1`` means using all processors. 

# -> proposed parameters for our bigger sample:
    # max_iter = 5, n_estimators = 50, max_features = 100, 
    # n_jobs = -1, random_state = 42

# in order to use the fit the imputer, we need to use the fit() function indicating
# all categorical variables. 

# Create an array of integers with our categorical variables
# in order to do so, we decrease the number of variables for now. 
# code from file 1, function taking a dataframe and the percentage 
# of missingness with which columns should be dropped (here everything over 5%)
PISA_reduced_2 = drop_columns_with_missingness(PISA_reduced_sample, 5)


fit(self, X, y=None, cat_vars=None):
    Fit the imputer on X.

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Input data, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    cat_vars : int or array of ints, optional (default = None)
        An int or an array containing column indices of categorical
        variable(s)/feature(s) present in the dataset X.
        ``None`` if there are no categorical variables in the dataset.

    Returns
    -------
    self : object
        Returns self.



# statistics_ : Dictionary of length two
    # The first element is an array with the mean of each numerical feature
    # being imputed while the second element is an array of modes of
    # categorical features being imputed (if available, otherwise it
    # will be None).


#%% compare different iterative imputation methods

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

N_SPLITS = 5

rng = np.random.RandomState(0)

X_full = PISA_reduced_sample.drop(columns=["read_score"])
y_full = PISA_reduced_sample["read_score"]

n_samples, n_features = X_full.shape

# Estimate the score on the entire dataset, without the missing values
br_estimator = BayesianRidge()
score_full_data = pd.DataFrame(
    cross_val_score(
        br_estimator, X_full, y_full, scoring="neg_mean_squared_error", cv=N_SPLITS
    ),
    columns=["Full Data"],
)

# Add a single missing value to each row
X_missing = X_full.copy()
y_missing = y_full
missing_samples = np.arange(n_samples)
missing_features = rng.choice(n_features, n_samples, replace=True)
X_missing[missing_samples, missing_features] = np.nan

# Estimate the score after imputation (mean and median strategies)
score_simple_imputer = pd.DataFrame()
for strategy in ("mean", "median"):
    estimator = make_pipeline(
        SimpleImputer(missing_values=np.nan, strategy=strategy), br_estimator
    )
    score_simple_imputer[strategy] = cross_val_score(
        estimator, X_missing, y_missing, scoring="neg_mean_squared_error", cv=N_SPLITS
    )

# Estimate the score after iterative imputation of the missing values
# with different estimators
estimators = [
    BayesianRidge(),
    DecisionTreeRegressor(max_features="sqrt", random_state=0),
    ExtraTreesRegressor(n_estimators=10, random_state=0),
    KNeighborsRegressor(n_neighbors=15),
]
score_iterative_imputer = pd.DataFrame()
for impute_estimator in estimators:
    estimator = make_pipeline(
        IterativeImputer(random_state=0, estimator=impute_estimator), br_estimator
    )
    score_iterative_imputer[impute_estimator.__class__.__name__] = cross_val_score(
        estimator, X_missing, y_missing, scoring="neg_mean_squared_error", cv=N_SPLITS
    )

scores = pd.concat(
    [score_full_data, score_simple_imputer, score_iterative_imputer],
    keys=["Original", "SimpleImputer", "IterativeImputer"],
    axis=1,
)

# plot california housing results
fig, ax = plt.subplots(figsize=(13, 6))
means = -scores.mean()
errors = scores.std()
means.plot.barh(xerr=errors, ax=ax)
ax.set_title("California Housing Regression with Different Imputation Methods")
ax.set_xlabel("MSE (smaller is better)")
ax.set_yticks(np.arange(means.shape[0]))
ax.set_yticklabels([" w/ ".join(label) for label in means.index.tolist()])
plt.tight_layout(pad=1)
plt.show()


#%% alternative multivariate imputation (links above, this is only the beginning)

# itearative imputation (k nearest neighbor or random forest - where is linear?) check out these links
# https://scikit-learn.org/stable/modules/impute.html
# https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#sphx-glr-auto-examples-impute-plot-missing-values-py

# set variables
rng = np.random.RandomState(42)
X_PISA, y_PISA = PISA_reduced_sample

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



