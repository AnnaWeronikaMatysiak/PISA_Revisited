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
- we should maybe not decide for one single imputer for everything - how do we decide this?

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


#%% MissForest simple try (disregarding categorical variables)

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

# type "pip install missingpy" in the console for installation
from missingpy import MissForest

# max_iter: default = 10. The maximum of iterations.
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
# in order to do so, we decrease the number of variables for now by dropping more

# create bigger reduced sample
# call raw sample 1000
PISA_raw_1000 = pd.read_csv("/Volumes/GoogleDrive/My Drive/PISA_Revisited/data/PISA_raw_1000.csv")
# renaming: function from file 1
rename_read_score_female(PISA_raw_1000)
# removing string columns: function from file 1
remove_string_columns(PISA_raw_1000)
# removing rows with a missingness ofer x percent: function from file 1
PISA_reduced_1000 = drop_columns_with_missingness(PISA_raw_1000, 5)

# dropping students without reading score doesnt work somehow. Also not clear if 
# "read_score" or 'PV1READ', because renaming does not always work.
# PISA_reduced_1000 = drop_students_without_read_score(PISA_reduced_1000)

# next line does not work, but there are 7 NaN's in the sample... :(
# maybe better finalize function in file 1 and see if that works!
# PISA_reduced_1000 = PISA_reduced_1000.dropna(subset=['read_score'], inplace=True)

# -> PISA_reduced_1000 is a only numerical, reduced sample with 1000 observations
# right now there is 7 students with missing read_score, needs to be changed later...

X = PISA_reduced_1000
imputer = MissForest(max_iter = 5, n_estimators = 50, max_features = 100, n_jobs = -1, random_state = 42)
# cat_vars : int or array of ints containing column indices of categorical variable(s)
# create array
cat_vars = np.array([0,1,2,5,7,8,9,10,11,12,13,14,15])

imputer.fit(X, cat_vars = cat_vars)
PISA_reduced_imputed = imputer.transform(X)

# convert to pandas dataframe
PISA_reduced_imputed = pd.DataFrame(PISA_reduced_imputed)

# save result as csv file
PISA_reduced_imputed.to_csv("data/PISA_1000_imputed.csv")

imputer.statistics_

# statistics_ : Dictionary of length two
    # The first element is an array with the mean of each numerical feature
    # being imputed while the second element is an array of modes of
    # categorical features being imputed (if available, otherwise it
    # will be None).

# creating a function. (finish later when it is more precise)
# "array" needs to be an array with the indices of categorical variables
def restricted_missforest(dataframe, array):
    X = dataframe
    imputer = MissForest(max_iter = 5, n_estimators = 50, max_features = 100, n_jobs = -1, random_state = 42)
    imputer.fit(X, cat_vars = array)
    dataframe = imputer.transform(X)
    dataframe = pd.DataFrame(dataframe)

    


#%% compare different iterative imputation methods

# https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html#sphx-glr-auto-examples-impute-plot-iterative-imputer-variants-comparison-py

X_full = PISA_reduced_sample.drop(columns=["read_score"])
y_full = PISA_reduced_sample["read_score"]

n_samples, n_features = X_full.shape

# ...
#
#

#%% alternative multivariate imputation (links above, this is only the beginning)

# itearative imputation (k nearest neighbor or random forest - where is linear?) check out these links
# https://scikit-learn.org/stable/modules/impute.html
# https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#sphx-glr-auto-examples-impute-plot-missing-values-py

# ...
#
#

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



