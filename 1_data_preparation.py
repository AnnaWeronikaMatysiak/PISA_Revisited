#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:56:52 2022

@author: Jo
"""

#%% call setup file

import openpyxl
import runpy
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt

runpy.run_path(path_name = '0_setup.py')

# imports sys, sklearn, numpy, os, matplotlib, pathlib
# checks versions, sets wd, sets random.seed 42, specifies plots
# defines function save_fig()

runpy.run_path(path_name = 'functions.py')

# runs preprocessing functions

#%% import additional packages

import pandas as pd


#%% read in data

PISA_raw = pd.read_csv("/Volumes/GoogleDrive/My Drive/PISA_Revisited/data/PISA_student_data.csv")

# rename target variable "PV1READ" = "read_score"
PISA_raw.rename(columns = {'PV1READ':'read_score'}, inplace = True)

# drop students with missing target variable
PISA_raw = PISA_raw.dropna(subset=['read_score'])

# create random sample of 100.000 observations -> reduction of data for the scope of the project
PISA_raw_100000 = PISA_raw.sample(100000)

# save as csv
PISA_raw_100000.to_csv("data/PISA_raw_100000.csv")


#%% feature selection

# create array with features to keep (read in column from excel doc)
codebook = pd.read_excel('/Users/max.eckert/Documents/GitHub/PISA_Revisited/codebook/codebook_covariates_PISA.xlsx')
covariates = codebook.iloc[:,3]
#transform to array, dropping SCHLTYPE and adding read_score so it doesn't get dropped
covariates_array = covariates.to_numpy()
covariates_array_new = np.delete(covariates_array, 1)
covariates_array_new = np.append(covariates_array_new, 'read_score')

# select the features included in the array
PISA_selection = PISA_raw_100000[covariates_array_new]

# save as csv
PISA_selection.to_csv("data/PISA_selection.csv")

# plot with missingness
NaN_count_rel = PISA_selection.isnull().sum()/len(PISA_selection)*100
NaN_count_rel.sort_values(ascending=False)

plt.hist(NaN_count_rel, 25)
plt.xlabel('Percentage Missingness')
plt.ylabel('Number of Covariates')
plt.title('Distribution of Missingness of Covariates',fontweight ="bold")
plt.show()


#%% OneHotEncoding of categorical variables (book? 2nd chapter)

# MAKE SURE COLUMN NAMES ARE GOOD -> look at book 2nd chapter

# select categorical features
PISA_cat = np.array(['country', 'home_language', 'immig', 'school_ownership', 'gender', 'mother_school', 'father_school'])


"""# apply OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
PISA_cat_1hot = cat_encoder.fit_transform(PISA_cat)
"""

# -> merge somehow or replace in original data set OR

# using a pipeline?
"""from sklearn.compose import ColumnTransformer

# LOOK INTO THIS AND TRY OUT WHEN WE HAVE DATA
pipeline = ColumnTransformer("cat", OneHotEncoder, PISA_cat)
PISA_encoded = pipeline.fit_transform(PISA)"""

# save as csv
PISA_encoded.to_csv("data/PISA_encoded.csv")

#%% normalization

# MinMaxScaler normalizes the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# fit scaler
scaler.fit(PISA_encoded)

# scale data
PISA_scaled = scaler.transform(PISA_encoded)

# convert to pandas dataframe AND ADD COLUMN NAMES FROM BEFORE!
PISA_scaled = pd.DataFrame(PISA_scaled, columns = PISA_encoded.columns)

# save result as csv file (just as a backup)
PISA_scaled.to_csv("data/midterm_scaled.csv")


#%% imputation (still change names)

# type "pip install missingpy" in the console for installation
from missingpy import MissForest

X = PISA_scaled

imputer = MissForest(max_iter = 5, n_estimators = 30, max_features = 80, n_jobs = -1, random_state = 42)

# cat_vars : int or array of ints containing column indices of categorical variable(s)
cat_vars = np.array(range(17)) # -> check if it is possible somehow with NAMES of columns to get indices

# fit imputer
imputer.fit(X, cat_vars = cat_vars)

# apply imputer
PISA_imputed = imputer.transform(X)

# convert to pandas dataframe AND ADD COLUMN NAMES FROM BEFORE!
PISA_imputed = pd.DataFrame(midterm_imputed, columns = midterm_reduced.columns)

# save result as csv file (just as a backup)
midterm_imputed.to_csv("data/midterm_imputed.csv")


#%% create boys and girls subsets for later feature importance interpretation

# filter by gender (see how it is called after OneHotEncoding!!!)


