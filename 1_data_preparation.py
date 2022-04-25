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

# read in if needed
PISA_raw_100000 = pd.read_csv("data/PISA_raw_100000.csv")

#%% feature selection

# create array with features to keep (read in column from excel doc)
codebook = pd.read_excel('codebook/codebook_covariates_PISA.xlsx')
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

# save plot (0 axis???)
save_fig("Distribution of Missingness of Covariates")


#%% imputation (still change names)

# type "pip install missingpy" in the console for installation
from missingpy import MissForest

X = PISA_selection

imputer = MissForest(max_iter = 5, n_estimators = 25, max_features = 50, n_jobs = -1, random_state = 42)

# cat_vars : int or array of ints containing column indices of categorical variable(s)
cat_names = ['CNTRYID', 'ST004D01T',  'ST005Q01TA', 'ST007Q01TA', 'ST022Q01TA', 'IMMIG']

def get_col_indices(df, names):
    return df.columns.get_indexer(names)

cat_indices = get_col_indices(PISA_selection, cat_names)

# fit imputer
imputer.fit(X, cat_vars = cat_indices)

# apply imputer
PISA_imputed = imputer.transform(X)

# convert to pandas dataframe and add column names from before
PISA_imputed = pd.DataFrame(PISA_imputed, columns = PISA_selection.columns)

# save result as csv file (just as a backup)
PISA_imputed.to_csv("data/PISA_imputed.csv")


#%% OneHotEncoding of categorical variables

# select categorical features
# 'country', 'gender', 'mother_school', 'father_school', 'home_language', 'immig'
# see variable cat_names

# transform categorical variables using OneHotEncoder and ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(transformers = [("cat", OneHotEncoder(), cat_names)], remainder = "passthrough")
PISA_encoded = transformer.fit_transform(PISA_imputed)

# save as csv
PISA_encoded.to_csv("data/PISA_encoded.csv")

#%% normalization

# MinMaxScaler normalizes the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# fit scaler
scaler.fit(PISA_encoded)

# scale data (OneHot-variables stay the same, they are already between 0 and 1)
PISA_prepared = scaler.transform(PISA_encoded)

# convert to pandas dataframe and add column names from before
PISA_prepared = pd.DataFrame(PISA_prepared, columns = PISA_encoded.columns)

# save result as csv file (just as a backup)
PISA_prepared.to_csv("data/PISA_prepared.csv")


#%% create boys and girls subsets for feature importance comparison

# filter by gender (see how e.g. female is called after OneHotEncoding...)
PISA_male = PISA_prepared[PISA_prepared["female"] == "0"] 
PISA_female = PISA_prepared[PISA_prepared["female"] == "1"]

# save as csv
PISA_male.to_csv("data/PISA_male.csv")
PISA_female.to_csv("data/PISA_female.csv")

