#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:56:52 2022

@author: Jo&Max
"""

#%% import packages

import pandas as pd
import runpy
import numpy as np
import matplotlib.pyplot as plt
#%% call setup file
runpy.run_path(path_name = '0_setup.py')

# imports sys, sklearn, numpy, os, matplotlib, pathlib
# checks versions, sets wd, sets random.seed 42, specifies plots
# defines function save_fig()

#%% runs preprocessing functions
runpy.run_path(path_name = '1a_functions.py')

#%% read in data

# raw data from pisa website
PISA_raw = pd.read_csv("/data/PISA_student_data.csv")

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


#%% imputation (still change names)

# type "pip install missingpy" in the console for installation
from missingpy import MissForest

X = PISA_selection

# fast version for now
imputer = MissForest(max_iter = 4, n_estimators = 10, max_features = 10, n_jobs = -1, random_state = 42)

# cat_vars : int or array of ints containing column indices of categorical variable(s)
# 'country', 'gender', 'mother_school', 'father_school', BINARIES ON HOME, 'home_language', 'immig', BINARIES ON READING
cat_names = ['CNTRYID', 'ST004D01T',  'ST005Q01TA', 'ST007Q01TA', "ST011Q01TA", 
             "ST011Q02TA", "ST011Q03TA", "ST011Q04TA", "ST011Q05TA", "ST011Q06TA",
             "ST011Q07TA", "ST011Q08TA", "ST011Q09TA", "ST011Q10TA", "ST011Q11TA", 
             "ST011Q12TA", "ST011Q16NA", 'ST022Q01TA', 'IMMIG', "ST153Q01HA", 
             "ST153Q02HA", "ST153Q03HA", "ST153Q04HA", "ST153Q05HA", "ST153Q06HA",
             "ST153Q08HA", "ST153Q09HA", "ST153Q10HA"]

def get_col_indices(df, names):
    return df.columns.get_indexer(names)

cat_indices = get_col_indices(PISA_selection, cat_names)

# fit imputer
imputer.fit(X, cat_vars = cat_indices)

# apply imputer
PISA_imputed = imputer.transform(X)

# convert to pandas dataframe and add column names from before
PISA_imputed = pd.DataFrame(PISA_imputed, columns = PISA_selection.columns)

# drop first column that was generated before
PISA_imputed = PISA_imputed.iloc[: , 1:]

# save result as csv file (just as a backup)
PISA_imputed.to_csv("data/PISA_imputed.csv")


#%% OneHotEncoding of categorical variables

# read in if needed
PISA_imputed = pd.read_csv("data/PISA_imputed.csv")

# transform categorical variables using OneHotEncoder and ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# select non-binary categorical features
# 'country', 'mother_school', 'father_school', 'home_language', 'immig'
non_binary_cat = ['CNTRYID', 'ST005Q01TA', 'ST007Q01TA', 'ST022Q01TA', 'IMMIG']

# transform categorical variables (this did not work when using the get_feature_names method afterwards...)
# transformer = ColumnTransformer(transformers = [("cat", encoder, non_binary_cat)], remainder = "passthrough")
# PISA_encoded = transformer.fit_transform(PISA_imputed)

# create df with categorical features
cat_df = PISA_imputed[non_binary_cat]
num_df = PISA_imputed.drop(non_binary_cat, axis=1)

encoder = OneHotEncoder(sparse = False)
encoder.fit(cat_df)

encoded_df = pd.DataFrame(encoder.fit_transform(cat_df))

# add column names
encoded_df.columns = encoder.get_feature_names(cat_df.columns)

# concatenate with numerical columns
PISA_encoded = pd.concat([num_df, encoded_df], axis=1)

# save as csv
PISA_encoded.to_csv("data/PISA_encoded.csv")

# read in if needed
PISA_encoded = pd.read_csv("data/PISA_encoded.csv")

#%% normalization

# move column "read_score" (target variable) to the end of the dataframe
temp_cols = PISA_encoded.columns.tolist()
index = PISA_encoded.columns.get_loc("read_score")
new_cols = temp_cols[0:index] + temp_cols[index+1:] + temp_cols[index:index+1]
PISA_encoded = PISA_encoded[new_cols]

# MinMaxScaler normalizes the data
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

scaler = MinMaxScaler()

# fit scaler
scaler.fit(PISA_encoded.loc[ : , PISA_encoded.columns != 'read_score'])

# choose all columns except of read_score (target) to be normalized
# OneHotEncoded columns don't change by applying MinMaxScaler
covar = PISA_encoded.loc[ : , PISA_encoded.columns != 'read_score'].columns

transformer = ColumnTransformer(transformers = [("norm", scaler, covar)], remainder = "passthrough")
PISA_prepared = transformer.fit_transform(PISA_encoded)

# convert to pandas dataframe and add column names from before
PISA_prepared = pd.DataFrame(PISA_prepared, columns = PISA_encoded.columns)

# drop first two columns that were unneccessarily generated during processing
PISA_prepared.drop(PISA_prepared.columns[[0, 1]], axis = 1, inplace = True)

# rename gender column for further observations
PISA_prepared.rename(columns = {'ST004D01T':'gender'}, inplace = True)

# save result as csv file (just as a backup)
PISA_prepared.to_csv("data/PISA_prepared.csv")

# PISA_prepared = pd.read_csv("data/PISA_prepared.csv")


#%% create boys and girls subsets for feature importance comparison

# filter by gender (see which is which later...)
PISA_female = PISA_prepared[PISA_prepared["gender"] == 0] 
PISA_male = PISA_prepared[PISA_prepared["gender"] == 1]

# save as csv
PISA_female.to_csv("data/PISA_female.csv")
PISA_male.to_csv("data/PISA_male.csv")

# drop first entries if needed
# drop_first_entry(PISA_female)
# drop_first_entry(PISA_male)

# split gender subsets
X_female = PISA_female.drop(columns=["read_score"])
y_female = PISA_female["read_score"]
y_female = y_female.to_frame()

X_male = PISA_male.drop(columns=["read_score"])
y_male = PISA_male["read_score"]
y_male = y_male.to_frame()

# save as csv
X_female.to_csv("data/X_female.csv")
y_female.to_csv("data/y_female.csv")

X_male.to_csv("data/X_male.csv")
y_male.to_csv("data/y_male.csv")


