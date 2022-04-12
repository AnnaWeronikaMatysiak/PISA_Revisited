#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 20:29:28 2022

@author: Jo
"""

#%% read raw data and create midterm sample

import pandas as pd

# The first to steps don't need to be run anymore, this is only to 
# show how midterm_raw was created. You can directly call the midterm_raw file below.

# read full csv file 
# PISA_raw = pd.read_csv("/Volumes/GoogleDrive/My Drive/PISA_Revisited/data/PISA_student_data.csv")

# take a sample of 5000 observations (tradeoff regarding runtime for midterm report)
# midterm_raw = PISA_raw.sample(6000)

# save midterm_raw as csv if someone wants to replicate this file
# midterm_raw.to_csv("data/midterm_raw.csv")

# read in midterm_raw (available in data folder) if you want to replicate 
# this file:
midterm_raw = pd.read_csv("data/midterm_raw.csv")

#%% first preprocessing

# renaming: function from file 1
rename_read_score_female(midterm_raw)

# drop students without reading score (these are 52 observations -> 5948)
midterm_raw.dropna(subset = ['read_score'], inplace = True)

# removing string columns: function from file 1
remove_string_columns(midterm_raw)

# removing rows with a missingness over x percent: function from file 1
midterm_reduced = drop_columns_with_missingness(midterm_raw, 10)


#%% imputing

# type "pip install missingpy" in the console for installation
from missingpy import MissForest

X = midterm_reduced

imputer = MissForest(max_iter = 5, n_estimators = 50, max_features = 100, n_jobs = -1, random_state = 42)

# cat_vars : int or array of ints containing column indices of categorical variable(s)
# For midterm, these include only the most obvious categorical, but not all ordinal variables
# Treatment of ordinal variables to be discussed...
cat_vars = np.array(range(15))

# fit imputer
imputer.fit(X, cat_vars = cat_vars)

# apply imputer
midterm_imputed = imputer.transform(X)

# convert to pandas dataframe
midterm_imputed = pd.DataFrame(midterm_imputed)

# save result as csv file (just as a backup)
midterm_imputed.to_csv("data/midterm_imputed.csv")


#%% scaling (runtime: 1h40m)

# Since in our research question, we don't care about the actual reading score
# but moreover we care about the predicor variables of it, we can also scale 
# the dependent variable without any harm to our project target.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fit scaler
scaler.fit(midterm_imputed)

# scale data
midterm_scaled = scaler.transform(midterm_imputed)

# convert to pandas dataframe
midterm_scaled = pd.DataFrame(midterm_scaled)

# save result as csv file (just as a backup)
midterm_scaled.to_csv("data/midterm_scaled.csv")


#%% splitting

# to make this notebook's output identical at every run
np.random.seed(42)

from sklearn.model_selection import train_test_split

# choose 1/6 of the data as validation set 
# rest is training set for now (5/6 of midterm_raw)
midterm_train, midterm_val = train_test_split(midterm_scaled, test_size=1/6, random_state=42)

len(midterm_train)
len(midterm_val)

# save as csv
midterm_train.to_csv("data/midterm_train.csv")
midterm_val.to_csv("data/midterm_val.csv")











