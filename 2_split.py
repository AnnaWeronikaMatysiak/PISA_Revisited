#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:28:18 2022

@author: Jo
"""

"""
TO DO:
- run again when preprocessed completely
- change the dataset to the "preprocessed 1000"
- split according to the lab
- add subsets by gender
"""
#%% call setup file

import runpy
runpy.run_path(path_name = '/0_setup.py')

# imports sys, sklearn, numpy, os, matplotlib, pathlib
# checks versions, sets wd, sets random.seed 42, specifies plots
# defines function save_fig()

#%% necessary packages

import pandas as pd
#import numpy as np


#%% read in data

# read csv file (current version, excahnge later)
# taken from file 1, but still missing imputing and normalization -> see file 1.1 and 1.2
PISA_raw_1000 = pd.read_csv("/My Drive/PISA_Revisited/data/PISA_raw_1000.csv")

#%% new split
from sklearn.model_selection import train_test_split

X=PISA_raw_1000.drop(columns=["read_score"])
y=PISA_raw_1000["read_score"]
y=y.to_frame()

# X = X.transpose()

X_train, X_test_val_1_val_2, y_train, y_test_val_1_val_2 = train_test_split(X, y, test_size=3/10, random_state=42)
X_test_val_1,X_val_2, y_test_val_1, y_val_2= train_test_split(X_test_val_1_val_2,y_test_val_1_val_2, test_size=1/3, random_state=42)
X_test, y_test, X_val_1, y_val_1 = train_test_split(X_test_val_1, y_test_val_1, test_size=1/2, random_state=42)


X_test.to_csv("data/X_test.csv")
y_test.to_csv("data/y_test.csv")
X_train.to_csv("data/X_train.csv")
y_train.to_scv("data/y_train.csv")

X_val_1.to_csv("data/X_val_1.csv")
y_val_1.to_csv("data/y_val_1.csv")
X_val_2.to_csv("data/X_val_2.csv")
y_val_2.to_scv("data/y_val_2.csv")


#%% split using scikit learn 
"""
# to make this notebook's output identical at every run
np.random.seed(42)

from sklearn.model_selection import train_test_split

# choose 30% of the data as test set (including two validation sets to be splitted later), 
# rest is training set (70%)
train_set, test_set = train_test_split(PISA_reduced, test_size=0.3, random_state=42)

len(train_set)
len(test_set)
"""
#%% create validation sets taken from the test set

"""
# validation set 1 (about a third of test_set)
test_set, val_1_set = train_test_split(test_set, test_size=1/3, random_state=42) # "test-size" is now val_1 size

# validation set 2 (half of test_set with val_1 already split up)
test_set, val_2_set = train_test_split(test_set, test_size=0.5, random_state=42) # "test-size" is now val_2 size

# should be approximately the same length, each around 10% of the data
len(train_set)
len(test_set)
len(val_1_set)
len(val_2_set)

"""


#%% save train, validation and test sets
"""
train_set.to_csv("data/train_set.csv")
test_set.to_csv("data/test_set.csv")
val_1_set.to_csv("data/val_1_set.csv")
val_2_set.to_csv("data/val_2_set.csv")
"""

#%% create smaller samples for observation and model tryouts 

# create a random sample for code developing
PISA_sample_1000 = train_set.sample(1000)

# create a smaller random sample for code developing
PISA_sample_100 = train_set.sample(100)

# create even smaller sample
PISA_sample_10 = train_set.sample(10)

# check if sampling worked out
len(PISA_sample_1000)
len(PISA_sample_100)
len(PISA_sample_10)

# saving samples as csv
PISA_sample_1000.to_csv("data/PISA_sample_1000.csv")
PISA_sample_100.to_csv("data/PISA_sample_100.csv")
PISA_sample_10.to_csv("data/PISA_sample_10.csv")






