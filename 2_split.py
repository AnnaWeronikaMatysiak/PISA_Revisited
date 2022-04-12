#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:28:18 2022

@author: Jo
"""

"""

To Do's
- run again when preprocessed completely (imputing and scaling of the whole dataset 
                                        is still missing)


Endproduct should be dataframes train_set, test_set, val_1_set and val_2_set 
saved to csv.

The file takes the final preprocessed file (to be created) as a starting point.
Right now it starts with the most preprocessed version "PISA_reduced".

"""


#%% call setup file

import runpy
runpy.run_path(path_name = '/My Drive/PISA_Revisited/0_setup.py')

# imports sys, sklearn, numpy, os, matplotlib, pathlib
# checks versions, sets wd, sets random.seed 42, specifies plots
# defines function save_fig()


#%% import additional packages

# load packages (common imports)
import pandas as pd
import numpy as np

# read csv file (current version, excahnge later)
# taken from file 1, but still missing imputing and normalization -> see file 1.1 and 1.2
PISA_reduced = pd.read_csv("/Volumes/GoogleDrive/My Drive/PISA_Revisited/data/PISA_reduced.csv")


#%% split using scikit learn 

# to make this notebook's output identical at every run
np.random.seed(42)

from sklearn.model_selection import train_test_split

# choose 30% of the data as test set (including two validation sets to be splitted later), 
# rest is training set (70%)
train_set, test_set = train_test_split(PISA_reduced, test_size=0.3, random_state=42)

len(train_set)
len(test_set)

#%% create validation sets taken from the test set

# validation set 1 (about a third of test_set)
test_set, val_1_set = train_test_split(test_set, test_size=1/3, random_state=42) # "test-size" is now val_1 size

# validation set 2 (half of test_set with val_1 already split up)
test_set, val_2_set = train_test_split(test_set, test_size=0.5, random_state=42) # "test-size" is now val_2 size

# should be approximately the same length, each around 10% of the data
len(train_set)
len(test_set)
len(val_1_set)
len(val_2_set)


#%% save train, validation and test sets

train_set.to_csv("data/train_set.csv")
test_set.to_csv("data/test_set.csv")
val_1_set.to_csv("data/val_1_set.csv")
val_2_set.to_csv("data/val_2_set.csv")


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


#%% create subsets by gender (of which data? probably whole dataset)

# STILL TO DO
#
#





