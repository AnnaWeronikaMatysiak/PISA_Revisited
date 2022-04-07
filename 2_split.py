#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:28:18 2022

@author: Jo
"""

#%% call setup file

import runpy
runpy.run_path(path_name = '/Volumes/GoogleDrive/My Drive/PISA_Revisited/0_setup.py')

# imports sys, sklearn, numpy, os, matplotlib, pathlib
# checks versions, sets wd, sets random.seed 42, specifies plots
# defines function save_fig()


#%% import additional packages

# load packages (common imports)
import pandas as pd

# read csv file
PISA_raw = pd.read_csv("/Volumes/GoogleDrive/My Drive/PISA_Revisited/data/PISA_student_data.csv")


#%% split using scikit learn 

# to make this notebook's output identical at every run
np.random.seed(42)

from sklearn.model_selection import train_test_split

# choose 20% of the data as test set, rest is training set (80%)
train_set, test_set = train_test_split(PISA_raw, test_size=0.2, random_state=42)

len(train_set)
len(test_set)

# save train and test set
train_set.to_csv("data/train_set.csv")
test_set.to_csv("data/test_set.csv")


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





