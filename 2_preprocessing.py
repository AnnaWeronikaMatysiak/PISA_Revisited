#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:14:57 2022

@author: Jo
"""

#%%

# set working directory
import os
os.getcwd()
os.chdir('/Volumes/GoogleDrive/My Drive/PISA_Revisited/')

#%% import common packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% import tryout samples (later full train data?)

PISA_sample_1000 = pd.read_csv("data/PISA_sample_1000.csv")
PISA_sample_100 = pd.read_csv("data/PISA_sample_100.csv")
PISA_sample_10 = pd.read_csv("data/PISA_sample_10.csv")
# PISA_train = pd.read_csv("data/train_set.csv")


#%% explore sample

# check datatypes
PISA_sample_10.info()

# generate description of variables
PISA_sample_10.describe()

# plot variables (not really possible with our over 1000 variables...)
#PISA_sample_100.hist(bins=50, figsize=(20,15))
#plt.show()

# --> select random features
PISA_plot_sample = PISA_sample_100.sample(n=10,axis='columns')
PISA_plot_sample.head()

# plot only those
PISA_plot_sample.hist(bins=50, figsize=(20,15))
plt.show()

# exploring the dependent variable "reading skills"
PISA_sample_1000.read_score.mean()
PISA_sample_1000[["read_score", "female"]].groupby("female").mean()

PISA_sample_1000.hist(column='read_score',bins=50)
plt.axvline(x=456.1, color='red', linestyle='--')
plt.show()

#%% Do something with NA's...


#%% normalize all features and build a pipeline for preprocessing 
# the numerical attributes (still check out what it does exactly)

#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler

#num_pipeline = Pipeline([
        #('imputer', SimpleImputer(strategy="median")), # what's this?
        #('attribs_adder', CombinedAttributesAdder()), #  what's this?
        #('std_scaler', StandardScaler()),
    #])

#data_transformed = num_pipeline.fit_transform("numerical_data")




