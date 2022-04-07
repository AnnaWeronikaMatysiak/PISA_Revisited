#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:14:57 2022

@author: Jo
"""

#%% call setup file

import runpy
runpy.run_path(path_name = '/Volumes/GoogleDrive/My Drive/PISA_Revisited/0_setup.py')

# imports sys, sklearn, numpy, os, matplotlib, pathlib
# checks versions, sets wd, sets random.seed 42, specifies plots
# defines function save_fig()

#%% import additional packages

import pandas as pd

#%% read in data

# read csv file
PISA_raw = pd.read_csv("/Volumes/GoogleDrive/My Drive/PISA_Revisited/data/PISA_student_data.csv")

# renaming relevant columns
PISA_raw.rename(columns = {'PV1READ':'read_score', 'ST004D01T':'female'}, inplace = True)


#%% create tryout samples (created as split of raw data)

# create a random sample for code developing
PISA_raw_1000 = PISA_raw.sample(1000)

# create a smaller random sample for code developing
PISA_raw_100 = PISA_raw.sample(100)

# create even smaller sample
PISA_raw_10 = PISA_raw.sample(10)

# check if sampling worked out
len(PISA_raw_1000)
len(PISA_raw_100)
len(PISA_raw_10)

# saving samples as csv
PISA_raw_1000.to_csv("data/PISA_raw_1000.csv")
PISA_raw_100.to_csv("data/PISA_raw_100.csv")
PISA_raw_10.to_csv("data/PISA_raw_10.csv")

#%% explore sample

# check datatypes
PISA_raw_10.info()

# generate description of variables
PISA_raw_10.describe()

# plot variables (not really possible with our over 1000 variables...)
# PISA_sample_100.hist(bins=50, figsize=(20,15))
# plt.show()

# --> plot variables: select random features
PISA_plot_sample = PISA_raw_100.sample(n=10,axis='columns')
PISA_plot_sample.head()

# plot only those
PISA_plot_sample.hist(bins=50, figsize=(20,15))
save_fig("distribution_examples")
plt.show()

# exploring the dependent variable "reading skills"
PISA_raw_1000.read_score.mean()
PISA_raw_1000[["read_score", "female"]].groupby("female").mean()

PISA_raw_1000.hist(column='read_score',bins=50)
plt.axvline(x=456.1, color='red', linestyle='--')
save_fig("read_score")
plt.show()


#%% Data Types



#%% Do something with NA's... -> or is this the 

# find how many NA's in each row, develop a strategy to deal with them



#%% any other preprocessing (pattern missingness? etc.)



#%% normalize all features and build a pipeline for preprocessing 

# the numerical attributes (still check out what it does exactly)

# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler

# num_pipeline = Pipeline([
        #('imputer', SimpleImputer(strategy="median")), # what's this?
        #('attribs_adder', CombinedAttributesAdder()), #  what's this?
        #('std_scaler', StandardScaler()),# this is our normalizer
    #])

# data_transformed = num_pipeline.fit_transform("numerical_data")



