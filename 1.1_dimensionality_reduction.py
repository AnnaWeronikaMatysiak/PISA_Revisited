#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:09:50 2022

@author: Jo
"""

#%% call setup file

import runpy
runpy.run_path(path_name = '/Volumes/GoogleDrive/My Drive/PISA_Revisited/0_setup.py')

# imports sys, sklearn, numpy, os, matplotlib, pathlib
# checks versions, sets wd, sets random.seed 42, specifies plots
# defines function save_fig()

#%% load additional packages

from sklearn.preprocessing import scale # Data scaling
from sklearn import decomposition #PCA
import pandas as pd

#%% read in data (later use only full raw data)

#PISA_sample_10 = pd.read_csv("data/PISA_sample_10.csv")
PISA_sample_100 = pd.read_csv("data/PISA_sample_100.csv")
#PISA_sample_1000 = pd.read_csv("data/PISA_sample_1000.csv")

# input features
print(PISA_sample_100.columns.values)

# output feature
PISA_sample_100.read_score.mean()
PISA_sample_100.read_score.describe()

#%% preprocess (we somehow need to replace all strings with other values... Ania is on it!)

pd.DataFrame(PISA_sample_100)

# this was to try to drop the VER_DAT and CNT columns, because we don't need them and they have another data type than float/integer/boolian
# but somehow didn't seem to work :()
PISA_sample_100a = PISA_sample_100.drop(columns = ["VER_DAT", "CNT"], inplace=True)


#%% assigning input and output variables, examine data dimensions

X = PISA_sample_100.columns.drop(["read_score"])
Y = PISA_sample_100a[["read_score"]]

X.shape
Y.shape

# scale X -> does not work for string variables!!! ANIA IS ON IT
X = scale(X)

#from sklearn.preprocessing import StandardScaler
#X = StandardScaler(X)

pca = decomposition.PCA(n_components=500)
pca.fit(X)

#%% Principal Component Analysis (PCA)

# score values

# loading values

# scree plot

# see github: https://github.com/dataprofessor/code/blob/master/python/PCA_analysis.ipynb
# and youtube: https://www.youtube.com/watch?v=oiusrJ0btwA












