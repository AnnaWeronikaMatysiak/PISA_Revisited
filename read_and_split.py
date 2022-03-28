#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:28:18 2022

@author: Jo
"""

# load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read csv file
PISA_raw = pd.read_csv("/Volumes/GoogleDrive/My Drive/PISA_Revisited/data/PISA_student_data.csv")

# take a random sample for code developing
PISA_sample_100 = PISA_raw.sample(100)
PISA_sample_100.head(10)

# check datatypes
PISA_sample_100.info()

# description of variables
PISA_sample_100.describe()

# plot variables
PISA_sample_100.hist(bins=50, figsize=(20,15))
plt.show()

# create test and validation set...
