#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 23:30:45 2022

@author: Jo
"""
#%% import packages
import numpy as np
import os
import pandas as pd

#%% set working directory
os.getcwd()
os.chdir('/Volumes/GoogleDrive/My Drive/PISA_Revisited/')

# to make the output stable across runs
np.random.seed(42)

#%% descriptions see preprocessing file

def rename_read_score_female(dataframe):
    dataframe = dataframe.rename(columns = {'PV1READ':'read_score', 'ST004D01T':'female'}, inplace = True)
    
def drop_students_without_read_score(dataframe):
    dataframe.dropna(subset=['PV1READ'], inplace = True)
    
def remove_string_columns(dataframe):
    dataframe = dataframe.drop(columns = ["VER_DAT", "CNT", "CYC", "STRATUM"], inplace = True)
    
# function to drop columns over a certain percentage of missingness
def drop_columns_with_missingness(dataframe, percentage):
    NaN_count_rel = dataframe.isnull().sum()/len(dataframe)*100
    NaN_count_rel = NaN_count_rel.reset_index(level=0)
    NaN_count_rel = pd.DataFrame(NaN_count_rel)
    NaN_count_rel.columns = ['variable', 'missingness']
    observations_to_drop = NaN_count_rel[NaN_count_rel["missingness"] > percentage]
    observation_names_to_drop = observations_to_drop["variable"]
    observation_names_to_drop = observation_names_to_drop.values.tolist()
    dataframe = dataframe.drop(columns = observation_names_to_drop, axis = 1)
    # print(len(observation_names_to_drop) should show how many were dropped
    return dataframe