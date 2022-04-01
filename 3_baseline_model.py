# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:14:00 2022

@author: Anna
"""
#%% import packages
from sklearn.linear_model import LinearRegression
import pandas as pd

#%%
PISA_sample_10=pd.read_csv("/My Drive/PISA_Revisited/data/PISA_sample_10.csv/")

X=PISA_sample_10.loc[:, PISA_sample_10.columns.drop(['read_score'])]
y = PISA_sample_10[["read_score"]]

lin_reg= LinearRegression()
lin_reg.fit(X, y)


