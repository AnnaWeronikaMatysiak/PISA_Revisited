# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:14:00 2022

@author: Anna
"""

#%% import packages
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#%% call setup file
#import runpy
#runpy.run_path(path_name = '/My Drive/PISA_Revisited/0_setup.py')
PISA_sample_100 = pd.read_csv("/My Drive/PISA_Revisited/data/PISA_sample_100.csv")
PISA_sample_100 = PISA_sample_100.columns.drop(["VER_DAT", "CNT", "CYC", "STRATUM"])

# imports sys, sklearn, numpy, os, matplotlib, pathlib
# checks versions, sets wd, sets random.seed 42, specifies plots
# defines function save_fig()



#%% preceding steps to enable us run these models (work in progress)

#we should make secision on NAs
#PISA_sample_100=PISA_sample_100.fillna(value=PISA_sample_100.mean())

#we need to use either LabelEncoder or OneHot thingy
#OneHotEncoder().fit_transform(PISA_sample_10)


#from sklearn.preprocessing import LabelEncoder
#def Encoder(PISA_sample_100):
#          columnsToEncode = list(PISA_sample_100.select_dtypes(include=['category','object']))
#          le = LabelEncoder()
#          for feature in columnsToEncode:
#              try:
#                  PISA_sample_100[feature] = le.fit_transform(PISA_sample_100[feature])
#              except:
#                  print('Error encoding '+feature)
#          return PISA_sample_100

#Encoder(PISA_sample_100)
      

#%%
X_tain=PISA_sample_100[:,:'VER_DAT', 'CNT', 'CYC', 'STRATUM']
#y = PISA_sample_100[['read_score']]
y_train=PISA_sample_100[:,'read_score']

#%% linear regression
lin_reg= LinearRegression()
lin_reg.fit(X_train, y_train)   

lin_reg.coef_
lin_reg.intercept_
predicted=lin_reg.predict(X_train)
predicted

#%%plots
plt.scatter(X_train, y_train, color="black")
plt.plot(X_train, predicted, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

#%% RMSE

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

mse = mean_squared_error(y_train, predicted, squared=False)
rmse= np.sqrt(metrics.mean_squared_error(y_train, predicted)

#%% polynomial regression

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]

lin_reg_pol = LinearRegression()
lin_reg_pol.fit(X_poly, y_train)
lin_reg_pol.intercept_, lin_reg.coef_
lin_reg_pol.predict(X_poly)

#%% ridge regression
rr = Ridge(alpha=0.01)
rr.fit(X_train, y_train) 
pred_train_rr= rr.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_rr)))
print(r2_score(y_train, pred_train_rr))


