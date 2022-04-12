# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:14:00 2022

@author: Anna
"""

"""
TO DO:
- add the validation sets and test models on them
- add plots
- add models with changes parameters like alpha, degree of polymomials etc. 
- at the end, change PISA_sample_100 to the final sample we are using'
"""

#%% import packages
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
#from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

#%% call setup file
#import runpy
#runpy.run_path(path_name = '/My Drive/PISA_Revisited/0_setup.py')

#%%
PISA_sample_100 = pd.read_csv("/My Drive/PISA_Revisited/data/PISA_sample_100.csv")
#utill we drop these variables from the whole data set, I manualy exclude them from sample
PISA_sample_100 = PISA_sample_100.drop(columns=["VER_DAT", "CNT", "CYC", "STRATUM"])

X_train=PISA_sample_100.drop(columns=["read_score"])
y_train=PISA_sample_100["read_score"]

#becuase y is an array, I change it back to data frame
y_train=y_train.to_frame()


#for current tests untill we have clean data I replaced the data columns
# with NAs witht the colum means. This does not work for "test" column, so I drop it. 
#By the way, what is that column about?
y_train=y_train.apply(lambda x: x.fillna(x.mean()))
X_train=X_train.apply(lambda x: x.fillna(x.mean())) 
X_train=X_train.drop(columns=["test"])

#code to check if any of the columns have NAs:
#y_train.isnull().any()

# repeat that for validation sets
y_validation=...
X_validation=...

#%% linear regression
lin_reg= LinearRegression()
lin_reg.fit(X_train, y_train)   

lin_reg.coef_
lin_reg.intercept_

y_predicted=lin_reg.predict(X_validation)
y_predicted.head()

#%% evaluation 
mse = mean_squared_error(y_validation, y_predicted)
rmse= np.sqrt(mean_squared_error(y_validation, y_predicted))
print(mse)
print(rmse)

#%%plots
#plt.plot(X_train, y_predicted, color="blue", linewidth=3)

#%% ridge regression
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X_std = scaler.fit_transform(X_train)

ridge_reg = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10], normalize=True)
ridge_reg_model=ridge_reg.fit(X_train, y_train)
predicted_ridge=ridge_reg.predict(X_validation)

#to check which alpha was used
ridge_reg_model.alpha_

#%% evaluation 

mse_ridge = mean_squared_error(y_validation, predicted_ridge)
rmse_ridge= np.sqrt(mean_squared_error(y_validation, predicted_ridge))

print(mse_ridge)
print(rmse_ridge)

#%% plots

#%% polynomial regressions 

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X_train)

lin_reg_pol = LinearRegression()
lin_reg_pol.fit(X_poly, y_train)
lin_reg_pol.intercept_, lin_reg_pol.coef_
y_predicted_poly=lin_reg_pol.predict(X_validation) 
y_predicted_poly.head()


poly_features_3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features_3.fit_transform(X_train)

lin_reg_pol_3 = LinearRegression()
lin_reg_pol_3.fit(X_poly, y_train)
lin_reg_pol_3.intercept_, lin_reg_pol_3.coef_
y_predicted_poly_3=lin_reg_pol.predict(X_validation) 
y_predicted_poly_3.head()

#%% evaluation 
mse_poly = mean_squared_error(y_validation, y_predicted_poly)
rmse_poly= np.sqrt(mean_squared_error(y_validation, y_predicted_poly))

r2_poly=r2_score(y_validation, y_predicted_poly)
r2_poly_3= r2_score(y_validation, y_predicted_poly_3)

print(r2_poly)
print(r2_poly_3)
print(mse_poly)
print(rmse_poly)

#%% Linear SVM base line 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


polynomial_svm_clf = Pipeline([("poly_features", PolynomialFeatures(degree=3)),
                               ("scaler", StandardScaler()),("svm_clf", 
                                LinearSVC(C=10, loss="hinge")) ])
polynomial_svm_clf.fit(X_train, y_train)



