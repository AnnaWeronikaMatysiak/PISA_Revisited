# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:14:00 2022

@author: Anna
"""

"""
TO DO:
- add the validation sets and test models on them
- at the end, change PISA_sample_100 to the final sample we are using'
- add basline of SVM-RFE 
- move SVM, do there hyperparamiter tuning, boosting
- next stage: plots
- next stage: visualisation of error
- next stage:predictions' description an ordering them
"""

#%% import packages
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

#%% call setup file
#import runpy
#runpy.run_path(path_name = '/My Drive/PISA_Revisited/0_setup.py')

#%%
#PISA_sample_100 = pd.read_csv("/My Drive/PISA_Revisited/data/PISA_sample_100.csv")
#utill we drop these variables from the whole data set, I manualy exclude them from sample
#PISA_sample_100 = PISA_sample_100.drop(columns=["VER_DAT", "CNT", "CYC", "STRATUM"])

#X_train=PISA_sample_100.drop(columns=["read_score"])
#y_train=PISA_sample_100["read_score"]

#becuase y is an array, I change it back to data frame
#y_train=y_train.to_frame()


#for current tests untill we have clean data I replaced the data columns
# with NAs witht the colum means. This does not work for "test" column, so I drop it. 
#By the way, what is that column about?
#y_train=y_train.apply(lambda x: x.fillna(x.mean()))
#X_train=X_train.apply(lambda x: x.fillna(x.mean())) 
#X_train=X_train.drop(columns=["test"])

#code to check if any of the columns have NAs:
#y_train.isnull().any()


# repeat that for validation sets

### MID-TERM:
midterm_train = pd.read_csv("/My Drive/PISA_Revisited/data/midterm_train.csv") 
midterm_validation=pd.read_csv("/My Drive/PISA_Revisited/data/midterm_val.csv")

X_train=midterm_train.drop(columns=["read_score"])
y_train=midterm_train["read_score"]

#y_train=y_train.to_frame()


X_validation=midterm_validation.drop(columns=["read_score"])
y_validation=midterm_validation["read_score"]

#y_validation=y_train.to_frame()

#%% linear regression
lin_reg= LinearRegression()
lin_reg.fit(X_train, y_train)   

lin_reg.coef_
lin_reg.intercept_

y_predicted=lin_reg.predict(X_validation)

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

#%% polynomial regressions degree=2

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X_train)

lin_reg_pol = LinearRegression()
lin_reg_pol.fit(X_poly, y_train)
lin_reg_pol.intercept_, lin_reg_pol.coef_

# The coefficients
print ('Coefficients: ', lin_reg_pol.coef_)
print ('Intercept: ',lin_reg_pol.intercept_)

#y_predicted_poly=lin_reg_pol.predict(X_validation) 
validation_X_poly = poly_features.fit_transform(X_validation)
validation_y_poly = lin_reg_pol.predict(validation_X_poly)

mse_poly = mean_squared_error(y_validation,validation_y_poly)
rmse_poly= np.sqrt(mean_squared_error(y_validation, validation_y_poly))
r2_poly=r2_score(y_validation,validation_y_poly)
print(mse_poly)
print(rmse_poly)
print(r2_poly)

#%%
poly_features_3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly_3 = poly_features_3.fit_transform(X_train)

lin_reg_pol_3 = LinearRegression()
lin_reg_pol_3.fit(X_poly_3, y_train)
lin_reg_pol_3.intercept_, lin_reg_pol_3.coef_

# The coefficients
print ('Coefficients: ', lin_reg_pol_3.coef_)
print ('Intercept: ',lin_reg_pol_3.intercept_)

#y_predicted_poly=lin_reg_pol.predict(X_validation) 
validation_X_poly_3 = poly_features_3.fit_transform(X_validation)
validation_y_poly_3 = lin_reg_pol_3.predict(validation_X_poly_3)

mse_poly_3 = mean_squared_error(y_validation,validation_y_poly_3)
rmse_poly_3= np.sqrt(mean_squared_error(y_validation, validation_y_poly_3))
r2_poly_3=r2_score(y_validation,validation_y_poly_3)
print(mse_poly_3)
print(rmse_poly_3)
print(r2_poly_3)

#%%
"""
from sklearn.pipeline import Pipeline
def plot_model(model_class, polynomial, alphas, **model_kargs):
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([
                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),

                    ("regul_reg", model),
                ])
        model.fit(X_train, y_train)
        y_new_regul = model.predict(X_validation)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_validation, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))
    plt.plot(X_train, y_train, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 3, 0, 4])

plt.figure(figsize=(8,4))
plt.subplot(121)
plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(122)
plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)


plt.show()

"""

#%%

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X_train)

poly_ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10], normalize=True)
poly_ridge.fit(X_poly, y_train)
poly_ridge.intercept_, poly_ridge.coef_



# The coefficients
print ('Coefficients: ', poly_ridge.coef_)
print ('Intercept: ',poly_ridge.intercept_)

#y_predicted_poly=lin_reg_pol.predict(X_validation) 
validation_poly_ridge = poly_features_3.fit_transform(X_validation)
validation_y_poly_ridge = lin_reg_pol_3.predict(validation_poly_ridge)

mse_poly_ridge = mean_squared_error(y_validation,validation_poly_ridge)
rmse_poly_ridge= np.sqrt(mean_squared_error(y_validation, validation_poly_ridge))

print(mse_poly_ridge)
print(rmse_poly_ridge)






