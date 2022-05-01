#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:11:16 2022

@author: Jo
"""

""" 
- take our best model
- train it on the girls and boys subsets (?)
- look at the strongest predictors
"""


#%% evaluation of the best model (ridge tuned) on the test set

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

# check dimensions -> dimensions should be 205 columns in X and 1 in y
def drop_first_entry(df):
    df.drop(df.columns[[0]], axis = 1, inplace = True)

drop_first_entry(X_test)
drop_first_entry(y_test)

# evaluate the best model on the test set... see the final score!
# load the best model
import joblib
final_model = joblib.load("models/final_ridge_model.pkl")

# prediction for X_val_1
y_pred = final_model.predict(X_test)

# evaluate
lin_reg_mse = mean_squared_error(y_test, y_pred)
lin_reg_rmse = np.sqrt(lin_reg_mse)
print(lin_reg_rmse)
 # result: 



#%% fit model on boys and girls subset and check feature importance to find the best predictors

# read in boys and girls subsets
X_female = pd.read_csv("data/X_female.csv")
y_female = pd.read_csv("data/y_female.csv")

X_male = pd.read_csv("data/X_male.csv")
y_male = pd.read_csv("data/y_male.csv")

# if needed, drop first entries
def drop_first_entry(df):
    df.drop(df.columns[[0]], axis = 1, inplace = True)

drop_first_entry(X_female)
drop_first_entry(y_female)
drop_first_entry(X_male)
drop_first_entry(y_male)


import joblib
final_model = joblib.load("models/final_ridge_model.pkl")

####################################

# fit the model to girls
final_model.fit(X_female, y_female)

# feature importance
feature_importances = final_model.feature_importances_
feature_importances

# get feature names and combine them with the coefficients, save as csv
feature_names = X_female.columns
predictors_girls = sorted(zip(feature_importances, feature_names), reverse=True)
predictors_girls = pd.DataFrame(predictors_girls, columns = ['coefficient', 'predictor'])
predictors_girls.to_csv("data/predictors_girls.csv")


#####################################

# fit the model to boys
ExtraTrees_loaded.fit(X_female, y_female)
ExtraTrees_loaded.fit(X_male, y_male)

# feature importance
feature_importances = ExtraTrees_loaded.feature_importances_
feature_importances

# get feature names and combine them with the coefficients, save as csv
feature_names = X_male.columns
predictors_boys = sorted(zip(feature_importances, feature_names), reverse=True)
predictors_boys = pd.DataFrame (predictors_boys, columns = ['coefficient', 'predictor'])
predictors_boys.to_csv("data/predictors_boys.csv")



