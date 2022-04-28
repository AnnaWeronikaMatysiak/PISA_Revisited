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

# read in boys and girls subsets
X_female = pd.read_csv("data/X_female.csv")
y_female = pd.read_csv("data/y_female.csv")

X_male = pd.read_csv("data/X_male.csv")
y_male = pd.read_csv("data/y_male.csv")

PISA_female = pd.read_csv("data/PISA_female.csv")
PISA_male = pd.read_csv("data/PISA_male.csv")

# if needed, drop first entries
# drop_first_entry(data)

# load the best model
import joblib
ExtraTrees_loaded=joblib.load("models/ExtraTrees.pkl")

#%% evaluation on the test set




# evaluate the best model on the test set... see the final score!







#%% fit model on boys and girls subset and check feature importance to find the best predictors

# fit the model to girls and boys
ExtraTrees_loaded.fit(X_female, y_female)
ExtraTrees_loaded.fit(X_male, y_male)

# feature importance
feature_importances = ExtraTrees_loaded.feature_importances_
feature_importances

# get feature names and combine them with the coefficients, save as csv
feature_names = X_female.columns
predictors_girls = sorted(zip(feature_importances, feature_names), reverse=True)
predictors_girls = pd.DataFrame (predictors_girls, columns = ['coefficient', 'predictor'])
predictors_girls.to_csv("data/predictors_girls.csv")

feature_names = X_male.columns
predictors_boys = sorted(zip(feature_importances, feature_names), reverse=True)
predictors_boys = pd.DataFrame (predictors_boys, columns = ['coefficient', 'predictor'])
predictors_boys.to_csv("data/predictors_boys.csv")



