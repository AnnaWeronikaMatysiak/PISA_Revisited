#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:58:43 2022

@author: Johanna & Anna
"""

#%% necessary packages
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

#%% ensemble made of the two best most different models

# load final ridge model and extra trees fine tuned
import joblib
ridge_tuned = joblib.load("models/final_ridge_model.pkl")
extra = joblib.load("models/ExtraTrees.pkl")


from sklearn.ensemble import VotingRegressor

named_estimators = [
    #("<model_description>", <model_variable>), # for each of the individual models
    ("ridge_tuned", ridge_tuned), 
    ("extra", extra)
]

voting_reg = VotingRegressor(named_estimators)

# fitting a VotingClassifier works as fitting any other model
voting_reg.fit(X_train, y_train)

# prediction for X_val_1
y_pred = voting_reg.predict(X_val_2)

# evaluate
voting_mse = mean_squared_error(y_val_2, y_pred)
print(voting_mse)
voting_rmse = np.sqrt(voting_mse)

voting_rmse # result: 66.40221618906632

# save the model
joblib.dump(voting_reg, "models/Ensemble.pkl")



