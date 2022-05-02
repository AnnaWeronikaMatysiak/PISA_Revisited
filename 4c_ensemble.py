#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:58:43 2022

@author: Johanna & Anna
"""


#%% ensemble made of the two best most different models

# load final ridge model and extra trees fine tuned
import joblib
ridge_tuned = joblib.load("models/final_ridge_model.pkl")
extra_tuned = joblib.load("models/ExtraTrees_tuned.pkl")


from sklearn.ensemble import VotingRegressor

named_estimators = [
    #("<model_description>", <model_variable>), # for each of the individual models
    ("ridge_tuned", ridge_tuned), 
    ("extra_tuned", extra_tuned)
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

voting_rmse # result: 66.5252708179837

# save the model
joblib.dump(voting_reg, "models/Ensemble.pkl")













# get inspiration from above where we evaluated the individual models
voting_reg.score(X_val_2, y_val_2)

# score: 

# Changing the voting method
#voting_reg.voting = "soft" # default is hard
#voting_reg.score(X_val_2, y_val_2)

# score: 