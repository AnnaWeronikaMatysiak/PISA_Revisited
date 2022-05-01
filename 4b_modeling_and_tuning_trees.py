# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:42:21 2022

@author: Jo&Anna

"""
#%% necessary packages
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

#%% call setup file
import runpy
runpy.run_path(path_name = '/0_setup.py')

# imports sys, sklearn, numpy, os, matplotlib, pathlib
# checks versions, sets wd, sets random.seed 42, specifies plots
# defines function save_fig()

#%% read in data
#X_test = pd.read_csv("/My Drive/PISA_Revisited/data/X_test.csv") 
#y_test = pd.read_csv("/My Drive/PISA_Revisited/data/y_test.csv")

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

X_val_1 = pd.read_csv("data/X_val_1.csv") 
y_val_1 = pd.read_csv("data/y_val_1.csv")

X_val_2 = pd.read_csv("data/X_val_2.csv") 
y_val_2 = pd.read_csv("data/y_val_2.csv")

# if needed, drop first column that was automatically generated 
# -> dimensions should be 205 columns in X and 1 in y
def drop_first_entry(df):
    df.drop(df.columns[[0]], axis = 1, inplace = True)

drop_first_entry(X_train)
drop_first_entry(y_train)
drop_first_entry(X_val_1)
drop_first_entry(y_val_1)
drop_first_entry(X_val_2)
drop_first_entry(y_val_2)

#%% Random Forest Regressor
forest_reg =  RandomForestRegressor()

# fit the model to our training data
forest_reg.fit(X_train, y_train)

# prediction for X_val_1
y_pred = forest_reg.predict(X_val_1)

# evaluate
forest_mse = mean_squared_error(y_val_1, y_pred)
print(forest_mse)
forest_rmse = np.sqrt(forest_mse)

forest_rmse # result: 71.15293129026736

# alternative: compute cross validation scores
#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(forest_reg, X_train, y_train, scoring = "neg_root_mean_squared_error", cv = 5) 

# saves the model 
joblib.dump(forest_reg, "models/RandomForests.pkl")

# load the model if needed
forest_reg = joblib.load("models/RandomForests.pkl")

#%% ExtraTrees (Extremely Randomized Trees)

# trades more bias for a lower variance, much faster to train than RandomForests
extra_reg =  ExtraTreesRegressor()

# fit the model to our training data
extra_reg.fit(X_train, y_train)

# prediction for X_val_1
y_pred = extra_reg.predict(X_val_1)

# evaluate
extra_mse = mean_squared_error(y_val_1, y_pred)
print(extra_mse)
extra_rmse = np.sqrt(extra_mse)

extra_rmse # result: 70.29067166539122

# saves the model 

joblib.dump(extra_reg, "models/ExtraTrees.pkl")

# load the model if needed
extra_reg = joblib.load("models/ExtraTrees.pkl")


#%% model 2 RandomForest

# n_estimators: number of trees in the forest. default = 100
# max_features: number of features to consider when looking for the best split. default = 

"""param_grid_1 = [
    {"n_estimators": [50, 100, 150], "max_features": [50, 100, 205]},
    ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid_1, cv=5,
                         scoring="neg_mean_squared_error",
                         return_train_score=True)

# perform grid search with the validdation set 2
grid_search.fit(X_val_2, y_val_2)

# get best parameters
grid_search.best_params_
# result: {'max_features': 50, 'n_estimators': 150} -> score = 72.73 which is worse than with default parameters

# look at evaluation scores of all parameter combinations
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)"""
    
#############################

"""# second try  with less features and more estimators (= number of trees) because both values were on the edge of the grid search
param_grid_2 = [
    {"n_estimators": [200, 300], "max_features": [10, 20]},
    ]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid_2, cv=5,
                         scoring="neg_mean_squared_error",
                         return_train_score=True)

# perform grid search with the validdation set 2
grid_search.fit(X_val_2, y_val_2)

# get best parameters
grid_search.best_params_
# result: {'max_features': 20, 'n_estimators': 300} -> score = 73.0656116 which is worse than with default parameters

# look at evaluation scores of all parameter combinations
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)"""

##############################

# third try  with more features and more estimators (= number of trees) because both values were on the edge of the grid search
param_grid_3 = [
    {"n_estimators": [300], "max_features": [20, 30, 40, 50]},
    ]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid_3, cv=5,
                         scoring="neg_mean_squared_error",
                         return_train_score=True)

# perform grid search with the validdation set 2
grid_search.fit(X_val_2, y_val_2)

# get best parameters
grid_search.best_params_
# result: {'max_features': 40, 'n_estimators': 300} -> score = 72.61856548195976 which is worse than with default parameters

# look at evaluation scores of all parameter combinations
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
# saves the model 
joblib.dump(forest_reg, "models/RandomForests_tuned.pkl")

# load the model if needed
# RandomForest_loaded=joblib.load("models/RandomForests_tuned.pkl")


#%% model 3 ExtraTrees

"""param_grid = [
    {"n_estimators": [50, 100, 150], "max_features": [30, 40, 50]},
    ]

from sklearn.ensemble import ExtraTreesRegressor

extra_reg =  ExtraTreesRegressor()

grid_search = GridSearchCV(extra_reg, param_grid, cv=5,
                         scoring="neg_mean_squared_error",
                         return_train_score=True)

# perform grid search with the validdation set 2 
grid_search.fit(X_val_2, y_val_2)

# get best parameters
grid_search.best_params_
# result: {'max_features': 50, 'n_estimators': 150} -> score = 72.33573422309566 which is worse than with default parameters

# look at evaluation scores of all parameter combinations
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)"""
    
    
####################

"""# second try with more max_features

param_grid = [
    {"n_estimators": [300], "max_features": [50, 75, 100]},
    ]

from sklearn.ensemble import ExtraTreesRegressor

extra_reg =  ExtraTreesRegressor()

grid_search = GridSearchCV(extra_reg, param_grid, cv=5,
                         scoring="neg_mean_squared_error",
                         return_train_score=True)

# perform grid search with the validdation set 2 
grid_search.fit(X_val_2, y_val_2)

# get best parameters
grid_search.best_params_
# result: {'max_features': 100, 'n_estimators': 300} -> score = 72.06615214928573 which is worse than with default parameters

# look at evaluation scores of all parameter combinations
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# saves the model 
import joblib
joblib.dump(forest_reg, "models/ExtraTrees_tuned.pkl")

# load the model if needed
# RandomForest_loaded=joblib.load("models/ExtraTree_tuned.pkl")""""""


######################

"""# third try with more max_features

param_grid = [
    {"n_estimators": [300], "max_features": [110, 125, 140]},
    ]

extra_reg =  ExtraTreesRegressor()

grid_search = GridSearchCV(extra_reg, param_grid, cv=5,
                         scoring="neg_mean_squared_error",
                         return_train_score=True)

# perform grid search with the validdation set 2 
grid_search.fit(X_val_2, y_val_2)

# get best parameters
grid_search.best_params_
# result: {'max_features': 110, 'n_estimators': 300} -> score = 72.17961678688113 which is worse than with default parameters

# look at evaluation scores of all parameter combinations
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# saves the model 
joblib.dump(forest_reg, "models/ExtraTrees_tuned_2.pkl")

# load the model if needed
# RandomForest_loaded=joblib.load("models/ExtraTree_tuned.pkl")"""


################################


# fourth try with more trees

param_grid = [
    {"n_estimators": [1000], "max_features": [100]},
    ]

extra_reg =  ExtraTreesRegressor()

grid_search = GridSearchCV(extra_reg, param_grid, cv=5,
                         scoring="neg_mean_squared_error",
                         return_train_score=True, verbose=1)

# perform grid search with the validdation set 2 
grid_search.fit(X_val_2, y_val_2.values.ravel())

# score = 71.94692616588618

# look at evaluation scores of all parameter combinations
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# saves the model 
joblib.dump(forest_reg, "models/ExtraTrees_tuned.pkl")

# load the model if needed
# RandomForest_loaded=joblib.load("models/ExtraTree_tuned.pkl")"""



##################################


# fifth try with even more trees

param_grid = [
    {"n_estimators": [10000], "max_features": [100]},
    ]

extra_reg =  ExtraTreesRegressor()

grid_search = GridSearchCV(extra_reg, param_grid, cv=5,
                         scoring="neg_mean_squared_error",
                         return_train_score=True, verbose=1)

# perform grid search with the validdation set 2 
grid_search.fit(X_val_2, y_val_2.values.ravel())

# score = 

# look at evaluation scores of all parameter combinations
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# saves the model 
joblib.dump(forest_reg, "models/ExtraTrees_tuned.pkl")

# load the model if needed
# RandomForest_loaded=joblib.load("models/ExtraTree_tuned.pkl")"""

