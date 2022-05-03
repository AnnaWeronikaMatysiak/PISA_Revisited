#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:47:56 2022

@author: Jo
"""

"""
The reading score scale has a mean of 500 points 
and a standard deviation of 100 points.

PISA defines "low performing" students in reading as having a score 
under 407 points, see p. 35 in:
https://read.oecd-ilibrary.org/education/low-performing-students_9789264250246-en#page37

40 score points is considered the equivalent of a full year of schooling.
"""

#%% create dataset with low performaing boys in reading

# read in prepared dataset
PISA_prepared = pd.read_csv("data/PISA_prepared.csv")
def drop_first_entry(df):
    df.drop(df.columns[[0]], axis = 1, inplace = True)

drop_first_entry(PISA_prepared)

# filter by gender
PISA_male = PISA_prepared[PISA_prepared["gender"] == 1] 

# filter for low performing boys (reading score below treshold)
PISA_low_male = PISA_male[PISA_male["read_score"] < 407]

# divide X and y
X_low = PISA_low_male.drop(columns=["read_score"])
y_low = PISA_low_male["read_score"]
y_low = y_low.to_frame()


#%% fit final model on low performing boys in reading

# load the final model
import joblib
final_model = joblib.load("models/final_ridge_model.pkl")

# fit the model to boys
final_model.fit(X_low, y_low)


#%% prepare coefficients ordered by weight

# choose ridge regression as a step in the model pipeline to get coefficients
coefficients = final_model.named_steps['ridge'].coef_

# reshape row to column and get feature names
coefficients = coefficients.reshape(-1, 1)
feature_names = X_low.columns

# generate dataframe combining coefficients and column names
predictors = pd.DataFrame(coefficients, feature_names)
zipped = list(zip(feature_names, coefficients))
predictors = pd.DataFrame(zipped, columns=["feature", "coefficient"])

# sort by the absolute value of coefficients
predictors = predictors.sort_values('coefficient', ascending=False, key=abs)

# filter out country-ID's (we are interested in individual features of students)
predictors_low_male = predictors[predictors["feature"].str.contains("CNTRYID")==False]

# save as csv
predictors_low_male.to_csv("data/predictors_low_male.csv")




