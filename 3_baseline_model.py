# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:14:00 2022

@author: Anna
"""
#%% call setup file
import runpy
runpy.run_path(path_name = '/0_setup.py')

#%% necessary packages
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import joblib

#%% read in data
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

X_val_1 = pd.read_csv("data/X_val_1.csv") 
y_val_1 = pd.read_csv("data/y_val_1.csv")

X_val_2 = pd.read_csv("data/X_val_2.csv") 
y_val_2 = pd.read_csv("data/y_val_2.csv")

#%% droping first column
def drop_first_entry(df):
    df.drop(df.columns[[0]], axis = 1, inplace = True)

drop_first_entry(X_train)
drop_first_entry(X_val_1)
drop_first_entry(X_val_2)

X_train.shape
X_val_2.shape
X_val_1.shape
#%% linear regression

# training
lin_reg= LinearRegression()
lin_reg.fit(X_train, y_train)   

lin_reg.coef_
lin_reg.intercept_

# predicting
y_predicted=lin_reg.predict(X_val_1)

#evaluation
rmse= np.sqrt(mean_squared_error(y_val_1, y_predicted))
mae=mean_absolute_error(y_val_1, y_predicted)
r_2=r2_score(y_val_1, y_predicted)
print('RMSE_linear: ',rmse) # result: 68.5229128573592
print('MAE_linear:', mae) # result: 54.40138962766195
print('R_2_linear:', r_2) #result: 0.5920794258682984
# saves the model 
joblib.dump(lin_reg, "models/LinearRegression.pkl")

# load the model if needed
# lin_reg = joblib.load("models/LinearRegression.pkl")

#%%Saving Baseline Metrics in Table
d = {'Model': ['Baseline: Linear Regression'], 'RMSE': [round(rmse, 4)], 'MAE': [round(mae, 4)], 'R2': [round(r_2, 4)]}
table_baseline = pd.DataFrame(data=d)
table_baseline
table_baseline.style.to_latex("baseline_table.tex", index=False, caption="Baseline Evaluation Metrics")

#%% visualizing baseline

# import matplotlib.pyplot as plt
# import numpy as np

# # Version 1 - Bad Code
# def plotGraph(y_val_1,y_pred,lin_reg):
#     if max(y_val_1) >= max(y_pred):
#         my_range = int(max(y_val_1))
#     else:
#         my_range = int(max(y_pred))
#     plt.scatter(range(len(y_val_1)), y_val_1, color='blue')
#     plt.scatter(range(len(y_pred)), y_pred, color='red')
#     plt.title(lin_reg)
#     plt.show()
#     return

# y_val_1 = range(100)
# y_pred = np.random.randint(0, 100, 10)

# plotGraph(y_val_1, y_pred, "test")

# #Version 3 - 

# plt.figure(figsize=(10,10))
# plt.scatter(y_val_1, y_pred, c='crimson')
# plt.yscale('log')
# plt.xscale('log')

# p1 = max(max(y_pred), max(y_val_1)
# p2 = min(min(y_pred), min(y_val_1))
# plt.plot([p1, p2], [p1, p2], 'b-')
# plt.xlabel('True Values', fontsize=15)
# plt.ylabel('Predictions', fontsize=15)
# plt.axis('equal')
# plt.show()

# #Version 4 - n
#  plt.scatter(y_val_1, y_pred)
#  plt.xlabel('True Values ')
#  plt.ylabel('Predictions ')
#  plt.axis('equal')
#  plt.axis('square')
#  plt.show()
#  plt.xlim([0, plt.xlim()])
#  plt.ylim([0, plt.ylim()])
#  plt.show()



#d = {'Model': ['Baseline: Linear Regression'], 'RMSE': [round(rmse, 4)], 'MAE': [round(mae, 4)], 'R_2': [round(r_2, 4)]}
#table_baseline = pd.DataFrame(data=d)
#table_baseline
#%%
