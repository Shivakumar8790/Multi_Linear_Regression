# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 21:12:11 2023

@author: 91879
"""
#  Problem--1
# Prepare a prediction model for profit of 50_startups data.
# Do transformations for getting better predictions of profit and
# make a table containing R^2 value for each prepared model.

# R&D Spend -- Research and devolop spend in the past few years
# Administration -- spend on administration in the past few years
# Marketing Spend -- spend on Marketing in the past few years
# State -- states from which data is collected
# Profit  -- profit of each state in the past few years


# Step-1 Import file
import pandas as pd

df1 = pd.read_csv("50_Startups.csv")
df1
df1.shape

#check for missing values
df1.isna().sum()


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df1["state"] = LE.fit_transform(df1["State"])
df1["state"]

# Correlation Matrix
df1.corr()

#Format the plot background and scatter plots for all the variables
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df1)

## Step-2 EDA Exploratory data analysis
import matplotlib.pyplot as plt
plt.scatter(df1["R&D Spend"],df1["Profit"])
plt.show()

plt.scatter(df1["Marketing Spend"],df1["Profit"])
plt.show()

plt.scatter(df1["Administration"],df1["Profit"])            ### No relation ship
plt.show()

## Step-3 Split the variables as X and Y
#X = df1[["R&D Spend"]]                                                 #============== Case-1
#X = df1[["R&D Spend","Marketing Spend"]]                        #============== Case-2
X = df1[["R&D Spend","Marketing Spend","Administration"]]             #============== Case-3   <<<Good>>>>
Y = df1["Profit"]

## Step-4 Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)

LR.intercept_
LR.coef_

## Step-5 Model Predicted the Values

Y_pred = LR.predict(X)
Y_pred

## Step-6 Model evaluation

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean square error",mse.round(3))

import numpy as np
print("Root mean square value",np.sqrt(mse).round(3))

from sklearn.metrics import r2_score

r2 = r2_score(Y, Y_pred)
print("R square value",r2.round(3))

#=====================================================================================================================
##### Case-1
#   Root mean square value 9226.101
#   R square value 0.947

##### Case-2
#   Root mean square value 8881.886
#   R square value 0.95

##### Case-3
#   Root mean square value 8855.344
#   R square value 0.951




