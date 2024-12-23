# -*- coding: utf-8 -*-
"""Untitled89.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rsU1cBbU-QhBFKQLxYFmqhpka-hFQmdW
"""

#----------------------------------------------------------------------
#NERS 590: Applied Machine Learning for Nuclear Engineers
#In-class sript: Linear regression demonstration to predict car price
#Date: 7/28/2024
#Author: Majdi I. Radaideh
#----------------------------------------------------------------------

#You need to install pandas, numpy, and scikit-learn
#e.g., pip install scikit-learn
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Define all metrics in a handy function
def metrics(y, yhat):
    diff=np.abs(y-yhat)
    perc_diff=((y-yhat)/y)
    ybar=np.mean(y)
    mae=np.mean(diff)
    mape=np.mean(np.abs(perc_diff))*100
    rmse=np.sqrt(np.mean(diff**2))
    rmspe=np.sqrt(np.mean(perc_diff**2))*100
    r2=1-np.sum(diff**2)/np.sum((y-ybar)**2)
    return mae, mape, rmse, rmspe, r2

#--------------------------
# Data loading & Processing
#--------------------------
url='https://raw.githubusercontent.com/MajdiRadaideh/S097data/main/cars.csv'
data=pd.read_csv(url)
data=data.dropna()
npdata=data.values

#assign features and outputs
X=npdata[:,0:8]
Y=npdata[:,-1]

#split the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=1)

#scale the features
xscaler =MinMaxScaler()
Xtrain=xscaler.fit_transform(Xtrain)
Xtest=xscaler.transform(Xtest)

#did not scale Y data since we have a single output

#----------------------------------
#Multiple Linear Regression (MLR)
#----------------------------------
lr = LinearRegression()
lr.fit(Xtrain, Ytrain)
ylr = lr.predict(Xtest)
lr_met = metrics(y=Ytest, yhat=ylr)

#--------------------------
#Ridge
#--------------------------
rg = Ridge(alpha=1.0)
rg.fit(Xtrain, Ytrain)
yrg = rg.predict(Xtest)
rg_met = metrics(y=Ytest, yhat=yrg)

#--------------------------
#LASSO
#--------------------------
ls = Lasso(alpha=1.0)
ls.fit(Xtrain, Ytrain)
yls = ls.predict(Xtest)
ls_met = metrics(y=Ytest, yhat=yls)

#--------------------------
#ElasticNet
#--------------------------
elnet = ElasticNet(alpha=1.0, l1_ratio=0.6)
elnet.fit(Xtrain, Ytrain)
y_elnet = elnet.predict(Xtest)
elnet_met = metrics(y=Ytest, yhat=y_elnet)

#print all metrics after combining them into a dataframe
all_metrics = pd.DataFrame([lr_met, rg_met, ls_met, elnet_met], columns=['MAE', 'MAPE', 'RMSE', 'RMSPE', 'R2'],
                           index=['MLR', 'Ridge', 'Lasso', 'ElasticNet'])

print(all_metrics)
all_metrics

# Plotting the prediction vs target values
plt.figure(figsize=(12, 8))
plt.scatter(Ytest, ylr, label='MLR', color='blue', alpha=0.6)
plt.scatter(Ytest, yrg, label='Ridge', color='red', alpha=0.6)
plt.scatter(Ytest, yls, label='Lasso', color='green', alpha=0.6)
plt.scatter(Ytest, y_elnet, label='ElasticNet', color='black', alpha=0.6)
plt.plot([min(Ytest), max(Ytest)], [min(Ytest), max(Ytest)], color='black', linestyle='--')
plt.xlabel('Target Values')
plt.ylabel('Predicted Values')
plt.title('Prediction vs Target Values')
plt.legend()
plt.show()