#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:38:19 2022

@author: macbook
"""

import numpy as np 
import sklearn.preprocessing as Polynomialfeatures
import sklearn.preprocessing as standardScaler
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("PositionDataset.csv")
X= dataset.iloc[:,1:-1].values 
Y=dataset.iloc[:,-1].values


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X, Y)
Y_pred= regressor.predict(X)

plt.scatter(X, Y, color= 'red')
plt.plot(X, regressor.predict(X), color= 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#----------------------------

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)

lin_reg = LinearRegression()

lin_reg.fit(x_poly, Y)
y_pred2 = regressor.predict(X)

x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(X, Y, color = 'red')
plt.plot(x_grid, lin_reg.predict(poly_reg.fit_transform(x_grid)),color = 'blue')
plt.title('Truth or Bluff (Plynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#-------------------------------------------------
from sklearn.preprocessing import StandardScaler

#feateur scalling
sc_x =StandardScaler()
sc_y= StandardScaler()
X= sc_x.fit_transform(X.reshape(-1,1))
Y=sc_y.fit_transform(Y.reshape(-1,1))
from sklearn.svm import SVR
#svr
regressor =SVR(kernel='rbf')
regressor.fit(X,Y)

Y_pred3 =regressor.predict(sc_x.transform([[5.5]]))
Y_pred3 =sc_y.inverse_transform(Y_pred)



plt.scatter(X, Y, color= 'red')
plt.plot(X, regressor.predict(X), color= 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


















