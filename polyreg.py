import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
lin_reg2 =LinearRegression()
lin_reg2.fit(X_poly, y)

#visualising the linear regression results
mpl.scatter(X, y, color='red')
mpl.plot(X, lin_reg.predict(X), color='blue')
mpl.xlabel('Position')
mpl.ylabel('Salary')
mpl.title('Linear regression visualisation')
mpl.show()
#visualising the polynomial regression results
mpl.scatter(X, y, color='red')
mpl.plot(X, lin_reg2.predict(X_poly),color='blue')
mpl.xlabel('position')
mpl.ylabel('salary')
mpl.title('polym=nomial regression visualisation')
mpl.show()