import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 1:2].values


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#fitting svr to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)
 

#predict a new result
#y_pred = regressor.predict(6.5)

#visualising the polynomial regression results
mpl.scatter(X, y, color='red')
mpl.plot(X, regressor.predict(X), color='blue')
mpl.xlabel('position')
mpl.ylabel('salary')
mpl.title('polynomial regression visualisation')
mpl.show()
