# Polynomial_Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #predictors in matrix form
y = dataset.iloc[:, 2].values

#for this dataset we won't need to split our data because when we look at our data we can see that we
#only have ten observations. it doesn't make much sense to split this data into a traing and a testing set
#we dont have so much information and we have to make the accurate [prediction]
# Splitting the dataset into the Training set and Test set
"""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# also no need for feature scalling
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial Regression to the datasest
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2= LinearRegression()
lin_reg2.fit(X_poly, y)

#Visualising the Linear Regression results
plt.scatter(X,y, color='red')
plt.plot(X,lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear_Regression)')
plt.xlabel('Position levels')
plt.ylabel('Salary')
plt.show()

#Visualising the polynomial Regression results
#X_grid = np.arange(min(X)),max(X), 0.1)
#X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X,y, color='red')
plt.plot(X,lin_reg2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Linear_Regression)')
plt.xlabel('Position levels')
plt.ylabel('Salary')
plt.show()
