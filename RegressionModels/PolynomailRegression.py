import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("positions.csv")
print(data.columns)
# Linear Regression 

level = data.iloc[:,1].values.reshape(-1,1)
salary = data.iloc[:,2].values.reshape(-1,1)

regression = LinearRegression()
regression.fit(level,salary)

guess1 = regression.predict([[8.3]])

# Poly Regression
regressionPoly = PolynomialFeatures(degree = 4)
levelPoly = regressionPoly.fit_transform(level)
regression2 = LinearRegression()
regression2.fit(levelPoly,salary)

guess2 = regression2.predict(regressionPoly.fit_transform([[8.3]]))


plt.scatter(level,salary,color="red")
plt.plot(level,regression.predict(level),color="blue")
plt.plot(level,regression2.predict(levelPoly),color="purple")
plt.show()



print("------Guesses-----\n")
print("Gues1 : ",guess1)
print("Guess2 : ",guess2)



