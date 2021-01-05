import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("positions.csv")
print(data.columns)

level = data.iloc[:,1].values.reshape(-1,1)
salary = data.iloc[:,2].values.reshape(-1,1)

# regression = LinearRegression()
# regression.fit(level,salary)

for i in range(2,8):
    # Her plotTa bir önceki derece ile karşılaştırılması yapılıyor.
    
    regressionPoly1 = PolynomialFeatures(degree=i)
    regressionPoly2 = PolynomialFeatures(degree=i-1)
    
    levelPoly1 = regressionPoly1.fit_transform(level)
    levelPoly2 = regressionPoly2.fit_transform(level)
    
    regression = LinearRegression()
    regression.fit(levelPoly1,salary)
    
    regression2 = LinearRegression()
    regression2.fit(levelPoly2,salary)
    
    plt.scatter(level,salary,color="red")
    plt.plot(level,regression.predict(levelPoly1),color="blue")
    plt.plot(level,regression2.predict(levelPoly2),color="black")
    plt.show()


