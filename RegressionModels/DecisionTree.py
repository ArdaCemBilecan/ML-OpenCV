import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


data = pd.read_csv("positions.csv")
print(data.columns)


level = data.iloc[:,1].values.reshape(-1,1)
salary = data.iloc[:,2].values.reshape(-1,1)
#print(level)
#print(salary)

regression = DecisionTreeRegressor()
regression.fit(level,salary)

print("Tahmin : ",regression.predict([[8.3]]))

plt.scatter(level,salary,color="blue")
x = np.arange(min(level),max(level),0.01).reshape(-1,1)
plt.plot(x,regression.predict(x),color="red")
plt.xlabel("Level")
plt.ylabel("Salary")

plt.title("Decision Model")
plt.show()
