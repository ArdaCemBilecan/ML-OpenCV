import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("positions.csv")
print(data.columns)

level = data.iloc[:,1].values.reshape(-1,1)
salary = data.iloc[:,2].values # y must be 1D

#print(level,salary)

regression = RandomForestRegressor(n_estimators=100,random_state=(0)) 
# kaç tane decision tree oluş. yazılır
# Random state çalıştırılacak algoritmayı söyler
regression.fit(level,salary)

print(regression.predict([[8.3]]))




