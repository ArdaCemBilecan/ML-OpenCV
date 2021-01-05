import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 

data = pd.read_csv("insurance.csv")

print(data.columns)
# Y Ekseni
expenses = data.expenses.values.reshape(-1,1)
print(expenses)

# X Ekseni

AgeBmis = data.iloc[:,[0,2]].values  # age ile bmi al

print(AgeBmis)
regression = LinearRegression()
regression.fit(AgeBmis,expenses)

print("--------PREDICT ------")

print(regression.predict([[20,20]])) # 20 yaşındaki biri ve 20bmi'se sahip birinin ort harcaması verir
print(regression.predict([[20,21]]))
print(regression.predict([[20,22]])) # Bmi arttıkça baklaım harcama artmış mı onu gördük
print(regression.predict([[20,23]]))
print(regression.predict([[20,24]]))

print("---------------------")


print(regression.predict(np.array([[20,20],[30,20],[40,20],[50,20]]))) # yaş arttıkça harcama artmış mı

print("--------------------------------")

print(r2_score(expenses,regression.predict(AgeBmis)))




  