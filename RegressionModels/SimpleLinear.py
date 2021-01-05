import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score   #R square için


data = pd.read_csv("hw_25000.csv")


boy = data.Height.values.reshape(-1,1)
kilo = data.Weight.values.reshape(-1,1)

regression = LinearRegression()
regression.fit(boy,kilo)  # x=boy , y=kilo için line fit yapılır 



print(regression.predict([[60]]).reshape(-1,1))
print(regression.predict([[62]]).reshape(-1,1)) #• 62 inç boyunda olan birinin hangi kiloda olduğunu verir
print(regression.predict([[64]]).reshape(-1,1))
print(regression.predict([[66]]).reshape(-1,1))
print(regression.predict([[68]]).reshape(-1,1))
print(regression.predict([[70]]).reshape(-1,1))

plt.scatter(data.Height,data.Weight)
x = np.arange(min(data.Height),max(data.Height)).reshape(-1,1)

plt.plot(x,regression.predict(x).reshape(-1,1),color="red")

plt.title("Simple Linear")

plt.xlabel("Boy")
plt.ylabel("Kilo")
plt.show()

print("------------------------------------------------------------------\n")
print(r2_score(kilo,regression.predict(boy))) # sapma miktarını söyleyecek