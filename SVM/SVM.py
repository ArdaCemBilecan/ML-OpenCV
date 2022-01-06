import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('C:/Users/ardac/OneDrive/Desktop/Udemy/veriler.csv')

print(data)

X = data.iloc[:,1:3] # height - weight
Y = data.iloc[:,3:4] # Age
x = X.values
y = Y.values
y = np.ravel(y.reshape(-1,1))

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=33) 

svr = SVR(kernel='poly')
svr.fit(x_train,y_train)
y_pred = svr.predict(x_test)



for i in range(len(y_pred)):
    print('Real Result: {real}  -  Prediction: {pre}'.format(real = y_test[i],pre = y_pred[i]))
    
