import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


data = pd.read_csv('veriler.csv')

X = data.iloc[:,1:3] #Height - Weight
x = X.values
gender = data.iloc[:,4:5] # Gender

label = LabelEncoder()
gender = label.fit_transform(gender) # y values   0--> man , 1--> woman

x_train,x_test,y_train,y_test = train_test_split(x,gender,test_size=0.25 , random_state=33)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


knn = KNeighborsClassifier(n_neighbors=2 , metric='minkowski')

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm) # 5 True 1 False 





