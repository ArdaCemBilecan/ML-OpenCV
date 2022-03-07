import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
# read datas
data = pd.read_excel('Iris.xls')
X = data.iloc[:,:4].values
Y = data.iloc[:,4:].values

#label encoder
le = LabelEncoder()
Y = le.fit_transform(Y)


# split data for train and test
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=33)


# Standard Scaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logR = LogisticRegression(random_state=0)
logR.fit(X_train , y_train)
logr_pred = logR.predict(X_test)
print('Logistic Regression Accuracy: ',accuracy_score(y_test,logr_pred)) #0.9
                         

# KNN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1 , metric='minkowski')
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
print('KNN-minkowski Accuracy: ',accuracy_score(y_test,knn_pred)) 
#K =5 : 0.88 , k=1 : 0.94

# Manhattan 
knn = KNeighborsClassifier(n_neighbors=1 , metric='manhattan')
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
print('KNN-Manhattan Accuracy: ',accuracy_score(y_test,knn_pred)) 
#K=5: 0.9   , K=1 : 0.92

# Euclidean
knn = KNeighborsClassifier(n_neighbors=1 , metric='euclidean')
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
print('KNN-Euclidean Accuracy: ',accuracy_score(y_test,knn_pred)) 
#K=5: 0.88  , K=1 : 0.94  the best


# Support Vector Classifier
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train,y_train)
svc_pred = svc.predict(X_test)
print('SVC-Linear Accuracy: ',accuracy_score(y_test,svc_pred)) #0.9

#RBF
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)
svc_pred = svc.predict(X_test)
print('SVC-RBF Accuracy: ',accuracy_score(y_test,svc_pred)) #0.9 

# sigmoid
svc = SVC(kernel='sigmoid')
svc.fit(X_train,y_train)
svc_pred = svc.predict(X_test)
print('SVC-sigmoid Accuracy: ',accuracy_score(y_test,svc_pred)) #0.88



# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
gnb_pred = gnb.predict(X_test)
print('Gaussian-Naive Bayes Accuracy: ',accuracy_score(y_test,gnb_pred)) #0.86



# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='gini')
dtc.fit(X_train,y_train)
dtc_pred = dtc.predict(X_test)
print('Gaussian-DTC-Gini Accuracy: ',accuracy_score(y_test,dtc_pred)) # 0.88

# Entropy
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)
dtc_pred = dtc.predict(X_test)
print('Gaussian-DTC-Entropy Accuracy: ',accuracy_score(y_test,dtc_pred)) #0.86


# Random-Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10,criterion='gini')
rfc.fit(X_train,y_train)
rfc_pred = dtc.predict(X_test)
print('Gaussian-RFC-Gini Accuracy: ',accuracy_score(y_test,rfc_pred)) #0.86

# Entropy
rfc = RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(X_train,y_train)
rfc_pred = dtc.predict(X_test)
print('Gaussian-RFC-Entropy Accuracy: ',accuracy_score(y_test,rfc_pred)) #0.86



# Artificial Neurol Network
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(y_train)
test_labels = to_categorical(y_test)

model = Sequential()
model.add(Dense(512,activation='relu',input_shape=(4,)))
model.add(Dense(256,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train, train_labels , epochs=50,validation_data=(X_test,test_labels))
# Validation Accuracy 0.9











