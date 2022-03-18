import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from tensorflow.keras.utils import to_categorical

# Data Preprocessing
data = pd.read_csv('train.csv') # this is train data
data = data.dropna(subset=['Embarked','Sex','Age','Parch','SibSp','Pclass'],how='any')

test_data = pd.read_csv('test.csv')
test_data = test_data.dropna(subset=['Embarked','Sex','Age','Parch','SibSp','Pclass'],how='any')

submission = pd.read_csv('titanic_submission.csv')
merged_test_data = pd.merge(test_data, submission, on= "PassengerId", how= "inner")
#['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived']


ohe = OneHotEncoder()
Pclass_train = (data.iloc[:,2].values).reshape(-1,1)
Pclass_train = ohe.fit_transform(Pclass_train).toarray()
Pclass_train = pd.DataFrame(data = Pclass_train , columns=['Class1','Class2','Class3'])

Pclass_test = (merged_test_data.iloc[:,1].values).reshape(-1,1)
Pclass_test = ohe.fit_transform(Pclass_test).toarray()
Pclass_test = pd.DataFrame(data = Pclass_test , columns=['Class1','Class2','Class3'])


Embarked_train = (data.iloc[:,11].values).reshape(-1,1)
Embarked_train = ohe.fit_transform(Embarked_train).toarray()
Embarked_train = pd.DataFrame(data = Embarked_train , columns=['C','Q','S'])

Embarked_test = (merged_test_data.iloc[:,10].values).reshape(-1,1)
Embarked_test = ohe.fit_transform(Embarked_test).toarray()
Embarked_test = pd.DataFrame(data = Embarked_test , columns=['C','Q','S'])



le = LabelEncoder()
sex_train = data.iloc[:,4].values
sex_train = le.fit_transform(sex_train)
sex_train= pd.DataFrame(data=sex_train,columns=['Sex']) # 1 male , 0 female

sex_test = merged_test_data.iloc[:,3].values
sex_test = le.fit_transform(sex_test)
sex_test= pd.DataFrame(data=sex_test,columns=['Sex']) # 1 male , 0 female

parch_train = data.iloc[:,7].values
parch_train = pd.DataFrame(data=parch_train,columns=['Parch'])

parch_test = merged_test_data.iloc[:,6]
parch_test = pd.DataFrame(data=parch_test,columns=['Parch'])

sibsp_train = data.iloc[:,5].values
sibsp_train = pd.DataFrame(data=sibsp_train , columns=['SibSp'])
sibsp_test= merged_test_data.iloc[:,4].values
sibsp_test = pd.DataFrame(data=sibsp_test , columns=['SibSp'])

survived_train = data.iloc[:,1]
survived_train = le.fit_transform(survived_train)
survived_train = pd.DataFrame(data=survived_train , columns=['Survived'])
survived_test = merged_test_data.iloc[:,11]
survived_test = le.fit_transform(survived_test)
survived_test = pd.DataFrame(data=survived_test , columns=['Survived'])


# Generated x_train ,x_test list
x_train_list = [Pclass_train,Embarked_train,sex_train,parch_train,sibsp_train]
x_test_list = [Pclass_test,Embarked_test,sex_test,parch_test,sibsp_test]

# x_train , y_train created
x_train = pd.concat(x_train_list,axis=1)
x_train = x_train.values

x_test = pd.concat(x_test_list,axis=1)
x_test = x_test.values

# created y_train , y_test

y_train =  to_categorical(survived_train)
y_test = to_categorical(survived_test)

my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='Checkpoint/model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]


# created ANN model
model = Sequential()
model.add(Input(shape=(9,)))
model.add(Dense(512))
model.add(LeakyReLU())
model.add(Dense(1024))
model.add(LeakyReLU())
model.add(Dropout(0.25))
model.add(Dense(1024))
model.add(LeakyReLU())
model.add(Dense(512))
model.add(LeakyReLU())
model.add(Dropout(0.25))
model.add(Dense(2,activation='softmax'))
model.compile(optimizer='Nadam',loss='categorical_crossentropy',metrics=['accuracy'])

history  = model.fit(x_train , y_train , epochs=50 , validation_data=(x_test , y_test) , callbacks=[my_callbacks])




