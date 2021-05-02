import numpy as np
import pandas as pd
import re 
import nltk
from nltk.stem.porter import PorterStemmer

comments = pd.read_excel("comments.xlsx")
ps = PorterStemmer()
nltk.download('stopwords') 
# anlamsız kelimeleri indirir ing için
#Restorant yorumları için bize lazım olan kelimeler örneğin:
# Awesome , fresh , bad , not,recommend vs vs duygu ve anlam belirtlenler

from nltk.corpus import stopwords

compiledList=[]
for i in range(len(comments)):
    comment = re.sub('[^a-zA-Z]',' ',comments['Review'][i])
    # a dan z ve A-Z değerleri olmayanları '' ile değiştir demek
    comment = comment.lower()
    comment= comment.split()  
    comment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    # Burada ise kelime stopword değilse kelimenin kökünü bulup listeye atıyoruz.
    comment = ' '.join(comment)
    compiledList.append(comment)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(compiledList).toarray()  #Bağımsız Değişken
y = comments.iloc[:,1].values # Bağımlı değişkenler ( liked 1 ve 0 ları alır)


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.20,random_state=0)
  
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

print(cm) # %72.5 accurary













