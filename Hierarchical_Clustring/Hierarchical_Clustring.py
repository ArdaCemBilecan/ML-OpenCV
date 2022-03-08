import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

data = pd.read_csv('customers.csv')
X = data.iloc[:,3:].values
ac = AgglomerativeClustering(n_clusters=4 , affinity='euclidean' , linkage='ward')
y_pred = ac.fit_predict(X)


plt.scatter(X[y_pred==0 ,0],X[y_pred==0 ,1] , s=100 , c='red')
plt.scatter(X[y_pred==1 ,0],X[y_pred==1 ,1] , s=100 , c='blue')
plt.scatter(X[y_pred==2 ,0],X[y_pred==2 ,1] , s=100 , c='green')
plt.scatter(X[y_pred==3 ,0],X[y_pred==3 ,1] , s=100 , c='purple')
plt.show()

dendogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()

