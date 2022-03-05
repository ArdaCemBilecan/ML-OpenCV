import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

datas = pd.read_csv('customers.csv')
X = datas.iloc[:,3:].values

kmeans = KMeans(n_clusters = 3 , init='k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)
 
# Finding the best number of 'k'
results = []
for i in range(1,11):
   kmeans = KMeans(n_clusters = i , init='k-means++', random_state=33)
   kmeans.fit(X)
   results.append(kmeans.inertia_)
    
plt.plot(range(1,11), results) 
plt.savefig('WCSS_Results.png')
