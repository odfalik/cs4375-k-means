import requests, io
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# data_url = 'https://utdallas.edu/~oxf170130/cs4375-k-means/echonest.csv'
data_url = 'data/echonest.csv'

print('Downloading dataset...')
df = pd.read_csv(
    data_url,
    header=[0,1,2]
)
print('Download complete.')

features = df['echonest', 'audio_features'].drop(0)
print(features.head())

# plt.scatter(features.iloc[:,1], features.iloc[:,6])
# plt.show()

# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10)
#     kmeans.fit(features)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=500, n_init=10)
pred_y = kmeans.fit_predict(features)
plt.scatter(features.iloc[:,1], features.iloc[:,6], s=1)
plt.scatter(kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,6], s=50, c='red')
plt.show()