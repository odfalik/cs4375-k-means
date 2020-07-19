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
    header=[0,1,2]                                  # Header of CSV file is 3 lines tall
)
print('Download complete.')

features = df['echonest', 'audio_features'].drop(0) # Drop line of NaNs
# features = features.iloc[:,0:3]                     # Limit dimensions
print(features.head())                              # Show subset of dataset

# # Finding optimal K value
# sses = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10)
#     kmeans.fit(features)
#     sses.append(kmeans.inertia_)
# plt.plot(range(1, 11), sses)
# plt.xlabel('Number of clusters')
# plt.ylabel('SSE')
# plt.title('Elbow Method')
# print('SSES:', sses)

# Run algorithm
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=500, n_init=10)
pred_y = kmeans.fit_predict(features)
axes = (0, 1, 2)

# # 3D plot of clusters
# ax = plt.axes(projection ='3d')
# ax.scatter(features.iloc[:,axes[0]],            features.iloc[:,axes[1]],             features.iloc[:,axes[2]],             c=pred_y,     s=.5);  # samples
# ax.scatter(kmeans.cluster_centers_[:,axes[0]],  kmeans.cluster_centers_[:,axes[1]],   kmeans.cluster_centers_[:,axes[2]],   color='red',  s=100, depthshade=False);  # means
# ax.set_xlabel(features.columns[axes[0]])
# ax.set_ylabel(features.columns[axes[1]])
# ax.set_zlabel(features.columns[axes[2]])

# # 2D plot of clusters
# plt.scatter(features.iloc[:,axes[0]],             features.iloc[:,axes[1]],             c=pred_y,   s=1)
# plt.scatter(kmeans.cluster_centers_[:,axes[0]],   kmeans.cluster_centers_[:,axes[1]],   c='red',    s=50)

plt.show()