import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.DataFrame({
    'x' : [4,12,33,20,54,39,11,46,60,71,78,8,24,30,34,64,73,16,29,41,75],
    'y' : [11,5,50,46,78,31,28,7,64,18,21,34,23,56,61,76,58,3,41,68,15]
})

Kmeans = KMeans(n_clusters=3)
Kmeans.fit(df)
labels=Kmeans.predict(df)
centroids = Kmeans.cluster_centers_

# plotting
fig = plt.figure(figsize=(5,5))

colmap = {1: 'r', 2:'g', 3:'b'}
colors = map(lambda x: colmap[x+1], labels)
colors1 = list(colors)
plt.scatter(df['x'],df['y'], color=colors1, alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0,80)
plt.ylim(0,80)
plt.show()