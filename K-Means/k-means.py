# import libraries
from sklearn.cluster import KMeans
import numpy as np

# Load the data
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# Initialize the K-Means model
kmeans = KMeans(n_clusters=2)

# Fit the model to data
kmeans.fit(X)

# Get the cluster labeling
labels = kmeans.labels_

# Get the cluster centroids
centroids = kmeans.cluster_centers_

print("Cluster Labels:", labels)
print("Cluster Centroids:", centroids)
