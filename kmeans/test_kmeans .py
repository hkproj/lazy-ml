import pytest
import numpy as np
from kmeans.kmeans import KMeans, InitializationMethod, StoppingCriteria
from sklearn.datasets import make_blobs

def test_initialization_random():
    # Randomly generate N data points with d features
    N = 100
    d = 2
    k = 3
    X = np.random.rand(N, d)
    for init_method in [InitializationMethod.Random, InitializationMethod.KMeansPlusPLus]:
        kmeans = KMeans(k=k, init_method=init_method)
        assert kmeans.k == k
        assert kmeans.init_method == init_method
        centroids = kmeans._initialize_centroids(X)
        assert centroids.shape == (k, d)

def test_assign_clusters():
    N = 100
    d = 32
    k = 5

    # Generate data points from gaussian clusters
    X, y = make_blobs(n_samples=N, n_features=d, centers=k, cluster_std=0.60, random_state=0, shuffle=True)

    for init_method in [InitializationMethod.Random, InitializationMethod.KMeansPlusPLus]:
        kmeans = KMeans(k=k, init_method=init_method)
        assignments, centroids = kmeans.fit(X, stop_criteria=StoppingCriteria.Convergence)
        assert assignments.shape == (N,)
        assert centroids.shape == (k, d)
