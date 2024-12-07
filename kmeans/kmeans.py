import numpy as np
from enum import Enum, Flag, auto
from typing import Optional

class InitializationMethod(Enum):
    Random = 1
    KMeansPlusPLus = 2

class StoppingCriteria(Flag):
    Convergence = auto()
    MaxIterations = auto()
    CentroidsChangeThreshold = auto()

class KMeans:

    def __init__(self, k: int, init_method: InitializationMethod):
        assert k > 0, "k should be greater than 0"
        self.k = k
        self.init_method = init_method

    def _initialize_centroids(self, X: np.array) -> np.array:
        # X: [N, d] where N is the number of data points and d is the number of features

        if self.init_method == InitializationMethod.Random:
            # Randomly choose K distinct data points from the dataset as centroids
            chosen_centroids_idx = np.random.choice(X.shape[0], size=self.k, replace=False)
            centroids = X[chosen_centroids_idx] # [k, d]
        elif self.init_method == InitializationMethod.KMeansPlusPLus:
            # The algorithm is as follows:
            # 1. Choose one center uniformly at random from among the data points.
            # 2. For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
            # 3. Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x).
            # 4. Repeat Steps 2 and 3 until k centers have been chosen.

            chosen_centroids_idx = np.random.choice(X.shape[0], size=1, replace=False)
            centroids = X[chosen_centroids_idx] # [1, d]
            for _ in range(self.k-1):
                # Compute the squared distance between each data point and each centroid
                diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :] # [N, k, d]
                assert diffs.shape == (X.shape[0], centroids.shape[0], X.shape[1]), f"Expected diffs to have shape (N, k, d) but got {diffs.shape}"
                distances = np.sum(diffs**2, axis=2, keepdims=False) # [N, k]
                assert distances.shape == (X.shape[0], centroids.shape[0]), f"Expected distances to have shape (N, k) but got {distances.shape}"
                # Compute the min distance between each data point and its nearest centroid
                min_distances = np.min(distances, axis=1, keepdims=False) # [N]
                assert min_distances.shape == (X.shape[0],), f"Expected min_distances to have shape {(X.shape[0])} but got {min_distances.shape}"
                # Compute the probability distribution
                prob = min_distances / np.sum(min_distances)
                # Choose a new centroid
                chosen_centroids_idx = np.concatenate([chosen_centroids_idx, np.random.choice(X.shape[0], size=1, replace=False, p=prob)], axis=0)
                centroids = X[chosen_centroids_idx]
        else:
            raise ValueError(f"Invalid initialization method: {self.init_method}")
    
        assert centroids is not None, "Centroids should not be None"
        assert centroids.shape == (self.k, X.shape[1]), f"Expected centroids to have shape ({self.k}, {X.shape[1]}) but got {centroids.shape}"
        return centroids
        
    def _assign_clusters(self, X: np.array, centroids: np.array) -> np.array:
        # X: [N, d]
        # centroids: [k, d]
        diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :] # [N, k, d]
        assert diffs.shape == (X.shape[0], centroids.shape[0], X.shape[1]), f"Expected diffs to have shape (N, k, d) but got {diffs.shape}"
        distances = np.sum(diffs**2, axis=2, keepdims=False) # [N, k]
        assignments = np.argmin(distances, axis=1) # [N]
        return assignments
    
    def _update_centroids(self, X: np.array, assignments: np.array) -> np.array:
        # X: [N, d]
        # assignments: [N]
        new_centroids = np.empty((self.k, X.shape[1]))
        for i in range(self.k):
            mask = assignments == i
            new_centroids[i] = np.mean(X[mask], axis=0) # [d]
        assert new_centroids.shape == (self.k, X.shape[1]), f"Expected new_centroids to have shape ({self.k}, {X.shape[1]}) but got {new_centroids.shape}"
        return new_centroids
    
    def _distance_threshold_convergence(self, centroids: np.array, old_centroids: np.array, threshold: float) -> bool:
        # centroids: [k, d]
        # old_centroids: [k, d]
        # threshold: float
        distances = np.linalg.norm(centroids - old_centroids, axis=1) # [k]
        return np.all(distances < threshold)        
    
    def _check_stop_criteria(self, stop_criteria: StoppingCriteria, max_iterations: int, dist_threshold: float, old_centroids: np.array, new_centroids: np.array, it: int) -> bool:
        if stop_criteria & StoppingCriteria.MaxIterations:
            if it >= max_iterations:
                return True
        if stop_criteria & StoppingCriteria.CentroidsChangeThreshold:
            if old_centroids is not None and self._distance_threshold_convergence(new_centroids, old_centroids, dist_threshold):
                return True
        return False

    def fit(self, X: np.array, stop_criteria: StoppingCriteria, max_iterations: Optional[int] = None, centroids_change_threshold: Optional[float] = None) -> tuple[np.array, np.array]:

        if self.k > X.shape[0]:
            raise ValueError(f"Number of clusters k={self.k} should be less than or equal to the number of data points N={X.shape[0]}")

        # By default, the convergence criteris is set
        if StoppingCriteria.Convergence not in stop_criteria:
            stop_criteria |= StoppingCriteria.Convergence

        # Check stopping criteria requirements
        if stop_criteria & StoppingCriteria.MaxIterations:
            assert max_iterations is not None, "max_iterations should be provided"
        if stop_criteria & StoppingCriteria.CentroidsChangeThreshold:
            assert centroids_change_threshold is not None, "centroids_change_threshold should be provided"

        old_centroids = None
        old_assignments = None

        # X: [N, d] 
        new_centroids = self._initialize_centroids(X)

        # Initialize the cluster assignments
        new_assignments = self._assign_clusters(X, new_centroids)

        it = 0

        while not self._check_stop_criteria(stop_criteria, max_iterations, centroids_change_threshold, old_centroids, new_centroids, it):
            old_centroids, old_assignments = new_centroids, new_assignments

            # Update the centroids
            new_centroids = self._update_centroids(X, new_assignments)
            # Re-assign the clusters
            new_assignments = self._assign_clusters(X, new_centroids)
            # Check if the assignments have changed
            if StoppingCriteria.Convergence in stop_criteria and np.array_equal(old_assignments, new_assignments):
                break
            it += 1
        
        return new_assignments, new_centroids

        

        



    
