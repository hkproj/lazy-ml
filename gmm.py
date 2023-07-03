import numpy as np
import utils

class GaussianMixtureModel():

    def __init__(self, k: int, X: np.ndarray) -> None:
        self.k = k

        self.n = X.shape[0]
        self.d = X.shape[1]
        self.X = X
        
        # Initialize the parameters
        self.mu = np.random.rand(k, self.d)
        self.cov_matrix = np.array([np.eye(self.d)] * k)
        self.pi = np.random.rand(k)
        # Normalize pi
        self.pi = self.pi / np.sum(self.pi)
        # Responsibility matrix
        self.r  = np.zeros((self.n, k))

    def _e_step(self) -> None:
        for i in range(self.n):
            for j in range(self.k):
                # Compute the pdf for the ith data point and the jth gaussian
                self.r[i, j] = self.pi[j] * utils.pdf_multivariate_normal_distribution(self.X[i], self.mu[j], self.cov_matrix[j])
        self.r = self.r / np.sum(self.r, axis=1, keepdims=True)        

    def _m_step(self) -> None:
        # Compute the new parameters
        for j in range(self.k):
            N_j = np.sum(self.r[:, j]) # Sum of the responsibilities for the jth gaussian
            self.mu[j] = np.sum(self.r[:, j].reshape(-1, 1) * self.X, axis=0) / N_j
            self.cov_matrix[j] = (1 / N_j) * np.sum(self.r[:, j].reshape(-1, 1, 1) * np.matmul((self.X - self.mu[j]).reshape(-1, self.d, 1), (self.X - self.mu[j]).reshape(-1, 1, self.d)), axis=0)
            self.pi[j] = N_j / self.n

    def fit(self) -> None:
        """
        Fits the model to the data.
        """
        # E-step
        self._e_step()
        # M-step
        self._m_step()