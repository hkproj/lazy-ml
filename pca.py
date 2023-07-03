import numpy as np

class PrincipalComponentAnalysis():

    def reduce(self, X: np.ndarray, k: int) -> np.ndarray:
        # 1. Standardize the data
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        # 2. Compute the covariance matrix
        cov_matrix = np.cov(X, ddof=0, rowvar=False)

        # 3. Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 4. Sort the eigenvalues and eigenvectors in descending order (the PCA is the eigenvector with the highest eigenvalue)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 5. Select the first k eigenvectors
        eigenvectors = eigenvectors[:, :k]

        # 6. Compute the new data
        reduced_data = np.matmul(X, eigenvectors)

        # How much of the variance is explained by the first k eigenvectors?
        explained_variance = np.sum(eigenvalues[:k]) / np.sum(eigenvalues)

        return reduced_data, explained_variance

        
        

