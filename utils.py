import numpy as np

def pdf_multivariate_normal_distribution(x: np.ndarray, mu: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Computes the pdf of the multivariate normal distribution.
    """
    d = x.shape[0]
    x = x.reshape(-1, 1)
    mu = mu.reshape(-1, 1)
    cov_matrix = cov_matrix.reshape(d, d)
    return (1 / np.sqrt(np.linalg.det(cov_matrix) * (2 * np.pi)**d)) * np.exp(-0.5 * np.matmul(np.matmul((x - mu).T, np.linalg.inv(cov_matrix)), (x - mu)))