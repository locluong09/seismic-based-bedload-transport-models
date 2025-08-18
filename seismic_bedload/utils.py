import numpy as np

def log_raised_cosine_pdf(D: np.ndarray, mu: float, s: float) -> np.ndarray:
    """Log-raised cosine probability density function."""
    D = np.log(D)
    mu = np.log(mu)
    res = np.where(np.abs(D - mu) < s, 0.5/s * (1 + np.cos(np.pi * (D - mu) / s)), 0)
    return res