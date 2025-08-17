import numpy as np

def log_raised_cosine_pdf(D: np.ndarray, mu: float, s: float) -> np.ndarray:
    """Log-raised cosine probability density function."""
    D = np.log(D)
    mu = np.log(mu)
    res = np.where(np.abs(D - mu) < s, 0.5/s * (1 + np.cos(np.pi * (D - mu) / s)), 0)

    return res

# n = 1000000
# D = np.linspace(0.001, 1, n)
# sigma = 0.52
# mu = 0.15
# s = sigma/np.sqrt(1/3-2/np.pi**2)
# pD = log_raised_cosine_pdf(D, mu, s)/D
# print(sum(pD/(1-0.001)/n))
# import matplotlib.pyplot as plt
# plt.plot(D, pD/100)
# plt.show()