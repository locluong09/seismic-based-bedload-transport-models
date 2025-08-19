# Example usage and testing
import numpy as np
from seismic_bedload.utils import log_raised_cosine_pdf
import matplotlib.pyplot as plt
from seismic_bedload import SaltationModel

def test_saltation_model():
    f = np.linspace(0.001, 20, 100)
    D = 0.3  
    # D = np.asarray(np.arange(0.1,1,0.1))
    sigma = 0.52
    mu = 0.15
    s = sigma/np.sqrt(1/3-2/np.pi**2)
    pD = log_raised_cosine_pdf(D, mu, s)/D
    D50 = 0.4
    H = 4.0     
    W = 50
    theta = np.tan(1.4*np.pi/180)
    r0 = 600
    qb = 1e-3
    model = SaltationModel()

    psd = model.forward_psd(f, D, H, W, theta, r0, qb, D50 = D50, pdf = pD)

    assert psd is not None

    psd_dB = 10*np.log10(psd)
    plt.figure(figsize=(10, 6))
    plt.plot(f, psd_dB)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.grid(True, alpha=0.3)
    plt.ylim(-170, -110)
    plt.show()

if __name__ == "__main__":
    test_saltation_model()
