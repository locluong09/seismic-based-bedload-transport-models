# Example usage and testing
import pytest
import numpy as np

from models import SaltationModel, SedimentParams, SeismicParams

def test_saltation_model():
    # Test with different approaches
    f = np.linspace(0.001, 20, 100)
    # D = 0.3  
    D = np.asarray(np.arange(0.1,1,0.1))
    D50 = 0.4
    print(D)
    H = 4.0     
    W = 50
    theta = np.tan(1.4*np.pi/180)
    r0 = 600
    qb = 1e-3
    # Method 1: Using the class directly
    model = SaltationModel()
    psd = model.forward_psd(f, D, H, W, theta, r0, qb, D50 = D50)
    assert psd is not None
    # print(np.shape(psd))
    # psd_dB = 10*np.log10(psd)

    # plt.figure(figsize=(10, 6))
    # plt.plot(f, psd_dB)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power Spectral Density')
    # plt.title('Saltation Mode Model - Sediment Transport PSD')
    # plt.grid(True, alpha=0.3)
    # plt.ylim(-170, -110)
    # plt.show()
