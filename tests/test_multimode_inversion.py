# Example usage and testing
import numpy as np
from seismic_bedload.utils import log_raised_cosine_pdf
import matplotlib.pyplot as plt
from seismic_bedload import MultimodeModel, SeismicParams, SedimentParams

from scipy.stats import lognorm

def test_multimode_inversion():
    seismic_params = SeismicParams()
    seismic_params.vc0 = 250
    seismic_params.zeta = 0.089

    sediment_params = SedimentParams()
    sediment_params.rho_s = 2650

    PSD_observe = np.loadtxt("data/pinos/PSD.txt")
    idx = np.arange(48, (48+317))
    PSD_obs = PSD_observe[idx]
    
    H = np.loadtxt("data/pinos/flowdepth.txt")
    H = H/100

    f = np.linspace(30, 80, 10)
    D = np.asarray(np.linspace(0.0001,0.07,100))
    sigma = 0.85
    mu = 0.009
    s = sigma/np.sqrt(1/3-2/np.pi**2)
    pD = log_raised_cosine_pdf(D, mu, s)/D
    
    tau_c50 = 0.045
    D50 = 0.005
    W = 10
    theta = np.tan(0.7*np.pi/180)
    r0 = 17
    qb = 1
    ks = 0.07

    t = 0.05
    t = np.linspace(0.01, 0.5, 100)
    mu = np.log(0.08)
    sigma = 1.0
    lower, upper = 0.01, 0.5
    dist = lognorm(sigma, scale=np.exp(mu))

    def truncated_pdf(x):
        Z = dist.cdf(upper) - dist.cdf(lower)  # normalization constant
        return np.where((x >= lower) & (x <= upper),
                        dist.pdf(x) / Z,
                        0.0)
    tD = truncated_pdf(t)

    model = MultimodeModel(seismic_params=seismic_params, sediment_params=sediment_params)

    bedload = model.inverse_bedload(PSD_obs, f, D, H, W, theta, r0, qb, t, ks, 
                                    D50 = D50, tau_c50 = tau_c50, pdf_D= pD, pdf_t=tD, clip_tau_c=True)

    plt.plot(np.arange(235), bedload[5:240]*2700, linestyle='-.')
    plt.show()


if __name__ == "__main__":
    test_multimode_inversion()

    
