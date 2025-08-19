# Example usage and testing
import numpy as np
from seismic_bedload.utils import log_raised_cosine_pdf
import matplotlib.pyplot as plt
from seismic_bedload import MultimodeModel, SeismicParams, SedimentParams

from scipy.stats import lognorm

def test_eshtemoa():
    seismic_params = SeismicParams()
    seismic_params.vc0 = 3000
    seismic_params.zeta = 0.59
    seismic_params.Q0 = 10

    sediment_params = SedimentParams()
    sediment_params.rho_s = 2650

    PSD_observe = np.loadtxt("data/eshtemoa/psd-data-one-minute.csv", delimiter=',')
    data = np.loadtxt("data/eshtemoa/bedload.csv", delimiter=',')

    psd_20_80 =  PSD_observe[2:-1, 101:400]
    psd_median = np.median(psd_20_80, axis=1)

    PSD_obs = psd_median[35:155]
    
    H = data[:, 1]
    H = H[0:120]

    f = np.linspace(30, 80, 10)
    D = np.asarray(np.linspace(0.0001,0.07,100))
    sigma = 1.35
    mu = 0.012
    s = sigma/np.sqrt(1/3-2/np.pi**2)
    pD = log_raised_cosine_pdf(D, mu, s)/D
    
    tau_c50 = 0.045
    D50 = 0.012
    W = 5
    theta = np.tan(0.43*np.pi/180)
    r0 = 5.5
    qb = 1
    ks = 0.07

    
    mu = np.log(0.08)
    sigma = 1.0
    lower, upper = 0.01, 0.5
    t = np.linspace(lower, upper, 100)
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

    plt.plot(np.arange(len(bedload)), bedload*2700, linestyle='-.')
    plt.show()


if __name__ == "__main__":
    test_eshtemoa()

    
