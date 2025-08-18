# Example usage and testing
import numpy as np
from utils import log_raised_cosine_pdf
import matplotlib.pyplot as plt
from models import MultimodeModel, SeismicParams, SedimentParams

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

    # psd = model.forward_psd(f, D, H, W, theta, r0, qb, pdf = pD)
    bedload = model.inverse_bedload(PSD_obs, f, D, H, W, theta, r0, qb, t, ks, 
                                    D50 = D50, tau_c50 = tau_c50, pdf_D= pD, pdf_t=tD, clip_tau_c=True)

    # res =np.zeros_like(D)
    # PSD_est = np.zeros_like(f)
    # ave = np.zeros_like(H)

    # # this is the hard way to do the inversion for bedload flux (3 nested-loop)
    # # My rough implementation from Matlab
    # for i in range(len(H)):
    #     for j in range(len(f)):
    #         for k in range(len(D)):
    #             res[k] = model.forward_psd(f[j], D[k], H[i], W, theta, r0, qb, D50=D50, tau_c50=tau_c50)
    #         PSD_est[j] = np.trapz(y = res * pD, x = D)
    #     ave[i] = np.median(PSD_est)
    # bed = 10**(PSD_obs/10)/ave

    # diff = bed - bedload
    # mask = ~np.isnan(diff)
    # assert(max(np.abs(diff[mask])) < 1e-6), "Inversion test failed: 2 methods do not match."

    # plt.plot(np.arange(len(bed)), bed*2700, linestyle='-.')
    plt.plot(np.arange(235), bedload[5:240]*2700, linestyle='-.')
    plt.show()


if __name__ == "__main__":
    test_multimode_inversion()

    
