"""
Test for MAP inversion of seismic data to infer flow depth and bedload flux.
I don't think this work yet, still need to be tested and debugged.
"""

from seismic_bedload.models import SaltationModel, SeismicParams, SedimentParams, MultimodeModel
import numpy as np
from seismic_bedload.MAP_inversion import PriorConfig, BandConfig, MAPInversion, ModelType
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from seismic_bedload.utils import log_raised_cosine_pdf

f     = np.linspace(1, 100, 200)    ## frequency array
data= np.loadtxt("/Users/locluong/Desktop/PhD/SeismicDataProcessing/2021_SeismicData/2021-07-06_GBB_1m_PSD_1_100.txt")
f = data[:,0]
D = np.asarray(np.linspace(0.0001,0.07,100))
sigma = 0.85
mu = 0.009
s = sigma/np.sqrt(1/3-2/np.pi**2)
pD = log_raised_cosine_pdf(D, mu, s)/D

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
model = SaltationModel(
            sediment_params = SedimentParams(rho_s=2700),
            seismic_params  = SeismicParams(v0=2206, Q0=20)
        )

model = MultimodeModel(
            sediment_params = SedimentParams(rho_s=2700),
            seismic_params  = SeismicParams(v0=250, Q0=20)
        )

## prior
prior = PriorConfig(
    h_mean   = 0.30,    ## Manning estimate
    h_sigma  = 0.10,    ## uncertainty in Manning
    qb_mean  = 0.005,   ## rough MPM estimate
    qb_sigma = 0.8,     ## log-space — factor ~2 uncertainty
    h_min    = 0.01,
    h_max    = 1.5,
    qb_min   = 1e-6,
    qb_max   = 5.0
)

## -band configuration
bands = BandConfig(
    low  = (1.0,  20.0),
    mid  = (20.0, 50.0),
    high = (50.0, 100.0),
    w_low  = 1.5,       ## trust turbulence band for H
    w_high = 1.5        ## trust bedload band for qb
)

## MAP Inversion setup
inverter = MAPInversion(
    model      = model,
    model_type = ModelType.MULTIMODE,
    f          = f,
    D          = 0.05,      ## D50 in meters
    W          = 5.0,       ## channel width
    theta      = 0.02,      ## channel slope
    r0         = 20.0,     ## source-receiver distance
    bands      = bands,
    prior      = prior,
    sigma_f    = 0.15,       ## noise level in log10 PSD space
    t          = t,
    pdf_t      = tD,
)

## PSD_obs_matrix : [n_freq x 60] in dB
def test_map_inversion():
    data= np.loadtxt("/Users/locluong/Desktop/PhD/SeismicDataProcessing/2021_SeismicData/2021-07-06_GBB_1m_PSD_1_100.txt")
    PSD_obs_matrix = data[:,50:55]
    results = inverter.invert_timeseries(PSD_obs_matrix, n_restarts=3)

## results contains MAP estimates and uncertainties for H and qb at each time step
    H_map  = results['H_map']    ## [60] flow depth profile
    qb_map = results['qb_map']   ## [60] bedload flux profile
    H_std  = results['H_std']    ## [60] uncertainty on H
    qb_std = results['qb_std']   ## [60] uncertainty on qb

    plt.plot(H_map, label='MAP Estimate')
    plt.show()

if __name__ == "__main__":
    test_map_inversion()

