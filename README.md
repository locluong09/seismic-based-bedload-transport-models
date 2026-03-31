# Seismic-based Bedload Transport Models

Implementation of seismic-based bedload transport models, including the **saltation-mode model** (Tsai et al., 2012) and **multi-mode model** (Luong et al., 2024). This package is designed to estimate bedload flux using seismic methods.

## Features

- Saltation-mode bedload transport model
- Multi-mode bedload transport model (including hop time distribution)
- Forward modeling for PSD, and backward inversion for bedload flux
- Support for grain size distributions (pD) in forward and inverse modeling

## Installation

You can install the package using either pip or by cloning this repo and install it locally:

```bash
pip install seismic_bedload
```

## Usage
### Saltation model
```python
import numpy as np
import matplotlib.pyplot as plt

from seismic_bedload import SaltationModel
from seismic_bedload import log_raised_cosine_pdf

f = np.linspace(0.001, 20, 100) #frequency window
D = 0.3  # grain size (m)
H = 4.0   # flow depth (m)
W = 50 # river width (m)
theta = np.tan(1.4*np.pi/180)
r0 = 600 # source-receiver distance (m)
qb = 1e-3 # volumetric bedload flux per unit width (m2/s)

model = SaltationModel()
# Forward modeling of PSD
psd = model.forward_psd(f, D, H, W, theta, r0, qb)
# Reproduce Tsai results
plt.plot(f, psd)
plt.show()

# Inverting  bedload flux using Pinos dataset
PSD_obs = np.loadtxt("data/pinos/PSD.txt")
H = np.loadtxt("data/pinos/flowdepth.txt")
# Need to make sure PSD_obs and H have the same length
idx = np.arange(49, (49+317))
PSD_obs = PSD_obs[idx]
H = H/100 # depth to meter

# Data for Pinos
f = np.linspace(30, 80, 10) # frequency (Hz)
D = np.asarray(np.linspace(0.0001,0.07,100))
sigma = 0.85
mu = 0.009
s = sigma/np.sqrt(1/3-2/np.pi**2)
pD = log_raised_cosine_pdf(D, mu, s)/D

tau_c50 = 0.045 # Critical shield for D50 (-)
D50 = 0.005 # mean grain size D50 (m)
W = 10 # river width (m)
theta = np.tan(0.7*np.pi/180) # slope (-)
r0 = 17 # source-receiver distance (m)
qb = 1 # Set qb = 1 for inverse mode m2/s
bedload_flux = model.inverse_bedload(PSD_obs, f, D, H, W, theta, r0, qb = 1, D50=D50, tau_c50=tau_c50, pdf = pD) # m2/s
bedload_flux = bedload_flux * 2700 # to (kg/m/s)
```
### Multimode model
```python
import numpy as np
import matplotlib.pyplot as plt

from seismic_bedload import MultimodeModel
from seismic_bedload import log_raised_cosine_pdf
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
```

## TODO

- ~~Allows grain size distribution (pD) to be called from forward and inverse method~~
- ~~Implementing empirical models~~
- ~~Multi-mode model testing, including the hop time distribution~~
- ~~Update equations for bedload velocity estimation~~

## References

Tsai, V.C., Minchew, B., Lamb, M.P. and Ampuero, J.P., 2012. A physical model for seismic noise generation from sediment transport in rivers. Geophysical Research Letters, 39(2).

Luong, L., Cadol, D., Bilek, S., McLaughlin, J.M., Laronne, J.B. and Turowski, J.M., 2024. Seismic modeling of bedload transport in a gravel‐bed alluvial channel. Journal of Geophysical Research: Earth Surface, 129(9), p.e2024JF007761.

Luong, L., Cadol, D., Turowski, J.M., Bilek, S., McLaughlin, J.M., Stark, K. and Laronne, J.B., 2026. An empirical model combining seismic noise and shear stress to predict bedload flux in a gravel‐bed alluvial channel. Water Resources Research, 62(2), p.e2025WR040371.

## License

This project is licensed under the MIT License.