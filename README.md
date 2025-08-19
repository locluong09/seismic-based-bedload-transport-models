# Seismic-based Bedload Transport Models

Implementation of seismic-based bedload transport models, including the **saltation-mode model** (Tsai et al., 2012) and **multi-mode model** (Luong et al., 2024). This package is designed to estimate bedload flux using seismic methods.

## Features

- Saltation-mode bedload transport model
- Multi-mode bedload transport model (in testing)
- Forward modeling for PSD, and backward inversion for bedload flux
- Support for grain size distributions (pD) in forward and inverse modeling

## Installation

You can install the package locally after building:

```bash
pip install ./dist/seismic_bedload-0.1.0-py3-none-any.whl
```

## Usage

```python
from seismic_bedload import SaltationModel

f = np.linspace(0.001, 20, 100)
D = 0.3  
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

# Forward modeling of PSD
psd = model.forward_psd(f, D, H, W, theta, r0, qb, D50 = D50, pdf = pD)

# Inverting  bedload flux
PSD_obs = np.loadtxt("data/pinos/PSD.txt")
H = np.loadtxt("data/pinos/flowdepth.txt")
bedload_flux = model.inverse_bedload(PSD_obs, f, D, H, W, theta, r0, qb, D50=D50, tau_c50=tau_c50, pdf = pD)
```

## TODO

- ~~Allows grain size distribution (pD) to be called from forward and inverse method~~
- Implementing empirical models
- Multi-mode model testing
- Particle velocity update

## References

Tsai, V.C., Minchew, B., Lamb, M.P. and Ampuero, J.P., 2012. A physical model for seismic noise generation from sediment transport in rivers. Geophysical Research Letters, 39(2).

Luong, L., Cadol, D., Bilek, S., McLaughlin, J.M., Laronne, J.B. and Turowski, J.M., 2024. Seismic modeling of bedload transport in a gravel‐bed alluvial channel. Journal of Geophysical Research: Earth Surface, 129(9), p.e2024JF007761.

## License

This project is licensed under the MIT License.