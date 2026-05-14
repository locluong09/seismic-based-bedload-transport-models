"""
FMI inversion — turbulence only (from pre-computed PSD matrix)
Matches the R script: loads psd_matrix.txt, psd_matrix_freq.txt,
flow_depth_m.txt, interpolates to target frequency grid, builds catalogue,
runs inversion.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from seismic_bedload import TurbulenceModel, SeismicParams
from seismic_bedload import build_reference_catalogue, lut_inversion, ParameterRange
from seismic_bedload.utils import log_raised_cosine_pdf

par = dict(
    d_s = 0.009,
    s_s = 0.85,
    r_s = 2650,
    w_w = 10.0,
    a_w = 0.0122,          # slope in radians
    f   = (30.0, 80.0),
    r_0 = 17.0,
    f_0 = 1.0,
    q_0 = 20.0,
    v_0 = 250.0,
    p_0 = 0.272,
    e_0 = 0.089,
    n_0 = (0.5, 0.8),
    res = 60,
)

seismic_params      = SeismicParams(f0=par["f_0"], Q0=par["q_0"],
                                    eta=par["e_0"])
seismic_params.vc0 = par["v_0"]
seismic_params.zeta = 0.089
seismic_params.N11  = par["n_0"][0]
seismic_params.N12  = par["n_0"][1]

# Load pre-computed PSD matrix
#     psd_mat  : (n_minutes, n_freq_orig)  — rows = minutes, cols = freq bins
#     psd_freq : (n_freq_orig,)
#     h_gauge  : (n_minutes,)  in metres
print("Loading PSD matrix …")
psd_mat  = np.loadtxt("data/test_data/psd_matrix.txt")        # (n_minutes, n_freq_orig)
psd_freq = np.loadtxt("data/test_data/psd_matrix_freq.txt")   # 1-D or 2-D; flatten below
h_gauge  = np.loadtxt("data/test_data/flow_depth_m.txt")

psd_freq = psd_freq.ravel()
h_gauge  = h_gauge.ravel()
n_minutes = psd_mat.shape[0]

# Target frequency grid (matches R: seq(f[1], f[2], length.out=res))
f_inv = np.linspace(par["f"][0], par["f"][1], par["res"])

# Interpolate each minute's spectrum to f_inv  (R uses spline; cubic here)
psd_obs = np.zeros((n_minutes, par["res"]))
for i in range(n_minutes):
    psd_obs[i] = np.interp(f_inv, psd_freq, psd_mat[i])
    # for cubic spline matching R's spline(), swap the line above with:
    # psd_obs[i] = make_interp_spline(psd_freq, psd_mat[i], k=3)(f_inv)

time_min = np.arange(n_minutes)
print(f"  PSD: {n_minutes} min × {par['res']} bins ({par['f'][0]:.0f}–{par['f'][1]:.0f} Hz)")

#Grain-size array and PDF
s     = par["s_s"] / np.sqrt(1/3 - 2/np.pi**2)
D50   = par["d_s"]
D_arr = np.linspace(np.exp(np.log(D50) - s),
                    np.exp(np.log(D50) + s), 80)
pD    = log_raised_cosine_pdf(D_arr, D50, s) / D_arr

# Reference catalogue — turbulence only, H range matches R h_w
parameter_ranges = [
    ParameterRange("H",     low=0.015, high=1.20, log=True),  # matches R h_w
    ParameterRange("W",     fixed=par["w_w"]),
    ParameterRange("theta", fixed=par["a_w"]),
    ParameterRange("r0",    fixed=par["r_0"]),
    ParameterRange("D50",   fixed=D50),
    ParameterRange("Dstd",  fixed=par["s_s"]),
]

def turbulence_factory(p):
    return TurbulenceModel(seismic_params=seismic_params)

def kw_turbulence(p):
    return dict(D=D_arr, W=p["W"], H=p["H"],
                theta=p["theta"], r0=p["r0"],
                D50=p["D50"], Dstd=p["Dstd"], pdf=pD)

print("Building catalogue (n=3000) …")
catalogue = build_reference_catalogue(
    f=f_inv, n=3000,
    parameter_ranges=parameter_ranges,
    model_factory=turbulence_factory,
    forward_kwargs=kw_turbulence,
    seed=42, n_workers=1, verbose=True,
)

#Inversion
print("Running inversion …")
result = lut_inversion(catalogue, psd_obs)
H_est  = result.parameters["H"]
rmse   = result.rmse_min

# plot
def running_mean(x, n):
    return np.convolve(x, np.ones(n)/n, mode='same')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(time_min, running_mean(H_est, 18), "k-",  lw=2, label="FMI (turbulence)")
ax.plot(time_min, h_gauge,                 "r-",  lw=2, label="Gauge")
ax.set_ylim(0, 1.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Flow depth (m)")
ax.set_title("Flow depth: FMI vs Gauge")
ax.legend(frameon=False)

ax = axes[1]
ax.plot(time_min, h_gauge, "r-", lw=2)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Flow depth (m)")
ax.set_title("Gauge only")

fig.tight_layout()
fig.savefig("pinos_turbulence_py.png", dpi=300)
print("Saved pinos_turbulence_py.png")
plt.show()