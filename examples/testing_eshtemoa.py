"""
test_fmi_eshtemoa.py
Mirrors the R Dietze script (eseis::fmi_*) using the Python seismic_bedload
package, so results can be compared side-by-side.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seismic_bedload import (
    SaltationModel, TurbulenceModel,
    SeismicParams,
    ParameterRange, build_joint_catalogue, lut_inversion,
)

BASE = ("/Users/locluong/Documents/FluvialSedimentTransport/"
        "dietze/supplementary_material_revised/")

PSD_FILE  = BASE + "spectrogram_nahal_eshtemoa_2016-02-22.txt"
DATA_FILE = BASE + "data_nahal_eshtemoa_flood_2016-02-22.txt"


print("Loading spectrogram ...")
psd_raw = pd.read_csv(PSD_FILE, sep="\t", header=0, index_col=0)

psd_f = np.array(psd_raw.index, dtype=float)
psd_t = np.array([float(c) for c in psd_raw.columns])

par = dict(
    d_s  = 0.01,
    s_s  = 1.35,
    r_s  = 2650,
    w_w  = 5.0,
    a_w  = 0.0075,
    f    = (10, 70),
    r_0  = 5.5,
    f_0  = 1.0,
    q_0  = 16.77,
    v_0  = 859.0,
    p_0  = 0.62,
    e_0  = 0.07,
    n_0  = (0.5, 0.8),
    res  = 100,
)

F_MIN, F_MAX = par["f"]
N_F          = par["res"]

i_cut     = (psd_f >= F_MIN) & (psd_f <= F_MAX)
psd_cut   = psd_raw.values[i_cut, :]
psd_f_cut = psd_f[i_cut]

f_inv = np.linspace(F_MIN, F_MAX, N_F)

print(f"Interpolating PSD to {N_F} freq bins ...")
n_t     = psd_cut.shape[1]
psd_obs = np.zeros((n_t, N_F))
for j in range(n_t):
    psd_obs[j] = np.interp(f_inv, psd_f_cut, psd_cut[:, j])

#catalogue is stored in dB — convert observed PSD to dB to match
# psd_obs_db = 10.0 * np.log10(np.maximum(psd_obs, 1e-40))

print("Loading flood data ...")
flood = pd.read_csv(DATA_FILE, sep="\t", header=0)
flood["time"] = pd.to_datetime(flood["time"], utc=True)

flood_cut = flood.iloc[:360].copy()
t_emp = flood_cut["time"].values
h_emp = flood_cut["h_w"].values
q_emp = flood_cut["q_s_median"].values
print(q_emp[:20])

# SeismicParams
#  set vc0 and zeta directly via setters — do NOT pass v_0 as v0
# to the constructor, which feeds a derived formula and gives wrong values.
#   R v_0=859  : vc0  (phase velocity at f0)
#   R p_0=0.62 :zeta (dispersion exponent)
#   R q_0=16.77: Q0

seismic_params = SeismicParams(
    f0 = par["f_0"],
    Q0 = par["q_0"],
)
seismic_params.vc0  = par["v_0"]   # 859 m/s  — bypasses v0/z0/a formula
seismic_params.zeta = par["p_0"]   # 0.62     — bypasses a formula

print(f"SeismicParams: vc0={seismic_params.vc0:.1f}, zeta={seismic_params.zeta:.3f}, Q0={seismic_params.Q0:.2f}")

#  Build Monte Carlo reference catalogue

parameter_ranges = [
    ParameterRange("H",     low=0.015, high=1.20,    log=False),
    ParameterRange("qb",    low=1e-8,  high=15/2650, log=True),
    ParameterRange("W",     fixed=par["w_w"]),
    ParameterRange("theta", fixed=par["a_w"]),
    ParameterRange("r0",    fixed=par["r_0"]),
    ParameterRange("D50",   fixed=par["d_s"]),
    ParameterRange("Dstd",  fixed=par["s_s"]),
]

s_gs  = par["s_s"] / np.sqrt(1/3 - 2/np.pi**2)
D_arr = np.linspace(np.exp(-s_gs + np.log(par["d_s"])),
                    np.exp( s_gs + np.log(par["d_s"])), 80)

def turbulence_factory(p):
    return TurbulenceModel(seismic_params=seismic_params)

def bedload_factory(p):
    return SaltationModel(seismic_params=seismic_params)

def kw_turbulence(p):
    return dict(D=D_arr, W=p["W"], H=p["H"],
                theta=p["theta"], r0=p["r0"],
                D50=p["D50"], Dstd=p["Dstd"])

def kw_bedload(p):
    return dict(D=p["D50"], H=p["H"], W=p["W"],
                theta=p["theta"], r0=p["r0"], qb=p["qb"])

N_CAT = 6000
print(f"Building reference catalogue (n={N_CAT}) ...")
catalogue = build_joint_catalogue(
    f                        = f_inv,
    n                        = N_CAT,
    turbulence_model_factory = turbulence_factory,
    bedload_model_factory    = bedload_factory,
    turbulence_fwd_kwargs    = kw_turbulence,
    bedload_fwd_kwargs       = kw_bedload,
    parameter_ranges         = parameter_ranges,
    seed                     = 42,
    n_workers                = 1,
    verbose                  = True,
)
print(f"Catalogue spectra shape: {catalogue.spectra.shape}")

# Sanity check: dB ranges should overlap
print(f"Catalogue dB range : [{catalogue.spectra.min():.1f}, {catalogue.spectra.max():.1f}]")
print(f"Observed  dB range : [{psd_obs.min():.1f}, {psd_obs.max():.1f}]")

#inversion — both catalogue and data in dB

print("Running inversion ...")
result = lut_inversion(catalogue, psd_obs)

H_est  = result.parameters["H"]
qb_est = result.parameters["qb"]
rmse   = result.rmse_min


# psd_t is in POSIX seconds — convert to minutes from flood start
t_model_min = (psd_t - psd_t[0]) / 60.0

# t_emp is already per-minute, index from 0
t_obs_min = np.arange(len(h_emp))

smooth = lambda x, k: np.convolve(x, np.ones(k)/k, mode="same")
k = 18

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.plot(t_model_min, rmse, lw=1, color="gray")
ax.set_title("Inversion RMSE (dB)")
ax.set_xlabel("Time (min)")
ax.set_ylabel("RMSE")

ax = axes[1]
ax.plot(t_model_min, smooth(H_est, k), "k-", lw=2, label="Python FMI")
ax.plot(t_obs_min + 40,   h_emp,            "r-", lw=1.5, label="Observed")
ax.set_title("Flow depth")
ax.set_xlabel("Time (min)")
ax.set_ylabel("H (m)")
ax.set_ylim(0, 1.2)
ax.legend()

ax = axes[2]
ax.plot(t_model_min, smooth(qb_est * par["r_s"], k), "k-", lw=2, label="Python FMI")
ax.plot(t_obs_min + 40,   q_emp,              "r-", lw=1.5, label="Observed")
ax.set_title("Bedload flux")
ax.set_xlabel("Time (min)")
ax.set_ylabel(r"$q_b \cdot \rho_s$ (kg m$^{-1}$ s$^{-1}$)")
ax.set_ylim(0, 20)
ax.legend()

fig.tight_layout()
fig.savefig("fmi_eshtemoa_python1.png", dpi=200)
print("Figure saved to fmi_eshtemoa_python1.png")