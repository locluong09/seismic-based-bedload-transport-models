"""
Mirrors the R Dietze script (eseis::fmi_*) using the Python seismic_bedload
package, so results can be compared side-by-side.

Inputs  (same files as the R script):
  spectrogram_nahal_eshtemoa_2016-02-22.txt  — tab-separated, rows=freq, cols=time
  data_nahal_eshtemoa_flood_2016-02-22.txt   — time / h_w / q_s_median
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from seismic_bedload import (
    SaltationModel, TurbulenceModel,
    SeismicParams,
    ParameterRange, build_joint_catalogue, lut_inversion,
)

# -----------------------------------------------------------------------
# 0. Paths  — edit these to match your local layout
# -----------------------------------------------------------------------
BASE = ("/Users/locluong/Documents/FluvialSedimentTransport/"
        "dietze/supplementary_material_revised/")

PSD_FILE  = BASE + "spectrogram_nahal_eshtemoa_2016-02-22.txt"
DATA_FILE = BASE + "data_nahal_eshtemoa_flood_2016-02-22.txt"

# -----------------------------------------------------------------------
# 1. Load observed PSD  (rows = frequency, columns = time)
#    Values are LINEAR power (same as R reads them)
# -----------------------------------------------------------------------
print("Loading spectrogram …")
psd_raw = pd.read_csv(PSD_FILE, sep="\t", header=0, index_col=0)

# Row index = frequency [Hz], column names encode POSIX timestamps
psd_f   = np.array(psd_raw.index, dtype=float)
psd_t   = np.array([float(c) for c in psd_raw.columns])    # POSIX seconds

# R model parameters  (kept identical to the R script)
par = dict(
    d_s  = 0.01,    # grain diameter [m]
    s_s  = 1.35,    # grain shape parameter
    r_s  = 2650,    # sediment density [kg/m³]
    w_w  = 5.0,     # channel width [m]
    a_w  = 0.0075,  # channel slope
    f    = (10, 70),
    r_0  = 5.5,     # source-receiver distance [m]
    f_0  = 1.0,     # coupling freq [Hz]
    q_0  = 16.77,   # coupling quality factor
    v_0  = 859.0,   # Rayleigh wave velocity [m/s]
    p_0  = 0.62,    # scattering coefficient exponent
    e_0  = 0.07,    # scattering coefficient
    n_0  = (0.5, 0.8),
    res  = 100,
)

F_MIN, F_MAX = par["f"]
N_F          = par["res"]        # 100, same as R

# -----------------------------------------------------------------------
# 2. Cut & interpolate PSD to model frequency grid  (mirrors R)
# -----------------------------------------------------------------------
i_cut      = (psd_f >= F_MIN) & (psd_f <= F_MAX)
psd_cut    = psd_raw.values[i_cut, :]      # shape: (n_freq_cut, n_time)
psd_f_cut  = psd_f[i_cut]

f_inv = np.linspace(F_MIN, F_MAX, N_F)    # same as R psd_f_agg

# Interpolate each time column to the model grid  (R uses spline; scipy cubic ≈)
print(f"Interpolating PSD to {N_F} freq bins …")
n_t = psd_cut.shape[1]
psd_obs = np.zeros((n_t, N_F))
for j in range(n_t):
    col = psd_cut[:, j]
    # Use linear if monotonic issues; cubic matches R's spline() better
    psd_obs[j] = np.interp(f_inv, psd_f_cut, col)   # linear fallback
    # For cubic: interp1d(psd_f_cut, col, kind="cubic")(f_inv)

# psd_obs shape: (n_time, N_F)  — LINEAR power, same as R

# -----------------------------------------------------------------------
# 3. Load empirical flood data  (for comparison panels)
# -----------------------------------------------------------------------
print("Loading flood data …")
flood = pd.read_csv(DATA_FILE, sep="\t", header=0)
flood["time"] = pd.to_datetime(flood["time"], utc=True)

# R uses first 300 rows
flood_cut = flood.iloc[:300].copy()
t_emp = flood_cut["time"].values
h_emp = flood_cut["h_w"].values        # flow depth [m]
q_emp = flood_cut["q_s_median"].values # bedload flux [m²/s] (check units)


# seismic_params.vc0  = par["v_0"]   # 859 m/s via setter

seismic_params = SeismicParams(
    f0 = par["f_0"],
    Q0 = par["q_0"],
)
seismic_params.vc0  = par["v_0"]   # 859 m/s  — bypasses v0/z0/a formula
seismic_params.zeta = par["p_0"]   # 0.62     — bypasses a formula


# If SeismicParams has p0, n0 etc., add them here:
# seismic_params.p0   = par["p_0"]
# seismic_params.n0_a = par["n_0"][0]
# seismic_params.n0_b = par["n_0"][1]

# -----------------------------------------------------------------------
# Monte Carlo reference catalogue  (mirrors R ref_pars_a + ref_pars_b)
# -----------------------------------------------------------------------
# R:  ref_pars_a  — q_s  [0, 15/2650], h_w  [0.015, 1.20], n=5000
#     ref_pars_b  — q_s = 0 (no bedload), h_w [0.015, 1.20], n=1000
# Python: approximate with a combined 6000-sample catalogue;
#         include qb=0 by lowering the lower bound to near-zero.

parameter_ranges = [
    ParameterRange("H",     low=0.015, high=1.20,        log=False),
    ParameterRange("qb",    low=1e-8,  high=15/2650,     log=True),   # ~q_s range
    ParameterRange("W",     fixed=par["w_w"]),
    ParameterRange("theta", fixed=par["a_w"]),
    ParameterRange("r0",    fixed=par["r_0"]),
    ParameterRange("D50",   fixed=par["d_s"]),
    ParameterRange("Dstd",  fixed=par["s_s"]),
]

# Grain-size array for TurbulenceModel (log-space around D50)
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


N_CAT = 6000   # 5000 + 1000 to mirror R
print(f"Building reference catalogue (n={N_CAT}) …")
catalogue = build_joint_catalogue(
    f                     = f_inv,
    n                     = N_CAT,
    turbulence_model_factory = turbulence_factory,
    bedload_model_factory    = bedload_factory,
    turbulence_fwd_kwargs    = kw_turbulence,
    bedload_fwd_kwargs       = kw_bedload,
    parameter_ranges         = parameter_ranges,
    seed                     = 42,
    n_workers                = 1,
    verbose                  = True,
)
print(f"Catalogue spectra shape: {catalogue.spectra.shape}")   # (N_CAT, N_F)

# Inversion
# NOTE: check whether lut_inversion expects LINEAR or LOG (dB) psd_obs.
# The catalogue is built in whatever units model_psd() returns.
# If the model returns linear power and psd_obs is also linear → consistent.
# Uncomment the dB conversion below if your lut_inversion works in log space.
# I need to check this against the R code to be sure, but I think R's fmi_* functions

# psd_obs_db = 10 * np.log10(psd_obs + 1e-40)   # only if needed
# result = lut_inversion(catalogue, psd_obs_db)

print("Running inversion …")
result = lut_inversion(catalogue, psd_obs)   # both linear

H_est  = result.parameters["H"]     # [m]
qb_est = result.parameters["qb"]    # [m²/s]
rmse   = result.rmse_min            # per-time-step RMSE

t_model_min = (psd_t - psd_t[0]) / 60.0
t_obs_min = np.arange(len(h_emp))

smooth = lambda x, k: np.convolve(x, np.ones(k)/k, mode="same")   # running mean

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.plot(rmse, lw=1, color="gray")
ax.set_title("Inversion RMSE")
ax.set_xlabel("Time step")
ax.set_ylabel("RMSE")

# Panel 2 — flow depth  (R: runmean with k=18)
ax = axes[1]
k = 18
ax.plot(t_model_min, smooth(H_est, k),   "k-",  lw=2, label="Python FMI")
ax.plot(t_obs_min,  h_emp,            "r-", lw=1.5, label="Observed")
ax.set_title("Flow depth")
ax.set_xlabel("Time step")
ax.set_ylabel("H (m)")
ax.set_ylim(0, 1.2)
ax.legend()

# Panel 3 — bedload flux (units: kg/m/s, matching R's q_s * 2650)
ax = axes[2]
ax.plot(t_model_min, smooth(qb_est * par["r_s"], k), "k-", lw=2, label="Python FMI")
ax.plot(t_obs_min,   q_emp,              "r-", lw=1.5, label="Observed")
ax.set_title("Bedload flux")
ax.set_xlabel("Time step")
ax.set_ylabel(r"$q_b \cdot \rho_s$ (kg m$^{-1}$ s$^{-1}$)")
ax.set_ylim(0, 5)
ax.legend()

fig.tight_layout()
fig.savefig("fmi_eshtemoa_python.png", dpi=200)
print("Figure saved to fmi_eshtemoa_python.png")