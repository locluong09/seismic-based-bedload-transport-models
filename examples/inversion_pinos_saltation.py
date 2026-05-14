"""
Monte Carlo FMI inversion — real data (joint turbulence + bedload)
------------------------------------------------------------------
Inputs:
  data.txt       : raw seismic waveform (fs = 1000 Hz)
  flow_depth.txt : flow depth time series (one value per minute)

Outputs:
  H_est, qb_est  : inverted flow depth and bedload flux (one per minute)
  fmi_results.npz: saved inversion results
  fmi_results.png: 4-panel summary figure
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram as sp_spectrogram

from seismic_bedload import (
    SaltationModel, TurbulenceModel, SeismicParams,
    ParameterRange, build_joint_catalogue, lut_inversion,
)
from seismic_bedload.utils import log_raised_cosine_pdf

# Site / model parameters
DATA_PATH = "/Users/locluong/Desktop/PhD/Codes_Zenodo_JGR/Seismic_modeling_bedload/data/seismic-data/data.txt"
FLOW_PATH = "/Users/locluong/Desktop/PhD/Codes_Zenodo_JGR/Seismic_modeling_bedload/data/seismic-data/flow_depth.txt"

fs       = 1000          # seismic sampling rate [Hz]
window   = 2**14
noverlap = 2**14 // 2
nfft     = 2**14

F_MIN = 1.0             # inversion frequency band [Hz]
F_MAX = 100.0
N_F   = 60               # number of frequency bins in catalogue

W     = 10.0             # channel width [m]
theta = np.tan(0.7 * np.pi / 180)   # dimensionless slope
r0    = 17.0             # source-receiver distance [m]
D50   = 0.009            # median grain size [m]
Dstd  = 0.85             # log-space std for grain-size PDF

seismic_params = SeismicParams(v0=2206, z0=1000, f0=1, a=0.272, Q0=20, eta=0)
seismic_params.vc0  = 250.0 # At pinos we estimate vco ~ 250 and zeta ~ 0.1
seismic_params.zeta = 0.1

# Grain-size array and log raised-cosine PDF
# Half-width of the raised-cosine support from Dstd
s     = Dstd / np.sqrt(1/3 - 2/np.pi**2)
D_arr = np.linspace(np.exp(np.log(D50) - s),
                    np.exp(np.log(D50) + s), 80)
pD    = log_raised_cosine_pdf(D_arr, D50, s) / D_arr   # normalised PDF

# Build observed PSD time series  (mean per minute)
print("Loading data …")
x    = np.loadtxt(DATA_PATH)
flow = np.loadtxt(FLOW_PATH)          # [cm] one value per minute
n_minutes = len(flow)

freqs, _, Sxx = sp_spectrogram(x, fs=fs, nperseg=window,
                                noverlap=noverlap, nfft=nfft)

f_mask     = (freqs >= F_MIN) & (freqs <= F_MAX)
freqs_band = freqs[f_mask]
Sxx_band   = Sxx[f_mask, :]          # (n_band, n_seg)

dt_seg       = (window - noverlap) / fs
segs_per_min = int(np.round(60.0 / dt_seg))
n_seg_total  = Sxx_band.shape[1]
n_minutes_avail = min(n_minutes, n_seg_total // segs_per_min)

f_inv   = np.linspace(F_MIN, F_MAX, N_F)
psd_obs = np.zeros((n_minutes_avail, N_F))

for i in range(n_minutes_avail):
    i0 = i * segs_per_min
    i1 = min(i0 + segs_per_min, n_seg_total)
    psd_mean_lin = np.mean(Sxx_band[:, i0:i1], axis=1)
    psd_db       = 10.0 * np.log10(psd_mean_lin + 1e-40)
    psd_obs[i]   = np.interp(f_inv, freqs_band, psd_db)

flow_used = flow[:n_minutes_avail]
time_min  = np.arange(n_minutes_avail)

print(f"  {n_minutes_avail} minutes of PSD ready, "
      f"{N_F} freq bins ({F_MIN}–{F_MAX} Hz)")

# Build Monte Carlo reference catalogue (joint turbulence + bedload)
parameter_ranges = [
    ParameterRange("H",     low=0.01,  high=2.00,  log=True),
    ParameterRange("qb",    low=1e-6,  high=10,    log=True),
    ParameterRange("W",     fixed=W),
    ParameterRange("theta", fixed=theta),
    ParameterRange("r0",    fixed=r0),
    ParameterRange("D50",   fixed=D50),
    ParameterRange("Dstd",  fixed=Dstd),
]


def turbulence_factory(p):
    return TurbulenceModel(seismic_params=seismic_params)


def bedload_factory(p):
    return SaltationModel(seismic_params=seismic_params)


def kw_turbulence(p):
    return dict(
        D=D_arr, W=p["W"], H=p["H"],
        theta=p["theta"], r0=p["r0"],
        D50=p["D50"], Dstd=p["Dstd"],
        pdf=pD,
    )


def kw_bedload(p):
    return dict(
        D=D_arr, H=p["H"], W=p["W"],
        theta=p["theta"], r0=p["r0"],
        qb=p["qb"], D50=p["D50"],
        pdf=pD,
    )


print("Building reference catalogue (n=2000) …")
catalogue = build_joint_catalogue(
    f=f_inv,
    n=2000,
    turbulence_model_factory=turbulence_factory,
    bedload_model_factory=bedload_factory,
    turbulence_fwd_kwargs=kw_turbulence,
    bedload_fwd_kwargs=kw_bedload,
    parameter_ranges=parameter_ranges,
    seed=42,
    n_workers=1,
    verbose=True,
)

# Inversion
print("Running inversion …")
result = lut_inversion(catalogue, psd_obs)

H_est  = result.parameters["H"]
qb_est = result.parameters["qb"]
rmse   = result.rmse_min


def running_mean(x, n):
    kernel = np.ones(n) / n
    return np.convolve(x, kernel, mode='same')

#
# Plot
fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

ax = axes[0]
im = ax.pcolormesh(time_min, f_inv, psd_obs.T,
                   shading="auto", cmap="jet", vmin=-170, vmax=-120)
ax.set_ylabel("Frequency (Hz)", fontsize=13)
cbar = fig.colorbar(im, ax=ax, pad=0.01)
cbar.set_label("dB", fontsize=11)
ax.text(0.01, 0.92, "a) Observed PSD", transform=ax.transAxes,
        fontsize=13, color="w", va="top")

ax = axes[1]
ax.plot(time_min, flow_used / 100, "b-", lw=2, label="Measured (gauge)")
ax.plot(time_min, running_mean(H_est, 18),           "r--", lw=2, label="Estimated (FMI)")
ax.set_ylabel("Flow depth (m)", fontsize=13)
ax.legend(fontsize=11)
ax.text(0.01, 0.92, "b) Flow depth", transform=ax.transAxes, fontsize=13, va="top")

ax = axes[2]
ax.semilogy(time_min, qb_est * 2650, "k-", lw=2)
ax.set_ylabel(r"Bedload flux $q_b \cdot \rho_s$ (kg m$^{-1}$ s$^{-1}$)", fontsize=13)
ax.text(0.01, 0.92, "c) Estimated bedload flux", transform=ax.transAxes,
        fontsize=13, va="top")

ax = axes[3]
ax.plot(time_min, rmse, "gray", lw=1.5)
ax.set_ylabel("Spectral RMSE (dB)", fontsize=13)
ax.set_xlabel("Time (minutes since flood start)", fontsize=13)
ax.text(0.01, 0.92, "d) Inversion fit quality", transform=ax.transAxes,
        fontsize=13, va="top")

for ax in axes:
    ax.tick_params(direction="out")

fig.tight_layout()
fig.savefig("inversion_pinos_saltation.png", dpi=150)
print("Figure saved to inversion_pinos_saltation.png")
plt.show()