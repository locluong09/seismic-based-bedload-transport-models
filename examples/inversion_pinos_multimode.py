"""
MonteCarlo based inversion using multimode model, 
applied to seismic data from Arroyo de Los Pinos.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram as sp_spectrogram
from scipy.stats import lognorm

from seismic_bedload import (
    MultimodeModel, TurbulenceModel, SeismicParams,
    ParameterRange, build_joint_catalogue, lut_inversion,
)
from seismic_bedload.utils import log_raised_cosine_pdf


DATA_PATH = "/Users/locluong/Desktop/PhD/Codes_Zenodo_JGR/Seismic_modeling_bedload/data/seismic-data/data.txt"
FLOW_PATH = "/Users/locluong/Desktop/PhD/Codes_Zenodo_JGR/Seismic_modeling_bedload/data/seismic-data/flow_depth.txt"

fs       = 1000           # seismic sampling rate [Hz]
window   = 2**14         
noverlap = window // 2
nfft     = 2**14

F_MIN = 30.0              # inversion frequency band [Hz]
F_MAX = 80.0
N_F   = 60               # catalogue frequency bins

# model parameters — Arroyo de Los Pinos

W     = 10.0                          # channel width [m]
theta = np.tan(0.7 * np.pi / 180)    # bed slope
r0    = 17.0                         # source-receiver distance [m]
D50   = 0.009                        # median grain size [m]
Dstd  = 0.85                         # log-space std of grain-size PDF

seismic_params = SeismicParams(v0=2206, z0=1000, f0=1, a=0.272, Q0=20, eta=0)
seismic_params.vc0  = 250.0          # phase velocity at f0 [m/s]
seismic_params.zeta = 0.089          # velocity dispersion exponent

# Grain-size array + log raised-cosine PDF
s     = Dstd / np.sqrt(1/3 - 2/np.pi**2)
D_arr = np.linspace(np.exp(np.log(D50) - s),
                    np.exp(np.log(D50) + s), 80)
pD    = log_raised_cosine_pdf(D_arr, D50, s) / D_arr   # normalised for log-space

# hop-time array + truncated log-normal PDF
T_MIN, T_MAX = 0.01, 0.5
t_arr = np.linspace(T_MIN, T_MAX, 70)

mu_t    = np.log(0.08)     # mode of log-normal ~0.08 s
sigma_t = 1.0
_dist   = lognorm(sigma_t, scale=np.exp(mu_t))
_Z      = _dist.cdf(T_MAX) - _dist.cdf(T_MIN)
pt      = np.where(
    (t_arr >= T_MIN) & (t_arr <= T_MAX),
    _dist.pdf(t_arr) / _Z,
    0.0,
)

# Build observed PSD time series  (mean per minute)
print("Loading seismic data …")
x    = np.loadtxt(DATA_PATH)
flow = np.loadtxt(FLOW_PATH)          # [cm]
n_minutes = len(flow)

freqs, _, Sxx = sp_spectrogram(x, fs=fs, nperseg=window,
                                noverlap=noverlap, nfft=nfft)

f_mask     = (freqs >= F_MIN) & (freqs <= F_MAX)
freqs_band = freqs[f_mask]
Sxx_band   = Sxx[f_mask, :]

dt_seg       = (window - noverlap) / fs
segs_per_min = max(1, int(np.round(60.0 / dt_seg)))
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
print(f"  PSD range: {psd_obs.min():.1f} to {psd_obs.max():.1f} dB")

# Monte Carlo reference catalogue — joint turbulence + MultimodeModel
parameter_ranges = [
    ParameterRange("H",     low=0.01,  high=2.0, log=True),
    ParameterRange("qb",    low=1e-6,  high=5.0,  log=True),
    ParameterRange("W",     fixed=W),
    ParameterRange("theta", fixed=theta),
    ParameterRange("r0",    fixed=r0),
    ParameterRange("D50",   fixed=D50),
    ParameterRange("Dstd",  fixed=Dstd),
]


def turbulence_factory(p):
    return TurbulenceModel(seismic_params=seismic_params)


def multimode_factory(p):
    return MultimodeModel(seismic_params=seismic_params)


def kw_turbulence(p):
    return dict(
        D=D_arr, W=p["W"], H=p["H"],
        theta=p["theta"], r0=p["r0"],
        D50=p["D50"], Dstd=p["Dstd"],
        pdf=pD,
    )


def kw_multimode(p):
    # MultimodeModel.forward_psd needs t, pdf_D, and pdf_t
    # in addition to the standard SaltationModel arguments
    return dict(
        D=D_arr,
        H=p["H"],
        W=p["W"],
        theta=p["theta"],
        r0=p["r0"],
        qb=p["qb"],
        t=t_arr,
        D50=p["D50"],
        pdf_D=pD,
        pdf_t=pt,
    )


print("Building reference catalogue (n=2000, joint turbulence + MultimodeModel) …")
catalogue = build_joint_catalogue(
    f=f_inv,
    n=1000,
    turbulence_model_factory=turbulence_factory,
    bedload_model_factory=multimode_factory,
    turbulence_fwd_kwargs=kw_turbulence,
    bedload_fwd_kwargs=kw_multimode,
    parameter_ranges=parameter_ranges,
    seed=42,
    n_workers=1,
    verbose=True,
)

print(f"Catalogue spectra range: "
      f"{catalogue.spectra.min():.1f} to {catalogue.spectra.max():.1f} dB")

# Montecarlo or  LUT inversion
print("Running LUT inversion …")
result = lut_inversion(catalogue, psd_obs)

H_est  = result.parameters["H"]
qb_est = result.parameters["qb"]
rmse   = result.rmse_min

print(f"H_est  : {np.nanmin(H_est):.3f} – {np.nanmax(H_est):.3f} m")
print(f"qb_est : {np.nanmin(qb_est):.2e} – {np.nanmax(qb_est):.2e} m²/s")
print(f"RMSE   : {np.nanmin(rmse):.2f} – {np.nanmax(rmse):.2f} dB")



def running_mean(x, n):
    return np.convolve(x, np.ones(n) / n, mode='same')


fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

# a) Observed PSD spectrogram
ax = axes[0]
im = ax.pcolormesh(time_min, f_inv, psd_obs.T, shading="auto", cmap="jet")
ax.set_ylabel("Frequency (Hz)", fontsize=13)
fig.colorbar(im, ax=ax, pad=0.01).set_label("dB", fontsize=11)
ax.text(0.01, 0.92, "a) Observed PSD", transform=ax.transAxes,
        fontsize=13, color="w", va="top")

# b) Flow depth
ax = axes[1]
ax.plot(time_min, flow_used / 100, "b-", lw=2, label="Measured (gauge)")
ax.plot(time_min, running_mean(H_est, 5), "r--", lw=2, label="Estimated (FMI)")
ax.set_ylabel("Flow depth (m)", fontsize=13)
ax.legend(fontsize=11)
ax.text(0.01, 0.92, "b) Flow depth", transform=ax.transAxes,
        fontsize=13, va="top")

# c) Bedload flux
ax = axes[2]
qb_kgms = running_mean(qb_est, 5) * 2700   # kg m⁻¹ s⁻¹
valid = qb_kgms > 0
if valid.any():
    ax.semilogy(time_min[valid], qb_kgms[valid], "k-", lw=2)
ax.set_ylabel(r"$q_b \cdot \rho_s$  (kg m$^{-1}$ s$^{-1}$)", fontsize=13)
ax.text(0.01, 0.92, "c) Estimated bedload flux (MultimodeModel)",
        transform=ax.transAxes, fontsize=13, va="top")

# d) RMSE
ax = axes[3]
ax.plot(time_min, rmse, color="gray", lw=1.5)
ax.set_ylabel("Spectral RMSE (dB)", fontsize=13)
ax.set_xlabel("Time (minutes since flood start)", fontsize=13)
ax.text(0.01, 0.92, "d) Inversion fit quality", transform=ax.transAxes,
        fontsize=13, va="top")

for ax in axes:
    ax.tick_params(direction="out")

fig.suptitle("Arroyo de Los Pinos — MultimodeModel FMI inversion", fontsize=14)
fig.tight_layout()
fig.savefig("inversion_pinos_multimode.png", dpi=150, bbox_inches="tight")
print("Figure saved → inversion_pinos_multimode.png")
plt.show()