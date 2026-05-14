import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.signal import spectrogram as sp_spectrogram

from seismic_bedload.models import SaltationModel
from seismic_bedload.turbulence import TurbulenceModel
from seismic_bedload import SedimentParams, SeismicParams

def test():
    x    = np.loadtxt("/Users/locluong/Desktop/PhD/Codes_Zenodo_JGR/Seismic_modeling_bedload/data/seismic-data/data.txt")
    flow = np.loadtxt("/Users/locluong/Desktop/PhD/Codes_Zenodo_JGR/Seismic_modeling_bedload/data/seismic-data/flow_depth.txt")
    time = np.arange(1, len(flow) + 1)

    #spectrogram parameters ---
    fs       = 1000
    window   = 2**14
    noverlap = 2**14 // 2
    nfft     = 2**14

    #site / grain parameters ---
    f_max = 250
    f     = np.linspace(0.001, f_max, 1000)
    W     = 10       # river width [m]
    H     = 0.5      # flow depth [m]
    # theta = 0.7      # slope [degrees]
    theta = np.tan(0.7*np.pi/180)
    r0    = 17       # source-receiver distance [m]
    D50   = 0.009    # median grain size [m]
    Dstd  = 0.85     # log-space std

    # grain size array (raised-cosine support)
    s       = Dstd / np.sqrt(1/3 - 2/np.pi**2)
    lim_min = np.exp(-s + np.log(D50))
    lim_max = np.exp( s + np.log(D50))
    D       = np.linspace(lim_min, lim_max, 100)

    # MATLAB varargin: vc0=250, epsilon=0.089, Q0=20, eta=0 ---
    seismic_params = SeismicParams(v0=2206, z0=1000, f0=1, a=0.272, Q0=20, eta=0)
    seismic_params.vc0   = 250
    seismic_params.zeta  = 0.089

    # forward models
    turb_model = TurbulenceModel(seismic_params=seismic_params)
    salt_model = SaltationModel(seismic_params=seismic_params)

    PSD_tur     = turb_model.forward_psd(f, D, W, H, theta, r0, D50=D50, Dstd=Dstd)
    PSD_bed     = salt_model.forward_psd(f, D50, H, W, theta, r0, qb=0.005,
                                          tau_c=0.045, D50=D50)

    PSD_turbulence = 10 * np.log10(PSD_tur)
    PSD_bedload    = 10 * np.log10(PSD_bed)

    # spetrogram of observed seismic data reproduced from MATLAB code
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    freqs, t_seg, Sxx = sp_spectrogram(x, fs=fs, nperseg=window,
                                        noverlap=noverlap, nfft=nfft)
    Sxx_dB = 10 * np.log10(Sxx + 1e-30)

    im = ax1.pcolormesh(t_seg, freqs, Sxx_dB, shading="gouraud",
                        cmap="jet", vmin=-160, vmax=-110)
    ax1.set_ylim(0, 200)
    ax1.set_yticks([0, 30, 80, 120, 200])
    ax1.set_xlabel("")
    ax1.tick_params(direction="out", length=6)
    ax1.set_ylabel("Frequency (Hz)", fontsize=20)

    cbar = fig1.colorbar(im, ax=ax1, orientation="horizontal", pad=0.02)
    cbar.set_label(r"Power [dB rel. (m/s)$^2$/Hz]", fontsize=16)
    ax1.text(0.02, 0.92, "a)", transform=ax1.transAxes, fontsize=24)

    ax2.plot(time / 60, flow, "b-", linewidth=3)
    ax2.set_ylabel("Water depth (cm)", fontsize=20)
    ax2.set_xlabel("Time since flood started (hour)", fontsize=20)
    ax2.tick_params(direction="out")

    fig1.tight_layout()

    #  comparison between turbulence and bedload models
    fig2, ax = plt.subplots(figsize=(12, 9))

    ax.plot(f, PSD_turbulence, "b-", linewidth=4, label="Turbulence")
    ax.plot(f, PSD_bedload,    "r-", linewidth=4, label="Bedload")

    ax.set_xlim(0, f_max)
    ax.set_ylim(-180, -130)
    ax.set_xlabel("Frequency (Hz)", fontsize=20)
    ax.set_ylabel(r"Power spectral density (dB rel. (m/s)$^2$/Hz)", fontsize=20)
    ax.legend(fontsize=20)
    ax.tick_params(direction="out")
    ax.set_box_aspect(None)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.text(0.04, 0.96, "b)", transform=ax.transAxes, fontsize=24,
            va="top")

    fig2.tight_layout()
    plt.show()

if __name__ == "__main__":
    test()
