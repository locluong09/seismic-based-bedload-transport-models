import numpy as np
import matplotlib.pyplot as plt
from seismic_bedload import SaltationModel, SeismicParams
from seismic_bedload.utils import log_raised_cosine_pdf

seismic_params = SeismicParams()
seismic_params.vc0  = 250.0
seismic_params.zeta = 0.089

W=10; theta=np.tan(0.7*np.pi/180); r0=17; D50=0.005; tau_c50=0.045
D  = np.linspace(0.0001, 0.07, 100)
s  = 0.85/np.sqrt(1/3-2/np.pi**2)
pD = log_raised_cosine_pdf(D, 0.009, s)/D
f  = np.linspace(30, 80, 10)

model = SaltationModel(seismic_params=seismic_params)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# vary H, fix qb
for H in [0.1, 0.3, 0.5, 0.8, 1.2]:
    psd = model.forward_psd(f, D, H, W, theta, r0, qb=0.001,
                            D50=D50, tau_c50=tau_c50, pdf=pD)
    axes[0].plot(f, 10*np.log10(psd), label=f"H={H}m")
axes[0].set_title("Vary H (qb fixed)"); axes[0].legend()
axes[0].set_xlabel("Frequency (Hz)"); axes[0].set_ylabel("dB")

# vary qb, fix H
for qb in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
    psd = model.forward_psd(f, D, 0.5, W, theta, r0, qb=qb,
                            D50=D50, tau_c50=tau_c50, pdf=pD)
    axes[1].plot(f, 10*np.log10(psd), label=f"qb={qb:.0e}")
axes[1].set_title("Vary qb (H fixed)"); axes[1].legend()
axes[1].set_xlabel("Frequency (Hz)")

plt.tight_layout(); plt.show()