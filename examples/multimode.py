"""
Test MultimodeModel forward PSD — Arroyo de Los Pinos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

from seismic_bedload import MultimodeModel, SeismicParams
from seismic_bedload.utils import log_raised_cosine_pdf

# Site parameters — Arroyo de Los Pinos
W     = 10.0
theta = np.tan(1.5 * np.pi / 180)
r0    = 17.0
D50   = 0.010          # 20 mm median grain size
Dstd  = 0.85           # log-space std
qb    = 1e-3           # bedload flux [m²/s]
H     = 0.3            # flow depth [m]

seismic_params = SeismicParams(v0=2206, z0=1000, f0=1, a=0.272, Q0=20, eta=0)
seismic_params.vc0  = 300.0
seismic_params.zeta = 0.089

# Frequency array
f = np.linspace(5.0, 80.0, 100)

# Grain-size array + log raised-cosine PDF
s     = Dstd / np.sqrt(1/3 - 2/np.pi**2)
D_arr = np.linspace(np.exp(np.log(D50) - s),
                    np.exp(np.log(D50) + s), 80)
pD    = log_raised_cosine_pdf(D_arr, D50, s) / D_arr

# Saltation time array + truncated log-normal PDF
T_MIN, T_MAX = 0.01, 0.5
t_arr = np.linspace(T_MIN, T_MAX, 70)

mu_t    = np.log(0.08)
sigma_t = 1.0
dist    = lognorm(sigma_t, scale=np.exp(mu_t))
Z       = dist.cdf(T_MAX) - dist.cdf(T_MIN)
pt      = np.where(
    (t_arr >= T_MIN) & (t_arr <= T_MAX),
    dist.pdf(t_arr) / Z,
    0.0,
)

# Forward model
model = MultimodeModel(seismic_params=seismic_params)

psd = model.forward_psd(
    f      = f,
    D      = D_arr,
    H      = H,
    W      = W,
    theta  = theta,
    r0     = r0,
    qb     = qb,
    t      = t_arr,
    D50    = D50,
    pdf_D  = pD,
    pdf_t  = pt,
)

psd_dB = 10 * np.log10(np.maximum(psd, 1e-40))
print(f"PSD range: {psd_dB.min():.1f} to {psd_dB.max():.1f} dB")

# Sensitivity: vary H and qb
H_vals  = [0.1, 0.3, 0.5, 0.8, 1.2]
qb_vals = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
cmap = plt.cm.Blues(np.linspace(0.3, 1.0, len(H_vals)))
for h, c in zip(H_vals, cmap):
    p = model.forward_psd(f, D_arr, h, W, theta, r0, qb, t_arr,
                          D50=D50, pdf_D=pD, pdf_t=pt)
    ax.plot(f, 10 * np.log10(np.maximum(p, 1e-40)),
            color=c, lw=2, label=f"H={h} m")
ax.set_xlabel("Frequency (Hz)", fontsize=12)
ax.set_ylabel("PSD (dB)", fontsize=12)
ax.set_title("Varying flow depth H\n" + r"$q_b=10^{-3}$ m²/s", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
cmap = plt.cm.Reds(np.linspace(0.3, 1.0, len(qb_vals)))
for qb_i, c in zip(qb_vals, cmap):
    p = model.forward_psd(f, D_arr, H, W, theta, r0, qb_i, t_arr,
                          D50=D50, pdf_D=pD, pdf_t=pt)
    ax.plot(f, 10 * np.log10(np.maximum(p, 1e-40)),
            color=c, lw=2, label=f"qb={qb_i:.0e}")
ax.set_xlabel("Frequency (Hz)", fontsize=12)
ax.set_title(f"Varying bedload flux\nH={H} m", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ax = axes[2]
# ax2 = ax.twinx()
# ax.plot(D_arr * 1000, pD / pD.max(), "b-", lw=2, label="Grain-size PDF")
# ax2.plot(t_arr, pt / pt.max(), "r-", lw=2, label="Saltation-time PDF")
# ax.set_xlabel("Grain size (mm)", fontsize=12)
# ax.set_ylabel("Normalised grain-size PDF", fontsize=12, color="b")
# ax2.set_ylabel("Normalised saltation-time PDF", fontsize=12, color="r")
# ax.set_title("Input PDFs", fontsize=12)
# lines1, labels1 = ax.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
# ax.grid(True, alpha=0.3)
ax = axes[2]

# Grain-size PDF — bottom x-axis, left y-axis
ax.plot(D_arr * 1000, pD / pD.max(), "b-", lw=2, label="Grain-size PDF")
ax.set_xlabel("Grain size (mm)", fontsize=12, color="b")
ax.set_ylabel("Normalised PDF", fontsize=12)
ax.tick_params(axis='x', colors='b')

# Saltation-time PDF — top x-axis, same y-axis
ax3 = ax.twiny()
ax3.plot(t_arr, pt / pt.max(), "r-", lw=2, label="Saltation-time PDF")
ax3.set_xlabel("Hop time distribution (s)", fontsize=12, color="r")
ax3.tick_params(axis='x', colors='r')

ax.set_title("Input PDFs", fontsize=12)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle("MultimodeModel — Arroyo de Los Pinos", fontsize=14)
fig.tight_layout()
fig.savefig("pinos_multimode_sensitivity.png", dpi=150, bbox_inches="tight")
print("Figure saved.")
plt.show()