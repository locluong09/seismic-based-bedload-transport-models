from seismic_bedload.abc_class import Empirical
from seismic_bedload.models import SedimentParams

import numpy as np
from typing import Optional, Union

rho_s = SedimentParams.rho_s
rho_f = SedimentParams.rho_f
R = (rho_s - rho_f) / rho_f  # submerged specific gravity of sediment
g = SedimentParams.g

class SeismicHydraulicModel(Empirical):
    def __init__(self, params=None, D50: Union[float, np.ndarray] = None):
        super().__init__()
        if params is None:
            self.params = {'k': 5.2, 'm': 0.25, 'n': 1.36}
        else:
            self.params = params
        self.D50 = D50

    def estimate_shear_from_depth(self, H: np.ndarray, S: float) -> np.ndarray:
        # Using the formula tau = rho * g * H * S, where S is the slope of the riverbed
        shear = rho_f * g * H * S
        dimless_shear = shear / ((rho_s - rho_f) * g * self.D50)  # dimensionless shear stress
        return dimless_shear

    def fit(self, PSD: np.ndarray, shear: np.ndarray) -> np.ndarray:
        """Fit a linear model to the log-transformed shear (dimensionless) and PSD data."""
        y = PSD
        x = np.log10(shear)
        a, b = np.polyfit(x, y, deg = 1)
        PSD_re = a * x + b
        return PSD_re

    def predict(self, PSD, shear):
        PSD_re = self.fit(PSD, shear)
        qb = self.params['k'] * (10 ** (PSD/10 - PSD_re/10)) ** self.params['m'] * shear ** self.params['n'] # dimensionless bedload flux
        conversion_factor = np.sqrt(R * g * self.D50**3) * rho_s
        bedload_flux = qb * conversion_factor # kg/m/s
        return bedload_flux
