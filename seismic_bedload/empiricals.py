from seismic_bedload.abc_class import Empirical
import numpy as np
from typing import Optionalm, Union

class SeismicHydraulicModel(Empirical):
    def __init__(self, params=None):
        super().__init__()
        if params is None:
            self.params = {'k': 5.2, 'm': 0.25, 'n': 1.36}
        else:
            self.params = params

    def fit(self, PSD: np.ndarray, shear: np.ndarray) -> np.ndarray:
        y = PSD
        x = np.log10(shear)
        a, b = np.polyfit(x, y, deg = 1)
        PSD_re = a * x + b
        return PSD_re

    def predict(self, PSD, shear):
        PSD_re = self.fit(PSD, shear)
        bedload_flux = self.params['k'] * (10 ** (PSD/10 - PSD_re/10)) ** self.params['m'] * shear ** self.params['n']
        return bedload_flux
