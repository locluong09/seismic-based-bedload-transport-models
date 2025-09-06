import numpy as np
import matplotlib.pyplot as plt
from seismic_bedload import SeismicHydraulicModel

def test_seismic_hydraulic_model():
    # Just a non-physical sample data for testing
    shear = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    PSD = np.array([10, 20, 30, 40, 50])

    model = SeismicHydraulicModel()

    # Fit the model
    PSD_re = model.fit(PSD, shear)
    print("Fitted PSD:", PSD_re)

    # Predict bedload flux
    bedload_flux = model.predict(PSD, shear)
    print("Predicted bedload flux:", bedload_flux)

    # Plotting for visualization
    plt.scatter(np.log10(shear), PSD, label='Observed PSD')
    plt.plot(np.log10(shear), PSD_re, color='red', label='Fitted PSD')
    plt.xlabel('log10(Shear)')
    plt.ylabel('PSD')
    plt.legend()
    plt.show()