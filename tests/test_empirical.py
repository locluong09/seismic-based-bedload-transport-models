import numpy as np
import matplotlib.pyplot as plt
from seismic_bedload import SeismicHydraulicModel

def test_seismic_hydraulic_model():
    # Just a non-physical sample data for testing
    shear = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) # dimensionless shear stress
    PSD = np.array([10, 20, 30, 40, 50]) #dB

    model = SeismicHydraulicModel(D50 = 0.01) # D50 in meters, for conversion to bedload flux in kg/m/s

    # Fit the model
    PSD_re = model.fit(PSD, shear)
    print("Fitted PSD:", PSD_re)

    # Predict bedload flux
    bedload_flux = model.predict(PSD, shear) # dimesionless bedload flux
    print("Predicted bedload flux:", bedload_flux)

    # Plotting for visualization
    plt.scatter(np.log10(shear), PSD, label='Observed PSD')
    plt.plot(np.log10(shear), PSD_re, color='red', label='Fitted PSD')
    plt.xlabel('log10(Shear)')
    plt.ylabel('PSD (dB)')
    plt.legend()
    plt.show()

    plt.scatter(shear, bedload_flux, label='Predicted Bedload Flux')
    plt.xlabel('Shear (-)')
    plt.ylabel('Bedload Flux (kg/m/s)')
    plt.legend()
    plt.show()

    # Different k, m, n parameters for testing
    custom_params = {'k': 4.0, 'm': 0.3, 'n': 1.5}
    model_custom = SeismicHydraulicModel(params=custom_params, D50 = 0.01)
    bedload_flux_custom = model_custom.predict(PSD, shear)
    print("Predicted bedload flux with custom parameters:", bedload_flux_custom)
    plt.scatter(shear, bedload_flux_custom, label='Predicted Bedload Flux (Custom values for k, m, n)', color='green')
    plt.xlabel('Shear (-)')
    plt.ylabel('Bedload Flux (kg/m/s)')
    plt.legend()
    plt.show()

    # testing with pinos data
    PSD_observe = np.loadtxt("data/pinos/PSD.txt")
    idx = np.arange(49, (49+317))
    PSD_obs = PSD_observe[idx]
    S = np.tan(0.7*np.pi/180)
    H = np.loadtxt("data/pinos/flowdepth.txt")
    H = H/100
    model_pinos = SeismicHydraulicModel(D50 = 0.005)
    shear_pinos = model_pinos.estimate_shear_from_depth(H, S)
    PSD_re_pinos = model_pinos.fit(PSD_obs, shear_pinos)
    bedload_flux_pinos = model_pinos.predict(PSD_obs, shear_pinos)
    
    plt.scatter(np.log10(shear_pinos), PSD_obs, label='Observed PSD (Pinos)')
    plt.plot(np.log10(shear_pinos), PSD_re_pinos, color='red', label='Fitted PSD (Pinos)')
    plt.xlabel('log10(Shear)')
    plt.ylabel('PSD (dB)')
    plt.legend()
    plt.show()  

    plt.scatter(shear_pinos, bedload_flux_pinos, label='Predicted Bedload Flux (Pinos)')
    plt.xlabel('Shear (-)')
    plt.ylabel('Bedload Flux (kg/m/s)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_seismic_hydraulic_model()