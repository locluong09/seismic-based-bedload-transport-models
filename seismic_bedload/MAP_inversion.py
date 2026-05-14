import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional, Union, Tuple
from enum import Enum

class ModelType(Enum):
    SALTATION  = "saltation"
    MULTIMODE  = "multimode"


@dataclass
class BandConfig:
    """
    Frequency band configuration.
    Physically motivated by spectral content:
        low  → turbulence dominated → sensitive to H
        mid  → mixed contribution
        high → bedload impact dominated → sensitive to qb
    """
    low:  Tuple[float, float]  = (1.0,  20.0)   # Hz
    mid:  Tuple[float, float]  = (20.0, 50.0)   # Hz
    high: Tuple[float, float]  = (50.0, 100.0)  # Hz

    ## Band weights in likelihood
    ## Upweight physically clean bands
    w_low:  float = 1.5    # H sensitive
    w_mid:  float = 1.0    # mixed
    w_high: float = 1.5    # qb sensitive


@dataclass
class PriorConfig:
    """
    Prior distributions for MAP inversion.
    Set from Manning equation and MPM bedload formula
    or from intuition / field observation.
    """
    ## h_w prior — Gaussian
    h_mean:     float = 0.3       # m — from Manning or gauge
    h_sigma:    float = 0.15      # m — your confidence in Manning

    ## qb prior — log-normal (flux spans orders of magnitude)
    qb_mean:    float = 0.01      # kg/m/s
    qb_sigma:   float = 1.0       # log-space sigma (~factor 3 uncertainty)

    ## Parameter bounds for optimizer
    h_min:      float = 0.01      # m
    h_max:      float = 2.0       # m
    qb_min:     float = 1e-6      # kg/m/s
    qb_max:     float = 10.0      # kg/m/s


def compute_band_averages(
        psd:    np.ndarray,
        f:      np.ndarray,
        bands:  BandConfig,
        ) -> np.ndarray:
    """
    Compress full PSD spectrum into 3 band averages.
    Works on both observed and modeled PSDs.

    Parameters
    ----------
    psd  : [n_freq] PSD in linear scale
    f    : [n_freq] frequency array in Hz
    bands: BandConfig

    Returns
    -------
    band_avgs : [3] array — [low, mid, high] band averages
                in log10 scale (dB-like, more Gaussian distributed)
    """
    def band_mean(f_lo, f_hi, log_scale=False):
        mask = (f >= f_lo) & (f < f_hi)
        if mask.sum() == 0:
            return np.nan
        ## Average in linear, convert to log — preserves spectral energy
        if log_scale:
            return np.log10(np.mean(psd[mask]))
        else:
            return np.mean(psd[mask])

    return np.array([
        band_mean(*bands.low),
        band_mean(*bands.mid),
        band_mean(*bands.high)
    ])


def log_likelihood(
        obs_bands:   np.ndarray,
        model_bands: np.ndarray,
        sigma_f:     float,
        band_weights: np.ndarray) -> float:
    """
    Gaussian log-likelihood in log10 PSD space.
    Each band is one observation with its own weight.

    log L = -0.5 * sum_bands [ w_b * (obs_b - model_b)^2 / sigma_f^2 ]
    """
    residual = obs_bands - model_bands               ## [3]
    weighted = band_weights * residual**2 / sigma_f**2
    return -0.5 * np.sum(weighted)


def log_prior(
        H:      float,
        qb:     float,
        prior:  PriorConfig) -> float:
    """
    Log-prior = log P(H) + log P(qb)

    H  ~ Gaussian  (Manning gives good point estimate)
    qb ~ LogNormal (flux spans orders of magnitude, always positive)
    """
    ## Gaussian prior on H
    lp_H  = -0.5 * ((H - prior.h_mean) / prior.h_sigma)**2

    ## Log-normal prior on qb
    lp_qb = -0.5 * ((np.log(qb) - np.log(prior.qb_mean)) / prior.qb_sigma)**2

    return lp_H + lp_qb


class MAPInversion:
    """
    MAP inversion for bedload flux and flow depth
    using 3-band PSD compression.

    Supports both SaltationModel (Tsai) and MultimodeModel (Luong).

    Usage
    -----
    inverter = MAPInversion(
        model      = saltation_model,
        model_type = ModelType.SALTATION,
        f          = f_array,
        D          = D50,
        W          = 5.0,
        theta      = 0.02,
        r0         = 100.0,
        bands      = BandConfig(),
        prior      = PriorConfig(h_mean=0.3, h_sigma=0.1,
                                 qb_mean=0.01, qb_sigma=0.8)
    )

    results = inverter.invert_timeseries(
        PSD_obs_matrix,       ## [n_freq x n_time] in dB
        f,
        n_restarts = 3
    )
    """

    def __init__(self,
                 model,
                 model_type:  ModelType,
                 f:           np.ndarray,
                 D:           Union[float, np.ndarray],
                 W:           float,
                 theta:       float,
                 r0:          float,
                 bands:       BandConfig  = None,
                 prior:       PriorConfig = None,
                 sigma_f:     float       = 0.1,
                 ## MultimodeModel only
                 t:           Union[float, np.ndarray] = None,
                 pdf_D:       Optional[np.ndarray]     = None,
                 pdf_t:       Optional[np.ndarray]     = None,
                 tau_c:       Optional[float]          = None,
                 D50:         Optional[float]          = None):

        self.model      = model
        self.model_type = model_type
        self.f          = np.asarray(f)
        self.D          = D
        self.W          = W
        self.theta      = theta
        self.r0         = r0
        self.bands      = bands if bands else BandConfig()
        self.prior      = prior if prior else PriorConfig()
        self.sigma_f    = sigma_f
        self.t          = t
        self.pdf_D      = pdf_D
        self.pdf_t      = pdf_t
        self.tau_c      = tau_c
        self.D50        = D50

        ## Band weights as array [low, mid, high]
        self.band_weights = np.array([
            self.bands.w_low,
            self.bands.w_mid,
            self.bands.w_high
        ])


    def _forward_bands(self, H: float, qb: float) -> np.ndarray:
        """
        Run forward model and compress to 3 band averages.
        This is called at every optimizer iteration.

        Returns log10 band averages [3].
        """
        if self.model_type == ModelType.SALTATION:
            psd = self.model.forward_psd(
                f      = self.f,
                D      = self.D,
                H      = H,
                W      = self.W,
                theta  = self.theta,
                r0     = self.r0,
                qb     = qb,
                tau_c  = self.tau_c,
                D50    = self.D50,
                pdf  = self.pdf_D
            )
        else:  ## MULTIMODE
            psd = self.model.forward_psd(
                f      = self.f,
                D      = self.D,
                H      = H,
                W      = self.W,
                theta  = self.theta,
                r0     = self.r0,
                qb     = qb,
                t      = self.t,
                tau_c  = self.tau_c,
                D50    = self.D50,
                pdf_D  = self.pdf_D,
                pdf_t  = self.pdf_t
            )

        return compute_band_averages(psd, self.f, self.bands)


    def _cost(self,
              params:     np.ndarray,
              obs_bands:  np.ndarray) -> float:
        """
        Negative log-posterior = cost to minimize.

        params[0] = H   (linear scale)
        params[1] = qb  (log scale — optimizer works in log space
                          to handle the orders-of-magnitude range)
        """
        H  = params[0]
        qb = np.exp(params[1])     ## optimizer sees log(qb)

        ## Hard bounds — return large cost if unphysical
        if (H  <= self.prior.h_min  or H  >= self.prior.h_max or
            qb <= self.prior.qb_min or qb >= self.prior.qb_max):
            return 1e10

        ## Forward model → band averages
        try:
            model_bands = self._forward_bands(H, qb)
        except Exception:
            return 1e10

        ## Guard against NaN from forward model
        if not np.all(np.isfinite(model_bands)):
            return 1e10

        ## Negative log-posterior
        nll = -log_likelihood(obs_bands, model_bands,
                              self.sigma_f, self.band_weights)
        nlp = -log_prior(H, qb, self.prior)

        return nll + nlp


    def invert_single(self,
                      PSD_obs_dB:  np.ndarray,
                      n_restarts:  int = 3
                      ) -> dict:
        """
        MAP inversion for a single timestep.

        Parameters
        ----------
        PSD_obs_dB : [n_freq] observed PSD in dB (10*log10 scale)
        n_restarts : number of random restarts to avoid local minima

        Returns
        -------
        dict with keys:
            H_map, qb_map       — MAP estimates
            H_std, qb_std       — uncertainty (Laplace approximation)
            log_post            — log-posterior at MAP
            obs_bands           — observed band averages
            model_bands_map     — modeled band averages at MAP
            residual_bands      — obs - model per band
            converged           — optimizer convergence flag
        """
        ## Convert observed PSD from dB to linear, then band average
        psd_linear = 10**(PSD_obs_dB / 10.0)
        obs_bands  = compute_band_averages(psd_linear, self.f, self.bands)

        if not np.all(np.isfinite(obs_bands)):
            return self._nan_result(obs_bands)

        best_result = None
        best_cost   = np.inf

        ## Multiple restarts — MAP cost surface can be multimodal
        ## First restart always from prior mean (physically motivated)
        init_points = [(self.prior.h_mean, np.log(self.prior.qb_mean))]

        ## Additional random restarts in parameter space
        rng = np.random.default_rng(seed=42)
        for _ in range(n_restarts - 1):
            h_rand  = rng.uniform(self.prior.h_min,         self.prior.h_max)
            qb_rand = rng.uniform(np.log(self.prior.qb_min), np.log(self.prior.qb_max))
            init_points.append((h_rand, qb_rand))

        for h0, log_qb0 in init_points:
            result = minimize(
                fun    = self._cost,
                x0     = [h0, log_qb0],
                args   = (obs_bands,),
                method = 'Nelder-Mead',       ## derivative-free
                options = {
                    'xatol':   1e-4,
                    'fatol':   1e-4,
                    'maxiter': 2000,
                    'adaptive': True          ## helps in 2D
                }
            )
            if result.fun < best_cost:
                best_cost   = result.fun
                best_result = result

        ## Extract MAP solution
        H_map  = best_result.x[0]
        qb_map = np.exp(best_result.x[1])

        ## Uncertainty via numerical Hessian (Laplace approximation)
        H_std, qb_std = self._laplace_uncertainty(
            best_result.x, obs_bands
        )

        ## Diagnostics
        model_bands_map = self._forward_bands(H_map, qb_map)
        residual_bands  = obs_bands - model_bands_map
        log_post        = -best_cost

        return {
            'H_map':            H_map,
            'qb_map':           qb_map,
            'H_std':            H_std,
            'qb_std':           qb_std,
            'log_post':         log_post,
            'obs_bands':        obs_bands,
            'model_bands_map':  model_bands_map,
            'residual_bands':   residual_bands,
            'converged':        best_result.success
        }


    def _laplace_uncertainty(self,
                             theta_map: np.ndarray,
                             obs_bands: np.ndarray,
                             eps:       float = 1e-4
                             ) -> Tuple[float, float]:
        """
        Estimate posterior uncertainty via Laplace approximation.
        Numerically compute diagonal of Hessian at MAP,
        invert to get posterior variance.

        Uncertainty on qb is in log-space, then converted back.
        """
        cost_map = self._cost(theta_map, obs_bands)

        hess_diag = np.zeros(2)
        for i in range(2):
            dtheta      = np.zeros(2)
            dtheta[i]   = eps
            c_plus      = self._cost(theta_map + dtheta, obs_bands)
            c_minus     = self._cost(theta_map - dtheta, obs_bands)
            ## Second derivative via finite difference
            hess_diag[i] = (c_plus - 2*cost_map + c_minus) / eps**2

        ## Posterior variance = 1 / Hessian diagonal
        ## Clip to avoid division by near-zero
        hess_diag = np.clip(hess_diag, 1e-8, np.inf)
        var       = 1.0 / hess_diag

        H_std  = np.sqrt(var[0])
        ## qb uncertainty: back-transform from log space
        ## sigma_qb ≈ qb_map * sigma_log_qb
        qb_map = np.exp(theta_map[1])
        qb_std = qb_map * np.sqrt(var[1])

        return H_std, qb_std


    def invert_timeseries(self,
                          PSD_obs_matrix: np.ndarray,
                          n_restarts:     int  = 3,
                          verbose:        bool = True
                          ) -> dict:
        """
        Joint MAP inversion over all timesteps.
        Each minute inverted independently but priors
        can be updated sequentially (see use_sequential).

        Parameters
        ----------
        PSD_obs_matrix : [n_freq x n_time] observed PSD in dB
        n_restarts     : restarts per timestep

        Returns
        -------
        dict with time series of all inversion outputs
        """
        n_time = PSD_obs_matrix.shape[1]

        ## Initialize output arrays
        H_map    = np.full(n_time, np.nan)
        qb_map   = np.full(n_time, np.nan)
        H_std    = np.full(n_time, np.nan)
        qb_std   = np.full(n_time, np.nan)
        log_post = np.full(n_time, np.nan)
        converged= np.zeros(n_time, dtype=bool)

        obs_bands_all   = np.full((n_time, 3), np.nan)
        model_bands_all = np.full((n_time, 3), np.nan)
        residual_all    = np.full((n_time, 3), np.nan)

        for t in range(n_time):
            if verbose:
                print(f"  Inverting timestep {t+1}/{n_time}...", end="\r")

            result = self.invert_single(
                PSD_obs_dB = PSD_obs_matrix[:, t],
                n_restarts = n_restarts
            )

            H_map[t]    = result['H_map']
            qb_map[t]   = result['qb_map']
            H_std[t]    = result['H_std']
            qb_std[t]   = result['qb_std']
            log_post[t] = result['log_post']
            converged[t]= result['converged']

            obs_bands_all[t,:]   = result['obs_bands']
            model_bands_all[t,:] = result['model_bands_map']
            residual_all[t,:]    = result['residual_bands']

        if verbose:
            print(f"\n  Done. Converged: {converged.sum()}/{n_time} timesteps")

        return {
            'H_map':        H_map,
            'qb_map':       qb_map,
            'H_std':        H_std,
            'qb_std':       qb_std,
            'log_post':     log_post,
            'converged':    converged,
            'obs_bands':    obs_bands_all,
            'model_bands':  model_bands_all,
            'residual':     residual_all
        }


    def _nan_result(self, obs_bands: np.ndarray) -> dict:
        """Return NaN result for bad timesteps."""
        return {
            'H_map':           np.nan,
            'qb_map':          np.nan,
            'H_std':           np.nan,
            'qb_std':          np.nan,
            'log_post':        np.nan,
            'obs_bands':       obs_bands,
            'model_bands_map': np.full(3, np.nan),
            'residual_bands':  np.full(3, np.nan),
            'converged':       False
        }