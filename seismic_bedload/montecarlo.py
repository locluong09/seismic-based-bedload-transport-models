"""
MonteCarlo-based inversion — Python port of R eseis fmi_inversion.

Two public functions:
  build_reference_catalogue  — sample parameter space and pre-compute PSDs
  lut_inversion              — match observed spectra to catalogue via min-RMSE

Compatible with:
  seismic_bedload.models.SaltationModel
  seismic_bedload.models.MultimodeModel
  seismic_bedload.turbulence.TurbulenceModel  (or any model with .forward_psd)
"""

from __future__ import annotations

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union
from tqdm import tqdm 


@dataclass
class ParameterRange:
    """
    Describes one free parameter for the Monte-Carlo sweep.

    Attributes
    ----------
    name    : parameter name (used as key in output dict)
    low     : lower bound of uniform prior
    high    : upper bound of uniform prior
    log     : if True, sample uniformly in log-space
    fixed   : if not None, ignore low/high and always use this value
    """
    name: str
    low: float = 0.0
    high: float = 1.0
    log: bool = False
    fixed: Optional[float] = None

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.fixed is not None:
            return np.full(n, self.fixed)
        if self.log:
            return np.exp(rng.uniform(np.log(self.low), np.log(self.high), n))
        return rng.uniform(self.low, self.high, n)


# Reference catalogue
@dataclass
class ReferenceCatalogue:
    """
    Container produced by build_reference_catalogue.

    Attributes
    ----------
    spectra    : (n_samples, n_freq) array of log10-PSD [dB]
    parameters : list of dicts, one per sample
    f          : frequency array used
    """
    spectra: np.ndarray
    parameters: list[dict]
    f: np.ndarray


def _compute_one(
    idx: int,
    params: dict,
    model_factory: Callable,
    f: np.ndarray,
    forward_kwargs: Callable[[dict], dict],
) -> tuple[int, np.ndarray]:
    """Worker: instantiate a fresh model and compute one PSD (dB)."""
    model = model_factory(params)
    kw = forward_kwargs(params)
    psd = model.forward_psd(f=f, **kw)
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-40))
    return idx, psd_db


def build_reference_catalogue(
    f: np.ndarray,
    n: int,
    parameter_ranges: list[ParameterRange],
    model_factory: Callable[[dict], Any],
    forward_kwargs: Callable[[dict], dict],
    seed: int = 42,
    n_workers: int = 1,
    verbose: bool = True,
) -> ReferenceCatalogue:
    """
    Build a Monte-Carlo reference catalogue for FMI.

    Parameters
    ----------
    f                : frequency array [Hz]
    n                : number of random parameter sets to sample
    parameter_ranges : list of ParameterRange objects defining the prior
    model_factory    : callable(params_dict) -> model instance
                       The callable receives the sampled parameter dict and
                       should return an initialised SaltationModel /
                       TurbulenceModel / MultimodeModel.
    forward_kwargs   : callable(params_dict) -> dict
                       Maps the parameter dict to keyword arguments for
                       model.forward_psd(f=f, **kw).
    seed             : RNG seed for reproducibility
    n_workers        : number of parallel processes (1 = serial)
    verbose          : show progress bar

    Returns
    -------
    ReferenceCatalogue
    """
    rng = np.random.default_rng(seed)
    param_sets = []
    for _ in range(n):
        p = {pr.name: pr.sample(1, rng)[0] for pr in parameter_ranges}
        param_sets.append(p)

    spectra = np.full((n, len(f)), np.nan)

    if n_workers == 1:
        iterator = enumerate(param_sets)
        if verbose:
            iterator = tqdm(iterator, total=n, desc="Building catalogue")
        for idx, params in iterator:
            try:
                _, psd_db = _compute_one(idx, params, model_factory, f, forward_kwargs)
                spectra[idx] = psd_db
            except Exception as e:
                if verbose:
                    print(f"  [warn] sample {idx} failed: {e}")
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for idx, params in enumerate(param_sets):
                fut = pool.submit(
                    _compute_one, idx, params, model_factory, f, forward_kwargs
                )
                futures[fut] = idx
            bar = tqdm(as_completed(futures), total=n, desc="Building catalogue",
                       disable=not verbose)
            for fut in bar:
                try:
                    idx, psd_db = fut.result()
                    spectra[idx] = psd_db
                except Exception as e:
                    idx = futures[fut]
                    if verbose:
                        print(f"  [warn] sample {idx} failed: {e}")

    # Drop samples where forward model failed
    valid = ~np.any(np.isnan(spectra), axis=1)
    if verbose:
        n_valid = valid.sum()
        print(f"Catalogue: {n_valid}/{n} valid spectra")

    return ReferenceCatalogue(
        spectra=spectra[valid],
        parameters=[param_sets[i] for i in np.where(valid)[0]],
        f=f,
    )


# Look-up table (LUT) inversion
@dataclass
class InversionResult:
    """
    Output of lut_inversion.

    Attributes
    ----------
    parameters : dict of str -> np.ndarray, one value per time step
    rmse       : (n_times, n_freq) frequency-resolved RMSE
    rmse_min   : (n_times,) scalar RMSE for the best-match spectrum
    best_idx   : (n_times,) index of best-match in catalogue
    """
    parameters: dict[str, np.ndarray]
    rmse: np.ndarray
    rmse_min: np.ndarray
    best_idx: np.ndarray


def lut_inversion(
    catalogue: ReferenceCatalogue,
    data: np.ndarray,
) -> InversionResult:
    """
    Match observed spectra to the reference catalogue using minimum RMSE.

    Parameters
    ----------
    catalogue : ReferenceCatalogue from build_reference_catalogue
    data      : (n_freq, n_times) or (n_times, n_freq) observed PSD [dB]
                Frequencies must match catalogue.f exactly.

    Returns
    -------
    InversionResult

    Notes
    -----
    The inversion is purely spectral-shape based: the catalogue spectra are
    normalised by subtracting their median before comparison, mirroring the
    approach in eseis fmi_inversion.
    """
    data = np.asarray(data, dtype=float)

    # Ensure data is (n_times, n_freq)
    n_freq = len(catalogue.f)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[0] == n_freq and data.shape[1] != n_freq:
        data = data.T  # transpose from (n_freq, n_times)
    if data.shape[1] != n_freq:
        raise ValueError(
            f"data has {data.shape[1]} frequency bins but catalogue has {n_freq}"
        )

    n_times = data.shape[0]
    ref = catalogue.spectra                # (n_ref, n_freq)
    params_list = catalogue.parameters

    best_idx = np.full(n_times, -1, dtype=int)
    rmse_min = np.full(n_times, np.nan)
    rmse_f = np.full((n_times, n_freq), np.nan)

    for t in range(n_times):
        obs = data[t]

        if np.any(np.isnan(obs)):
            continue

        # Spectral-shape RMSE: subtract median from both sides
        # obs_norm = obs - np.median(obs)
        # ref_norm = ref - np.median(ref, axis=1, keepdims=True)

        # diff = ref_norm - obs_norm[np.newaxis, :]   # (n_ref, n_freq)
        # rmse_all = np.sqrt(np.mean(diff**2, axis=1))  # (n_ref,)

        # best = int(np.argmin(rmse_all))
        # best_idx[t] = best
        # rmse_min[t] = rmse_all[best]

        # # Frequency-resolved RMSE for best match
        # rmse_f[t] = np.sqrt((ref_norm[best] - obs_norm)**2)
        ## above does not work well -- need to test more

        ### NEW
        # Raw dB RMSE — matches R fmi_inversion exactly
        diff     = ref - obs[np.newaxis, :]           # (n_ref, n_freq)
        rmse_all = np.sqrt(np.mean(diff**2, axis=1))  # (n_ref,)

        best = int(np.argmin(rmse_all))
        best_idx[t] = best
        rmse_min[t] = rmse_all[best]

        # Frequency-resolved RMSE for best match
        rmse_f[t] = np.sqrt((ref[best] - obs)**2)

    # Collect best-match parameters for each time step
    param_keys = list(params_list[0].keys()) if params_list else []
    parameters_out: dict[str, np.ndarray] = {}
    for key in param_keys:
        arr = np.full(n_times, np.nan)
        for t in range(n_times):
            if best_idx[t] >= 0:
                arr[t] = params_list[best_idx[t]][key]
        parameters_out[key] = arr

    return InversionResult(
        parameters=parameters_out,
        rmse=rmse_f,
        rmse_min=rmse_min,
        best_idx=best_idx,
    )


#joint turbulence + bedload catalogue

def build_joint_catalogue(
    f: np.ndarray,
    n: int,
    turbulence_model_factory: Callable[[dict], Any],
    bedload_model_factory: Callable[[dict], Any],
    turbulence_fwd_kwargs: Callable[[dict], dict],
    bedload_fwd_kwargs: Callable[[dict], dict],
    parameter_ranges: list[ParameterRange],
    seed: int = 42,
    n_workers: int = 1,
    verbose: bool = True,
) -> ReferenceCatalogue:
    """
    Build a catalogue where each reference spectrum is turbulence + bedload PSD.

    This mirrors the R example in fmi_inversion where psd_sum = psd_turbulence
    + psd_bedload before taking 10*log10.

    Parameters mirror build_reference_catalogue; model_factory and fwd_kwargs
    are provided separately for turbulence and bedload.
    """
    rng = np.random.default_rng(seed)
    param_sets = []
    for _ in range(n):
        p = {pr.name: pr.sample(1, rng)[0] for pr in parameter_ranges}
        param_sets.append(p)

    spectra = np.full((n, len(f)), np.nan)

    iterator = enumerate(param_sets)
    if verbose:
        iterator = tqdm(iterator, total=n, desc="Building joint catalogue")

    for idx, params in iterator:
        try:
            t_model = turbulence_model_factory(params)
            b_model = bedload_model_factory(params)
            t_kw = turbulence_fwd_kwargs(params)
            b_kw = bedload_fwd_kwargs(params)

            psd_t = t_model.forward_psd(f=f, **t_kw)
            psd_b = b_model.forward_psd(f=f, **b_kw)
            psd_sum = psd_t + psd_b
            spectra[idx] = 10.0 * np.log10(np.maximum(psd_sum, 1e-40))
        except Exception as e:
            if verbose:
                print(f"  [warn] sample {idx} failed: {e}")

    valid = ~np.any(np.isnan(spectra), axis=1)
    if verbose:
        print(f"Joint catalogue: {valid.sum()}/{n} valid spectra")

    return ReferenceCatalogue(
        spectra=spectra[valid],
        parameters=[param_sets[i] for i in np.where(valid)[0]],
        f=f,
    )
