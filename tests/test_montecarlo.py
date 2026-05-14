# tests/test_montecarlo.py
import numpy as np
import pytest
from seismic_bedload import (
    SaltationModel, TurbulenceModel, SeismicParams,
    ParameterRange, build_joint_catalogue, lut_inversion,
)

f   = np.logspace(np.log10(5), np.log10(80), 50)
W     = 6.0
theta = 0.0075
r0    = 6.0
D50   = 0.01

ranges = [
    ParameterRange("H",     low=0.02, high=1.20, log=True),
    ParameterRange("qb",    low=1e-6, high=3e-3, log=True),
    ParameterRange("W",     fixed=W),
    ParameterRange("theta", fixed=theta),
    ParameterRange("r0",    fixed=r0),
    ParameterRange("D50",   fixed=D50),
]

def factory_t(p): return TurbulenceModel()
def factory_b(p): return SaltationModel()
def kw_t(p): return dict(D=p["D50"], W=p["W"], H=p["H"], theta=p["theta"], r0=p["r0"], D50=p["D50"])
def kw_b(p): return dict(D=p["D50"], H=p["H"], W=p["W"], theta=p["theta"], r0=p["r0"], qb=p["qb"])


def test_catalogue_builds():
    cat = build_joint_catalogue(
        f=f, n=20,
        turbulence_model_factory=factory_t,
        bedload_model_factory=factory_b,
        turbulence_fwd_kwargs=kw_t,
        bedload_fwd_kwargs=kw_b,
        parameter_ranges=ranges,
        verbose=False,
    )
    assert cat.spectra.shape[1] == len(f)
    assert len(cat.parameters) == cat.spectra.shape[0]


def test_inversion_returns_H_qb():
    cat = build_joint_catalogue(
        f=f, n=50,
        turbulence_model_factory=factory_t,
        bedload_model_factory=factory_b,
        turbulence_fwd_kwargs=kw_t,
        bedload_fwd_kwargs=kw_b,
        parameter_ranges=ranges,
        verbose=False,
    )
    # synthetic observation: one time step
    obs = cat.spectra[[0]]   # (1, n_freq)
    result = lut_inversion(cat, obs)
    assert "H" in result.parameters
    assert "qb" in result.parameters
    assert result.rmse_min.shape == (1,)


def test_inversion_recovers_catalogue_entry():
    """Best match for a catalogue spectrum should be itself (RMSE ~ 0)."""
    cat = build_joint_catalogue(
        f=f, n=30,
        turbulence_model_factory=factory_t,
        bedload_model_factory=factory_b,
        turbulence_fwd_kwargs=kw_t,
        bedload_fwd_kwargs=kw_b,
        parameter_ranges=ranges,
        verbose=False,
    )
    obs = cat.spectra[[5]]
    result = lut_inversion(cat, obs)
    assert result.best_idx[0] == 5