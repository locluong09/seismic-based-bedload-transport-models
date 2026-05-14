# Changelog

All notable changes to `seismic_bedload` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-14

### Added
- `TurbulenceModel` class implementing the Gimbert et al. (2014) seismic turbulence model, enabling combined turbulence + bedload inversion across a wide frequency band (10–100 Hz)
- Monte Carlo LUT (Look-Up Table) inversion module (`montecarlo.py`) with:
  - `build_joint_catalogue` for constructing joint parameter catalogues
  - `lut_inversion` for inverting seismic PSD to bedload flux and flow depth
- Unit tests covering new and existing modules
- Example scripts demonstrating FMI inversion workflow

### Notes
- H/q_b degeneracy problem during inversion still need to be explored!

## [0.1.2] - 2025

### Fixed
- NumPy 2.0+ compatibility issues across core modules

## [0.1.1] - 2025

### Fixed
- Minor bug fixes and packaging corrections post initial release

## [0.1.0] - 2025

### Added
- Initial release of `seismic_bedload`
- `SaltationModel` based on Tsai et al. (2012)
- `MultimodeModel` based on Luong et al. (2024, *Water Resources Research*)
- Core FMI (Fluvial Model Inversion) framework ported from R `eseis` package