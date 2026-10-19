<!-- SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder -->
<!-- SPDX-License-Identifier: ISC -->
<div align="center">
  <img src="docs/images/ccar_logo_w_name_transparent.png" width="30%">&nbsp;&nbsp;&nbsp;
  <img src="docs/images/scarabaeus_logo.png" width="30%">&nbsp;&nbsp;&nbsp;
  <img src="docs/images/ORCCA_Logo_Reverse.png" width="30%">
</div>

<h1 align="center">Scarabaeus</h1>

<p align="center">
  <b>Open-source spacecraft navigation & orbit determination framework</b><br>
  Developed by the <a href="https://www.colorado.edu/faculty/mcmahon/orcca">Orbital Research Cluster for Celestial Applications (ORCCA)</a><br>
  Colorado Center for Astrodynamics Research (CCAR) · University of Colorado Boulder
</p>

<p align="center">
  <a href="https://ccar-orcca.github.io/scarabaeus-docs/"><img alt="Documentation" src="https://img.shields.io/badge/docs-online-blue"></a>
  <a href="LICENSE"><img alt="License: ISC" src="https://img.shields.io/badge/license-ISC-green"></a>
  <img alt="Version" src="https://img.shields.io/badge/version-2026.0.0-informational">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue">
  <img alt="Rust backend" src="https://img.shields.io/badge/backend-Rust-orange">
</p>

---

## Overview

**Scarabaeus (SCB)** is a Python framework for spacecraft navigation and orbit determination (OD). It provides a unified, mission-agnostic environment for:

- Building and propagating spacecraft trajectories with multiple force models
- Simulating or ingesting radiometric and optical measurements
- Running sequential and batch OD filters (LKF, SRIF, LSB, SRIFB, …) and sampling their posteriors with MCMC
- Estimating dynamical, measurement, and consider parameters
- Performing maneuver planning and finite/impulsive burn modelling

SCB interfaces with [NASA SPICE](https://naif.jpl.nasa.gov/naif/toolkit.html) for time, frame, and kernel management and ships a compiled **Rust** back-end (`scarabaeus.rust`) for the integrator, the ephemeris cache and the gravity and SRP kernels.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Requirements](#requirements)
4. [Installation](#installation)
   - [User Installation](#user-installation)
   - [Developer Installation](#developer-installation)
5. [Tutorials](#tutorials)
6. [Testing](#testing)
7. [Contributing](#contributing)
8. [Citing Scarabaeus](#citing-scarabaeus)
9. [License](#license)

---

## Features

### Dynamics & Force Models
| Class | Description |
|---|---|
| `PointMassGravity` | Two-body (Keplerian) gravitational acceleration |
| `ThreeBodyGravity` | Third-body perturbation via SPICE ephemeris |
| `SphericalHarmonicsGravity` *(Rust core)* | Gravity field up to arbitrary degree/order; several fields per spacecraft |
| `PolyhedralGravity` *(Rust core)* | Constant-density polyhedral shape-model gravity, valid down to the surface |
| `CannonballSRP` | Solar radiation pressure (cannonball model) |
| `nPlateSRP` *(Rust core)* | Solar radiation pressure (N-plate model with CK-frame solar arrays) |
| `ThermalRadiation` | Thermal re-radiation recoil acceleration |
| `AntennaRadiationPressure` | Recoil from the radiated antenna power |
| `YarkovskyEffect` | Yarkovsky non-gravitational force |
| `FirstOrderGaussMarkov` | First-order Gauss–Markov stochastic accelerations |
| `PiecewiseFirstOrderGaussMarkov` | Piecewise Gauss–Markov accelerations |
| `PiecewisePolynomialAcceleration` | Piecewise constant or linear empirical accelerations |
| `FourierAcceleration` | Empirical accelerations as a Fourier series in time, argument of latitude or solar longitude |
| `ImpulsiveBurn` | Instantaneous delta-v manoeuvres |
| `SmoothedImpulsiveBurn` | Impulsive manoeuvre as a narrow smooth pulse, without splitting the trajectory |
| `FiniteBurn` | Continuous thrust finite burn arcs |
| `CustomDynamicModel` / `CustomParameter` | User-defined force models and solve-for parameters, written in a script |

`ForceModelTranslation` assembles the models above for one spacecraft and is what `Propagator` takes as `force_models`. `DynamicModel` and `ForceModel` are their abstract base classes, and `DynamicalUtils` holds the attitude and component-frame geometry they share. `Maneuver` and `ManeuverParser` describe finite burns as time polynomials of thrust, mass flow and pointing. `ForceModelRotation` (attitude dynamics) is a placeholder and not implemented yet.

### Integrators
Selected with `Propagator(..., integrator=...)`:

| `integrator` | Description |
|---|---|
| `"DOP853"` (default) | Dormand–Prince 8(5,3) via SciPy `solve_ivp`, with the state transition matrix |
| `"IAS15"` *(Rust)* | Implicit adaptive 15th-order integrator (Rein & Spiegel 2015); state only, so pass `propagate_STM=False` |

### Measurement Models
| Class | Type |
|---|---|
| `RangeIdeal` | Range (geometric, no light time or media) |
| `RangeRateIdeal` | Line-of-sight range rate (geometric, no light time or media) |
| `DopplerIdeal` | Two- and three-way Doppler shift (no light time or media) |
| `DopplerReal` | Doppler from real DSN tracking data |
| `SequentialRangingReal` | Sequential ranging from real DSN tracking data |
| `DiffOneWayRangeIdeal` | Differential one-way range (DOR) between two stations |
| `AngularIdeal` | Right ascension / declination optical measurements |
| `CentroidingIdeal` | Pixel-plane centroiding measurements |
| `CustomMeasurement` | User-defined observable and Jacobian |

`Measurement` is the base class of every model above. `MeasurementScheduler` builds simulated tracking schedules. `MeasurementRegistry` (with `MeasurementType`), `MeasurementStream` and `MeasurementConfig` describe real data streams, which are read into `ObsData`. `MeasurementSpec` and `MeasurementDataSet` package the measurements for the filters, and `OpNavManager` is the pinhole projection the optical models share. `MediaCorrections`, `RampTableManager`, `SolidTide`, `PoleTide`, `OceanLoading` and `BodyOrientation` model the media, ramps, station displacements and Earth orientation of the real DSN observables.

### Orbit Determination
| Class | Description |
|---|---|
| `LKF` | Linearised Kalman Filter (sequential), with a Rauch–Tung–Striebel smoother |
| `SRIF` | Square Root Information Filter (sequential), with a Bierman–Thornton backward smoother |
| `LSB` | Weighted least-squares batch estimator |
| `SRIFB` | Square Root Information Filter in batch form (no process noise) |
| `MCMC` | Markov chain Monte Carlo sampling (`emcee`) of the posterior about a converged filter solution |
| `MultiFilterOD` | Multi-arc OD combining independent filters, each with its own propagator, data and settings |
| `CustomFilter` | User-supplied estimator run on the library's residuals, partials and STMs |
| `MeasurementEditing` | Outlier editing: lasso selection, chi-squared, kurtosis, date ranges, isolation forest |
| `StateNoiseCompensation` | SNC process noise |
| `DynamicalModelCompensation` | DMC process noise |
| `SolutionOD` | OD solution with metadata, saved to JSON, with Keplerian covariance mapping |
| `SolutionCloseness` | Statistical distance between OD solutions |

`FilterOD` is the base class of the filters. A run is configured with `FilterSettings` (a-priori covariance, process noise, measurement noise, editing and output), `ProcessNoiseSettings` and `OutputSettings`. `ProcessNoise` is the base class of SNC and DMC, `FilterDataManager` combines and edits the measurement data sets, and `UtilsOD` holds static helpers for comparing and merging solutions.

### Uncertainty Quantification
- `CovarianceMatrix` — covariance with units, standard deviations and correlations
- `MonteCarloSamples` — weighted sample clouds with their moments, percentiles and effective sample size
- `UncertaintyQuantification` — common base class of the two

### Spacecraft & Environment
- `Body` — base class of `Spacecraft`, `CelestialBody` and `GroundStation`
- `nPlateModel` — N-plate geometry with static and CK-frame (rotating) panels
- `Instrument` — base class of the `Camera` and `Antenna` instruments; `Actuator` for thrusting components
- `StateDefinition` / `StateArray` — build and hold the estimation state vector
- `Propagator` — numerical propagation of the state and STM under a set of force models
- `MissionSequence` — trajectory legs joined at nodes, with impulsive burns at the nodes; `ScenarioSetup` propagates it leg by leg
- `Trajectory` — post-processing and visualisation of propagated states

### Utilities
- `ArrayWUnits` / `Units` / `Dimensions` — dimension-safe arithmetic with full unit tracking
- `Frame` / `ArrayWFrame` — SPICE reference frames and arrays attached to them
- `EpochArray` — time object supporting UTC, TDB, TT, SCLK conversions
- `SpiceManager` — unified SPICE wrapper (kernels, frames, ephemeris queries)
- `Constants` — physical and planetary constants
- `OrbitalElements` — Cartesian ↔ Keplerian conversions
- `Bplane` — B-plane targeting utilities
- `Noise` — Gaussian noise for simulated measurements
- `SCBPolynomial` — polynomials for time-varying scalar profiles
- `CustomCallable` — keyword-checked user functions behind the `Custom*` classes
- `Plotting` — styled matplotlib helpers
- `EphemerisCache` *(Rust)* — Chebyshev ephemeris cache for third-body accelerations
- `ShapeHarmonics` — spherical-harmonic coefficients of a polyhedral shape model
- `Serialization` — JSON / pickle save and load helpers
- `Utils` — general array and file helpers

---

## Architecture

```
scarabaeus/
├── src/
│   ├── scarabaeus/               # Python front-end
│   │   ├── body/                 # Body, CelestialBody, GroundStation, station tides and ocean loading
│   │   ├── dynamics/             # Force models, burn models, ForceModelTranslation
│   │   ├── environment/          # Propagator, StateDefinition, StateArray, Trajectory, MissionSequence, ScenarioSetup
│   │   ├── finiteBurn/           # Maneuver, ManeuverParser
│   │   ├── guidance/             # B-plane targeting
│   │   ├── measurements/         # Measurement models, scheduling, real data streams, media corrections
│   │   ├── orbitDetermination/   # Filters (LKF, SRIF, LSB, SRIFB, MCMC, …), process noise, solutions
│   │   ├── spacecraft/           # Spacecraft, Instrument, Camera, Antenna, Actuator, nPlateModel
│   │   ├── timeAndFrame/         # EpochArray, Frame, ArrayWFrame, SpiceManager, BodyOrientation
│   │   ├── uncertaintyQuantification/  # CovarianceMatrix, MonteCarloSamples
│   │   ├── units/                # Units, Dimensions, ArrayWUnits
│   │   └── utils/                # Constants, Plotting, OrbitalElements, EphemerisCache, ShapeHarmonics, SolutionCloseness, Serialization, Utils
│   └── scarabaeus_rust/          # Rust back-end (compiled via maturin / PyO3)
│       └── src/
│           ├── ias15.rs          # IAS15 integrator
│           ├── integration_event.rs
│           ├── ephemeris_cache.rs
│           ├── spherical_harmonics.rs
│           ├── nplate_srp.rs
│           ├── polyhedron_gravity.rs
│           ├── polyhedron_harmonics.rs
│           └── lib.rs
├── tutorials/                    # Worked examples (basics → advanced)
├── tests/                        # pytest test suite
├── docs/                         # Sphinx documentation source
├── pyproject.toml
├── DEVELOPMENT.md
└── LICENSE
```

---

## Requirements

| Dependency | Minimum | Used for |
|---|---|---|
| Python | 3.11.9 | |
| Rust (rustup) | 1.85 | compiling the Rust back-end, which `pip install` does (the crate uses the 2024 edition) |
| NumPy / SciPy | | arrays, linear algebra, DOP853 integration |
| SpiceyPy | | SPICE toolkit Python bindings |
| pyerfa | 2.0 | Earth orientation (ERFA routines) |
| autograd | | automatic differentiation |
| scikit-learn | | isolation-forest measurement editing |
| emcee / corner | | MCMC sampling and its corner plots |
| matplotlib / mpld3 | | plotting |
| tqdm | | progress bars |
| ipykernel / ipympl | | Jupyter notebook support |

Only Python and Rust have to be installed beforehand: pip installs the Python packages, and the rest of the list in [`pyproject.toml`](pyproject.toml), automatically.

> **pip version** — the developer install uses `--group dev` syntax, which requires **pip ≥ 25.1**. Run `pip install --upgrade pip` if you encounter an error.

---

## Installation

### User Installation

#### 1. Install Rust

pip compiles the Rust back-end while it installs Scarabaeus, so Rust 1.85 or newer is needed. Follow the [rustup installer](https://rustup.rs/) instructions, then verify:

```bash
rustc --version
```

#### 2. Clone the repository

```bash
git clone https://github.com/ccar-orcca/scarabaeus.git
cd scarabaeus
```

#### 3. Create a virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

#### 4. Install Scarabaeus

```bash
pip install .
```

Scarabaeus is not published on PyPI yet, so it is installed from the clone.

> **SPICE kernels** — SCB relies on SPICE kernels for ephemeris, frame, and spacecraft clock data. These are not distributed with the package. Load them in your script using `scb.SpiceManager.load_kernel_from_mkfile(path_to_metakernel)`.

---

### Developer Installation

> See [DEVELOPMENT.md](DEVELOPMENT.md) for the full guide. The summary is below.

#### 1. Follow the user installation steps above

#### 2. Install in editable mode with the developer extras

```bash
pip install -e . --group dev
```

#### 3. Rebuild the Rust back-end after Rust changes

```bash
maturin develop
```

This recompiles the Rust code and binds it to the Python package. Re-run it whenever Rust source files change.

#### 4. Install commit hooks

```bash
pre-commit install
```

This installs `nbstripout` and other hooks that run before each commit.

---

## Tutorials

A tiered tutorial suite lives in [`tutorials/`](tutorials/) as Jupyter Notebooks (`.ipynb`).

### Basics
| Tutorial | Description |
|---|---|
| [`basics_AWU_and_AWF.ipynb`](tutorials/basics_AWU_and_AWF.ipynb) | `ArrayWUnits` and `ArrayWFrame` — dimension-safe arithmetic and frame-attached arrays |
| [`basics_EpochArray.ipynb`](tutorials/basics_EpochArray.ipynb) | Time representations and conversions (UTC, TDB, TT, SCLK) |
| [`basics_SpiceManager_CelestialBody_Constants.ipynb`](tutorials/basics_SpiceManager_CelestialBody_Constants.ipynb) | Loading SPICE kernels, time/frame queries, and defining celestial bodies |

### Intermediate
| Tutorial | Description |
|---|---|
| [`intermediate_Propagator_and_Trajectory.ipynb`](tutorials/intermediate_Propagator_and_Trajectory.ipynb) | Propagator configuration with multiple force models; trajectory post-processing |
| [`intermediate_MissionSequence_and_FiniteBurn.ipynb`](tutorials/intermediate_MissionSequence_and_FiniteBurn.ipynb) | Mission sequence with chained propagation arcs and finite burns |
| [`intermediate_Measurements.ipynb`](tutorials/intermediate_Measurements.ipynb) | Simulating and computing radiometric and optical measurements |
| [`intermediate_nPlateModel_and_Attitude.ipynb`](tutorials/intermediate_nPlateModel_and_Attitude.ipynb) | N-plate SRP model with CK-frame rotating panels and spacecraft attitude |
| [`intermediate_BPlane.ipynb`](tutorials/intermediate_BPlane.ipynb) | B-plane targeting — OSIRIS-REx 2017 Earth gravity assist example |

### Advanced
| Tutorial | Description |
|---|---|
| [`advanced_IdealMSR_BatchOD.ipynb`](tutorials/advanced_IdealMSR_BatchOD.ipynb) | End-to-end batch OD (LSB, SRIFB) with simulated measurements, process noise, and measurement editing |
| [`advanced_IdealMSR_SequentialOD.ipynb`](tutorials/advanced_IdealMSR_SequentialOD.ipynb) | Sequential OD (LKF, SRIF, RTS smoother) with SNC/DMC process noise and consider parameters |
| [`advanced_RealMSR_MediaCorrections.ipynb`](tutorials/advanced_RealMSR_MediaCorrections.ipynb) | OSIRIS-REx real data — tropospheric/ionospheric media corrections and DSN ramp table management |
| [`advanced_RealMSR_OSIRIS_REx_OD.ipynb`](tutorials/advanced_RealMSR_OSIRIS_REx_OD.ipynb) | OSIRIS-REx two-pass batch OD with real DSN radiometric tracking data |

---

## Testing

Scarabaeus ships a multi-tier pytest suite under [`tests/`](tests/):

```bash
pytest tests/python_tests/unit_testing/        # fast, isolated unit tests
pytest tests/python_tests/integration_testing/ # end-to-end scenarios
pytest                                          # full suite with coverage report
```

Coverage and JUnit XML reports are written to `tests/python_tests/reports/`.

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and the [developer guide](https://ccar-orcca.github.io/scarabaeus-docs/development/development_frontpage.html) before opening a pull request.

### Reporting issues

Open a GitHub Issue and include: a short title, steps to reproduce (minimal working example preferred), expected vs. actual behaviour, and your Scarabaeus version, Python version, and OS. For security vulnerabilities, contact the maintainers directly.

### Submitting code

1. Fork the repository and set up the [developer environment](#developer-installation) on your fork.
2. Branch from `main` with a descriptive name, e.g. `feature/add-xyz` or `fix/filter-bug`.
3. Follow the **NumPy docstring convention** for all public API.
4. Run the test suite and ensure all tests pass: `pytest`.
5. Open a PR against the `main` branch with a clear description of what changed and why. Reference related Issues (e.g. `Closes #42`).

> Jupyter notebook outputs are stripped automatically by the `nbstripout` pre-commit hook — do not commit notebooks with cell outputs.

---

## Citing Scarabaeus

If you use Scarabaeus in academic work, please cite the paper:

```bibtex
@inproceedings{scarabaeus2026,
  author    = {McMahon, J. and Fereoli, G. and Wolf, T. and Ellis, Z. and Aldhanhani, B. and Kuleib, M. and Frank, W. and Pugliatti, M. and Almashjari, M. and Knittel, J.},
  title     = {Scarabaeus: An Open-Source Tool for Interplanetary Spacecraft Navigation},
  booktitle = {30th International Symposium on Space Flight Dynamics (ISSFD)},
  address   = {Toulouse, France},
  month     = jun,
  year      = {2026},
  url       = {https://www.researchgate.net/publication/404941039_Scarabaeus_An_Open-Source_Tool_for_Interplanetary_Spacecraft_Navigation},
}
```

---

## License

Scarabaeus is distributed under the [ISC License](LICENSE).

Copyright (c) 2026, Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder.

Some files are not covered by the ISC License and are included only so that the tutorials, tests and documentation work out of the box: the vendored numpydoc Sphinx extension (BSD-2-Clause, see its `LICENSE.txt`), NASA/JPL NAIF kernels and Deep Space Network tracking and media calibration data, and the University of Colorado Boulder logos. Each stays under its owner's terms; the logos are trademarks and may not be used to suggest endorsement.
