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

**Scarabaeus (SCB)** is a Python and Rust library for spacecraft navigation and orbit determination (OD). It provides capabilities for:

- Building and propagating spacecraft trajectories with multiple force models
- Simulating and/or ingesting radiometric and optical measurements
- Running sequential and batch OD filters (LKF, SRIF, LSB, SRIFB, ...) and sampling their posteriors with MCMC
- Estimating dynamical, measurement, and consider parameters
- Performing maneuver planning and finite/impulsive burn modeling

To get started with Scarabaeus, follow the [Installation Guide](https://ccar-orcca.github.io/scarabaeus-docs/user_guide/install.html). Once SCB is set up, see [Scarabaeus Quickstart](https://ccar-orcca.github.io/scarabaeus-docs/user_guide/quickstart/quickstart.html) or dive deeper with [Scarabaeus Tutorials](https://ccar-orcca.github.io/scarabaeus-docs/examples/examples_frontpage.html).

---

## More Info
- Documentation: https://ccar-orcca.github.io/scarabaeus-docs/index.html
- Contributing: [Contributing](#contributing)
- Bug Reports: https://github.com/CCAR-ORCCA/scarabaeus/issues
- Citation: [Citing Scarabaeus](#citing-scarabaeus)


## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and the [developer guide](https://ccar-orcca.github.io/scarabaeus-docs/development/development_frontpage.html) before opening a pull request.

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
