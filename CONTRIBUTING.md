<!-- SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder -->
<!-- SPDX-License-Identifier: ISC -->
# Contributing to Scarabaeus

Thank you for your interest in Scarabaeus! This document covers how to contribute code,
report issues, and get support.

## Reporting Issues and Bugs

Open a GitHub Issue and include:

- A short, descriptive title.
- Steps to reproduce the problem (minimal working example preferred).
- Expected vs. actual behaviour.
- Scarabaeus version, Python version, and OS.

For security vulnerabilities, please contact the maintainers directly rather than filing a
public issue.

## Requesting Features

Feature requests are also tracked via GitHub Issues. Describe the use case and why
existing functionality does not cover it.

## Contributing Code

### 1. Set up a development environment

Follow the [Developer Installation](docs/online_documentation/sphinx_files/user_guide/install.rst)
instructions (clone → venv → `pip install -e . --group dev` → `maturin develop` → `pre-commit install`).

For a complete guide see [DEVELOPMENT.md](DEVELOPMENT.md).

### 2. Branch naming

Branch from `develop` and use descriptive names, e.g. `feature/add-xyz`, `fix/filter-bug`.

### 3. Code style

- Follow the [style guide](docs/online_documentation/sphinx_files/dev_guide/style_guide.rst).
- Document all public classes and methods using **NumPy-style docstrings**.
- Do not commit Jupyter notebook outputs (`nbstripout` pre-commit hook enforces this).

### 4. Tests

Add or update tests in `tests/` for any changed behaviour:

```bash
pytest tests/unit_testing/       # fast, isolated
pytest tests/integration_testing/ # end-to-end scenarios
```

All tests must pass before a pull request will be merged.

### 5. Submit a pull request

- Open a PR against the `develop` branch.
- Write a clear description of what changed and why.
- Reference any related Issues (e.g. `Closes #42`).
- A maintainer will review and may request changes before merging.

## Seeking Support

- **Documentation**: [online docs](https://ccar-orcca.github.io/scarabaeus-docs/)
- **Tutorials**: the `tutorials/` directory in this repository
- **Questions**: open a GitHub Issue with the `question` label

## Code of Conduct

Contributors are expected to be respectful and professional in all project spaces.
Harassment or discrimination of any kind will not be tolerated.
