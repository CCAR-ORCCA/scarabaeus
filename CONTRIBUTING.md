<!-- SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder -->
<!-- SPDX-License-Identifier: ISC -->
# Contributing to Scarabaeus

Thank you for your interest in Scarabaeus! This document covers how to contribute code,
report issues, and get support.

## Reporting Issues and Bugs

Open a GitHub Issue and include:

- A short, descriptive title
- Steps to reproduce the problem (minimal working example preferred)
- Expected vs. actual behaviour
- Scarabaeus version, Python version, and OS

For security vulnerabilities, please contact the maintainers directly rather than filing a public issue.

## Requesting Features

Feature requests are also tracked via GitHub Issues. Describe the use case and why
existing functionality does not cover it.

## Contributing Code
### Branch naming

Branch from `main` and use descriptive names, e.g. `feature/add-xyz`, `fix/filter-bug`.

### Code style

- Follow the [style guide](docs/online_documentation/sphinx_files/development/contributing/style_guide.rst).
- Document all public classes and methods using **NumPy-style docstrings**.
- Ensure pre-commit hooks are applied.

### Tests

Add or update tests in `tests/` for any new or changed behaviour. See [Writing SCB Tests](https://ccar-orcca.github.io/scarabaeus-docs/development/contributing/testing_in_scb.html#adding-to-the-testing-suite) for more. All tests must pass before a pull request will be merged.

### Submit a pull request

- Open a PR from your fork against the `main` branch.
- Write a clear description of what changed and why.
- Reference any related Issues (e.g. `Closes #42`).
- A maintainer will review and may request changes before merging.

## Code of Conduct

Contributors are expected to be respectful and professional in all project spaces.
Harassment or discrimination of any kind will not be tolerated.
