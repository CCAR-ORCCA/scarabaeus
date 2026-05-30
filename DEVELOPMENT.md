<!-- SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder -->
<!-- SPDX-License-Identifier: ISC -->
# Scarabaeus Developer Setup Guide
> **IMPORTANT:** The following covers how to build compiled code, run tests, configure Git commit hooks, and write/maintain documentation for Scarabaeus. This is only necessary if you are working as a developer. If you are running Scarabeus as a general user, see [Getting Started](README.md#getting-started).

Setting up to devlop for Scarabaeus (SCB) includes a few more steps, outlined below. This guide assumes that you have followed the neccessary steps to get SCB up and running outlined in [Getting Started](README.md#getting-started).

**Contents:**
- [Scarabaeus Dev Environment](#set-up-scarabaeus-dev-environment)
- [Compiled Code](#compiled-code)
- [Running and Writing Tests](#scarabaeus-testing-suite)
- [Commit Hooks](#commit-hooks)
- [Documentation](#documentation)

# Set Up Scarabaeus Dev Environment
There are a few tools that developers will need while working on SCB that aren't required for a general user. In order to use these, we'll need to install SCB as an editable package with a few more developer tools. In your virtual environment, run:
```
(.venv) pip install -e . --group dev
```
The developer environment includes the following additional dependencies:

**Rust Compilation**:
- [maturin](https://github.com/pyo3/maturin): build Rust code and bind it to Python

**Git Filtering**:
- [pre-commit](https://pypi.org/project/pre-commit/): manage Scarabaeus Git hooks
- [nbstripout](https://github.com/kynan/nbstripout): remove cell results from Jupyter Notebooks when pushing to Git

**Documentation**:
- [numpydoc](https://numpydoc.readthedocs.io/en/latest/): formatting for documentation generation

**Testing**:
- [pytest](https://docs.pytest.org/en/stable/): testing suite architecture

The following sections require the packages installed with the dev environment to function.

# Compiled Code
Scarabaeus utilizes Rust code for computationally expensive tasks. In order to develop within SCB, this code must be compiled. This section will guide a developer through the steps required to compile Rust code and bind it to SCB's Python front end.

Rust setup instructions adapted from [Visual Studio Code](https://code.visualstudio.com/docs/languages/rust) for Scarabaeus-specific development.

## 1. Install Rust
Follow the instructions provided by the [rustup installer](https://rustup.rs/), which supports installation for Windows, macOS, and Linux. Once Rust is installed, restart any terminal/Command Prompt and VS Code instances.

Once you've installed Rust and restarted all terminals, check to make sure everything is installed by typing:
```
rustc --version
```
This will output the version of the Rust compiler if it's installed.

## 2. Install and Link rust-analyzer Extension
The [rust analyzer extension](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer) makes writing Rust code in VS Code significantly easier by providing IDE functionalities like auto-complete, inline errors, and auto-formatting. Install it via the VS Code Extensions tab.

Due to the structure of the [src](src/) folder, we'll have to manually link [SCB's Rust module](src/scarabaeus_rust/) to rust-analyzer in its settings:

1. `Ctrl` + `Shift` + `P` on Windows/Linux or `Cmd` + `Shift` + `P` on Mac
2. Search for `Preferences: Open User Settings` and open it
3. In User Settings, search for `rust-analyzer: Linked Projects` and select `Edit in settings.json`
4. In the `rust-analyzer.linkedProjects` field, add
    ```
    ["src/scarabaeus_rust/Cargo.toml"]
    ```
This will allow rust-analyzer to link to our Cargo file. Without it, we won't be able to utilize the IDE functionalities provided by rust-analyzer.

## 3. Build `scarabaeus_rust`
The [SCB dev environment](#set-up-scarabaeus-dev-environment) includes [maturin](https://github.com/pyo3/maturin), which will allow us to build our Rust binaries as a Python package so that they're callable within SCB's frontend.
SCB's Rust source code is separated from Python source code, placed in the [src/scarabaeus_rust](src/scarabaeus_rust/) folder. To compile this code, run in your terminal:
```
(.venv) maturin develop
```
This will build and bind the Rust code to the Python component of SCB, allowing compiled Rust code to be called within Python. If you are working solely on Python code (under [src/scarabaeus](src/scarabaeus/)), you will only need to run this the first time you've cloned the Rust code. However, if you are developing new Rust code, continue to the next step.

## 4. Expose Rust to Python (Writing Rust Code Only)
If you are modifying or writing any Rust code, you'll need to define and/or update its corresponding stub in the [stub file](src/scarabaeus/scarabaeus_rust.pyi) before building with `maturin develop`. This stub file provides the docstrings used for documentation generation as well as inline definitions for any Rust code bound to Python.

Additionally, if you've created a brand new class (Rust struct bound to Python class), youll need to add it to the `scarabaeus_rust` package in the [lib file](src/scarabaeus_rust/src/lib.rs) using:
```
m.add_class::<ClassName>()?;
```
Note that the `<>` are part of the code, not to denote an insertion.

# Commit Hooks
Jupyter Notebooks (see [Tutorials](#tutorials)) store their cell information as metadata when run. We don't want to track this info on git, so we us nbstripout to remove this data when tracking commits. This is handled by the [.pre-commit-config.yaml](.pre-commit-config.yaml) file, you just need to install it:

Whenever you create or pull a branch for the first time, in the terminal run:
```
(.venv) pre-commit install
```
This will add Scarabaeus' development hooks to your Git configuration, acting as an extra layer of quality assurance when you push a commit. If the `pre-commit` command is not found, ensure that you've installed the [Scarabaeus dev environment](#set-up-scarabaeus-dev-environment).