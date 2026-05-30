.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC
.. meta::
    :description lang=en:
        Tutorial for Scarabaeus developers to build compiled Rust code from source.
    :keywords:
        Scarabaeus, developer guide, Rust, compiled code

.. _buildsource:

.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

.. create a role that makes bolded text colored blue as well
.. raw:: html

    <style> .bold {font-weight: bold; color:rgb(2 119 189)} </style>

.. role:: bold

=======================
:gold:`Building From Source`
=======================
Last revised on 2026 MAY 28 by Z. Ellis.

Scarabaeus utilizes Rust code for computationally expensive tasks. If you are developing within SCB or attempting a local build, this code must be compiled. The following will guide a developer through the steps required to compile Rust code and bind it to SCB's Python front end.

Rust setup instructions adapted from `Visual Studio Code <https://code.visualstudio.com/docs/languages/rust>`_ for Scarabaeus-specific development.

-------------------------
:bold:`Install Rust`
-------------------------
Follow the instructions provided by the `rustup installer <https://rustup.rs/>`_, which supports installation for Windows, macOS, and Linux. Once Rust is installed, restart any terminal/Command Prompt and VS Code instances.

Once you've installed Rust and restarted all terminals, check to make sure everything is installed by typing in your terminal:

.. code-block:: console

    rustc --version

This will output the version of the Rust compiler if it's installed.

-------------------------
:bold:`Install and Link rust-analyzer Extension`
-------------------------
The `rust analyzer extension <https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer>`_ makes writing Rust code in VS Code significantly easier by providing IDE functionalities like auto-complete, inline errors, and auto-formatting. Install it via the VS Code Extensions tab.

Due to the structure of the `src` folder, we'll have to manually link SCB's Rust module ``src/scarabaeus_rust/`` to rust-analyzer in its settings:

1. ``Ctrl`` + ``Shift`` + ``P`` on Windows/Linux or ``Cmd`` + ``Shift`` + ``P`` on Mac
2. Search for `Preferences: Open User Settings` and open it
3. In User Settings, search for `rust-analyzer: Linked Projects` and select `Edit in settings.json`
4. In the `rust-analyzer.linkedProjects` field, add:
   ``["src/scarabaeus_rust/Cargo.toml"]``
This will allow rust-analyzer to link to our Cargo file. Without it, we won't be able to utilize the IDE functionalities provided by rust-analyzer.

-------------------------
:bold:`Build Rust Source`
-------------------------
The `SCB dev environment <devenviron>`_ includes `maturin <https://github.com/pyo3/maturin>`__, which will allow us to build our Rust binaries as a Python package so that they're callable within SCB's frontend.
SCB's Rust source code is separated from Python source code, placed in the ``src/scarabaeus_rust`` folder. To compile this code, run in your terminal:

.. code-block:: console

    (.venv) maturin develop

This will build and bind the Rust code to the Python component of SCB, allowing compiled Rust code to be called within Python. This will generate a few build artifacts that will clutter the file explorer. The next step will help you configure VSCode to hide these artifacts.

-------------------------
:bold:`Hide Build Artifacts`
-------------------------
After running ``maturin develop``, two files will be generated in the ``src/scarabaeus`` path: 

1. a decoder ring that provides a map from compiled Rust code to the terminal so that errors can be displayed
2. the actual compiled Rust code

Depending on your OS, these two files will take different forms. Regardless, they're not files that we will need access to 
and clutter the folder structure, so we'll exclude them from VSCode's explorer:

1. ``Ctrl`` + ``,`` on Windows/Linux or ``Cmd`` + ``,`` on Mac
2. Search for `Files:Exclude` and select ``Add Pattern``

For Windows, add the following patterns:
    - ``**/*.pdb`` to hide the decoder ring
    - ``**/*.pyd`` to hide the compiled Rust binaries

For Linux, add the following patterns:
    - ``**/*.dwp`` to hide the decoder ring
    - ``**/*.so`` to hide the compiled Rust binaries
  
For Mac, add the following patterns:
    - ``**/*.dSYM`` to hide the decoder ring
    - ``**/*.so`` to hide the compiled Rust binaries

-------------------------
:bold:`Expose (New) Rust Code to Python`
-------------------------
If you are modifying or writing any Rust code, you'll need to define and/or update its corresponding stub in the stub file ``src/scarabaeus/scarabaeus_rust.pyi`` before building with ``maturin develop``. This stub file provides the docstrings used for documentation generation as well as inline definitions for any Rust code bound to Python.

Additionally, if you've created a brand new class (Rust struct bound to Python class), youll need to add it to the ``scarabaeus_rust`` package in the lib file ``src/scarabaeus_rust/src/lib.rs`` using:

.. code-block:: rust

    m.add_class::<ClassName>()?;

Note that the `<>` are part of the code, not to denote an insertion.