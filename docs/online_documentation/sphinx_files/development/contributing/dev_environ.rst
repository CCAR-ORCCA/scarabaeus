.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC
.. meta::
    :description lang=en:
        Tutorial for Scarabaeus developers to set up their development environment
    :keywords:
        Scarabaeus, developer guide, set up dev environment

.. _devenviron:

.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

.. create a role that makes bolded text colored blue as well
.. raw:: html

    <style> .bold {font-weight: bold; color:rgb(2 119 189)} </style>

.. role:: bold

=======================
:gold:`Setting Up and Using the Development Environment`
=======================
Last revised on 2026 MAY 28 by Z. Ellis.

.. note:: The following covers setting up a development environment for Scarabaeus. If you are simply using Scarabaeus, see the :ref:`User Guide <user>`.

Developers on the Scarabaeus team use a few more tools in addition to those included in the distributed package to support running tests, generating documentation, writing tutorials, and compiling source code. This document will guide a developer through setting up the development environment necessary for these tasks.

-------------------------
:bold:`Prerequisites`
-------------------------
Most of the development team uses VSCode as their IDE because of its extension ecosystem. Below are the recommended extensions for development:

- **Python:** provides language support for Python
- **Rust Analyzer:** provides language support for Rust
- **GitLens:** most of the development team uses GitLens for managing version control
- **Jupyter:** SCB utilizes Jupyter Notebooks for all tutorials

-------------------------
:bold:`Clone the Repository`
-------------------------
Clone the Scarabaeus repository. THIS WILL CHANGE WITH OPEN SOURCE?

-------------------------
:bold:`Set Up A Virtual Environment`
-------------------------
Before installing the developer environment, we'll need to create a virtual environment to place hold it. In your terminal, run:

.. code-block:: console

    python3 -m venv .venv

And activate it with:

.. tabs::

   .. group-tab:: Mac/Linux

      .. code-block:: console

        source .venv/bin/activate

   .. group-tab:: Windows

      .. code-block:: console

        .venv\Scripts\activate

You'll know it's activated when you see ``(.venv)`` in your terminal. To deactivate, run:

.. code-block:: console

    deactivate

-------------------------
:bold:`Install the Development Environment`
-------------------------
With the virtual environment activated, run the following command to install the development environment:

.. code-block:: console

    pip install -e . --group dev

.. note:: Sometimes pip might not recognize the ``--group`` command. This is usually fixed by updating with ``python -m pip install --upgrade pip``

This will install more dependencies than the standard installation, which include:

- `maturin <https://github.com/pyo3/maturin>`__: build Rust code and bind it to Python
- `pre-commit <https://pre-commit.com/>`__: manage Scarabaeus Git hooks
- `nbstripout <https://github.com/kynan/nbstripout>`__: remove cell results from Jupyter Notebooks when pushing to Git
- `numpydoc <https://numpydoc.readthedocs.io/en/latest/>`__: formatting for documentation generation
- `pytest <https://docs.pytest.org/en/stable/>`__: testing suite architecture

The following sections require the packages installed with the dev environment to function.

-------------------------
:bold:`Build From Source`
-------------------------
See :ref:`Building From Source <buildsource>`.

.. _dev-commit-hooks:
-------------------------
:bold:`Enable Git Commit Hooks`
-------------------------
The first time you clone the repository or create/pull a new branch, in the terminal run:

.. code-block:: console

    pre-commit install

This will add Scarabaeus' development hooks to your Git configuration, acting as an extra layer of quality assurance when you push a commit. If the `pre-commit` command is not found, ensure that you've installed the [Scarabaeus dev environment](#set-up-scarabaeus-dev-environment).

The hooks we use are:

- :bold:`nbstripout:` Jupyter Notebooks are a powerful tool for :ref:`writing tutorials <tutorialshowto>`, but they also produce metadata 
  in their cells when run that we don't want to track on git. We use nbstripout to automate this removal.
- :bold:`reuse:` We use reuse to ensure that all files committed to the repository are `SPDX compliant <https://spdx.dev/about/overview/>`_.