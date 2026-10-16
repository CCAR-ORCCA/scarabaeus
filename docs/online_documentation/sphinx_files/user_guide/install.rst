.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC

.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

.. create a role that makes bolded text colored blue as well
.. raw:: html

    <style> .bold {font-weight: bold; color:rgb(2 119 189)} </style>

.. role:: bold

====================
:gold:`Installation`
====================

.. note:: Scarabaeus has not been added to PyPi yet. You will need to compile Rust source code in order for it to function correctly, see :ref:`the developer guide <devenviron>` for further instructions.

------------------------
:gold:`Requirements`
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Dependency
     - Version
     - Notes
   * - Python
     - 3.11+
     - Required
   * - NumPy
     - latest
     - Array mathematics
   * - SciPy
     - latest
     - Integrators and linear algebra
   * - SpiceyPy
     - latest
     - SPICE toolkit Python wrapper
   * - matplotlib
     - latest
     - Plotting utilities
   * - Rust / Cargo
     - 1.75+
     - **Developer builds only** — not required for users

The full dependency list is managed in ``pyproject.toml`` and is installed automatically by ``pip``.

------------------------------
:gold:`User Installation`
------------------------------

**1. Clone the repository**

.. code-block:: bash

   git clone <https://github.com/CCAR-ORCCA/scarabaeus>
   cd scarabaeus

**2. Create a virtual environment**

.. code-block:: bash

   python -m venv .venv
   # macOS / Linux
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate

**3. Install Scarabaeus**

.. .. code-block:: bash

..    pip install .

Currently, Scarabaeus can only be installed in developer mode, see :ref:`the Dev Environment Guide <devenviron>` for more on how to do this.

.. Once the package is published on PyPI, users will also be able to install with:

.. .. code-block:: bash

..    pip install scarabaeus

.. .. note::

..    **SPICE kernels** — Scarabaeus uses SPICE kernels for ephemeris, reference frames, and
..    spacecraft clock data. These are not distributed with the package. Load them in your
..    script with:

..    .. code-block:: python

..       import scarabaeus as scb
..       scb.SpiceManager.load_kernel_from_mkfile("path/to/metakernel.tm")

..    See `NAIF's kernel guide <https://naif.jpl.nasa.gov/naif/toolkit.html>`_ for details on
..    obtaining and configuring kernels.
