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

------------------------------
:gold:`User Installation`
------------------------------

**1. Clone the repository**

.. code-block:: bash

   git clone <https://github.com/CCAR-ORCCA/scarabaeus>
   cd scarabaeus

**2. Create a virtual environment**

.. code-block:: bash

   # macOS / Linux
  python3 -m venv .venv
  source .venv/bin/activate

  # Windows
  python -m venv .venv
  .venv\Scripts\activate

**3. Install Scarabaeus**

Currently, Scarabaeus can only be installed in developer mode, see :ref:`the Dev Environment Guide <devenviron>` for more on how to do this. Developer mode includes all SCB functionality in addition to tools for developing tests and Rust binding.
