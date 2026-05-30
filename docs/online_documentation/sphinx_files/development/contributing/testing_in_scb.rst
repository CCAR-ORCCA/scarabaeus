.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC
.. meta::
    :description lang=en:
        Overview of testing set up in Scarabaeus.
    :keywords:
        Scarabaeus, testing

.. _testing-in-scb:

.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold


=================================
:gold:`Scarabaeus Testing Suite`
=================================
Last revised on 2026 MAY 28 by G. Fereoli.

Scarabaeus (SCB) uses `pytest <https://docs.pytest.org>`_ for automated testing. The test
suite lives in ``tests/`` at the repository root and is split into unit and system
tests. The system tests are further divided into functional, integration, V&V, and 
performance tests, which themselves include code profiling and code coverage.

.. code-block:: text

   tests/
   ├── unit_testing/          # isolated tests per class/module
   │   ├── body_tests/
   │   ├── dynamics_tests/
   │   ├── environment_tests/
   │   ├── guidance_tests/
   │   ├── measurements_tests/
   │   ├── orbitDetermination_tests/
   │   ├── spacecraft_tests/
   │   ├── timeAndFrame_tests/
   │   ├── units_tests/
   │   └── utils_tests/
   └── system_testing/
       ├── functional_testing/    # functional verification tests
       ├── integration_testing/   # end-to-end scenario tests
       ├── performance_testing/   # profiling and code coverage
       └── V&V                    # validation & verification tests

----------------------------
:gold:`Running All Tests`
----------------------------

From the repository root (with your virtual environment active):

.. code-block:: bash

   pytest tests/

To run only a specific tier:

.. code-block:: bash

   pytest tests/unit_testing/
   pytest tests/system_testing/integration_testing/

To run a single test file:

.. code-block:: bash

   pytest tests/unit_testing/body_tests/test_CelestialBody.py

-----
:gold:`Adding to the Testing Suite`
-----
Whenever you add any new functionality to Scarabaeus, you must also include this functionality in the testing suite. 

Depending on the complexity and depth of your addition, you may need to create multiple tests across different levels of the suite. For example, 
a simple new class that only holds information within itself would only require a unit test. However, a more complex new class that 
interacts with other SCB classes would require integration tests in addition to unit testing.

The following sections provide more information on each of the different levels of testing and their expected use cases.

-----
:gold:`Unit Tests`
-----
Unit tests verify the behaviour of individual Scarabaeus classes and methods in isolation.
Each test file maps to one class (e.g. ``test_CelestialBody.py`` tests :class:`~scarabaeus.CelestialBody`)
and lives under the matching module subdirectory of ``tests/unit_testing/``.

**Running unit tests:**

.. code-block:: bash

   pytest tests/unit_testing/

**Running a single module's tests:**

.. code-block:: bash

   pytest tests/unit_testing/body_tests/


^^^^^^
Writing a New Unit Test
^^^^^^
Whenever you create a new class or add a new method to an existing class, you must also include a test or tests 
for it in the testing suite:

1. Create ``tests/unit_testing/<module>_tests/test_ClassName.py``.
2. For new classes, ensure that initialization functions properly.

-----
:gold:`Integration Tests`
-----
Integration tests run end-to-end OD scenarios to verify that the full pipeline—propagation,
measurement simulation, filtering, and output—produces correct results. They live in
``tests/integration_testing/`` and typically require SPICE kernels and data files.

**Running integration tests:**

.. code-block:: bash

   pytest tests/integration_testing/

.. note::

   Integration tests load SPICE kernels via metakernel files stored in the test data
   directory. Per the `SPICE metakernel specification
   <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/kernel.html>`_, file paths
   within a metakernel cannot exceed 255 characters. If the full path to the test data
   directory is long, use SPICE's ``+`` line-continuation marker for kernel filenames
   that would otherwise exceed the limit.

Each integration test is self-contained and uses ``pytest-dependency`` markers to express
ordering constraints where one scenario must complete before another.