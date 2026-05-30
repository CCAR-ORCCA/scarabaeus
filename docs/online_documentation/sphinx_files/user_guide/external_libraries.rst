.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC
.. _external-libs:

.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

.. create a role that makes bolded text colored blue as well
.. raw:: html

    <style> .bold {font-weight: bold; color:rgb(2 119 189)} </style>

.. role:: bold

=================
:gold:`Libraries`
=================
Last revised by M. Pugliatti on 2025 FEB 4

One of the pillars of the software design principles of SCB is to not reinvent the wheel and to make use of existing libraries whenever possible. To that end, the following python, C++ libraries and software are extensively used by SCB.

------------------------------
:bold:`Geometry and Ephemeris`
------------------------------

- SpiceyPy: `Online documentation <https://spiceypy.readthedocs.io/en/main/index.html#>`__

- Trimesh: `Online documentation <https://trimesh.org/>`__

-------------------
:bold:`Integration`
-------------------

Two integrators are used in SCB: 

- DOP853 from the Scipy library: `Online documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.DOP853.html>`_

- PyASA (C++), an internal implementation of IAS15 from rebound: `Online documentation <https://rebound.readthedocs.io/en/latest/integrators/#ias15>`_

------------
:bold:`Math`
------------

- Numpy: `Online documentation <https://numpy.org/>`__

- SciPy: `Online documentation <https://scipy.org/>`__

- Eigen (C++): `Online documentation <https://eigen.tuxfamily.org/index.php?title=Main_Page>`__

- Armadillo (C++): `Online documentation <https://arma.sourceforge.net/>`__

----------------
:bold:`Plotting`
----------------

- Matplotlib: `Online documentation <https://matplotlib.org/>`__

------------------------------------------
:bold:`Integrated Tests and Documentation`
------------------------------------------

- PyTest: `Online documentation <https://docs.pytest.org/en/stable/>`__

- Sphinx: `Online documentation <https://www.sphinx-doc.org/en/master/>`__

- PyData `Online documentation <https://pydata-sphinx-theme.readthedocs.io/en/stable/>`__

- Napoleon: `Online documentation <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/>`__

- Hypothesis: `Online documentation <https://hypothesis.readthedocs.io/en/latest/>`__

----------------
:bold:`Database`
----------------

- PyMongo: to interact with the database `Online documentation <https://pymongo.readthedocs.io/en/stable/>`__

- Pandas: to interact with the database `Online documentation <https://pandas.pydata.org/docs/index.html>`__

-------------------------
:bold:`Others from ORCCA`
-------------------------

- DWN: Deep Whale Network, data-parser for ground-station tracking data (Release date: TBD)

- PyASA: IAS15 integrator (Release date: TBD)

-------------------------
:bold:`External Software`
-------------------------

- MongoDB: to handle the database `Online documentation <https://www.mongodb.com/>`__

- Visual Studio Code (VSC): code editor `Online documentation <https://code.visualstudio.com/>`__
