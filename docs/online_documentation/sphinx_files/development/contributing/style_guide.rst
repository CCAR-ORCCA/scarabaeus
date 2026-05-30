.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC
.. _style_guide:
.. create a gold color role for headers
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

.. create a role that makes bolded text colored blue as well
.. raw:: html

    <style> .bold {font-weight: bold; color:rgb(2 119 189)} </style>

.. role:: bold

=====================================
:gold:`Scarabaeus Coding Conventions`
=====================================
Last revised on 2026 MAY 28 by Z. Ellis.

--------------------
:bold:`Introduction`
--------------------
This document describes conventions for developing in the Scarabaeus (SCB) library. Much of these 
conventions are derived from Python's `PEP 8 style guide <https://peps.python.org/pep-0008/>`_; 
wherever some convention is not explicitly stated in the following guide, refer to it for guidance.

The coding guidelines layed out below exist to serve two purposes:

- to ensure a coherent, readable protocol while developing for Scarabaeus
- to make maintaining Scarabaeus's documentation as easy as possible

Each section of this document explains a section of a given template script, ``TemplateClass.py``, available to view 
:doc:`here <show_tmplt_clss>`, as well as in Scarabaeus itself under ``docs/TemplateClass.py``.
Following along in order should explain the purpose and formatting of each section, allowing us to build 
the template over the course of this reading.

One final note to remember from the PEP 8 style guide:
    
    "... code is read much more often than it is written... know when to be inconsistent - sometimes style guide recommendations just aren't applicable. When in doubt, use your best judgment. Look at other examples and decide what looks best."

----

---------------------
:bold:`General Rules`
---------------------
Before we go line by line through the example script, let's quickly cover a few general rules that you should follow when writing code in Scarabaeus. You'll 
notice throughout the example script that these rules are followed:

- :bold:`Commenting:` Ensure that your comments are frequent and descriptive.
- :bold:`Type-Hinting:` Type-hinting is an important extension of commenting. It tells both users and developers what should be going where. 
  It appears in the online documentation (where you are right now), as well as in the code through autocompletion tools 
  like `Intellisense <https://code.visualstudio.com/docs/editor/intellisense>`_. Make sure to type-hint both arguments and returns. You 
  can see how this is done throughout the Code Layout section.
- :bold:`Naming Conventions:` Both method and variable names should be written in snake_case.

----

-------------------
:bold:`Code Layout`
-------------------
In the following sections of the Code Layout, we will walk through each component of a template file ``TemplateClass.py`` and explain the 
reasoning behind the decisions that have been made. Note that the entire file is available to view :doc:`here <show_tmplt_clss>`, as well as in Scarabaeus itself 
under ``docs/TemplateClass.py`` and contains more comments to supplement it. Many of these comments have been removed in the following code 
snippets for brevity.

SPDX License Identifier
===========
You'll notice that every file in Scarabaeus begins with these lines:

.. code-block:: python

    # SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
    # SPDX-License-Identifier: ISC

This is a license identifier necessary to maintain `SPDX compliance <https://spdx.dev/about/overview/>`_. It is required at the top of every file, and is checked 
by :ref:`our commit hooks <dev-commit-hooks>`. Without it, you will not be allowed to push any commits.

See the :doc:`Template Class <show_tmplt_clss>` for further examples.

-------------------
:bold:`Docstrings`
-------------------
Docstrings in SCB follow the [numpydocs format](https://numpydoc.readthedocs.io/en/latest/example.html). When writing them, make sure to adhere to this standard. See any SCB module for examples. A few more notes:

- When writing equations, it is encouraged to write them with the rst `.. math::` directive.
- Citations are supported in any context.
- Docstrings are collected and displayed by `Intellisense <https://code.visualstudio.com/docs/editing/intellisense>`_ when using VSCode, 
  so you can quickly check that your docstring works and is formatted correctly by hovering over your class or function with your mouse.