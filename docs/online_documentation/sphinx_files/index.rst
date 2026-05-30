.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC
.. meta::
    :description lang=en:
        Scarabaeus is an open-source navigation tool being developed by the Orbital Research 
        Cluster for Celestial Applications (ORCCA) laboratory at the University of Colorado Boulder. 

        Scarabaeus’ architecture is based on a modular, object-oriented, Python front end with a Rust backend for 
        computationally intensive components.

        - Measurement data-centric approach and organization.
        - All data and calculations are associated and carried out with attached units to ensure correctness by construction.
        - Integration with the open-source SPICE toolkit.
        - User interaction entirely through Python and/or graphical interfaces for improved usability. 

        Under the hood, Scarabaeus is designed for high-numerical accuracy propagation using state-of-the-art 
        integrators with a range of dynamic models including multi-body gravity, spherical harmonics or 
        polyhedral gravity, central body gravity, cannonball or plate-based solar radiation pressure, 
        and impulsive or finite burn thrust models.

        Scarabaeus is designed to process radiometric data as well as optical navigation measurements, 
        to answer the needs of future deep-space exploration missions. 
    :keywords:
        Scarabaeus, documentation, open-source, orbit determination, navigation, measurements, SPICE, ORCCA Labratory, CCAR, CU Boulder

:html_theme.sidebar_secondary.remove:

.. create a gold color role for headers
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

.. create a role that makes bolded text colored blue as well
.. raw:: html

    <style> .bold {font-weight: bold; color:rgb(2 119 189)} </style>

.. role:: bold

.. use relative path extension (ctrl+shift+h) to get paths to images easily
.. image:: ../../images/scarabaeus_logo_dark.png
   :class: only-dark
   :width: 250
   :align: right

.. image:: ../../images/scarabaeus_logo.png
   :class: only-light
   :width: 250
   :align: right

==============================
:gold:`Welcome to Scarabaeus!`
==============================
Scarabaeus is an open-source navigation tool being developed by the Orbital Research 
Cluster for Celestial Applications (ORCCA) laboratory at the University of Colorado Boulder. 

Scarabaeus’ architecture is based on a modular, object-oriented, Python front end with a Rust backend for 
computationally intensive components. The tool has been built with several core principles including: 

- Measurement data-centric approach and organization.
- All data and calculations are associated and carried out with attached units to ensure correctness by construction.
- Integration with the open-source SPICE toolkit.
- User interaction entirely through Python and/or graphical interfaces for improved usability. 

Under the hood, Scarabaeus is designed for high-numerical accuracy propagation using state-of-the-art 
integrators with a range of dynamic models including multi-body gravity, spherical harmonics or 
polyhedral gravity, central body gravity, cannonball or plate-based solar radiation pressure, 
and impulsive or finite burn thrust models.

Scarabaeus is designed to process radiometric data as well as optical navigation measurements, 
to answer the needs of future deep-space exploration missions. 

.. grid::

   .. grid-item-card::
      :text-align: center
      
      | :fa:`file-lines, fa-file-lines fa-3x`
      |
      | **User Guide**
      ^^^^^^^^^^^^^^^^

      Guides on how to install and get 
      started as well as information detailing the 
      use and purpose of the project. New to 
      Scarabaeus? Start here!
      ++++++++++++++++++++++++++++++++++++++++++++

      .. button-ref:: user-guide-front
         :ref-type: ref
         :color: info
         :align: center
         :click-parent:

         To the User Guide
   
   .. grid-item-card::
      :text-align: center
      
      | :fa:`file-code, fa-file-code fa-3x`
      |
      | **API Reference**
      ^^^^^^^^^^^^^^^^^^^

      Detailed descriptions of modules, classes, 
      and methods included in Scarabaeus. Includes 
      explanations for how each object functions
      and accepted input parameters.
      +++++++++++++++++++++++++++++++++++++++++++++

      .. button-ref:: api
         :ref-type: ref
         :color: info
         :align: center
         :click-parent:

         To the API Reference

.. grid::

   .. grid-item-card::
      :text-align: center
      
      | :fa:`terminal, fa-terminal fa-3x`
      |
      | **Learn Scarabaeus**
      ^^^^^^^^^^^^^^^^^^^^^^

      Line-by-line walkthroughs for 
      Scarabaeus tasks. Example 
      demos to show holistic Scarabaeus 
      functionality.
      +++++++++++++++++++++++++++++++++

      .. button-ref:: learn
         :ref-type: ref
         :color: info
         :align: center
         :click-parent:

         To Demos & Tutorials
   
   .. grid-item-card::
      :text-align: center
      
      | :fa:`code-commit, fa-code-commit fa-3x`
      |
      | **Developer Guide**
      ^^^^^^^^^^^^^^^^^^^^^

      Style guide, best practices, and 
      expected standards for developing 
      in Scarabaeus.
      +++++++++++++++++++++++++++++++++++++++

      .. button-ref:: dev-guide
         :ref-type: ref
         :color: info
         :align: center
         :click-parent:

         To the Developer Guide

----

|pic1|  |pic2|

.. |pic1| image:: ../../images/ccar_logo_w_name_transparent.png
   :width: 45%

.. |pic2| image:: ../../images/ORCCA_Logo_Reverse.png
   :width: 45%

.. toctree::
   :hidden:

   User Guide <user_guide/user_guide_frontpage.rst>
   API Reference <api_ref/api_ref_frontpage.rst>
   Examples <examples/examples_frontpage.rst>
   Development <development/development_frontpage.rst>
   About <about_scb/about_frontpage.rst>
   Release Notes <release_notes/release_frontpage.rst>