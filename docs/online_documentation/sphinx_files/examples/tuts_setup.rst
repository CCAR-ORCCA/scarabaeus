
.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC

.. meta::
    :description lang=en:
        Explanation of how supplementary data is handled and provided in Scarabaeus tutorials.
    :keywords:
        Scarabaeus, supplementary data, tutorial set up

.. _tuts-setup:

.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

===================================
:gold:`Scarabaeus Tutorials Primer`
===================================
Last revised on 2026 OCT 22 by Z. Ellis.

The Supplementary Folder
----
In the `tutorials` directory, you will see a folder called `supplementary`. It contains two folders, `data` and `supp_data`, in addition to a Python files. The purpose of this folder is to make data handling and plotting simpler in the tutorials so that they can more easily focus on using Scarabaeus. Contents of the `supplementary` folder are:

- `data/`: a folder to hold any files written in a tutorial
- `supp_data/`: a folder containing given data files either hosted directly in the SCB repo or downloaded from publically available sources (see `supp_data.py` below)
- `file_gen.py`: Python script to generate small files like a metakernel or N-plate coefficient configuration file
- `supp_data.py`: Python script to download publically available data to `supp_data/`
- `supp_plotting.py`: plotting functions used throughout the tutorials to reduce code in the tutorials themselves