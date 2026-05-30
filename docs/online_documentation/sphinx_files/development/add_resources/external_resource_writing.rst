.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC
.. _writing-for-ext-res:
.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

.. create a role that makes bolded text colored blue as well
.. raw:: html

    <style> .bold {font-weight: bold; color:rgb(2 119 189)} </style>

.. role:: bold

==========================================
:gold:`Scarabaeus External Resource Guide`
==========================================
Last revised on 2026 MAY 28 by Z. Ellis.

--------------------
:bold:`Introduction`
--------------------
The external resource, where you are now, consists of two separate kinds of information:

#. Auto-Generated Script Documents in the Library Reference
#. Custom Non-Script Articles

Non-script articles are written by developers to provide information that doesn't already live within the code, like this 
document. Auto-generated script documents are created by reading docstrings already present in the source code. For more information 
about writing these docstrings, see the :ref:`Scarabaeus Coding Conventions <style-guide>`.

This document explains the steps necessary for ensuring that auto-generated documentation builds and how to 
deploy it online.

--------------------
:bold:`Building Docs`
--------------------
SCB online docs are built using Sphinx, which collects all of the docstrings and tutorials within SCB, organizes them in a structure defined in the online docs folder ``docs/online_documentation/sphinx_files/``, and renders it all as html, which is then hosted online.

In order to build the online docs, you'll need to:
- make sure in dev environment 
- make sure `pandoc <https://pandoc.org/>`_ is installed

Depending on your needs, there are two ways to build docs, outlined below.

Full Build
^^^^^^
For a completely clean build, run in your terminal:

.. code-block:: console

    python3 -m sphinx -b html docs/online_documentation/sphinx_files docs/online_documentation/_build

This will perform a build of the documentation from the ground up, which is important for major updates to the docs. Sometimes live build can mask errors, so if you're planning to push an update to the online docs, make sure that this builds.

Live Editing
^^^^^
When making changes to docstrings or updating the structure of the online docs, 

Additionally, you'll need to:
- make sure Esbonio and Live Preview extensions are both installed

Once you've done that, run:

.. code-block:: console
    
    sphinx-autobuild docs/online_documentation/sphinx_files docs/online_documentation/_build

This will start a process that continually rebuilds sphinx when it sees changes made to its code. In order to actually view these changes, right click ``docs/online_documentation/_build/index.html`` and select "Show Preview" (this will only appear if you have Live Preview installed). This will open a preview that automatically updates with whatever is produced by autobuild. When done editing, stop the autobuild process by pressing ``Ctrl`` + ``C`` in the terminal.

--------------------
:bold:`Deploying Docs`
--------------------
The above section only showed how to build a local copy of the docs. In order to actually deploy them online, we need to push them to the `docs repo <https://github.com/CCAR-ORCCA/scarabaeus-docs>`_. This is done automatically for major merges via CI/CD, but if you need to update the docs without making a push, run:

.. code-block:: console

    bash docs/deploy_docs.sh

This will perform a clean build of the documentations and push it to the repository with a time tag as the commit message. This resource is hosted 
using `GitHub Pages <https://docs.github.com/en/pages>`_ which updates fairly quickly. Within a few minutes of pushing to the repo, the website should 
update as well.