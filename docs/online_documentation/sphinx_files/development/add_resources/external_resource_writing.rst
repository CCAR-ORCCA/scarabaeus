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
Last revised on 2026 OCT 16 by G. Fereoli.

--------------------
:bold:`Introduction`
--------------------
The external resource, where you are now, consists of two separate kinds of information:

#. Auto-Generated Script Documents in the Library Reference
#. Custom Non-Script Articles

Non-script articles are written by developers to provide information that doesn't already live within the code, like this 
document. Auto-generated script documents are created by reading docstrings already present in the source code. For more information 
about writing these docstrings, see the :ref:`Scarabaeus Coding Conventions <style_guide>`.

This document explains the steps necessary for ensuring that auto-generated documentation builds and how to 
deploy it online.

---------------------
:bold:`Building Docs`
---------------------
SCB online docs are built using Sphinx, which collects all of the docstrings and tutorials within SCB, organizes them in a structure defined in the online docs folder ``docs/online_documentation/sphinx_files/``, and renders it all as html, which is then hosted online.

In order to build the online docs, you'll need to:
- make sure in dev environment 
- make sure `pandoc <https://pandoc.org/>`_ is installed

Depending on your needs, there are two ways to build docs, outlined below.

Full Build
^^^^^^^^^^
For a completely clean build, run in your terminal:

.. code-block:: console

    sphinx-build -j auto -W --keep-going -E -a -b html docs/online_documentation/sphinx_files docs/online_documentation/_build

This will perform a build of the documentation from the ground up, which is important for major updates to the docs. Sometimes live build can mask errors, so if you're planning to push an update to the online docs, make sure that this builds.

If you want to do a single build but want to hold on to anything cached by Sphinx (this will run much faster), run:

.. code-block:: console

    sphinx-build -j auto -W --keep-going -b html docs/online_documentation/sphinx_files docs/online_documentation/_build


This **WILL NOT** re-run all code snippets and tutorials like the previous command does, only whatever you've changed. This is good for trouble shooting any tutorials or code snippets you've added, but is not sufficient for a full docs build before pushing it to public.

Live Editing
^^^^^^^^^^^^
When making changes to docstrings or updating the structure of the online docs, 

Additionally, you'll need to:
- make sure Esbonio and Live Preview extensions are both installed

Once you've done that, run:

.. code-block:: console
    
    sphinx-autobuild docs/online_documentation/sphinx_files docs/online_documentation/_build -o "-j auto -W --keep-going"

This will start a process that continually rebuilds sphinx when it sees changes made to its code. In order to actually view these changes, right click ``docs/online_documentation/_build/index.html`` and select "Show Preview" (this will only appear if you have Live Preview installed). This will open a preview that automatically updates with whatever is produced by autobuild. When done editing, stop the autobuild process by pressing ``Ctrl`` + ``C`` in the terminal.

---------------------
:bold:`Release Notes`
---------------------
Release notes follow the `keep a changelog <https://keepachangelog.com/en/1.1.0/>`_ format and are maintained in the ``CHANGELOG.md`` file. See keep a changelog for more info.

They're also automatically generated in Sphinx and added to the online resource from the ``CHANGELOG.md`` file whenever the docs are built with ``docs/build_docs_local.sh`` or deployed. In order to update them yourself before a manual Sphinx build, run:

.. code-block:: console

    python docs/online_documentation/sphinx_files/release_notes/release_gen.py

----------------------
:bold:`Deploying Docs`
----------------------
The above section only showed how to build a local copy of the docs. The online docs are the content of the `docs repo <https://github.com/CCAR-ORCCA/scarabaeus-docs>`_, hosted
using `GitHub Pages <https://docs.github.com/en/pages>`_. The maintainers deploy them after changes are merged into ``main`` of the `GitHub repository <https://github.com/CCAR-ORCCA/scarabaeus>`_: the deploy performs a clean build of the documentation from the public code, checks it and pushes it to the docs repo. Within a few minutes of the push, the website updates as well.

.. note::

    Contributors don't need to deploy anything: make sure the full build above passes before opening a pull request. Pushing to the docs repo needs write access to it; contact a maintainer if you need it.

------------------------
:bold:`Writing Articles`
------------------------
Note that any code snippets in an article should be run by Sphinx during build to ensure that the code we're providing functions the way that we intend it to. This is done through the ``.. plot::`` directive. See ``docs/online_documentation/sphinx_files/user_guide/quickstart/quickstart.rst`` for an example.