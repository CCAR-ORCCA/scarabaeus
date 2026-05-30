.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC
.. meta::
    :description lang=en:
        Setup and installation guide for Scarabaeus.
    :keywords:
        Scarabaeus, setup guide, install

.. _scarabaeus-setup:

.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

.. create a role that makes bolded text colored blue as well
.. raw:: html

    <style> .bold {font-weight: bold; color:rgb(2 119 189)} </style>

.. role:: bold

=====================================
:gold:`Install and Set Up Scarabaeus`
=====================================
Last revised by M. Pugliatti on 2025 FEB 4

---------------------
:bold:`Prerequisites`
---------------------
There are a few additional tools that must be intalled before installing Scarabaeus.

Python
======
Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.

Install the recomemended version of Python - `3.11.9 <https://www.python.org/downloads/release/python-3119/>`_

Git
===
Git is a free and open source distributed version control system designed to handle everything from small to very large projects with speed and efficiency. Install 
Git if your machine does not already have it. Most Mac and Linux machines will have Git already installed.

Install Git - `Git Download <https://git-scm.com/downloads/>`_

Integrated Development Environment (IDE)
========================================
Use any IDE or text editor you like. We encourage the use of an IDE that has add-ons to make development a bit easier. Most developers on the team use VSCode for its large 
ecosystem of powerful extensions. For SCB, we recommend using **VSCode** as your IDE, with the following extensions:

- GitLens - `link <https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens>`_
- SonarQube - `link <https://marketplace.visualstudio.com/items?itemName=SonarSource.sonarlint-vscode>`_
.. - autoDocstring - `link <https://marketplace.visualstudio.com/items?itemName=SonarSource.sonarlint-vscode>`_
- Python - `link <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`_
- Pylance - `link <https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance>`_

-----------------------------------
:bold:`Clone Scarabaeus Repository`
-----------------------------------

Clone the Scarabaeus repository to your local machine. You will need to setup some form of secure access between your machine and Gitlab before you are able to do this. The 
options for secure access are below. We highly recommend using SSH.

To setup a secure access method:

- **SSH - Recommended**:
    - On Gitlab, go to your **Preferences**. This can be done by clicking on your **user icon** in the top right corner, and the selecting **Preferences**.
    - Navigate to the menu on the far left of the screen and click on the **SSH Keys** option. This will have a key icon to the left of the text.
    - Follow the instructions on this page for adding an SSH key. The page contains links to documentation for how to set up the private and public key pair. We understand this process can be somewhat confusing so if you have any trouble please reach out to a fellow developer for assistance.
    - General GitLab SSH documentation can be found on the `GitLab SSH Docs <https://docs.gitlab.com/ee/user/ssh.html>`_

Alternatively: 

- **HTTPS - Not Recommended**:
    - This process involves generating Gitlab Tokens. If you would like to use this method please reference the Gitlab documentation for a how-to.

Once you have a secure access method set up you can clone the Scarabaeus repository. You may do this via a terminal or your IDE. We recommend using your IDE as it makes 
things slightly simpler but this is up to you. The following instructions will be for cloning via SSH.

**Cloning via VSC**:

- Navigate to the Scarabaeus project on Gitlab and click on the Clone button shown in the image below.
- Copy the link under "Clone with SSH" (the one that starts with git@).
- Now navigate to VSCode, open a new window, and clock on the Clone Git Repository... option under the Start menu (center left of the window). Your screen should look like the image below.
- Next, paste the link you copied from Gitlab into the bar at the top of the VSCode window. From there follow the prompts to clone the repository.

|pic1|  |pic2|

.. |pic1| image:: ../../../images/readme_images/clone_repo_screenshot.png
   :width: 45%

.. |pic2| image:: ../../../images/readme_images/VSCode_New_Window.png
   :width: 45%

**Cloning via Command Line**:

In the terminal type:

.. code-block::

    git clone git@orccagitlab.colorado.edu:javi9068/scarabaeus.git

Refer to `GitLab CLI Cloning Guide <https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html>`_ for extra documentations on cloning via command line.

-------------------------------------------------
:gold:`Add Scarabaeus Package to Python Path`
-------------------------------------------------

Steps to add the ``scarabaeus/src`` folder path to the Python path for both macOS and Windows.

**Steps for macOS**

1. Open Terminal.
2. Run the following command to edit your ``.zshrc`` file:

    .. code-block::

        sudo nano ~/.zshrc

    - Enter your PC's password

3. In the ``.zshrc`` file, add the following line to set the Python path:

    .. code-block::

        export PYTHONPATH=$PYTHONPATH:/path/to/scarabaeus/src

    - Replace ``/path/to/scarabaeus/src`` with the absolute path of the **src folder** (e.g., ``/Users/yourusername/path/to/scarabaeus/src``).

4. Press `Ctrl + O` and press `Enter`, then `Ctrl + X` to exit the editor and save the changes.

5. Apply the changes by running:

    .. code-block::

        source ~/.zshrc

6. Close this terminal and connect to the **LASP VPN**.

7. Make sure to add the ``db_config.json`` file to the config folder (It contains the DB Credentials - DB Name, HostName, URI, Port Number).

**Steps for Windows**

1. Press ``Win + R`` to open the Run dialog.
2. Type ``sysdm.cpl`` and press Enter to open the System Properties window.
3. In the System Properties window, click on the "Advanced" tab.
4. Click on the "Environment Variables" button at the bottom.
5. In the Environment Variables window, under the "User variables" section, click on "New..." to create a new environment variable.
6. In the "Variable name" field, enter ``PYTHONPATH``.
7. In the "Variable value" field, enter the absolute path of the src folder (e.g., ``C:\path\to\SCB\src``).
8. Click OK to save the new environment variable.
9. Close the remaining windows by clicking OK.

--------------------------------------
:bold:`Creating a Virtual Environment`
--------------------------------------
To create a virtual environment (venv) you have two options: 
 
From Terminal
=============

.. code-block::

   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux

To deactivate:

.. code-block::

   deactivate

In VSC
======

- Go to the search bar on top and click it
- Select **Show and Run Commands \>**
- Select **Python: Create Environment...**
- Select **Venv**
- Either **Use Existing** or **Delete and Recreate**
- Select the python interpreter as **Python 3.11.9**
- Select ``requirements.txt``

and the venv will be generated following the list of libraries specified in ``requirements.txt``

----------------------------
:bold:`Library Installation`
----------------------------

From Terminal
=============

Install required packages:

.. code-block::

   pip install -r requirements.txt

In VSC
======
    
Libraries are already installed in the last step of the venv generation.

External
========

Additionally, othe rdependencies (DWN, IAS15) needs to be installed manually from terminal

Install IAS15 integrator dependencies:

.. code-block::

   git submodule update --init
   pip install pyasa/.

On Windows, manually copy the shared library:

.. code-block::

   cp <venv>/Lib/site-packages/bin/pyasalib.dll <venv>/Lib/site-packages/pyasa/

where `<venv>` is your virtual environment directory.
