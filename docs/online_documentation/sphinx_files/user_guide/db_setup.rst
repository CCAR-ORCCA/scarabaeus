.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC
.. meta::
    :description lang=en:
        Guide detailing how to install and access the MongoDB database for Scarabaeus.
    :keywords:
        MongoDB, database, Scarabaeus

.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

.. create a role that makes bolded text colored blue as well
.. raw:: html

    <style> .bold {font-weight: bold; color:rgb(2 119 189)} </style>

.. role:: bold

======================================
:gold:`MongoDB Installation and Setup`
======================================
Last revised by S. Pattamudu-Manoharan on 2025 FEB 11

--------------------
:bold:`Introduction`
--------------------

MongoDB is an optional installation that facilitates the use of the tool when working with a large amount of data.

---------------
:bold:`Windows`
---------------
1. **Download**: Visit the `MongoDB download page <https://www.mongodb.com/try/download/community>`_ and download the MSI installer for Windows.
2. **Install**: Run the MSI file and follow the installation prompts. Choose "Complete" setup type and ensure "Install MongoDB as a Service" is checked.
3. **Set Up Environment**: Add MongoDB to the system PATH:
   
   - Go to **Control Panel** → **System and Security** → **System** → **Advanced system settings** → **Environment Variables**.
   - Select **Path**, click **Edit**, then **New** and add:
     
     .. code-block:: text
     
        C:\Program Files\MongoDB\Server\<version>\bin

4. **Start MongoDB**:
   
   - Open **Command Prompt**.
   - Run:

     .. code-block:: sh
     
        mongod --version
     
   - Ensure MongoDB is installed properly.

-------------
:bold:`macOS`
-------------
Prerequisites
=============
Homebrew requires the Xcode command-line tools from Apple's Xcode.

Install Xcode Command-Line Tools
================================
- Run:

  .. code-block:: sh

     xcode-select --install

Install Homebrew
================
macOS does not include the Homebrew package manager by default.

- Install Homebrew using the official documentation: `Homebrew Documentation <https://brew.sh/#install>`_.

1. **Install MongoDB**:
   - Open **Terminal**.
   - Run:

     .. code-block:: sh
     
        brew tap mongodb/brew
        brew update
        brew install mongodb-community

   - Start MongoDB service:

     .. code-block:: sh
     
        brew services start mongodb/brew/mongodb-community

2. **(Optional) Install MongoDB Compass (Graphical Interface)**
    - Visit the `MongoDB Compass download page <https://www.mongodb.com/try/download/compass>`_ and download the DMG installer for macOS.
    - Run the DMG file and follow the installation prompts.

------------------------------------------------
:bold:`Setup LASP Credentials to Access MongoDB`
------------------------------------------------

This step is required only for members of the SCB team working with the MongoDB database.

Windows
=======
1. Open the **Start Menu** and search for **Environment Variables**.
2. Select **Edit the system environment variables**.
3. In the **System Properties** dialog, click **Environment Variables**.
4. Under **System variables** or **User variables**, click **New**.
5. Add the following details:
    - **Variable name**: ``LASP_USER_NAME``
    - **Variable value**: ``<Your Value>``
    - **Variable name**: ``LASP_PASSWORD``

    If your LASP_PASSWORD contains any shell special characters (also called *metacharacters*),
    you must escape them using a backslash (``\``) so the shell interprets them as literal characters.

    Common shell special characters include:
    ``$``, ``"``, ``'``, ``\``, ``!``, ``*``, ``&``, ``(``, ``)``, ``|``, ``;``, ``<``, ``>``, ``~``, and ````` ``.

    **Examples**:

    - If your password contains ``$``, write it as ``\$``
    - If it contains ``!``, write it as ``\!``
    - A backslash itself must be written as ``\\``
    - **Variable value**: ``<Your Value>``

6. Click **OK** to save changes.
7. Connect to the **LASP VPN**.
8. Make sure to add the ``db_config.json`` file to the config folder (It contains the DB Credentials - DB Name, HostName, URI, Port Number).


macOS/Linux
===========
1. Open a **Terminal**.
2. Edit your shell configuration file:

   - For **bash**, run:

     .. code-block:: sh
     
        sudo nano ~/.zshrc

3. Add the following lines at the end of the file:

   .. code-block:: sh

      export LASP_USER_NAME='YOUR_LASP_USER_NAME'
      export LASP_PASSWORD='YOUR_LASP_PASSWORD'

4. Save and exit:
   - Press **CTRL + O**, then **Enter** to save.
   - Press **CTRL + X** to exit.

5. Reload the profile to apply changes immediately:

   .. code-block:: sh

      source ~/.zshrc

6. Verify the variable is set (do not run while sharing screen):

   .. code-block:: sh

      echo $LASP_USER_NAME

7. Close this terminal and connect to the **LASP VPN**.

8. Make sure to add the ``db_config.json`` file to the config folder (It contains the DB Credentials - DB Name, HostName, URI, Port Number).
