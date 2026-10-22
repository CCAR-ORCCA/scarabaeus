<!-- SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder -->
<!-- SPDX-License-Identifier: ISC -->
# Testing in Scarabaeus
Below is a quick guide to get the Scarabaeus testing suite running. For a complete guide, see [Testing in Scarabaeus](https://ccar-orcca.github.io/scarabaeus-docs/development/contributing/testing_in_scb.html).

### Configuring and Setting Up the Testing Suite<a name="conf"></a>
In order to easily use the testing suite, you'll need to first configure VSCode's testing window.

1. Press `Ctrl/Cmd` + `Shift` + `P`. The VSCode command palette will open.
2. Search for `Python: Configure Tests` and run the command.
3. Select the `pytest` framework.
4. Select the `tests` root directory.

A beaker icon should appear in the VSCode sidebar. Select it to open the testing window. For more information on setting up the testing window, see [VSCode's guide](https://code.visualstudio.com/docs/debugtest/testing).


### Running Tests<a name="conf"></a>
Now that the testing window is configured, you can hover over a test in the hierarchy and select the play button to the right of its name to run it. Tests run from the top down, so any tests categorized below the one you've selected will also be run.

If a test passes, a green check mark will appear beside it in the hierarchy. A failure will place a red X. Finally, some tests may be purposefully skipped, placing a gray arrow beside the test.

Additionally, a testing summary will print in the `TEST RESULTS` tab of the VSCode terminal.