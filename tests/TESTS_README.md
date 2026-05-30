# Testing in Scarabaeus

* [Pytest](#pytest)
    - [Configuration and Setup](#conf)
    - [Running Tests](#oper)

* [Unit Tests](#unit)
    - [Purpose](#unitpurpose)

* [Integration Tests](#integration)
    - [Purpose](#integpurpose)
    - [Using Data File](#integdata)

## Pytest<a name="pytest"></a>

### Configuring and Setting Up the Testing Suite<a name="conf"></a>
In order to easily use the testing suite, you'll need to first configure VSCode's tesing window.

1. Press `Ctrl/Cmd` + `Shift` + `P`. The VSCode command palette will open.
2. Search for `Python: Configure Tests` and run the command.
3. Select the `pytest` framework.
4. Select the `tests` root directory.

A beaker icon should appear in the VSCode sidebar. Select it to open the testing window. There should a series of drop down lists with the hierarchy:

```
tests
├─ integration_testing
│  ├─ ...
├─ unit_testing
│  ├─ ...
```

The following sections will detail how to use this testing window as well as the purpose of the different kinds of tests. For more information on setting up the testing window, see [VSCode's guide](https://code.visualstudio.com/docs/debugtest/testing).


### Running Tests<a name="conf"></a>
Now that the testing window is configured, you can hover a test in the hierarchy and select the the play button to the right of its name to run it. Tests run from the top down, so any tests categorized below the one you've selected will also be run.

If a test passes, a green check mark will appear beside it in the hierarchy. A failure will place a red X. Finally, some tests may be purposefully skipped, placing a gray arrow beside the test.

Additionally, a testing summary will print in the `TEST RESULTS` tab of the VSCode terminal.

## Scarabaeus Unit Tests<a name="unit"></a>

### The Purpose of Unit Tests<a name="unitpurpose"></a>

## Scarabaeus Integration Tests<a name="integration"></a>

### The Purpose of Integration Tests<a name="integpurpose"></a>

### Data Files for Integration Tests<a name="integdata"></a>
Remember that, per [SPICE Metakernel Specifications](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/kernel.html#Additional%20Meta-kernel%20Specifications), the length of any given file cannot exceed 255 characters. If a file has been specified in a metakernel with a length greater than 255 characters, SPICE will say that it's unable to find it. Due to the location of the testing data directory, a large amount of the 255 character limit is already consumed just by pointing to it. For kernels with longer names, a [plus sign string continuation marker](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/kernel.html#Additional%20Text%20Kernel%20Syntax%20Rules) will need to be included.

<!-- MORE HERE
  
* [Building a Unit Test](#build)
	- [What is a Unit Test?](#what_is_test)
	- [Scarabaeus Test Walkthrough](#walkthrough)
	- [TruthData]()
	- [TestWide]()
	- [Generating a Test Template With TestWide](#gen_test)
* [Using Pytest]()
    - [The Testing Window]()
    - [Test Statuses]()
* [Examples]()
    - [Ex: Troubleshooting a Module]()
    - [Ex: Checking Exceptions]()

## Building a Unit Test<a name="build"></a>
  
### What is a Unit Test? <a name="what_is_test"></a>

### Scarabaues Test Walkthrough <a name="walkthrough"></a>

### TruthData <a name="truthdata"></a>

### TestWide <a name="testwide"></a>


### Generating a Test Skeleton With TestWide  <a name="gen_test"></a>
In order to make creating new tests as simple as possible, `TestWide` has a method that generates a test following the template outlined in the above sections. It will be up to you to populate it with the correct test and expected values, exceptions, and any logic required to test individual methods.

In the following steps we will walk through generating a test for the hypothetical `ExampleStar.py` under the assumption that you are using an **active Python virtual environment** as suggested in the ReadMe.


 In a Terminal window, first navigate to the `test` directory:
 ```
 (venv) cd test
```
<sup  style="display: inline-block;">**NOTE:** Remember that the command `cd ..` will move up by one directory, which in this case returns you to the Scarabaeus root directory.</sup>


Now, initialize `TestWide`'s test-generation tool:
```
(venv) python -c "import TestWide; TestWide.TestTools.generate_test(TestWide.TestTools)"
```

This will begin taking a series of inputs in the terminal:

1) The first prompt will ask for the name of the Scarabaeus class you want to create a test for. In our example, we provide `ExampleStar`.

2) If the generator finds a matching class inside of Scarabaeus, it will then ask for a name for the init fixture. In most cases this will be of the form `init_class_name`, but the flexibility to abbreviate is there if necessary - for example, we choose to give a shorter name: `init_ex_star`. 

3) Finally, the generator provides a list of available folders to place the newly built test in. If no test exists in the folder you select, a new file will appear with the test naming convention `test_ClassName.py`. If a test file following the correct naming convention already exists, the generated test will be placed below all existing code, separated by a comment header noting where the generated test begins.

   In our case, since we haven't created a test for `ExampleStar` yet, when we select `bodyTests` (because `ExampleStar` is a Scarabaeus Body module), the new file will be created, as noted by the completion message in the terminal. -->
