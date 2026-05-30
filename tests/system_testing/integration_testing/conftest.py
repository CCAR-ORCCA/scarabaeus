# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import pytest, os
import spiceypy as spice

#-----------------------#
# region Test Constants #
#-----------------------#
DATA_DIR = '/tests/system_testing/integration_testing/integration_test_data'
REQ_DATA = []

#----------------------#
# region Session Setup #
#----------------------#
def pytest_sessionstart(session):
    """
        Prepare test session for integration testing.
    """
    # ensure data is accessible
    check_data(session)

    # clean the kernel pool just in case anything is currently loaded
    import src.scarabaeus
    spice.kclear()

def check_data(session):
    """
        Ensures that required testing data exists before running integration tests.

        Compares the files in DATA_DIR directory to the expected files specified 
        in the REQ_DATA list and ends test session before collection if one or 
        more data files is missing.
    """
    # get all files in the testing folder
    found_data = os.listdir(os.getcwd() + DATA_DIR)

    # find and format any missing files
    missing_data = '\n'.join([ele for ele in REQ_DATA if ele not in found_data])

    # stop test session if one or more files is missing
    if missing_data:
        pytest.exit(reason = (f'\n{"!"*10} One or more data files necessary for testing could not '
                              f'be located: {"!"*10}\n'
                              f'{missing_data}\n'
                              'See tests/TESTS_README.md for more.'))
        
def pytest_sessionfinish(session):
    """
        Final clean up/teardown for integration testing.
    """
    # make sure the kernel pool is empty
    spice.kclear()

#-----------------#
# region Fixtures #
#-----------------#
@pytest.fixture(scope = 'session')
def testing_data_path() -> str:
    """
        Fetches the path to the root directory for all integration testing data.

        Returns
        -------
        path : str
            File path to the testing data root directory.
    """
    return (os.getcwd() + DATA_DIR)
