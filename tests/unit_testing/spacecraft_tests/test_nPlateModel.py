#========================================#
#  Unit Tests for the nPlateModel class  #
#========================================#
import src.scarabaeus as scb
import pytest
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu

#--------------------#
# region    Fixtures #
#--------------------#

#--------------------#
# endregion Fixtures #
#--------------------#


#--------------#
# region Tests #
#--------------#
def test_initialization(nplate_config_file):
    """
        Verifies that object is constructed correctly.
    """
    # construct N-plate model using a configuration file
    plate_config = str(nplate_config_file)
    n_model = scb.nPlateModel(plate_config)
    