# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
from scarabaeus import SpiceManager
import pytest
import spiceypy as spice
import numpy as np

N, kg, sec, km, dam, m, mm, mu, rad, mrad, deg, g, hr = scb.Units.get_units(['N', 'kg', 'sec', 'km', 'dam', 'm', 'mm', 'mu', 'rad', 'mrad', 'deg', 'g', 'hr'])

#--------------------#
# region    Fixtures #
#--------------------#

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
class TestInitialization:
    @pytest.mark.parametrize(
        'init_args',
        [('Sun', 10, None, None),
         ('ORCCA', -1234, None, None),
         ('ORCCA', -1234, scb.ArrayWUnits(300, kg), None),
         ('TEST_BODY', -1000, scb.ArrayWUnits(300, kg), 3)],
         ids = ['Name in Kernels', 'Name & New ID', 'Name, ID, Const Mass', 'Name, ID, Mass, Mass Poly']
    )
    def test_proper_init(self, init_args):
        """
            Verifies that object is constructed correctly for all valid initialization configurations.
        """
        # construct body
        body = scb.Body(*init_args)

        # verify that name is correct
        assert body._name == init_args[0]
        assert body.spice_id == init_args[1]
        assert body._mass_profile == init_args[2]
        assert body._mass_poly_deg == init_args[3]
            
    @pytest.mark.parametrize(
            'init_args',
            [((10))],
            ids = ['ID not in Kernels']
    )
    def test_improper_init(self, init_args):
        """
            Verifies that exceptions are handled correctly for all invalid initialization configurations.
        """
        pytest.skip('UPDATE')
        with pytest.raises(Exception):
            scb.Body('TEST_BODY', None)

@pytest.mark.parametrize(
    "body, date_str",
    [(scb.Body("Sun"), '2000 JAN 01 12:00:00.000'), 
     (scb.Body("Sun"), '2000 JAN 02 12:00:00.000')],
    ids=["State at J2000", "State at J2000 + 1 day"],
)
def test_get_state(body, date_str):
    epoch = scb.EpochArray(date_str, 'TDB')

    spice_state = np.array(spice.spkezr('SUN', float(epoch.times.values), 'J2000', 'None', 'Earth')[0])
    body_state = body.get_state(epoch)
    assert np.array_equal(body_state.quantity.values, spice_state)


# @pytest.mark.parametrize(
#     'body, epoch, mass',
#     [(scb.Body('TEST_BODY', -1000, scb.ArrayWUnits(300, kg), None), scb.EpochArray('1096804869.182','TDB'), scb.ArrayWUnits(300,kg)),
#      (scb.Body('TEST_BODY', -1000, scb.SCBPolynomial([0, 2, 1]), 3), scb.EpochArray('1096805869.182','TDB'), scb.ArrayWUnits(300,kg))],
#      ids = ['State at J2000', 'State at J2000 + 1000 sec']
# )
# def test_get_mass(body, epoch, mass):
#     """
#         Verifies that the get_mass method functions correctly.
#     """
#     assert body.get_mass(epoch) == mass
