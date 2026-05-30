# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb

import pytest
kg, km, sec, mu = scb.Units.get_units(['kg', 'km', 'sec', 'mu'])
ITRF93 = scb.Frame('ITRF93')

# --------------#
# region Tests #
# --------------#
class TestInitialization:
    def test_proper_init(self):
        """
        Verifies that object is constructed correctly for all valid initialization configurations.
        """
        # parameters
        test_name  = 'TEST_CB'
        test_mass  = scb.ArrayWUnits(1, kg)
        test_rad   = scb.ArrayWUnits(1, km)
        test_GM    = scb.ArrayWUnits(1, km**3 / sec**2)
        test_frame = ITRF93
        test_id    = 10000000

        # construct body
        cb = scb.CelestialBody(name        = test_name,
                               mass        = test_mass,
                               mean_radius = test_rad,
                               grav_param  = test_GM,
                               base_frame  = test_frame,
                               spice_id    = test_id)
        
        # check
        assert cb.name == test_name
        assert cb.mass() == test_mass
        assert cb.mean_radius == test_rad
        assert cb.grav_param == test_GM
        assert cb.base_frame == test_frame
        assert cb.spice_id == test_id

    @pytest.mark.parametrize(
        "cb_name", [("Jupiter"), ("Not Jupiter")], ids=["Valid Query", "Invalid Query"]
    )
    def test_from_constants(self, cb_name):
        """
        Verifies that alternate construction method builds a CelestialBody correctly.
        """
        jov_consts = scb.constants.JUPITER
        if cb_name == 'Jupiter':
            # valid construction -> make sure correct values match
            cb = scb.CelestialBody.from_constants(cb_name)
            assert cb.name == cb_name
            assert cb.mean_radius == jov_consts.mean_radius
            assert cb.grav_param == jov_consts.GM
            assert cb._base_frame == jov_consts.ref_name
            assert cb.spice_id == jov_consts.body_center_id
            assert cb.barycenter_id == jov_consts.barycenter_id
        else:
            # invalid construction -> make sure error is raised
            with pytest.raises(Exception):
                scb.CelestialBody.from_constants(cb_name)
