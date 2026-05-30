# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu

import pytest
import numpy as np

kg, km, sec, rad = scb.Units.get_units(["kg", "km", "sec", "rad"])
J2000 = scb.Frame("J2000")


# --------------------#
# region    Fixtures #
# --------------------#
# setup
@pytest.fixture(scope="module")
def common_params():
    spice_id = -1001
    tot_mass = scb.ArrayWUnits(2000.0, kg)
    sc_area = scb.ArrayWUnits(1e-06, km**2)
    sc_ref_coeff = 1.5
    return spice_id, tot_mass, sc_area, sc_ref_coeff


@pytest.fixture(scope="module")
def config1(common_params):
    """
    Basic configuration.
        - no instruments
        - minimal parameters
    """
    name = "orbiter"
    spice_id, tot_mass, sc_area, sc_ref_coff = common_params
    sc1 = scb.Spacecraft(name, spice_id, tot_mass, sc_area, sc_ref_coff)
    return sc1


@pytest.fixture(scope="module")
def sc_w_instr(camera, common_params):
    """
    Basic configuration with instrument.
        - add instruments
        - minimal parameters
    """
    name = "orbiter"
    spice_id, tot_mass, sc_area, sc_ref_coff = common_params
    sc2 = scb.Spacecraft(
        name, spice_id, tot_mass, sc_area, sc_ref_coff, instrument_list=[camera]
    )
    return sc2


@pytest.fixture(scope="module")
def sc_w_nplate(nplate_config_file, common_params):
    """
    Basic configuration with nadir plate.
        - add nplate
        - minimal parameters
    """
    name = "orbiter"
    spice_id, tot_mass, sc_area, sc_ref_coff = common_params
    plate_config = str(nplate_config_file)
    n_model = scb.nPlateModel(plate_config)
    sc3 = scb.Spacecraft(
        name,
        spice_id,
        tot_mass,
        sc_area,
        sc_ref_coff,
        n_plate_model=n_model,
        attitudeMode="nadir_pointing_to_sun",
    )
    return sc3


def get_state(spacecraft, epochs, origin):
    """Define stateArray object"""
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0, 70.0, 0.0]), km / sec, J2000)

    state_dict = [
        ("position", 3, "estimated", "dynamic", spacecraft, pos_0),
        ("velocity", 3, "estimated", "dynamic", spacecraft, vel_0),
    ]
    return scb.StateArray(
        epochs[0], origin, state=scb.StateDefinition.from_components(state_dict)
    )


def get_force_model(
    spacecraft, canonball_srp=False, nplate_srp=False, finite_burn=False
):
    return scb.ForceModelTranslation(
        primary_body=spacecraft,
        cannonball_SRP=canonball_srp,
        nplate_SRP=nplate_srp,
        finite_burn=finite_burn,
    )


def get_force_model_sph(earth_config_file,
                        spacecraft,
                        sph_harm=False,
                        canonball_srp=False):
    return scb.ForceModelTranslation(primary_body       = spacecraft,
                                     cannonball_SRP     = canonball_srp,
                                     sph_harm           = sph_harm,
                                     sph_harm_order     = 3,
                                     sph_harm_cs_file   = str(earth_config_file),
                                     sph_harm_body      =scb.CelestialBody.from_constants('EARTH'),
                                     sph_harm_norm_flag = True)


@pytest.fixture(scope="module")
def spacecraft(request):
    """Resolves the string name into the actual fixture object."""
    return request.getfixturevalue(request.param)


class TestSCConfigs:
    # for configs with instruments, check get_dependednt_spice_ids

    # skip for get_attitude

    # skip for nadir

    @pytest.mark.parametrize(
        "spacecraft, expected",
        [
            ("config1", {"instruments": [], "plates": []}),
            ("sc_w_instr", {"instruments": [-1000], "plates": []}),
            ("sc_w_nplate", {"instruments": [], "plates": [-64027]}),
        ],
        ids=["sc", "sc_w_instr", "sc_w_nplate"],
        indirect=["spacecraft"],
    )
    def test_sc_config1(self, spacecraft, expected):
        spice_ids = spacecraft.get_dependent_spice_ids()

        assert spice_ids == expected

class TestSCPropagatorCompatibility:
    # check that a propagator will take it (dont need to integrate)
    @pytest.mark.parametrize(
        "spacecraft",
        [("config1"), ("sc_w_instr"), ("sc_w_nplate")],
        ids=["sc", "sc_w_instr", "sc_w_nplate"],
        indirect=["spacecraft"],
    )
    def test_sc_prop(self, spacecraft, epochs, origin):
        state = get_state(spacecraft, epochs, origin)
        force_model = get_force_model(spacecraft)
        prop = scb.Propagator(
            primary_body=spacecraft,
            state_vector=state,
            tspan=epochs,
            force_models=force_model,
        )

    @pytest.mark.parametrize(
        "spacecraft",
        [("config1"), ("sc_w_instr"), ("sc_w_nplate")],
        ids=["sc", "sc_w_instr", "sc_w_nplate"],
        indirect=["spacecraft"],
    )
    def test_sc_prop_cannonballSRP(self, spacecraft, epochs, origin):
        state = get_state(spacecraft, epochs, origin)
        cannon = get_force_model(spacecraft, canonball_srp=True)
        prop = scb.Propagator(
            primary_body=spacecraft,
            state_vector=state,
            tspan=epochs,
            force_models=cannon,
        )

    @pytest.mark.parametrize(
        "spacecraft", ["sc_w_nplate"], ids=["sc_w_nplate"], indirect=["spacecraft"]
    )
    def test_sc_prop_nplateSRP(self, spacecraft, epochs, origin):
        state = get_state(spacecraft, epochs, origin)
        nplate = get_force_model(spacecraft, nplate_srp=True)
        prop = scb.Propagator(
            primary_body=spacecraft,
            state_vector=state,
            tspan=epochs,
            force_models=nplate,
        )

    @pytest.mark.parametrize(
        "spacecraft",
        [("config1"), ("sc_w_instr"), ("sc_w_nplate")],
        ids=["sc", "sc_w_instr", "sc_w_nplate"],
        indirect=["spacecraft"],
    )
    def test_sc_prop_sphHarm(self, spacecraft, epochs, origin, earth_config_file):
        state = get_state(spacecraft, epochs, origin)
        sph_harm = get_force_model_sph(earth_config_file, spacecraft, sph_harm=True)
        prop = scb.Propagator(
            primary_body=spacecraft,
            state_vector=state,
            tspan=epochs,
            force_models=sph_harm,
        )

    @pytest.mark.parametrize(
        "spacecraft",
        [("config1"), ("sc_w_instr"), ("sc_w_nplate")],
        ids=["sc", "sc_w_instr", "sc_w_nplate"],
        indirect=["spacecraft"],
    )
    def test_sc_prop_thrust(self, spacecraft, epochs, origin):
        state = get_state(spacecraft, epochs, origin)
        force_model = get_force_model(spacecraft, finite_burn=True)
        prop = scb.Propagator(
            primary_body=spacecraft,
            state_vector=state,
            tspan=epochs,
            force_models=force_model,
        )

    @pytest.mark.parametrize(
        "spacecraft",
        [("config1"), ("sc_w_instr"), ("sc_w_nplate")],
        ids=["sc", "sc_w_instr", "sc_w_nplate"],
        indirect=["spacecraft"],
    )
    def test_sc_prop_SRP(self, spacecraft, epochs, origin):
        state = get_state(spacecraft, epochs, origin)
        force_model = get_force_model(spacecraft, canonball_srp=True)
        prop = scb.Propagator(
            primary_body=spacecraft,
            state_vector=state,
            tspan=epochs,
            force_models=force_model,
        )
