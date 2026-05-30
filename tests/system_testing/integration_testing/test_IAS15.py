# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu

import pytest
import numpy as np
import os

#--------------#
# region Setup #
#--------------#
# units, frame, epochs
km, m, s, kg, AU = scb.Units.get_units(['km', 'm', 'sec', 'kg', 'AU'])
J2000 = scb.Frame('J2000')

scb.SpiceManager.load_kernel_from_mkfile(os.getcwd() + '/data/kernels/locked/locked_generic.tm')
t0 = scb.SpiceManager.cal2et('2026 JAN 01 00:00:00.0000')
tf = scb.SpiceManager.cal2et('2026 JAN 01 06:00:00.0000')

epochs = scb.EpochArray(np.arange(t0, tf, 60), sys = 'TDB')

# spacecraft parameters (not every configuration uses all of them)
sc_params = {'mass' : awu(1e3, kg),
             'ID'   : -1000,
             'A'    : 1*m**2,
             'Cr'   : awu(1.5, None)}

# propagation scenario parameters
earth = scb.CelestialBody.from_constants('EARTH')

#--------------------#
# region    Fixtures #
#--------------------#
## different spacecraft configurations
@pytest.fixture(scope = 'function')
def sc_pm() -> scb.Spacecraft:
    """ Simplified point mass spacecraft """
    return scb.Spacecraft(name     = 'SC_PM',
                          spice_id = sc_params['ID'],
                          tot_mass = sc_params['mass'])

@pytest.fixture(scope = 'function')
def sc_cb() -> scb.Spacecraft:
    """ Cannonball spacecraft """
    return scb.Spacecraft(name      = 'SC_CB',
                          spice_id  = sc_params['ID'],
                          tot_mass  = sc_params['mass'],
                          area      = sc_params['A'],
                          ref_coeff = sc_params['Cr'])

## different initial conditions
@pytest.fixture(scope = 'function')
def x0_kep() -> scb.StateArray:
    """ Creates an initial state that results in a circular orbit for the given spacecraft """
    def _x0(sc: scb.Spacecraft):
        R   = 8000
        v_c = np.sqrt(earth.grav_param.values/R)

        pos0 = scb.ArrayWFrame(np.array([R, 0  , 0]), km  , J2000)
        vel0 = scb.ArrayWFrame(np.array([0, v_c, 0]), km/s, J2000)

        x0 = scb.StateArray(epoch  = epochs[0],
                            origin = earth,
                            state  = scb.StateDefinition().position(sc, pos0)
                                                          .velocity(sc, vel0))
        return x0
    return _x0

#--------------#
# region Tests #
#--------------#
def test_conserved_h(sc_pm, x0_kep):
    """
        Verifies that specific angular momentum is preserved across integration 
        period of a Keplerian orbit configuration.
    """
    # calculate initial specific angular momentum
    pos0, vel0 = x0_kep(sc_pm).state[0][5], x0_kep(sc_pm).state[1][5]
    h0 = pos0.cross(vel0).quantity.values

    # propagate keplerian orbit
    prop = scb.Propagator(integrator       = 'IAS15',
                          primary_body     = sc_pm,
                          state_vector     = x0_kep(sc_pm),
                          tspan            = epochs,
                          force_models     = scb.ForceModelTranslation(sc_pm),
                          propagate_STM    = False,
                          display_progress = False)
    prop.propagate()

    # compute time history of h
    h_x, h_y, h_z = [], [], []
    for state in prop.ys[0:5]:
        # compute h at time
        h_t = np.cross(state[0:3], state[3:])

        # save difference between each component
        h_x.append(h0[0] - h_t[0])
        h_y.append(h0[1] - h_t[1])
        h_z.append(h0[2] - h_t[2])
    
    # make sure average of each component is below a reasonable tolerance
    tol = 1e-6  # km^2/s = 1 m^2/s

    assert np.mean(h_x) <= tol
    assert np.mean(h_y) <= tol
    assert np.mean(h_z) <= tol

def test_conserved_e(sc_pm, x0_kep):
    """
        Verifies that specific energy is preserved across integration 
        period of a Keplerian orbit configuration.
    """
    # calculate initial specific energy
    pos0, vel0 = x0_kep(sc_pm).state[0][5], x0_kep(sc_pm).state[1][5]
    r0, v0 = pos0.norm().quantity, vel0.norm().quantity
    e0 = ((v0**2/2) - (earth.grav_param/r0)).values

    # propagate keplerian orbit
    prop = scb.Propagator(integrator       = 'IAS15',
                          primary_body     = sc_pm,
                          state_vector     = x0_kep(sc_pm),
                          tspan            = epochs,
                          force_models     = scb.ForceModelTranslation(sc_pm),
                          propagate_STM    = False,
                          display_progress = False)
    prop.propagate()

    # compute time history of e
    e = []
    for state in prop.ys[0:5]:
        # compute e at time
        r_t, v_t = np.linalg.norm(state[0:3]), np.linalg.norm(state[3:])
        e_t = (v_t**2/2) - (earth.grav_param.values/r_t)

        # save difference 
        e.append(e0 - e_t)
    
    # make sure average is below a reasonable tolerance
    tol = 1e-9  # km^2/s^2 = 1e-3m^2/s^2

    assert np.mean(e) <= tol

def test_self_consistency(sc_pm, x0_kep):
    """
        Verfiies that solutions match at the same time points across different 
        total integration periods. Ensures consistent results.
    """
    # short interval
    tf_short     = scb.SpiceManager.cal2et('2026 JAN 01 06:00:00.0000')
    short_times  = np.arange(t0, tf_short, 60)

    epochs_short = scb.EpochArray(short_times, sys = 'TDB')

    # long interval with matching time steps during short interval
    tf_long     = scb.SpiceManager.cal2et('2026 JAN 01 12:00:00.0000')
    epochs_long = scb.EpochArray(np.concatenate([short_times, np.arange(tf_short, tf_long, 50)[1:]]), 'TDB')

    # propagate both intervals
    fm = scb.ForceModelTranslation(sc_pm)
    prop_short = scb.Propagator(integrator       = 'IAS15',
                                primary_body     = sc_pm,
                                state_vector     = x0_kep(sc_pm),
                                tspan            = epochs_short,
                                force_models     = fm,
                                propagate_STM    = False,
                                display_progress = False)
    prop_short.propagate()
    
    prop_long  = scb.Propagator(integrator       = 'IAS15',
                                primary_body     = sc_pm,
                                state_vector     = x0_kep(sc_pm),
                                tspan            = epochs_long,
                                force_models     = fm,
                                propagate_STM    = False,
                                display_progress = False)
    prop_long.propagate()

    # compare long interval solution to short interval
    interval_diff = np.abs(prop_long.ys[:len(prop_short.ys), :] - prop_short.ys)
    np.testing.assert_allclose(interval_diff, 0, atol = 1e-8)

class TestEventDetect:
    def test_detect_state_based_event(self):
        pytest.skip(reason = 'Propagator has not yet fully implemented event detection, skipping for now.')
    
    def test_detect_time_based_event(self):
        pytest.skip(reason = 'Propagator has not yet fully implemented event detection, skipping for now.')
