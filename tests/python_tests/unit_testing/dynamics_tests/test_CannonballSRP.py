# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==========================================#
#  Unit Tests for the CannonballSRP class  #
# ==========================================#
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import pytest
import numpy as np

km, kg, sec = scb.Units.get_units(["km", "kg", "sec"])
J2000 = scb.Frame("J2000")


# --------------------#
# region    Fixtures #
# --------------------#
# setup
@pytest.fixture
def epoch():
    """Epoch to define state."""
    return scb.EpochArray("2000 JAN 01 12:00:00.000", "UTC")


@pytest.fixture
def origin():
    """Origin body to define state."""
    return scb.CelestialBody.from_constants("Earth")


@pytest.fixture
def sc_area():
    return awu(1e-06, km**2)


@pytest.fixture
def sc_ref_coeff():
    return 1.5


# construction
@pytest.fixture
def sc(sc_area, sc_ref_coeff):
    """Spacecraft to define the cannonball model for."""
    return scb.Spacecraft("ORCCA", -1234, area=sc_area, ref_coeff=sc_ref_coeff)


@pytest.fixture
def state(epoch, origin, sc):
    """StateArray object to construct CannonballSRP."""
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0, 70.0, 0.0]), km / sec, J2000)

    return scb.StateArray(epoch, origin, state=scb.StateDefinition()
                                                .position(sc, pos_0)
                                                .velocity(sc, vel_0),)


# validation
@pytest.fixture
def exp_const_scale_factor():
    """Expected constant scale factor computed by CannonballSRP."""
    return awu(152969156.355, km**3 * kg * sec**-2)


@pytest.fixture
def frame():
    """Define frame"""

    return J2000


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(state, sc, origin, sc_area, sc_ref_coeff, exp_const_scale_factor):
    """
    Verifies that object is constructed correctly.
    """
    # initialize
    cb_srp = scb.CannonballSRP(state.origin, sc, J2000)

    # ensure correct properties
    assert cb_srp.origin == origin
    assert cb_srp.origin_name == origin.name
    assert cb_srp._spacecraft == sc
    assert cb_srp.ref_frame == J2000

    assert cb_srp.area == scb.ArrayWUnits.get_value_in_target_units(sc_area, km**2)  # stored in km^2
    assert cb_srp.ref_coeff == sc_ref_coeff

    assert cb_srp._const_scale_factor == scb.ArrayWUnits.get_value_in_target_units(
                                        exp_const_scale_factor, (kg * km**3) / sec**2)


# NOTE: the rest of this class' methods are tested in integration tests


# ------------------------------------------------------------------#
# region  eta_srp partial                                            #
# ------------------------------------------------------------------#
# The acceleration scales linearly with eta_srp, so the partial is the
# acceleration at unit scale — a constant, defined even at eta_srp = 0, which
# is the natural starting guess when the factor is estimated as a correction.


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0, 2.0],
                         ids=["zero", "half", "unit", "double"])
def test_eta_partial_is_independent_of_eta(origin, sc_area, sc_ref_coeff, epoch, eta):
    """The partial is the unit-scale acceleration, whatever eta happens to be."""
    # SRP divides by the spacecraft mass, which the shared fixture leaves unset
    sc = scb.Spacecraft(
        "ORCCA", -1234, awu(2000.0, kg), area=sc_area, ref_coeff=sc_ref_coeff
    )
    epoch = float(np.atleast_1d(epoch.et).ravel()[0])
    model = scb.CannonballSRP(origin, sc, J2000, t0=epoch, tf=epoch + 100.0)
    position = np.array([7000.0, 0.0, 0.0])

    reference = np.asarray(
        model._compute_partial_by_eta(
            position, {("eta_srp", sc.spice_id): 1.0}, epoch, sc
        )
    ).ravel()
    partial = np.asarray(
        model._compute_partial_by_eta(
            position, {("eta_srp", sc.spice_id): eta}, epoch, sc
        )
    ).ravel()

    np.testing.assert_allclose(partial, reference, rtol=1e-12)

    # and away from zero it still equals acceleration / eta, as before
    if eta != 0.0:
        acceleration = np.asarray(
            model.compute_acceleration(
                position, {("eta_srp", sc.spice_id): eta}, epoch, sc
            )
        ).ravel()
        np.testing.assert_allclose(partial, acceleration / eta, rtol=1e-9)


# endregion                                                          #
# ------------------------------------------------------------------#


# ---------------------------------------------------#
# region  Scalar state reads under autograd          #
# ---------------------------------------------------#
"""
A partial verified by automatic differentiation is evaluated with an autograd
``ArrayBox`` in the state, and ``float()`` raises on one — so a model that cast
its scalar parameters that way both broke the trace and failed outright. The
models now read them through ``DynamicModel._scalar_from_state``, which hands a
traced value back untouched.

Tests
-----
test_scalar_from_state_unwraps_plain_values
    A float, an array and an ArrayWUnits all come back as a float.
test_scalar_from_state_passes_a_traced_value_through
    A traced value is returned as-is so the derivative survives.
test_models_do_not_cast_scalar_state_with_float
    No traceable model still forces the cast that raised on an ArrayBox.
"""

import inspect

from scarabaeus.dynamics.DynamicModel import DynamicModel


class _Boxed:
    """Stand-in for a traced value, without requiring the autograd backend."""


def test_scalar_from_state_unwraps_plain_values():
    """The shapes a state actually holds all come back as a float."""
    # ArrayWUnits normalises a one-element array to a 0-d scalar, so these are
    # the forms a scalar parameter really takes in ``current_state``.
    assert DynamicModel._scalar_from_state(2.5) == 2.5
    assert DynamicModel._scalar_from_state(np.float64(2.5)) == 2.5
    assert DynamicModel._scalar_from_state(awu(np.array([2.5]), None)) == 2.5
    assert isinstance(DynamicModel._scalar_from_state(awu(2.5, None)), float)


def test_scalar_from_state_passes_a_traced_value_through(monkeypatch):
    """A traced value is returned as-is so the derivative survives."""
    import scarabaeus.utils.NumpyWrapper as npw

    monkeypatch.setattr(npw, "autograd_types", (_Boxed,))
    traced = _Boxed()

    assert DynamicModel._scalar_from_state(traced) is traced, (
        "float() raises on an autograd ArrayBox, which is exactly the value "
        "present while a partial is being verified by differentiation"
    )


@pytest.mark.parametrize(
    "model_name",
    ["CannonballSRP", "nPlateSRP", "YarkovskyEffect", "FiniteBurn", "PointMassGravity"],
)
def test_models_do_not_cast_scalar_state_with_float(model_name):
    """No traceable model still forces the cast that raised on an ArrayBox."""
    module = __import__(
        f"scarabaeus.dynamics.{model_name}", fromlist=[model_name]
    )
    source = inspect.getsource(module)

    assert 'float(val.values if hasattr(val, "values") else val)' not in source
    assert "float(val.values if hasattr(val, 'values') else val)" not in source
    assert "_scalar_from_state" in source
