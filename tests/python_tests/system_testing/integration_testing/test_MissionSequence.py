# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =============================================#
#  Integration Tests for the MissionSequence  #
# =============================================#
import src.scarabaeus as scb
import os
import pytest
import numpy as np

km, kg, sec = scb.Units.get_units(["km", "kg", "sec"])


J2000 = scb.Frame("J2000")

# --------------------#
# region    Fixtures #
# --------------------#


### use conftest fixture
@pytest.fixture
def epochs():
    time_0 = scb.SpiceManager.jd2et(2461809.72995654 + 1 / 3)
    time_f = scb.SpiceManager.jd2et(2461809.72995654 + 1)
    dt = 60 * 60
    epoch_array = scb.EpochArray(np.arange(time_0, time_f, dt), sys="TDB")

    return epoch_array


### use conftest fixture
@pytest.fixture
def origin():

    origin = scb.CelestialBody.from_constants("SUN")

    return origin


### use conftest fixture
@pytest.fixture
def sc():
    Orbiter_area = scb.ArrayWUnits(1e-06, km**2)
    Orbiter_cr_srp = scb.ArrayWUnits(1.5, None)
    return scb.Spacecraft(
        "Orbiter", -1000, scb.ArrayWUnits(2000.0, kg), Orbiter_area, Orbiter_cr_srp
    )


### use conftest fixture
@pytest.fixture
def state_dict(sc):
    state_definition = [
        (
            "position",
            3,
            "estimated",
            "dynamic",
            sc,
            scb.ArrayWFrame(
                scb.ArrayWUnits(
                    np.array(
                        [-2.6561672815488e08, 1.8837080839972e08, 1.0199307996915e08]
                    ),
                    km,
                ),
                J2000,
            ),
        ),
        (
            "velocity",
            3,
            "estimated",
            "dynamic",
            sc,
            scb.ArrayWFrame(
                scb.ArrayWUnits(
                    np.array([-9.0779264528807, -12.4823165310294, -4.4599484708606]),
                    km / sec,
                ),
                J2000,
            ),
        ),
    ]
    return state_definition


@pytest.fixture
def state(epochs, origin, state_dict):
    """Define stateArray object"""
    return scb.StateArray(
        epochs[0], origin, state=scb.StateDefinition.from_components(state_dict)
    )


@pytest.fixture
def force_model(sc):
    return scb.ForceModelTranslation(
        primary_body=sc,
    )


@pytest.fixture
def propagator(sc, state, epochs, force_model):

    prop = scb.Propagator(
        primary_body=sc, state_vector=state, tspan=epochs, force_models=force_model
    )
    return prop


@pytest.fixture
def ias15_leg(sc, origin, state_dict, force_model):
    """
    Factory for a leg propagator that is handed only its ``(t0, tf)`` bounds.

    IAS15 chooses its own steps, so such a leg carries no output grid until it
    has been propagated. ``propagate_STM=False`` because IAS15 integrates the
    6-state only.
    """

    def factory(t_start, t_end):
        state_vector = scb.StateArray(
            t_start, origin, state=scb.StateDefinition.from_components(state_dict)
        )
        return scb.Propagator(
            integrator="IAS15",
            primary_body=sc,
            state_vector=state_vector,
            tspan=(t_start, t_end),
            force_models=force_model,
            propagate_STM=False,
            display_progress=False,
        )

    return factory


@pytest.fixture
def ias15_orbit_leg(epochs):
    """
    Bounds-only IAS15 leg on a fast circular Earth orbit, where the error
    control has to take many short steps within the interval.
    """
    earth = scb.CelestialBody.from_constants("EARTH")
    sc_orbit = scb.Spacecraft("Orbiter LEO", -1002, scb.ArrayWUnits(1e3, kg))

    radius = 8000.0
    speed = float(np.sqrt(earth.grav_param.values / radius))
    t_start, t_end = epochs[0], epochs[-1]

    state_vector = scb.StateArray(
        t_start,
        earth,
        state=scb.StateDefinition()
        .position(sc_orbit, scb.ArrayWFrame(np.array([radius, 0.0, 0.0]), km, J2000))
        .velocity(sc_orbit, scb.ArrayWFrame(np.array([0.0, speed, 0.0]), km / sec, J2000)),
    )
    prop = scb.Propagator(
        integrator="IAS15",
        primary_body=sc_orbit,
        state_vector=state_vector,
        tspan=(t_start, t_end),
        force_models=scb.ForceModelTranslation(primary_body=sc_orbit),
        propagate_STM=False,
        display_progress=False,
    )
    return prop, t_start, t_end


@pytest.fixture
def dv1_model():
    return scb.ImpulsiveBurn(
        scb.ArrayWFrame(scb.ArrayWUnits(np.array([5]), km / sec), J2000),
        scb.ArrayWFrame(
            scb.ArrayWUnits(np.array([0.16384638, 0.9830783, 0.08192319]), km / km),
            J2000,
        ),
    )


@pytest.fixture
def state_fb_def():
    leg_state_definition_fb_M1 = [
        (
            "position",
            3,
            "estimated",
            "dynamic",
            sc,
            scb.ArrayWFrame(np.full(3, np.nan), km, J2000),
        ),
        (
            "velocity",
            3,
            "estimated",
            "dynamic",
            sc,
            scb.ArrayWFrame(np.full(3, np.nan), km / sec, J2000),
        ),
    ]
    return leg_state_definition_fb_M1


@pytest.fixture
def leg_model_fb(origin, state_fb_def):
    return scb.StateArray(
        epoch=scb.EpochArray(np.full(1, np.nan), sys="TDB"),
        origin=origin,
        state=scb.StateDefinition.from_components(state_fb_def),
    )


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
@pytest.mark.parametrize("init_args", [("TEST SEQUENCE",)], ids=["name"])
def test_initialization(init_args):
    """
    Verifies that object is constructed correctly.
    """
    miss_seq = scb.MissionSequence(*init_args)
    assert miss_seq.name == "TEST SEQUENCE"


class TestAddSegment:
    """Segments (legs, burns) are appended to the sequence correctly."""

    # fix this
    def test_leg(self, epochs, origin, state_dict, propagator):
        """
        Verifies that legs are added to the sequence correctly.
        """
        leg1_model = scb.StateArray(
            epoch=epochs[0],
            origin=origin,
            state=scb.StateDefinition.from_components(state_dict),
        )

        miss_seq = scb.MissionSequence("TEST SEQUENCE")
        miss_seq.addLeg(
            "LEG 1",
            scb.ArrayWUnits(58000, sec),
            leg1_model,
            propagator,
            scb.ArrayWUnits(360, sec),
        )
        assert miss_seq._names == ["LEG 1"]

    # fix this
    def test_burn(self, epochs, origin, state_dict, propagator, dv1_model):
        """
        Verifies that legs are added to the sequence correctly.
        """
        leg1_model = scb.StateArray(
            epoch=epochs[0],
            origin=origin,
            state=scb.StateDefinition.from_components(state_dict),
        )

        miss_seq = scb.MissionSequence("TEST SEQUENCE")
        miss_seq.addLeg(
            "LEG 1",
            scb.ArrayWUnits(58000, sec),
            leg1_model,
            propagator,
            scb.ArrayWUnits(360, sec),
        )
        miss_seq.addBurn("Burn 1", dv1_model, propagator)
        assert miss_seq._names == ["LEG 1", "Burn 1"]


class TestListForces:
    """MissionSequence.list_forces: the forces of each propagated segment."""

    def test_propagated_segments_only(
        self, sc, epochs, origin, state, state_dict, propagator, dv1_model
    ):
        """Legs are listed by name and per object; impulsive burns are not."""
        leg1_model = scb.StateArray(
            epoch=epochs[0],
            origin=origin,
            state=scb.StateDefinition.from_components(state_dict),
        )
        srp_propagator = scb.Propagator(
            primary_body=sc,
            state_vector=state,
            tspan=epochs,
            force_models=scb.ForceModelTranslation(primary_body=sc, cannonball_SRP=True),
        )

        miss_seq = scb.MissionSequence("TEST SEQUENCE")
        miss_seq.addLeg(
            "LEG 1",
            scb.ArrayWUnits(58000, sec),
            leg1_model,
            propagator,
            scb.ArrayWUnits(360, sec),
        )
        miss_seq.addBurn("Burn 1", dv1_model, propagator)
        miss_seq.addLeg(
            "LEG 2",
            scb.ArrayWUnits(58000, sec),
            leg1_model,
            srp_propagator,
            scb.ArrayWUnits(360, sec),
        )

        assert miss_seq.list_forces() == {
            "LEG 1": {"Orbiter": ["PointMassGravity(SUN)"]},
            "LEG 2": {"Orbiter": ["CannonballSRP", "PointMassGravity(SUN)"]},
        }

    def test_empty_sequence(self):
        """A sequence with no segments lists nothing."""
        assert scb.MissionSequence("EMPTY").list_forces() == {}


class TestDeferredGrid:
    """Legs given only (t0, tf) bounds defer their grid until propagated."""

    def test_dop853_places_own_grid(self, sc, state, epochs, force_model):
        """
        DOP853 places its own output times when given no grid (t_eval=None), so a
        (t0, tf) tuple defers the grid exactly as it does for IAS15.
        """
        prop_tuple = scb.Propagator(
            primary_body=sc,
            state_vector=state,
            tspan=(epochs[0], epochs[-1]),
            force_models=force_model,
        )

        miss_seq = scb.MissionSequence("TEST SEQUENCE")
        miss_seq.addLeg("LEG 1", prop=prop_tuple)

        assert miss_seq.deferred_grids == [True]
        assert len(miss_seq.epochs_vec[0]) == 2  # bounds only, so far

        miss_seq.propagate(verbose=False)

        grid = miss_seq.epochs_vec[0]
        assert len(grid) > 2
        assert float(grid[0].et) == pytest.approx(float(epochs[0].et))
        assert float(grid[-1].et) == pytest.approx(float(epochs[-1].et))

    def test_explicit_grid_overrides_tuple(self, sc, state, epochs, force_model):
        """
        A leg given `epochs` keeps that grid even when its propagator carries a
        tuple, and is not marked deferred.
        """
        prop_tuple = scb.Propagator(
            primary_body=sc,
            state_vector=state,
            tspan=(epochs[0], epochs[-1]),
            force_models=force_model,
        )

        miss_seq = scb.MissionSequence("TEST SEQUENCE")
        miss_seq.addLeg("LEG 1", prop=prop_tuple, epochs=epochs)

        assert miss_seq.names == ["LEG 1"]
        assert len(miss_seq.epochs_vec[0]) == len(epochs)
        assert miss_seq.deferred_grids == [False]

    def test_adopts_integrator_grid(self, ias15_leg, epochs):
        """
        An IAS15 leg given only (t0, tf) holds its two bounds until it is
        propagated, then carries the epochs the integrator's steps landed on —
        bounds included, so the segment still starts and ends where it said it
        would.  How many samples that is, is the integrator's call: this arc is
        slow enough that it may well be the two bounds alone.
        """
        miss_seq = scb.MissionSequence("ADAPTIVE SEQUENCE")
        miss_seq.addLeg("coast", prop=ias15_leg(epochs[0], epochs[-1]))

        assert miss_seq.deferred_grids == [True]
        assert len(miss_seq.epochs_vec[0]) == 2  # bounds only, so far

        scenario = miss_seq.propagate(verbose=False)

        grid = miss_seq.epochs_vec[0]
        assert float(grid[0].et) == pytest.approx(float(epochs[0].et))
        assert float(grid[-1].et) == pytest.approx(float(epochs[-1].et))

        # The bookkeeping and the propagated segment agree with the adopted grid
        assert miss_seq.num_steps[0] == len(grid)
        assert miss_seq.last_idx[0] == len(grid)
        assert len(scenario.segment_epochsTDB["coast"]) == len(grid)
        np.testing.assert_allclose(
            np.atleast_1d(scenario.segment_epochsTDB["coast"]), grid.et
        )

    def test_keeps_steps_dynamics_need(self, ias15_orbit_leg):
        """
        On dynamics fast enough to force short steps, the adopted grid is the fine
        one the error control produced — which is the point of handing the segment
        only its bounds.
        """
        miss_seq = scb.MissionSequence("ADAPTIVE ORBIT")
        prop, t_start, t_end = ias15_orbit_leg
        miss_seq.addLeg("orbit", prop=prop)

        miss_seq.propagate(verbose=False)

        grid = miss_seq.epochs_vec[0]
        assert len(grid) > 2
        assert float(grid[0].et) == pytest.approx(float(t_start.et))
        assert float(grid[-1].et) == pytest.approx(float(t_end.et))
        assert np.all(np.diff(grid.et) > 0.0)

    def test_two_legs_stay_continuous(self, ias15_leg, epochs):
        """
        Two adaptive legs meet at the epoch they share, and the auto-inserted node
        keeps overlapping the previous leg's endpoint in the step bookkeeping.
        """
        t_mid = epochs[len(epochs) // 2]

        miss_seq = scb.MissionSequence("TWO ADAPTIVE LEGS")
        miss_seq.addLeg("leg1", prop=ias15_leg(epochs[0], t_mid))
        miss_seq.addLeg("leg2", prop=ias15_leg(t_mid, epochs[-1]))

        scenario = miss_seq.propagate(verbose=False)

        assert miss_seq.types == ["Leg", "Node", "Leg"]
        assert miss_seq.deferred_grids == [True, False, True]

        first, node, second = miss_seq.epochs_vec
        assert float(first[-1].et) == pytest.approx(float(t_mid.et))
        assert float(node[0].et) == pytest.approx(float(t_mid.et))
        assert float(second[0].et) == pytest.approx(float(t_mid.et))

        # The node overlaps the boundary sample, so it is counted once overall
        n_first, n_second = len(first), len(second)
        assert miss_seq.last_idx == [n_first - 1, n_first - 1, n_first - 1 + n_second]
        assert len(scenario.total_epochsTDB) == n_first + n_second - 1


class TestGridBoundTol:
    """How far an adopted grid may stop short of the segment's bounds."""

    @staticmethod
    def deferred_sequence(ias15_leg, epochs, **kwargs):
        """Sequence holding one bounds-only leg, ready to adopt a grid."""
        miss_seq = scb.MissionSequence("TOLERANCE SEQUENCE", **kwargs)
        miss_seq.addLeg("coast", prop=ias15_leg(epochs[0], epochs[-1]))
        return miss_seq

    @staticmethod
    def grid_short_by(epochs, gap):
        """Grid over the same bounds, ending ``gap`` seconds early."""
        return scb.EpochArray(
            np.array([float(epochs[0].et), float(epochs[-1].et) - gap]), sys="TDB"
        )

    def test_default(self, ias15_leg, epochs):
        assert self.deferred_sequence(ias15_leg, epochs).grid_bound_tol == 1e-6

    def test_grid_within_tolerance_is_adopted(self, ias15_leg, epochs):
        miss_seq = self.deferred_sequence(ias15_leg, epochs, grid_bound_tol=1e-3)
        grid = self.grid_short_by(epochs, 1e-4)

        miss_seq.adopt_propagated_grid(0, grid)

        assert miss_seq.epochs_vec[0] is grid
        assert miss_seq.num_steps[0] == len(grid)

    def test_grid_outside_tolerance_is_refused(self, ias15_leg, epochs):
        miss_seq = self.deferred_sequence(ias15_leg, epochs, grid_bound_tol=1e-3)

        with pytest.raises(scb.MissionSequence.TimeSpanError, match="coast"):
            miss_seq.adopt_propagated_grid(0, self.grid_short_by(epochs, 1.0))

    def test_tolerance_is_settable_after_construction(self, ias15_leg, epochs):
        miss_seq = self.deferred_sequence(ias15_leg, epochs)
        grid = self.grid_short_by(epochs, 0.5)

        with pytest.raises(scb.MissionSequence.TimeSpanError):
            miss_seq.adopt_propagated_grid(0, grid)

        miss_seq.grid_bound_tol = 1.0
        miss_seq.adopt_propagated_grid(0, grid)
        assert miss_seq.epochs_vec[0] is grid

    @pytest.mark.parametrize("bad", [-1e-6, float("nan"), float("inf"), "1e-6", None, True])
    def test_invalid_tolerance_raises(self, bad):
        with pytest.raises(scb.MissionSequence.GridToleranceError):
            scb.MissionSequence("BAD TOLERANCE", grid_bound_tol=bad)


# fix this
# def test_add_finite_burn(epochs, origin, state_dict, propagator, leg_model_fb):
#     """
#     Verifies that legs are added to the sequence correctly.
#     """
#     leg1_model = scb.StateArray(
#         epoch=epochs[0],
#         origin=origin,
#         state=scb.StateDefinition.from_components(state_dict),
#     )

#     miss_seq = scb.MissionSequence("TEST SEQUENCE")
#     miss_seq.addLeg(
#         "LEG 1",
#         scb.ArrayWUnits(58000, sec),
#         leg1_model,
#         propagator,
#         scb.ArrayWUnits(360, sec),
#     )
#     miss_seq.add_finite_burn(
#         "Burn 2",
#         scb.ArrayWUnits(30, sec),
#         leg_model_fb,
#         propagator,
#         scb.ArrayWUnits(1, sec),
#     )
#     assert miss_seq._names == ["LEG 1", "Burn 2"]
