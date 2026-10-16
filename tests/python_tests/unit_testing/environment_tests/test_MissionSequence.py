# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ============================================#
#  Unit Tests for the MissionSequence class  #
# ============================================#
import src.scarabaeus as scb
import numpy as np
import pytest

km, kg, sec, hr, day, rad, N, unitless = scb.Units.get_units(
    ["km", "kg", "sec", "hr", "day", "rad", "N", "unitless"])
J2000 = scb.Frame("J2000")

SEGMENT = 600.0

# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture
def origin():
    """Origin body to define state."""
    return scb.CelestialBody.from_constants("Earth")


@pytest.fixture
def sc():
    """Define spacecraft"""
    return scb.Spacecraft(
        "ORCCA",
        -1234,
        tot_mass=scb.ArrayWUnits(1500.0, kg),
        area=scb.ArrayWUnits(1e-06, km**2),
        ref_coeff=1.5,
    )


@pytest.fixture
def build(origin, sc):
    """
    Build a coast / burn / coast / ... sequence of contiguous 600 s segments.

    ``burns`` holds, per burn, the extra state components as
    ``(name, "estimated" | "considered")``; a coast goes before every burn and
    after the last one.
    """

    def pos_vel(first):
        position = np.array([7000.0, 0.0, 0.0]) if first else np.full(3, np.nan)
        velocity = np.array([0.0, 7.5, 0.0]) if first else np.full(3, np.nan)
        return [
            ("position", 3, "estimated", "dynamic", sc, scb.ArrayWFrame(position, km, J2000)),
            ("velocity", 3, "estimated", "dynamic", sc, scb.ArrayWFrame(velocity, km / sec, J2000)),
        ]

    t0 = float(scb.EpochArray("2021 January 1, 00:00:00", sys="TDB").et)

    def state(components, first):
        epoch = scb.EpochArray(t0, sys="TDB") if first else scb.EpochArray(np.full(1, np.nan), sys="TDB", rep="NUM")
        return scb.StateArray(epoch, origin, state=scb.StateDefinition.from_components(components))

    def _build(burns):
        MS = scb.MissionSequence("Seq")
        t = t0
        for k in range(len(burns) + 1):
            epochs = scb.EpochArray.interval(t, t + SEGMENT, n_epochs=11, sys="TDB")
            coast = state(pos_vel(k == 0), k == 0)
            prop = scb.Propagator(primary_body=sc, state_vector=coast, tspan=epochs,
                                  force_models=scb.ForceModelTranslation(primary_body=sc))
            MS.addLeg(f"Coast {k}", state_vector=coast, prop=prop, epochs=epochs)
            t += SEGMENT
            if k == len(burns):
                break

            epochs = scb.EpochArray.interval(t, t + SEGMENT, n_epochs=11, sys="TDB")
            maneuver = scb.Maneuver(scb.ArrayWUnits(0.5, N), scb.ArrayWUnits(3e-4, kg / sec), epochs[0], epochs[-1],
                                    scb.ArrayWUnits(1, None), scb.ArrayWUnits(0, None), scb.ArrayWUnits(0, None))
            burn = state(pos_vel(False) + [
                (name, 1, kind, "static", sc,
                 scb.ArrayWFrame(scb.ArrayWUnits(1.0, unitless), None) if name.startswith("k_thrust")
                 else scb.ArrayWFrame(0.0, rad, J2000))
                for name, kind in burns[k]
            ], False)
            prop = scb.Propagator(primary_body=sc, state_vector=burn, tspan=epochs,
                                  force_models=scb.ForceModelTranslation(primary_body=sc, finite_burn=True,
                                                                         maneuver=maneuver))
            MS.addFiniteBurn(f"Burn {k}", state_vector=burn, prop=prop, epochs=epochs)
            t += SEGMENT
        return MS

    return _build


@pytest.fixture
def two_legs(origin, sc):
    """
    Build a sequence of two legs from ``period`` + ``dt`` pairs, the way
    case 7a does: the first leg starts at 2021-01-01 TDB, the second one
    continues from its end through an automatic boundary node.
    """
    t0 = float(scb.EpochArray("2021 January 1, 00:00:00", sys="TDB").et)

    def components(first):
        position = np.array([7000.0, 0.0, 0.0]) if first else np.full(3, np.nan)
        velocity = np.array([0.0, 7.5, 0.0]) if first else np.full(3, np.nan)
        return [
            ("position", 3, "estimated", "dynamic", sc, scb.ArrayWFrame(position, km, J2000)),
            ("velocity", 3, "estimated", "dynamic", sc, scb.ArrayWFrame(velocity, km / sec, J2000)),
        ]

    def _build(periods, dt):
        MS = scb.MissionSequence("Seq")
        for k, period in enumerate(periods):
            epoch = scb.EpochArray(t0, sys="TDB") if k == 0 else scb.EpochArray(np.full(1, np.nan), sys="TDB", rep="NUM")
            state = scb.StateArray(epoch, origin, state=scb.StateDefinition.from_components(components(k == 0)))
            prop = scb.Propagator(primary_body=sc, state_vector=state,
                                  tspan=scb.EpochArray.interval(t0, t0 + SEGMENT, n_epochs=11, sys="TDB"),
                                  force_models=scb.ForceModelTranslation(primary_body=sc))
            MS.addLeg(f"Leg {k}", period=period, state_vector=state, prop=prop, dt=dt)
        return MS

    return _build


# endregion Fixtures #
# --------------------#

# --------------#
# region Tests #
# --------------#


def names(model):
    return [entry[0] for entry in model.state]


def test_interrupted_group_is_carried_through_the_coast(build, sc):
    """A group declared on two burns is carried, dormant, through the coast between them only."""
    MS = build([[("k_thrust1", "estimated")], [("k_thrust1", "estimated")]])
    middle = MS.names.index("Coast 1")

    dormant = MS.carry_static_parameters()

    assert len(dormant) == len(MS.types)
    assert dormant[middle] == {("k_thrust1", sc.spice_id)}
    assert all(not d for i, d in enumerate(dormant) if i != middle)
    assert names(MS.models[middle]) == ["position", "velocity", "k_thrust1"]
    assert MS.states_n[middle] == 7
    for name in ("Coast 0", "Coast 2"):
        i = MS.names.index(name)
        assert names(MS.models[i]) == ["position", "velocity"]
        assert MS.states_n[i] == 6


def test_carry_is_idempotent(build):
    """A second call returns the same carry and keeps the carrying state object."""
    MS = build([[("k_thrust1", "estimated")], [("k_thrust1", "estimated")]])
    middle = MS.names.index("Coast 1")

    first = MS.carry_static_parameters()
    carrying = MS.models[middle]
    second = MS.carry_static_parameters()

    assert second == first
    assert MS.models[middle] is carrying
    assert names(MS.models[middle]) == ["position", "velocity", "k_thrust1"]


def test_consecutive_group_carries_nothing(build):
    """A group on adjacent burns has no segment in between, so nothing changes."""
    MS = build([[("k_thrust1", "estimated")], [("k_thrust2", "estimated")]])
    before = list(MS.models)

    dormant = MS.carry_static_parameters()

    assert not any(dormant)
    assert all(after is model for after, model in zip(MS.models, before))


def test_carried_estimates_go_before_considered(build, sc):
    """Mixed groups k1 / k2 / k1: carried estimated entries precede the segment's considered ones."""
    group1 = [("k_thrust1", "estimated"), ("dRA1", "considered"), ("dDEC1", "considered")]
    group2 = [("k_thrust2", "estimated"), ("dRA2", "considered"), ("dDEC2", "considered")]
    MS = build([group1, group2, group1])
    burn2 = MS.names.index("Burn 1")
    carried = {("k_thrust1", sc.spice_id), ("dRA1", sc.spice_id), ("dDEC1", sc.spice_id)}

    dormant = MS.carry_static_parameters()

    assert names(MS.models[burn2]) == [
        "position", "velocity", "k_thrust2", "k_thrust1", "dRA2", "dDEC2", "dRA1", "dDEC1"]
    assert [entry[2] for entry in MS.models[burn2].state] == ["estimated"] * 4 + ["considered"] * 4
    assert dormant[burn2] == carried
    # The coasts either side of the group-2 burn carry group 1 but not group 2.
    for name in ("Coast 1", "Coast 2"):
        i = MS.names.index(name)
        assert names(MS.models[i]) == ["position", "velocity", "k_thrust1", "dRA1", "dDEC1"]
        assert dormant[i] == carried
    assert not dormant[MS.names.index("Coast 0")] and not dormant[MS.names.index("Coast 3")]


def test_scenario_union_puts_considered_last(build):
    """A burn's considered parameter goes after the estimated one a later burn adds."""
    MS = build([[("k_thrust1", "considered")], [("k_thrust2", "estimated")]])

    scenario = scb.ScenarioSetup(MS)

    union = scenario.full_state_vector
    assert names(union) == ["position", "velocity", "k_thrust2", "k_thrust1"]
    assert [entry[2] for entry in union.state] == ["estimated"] * 3 + ["considered"]
    offsets = {key[0]: span for key, span in scenario._scenario_index_map.items()}
    assert offsets["k_thrust2"] == (6, 7) and offsets["k_thrust1"] == (7, 8)
def test_period_and_dt_in_other_time_units(two_legs):
    """A period in days and a step in hours give the same grid as in seconds (case 7a)."""
    in_sec = two_legs([scb.ArrayWUnits(2 * 86400.0, sec), scb.ArrayWUnits(86400.0, sec)],
                      scb.ArrayWUnits(3600.0, sec))
    in_day = two_legs([scb.ArrayWUnits(2.0, day), scb.ArrayWUnits(1.0, day)], scb.ArrayWUnits(1.0, hr))
    t0 = float(in_sec.epochs_vec[0][0].et)

    assert in_day.types == ["Leg", "Node", "Leg"]
    for mine, ref in zip(in_day.epochs_vec, in_sec.epochs_vec):
        np.testing.assert_allclose(np.atleast_1d(mine.et), np.atleast_1d(ref.et), rtol=0.0, atol=1e-6)
    assert in_day.periods[1].epoch == pytest.approx(t0 + 2 * 86400.0, abs=1e-6)
    assert float(in_day.epochs_vec[-1][-1].et) == pytest.approx(t0 + 3 * 86400.0, abs=1e-6)
    assert in_day.num_steps[0] == in_sec.num_steps[0] == 48
    for k in (0, 2):
        assert in_day.periods[k].units == sec
        assert float(in_day.periods[k].values) == pytest.approx(float(in_sec.periods[k].values))


@pytest.mark.parametrize("period, dt", [
    (scb.ArrayWUnits(600.0, km), scb.ArrayWUnits(60.0, sec)),
    (scb.ArrayWUnits(600.0, sec), scb.ArrayWUnits(60.0, unitless)),
])
def test_period_or_dt_not_a_time_raises(two_legs, period, dt):
    """A period or step that is not a time is rejected instead of read as seconds."""
    with pytest.raises(scb.MissionSequence.TimeSpanError):
        two_legs([period], dt)


# endregion Tests #
# --------------#
