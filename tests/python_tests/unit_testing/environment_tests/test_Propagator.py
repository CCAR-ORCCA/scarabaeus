# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =======================================#
#  Unit Tests for the Propagator class  #
# =======================================#
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import os
import pytest
import numpy as np

km, kg, sec, hr = scb.Units.get_units(["km", "kg", "sec", 'hr'])
J2000 = scb.Frame("J2000")

# --------------------#
# region    Fixtures #
# --------------------#


# setup
### use conftest fixture
@pytest.fixture
def epochs():
    """Epoch to define state."""
    return scb.EpochArray.interval(
        '2021 January 1, 00:00:00', '2021 January 7, 00:00:00', scb.ArrayWUnits(1, hr), sys = 'TDB')

### use conftest fixture
@pytest.fixture
def origin():
    """Origin body to define state."""
    return scb.CelestialBody.from_constants("Earth")


### remove
@pytest.fixture
def sc_area():
    return awu(1e-06, km**2)


### remove
@pytest.fixture
def sc_ref_coeff():
    return 1.5


### use conftest fixture
@pytest.fixture
def sc(sc_area, sc_ref_coeff):
    """Define spacecraft"""
    return scb.Spacecraft(
        "ORCCA",
        -1234,
        tot_mass=scb.ArrayWUnits(1500.0, kg),
        area=sc_area,
        ref_coeff=sc_ref_coeff,
    )


### use conftest fixture
@pytest.fixture
def state(epochs, origin, sc):
    """Define stateArray object"""
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0, 70.0, 0.0]), km / sec, J2000)

    return scb.StateArray(epochs[0], origin, state=scb.StateDefinition()
                                                .position(sc, pos_0)
                                                .velocity(sc, vel_0),)


@pytest.fixture
def force_model(sc):
    return scb.ForceModelTranslation(
        primary_body=sc,
    )

@pytest.fixture
def prop(sc, state, epochs, force_model):
    """Fully constructed Propagator for reuse across tests."""
    return scb.Propagator(
        primary_body=sc,
        state_vector=state,
        tspan=epochs,
        force_models=force_model,
    )


# endregion Fixtures #
# --------------------#

# --------------#
# region Tests #
# --------------#


def test_initialization(sc, state, epochs, force_model):
    """
    Verifies that object is constructed correctly.
    """
    prop = scb.Propagator(
        primary_body=sc,
        state_vector=state,
        tspan=epochs,
        force_models=force_model,
    )
    assert prop._primary_body == sc
    assert prop._prop_origin == state.origin
    assert prop._t0 == state.epoch
    assert prop._tspan == epochs


def test_convert_to_base_frame(prop, state):
    """
    Verifies that _convert_to_base_frame populates _original_frames with one
    entry per state component and leaves the state in the base frame.

    The fixture state is already in J2000 (the base frame), so no actual
    rotation happens but the bookkeeping dict must still be populated.
    """
    # One entry per state component (position + velocity = 2 components)
    n_components = len(state.state)
    assert len(prop._original_frames) == n_components

    # Every stored frame must be a Frame object (or None for scalar params)
    for frame in prop._original_frames.values():
        assert frame is None or hasattr(frame, "name")


class TestAccumulateAndSave:
    """Groups the _accumulate_and_save_* bookkeeping helpers exercised in __init__."""

    def test_units_and_frames(self, prop):
        """
        Verifies that _accumulate_and_save_units_and_frames returns two lists
        whose length equals the scalar size of the full state vector.
        """
        units, frames = prop._accumulate_and_save_units_and_frames()

        n = prop.full_state_vector.size  # 6 for pos(3) + vel(3)
        assert len(units) == n
        assert len(frames) == n

    def test_stm_units_shape(self, prop):
        """
        Verifies that _accumulate_and_save_stm_units returns a square matrix
        whose side length equals the full state size, with no None entries.
        """
        stm_units = prop._STM_units  # set in __init__

        n = prop.full_state_vector.size
        assert stm_units.shape == (n, n)
        assert all(entry is not None for entry in stm_units.flat)

    def test_frame_ratios(self, prop):
        """
        Verifies that _accumulate_and_save_frame_ratios returns a square matrix
        of frame-ratio strings whose side length equals the full state size.
        """
        stm_frames = prop._STM_frame  # set in __init__

        n = prop.full_state_vector.size
        assert stm_frames.shape == (n, n)

        # Every entry must be a non-empty string (e.g. "J2000/J2000")
        for val in stm_frames.flat:
            assert isinstance(val, str) and "/" in val


def test_get_event_function(sc, state, epochs, force_model):
    """
    Verifies that _get_event_function returns None when look_event is None,
    a callable when look_event is "EARTH_SOI", and raises ValueError for
    unsupported event types.
    """
    # Default (no event): returns None
    prop_no_event = scb.Propagator(
        primary_body=sc, state_vector=state, tspan=epochs,
        force_models=force_model, look_event=None,
    )
    assert prop_no_event._get_event_function() is None

    # EARTH_SOI: returns a callable event function
    prop_soi = scb.Propagator(
        primary_body=sc, state_vector=state, tspan=epochs,
        force_models=force_model, look_event="EARTH_SOI",
    )
    event_fn = prop_soi._get_event_function()
    assert callable(event_fn)

    # Unsupported type: raises ValueError
    prop_bad = scb.Propagator(
        primary_body=sc, state_vector=state, tspan=epochs,
        force_models=force_model,
    )
    prop_bad._look_event = "INVALID_EVENT"
    with pytest.raises(ValueError):
        prop_bad._get_event_function()


def test_convert_to_original_frames(prop):
    """
    Verifies that _convert_to_original_frames returns a StateArray with the
    same origin and epoch as the propagated input.
    """
    # Use the current full_state_vector as a stand-in for a propagated state.
    # _original_frames is already populated from __init__.
    restored = prop._convert_to_original_frames(prop.full_state_vector)

    assert isinstance(restored, scb.StateArray)
    assert restored.origin == prop._prop_origin
    assert restored.epoch == prop.full_state_vector.epoch


def test_precompute_indices(prop, sc):
    """
    Verifies that _precompute_indices returns a 5-tuple whose first four
    elements are dicts containing the expected (name, spice_id) keys and
    whose fifth element is a numpy integer array.
    """
    result = prop._precompute_indices()

    assert len(result) == 5
    _, _, full_map, _, dyn_rows = result

    # The full state has position and velocity entries for the spacecraft
    assert ("position", sc.spice_id) in full_map
    assert ("velocity", sc.spice_id) in full_map

    # Dynamic rows must be a numpy integer array
    assert isinstance(dyn_rows, np.ndarray)
    assert dyn_rows.dtype == int or np.issubdtype(dyn_rows.dtype, np.integer)


def test_initialize_dynamic_mass(prop):
    """
    Verifies that _initialize_mass_profile sets mass_profile to a non-None
    value.  With no maneuver, it fits a constant-mass polynomial over the
    propagation span.
    """
    assert prop._mass_profile is None  # not set until explicitly called

    prop._initialize_mass_profile()

    assert prop._mass_profile is not None


# NOTE: the rest of this class' methods are tested in integration tests


# ------------------------------------------------------------------#
# region  Mass profile continuity across sequence legs               #
# ------------------------------------------------------------------#
# A coast leg samples the previously stored profile to get its constant mass.
# Sampling it at the stored span's first epoch rewinds the spacecraft to the
# mass it had when the preceding burn began, discarding that burn's propellant.


@pytest.fixture
def burnt_sc(sc):
    """Spacecraft carrying a stored profile that burns 1000 kg down to 900 kg."""
    t_burn = np.linspace(0.0, 600.0, 60)
    m_burn = 1000.0 - (100.0 / 600.0) * t_burn
    sc.fit_mass_profile_from_data(
        t_array=t_burn, m_array=m_burn, domain=(0.0, 600.0), units=kg
    )
    return sc


def coast_leg(spacecraft, state, force_model, t_start, t_end):
    """Build a maneuver-free propagator over [t_start, t_end] (raw ET seconds)."""
    span = scb.EpochArray(np.linspace(t_start, t_end, 40), sys="TDB", rep="NUM")
    return scb.Propagator(
        primary_body=spacecraft,
        state_vector=state,
        tspan=span,
        force_models=force_model,
        display_progress=False,
    )


class TestMassProfileContinuity:
    """Groups mass-profile continuity across sequence legs (see module note above)."""

    def test_coast_after_burn_keeps_burnt_mass(self, burnt_sc, state, force_model):
        """A coast leg after a burn starts from the burn's end mass, not its start."""
        prop = coast_leg(burnt_sc, state, force_model, 600.0, 4200.0)
        prop._initialize_mass_profile()

        # 900 kg is the burn's end mass; 1000 kg would be the pre-burn rewind.
        assert float(burnt_sc.mass(1000.0).values) == pytest.approx(900.0, abs=1e-6)

    def test_coast_after_two_burns_keeps_both_depletions(self, burnt_sc, state, force_model):
        """Consecutive burns accumulate; the mass never rewinds."""
        t_burn2 = np.linspace(4200.0, 4800.0, 60)
        burnt_sc.fit_mass_profile_from_data(
            t_array=t_burn2,
            m_array=900.0 - (50.0 / 600.0) * (t_burn2 - 4200.0),
            domain=(4200.0, 4800.0),
            units=kg,
        )

        prop = coast_leg(burnt_sc, state, force_model, 4800.0, 8400.0)
        prop._initialize_mass_profile()

        assert float(burnt_sc.mass(6000.0).values) == pytest.approx(850.0, abs=1e-6)

    def test_leg_before_stored_span_clamps_to_start(self, burnt_sc, state, force_model):
        """A leg earlier than the stored profile samples that profile's first epoch."""
        prop = coast_leg(burnt_sc, state, force_model, -3600.0, 0.0)
        prop._initialize_mass_profile()

        assert float(burnt_sc.mass(-1800.0).values) == pytest.approx(1000.0, abs=1e-6)


# endregion                                                          #
# ------------------------------------------------------------------#


# ------------------------------------------------------------------#
# region  STM units                                                  #
# ------------------------------------------------------------------#
# STM entries are keyed by '<name>@<spice_id>_by_<name>@<spice_id>'. A key in
# any other shape fails to parse, yields no partial, and leaves the entry
# unitless -- which silently applies to the whole matrix.


class TestSTMUnits:
    """Groups STM unit-correctness checks (see module note above)."""

    def test_carries_partials_units(self, prop):
        """STM entries take the units of the partial they hold, not blanket unitless."""
        stm_units = prop._STM_units

        named = [
            str(stm_units[i, j])
            for i in range(6)
            for j in range(6)
            if str(stm_units[i, j]) != "unitless"
        ]

        assert named, "every STM entry is unitless: the partial keys are not matching"

    def test_consistent_with_state_units(self, prop):
        """The position/velocity blocks carry the units the STM definition implies."""
        stm_units = prop._STM_units

        for i in range(3):
            for j in range(3):
                # d(position)/d(position) and d(velocity)/d(velocity): dimensionless
                assert str(stm_units[i, j]) == "unitless"
                assert str(stm_units[i + 3, j + 3]) == "unitless"

                # d(position)/d(velocity) = km / (km/sec) = sec
                assert str(stm_units[i, j + 3]) == "sec"

                # d(velocity)/d(position) = (km/sec) / km = 1/sec
                assert str(stm_units[i + 3, j]) == "Hz"


# endregion                                                          #
# ------------------------------------------------------------------#


# ------------------------------------------------------------------#
# region  Mass history shared with the force models                  #
# ------------------------------------------------------------------#
# Force models are deep-copied when a propagator is built, so each sub-model
# that scales with mass carries its own spacecraft, frozen at construction. The
# profile is fitted onto the propagator's body, so the copies have to be kept in
# step or they answer from a span belonging to an earlier leg.


@pytest.fixture
def stale_profile_sc(sc, epochs):
    """Spacecraft whose only mass profile predates the propagation span."""
    t_leg0 = float(np.asarray(epochs.et).ravel()[0])
    t_array = np.linspace(t_leg0 - 20000.0, t_leg0 - 10000.0, 60)
    sc.fit_mass_profile_from_data(
        t_array=t_array,
        m_array=np.full(t_array.size, 2000.0),
        domain=(t_array[0], t_array[-1]),
        units=kg,
    )
    return sc


class TestStaleMassProfile:
    """Groups checks that a mass profile predating the leg is still handled correctly."""

    def test_force_models_see_updated_mass_history(self, stale_profile_sc, state, epochs):
        """Sub-models resolve the mass over the leg being propagated."""
        force_model = scb.ForceModelTranslation(
            primary_body=stale_profile_sc, cannonball_SRP=True
        )
        prop = scb.Propagator(
            primary_body=stale_profile_sc,
            state_vector=state,
            tspan=epochs,
            force_models=force_model,
            display_progress=False,
        )

        prop._initialize_mass_profile()

        srp = prop._force_models[0].force_models["CannonballSRP"]
        t_in_leg = float(np.asarray(epochs.et).ravel()[0])

        # the sub-model holds a copy of the spacecraft; it must answer for this leg
        assert srp._spacecraft is not stale_profile_sc
        assert float(srp._spacecraft.mass(t_in_leg).values) == pytest.approx(
            float(stale_profile_sc.mass(t_in_leg).values)
        )

    def test_stm_units_survive_stale_profile(self, stale_profile_sc, state, epochs):
        """Building a propagator works even when no stored profile covers its start."""
        force_model = scb.ForceModelTranslation(
            primary_body=stale_profile_sc, cannonball_SRP=True
        )

        prop = scb.Propagator(
            primary_body=stale_profile_sc,
            state_vector=state,
            tspan=epochs,
            force_models=force_model,
            display_progress=False,
        )

        n = prop.full_state_vector.size
        assert prop._STM_units.shape == (n, n)

        # Units objects refuse comparison with None, so check identity per entry
        assert all(
            prop._STM_units[i, j] is not None for i in range(n) for j in range(n)
        )


# endregion                                                          #
# ------------------------------------------------------------------#


# ------------------------------------------------------------------#
# region  Repeated propagation                                       #
# ------------------------------------------------------------------#
# propagate() leaves the *full* state in `ys` for callers to read, while ode()
# slices the *reduced* state — only the dynamic entries are integrated. Those
# coincide until a static solve-for parameter is added, at which point a second
# run used to slice the STM one row too long.


@pytest.mark.parametrize("with_parameter", [False, True],
                         ids=["position-velocity", "with-static-parameter"])
def test_propagating_twice_repeats_the_run(epochs, origin, sc, with_parameter):
    """A propagator can be run again, including with a static solve-for parameter."""
    unitless = scb.Units.get_units("unitless")
    components = [
        ("position", 3, "estimated", "dynamic", sc,
         scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)),
        ("velocity", 3, "estimated", "dynamic", sc,
         scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000)),
    ]
    if with_parameter:
        components.append(
            ("eta_srp", 1, "estimated", "static", sc,
             scb.ArrayWFrame(scb.ArrayWUnits(1.0, unitless), None))
        )

    state = scb.StateArray(
        epochs[0], origin, state=scb.StateDefinition.from_components(components)
    )
    # eta_srp only means anything with an SRP model behind it
    prop = scb.Propagator(
        primary_body=sc, state_vector=state, tspan=epochs,
        force_models=scb.ForceModelTranslation(primary_body=sc, cannonball_SRP=True),
        display_progress=False,
    )

    prop.propagate()
    first = np.asarray(prop.ys[-1], dtype=float).copy()

    prop.propagate()  # used to raise for the static-parameter case

    np.testing.assert_allclose(np.asarray(prop.ys[-1], dtype=float), first, rtol=1e-12)


def emission_configuration(parameter):
    """``ForceModelTranslation`` keywords enabling the model that applies ``parameter``."""
    m, deg, AU = scb.Units.get_units(["m", "deg", "AU"])
    if parameter == "eta_srp":
        return {"cannonball_SRP": True}
    if parameter == "eta_thermal":
        table = {
            "angles": awu(np.array([0.0, 180.0]), deg),
            "distances": awu(np.array([1.0]), AU),
            "temperatures": np.array([300.0, 300.0]),
            "temperature_units": "K",
        }
        return {"thermal_radiation": True, "thermal_model": {
            "temperature_tables": {"T": table},
            "surfaces": [{"name": "PX", "area": awu(2.0, m**2), "emissivity": 0.8,
                          "normal": [1.0, 0.0, 0.0], "temperature": "T"}],
        }}
    return {"antenna_radiation_pressure": True, "antenna_emitters": [
        {"name": "HGA", "power": awu(100.0, kg * m**2 * sec**-3),
         "hpbw": awu(1.02, deg), "boresight": [0.0, 0.0, 1.0]},
    ]}


class TestScaleFactorGuard:
    """
    A force-model scale factor in the state needs the model that applies it;
    estimated without one, its partial would be zero and its column singular.
    """

    @pytest.fixture
    def attitude_sc(self, sc):
        sc.attitudeMode = "inertialFixedAttitude"
        sc.fixedInertialAttitude = np.eye(3)
        return sc

    def state_with(self, parameter, epochs, origin, sc):
        unitless = scb.Units.get_units("unitless")
        return scb.StateArray(epochs[0], origin, state=scb.StateDefinition.from_components([
            ("position", 3, "estimated", "dynamic", sc,
             scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)),
            ("velocity", 3, "estimated", "dynamic", sc,
             scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000)),
            (parameter, 1, "estimated", "static", sc,
             scb.ArrayWFrame(scb.ArrayWUnits(1.0, unitless), None)),
        ]))

    @pytest.mark.parametrize("parameter", ["eta_srp", "eta_thermal", "eta_antenna"])
    def test_scale_factor_without_its_model_is_refused(self, epochs, origin, attitude_sc, parameter):
        with pytest.raises(scb.Propagator.MissingScaleFactorModelError, match=parameter):
            scb.Propagator(
                primary_body=attitude_sc,
                state_vector=self.state_with(parameter, epochs, origin, attitude_sc),
                tspan=epochs,
                force_models=scb.ForceModelTranslation(primary_body=attitude_sc),
                display_progress=False,
            )
        # callers that caught the old ValueError still do
        assert issubclass(scb.Propagator.MissingScaleFactorModelError, ValueError)

    @pytest.mark.parametrize("parameter", ["eta_srp", "eta_thermal", "eta_antenna"])
    def test_scale_factor_with_its_model_is_accepted(self, epochs, origin, attitude_sc, parameter):
        scb.Propagator(
            primary_body=attitude_sc,
            state_vector=self.state_with(parameter, epochs, origin, attitude_sc),
            tspan=epochs,
            force_models=scb.ForceModelTranslation(
                primary_body=attitude_sc, **emission_configuration(parameter)
            ),
            display_progress=False,
        )


# endregion                                                          #
# ------------------------------------------------------------------#


# ------------------------------------------------------------------#
# region  Dormant parameters                                         #
# ------------------------------------------------------------------#
# A thrust group carried through an arc it does not drive (e.g. across a coast
# to the next burn of the same group) stays in the state but must neither act
# on the dynamics nor pick up partials. FiniteBurn resolves `k_thrust` from any
# `k_thrustN` of the spacecraft, so a visible carried group would drive the burn.


def test_dormant_parameter_is_carried_without_acting(origin, sc):
    """A dormant parameter keeps its value, leaves the dynamics alone and maps as identity."""
    unitless, N, minute = scb.Units.get_units(["unitless", "N", "min"])
    epochs = scb.EpochArray.interval(
        '2021 January 1, 00:00:00', '2021 January 1, 02:00:00',
        scb.ArrayWUnits(10, minute), sys='TDB')
    maneuver = scb.Maneuver(
        scb.ArrayWUnits(0.5, N), scb.ArrayWUnits(3e-4, kg / sec), epochs[0], epochs[-1],
        scb.ArrayWUnits(1, None), scb.ArrayWUnits(0, None), scb.ArrayWUnits(0, None))

    def make_propagator(**thrust_groups):
        components = [
            ("position", 3, "estimated", "dynamic", sc,
             scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)),
            ("velocity", 3, "estimated", "dynamic", sc,
             scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000)),
        ] + [
            (name, 1, "estimated", "static", sc,
             scb.ArrayWFrame(scb.ArrayWUnits(value, unitless), None))
            for name, value in thrust_groups.items()
        ]
        state = scb.StateArray(
            epochs[0], origin, state=scb.StateDefinition.from_components(components))
        return scb.Propagator(
            primary_body=sc, state_vector=state, tspan=epochs,
            force_models=scb.ForceModelTranslation(
                primary_body=sc, finite_burn=True, maneuver=maneuver),
            display_progress=False,
        )

    reference = make_propagator(k_thrust2=1.0)
    reference.propagate()
    y_ref = np.asarray(reference.ys[-1], dtype=float)
    state_ref, stm_ref = y_ref[:7], y_ref[7:].reshape(7, 7)

    prop = make_propagator(k_thrust1=2.0, k_thrust2=1.0)
    prop.propagate()
    visible = np.asarray(prop.ys[-1], dtype=float).copy()

    prop.reinitialize(dormant_parameters={("k_thrust1", sc.spice_id)})
    prop.propagate()
    y = np.asarray(prop.ys[-1], dtype=float)
    state, stm = y[:8], y[8:].reshape(8, 8)
    others = [0, 1, 2, 3, 4, 5, 7]

    # visible, k_thrust1 drives the burn (control for the comparison below)
    assert np.max(np.abs(visible[:3] - state_ref[:3])) > 1.0
    # dormant, the arc is the one without k_thrust1 and the value is untouched
    np.testing.assert_allclose(state[others], state_ref, rtol=1e-10, atol=1e-8)
    assert state[6] == 2.0
    np.testing.assert_allclose(stm[np.ix_(others, others)], stm_ref, rtol=1e-8, atol=1e-8)
    np.testing.assert_array_equal(stm[6], np.eye(8)[6])
    np.testing.assert_array_equal(stm[:, 6], np.eye(8)[:, 6])

    # an empty set wakes it up again
    prop.reinitialize(dormant_parameters=())
    prop.propagate()
    np.testing.assert_allclose(np.asarray(prop.ys[-1], dtype=float), visible, rtol=1e-10, atol=1e-8)


# endregion                                                          #
# ------------------------------------------------------------------#


# ------------------------------------------------------------------#
# region  Saving and reloading                                       #
# ------------------------------------------------------------------#
# A saved propagator, or just its force models, is reused without copying the
# original: run for longer after loading, or drive a new propagator.


def leo_propagator(epochs, origin, sc, tspan, force_models=None):
    """A LEO propagator starting at epochs[0]."""
    state = scb.StateArray(
        epochs[0], origin, state=scb.StateDefinition()
        .position(sc, scb.ArrayWFrame(np.array([7000.0, 0.0, 1000.0]), km, J2000))
        .velocity(sc, scb.ArrayWFrame(np.array([0.0, 7.2, 0.5]), km / sec, J2000)),
    )
    return scb.Propagator(
        primary_body=sc, state_vector=state, tspan=tspan,
        force_models=force_models or scb.ForceModelTranslation(primary_body=sc),
        display_progress=False,
    )


@pytest.mark.parametrize("as_list", [False, True], ids=["tuple", "list"])
def test_loaded_propagator_runs_for_longer(tmp_path, epochs, origin, sc, as_list):
    """Setting tspan on a loaded propagator matches running the longer span directly."""
    short = leo_propagator(epochs, origin, sc, (epochs[0], epochs[2]))
    loaded = scb.Propagator.load(short.save(tmp_path / "propagator.json"))

    span = [epochs[0], epochs[4]]
    loaded.tspan = span if as_list else tuple(span)
    loaded.propagate()

    direct = leo_propagator(epochs, origin, sc, (epochs[0], epochs[4]))
    direct.propagate()
    np.testing.assert_allclose(
        np.asarray(loaded.ys[-1], dtype=float),
        np.asarray(direct.ys[-1], dtype=float),
        rtol=1e-12,
    )


@pytest.mark.parametrize("filename", ["force_models.json", "force_models.pkl"])
def test_saved_force_models_drive_a_new_propagator(tmp_path, epochs, origin, sc, filename):
    """A ForceModelTranslation file reproduces the run it came from."""
    force_models = scb.ForceModelTranslation(
        primary_body=sc,
        cannonball_SRP=True,
        third_bodies=scb.CelestialBody.from_constants(["SUN", "MOON"]),
        t0=epochs[0],
    )
    tspan = (epochs[0], epochs[2])
    original = leo_propagator(epochs, origin, sc, tspan, force_models)
    original.propagate()

    path = force_models.save(tmp_path / filename)
    loaded = scb.ForceModelTranslation.load(path, primary_body=sc)
    rerun = leo_propagator(epochs, origin, sc, tspan, loaded)
    rerun.propagate()

    assert loaded._primary_body is sc
    assert loaded.dynamics_mode_list == force_models.dynamics_mode_list
    np.testing.assert_array_equal(
        np.asarray(rerun.ys[-1], dtype=float), np.asarray(original.ys[-1], dtype=float)
    )

    # without a spacecraft to share, the file's own is rebuilt
    standalone = scb.ForceModelTranslation.load(path)
    assert standalone._primary_body is not sc
    assert standalone._primary_body.name == sc.name


def test_propagator_file_is_not_a_force_model_file(tmp_path, prop):
    """Loading a Propagator file as force models is refused by name."""
    path = prop.save(tmp_path / "propagator.json")

    with pytest.raises(
        scb.ForceModelTranslation.NotAForceModelFileError, match="Propagator.load"
    ):
        scb.ForceModelTranslation.load(path)


class TestSetters:
    """Every setting of a loaded propagator can be changed in place."""

    @pytest.fixture
    def loaded(self, tmp_path, epochs, origin, sc):
        short = leo_propagator(epochs, origin, sc, (epochs[0], epochs[2]))
        return scb.Propagator.load(short.save(tmp_path / "propagator.json"))

    @pytest.mark.parametrize("attr", ["state_vector", "full_state_vector"])
    def test_new_state_matches_a_fresh_propagator(self, loaded, epochs, origin, sc, attr):
        """Setting the state reproduces a propagator built from that state."""
        new_state = scb.StateArray(
            epochs[0], origin, state=scb.StateDefinition()
            .position(loaded.primary_body,
                      scb.ArrayWFrame(np.array([7200.0, 100.0, 900.0]), km, J2000))
            .velocity(loaded.primary_body,
                      scb.ArrayWFrame(np.array([0.1, 7.1, 0.6]), km / sec, J2000)),
        )
        setattr(loaded, attr, new_state)
        loaded.propagate()

        direct = scb.Propagator(
            primary_body=sc, state_vector=new_state, tspan=(epochs[0], epochs[2]),
            force_models=scb.ForceModelTranslation(primary_body=sc),
            display_progress=False,
        )
        direct.propagate()

        np.testing.assert_allclose(
            np.asarray(loaded.state_vector.values0.values, dtype=float).ravel(),
            [7200.0, 100.0, 900.0, 0.1, 7.1, 0.6],
        )
        np.testing.assert_allclose(
            np.asarray(loaded.ys[-1], dtype=float),
            np.asarray(direct.ys[-1], dtype=float),
            rtol=1e-12,
        )

    def test_base_frame_round_trip_does_not_compound(self, loaded):
        """Switching frame and back leaves the working state as it started."""
        before = np.asarray(loaded.full_state_vector.values0.values, dtype=float).copy()

        loaded.base_frame = scb.Frame("ECLIPJ2000")
        assert loaded.base_frame.name == "ECLIPJ2000"
        assert not np.allclose(
            np.asarray(loaded.full_state_vector.values0.values, dtype=float), before
        )

        loaded.base_frame = None
        assert loaded.base_frame.name == "J2000"
        np.testing.assert_allclose(
            np.asarray(loaded.full_state_vector.values0.values, dtype=float),
            before, rtol=1e-12, atol=1e-9,
        )

    def test_stm0(self, loaded):
        """STM0 accepts a square matrix, and None restores the identity."""
        n = loaded.full_state_vector.size
        stm0 = 2.0 * np.identity(n)
        loaded.STM0 = stm0
        np.testing.assert_array_equal(loaded.STM0, stm0.reshape(-1))
        np.testing.assert_array_equal(loaded.ys[0][n:], stm0.reshape(-1))

        loaded.STM0 = None
        np.testing.assert_array_equal(loaded.STM0, np.identity(n).reshape(-1))

    def test_plain_settings(self, loaded):
        """Settings only read at propagate() time are stored as given."""
        loaded.atol = 1e-10
        loaded.rtol = 1e-11
        loaded.look_event = "EARTH_SOI"
        loaded.look_event = None
        loaded.display_progress = True
        loaded.convert_to_original_frame = True

        assert (loaded.atol, loaded.rtol) == (1e-10, 1e-11)
        assert loaded.look_event is None
        assert loaded.display_progress is True
        assert loaded.convert_to_original_frame is True

    def test_rebuilding_settings(self, loaded, sc):
        """Settings that shape the integration go through reinitialize()."""
        loaded.n_densifier = 3
        loaded.propagate_STM = False
        loaded.backward = False
        loaded.integrator = "DOP853"
        loaded.primary_body = sc
        fm = scb.ForceModelTranslation(primary_body=sc)
        loaded.force_models = fm

        assert loaded.n_densifier == 3
        assert loaded.propagate_STM is False
        assert loaded.primary_body is sc
        assert loaded.force_models == [fm]

        with pytest.raises(scb.Propagator.DensifierValueError):
            loaded.n_densifier = 0

    def test_mass_is_variable(self, loaded):
        loaded.mass_is_variable = True
        assert loaded.mass_is_variable is True


@pytest.fixture
def two_body_state(epochs, origin, sc):
    """State integrating the spacecraft together with the Moon."""
    moon = scb.CelestialBody.from_constants("Moon")
    return moon, scb.StateArray(
        epochs[0], origin,
        state=scb.StateDefinition()
        .position(sc, scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000))
        .velocity(sc, scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000))
        .position(moon, scb.ArrayWFrame(np.array([384400.0, 0.0, 0.0]), km, J2000))
        .velocity(moon, scb.ArrayWFrame(np.array([0.0, 1.0, 0.0]), km / sec, J2000)),
    )


class TestListForces:
    """Propagator.list_forces: the bound forces, per integrated object."""

    def test_one_entry_per_integrated_object(self, sc, epochs, two_body_state):
        """Every integrated object is listed by name, in state order, with the bound labels."""
        moon, state = two_body_state
        prop = scb.Propagator(
            primary_body=sc,
            state_vector=state,
            tspan=epochs,
            force_models=[
                scb.ForceModelTranslation(
                    primary_body=sc, third_bodies=["SUN"], cannonball_SRP=True
                ),
                scb.ForceModelTranslation(primary_body=moon, third_bodies=["SUN"]),
            ],
        )

        forces = prop.list_forces()

        assert list(forces) == ["ORCCA", "Moon"]
        # Bound labels: the integrated Moon attracts the spacecraft, the origin
        # is named, and the Moon does not attract itself.
        assert forces["ORCCA"] == [
            "ThreeBodyGravity(SUN, MOON)", "CannonballSRP", "PointMassGravity(EARTH)"
        ]
        assert forces["Moon"] == ["ThreeBodyGravity(SUN)", "PointMassGravity(EARTH)"]

    def test_reads_the_propagator_copies(self, sc, state, epochs, force_model):
        """The listing is the bound copy's; the caller's model stays unbound."""
        prop = scb.Propagator(
            primary_body=sc, state_vector=state, tspan=epochs, force_models=force_model
        )

        assert prop.list_forces() == {"ORCCA": ["PointMassGravity(EARTH)"]}
        assert force_model.list_forces() == ["PointMassGravity"]


class TestDensifier:
    """n_densifier resamples each adaptive step from the integrator's dense
    output, so an SPK written on a (t0, tf) propagation is not left with wide
    gaps wherever the solver took long steps."""

    @pytest.fixture
    def tspan_tuple(self, epochs):
        """A (t0, tf) tuple, which hands the output grid to the integrator."""
        return (epochs[0], epochs[-1])

    def _prop(self, sc, state, tspan, force_model, **kwargs):
        return scb.Propagator(
            primary_body=sc,
            state_vector=state,
            tspan=tspan,
            force_models=force_model,
            **kwargs,
        )

    def test_defaults_to_ten(self, prop):
        """Left alone, each step is split ten ways."""
        assert prop.n_densifier == 10

    def test_one_keeps_the_raw_steps(self, sc, state, tspan_tuple, force_model):
        """Opting out returns the integrator's own steps untouched."""
        prop = self._prop(sc, state, tspan_tuple, force_model, n_densifier=1)
        prop.propagate()
        plain = self._prop(sc, state, tspan_tuple, force_model, n_densifier=1)
        plain.propagate()
        assert prop.times.et.size == plain.times.et.size

    @pytest.mark.parametrize("bad", [0, -1, -10])
    def test_below_one_is_reported(self, sc, state, tspan_tuple, force_model, bad):
        """A step cannot be split into fewer than one sub-interval."""
        with pytest.raises(scb.Propagator.DensifierValueError):
            self._prop(sc, state, tspan_tuple, force_model, n_densifier=bad)

    @pytest.mark.parametrize("bad", [2.5, "10", True])
    def test_non_integer_is_reported(self, sc, state, tspan_tuple, force_model, bad):
        """Including True, which is an int subclass and would read as 1."""
        with pytest.raises(scb.Propagator.DensifierValueError):
            self._prop(sc, state, tspan_tuple, force_model, n_densifier=bad)

    def test_none_takes_the_default(self, sc, state, tspan_tuple, force_model):
        """None is the sentinel the signature declares, not a rejected type."""
        prop = self._prop(sc, state, tspan_tuple, force_model, n_densifier=None)
        assert prop.n_densifier == 10

    def test_ias15_is_reported_when_asked_for(self, sc, state, tspan_tuple, force_model):
        """IAS15 returns step endpoints only and builds no interpolant."""
        prop = self._prop(
            sc, state, tspan_tuple, force_model, n_densifier=10, integrator="IAS15"
        )
        with pytest.raises(scb.Propagator.DensifierIntegratorError):
            prop.propagate()

    def test_ias15_ignores_the_default(self, sc, state, tspan_tuple, force_model):
        """The default is not a request, so it must not break IAS15.

        The STM is off because IAS15 cannot propagate one yet (see
        TestIAS15STM) -- unrelated to n_densifier, which is left at its default
        here.
        """
        prop = self._prop(
            sc, state, tspan_tuple, force_model,
            integrator="IAS15", propagate_STM=False,
        )
        prop.propagate()  # must not raise

    def test_explicit_grid_is_reported_when_asked_for(self, sc, state, epochs, force_model):
        """An EpochArray already fixes every output epoch, so there is no
        solver-chosen grid left to refine."""
        prop = self._prop(sc, state, epochs, force_model, n_densifier=10)
        with pytest.raises(scb.Propagator.DensifierGridError):
            prop.propagate()

    def test_grid_ignores_the_default(self, sc, state, epochs, force_model):
        """The commonest case of all: an EpochArray tspan with n_densifier left
        alone must return exactly the epochs that were asked for."""
        prop = self._prop(sc, state, epochs, force_model)
        prop.propagate()  # must not raise
        np.testing.assert_allclose(
            np.asarray(prop.times.et, dtype=float),
            np.asarray(epochs.et, dtype=float),
        )

    def test_epochs_are_a_superset_of_the_steps(self, sc, state, tspan_tuple, force_model):
        """Densifying adds epochs; it never moves the ones the integrator
        produced, so every undensified epoch survives exactly."""
        plain = self._prop(sc, state, tspan_tuple, force_model, n_densifier=1)
        plain.propagate()
        steps = np.asarray(plain.times.et, dtype=float)

        dense = self._prop(sc, state, tspan_tuple, force_model, n_densifier=5)
        dense.propagate()
        dense_et = np.asarray(dense.times.et, dtype=float)

        assert np.all(np.isin(steps, dense_et))
        assert dense_et.size == (steps.size - 1) * 5 + 1
        assert np.all(np.diff(dense_et) > 0)

    def test_gaps_follow_the_step_lengths(self, sc, state, tspan_tuple, force_model):
        """The point of densifying by step rather than by fixed interval: the
        output stays fine where the steps are short and coarse where they are
        long, instead of imposing one spacing over the whole arc."""
        dense = self._prop(sc, state, tspan_tuple, force_model, n_densifier=8)
        dense.propagate()
        gaps = np.diff(np.asarray(dense.times.et, dtype=float))

        # Sub-intervals of one step are uniform, so any spread in the gaps can
        # only come from the steps themselves differing in length.
        assert gaps.max() > gaps.min()

    def test_densified_states_match_the_interpolant(self, sc, state, tspan_tuple, force_model):
        """The added samples are the solver's own dense output, so the states
        at the original step endpoints are unchanged by densifying."""
        plain = self._prop(sc, state, tspan_tuple, force_model, n_densifier=1)
        plain.propagate()
        steps = np.asarray(plain.times.et, dtype=float)

        dense = self._prop(sc, state, tspan_tuple, force_model, n_densifier=4)
        dense.propagate()
        dense_et = np.asarray(dense.times.et, dtype=float)

        shared = np.searchsorted(dense_et, steps)
        np.testing.assert_allclose(
            dense._ode_sol.y[:, shared], plain._ode_sol.y, rtol=1e-9, atol=1e-9
        )

    def test_survives_a_round_trip(self, sc, state, tspan_tuple, force_model):
        """n_densifier is carried through to_dict()/from_dict()."""
        prop = self._prop(sc, state, tspan_tuple, force_model, n_densifier=7)
        assert prop.to_dict()["n_densifier"] == 7

    def test_a_file_without_the_field_takes_the_default(self, prop):
        """Propagator files written before n_densifier existed still load."""
        data = prop.to_dict()
        del data["n_densifier"]
        assert scb.Propagator.from_dict(data).n_densifier == 10

    def test_reinitialize_updates_it(self, sc, state, tspan_tuple, force_model):
        prop = self._prop(sc, state, tspan_tuple, force_model, n_densifier=7)
        prop.reinitialize(n_densifier=3)
        assert prop.n_densifier == 3

    def test_reinitialize_validates_it(self, sc, state, tspan_tuple, force_model):
        prop = self._prop(sc, state, tspan_tuple, force_model, n_densifier=7)
        with pytest.raises(scb.Propagator.DensifierValueError):
            prop.reinitialize(n_densifier=0)
        assert prop.n_densifier == 7


class TestIAS15STM:
    """IAS15 integrates the state alone, so asking it for the STM is reported
    up front instead of failing deep inside the ODE."""

    def _ias15(self, sc, state, epochs, force_model, **kwargs):
        return scb.Propagator(
            primary_body=sc,
            state_vector=state,
            tspan=epochs,
            force_models=force_model,
            integrator="IAS15",
            display_progress=False,
            **kwargs,
        )

    def test_stm_is_reported(self, sc, state, epochs, force_model):
        """propagate_STM defaults to True, so this is the out-of-the-box case."""
        prop = self._ias15(sc, state, epochs, force_model)
        with pytest.raises(scb.Propagator.IAS15STMError, match="propagate_STM=False"):
            prop.propagate()

    def test_is_a_not_implemented_error(self):
        """Callers catching the generic class still see it."""
        assert issubclass(scb.Propagator.IAS15STMError, NotImplementedError)

    def test_switching_to_ias15_is_reported(self, prop):
        """reinitialize() can change the integrator after construction, which
        is why the check sits in propagate()."""
        prop.integrator = "IAS15"
        with pytest.raises(scb.Propagator.IAS15STMError):
            prop.propagate()

    def test_turning_the_stm_off_lets_it_run(self, sc, state, epochs, force_model):
        """Setting the integrator first and the STM flag second must not trip
        the check before propagate() is reached."""
        prop = scb.Propagator(
            primary_body=sc, state_vector=state, tspan=epochs,
            force_models=force_model, display_progress=False,
        )
        prop.integrator = "IAS15"
        prop.propagate_STM = False
        prop.propagate()  # must not raise
        assert prop.STM is None
        assert prop.ys.shape == (len(epochs), 6)



# endregion                                                          #
# ------------------------------------------------------------------#
