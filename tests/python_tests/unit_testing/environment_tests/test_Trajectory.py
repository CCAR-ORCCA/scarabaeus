# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =======================================#
#  Unit Tests for the Trajectory class  #
# =======================================#
import src.scarabaeus as scb

from types import SimpleNamespace
from pathlib import Path
import os
import pytest
import numpy as np
from scipy.linalg import expm

km, sec, kg, hr = scb.Units.get_units(["km", "sec", "kg", 'hr'])
J2000 = scb.Frame("J2000")


# --------------------#
# region    Fixtures #
# --------------------#


# setup
### use conftest fixture
@pytest.fixture
def epochs():
    """Epoch to define state."""
    return scb.EpochArray.interval('2021 January 1, 00:00:00', '2021 January 7, 00:00:00',
                                   scb.ArrayWUnits(1, hr), sys = 'TDB')


### use conftest fixture
@pytest.fixture
def origin():
    """Origin body to define state."""
    return scb.CelestialBody.from_constants("Earth")


### remove
@pytest.fixture
def sc_area():
    return scb.ArrayWUnits(1e-06, km**2)


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
    vel_0 = scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000)

    return scb.StateArray(epochs[0], origin, state=scb.StateDefinition()
                                                .position(sc, pos_0)
                                                .velocity(sc, vel_0),)


@pytest.fixture
def state_mixed_units(epochs, origin, sc):
    """
    StateArray whose body solves for two parameters with DIFFERENT units.

    A delta-v (km/s) alongside a maneuver-epoch correction (sec) is the
    motivating case, but the same shape occurs for FiniteBurn's dimensionless
    k_thrust next to the angular dRA/dDEC.
    """
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000)

    return scb.StateArray(
        epochs[0],
        origin,
        state=(
            scb.StateDefinition()
            .position(sc, pos_0)
            .velocity(sc, vel_0)
            .param(
                "dv_smooth1",
                sc,
                scb.ArrayWFrame(np.array([1e-3, -2e-3, 3e-3]), km / sec, J2000),
                estimation="estimated",
                dynamics="static",
            )
            .param(
                "dt_dv1",
                sc,
                scb.ArrayWFrame(scb.ArrayWUnits(np.array([-12.5]), sec), None),
                estimation="estimated",
                dynamics="static",
            )
        ),
    )


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#


class TestInitialization:
    def test_from_state_array(self, state, origin, sc):
        """
        Verifies that object is constructed correctly from a StateArray,
        including the derived origin, reference frame, body, position,
        velocity, epoch, and every simple passthrough property (poly_deg,
        parameter_units, offset, STMs, STMs_timestamp,
        propagator_settings_dict, use_saved_stms, spk_path). start_epoch/
        end_epoch are pinned as raising, since `state`'s epoch is scalar
        (0-dimensional) and EpochArray indexing is undefined for that case.
        """
        offset = scb.ArrayWUnits(np.array([1.0, 2.0, 3.0]), km)
        stms = [np.eye(6)]
        stms_ts = [state.epoch]
        prop_settings = {"integrator": "RK4"}

        orbiter_traj = scb.Trajectory(
            state_array=state,
            degree_poly=5,
            offset=offset,
            STMs=stms,
            STMs_timestamp=stms_ts,
            propagator_settings_dict=prop_settings,
            use_saved_stms=True,
        )

        assert orbiter_traj.origin == origin.name
        assert orbiter_traj.ref_frame == J2000
        assert orbiter_traj.body is sc
        assert np.allclose(orbiter_traj.position.values, [7000.0, 0.0, 0.0])
        assert np.allclose(orbiter_traj.velocity.values, [0.0, 7.5, 0.0])
        assert sc.spice_id in orbiter_traj._body_map

        assert orbiter_traj.epoch == state.epoch
        with pytest.raises(IndexError):
            orbiter_traj.start_epoch
        with pytest.raises(IndexError):
            orbiter_traj.end_epoch

        assert orbiter_traj.poly_deg == 5
        assert orbiter_traj.parameter_units is None  # `state` has no non-kinematic params
        assert orbiter_traj.offset is offset
        assert orbiter_traj.STMs is stms
        assert orbiter_traj.STMs_timestamp is stms_ts
        assert orbiter_traj.propagator_settings_dict is prop_settings
        assert orbiter_traj.spk_path is None

        assert orbiter_traj.use_saved_stms is True
        orbiter_traj.use_saved_stms = 0
        assert orbiter_traj.use_saved_stms is False

    def test_bare(self):
        """
        Verifies the bare construction path (neither state_array nor
        leg_array provided): attributes are assigned directly from the
        given ref_frame/origin/body arguments, and every simple passthrough
        property (poly_deg, offset, STMs, STMs_timestamp,
        propagator_settings_dict, use_saved_stms, spk_path) still reflects
        its constructor argument, since those are set unconditionally before
        the state_array/leg_array branch is chosen.
        """
        offset = scb.ArrayWUnits(np.array([1.0, 2.0, 3.0]), km)
        stms = [np.eye(6)]
        stms_ts = [scb.EpochArray(0.0, "TDB")]
        prop_settings = {"integrator": "RK4"}

        orbiter_traj = scb.Trajectory(
            ref_frame="J2000",
            origin="EARTH",
            body=None,
            degree_poly=5,
            offset=offset,
            STMs=stms,
            STMs_timestamp=stms_ts,
            propagator_settings_dict=prop_settings,
            use_saved_stms=True,
        )

        assert orbiter_traj.ref_frame == "J2000"
        assert orbiter_traj.origin == "EARTH"
        assert orbiter_traj.body is None

        assert orbiter_traj.poly_deg == 5
        assert orbiter_traj.parameter_units is None
        assert orbiter_traj.offset is offset
        assert orbiter_traj.STMs is stms
        assert orbiter_traj.STMs_timestamp is stms_ts
        assert orbiter_traj.propagator_settings_dict is prop_settings
        assert orbiter_traj.spk_path is None

        assert orbiter_traj.use_saved_stms is True
        orbiter_traj.use_saved_stms = 0
        assert orbiter_traj.use_saved_stms is False

    def test_bare_get_state_reads_spice(self, epochs, origin):
        """
        A bare trajectory over a body already in the SPICE pool has no state
        or sequence of its own: get_state must return the SPICE state with no
        state definition, not fail on attributes only the other branches set.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="SUN", body=origin)
        assert traj.state_definition is None
        assert traj.sequence_definition is None
        assert traj.epoch is None

        state = traj.get_state(epochs[0])
        expected = scb.SpiceManager.get_state(
            "EARTH", float(np.atleast_1d(epochs[0].et)[0]), "J2000", "SUN"
        )
        np.testing.assert_array_equal(state["position"], expected[0:3])
        np.testing.assert_array_equal(state["velocity"], expected[3:6])
        assert state["state_definition"] is None
        assert state["parameters"] is None
        assert state["propagator_settings"] is None


@pytest.mark.parametrize(
    "case",
    [
        "mutual_exclusivity_raises",
        "non_celestial_owner_skipped",
        "missing_values_array_key_skipped",
    ],
    ids=[
        "state_array + leg_array raises",
        "Non-CelestialBody/Spacecraft owner skipped",
        "Missing values_array key skipped (empty body_map raises)",
    ],
)
def test_state_array_init_errors_and_edge_cases(state, origin, sc, case):
    """
    Verifies the StateArray-based init branch's error/edge-case
    handling: mutual exclusivity of state_array/leg_array raises,
    a "position"/"velocity" entry owned by something other than a
    CelestialBody/Spacecraft (e.g. GroundStation) is skipped rather
    than included in _body_map, and a StateArray whose values_array is
    missing an entry's key is likewise skipped -- leaving an empty
    body_map, which raises ValueError.
    """
    match case:
        case "mutual_exclusivity_raises":
            with pytest.raises(ValueError, match="Provide either"):
                scb.Trajectory(state_array=state, leg_array=state)

        case "non_celestial_owner_skipped":
            gs = scb.GroundStation("GS_TRAJ_TEST", spice_id=-9992)
            pos_gs = scb.ArrayWFrame(np.array([1000.0, 0.0, 0.0]), km, J2000)
            sd = (
                scb.StateDefinition.from_components(state.state)
                .position(gs, pos_gs)
            )
            state_with_gs = scb.StateArray(state.epoch, origin, state=sd)

            traj = scb.Trajectory(state_array=state_with_gs)

            assert sc.spice_id in traj._body_map
            assert gs.spice_id not in traj._body_map

        case "missing_values_array_key_skipped":
            # values_array is derived from .state at construction time, so
            # a real StateArray never has a naturally missing entry; force
            # the branch by removing one after construction (values_array
            # returns the live dict, not a copy).
            del state.values_array[("position", sc.spice_id)]

            with pytest.raises(ValueError, match="at least one"):
                scb.Trajectory(state_array=state)


class TestLegArrayConstruction:
    def test_success(self, origin, sc, state, epochs, tmp_path):
        """
        Verifies the success path of leg_array-based construction against a
        real, propagated 2-leg MissionSequence: the auto-inserted "Node"
        between the two legs is skipped, the two "Leg" segments are
        collected with their (position, velocity, epoch, body, origin,
        frame), the second leg's overlapping boundary epoch sample is
        dropped from the concatenated arrays, and traj.epoch/position/
        velocity come straight from the sequence's totals.

        Also writes a real SPK from this trajectory (single-body sequence,
        so this drives _write_segments_only and _write_metadata_only_json),
        and separately calls _write_metadata_only_json directly with an
        empty body_legs to check its early-return guard.
        """
        force_model = scb.ForceModelTranslation(primary_body=sc)
        # `epochs` (a week-long template array) is only used to construct these
        # Propagators; addLeg's explicit period+dt below is what actually
        # decides each leg's output grid (see ScenarioSetup.propagate, which
        # reinitializes the propagator's tspan from the sequence's own
        # per-leg epochs before integrating).
        prop1 = scb.Propagator(primary_body=sc, state_vector=state, tspan=epochs, force_models=force_model)
        prop2 = scb.Propagator(primary_body=sc, state_vector=state, tspan=epochs, force_models=force_model)

        miss_seq = scb.MissionSequence("TRAJ TEST SEQUENCE")
        # period+dt=300s/100s -> 4 samples per leg: [t0, t0+100, t0+200, t0+300]
        miss_seq.addLeg("leg1", scb.ArrayWUnits(300, sec), state, prop1, scb.ArrayWUnits(100, sec))
        miss_seq.addLeg("leg2", scb.ArrayWUnits(300, sec), state, prop2, scb.ArrayWUnits(100, sec))

        # Two adjacent legs auto-insert a boundary "Node" between them (see
        # test_MissionSequence.TestDeferredGrid.test_two_legs_stay_continuous)
        # -- that Node is a real instance of the "skipped, non-propagated leg"
        # case this test exercises.
        assert miss_seq.types == ["Leg", "Node", "Leg"]

        scenario = miss_seq.propagate(verbose=False)

        # degree_poly=2 needs >=3 states per leg; each leg has 4.
        traj = scb.Trajectory(leg_array=scenario, degree_poly=2)

        # the "Node" (sequence index 1) was skipped; only the two "Leg"s remain
        assert [leg["key"] for leg in traj._legs] == ["leg1", "leg2"]
        assert [leg["seq_idx"] for leg in traj._legs] == [0, 2]
        assert all(leg["body"] is sc for leg in traj._legs)

        # leg2's first (duplicate boundary) sample is dropped: 4 + (4 - 1) = 7
        assert len(traj._body) == 7
        assert all(b is sc for b in traj._body)
        assert len(traj._origin) == 7
        assert len(traj._ref_frame) == 7

        assert traj.epoch is scenario.total_epochsTDB
        assert traj.position is scenario.total_position.quantity
        assert traj.velocity is scenario.total_velocity.quantity

        # a genuinely multi-epoch trajectory: start_epoch/end_epoch resolve
        # to the first/last sample rather than raising (unlike the scalar
        # case pinned in test_initialization).
        assert float(traj.start_epoch.et) == pytest.approx(float(scenario.total_epochsTDB.et[0]))
        assert float(traj.end_epoch.et) == pytest.approx(float(scenario.total_epochsTDB.et[-1]))

        # single-body sequence -> write_to_spk drives _write_segments_only
        # (one write_spk_segment_type13 call per leg) and _write_metadata_only_json
        spk_path = str(tmp_path / "legs.bsp")
        traj.write_to_spk(spk_path)
        assert Path(spk_path).exists()

        # get_state without idx_leg_input on a sequence-built trajectory:
        # resolves the leg covering the requested epoch (50s into the
        # sequence is inside leg1's span) and uses its per-leg body/origin/
        # frame plus sequence_definition.models[0].state_definition.
        t0 = float(scenario.total_epochsTDB.et[0])
        result_leg1 = traj.get_state(scb.EpochArray(t0 + 50.0, "TDB"))
        assert result_leg1["body"] == sc.name
        assert result_leg1["origin"] is origin  # leg's origin object, not the trajectory's name-string origin
        assert result_leg1["ref_frame"].name == J2000.name
        assert result_leg1["state_definition"] == state.state_definition

        # explicit idx_leg_input=1 ("leg2") -> idx_leg = 2 * 1 = 2
        result_leg2 = traj.get_state(scb.EpochArray(t0 + 350.0, "TDB"), idx_leg_input=1)
        assert result_leg2["state_definition"] == state.state_definition

        # _write_metadata_only_json's own early-return guard: an empty
        # body_legs writes nothing and doesn't touch seq at all
        assert traj._write_metadata_only_json(
            spk_name=spk_path,
            body_legs=[],
            params_json=None,
            stms_json=None,
            settings_json=None,
            mass_prof=None,
            seq=None,
        ) is None


def test_get_state_leg_fallback_outside_span():
    """
        Verifies get_state's "epoch outside every leg's span" fallback:
        with idx_leg_input=None and a requested epoch outside every leg's
        [t0, tf] window, the leg with the closest boundary is selected
        instead of raising (the for/else on leg_bounds). Patches just the
        two attributes this branch reads (_body as an ndarray, marking a
        sequence-built trajectory, and _legs) directly onto a bare
        Trajectory, rather than constructing a full fake leg_array
        sequence. Uses a real CelestialBody (Earth) as the leg body so
        SpiceManager.get_state succeeds via the kernel pool's real
        ephemeris — this branch can't be reached through a written SPK,
        since its own span *is* exactly the legs' span, so any epoch
        outside it also makes SpiceManager.get_state raise
        SpiceSPKINSUFFDATA first.
    """
    earth = scb.CelestialBody.from_constants("EARTH")
    traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
    traj._body = np.array([earth, earth], dtype=object)  # marks this as sequence-built
    traj._legs = [
        {"body": earth, "origin": earth, "frame": J2000,
         "epoch_leg": scb.EpochArray(np.array([0.0, 100.0]), "TDB")},
        {"body": earth, "origin": earth, "frame": J2000,
         "epoch_leg": scb.EpochArray(np.array([100.0, 200.0]), "TDB")},
    ]
    traj._sequence_definition = SimpleNamespace(
        models=[SimpleNamespace(state_definition="leg1_def"),
                SimpleNamespace(state_definition="unused"),
                SimpleNamespace(state_definition="leg2_def")]
    )

    # far outside both legs' [0, 200] span, but Earth's real ephemeris
    # covers it -> SpiceManager.get_state still succeeds, letting the
    # closest-boundary fallback resolve instead of raising.
    far_epoch = scb.EpochArray(1.0e8, "TDB")
    result = traj.get_state(far_epoch)
    assert result["body"] == earth.name
    assert result["state_definition"] in ("leg1_def", "leg2_def")


class TestNormalizeSTMs:
    @pytest.mark.parametrize(
        "stms, expected",
        [
            pytest.param(None, None, id="none"),
            pytest.param(
                np.array([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]]),
                [np.eye(2), 2.0 * np.eye(2)],
                id="3d_array",
            ),
            pytest.param(
                np.array([[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 2.0]]),  # (N=2, n*n=4)
                [np.eye(2), 2.0 * np.eye(2)],
                id="flattened_time_series",
            ),
            pytest.param(
                np.array([[1.0, 0.0], [0.0, 1.0]]),  # single (n, n) matrix
                [np.eye(2)],
                id="single_matrix",
            ),
            pytest.param(
                np.array([1.0, 0.0, 0.0, 1.0]),  # single flattened (n*n,) vector
                [np.eye(2)],
                id="single_flattened_vector",
            ),
        ],
    )
    def test_supported_shapes(self, stms, expected):
        """
            Verifies _normalize_stms handles None, a homogeneous (N, n, n)
            array, an (N, n*n) flattened time series, a single (n, n)
            matrix, and a single flattened (n*n,) vector.
        """
        result = scb.Trajectory._normalize_stms(stms)
        if expected is None:
            assert result is None
        else:
            assert len(result) == len(expected)
            for got, exp in zip(result, expected):
                assert np.allclose(got, exp)

    def test_heterogeneous_list(self):
        """
            Verifies _normalize_stms' fallback for a list it can't cast to a
            single homogeneous numeric array (e.g. ArrayWFrame-like objects,
            one already (n, n), one flattened and needing a reshape).
        """
        het_list = [
            SimpleNamespace(quantity=SimpleNamespace(values=np.eye(2))),
            SimpleNamespace(quantity=SimpleNamespace(values=np.array([2.0, 0.0, 0.0, 2.0]))),
        ]
        result = scb.Trajectory._normalize_stms(het_list)
        assert np.allclose(result[0], np.eye(2))
        assert np.allclose(result[1], 2.0 * np.eye(2))


def test_stm_units_dict():
    """
        Verifies _stm_units_dict's fallback chain: an explicit
        _STM_units wins; otherwise a single ArrayWFrame-like STM's units
        are used; otherwise the first of a list of ArrayWFrame-like
        STMs; a list-valued units takes its first entry; and no STMs or
        units at all yields None.
    """
    traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)

    # no STMs, no explicit units -> None
    traj._STMs = None
    assert traj._stm_units_dict() is None

    # explicit _STM_units wins over _STMs entirely
    traj._STM_units = km
    assert traj._stm_units_dict() == km.to_dict()

    # fall back to a single ArrayWFrame-like STM's units
    traj._STM_units = None
    traj._STMs = SimpleNamespace(quantity=SimpleNamespace(units=km / sec))
    assert traj._stm_units_dict() == (km / sec).to_dict()

    # fall back to the first of a list of ArrayWFrame-like STMs
    traj._STMs = [
        SimpleNamespace(quantity=SimpleNamespace(units=km)),
        SimpleNamespace(quantity=SimpleNamespace(units=km / sec)),
    ]
    assert traj._stm_units_dict() == km.to_dict()

    # a list-valued _STM_units (e.g. per-leg) takes its first entry
    traj._STM_units = [km, km / sec]
    assert traj._stm_units_dict() == km.to_dict()

    # units with no to_dict() -> None
    traj._STM_units = 42
    assert traj._stm_units_dict() is None


class TestWriteToSpk:
    def test_validation_errors(self, state):
        """
            Verifies write_to_spk's two validation checks, both raised
            before any SPICE I/O is reached: a spk_path not ending in .bsp,
            and a second call with append_if_exists=True (the default) on a
            Trajectory that already has an associated SPK path.
        """
        traj = scb.Trajectory(state_array=state)

        with pytest.raises(ValueError, match=r"must end with \.bsp"):
            traj.write_to_spk("not_a_bsp_file.txt")

        traj._spk_path = "/some/existing/path.bsp"  # simulate a prior write
        with pytest.raises(UserWarning, match="already exists"):
            traj.write_to_spk("/new/path.bsp")
        assert traj._spk_path == "/some/existing/path.bsp"  # unchanged

    def test_state_array_branch(self, origin, sc, tmp_path):
        """
            Writes a real (minimal, degree_poly=1 so a 2-sample state
            satisfies SPK type-13's states-vs-degree requirement) SPK to
            tmp_path, verifying append_if_exists=False deletes a pre-
            existing file at spk_path, associates the new path, and that
            the .bsp file exists on disk afterward.

            Uses a genuinely multi-sample (2-epoch) state with position/
            velocity as (2, 3) arrays: every real write_to_spk call site
            writes a multi-epoch propagated trajectory, and a single-sample
            state hits ArrayWUnits' one-element-collapses-to-0-d behavior
            two different ways (`_state_spice_from_pv`'s 2D-array assumption,
            and `epoch.times.values.tolist()` yielding a bare float instead
            of a list), so `state`'s scalar single epoch (from `epochs[0]`)
            isn't a representative input here.
        """
        two_epochs = scb.EpochArray(np.array([700000000.0, 700003600.0]), "TDB")
        pos_0 = scb.ArrayWFrame(np.array([[7000.0, 0.0, 0.0], [7001.0, 0.1, 0.05]]), km, J2000)
        vel_0 = scb.ArrayWFrame(np.array([[0.0, 7.5, 0.0], [0.0, 7.55, 0.0]]), km / sec, J2000)
        eta_val = scb.ArrayWFrame(scb.ArrayWUnits(np.array([1.2]), None), J2000)
        two_sample_state = scb.StateArray(two_epochs, origin, state=scb.StateDefinition()
                                                                .position(sc, pos_0)
                                                                .velocity(sc, vel_0)
                                                                .param("eta_srp", sc, eta_val, dynamics="static"))

        prop_settings = {"integrator": "RK4"}
        traj = scb.Trajectory(state_array=two_sample_state, degree_poly=1,
                               propagator_settings_dict=prop_settings)

        spk_path = tmp_path / "traj.bsp"
        spk_path.write_text("stale")  # pre-existing file to be deleted, not appended to

        traj.write_to_spk(str(spk_path), append_if_exists=False)

        assert traj._spk_path == str(spk_path)
        assert spk_path.exists()

        # real round trip: get_state reads the just-written SPK/sidecars back
        result = traj.get_state(two_epochs[0])
        assert np.allclose(result["position"].values, [7000.0, 0.0, 0.0], atol=1e-6)
        assert np.allclose(result["velocity"].values, [0.0, 7.5, 0.0], atol=1e-6)
        assert result["body"] == sc.name
        assert result["origin"] == origin.name
        assert result["ref_frame"] == J2000
        assert result["state_definition"] == traj.state_definition
        assert result["parameters"] is not None
        assert result["propagator_settings"] == prop_settings
        assert spk_path.read_bytes() != b"stale"  # the stale placeholder was overwritten

        # non-TDB epoch: converted via .to(sys="TDB") before querying SPICE
        tt_epoch = two_epochs[0].to(sys="TT")
        result_tt = traj.get_state(tt_epoch)
        assert np.allclose(result_tt["position"].values, [7000.0, 0.0, 0.0], atol=1e-6)

        # array-wrapped single epoch: et_vals.ndim is truthy -> converted via .tolist()
        array_epoch = scb.EpochArray(np.array([700000000.0]), "TDB")
        result_arr = traj.get_state(array_epoch)
        assert np.allclose(result_arr["position"].values, [7000.0, 0.0, 0.0], atol=1e-6)

    def test_multi_body_writes_primary_metadata(self, origin, sc, sc_area, sc_ref_coeff, tmp_path):
        """
            Verifies the multi-body branch of write_to_spk writes the
            primary body's metadata sidecar. Previously the metadata write
            (and its `if any(...)` guard) sat outside the per-body for loop
            (same indentation as the `for` statement), so it ran exactly
            once after the loop using whichever body's params/stms/settings/
            mass_profile variables the *last* iteration happened to leave
            behind — almost never the primary body, since legs_by_body
            preserves insertion order and the primary body (self._legs[0]'s
            body) is normally inserted *first*, not last. That silently
            dropped the primary body's settings sidecar entirely.
        """
        class _FakeProps:
            def __init__(self, body):
                self.primary_body = body

        class _FakeMissionSequence:
            def __init__(self, types, names, props, origins, models):
                self.types = types
                self.names = names
                self.props = props
                self.origins = origins
                self.models = models

        class _FakeQuantityWrapper:
            def __init__(self, awu):
                self.quantity = awu

        class _FakeSeq:
            def __init__(self, mission_sequence, total_epochsTDB, segment_positions,
                         segment_velocities, segment_epochsTDB, total_position,
                         total_velocity, len_sequence):
                self.mission_sequence = mission_sequence
                self.total_epochsTDB = total_epochsTDB
                self.segment_positions = segment_positions
                self.segment_velocities = segment_velocities
                self.segment_epochsTDB = segment_epochsTDB
                self.total_position = total_position
                self.total_velocity = total_velocity
                self.total_parameters = None
                self.STM = None
                self.len_sequence = len_sequence

        sc2 = scb.Spacecraft("ORCCA2", -5678, area=sc_area, ref_coeff=sc_ref_coeff)

        leg1_epochs = np.array([0.0, 100.0, 200.0])
        leg2_epochs = np.array([0.0, 100.0, 200.0])
        leg1_pos = scb.ArrayWFrame(np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]]), km, J2000)
        leg1_vel = scb.ArrayWFrame(np.array([[0, 1.0, 0], [0, 2.0, 0], [0, 3.0, 0]]), km / sec, J2000)
        leg2_pos = scb.ArrayWFrame(np.array([[4.0, 0, 0], [5.0, 0, 0], [6.0, 0, 0]]), km, J2000)
        leg2_vel = scb.ArrayWFrame(np.array([[0, 4.0, 0], [0, 5.0, 0], [0, 6.0, 0]]), km / sec, J2000)

        # sc (primary, first leg) then sc2 -> primary is inserted first into
        # legs_by_body, so the bug's "last iteration" would leave sc2's
        # (all-None) metadata behind instead.
        mseq = _FakeMissionSequence(
            types=["Coast", "Coast"],
            names=["leg1", "leg2"],
            props=[_FakeProps(sc), _FakeProps(sc2)],
            origins=[origin, origin],
            models=[SimpleNamespace(state_definition="leg1"), SimpleNamespace(state_definition="leg2")],
        )
        total_epochs = scb.EpochArray(np.concatenate([leg1_epochs, leg2_epochs]), "TDB")
        seq = _FakeSeq(
            mission_sequence=mseq,
            total_epochsTDB=total_epochs,
            segment_positions={"leg1": leg1_pos, "leg2": leg2_pos},
            segment_velocities={"leg1": leg1_vel, "leg2": leg2_vel},
            segment_epochsTDB={"leg1": leg1_epochs, "leg2": leg2_epochs},
            total_position=_FakeQuantityWrapper(scb.ArrayWUnits(leg1_pos.quantity.values, km)),
            total_velocity=_FakeQuantityWrapper(scb.ArrayWUnits(leg1_vel.quantity.values, km / sec)),
            len_sequence=2,
        )

        prop_settings = {"integrator": "RK4"}
        traj = scb.Trajectory(leg_array=seq, degree_poly=2, propagator_settings_dict=prop_settings)
        assert len({leg["body"].spice_id for leg in traj._legs}) == 2  # genuinely multi-body

        spk_path = tmp_path / "multibody.bsp"
        traj.write_to_spk(str(spk_path))

        settings_sidecar = tmp_path / "multibody_settings.json"
        assert settings_sidecar.exists()


def test_get_state_leg_specific_parameters_and_settings(origin, sc, tmp_path):
    """
        Verifies get_state's idx_leg_input-specific branches for
        parameters and propagator settings: when self.parameters[idx] is
        a valid (non-NaN) ArrayWFrame entry and idx is a key in
        self._propagator_settings_dict, both are read back from the SPK's
        sidecars — SpiceManager.get_parameters/get_propagator_settings
        take the same (body, ref_frame, origin, folder_path_override)
        regardless of idx_leg_input, so a state_array-built Trajectory's
        already-written sidecars are retrofitted here with the leg-array
        in-memory shapes these branches check, rather than reconstructing
        a full fake leg_array sequence (whose JSON export requires a much
        larger duck-typed mission_sequence than get_state itself needs).
    """
    two_epochs = scb.EpochArray(np.array([700000000.0, 700003600.0]), "TDB")
    pos_0 = scb.ArrayWFrame(np.array([[7000.0, 0.0, 0.0], [7001.0, 0.1, 0.05]]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([[0.0, 7.5, 0.0], [0.0, 7.55, 0.0]]), km / sec, J2000)
    eta_val = scb.ArrayWFrame(scb.ArrayWUnits(np.array([1.2]), None), J2000)
    two_sample_state = scb.StateArray(two_epochs, origin, state=scb.StateDefinition()
                                                            .position(sc, pos_0)
                                                            .velocity(sc, vel_0)
                                                            .param("eta_srp", sc, eta_val, dynamics="static"))
    prop_settings = {"integrator": "RK4"}
    traj = scb.Trajectory(state_array=two_sample_state, degree_poly=1,
                           propagator_settings_dict=prop_settings)

    spk_path = tmp_path / "leg_specific.bsp"
    traj.write_to_spk(str(spk_path))

    traj._parameters = [[("eta_srp", eta_val)]]
    traj._propagator_settings_dict = {0: prop_settings}
    traj._sequence_definition = SimpleNamespace(
        models=[SimpleNamespace(state_definition="leg0_state_def")]
    )

    result = traj.get_state(two_epochs[0], idx_leg_input=0)
    assert result["state_definition"] == "leg0_state_def"
    assert result["parameters"] is not None
    assert result["propagator_settings"] == prop_settings


class TestTdbListHelpers:
    def test_ensure_tdb_list(self):
        """
            Verifies _ensure_tdb_list passes a TDB epoch through unchanged
            and converts a non-TDB epoch (via .to(sys="TDB")) before listing.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)

        tdb_epoch = scb.EpochArray(np.array([100.0, 200.0]), "TDB")
        assert traj._ensure_tdb_list(tdb_epoch) == [100.0, 200.0]

        tt_epoch = scb.EpochArray(np.array([100.0, 200.0]), "TT")
        tdb_list = traj._ensure_tdb_list(tt_epoch)
        assert len(tdb_list) == 2
        assert tdb_list != [100.0, 200.0]  # genuinely converted, not just re-labeled

    def test_epochs_tdb_list_passthrough(self):
        """Verifies _epochs_tdb_list passes a TDB epoch through unchanged."""
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
        tdb_epoch = scb.EpochArray(np.array([100.0, 200.0]), "TDB")
        assert traj._epochs_tdb_list(tdb_epoch) == [100.0, 200.0]

    def test_epochs_tdb_list_conversion(self):
        """
            Verifies _epochs_tdb_list converts a non-TDB epoch via
            .to(sys="TDB") before listing (previously it called the
            nonexistent EpochArray.convert_to("TDB"), always raising
            AttributeError for a genuinely non-TDB epoch).
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
        tt_epoch = scb.EpochArray(np.array([100.0, 200.0]), "TT")
        tdb_list = traj._epochs_tdb_list(tt_epoch)
        assert len(tdb_list) == 2
        assert tdb_list != [100.0, 200.0]  # genuinely converted, not just re-labeled


def test_state_spice_from_awf():
    """
        Verifies _state_spice_from_awf concatenates position/velocity
        ArrayWFrames into a list of 6-vectors.
    """
    traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
    pos = scb.ArrayWFrame(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), km, J2000)
    vel = scb.ArrayWFrame(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), km / sec, J2000)

    result = traj._state_spice_from_awf(pos, vel)

    assert result == [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3], [4.0, 5.0, 6.0, 0.4, 0.5, 0.6]]


def test_build_leg_data_map(origin, sc):
    """
        Verifies _build_leg_data_map maps each leg's seq_idx to its
        body/origin spice_id and frame name.
    """
    traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
    legs = [
        {"seq_idx": 1, "body": sc, "origin": origin, "frame": J2000},
        {"seq_idx": 3, "body": sc, "origin": origin, "frame": J2000},
    ]

    result = traj._build_leg_data_map(legs)

    assert result == {
        "by_seq_idx": {
            1: {"body_ID": sc.spice_id, "origin_ID": origin.spice_id, "reference_frame": "J2000"},
            3: {"body_ID": sc.spice_id, "origin_ID": origin.spice_id, "reference_frame": "J2000"},
        }
    }


_EPBFS_BODY = SimpleNamespace(spice_id=-1234)
_EPBFS_OTHER_BODY = SimpleNamespace(spice_id=-9999)


def _epbfs_state(epoch_size, state_definition, values_array):
    return SimpleNamespace(
        epoch=SimpleNamespace(size=epoch_size),
        state_definition=state_definition,
        values_array=values_array,
    )


class TestExtractBodyParametersFromState:
    @pytest.mark.parametrize(
        "state_array, target_body",
        [
            pytest.param(None, _EPBFS_BODY, id="state_array_none"),
            pytest.param(_epbfs_state(1, [], {}), None, id="target_body_none"),
            pytest.param(_epbfs_state(0, [], {}), _EPBFS_BODY, id="epoch_size_zero"),
            pytest.param(
                _epbfs_state(
                    1,
                    [("eta_srp", 1, "estimated", "static", _EPBFS_OTHER_BODY)],
                    {("eta_srp", _EPBFS_OTHER_BODY.spice_id): (np.array([1.2]), km)},
                ),
                _EPBFS_BODY,
                id="owner_mismatch_skipped",
            ),
            pytest.param(
                _epbfs_state(
                    1,
                    [("eta_srp", 1, "estimated", "static", _EPBFS_BODY)],
                    {},  # values_array missing the matching key
                ),
                _EPBFS_BODY,
                id="missing_values_array_key_skipped",
            ),
        ],
    )
    def test_returns_none_cases(self, state_array, target_body):
        """
            Verifies (None, None) is returned for: no state_array, no
            target_body, a non-positive epoch size, an owner that doesn't
            match target_body (parameter skipped), and a state_definition
            entry with no matching values_array key (also skipped) — leaving
            nothing to stack in every case.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
        assert traj._extract_body_parameters_from_state(state_array, target_body) == (None, None)

    @pytest.mark.parametrize(
        "state_definition, values_array, match",
        [
            pytest.param(
                [("eta_srp", 3, "estimated", "static", _EPBFS_BODY)],
                {("eta_srp", _EPBFS_BODY.spice_id): (np.array([1.2]), km)},  # size 1, declared 3
                "must have size",
                id="static_size_mismatch",
            ),
            pytest.param(
                [("eta_srp", 1, "estimated", "dynamic", _EPBFS_BODY)],
                {("eta_srp", _EPBFS_BODY.spice_id): (np.array([1.2, 1.3, 1.4]), km)},  # length 3, epoch_size 2
                "has length",
                id="dynamic_length_mismatch",
            ),
            pytest.param(
                [("eta_srp", 1, "estimated", "bogus", _EPBFS_BODY)],
                {("eta_srp", _EPBFS_BODY.spice_id): (np.array([1.2, 1.3]), km)},
                "Unknown dynamics type",
                id="unknown_dynamics_type",
            ),
        ],
    )
    def test_raises(self, state_definition, values_array, match):
        """
            Verifies ValueError for a static parameter whose stored size
            doesn't match its declared dimension, a dynamic parameter whose
            stored time length doesn't match epoch.size, and an unrecognized
            dynamics-type token.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
        state_array = _epbfs_state(2, state_definition, values_array)
        with pytest.raises(ValueError, match=match):
            traj._extract_body_parameters_from_state(state_array, _EPBFS_BODY)

    def test_success(self, origin, sc):
        """
            Verifies the success path with a real StateArray: a static
            eta_srp parameter for the target body is extracted, tiled across
            epochs, and returned as a (stacked ArrayWUnits, param_meta) pair.
        """
        two_epochs = scb.EpochArray(np.array([700000000.0, 700003600.0]), "TDB")
        pos_0 = scb.ArrayWFrame(np.array([[7000.0, 0.0, 0.0], [7001.0, 0.1, 0.05]]), km, J2000)
        vel_0 = scb.ArrayWFrame(np.array([[0.0, 7.5, 0.0], [0.0, 7.55, 0.0]]), km / sec, J2000)
        eta_val = scb.ArrayWFrame(scb.ArrayWUnits(np.array([1.2]), None), J2000)
        state_with_param = scb.StateArray(two_epochs, origin, state=scb.StateDefinition()
                                                                .position(sc, pos_0)
                                                                .velocity(sc, vel_0)
                                                                .param("eta_srp", sc, eta_val, dynamics="static"))

        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
        stacked, param_meta = traj._extract_body_parameters_from_state(state_with_param, sc)

        assert len(param_meta) == 1
        assert param_meta[0][0] == "eta_srp"
        assert param_meta[0][1] == 1
        assert param_meta[0][2] == eta_val.quantity.units
        assert np.allclose(stacked.values, [[1.2], [1.2]])  # tiled across both epochs

    @pytest.mark.parametrize(
        "target_body, expected_names, expected_values",
        [
            pytest.param(
                _EPBFS_BODY,
                ["eta_srp", "range_bias"],
                [[1.2, 0.5], [1.2, 0.5]],
                id="first_propagated_body_hosts_station_parameter",
            ),
            pytest.param(
                _EPBFS_OTHER_BODY,
                ["gm"],
                [[3.0], [3.0]],
                id="second_propagated_body_keeps_only_its_own",
            ),
        ],
    )
    def test_unpropagated_owner_goes_to_first_propagated_body(
        self, target_body, expected_names, expected_values
    ):
        """
            Verifies a parameter whose owner is not propagated (a ground
            station's range bias) is written with the first propagated body's
            parameters, and nowhere else, so every parameter in the state
            lands in exactly one sidecar.
        """
        station = SimpleNamespace(spice_id=399014)
        state_definition = [
            ("position", 3, "estimated", "dynamic", _EPBFS_BODY),
            ("velocity", 3, "estimated", "dynamic", _EPBFS_BODY),
            ("position", 3, "estimated", "dynamic", _EPBFS_OTHER_BODY),
            ("velocity", 3, "estimated", "dynamic", _EPBFS_OTHER_BODY),
            ("eta_srp", 1, "estimated", "static", _EPBFS_BODY),
            ("range_bias", 1, "estimated", "static", station),
            ("gm", 1, "estimated", "static", _EPBFS_OTHER_BODY),
        ]
        values_array = {
            ("eta_srp", _EPBFS_BODY.spice_id): (np.array([1.2]), km),
            ("range_bias", station.spice_id): (np.array([0.5]), km),
            ("gm", _EPBFS_OTHER_BODY.spice_id): (np.array([3.0]), km),
        }
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
        stacked, param_meta = traj._extract_body_parameters_from_state(
            _epbfs_state(2, state_definition, values_array), target_body
        )

        assert [m[0] for m in param_meta] == expected_names
        assert np.allclose(stacked.values, expected_values)

    def test_station_parameter_round_trips_through_sidecar(self, origin, sc, tmp_path):
        """
            Verifies a ground station's range bias, estimated alongside the
            spacecraft but never propagated, is written to the spacecraft's
            parameters sidecar and read back by get_state(), so the stacked
            nominal has the same length as the estimated parameter block.
        """
        station = scb.GroundStation("DSS-14")
        epochs = scb.EpochArray(np.arange(0.0, 600.0, 60.0), sys="TDB")
        state = (
            scb.StateDefinition()
            .position(sc, scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000))
            .velocity(sc, scb.ArrayWFrame(np.array([0.0, 7.5460491, 0.0]), km / sec, J2000))
            .param(
                "dv_smooth1",
                sc,
                scb.ArrayWFrame(np.array([1e-3, -2e-3, 3e-3]), km / sec, J2000),
                dynamics="static",
            )
            .param(
                "range_bias_DSS14",
                station,
                scb.ArrayWFrame(np.array([0.005]), km, J2000),
                dynamics="static",
            )
        )
        prop = scb.Propagator(
            primary_body=sc,
            state_vector=scb.StateArray(epochs[0], origin, state=state),
            tspan=epochs,
            force_models=scb.ForceModelTranslation(primary_body=sc),
            propagate_STM=False,
        )
        prop.propagate()

        spk = str(tmp_path / "station_bias.bsp")
        scb.Trajectory(state_array=prop.propagated_state_array).write_to_spk(
            spk, append_if_exists=False
        )

        out = scb.SpiceManager.get_parameters(
            sc.name,
            300.0,
            J2000,
            origin.name,
            folder_path_override=str(tmp_path),
            per_parameter=True,
        )

        assert set(out) == {"dv_smooth1", "range_bias_DSS14"}
        np.testing.assert_allclose(
            np.ravel(out["dv_smooth1"].values), [1e-3, -2e-3, 3e-3], atol=1e-9
        )
        np.testing.assert_allclose(
            float(np.ravel(out["range_bias_DSS14"].values)[0]), 0.005
        )

    def test_dynamic_reshape(self):
        """
            Verifies a "dynamic" parameter whose stored values are a flat 1D
            time history (one scalar sample per epoch, matching epoch_size)
            is reshaped to (epoch_size, 1) before stacking, rather than
            hitting the static-parameter tiling path.
        """
        state_array = _epbfs_state(
            2,
            [("eta_srp", 1, "estimated", "dynamic", _EPBFS_BODY)],
            {("eta_srp", _EPBFS_BODY.spice_id): (np.array([1.2, 1.3]), km)},
        )
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
        stacked, param_meta = traj._extract_body_parameters_from_state(state_array, _EPBFS_BODY)

        assert param_meta[0] == ("eta_srp", 1, km)
        assert np.allclose(stacked.values, [[1.2], [1.3]])  # genuine per-epoch history, not tiled


class TestGetSTMs:
    def test_memory_source(self):
        """
            Verifies get_STMs' in-memory path (use_saved_stms=False, the
            default): raises when no STMs are attached, falls back to the
            first STM when no timestamps are given, and otherwise returns
            the STM nearest the requested epoch.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)

        traj._STMs = None
        with pytest.raises(ValueError, match="No STMs are attached"):
            traj.get_STMs(scb.EpochArray(0.0, "TDB"))

        traj._STMs = [np.eye(2), 2.0 * np.eye(2)]
        traj._STMs_timestamp = None
        result = traj.get_STMs(scb.EpochArray(999.0, "TDB"))
        assert np.allclose(result, np.eye(2))

        traj._STMs_timestamp = [0.0, 100.0]  # raw ET floats, not EpochArray objects
        result = traj.get_STMs(scb.EpochArray(90.0, "TDB"))  # nearer to the second timestamp
        assert np.allclose(result, 2.0 * np.eye(2))

    def test_disk_source_and_tdb_conversion(self, origin, sc, tmp_path):
        """
            Verifies get_STMs' disk-source path (use_saved_stms=True): a
            non-TDB epoch is converted before lookup, folder_path_override
            defaults to this trajectory's own written SPK, and the STM is
            read back via SpiceManager.get_STMs.
        """
        two_epochs = scb.EpochArray(np.array([700000000.0, 700003600.0]), "TDB")
        pos_0 = scb.ArrayWFrame(np.array([[7000.0, 0.0, 0.0], [7001.0, 0.1, 0.05]]), km, J2000)
        vel_0 = scb.ArrayWFrame(np.array([[0.0, 7.5, 0.0], [0.0, 7.55, 0.0]]), km / sec, J2000)
        two_sample_state = scb.StateArray(two_epochs, origin, state=scb.StateDefinition()
                                                                .position(sc, pos_0)
                                                                .velocity(sc, vel_0))
        traj = scb.Trajectory(state_array=two_sample_state, degree_poly=1, use_saved_stms=True)
        traj.add_STMs(
            scb.ArrayWFrame(scb.ArrayWUnits(np.stack([np.eye(6), 2.0 * np.eye(6)]), None), J2000),
            timestamps=two_epochs,
        )

        spk_path = tmp_path / "stm_disk.bsp"
        traj.write_to_spk(str(spk_path))

        # non-TDB epoch, representing the same instant as two_epochs[0]
        tt_epoch = two_epochs[0].to(sys="TT")
        result = traj.get_STMs(tt_epoch)
        assert np.allclose(result, np.eye(6))


class TestPropertySetters:
    def test_offset(self):
        """
            Verifies set_offset accepts a Length-dimensioned ArrayWUnits and
            rejects any other physical type (previously this always raised
            AttributeError instead: ArrayWUnits has no `physical_type`
            attribute, and the codebase's own convention for a position-like
            quantity is `.units.dimensions.name == "Length"`, not "Position").
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)

        good_offset = scb.ArrayWUnits(np.array([1.0, 2.0, 3.0]), km)
        traj.set_offset(good_offset)
        assert traj.offset is good_offset

        bad_offset = scb.ArrayWUnits(np.array([1.0]), sec)
        with pytest.raises(Exception, match="Length"):
            traj.set_offset(bad_offset)

    def test_max_interp_error(self):
        """
            Verifies the max_interp_error setter accepts a positive float,
            rejects a non-positive value, and allows None to disable the check.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)

        traj.max_interp_error = "1e-6"  # coerced to float
        assert traj.max_interp_error == pytest.approx(1e-6)

        with pytest.raises(ValueError, match="must be a positive relative tolerance"):
            traj.max_interp_error = 0.0
        with pytest.raises(ValueError, match="must be a positive relative tolerance"):
            traj.max_interp_error = -1.0

        traj.max_interp_error = None
        assert traj.max_interp_error is None


class TestAddSTMs:
    def test_list_with_explicit_timestamps(self, sc):
        """
            Verifies add_STMs accepts a list of per-leg ArrayWFrame STM
            blocks (each an (N_leg, n, n) time series), computes n_stms as
            the sum of each block's leading dimension, and validates the
            given timestamps' length against that total.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)

        stm_leg1 = scb.ArrayWFrame(scb.ArrayWUnits(np.stack([np.eye(2), 2.0 * np.eye(2)]), None), J2000)  # 2 STMs
        stm_leg2 = scb.ArrayWFrame(scb.ArrayWUnits(np.stack([3.0 * np.eye(2)]), None), J2000)  # 1 STM
        timestamps = scb.EpochArray(np.array([0.0, 100.0, 200.0]), "TDB")  # 3 total

        traj.add_STMs([stm_leg1, stm_leg2], timestamps=timestamps)

        assert len(traj._STMs) == 2
        assert np.allclose(traj._STMs[0], stm_leg1.quantity.values)
        assert np.allclose(traj._STMs[1], stm_leg2.quantity.values)
        assert list(traj._STMs_timestamp) == [0.0, 100.0, 200.0]

        mismatched_timestamps = scb.EpochArray(np.array([0.0, 100.0]), "TDB")  # only 2, need 3
        with pytest.raises(ValueError, match="does not match number of STMs"):
            traj.add_STMs([stm_leg1, stm_leg2], timestamps=mismatched_timestamps)

    def test_single_arrayframe_no_timestamps(self, origin, sc):
        """
            Verifies add_STMs' single-ArrayWFrame (non-sequence) branch: with
            no timestamps given, it falls back to the trajectory's own
            epochs (warning), or raises if the STM/epoch counts disagree.
        """
        two_epochs = scb.EpochArray(np.array([700000000.0, 700003600.0]), "TDB")
        pos_0 = scb.ArrayWFrame(np.array([[7000.0, 0.0, 0.0], [7001.0, 0.1, 0.05]]), km, J2000)
        vel_0 = scb.ArrayWFrame(np.array([[0.0, 7.5, 0.0], [0.0, 7.55, 0.0]]), km / sec, J2000)
        two_sample_state = scb.StateArray(two_epochs, origin, state=scb.StateDefinition()
                                                                .position(sc, pos_0)
                                                                .velocity(sc, vel_0))
        traj = scb.Trajectory(state_array=two_sample_state)

        matching_stms = scb.ArrayWFrame(scb.ArrayWUnits(np.stack([np.eye(2), 2.0 * np.eye(2)]), None), J2000)
        with pytest.warns(UserWarning, match="falling back to trajectory epochs"):
            traj.add_STMs(matching_stms)
        assert list(traj._STMs_timestamp) == list(two_epochs.times.values)

        mismatched_stms = scb.ArrayWFrame(scb.ArrayWUnits(np.stack([np.eye(2)]), None), J2000)  # 1 STM, 2 epochs
        with pytest.raises(ValueError, match="cannot align"):
            traj.add_STMs(mismatched_stms)


class TestGetSTM:
    @pytest.mark.parametrize(
        "stms, timestamp, idx, exc_type, match, expected",
        [
            pytest.param([np.eye(2).ravel(), (2 * np.eye(2)).ravel()], None, 1,
                         None, None, 2.0 * np.eye(2), id="no_timestamps_uses_idx"),
            pytest.param([np.eye(2).ravel()], None, None,
                         ValueError, "Neither STM timestamps nor idx", None, id="no_timestamps_no_idx_raises"),
            pytest.param([np.eye(2).ravel(), (3 * np.eye(2)).ravel()], [0.0, 100.0], 0,
                         None, None, np.eye(2), id="exact_match_at_first_timestamp"),
            pytest.param([np.eye(2).ravel(), (3 * np.eye(2)).ravel()], [0.0, 100.0], 50,
                         None, None, 2.0 * np.eye(2), id="linear_interpolation_midpoint"),
            pytest.param([np.eye(2).ravel(), np.array([1.0, 2.0, 3.0])], None, 1,
                         ValueError, "not a perfect square", None, id="non_square_stm_size"),
        ],
    )
    def test_direct_index_and_interpolation(self, stms, timestamp, idx, exc_type, match, expected):
        """
            Verifies get_STM's direct-indexing path (no timestamps), exact-
            match short-circuit, and linear interpolation between bracketing
            STMs — parametrized since these all share the same call shape.
            `idx` doubles as the query epoch for the timestamp-bearing cases.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
        traj._STMs = stms
        traj._STMs_timestamp = timestamp

        if exc_type is not None:
            with pytest.raises(exc_type, match=match):
                traj.get_STM(epoch=idx if timestamp is not None else 0, idx=idx if timestamp is None else None)
        else:
            result = traj.get_STM(epoch=idx, idx=idx if timestamp is None else None)
            assert np.allclose(result, expected)

    def test_raises_out_of_bounds_and_degenerate_cases(self):
        """
            Verifies get_STM raises for a query before the first timestamp,
            beyond the last, and when the estimated interpolation error
            exceeds max_interp_error -- a Trajectory-level property (not a
            get_STM() kwarg; that API predates max_interp_error's move to a
            trajectory-wide setting). The error-threshold check also needs a
            3+ point grid: with only two points there's no third node to
            estimate curvature from, so _stm_interp_rel_error always returns
            None and the check can never fire, however big the STM jump is.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
        traj._STMs = [np.eye(2).ravel(), (2.0 * np.eye(2)).ravel(), (100.0 * np.eye(2)).ravel()]
        traj._STMs_timestamp = [0.0, 100.0, 200.0]

        with pytest.raises(ValueError, match="before the earliest"):
            traj.get_STM(epoch=-10.0)
        with pytest.raises(ValueError, match="beyond the latest"):
            traj.get_STM(epoch=300.0)

        with pytest.raises(ValueError, match="Estimated relative interpolation error"):
            traj.get_STM(epoch=150.0)  # sharp jump from 2I to 100I far exceeds the default 1e-3 tolerance


class TestGetSTMSequence:
    def test_non_sequence_block(self):
        """
            Verifies get_STM_sequence's "Case A" (self._STMs is not a list):
            direct indexing when no timestamps are given, raising when
            neither timestamps nor idx are available, and linear
            interpolation between timestamped STMs.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)

        traj._STMs = np.array([np.eye(2).ravel(), (3.0 * np.eye(2)).ravel()])
        traj._STMs_timestamp = None
        assert np.allclose(traj.get_STM_sequence(epoch=0.0, idx=1), 3.0 * np.eye(2))
        with pytest.raises(ValueError, match="idx not provided"):
            traj.get_STM_sequence(epoch=0.0)

        traj._STMs_timestamp = np.array([0.0, 100.0])
        assert np.allclose(traj.get_STM_sequence(epoch=50.0), 2.0 * np.eye(2))

        # _reshape_stm rejects a flat STM whose size isn't a perfect square
        traj._STMs = np.array([np.eye(2).ravel(), np.array([1.0, 2.0, 3.0])], dtype=object)
        traj._STMs_timestamp = None
        with pytest.raises(ValueError, match="not a perfect square"):
            traj.get_STM_sequence(epoch=0.0, idx=1)

    def test_block_list_explicit_idx_leg(self):
        """
            Verifies get_STM_sequence's "Case B" (self._STMs is a list of
            per-leg blocks) with an explicitly given idx_leg, using per-leg
            timestamps from self._STMs_timestamp (a list), including the
            out-of-range idx_leg check.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
        traj._STMs = [
            [np.eye(2).ravel(), (3.0 * np.eye(2)).ravel()],
            [(5.0 * np.eye(2)).ravel()],
        ]
        traj._STMs_timestamp = [[0.0, 100.0], [500.0]]

        assert np.allclose(traj.get_STM_sequence(epoch=50.0, idx_leg=0), 2.0 * np.eye(2))

        with pytest.raises(IndexError, match="out of range"):
            traj.get_STM_sequence(epoch=0.0, idx_leg=5)

    def test_no_per_leg_grid(self):
        """
            Verifies get_STM_sequence's no-per-leg-grid handling: with
            self._STMs_timestamp not a list (so _leg_time_grid can't index
            into it per-leg) and no self._legs to derive epoch spans from,
            an explicit idx falls back to direct indexing within the leg,
            and omitting idx raises.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
        traj._STMs = [[(9.0 * np.eye(2)).ravel()]]
        traj._STMs_timestamp = None
        traj._legs = None

        assert np.allclose(traj.get_STM_sequence(epoch=0.0, idx_leg=0, idx=0), 9.0 * np.eye(2))
        with pytest.raises(ValueError, match="No timestamps/epoch grid"):
            traj.get_STM_sequence(epoch=0.0, idx_leg=0)

    def test_block_list_auto_detect_leg(self):
        """
            Verifies get_STM_sequence auto-detects the leg from
            self._legs[*]["epoch_leg"] spans when idx_leg isn't given, and
            raises when no per-leg grid (timestamps or leg epoch spans) is
            available at all. Two throwaway legs precede the real ones — one
            with no epoch_leg at all, one with an empty epoch grid — so the
            auto-detect loop's `continue` on each of those is also exercised
            before it reaches a leg that actually matches.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
        traj._STMs = [
            [],
            [],
            [np.eye(2).ravel(), (3.0 * np.eye(2)).ravel()],
            [(5.0 * np.eye(2)).ravel(), (7.0 * np.eye(2)).ravel()],
        ]
        traj._STMs_timestamp = None
        traj._legs = [
            {"epoch_leg": None},
            {"epoch_leg": SimpleNamespace(times=SimpleNamespace(values=np.array([])))},
            {"epoch_leg": scb.EpochArray(np.array([0.0, 100.0]), "TDB")},
            {"epoch_leg": scb.EpochArray(np.array([100.0, 200.0]), "TDB")},
        ]

        # epoch 50.0 falls in leg 2's [0, 100] span
        assert np.allclose(traj.get_STM_sequence(epoch=50.0), 2.0 * np.eye(2))
        # epoch 150.0 falls in leg 3's [100, 200] span
        assert np.allclose(traj.get_STM_sequence(epoch=150.0), 6.0 * np.eye(2))

        # no legs, no per-leg timestamps, no global timestamp vector -> nothing to detect the leg with
        traj._legs = None
        with pytest.raises(ValueError, match="needs idx_leg OR leg epoch spans"):
            traj.get_STM_sequence(epoch=50.0)

    def test_block_list_global_timestamp_fallback(self):
        """
            Verifies get_STM_sequence's fallback when idx_leg can't be
            resolved but self._STMs_timestamp is a single flat (non-list)
            array spanning all blocks: flattens the blocks and interpolates
            against that global timestamp vector.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
        traj._STMs = [
            [np.eye(2).ravel()],
            [(3.0 * np.eye(2)).ravel()],
        ]
        traj._STMs_timestamp = np.array([0.0, 100.0])  # flat, not a list -> global vector
        traj._legs = None

        assert np.allclose(traj.get_STM_sequence(epoch=50.0), 2.0 * np.eye(2))

    def test_interp_flat_edge_cases(self):
        """
            Verifies _interp_flat's boundary and exact-match handling: an
            empty time grid raises, an unsorted grid raises, a query at/
            before the first timestamp and at/beyond the last timestamp
            either short-circuits (within tol_time) or raises (outside it),
            and an interior query landing exactly on a grid node (either
            side of the bracket) short-circuits instead of interpolating.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)

        traj._STMs = np.array([])
        traj._STMs_timestamp = np.array([])
        with pytest.raises(ValueError, match="Empty STM time grid"):
            traj.get_STM_sequence(epoch=0.0)

        traj._STMs = np.array([np.eye(2).ravel(), (2.0 * np.eye(2)).ravel()])
        traj._STMs_timestamp = np.array([100.0, 0.0])  # decreasing -> unsorted
        with pytest.raises(ValueError, match="not sorted"):
            traj.get_STM_sequence(epoch=50.0)

        traj._STMs = np.array(
            [np.eye(2).ravel(), (2.0 * np.eye(2)).ravel(), (3.0 * np.eye(2)).ravel()]
        )
        traj._STMs_timestamp = np.array([0.0, 100.0, 200.0])

        # exact match at the first timestamp: within tol -> short-circuits
        assert np.allclose(traj.get_STM_sequence(epoch=0.0), np.eye(2))
        # genuinely before the first timestamp (outside tol) -> raises
        with pytest.raises(ValueError, match="before first STM timestamp"):
            traj.get_STM_sequence(epoch=-1000.0)

        # just past the last timestamp but within tol -> short-circuits. An
        # exact match at 200.0 doesn't reach this branch: bisect_left finds
        # it at the last *valid* index (i=2, not i=times.size), so it's
        # handled as an ordinary interior exact match instead.
        assert np.allclose(
            traj.get_STM_sequence(epoch=200.5, tol_time=1.0), 3.0 * np.eye(2)
        )
        # genuinely beyond the last timestamp (outside tol) -> raises
        with pytest.raises(ValueError, match="beyond last STM timestamp"):
            traj.get_STM_sequence(epoch=1000.0)

        # exact match at an interior grid node (times[i])
        assert np.allclose(traj.get_STM_sequence(epoch=100.0), 2.0 * np.eye(2))

        # within tol_time of the *lower* bracket node (times[i-1]), but on
        # its upper side, so bisect places the bracket one node later
        assert np.allclose(
            traj.get_STM_sequence(epoch=100.5, tol_time=1.0), 2.0 * np.eye(2)
        )

    def test_interp_rel_error_and_threshold(self):
        """
            Verifies _stm_interp_rel_error's real curvature-based error
            estimate (not just its "too few points" None fallback), driven
            entirely through get_STM_sequence -- per the module docstring,
            this internal helper is exercised through the public method
            rather than called directly. A 3-point grid queried in the first
            bracket exercises the "look one point ahead" branch, the second
            bracket exercises the "look two points back" branch, a
            near-linear grid returns without raising (normal, non-degenerate
            scale), an all-zero grid exercises the degenerate scale<=0
            fallback, and a sharply curved grid exceeds max_interp_error and
            raises.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
        traj._STMs_timestamp = np.array([0.0, 100.0, 200.0])

        # near-linear (zero curvature): both brackets interpolate cleanly,
        # exercising the "look ahead" (i=1) and "look back" (i=2) branches
        traj._STMs = np.array(
            [np.eye(2).ravel(), (2.0 * np.eye(2)).ravel(), (3.0 * np.eye(2)).ravel()]
        )
        assert np.allclose(traj.get_STM_sequence(epoch=50.0), 1.5 * np.eye(2))
        assert np.allclose(traj.get_STM_sequence(epoch=150.0), 2.5 * np.eye(2))

        # all-zero STMs: the interpolated matrix has zero norm -> degenerate
        # scale<=0 fallback returns the (also zero) absolute error directly
        traj._STMs = np.array([np.zeros((2, 2)).ravel()] * 3)
        assert np.allclose(traj.get_STM_sequence(epoch=150.0), np.zeros((2, 2)))

        # sharp curvature: estimated relative error exceeds max_interp_error
        # (the trajectory's default, 1e-3)
        traj._STMs = np.array(
            [np.eye(2).ravel(), (2.0 * np.eye(2)).ravel(), (100.0 * np.eye(2)).ravel()]
        )
        with pytest.raises(ValueError, match="Estimated relative interpolation error"):
            traj.get_STM_sequence(epoch=150.0)

        # duplicate leading timestamp: the curvature lookup's third node
        # (i-2) lands on it, making ta == tb degenerate -> no error estimate,
        # so the (otherwise sharp) jump between the duplicate and the next
        # point can't be checked and interpolation proceeds unchecked
        traj._STMs_timestamp = np.array([0.0, 0.0, 100.0, 200.0])
        traj._STMs = np.array(
            [
                np.eye(2).ravel(),
                np.eye(2).ravel(),
                (2.0 * np.eye(2)).ravel(),
                (3.0 * np.eye(2)).ravel(),
            ]
        )
        assert np.allclose(traj.get_STM_sequence(epoch=50.0), 1.5 * np.eye(2))

    def test_no_per_leg_grid_idx_fallback(self):
        """
            Verifies get_STM_sequence's last-resort direct-indexing fallback
            when the requested leg has no derivable time grid (self._legs is
            None and self._STMs_timestamp isn't a per-leg list) but idx is
            provided. Distinct from the (self._STMs_timestamp is None) direct
            shortcut taken earlier in the function: here self._STMs_timestamp
            is a plain array, so that shortcut is skipped and times_leg
            genuinely resolves to None before the idx fallback is reached.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
        traj._STMs = [[(9.0 * np.eye(2)).ravel()]]
        traj._STMs_timestamp = np.array([0.0])
        traj._legs = None

        assert np.allclose(
            traj.get_STM_sequence(epoch=0.0, idx_leg=0, idx=0), 9.0 * np.eye(2)
        )


class TestGetOffsetPosition:
    def test_no_transform(self, origin, sc):
        """
            Verifies get_offset_position returns the raw stored offset
            unchanged when neither ref_frame_input nor origin_input is given.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
        offset = scb.ArrayWUnits(np.array([1.0, 2.0, 3.0]), km)
        traj.set_offset(offset)

        result = traj.get_offset_position(scb.EpochArray(700000000.0, "TDB"))
        assert result is offset

    def test_frame_rotation_dcm_lookup(self):
        """
            Verifies the real SpiceManager.get_xfrm DCM lookup itself
            succeeds (correct scalar-epoch and frame-name plumbing) when
            ref_frame_input is given. Stops short of asserting a rotated
            result: get_offset_position's next line calls
            ArrayWUnits.rotate_dcm, which doesn't exist anywhere in the
            codebase (grep finds no definition) — a missing implementation,
            not a naming bug, so this pins the AttributeError rather than
            guessing at what that method should do.
        """
        traj = scb.Trajectory(ref_frame="J2000", origin="EARTH", body=None)
        traj.set_offset(scb.ArrayWUnits(np.array([1.0, 0.0, 0.0]), km))

        with pytest.raises(AttributeError, match="rotate_dcm"):
            traj.get_offset_position(
                scb.EpochArray(700000000.0, "TDB"), ref_frame_input="ECLIPJ2000"
            )

    def test_frame_rotation_dcm_lookup_with_frame_object(self, origin):
        """
            Same DCM-lookup check as above, but for a StateArray-derived
            Trajectory whose self.ref_frame is a real Frame object rather
            than a plain string — get_xfrm needs a plain frame-name string,
            so this exercises the .name extraction for that case too.
        """
        traj = scb.Trajectory(state_array=scb.StateArray(
            scb.EpochArray(700000000.0, "TDB"), origin,
            state=scb.StateDefinition()
            .position(origin, scb.ArrayWFrame(np.array([0.0, 0.0, 0.0]), km, J2000))
            .velocity(origin, scb.ArrayWFrame(np.array([0.0, 0.0, 0.0]), km / sec, J2000)),
        ))
        traj.set_offset(scb.ArrayWUnits(np.array([1.0, 0.0, 0.0]), km))

        with pytest.raises(AttributeError, match="rotate_dcm"):
            traj.get_offset_position(
                scb.EpochArray(700000000.0, "TDB"),
                origin_input="Sun", ref_frame_input="ECLIPJ2000",
            )


class TestGetSTMJSON:
    def test_round_trip(self, origin, sc, tmp_path):
        """
            Real round trip for get_STM_JSON: writes an SPK for a state with
            STMs and propagator settings attached, then queries both back
            (idx_leg_input=None -> the general, not leg-specific, branches).
            Previously this always raised AttributeError (epoch_input.time
            doesn't exist on EpochArray, only .times), and passed a list
            where SpiceManager.get_STMs expects a scalar epoch.
        """
        two_epochs = scb.EpochArray(np.array([700000000.0, 700003600.0]), "TDB")
        pos_0 = scb.ArrayWFrame(np.array([[7000.0, 0.0, 0.0], [7001.0, 0.1, 0.05]]), km, J2000)
        vel_0 = scb.ArrayWFrame(np.array([[0.0, 7.5, 0.0], [0.0, 7.55, 0.0]]), km / sec, J2000)
        two_sample_state = scb.StateArray(two_epochs, origin, state=scb.StateDefinition()
                                                                .position(sc, pos_0)
                                                                .velocity(sc, vel_0))
        prop_settings = {"integrator": "RK4"}
        traj = scb.Trajectory(state_array=two_sample_state, degree_poly=1,
                               propagator_settings_dict=prop_settings)
        traj.add_STMs(
            scb.ArrayWFrame(scb.ArrayWUnits(np.stack([np.eye(6), 2.0 * np.eye(6)]), None), J2000),
            timestamps=two_epochs,
        )

        spk_path = tmp_path / "stm_traj.bsp"
        traj.write_to_spk(str(spk_path))

        result = traj.get_STM_JSON(two_epochs[0])
        assert np.allclose(result["STM"], np.eye(6))
        assert result["propagator_settings"] == prop_settings

    def test_leg_specific(self, origin, sc, tmp_path):
        """
            Verifies get_STM_JSON's idx_leg_input-specific branches for STM
            and propagator settings, and its leg-specific state_definition
            resolution. SpiceManager.get_STMs/get_propagator_settings take
            the same (body, ref_frame, origin, folder_path_override)
            regardless of idx_leg_input, so this retrofits a state_array-
            built Trajectory's already-written sidecars with the leg-array
            in-memory shapes these branches gate on (self.STMs as a dict
            keyed by leg index, self._propagator_settings_dict likewise),
            rather than reconstructing a full fake leg_array sequence.
        """
        two_epochs = scb.EpochArray(np.array([700000000.0, 700003600.0]), "TDB")
        pos_0 = scb.ArrayWFrame(np.array([[7000.0, 0.0, 0.0], [7001.0, 0.1, 0.05]]), km, J2000)
        vel_0 = scb.ArrayWFrame(np.array([[0.0, 7.5, 0.0], [0.0, 7.55, 0.0]]), km / sec, J2000)
        two_sample_state = scb.StateArray(two_epochs, origin, state=scb.StateDefinition()
                                                                .position(sc, pos_0)
                                                                .velocity(sc, vel_0))
        prop_settings = {"integrator": "RK4"}
        traj = scb.Trajectory(state_array=two_sample_state, degree_poly=1,
                               propagator_settings_dict=prop_settings)
        traj.add_STMs(
            scb.ArrayWFrame(scb.ArrayWUnits(np.stack([np.eye(6), 2.0 * np.eye(6)]), None), J2000),
            timestamps=two_epochs,
        )

        spk_path = tmp_path / "stm_json_leg.bsp"
        traj.write_to_spk(str(spk_path))

        traj._STMs = {0: np.eye(6)}
        traj._propagator_settings_dict = {0: prop_settings}
        traj._sequence_definition = SimpleNamespace(
            models=[SimpleNamespace(state_definition="leg0_state_def")]
        )

        result = traj.get_STM_JSON(two_epochs[0], idx_leg_input=0)
        assert np.allclose(result["STM"], np.eye(6))
        assert result["state_definition"] == "leg0_state_def"
        assert result["propagator_settings"] == prop_settings


# NOTE: the rest of this class' methods are tested in integration tests


# Mixed-unit estimated parameters
class TestMixedUnitParameters:
    def test_mixed_unit_parameters_are_accepted(self, state_mixed_units):
        """
        Parameters of one body must not be required to share units.

        This previously raised "Inconsistent units across parameters", which made
        it impossible to solve for a delta-v and a maneuver epoch together.
        """
        traj = scb.Trajectory(state_array=state_mixed_units)

        # Values stay stacked, so every existing consumer keeps its shape.
        assert traj.parameters is not None
        assert traj.parameters.values.shape[1] == 4

        # The honest per-parameter units are reported separately.
        names = [n for n, _, _ in traj.parameter_units]
        dims = [d for _, d, _ in traj.parameter_units]
        assert names == ["dv_smooth1", "dt_dv1"]
        assert dims == [3, 1]

        # No single unit describes the block, so the stacked array is left
        # dimensionless rather than mislabelled with whichever unit came first.
        assert not np.any(traj.parameters.units.dimensions.powers)

    def test_uniform_unit_parameters_keep_shared_units(self, epochs, origin, sc):
        """The uniform case is unchanged: the stacked array still carries units."""
        traj = scb.Trajectory(
            state_array=scb.StateArray(
                epochs[0],
                origin,
                state=(
                    scb.StateDefinition()
                    .position(sc, scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000))
                    .velocity(sc, scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000))
                    .param(
                        "dv_smooth1",
                        sc,
                        scb.ArrayWFrame(np.array([1e-3, -2e-3, 3e-3]), km / sec, J2000),
                        estimation="estimated",
                        dynamics="static",
                    )
                ),
            )
        )

        assert traj.parameters.units == (km / sec)
        assert [n for n, _, _ in traj.parameter_units] == ["dv_smooth1"]

    def test_mixed_unit_parameters_round_trip_through_sidecar(self, origin, sc, tmp_path):
        """
        The sidecar must record units per parameter, not one unit per file.

        Reading back a mixed-unit body yields the parameters individually, each
        with its own units, since no single ArrayWUnits could describe them.
        """
        epochs = scb.EpochArray(np.arange(0.0, 600.0, 60.0), sys="TDB")
        state = (
            scb.StateDefinition()
            .position(sc, scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000))
            .velocity(sc, scb.ArrayWFrame(np.array([0.0, 7.5460491, 0.0]), km / sec, J2000))
            .param(
                "dv_smooth1",
                sc,
                scb.ArrayWFrame(np.array([1e-3, -2e-3, 3e-3]), km / sec, J2000),
                estimation="estimated",
                dynamics="static",
            )
            .param(
                "dt_dv1",
                sc,
                scb.ArrayWFrame(scb.ArrayWUnits(np.array([-12.5]), sec), None),
                estimation="estimated",
                dynamics="static",
            )
        )
        prop = scb.Propagator(
            primary_body=sc,
            state_vector=scb.StateArray(epochs[0], origin, state=state),
            tspan=epochs,
            force_models=scb.ForceModelTranslation(primary_body=sc),
            propagate_STM=False,
        )
        prop.propagate()

        spk = str(tmp_path / "mixed_units.bsp")
        scb.Trajectory(state_array=prop.propagated_state_array).write_to_spk(
            spk, append_if_exists=False
        )

        # Default return stays type-stable: one stacked ArrayWUnits, always.
        stacked = scb.SpiceManager.get_parameters(
            sc.name, 300.0, J2000, origin.name, folder_path_override=str(tmp_path)
        )
        assert isinstance(stacked, scb.ArrayWUnits)
        assert np.size(stacked.values) == 4
        # Mixed units have no single honest label, so the stack is dimensionless.
        assert not np.any(stacked.units.dimensions.powers)

        out = scb.SpiceManager.get_parameters(
            sc.name,
            300.0,
            J2000,
            origin.name,
            folder_path_override=str(tmp_path),
            per_parameter=True,
        )

        assert isinstance(out, dict), "per_parameter=True must split by parameter"
        assert set(out) == {"dv_smooth1", "dt_dv1"}
        assert out["dv_smooth1"].units == (km / sec)
        assert out["dt_dv1"].units == sec
        np.testing.assert_allclose(
            np.ravel(out["dv_smooth1"].values), [1e-3, -2e-3, 3e-3], atol=1e-9
        )
        np.testing.assert_allclose(
            float(np.ravel(out["dt_dv1"].values)[0]), -12.5, atol=1e-6
        )


# STM interpolation gate
def _analytic_stm_grid(step, n, size=4):
    """
    A smooth, exactly-known matrix history on a uniform grid.

    Built from ``exp(A t)`` for a fixed ``A``, so the true value at any epoch is
    available in closed form and the gate's estimate can be checked against a
    real interpolation error rather than against itself.
    """
    rng = np.random.default_rng(3)
    A = rng.normal(scale=0.35, size=(size, size))
    times = np.arange(n, dtype=float) * step
    flats = [expm(A * t).ravel() for t in times]
    return times, flats, A


def _trajectory_with_stms(state, epochs, sc, step_seconds, hours):
    """Propagate a short arc and hang its STMs off a Trajectory."""
    grid = scb.EpochArray.interval(
        float(np.asarray(epochs.et).ravel()[0]),
        float(np.asarray(epochs.et).ravel()[0]) + hours * 3600.0,
        scb.ArrayWUnits(step_seconds, sec),
        sys="TDB",
    )
    propagator = scb.Propagator(
        primary_body=sc,
        state_vector=state,
        tspan=grid,
        force_models=scb.ForceModelTranslation(primary_body=sc),
        display_progress=False,
    )
    propagator.propagate()
    trajectory = scb.Trajectory("gate", state_array=propagator.propagated_state_array)
    trajectory.add_STMs(propagator.STM, timestamps=grid)
    return trajectory, np.asarray(grid.et).ravel()


class TestSTMInterpolationGate:
    """
    Every get_STM / get_STM_sequence lookup that lands between two stored epochs
    interpolates linearly and rejects the result when the estimated error exceeds
    max_interp_error. The estimate is curvature-based and relative to the matrix
    being interpolated: the error of linear interpolation is set by the second
    derivative, not the first, and an STM's Frobenius norm grows with the arc.
    """

    def test_error_estimate_matches_repropagated_truth(self):
        """The reported estimate agrees with the error measured against truth."""
        step = 0.25
        times, flats, A = _analytic_stm_grid(step, 12)

        i = 6
        alpha = 0.5
        reported = scb.Trajectory._stm_interp_rel_error(times, flats, i, alpha)

        t_mid = times[i - 1] + alpha * step
        interpolated = (1.0 - alpha) * flats[i - 1] + alpha * flats[i]
        truth = expm(A * t_mid).ravel()
        measured = np.linalg.norm(interpolated - truth) / np.linalg.norm(truth)

        assert reported == pytest.approx(measured, rel=0.05)

    def test_error_estimate_falls_as_the_square_of_the_step(self):
        """
        Halving the grid step cuts the estimate by about four, which is what makes
        "use a finer grid" an actionable instruction.
        """
        # The bracket must start at the same epoch every time: refining the grid
        # while holding the *index* fixed would sample a different part of the
        # curve and measure nothing about the step.
        t_bracket = 1.2
        estimates = []
        for step in (0.4, 0.2, 0.1):
            i = int(round(t_bracket / step)) + 1
            times, flats, _ = _analytic_stm_grid(step, i + 3)
            assert times[i - 1] == pytest.approx(t_bracket)
            estimates.append(scb.Trajectory._stm_interp_rel_error(times, flats, i, 0.5))

        assert estimates[0] / estimates[1] == pytest.approx(4.0, rel=0.15)
        assert estimates[1] / estimates[2] == pytest.approx(4.0, rel=0.15)

    def test_fine_grid_on_a_long_arc_is_accepted(self, state, epochs, sc):
        """
        A long arc no longer trips the gate at a fine step; the first-difference
        form it replaces failed here regardless of the grid.
        """
        trajectory, times = _trajectory_with_stms(state, epochs, sc, 10.0, hours=8)

        # Probes strictly between nodes, so every one of them interpolates.
        for t in times[1:-1:60]:
            trajectory.get_STM(epoch=float(t) + 4.0)

    def test_coarse_grid_is_rejected(self, state, epochs, sc):
        """A step genuinely too coarse to interpolate is still refused."""
        trajectory, times = _trajectory_with_stms(state, epochs, sc, 900.0, hours=8)

        with pytest.raises(ValueError, match="relative interpolation error"):
            for t in times[1:-1]:
                trajectory.get_STM(epoch=float(t) + 400.0)

    def test_tolerance_can_be_relaxed_and_disabled(self, state, epochs, sc):
        """The trajectory's tolerance is honoured by the lookups, and None skips."""
        trajectory, times = _trajectory_with_stms(state, epochs, sc, 900.0, hours=4)
        probe = float(times[2]) + 400.0

        with pytest.raises(ValueError, match="relative interpolation error"):
            trajectory.get_STM(epoch=probe)

        trajectory.max_interp_error = 1.0
        relaxed = trajectory.get_STM(epoch=probe)

        trajectory.max_interp_error = None
        unchecked = trajectory.get_STM(epoch=probe)

        assert np.allclose(relaxed, unchecked)

    def test_negative_tolerance_is_reported(self, state):
        """A tolerance that is not a positive relative number is refused."""
        trajectory = scb.Trajectory("gate", state_array=state)

        with pytest.raises(ValueError, match="positive relative tolerance"):
            trajectory.max_interp_error = -1.0

    @pytest.mark.parametrize("configured", [1e-2, None])
    def test_tolerance_can_be_set_at_construction(self, state, configured):
        """The constructor argument reaches the property, and defaults to 1e-3."""
        trajectory = scb.Trajectory(
            "gate", state_array=state, max_interp_error=configured
        )

        assert trajectory.max_interp_error == configured
        assert scb.Trajectory("gate", state_array=state).max_interp_error == 1e-3

    def test_two_point_grid_has_no_curvature_to_check(self):
        """With only two nodes the lookup interpolates instead of failing."""
        times = np.array([0.0, 1.0])
        flats = [np.eye(3).ravel(), (2.0 * np.eye(3)).ravel()]

        assert scb.Trajectory._stm_interp_rel_error(times, flats, 1, 0.5) is None


# Skipping the STM sidecar
def _written_trajectory(state, epochs, sc, tmp_path, **write_kwargs):
    """Propagate a short arc and write it, returning the trajectory and paths."""
    trajectory, _ = _trajectory_with_stms(state, epochs, sc, 60.0, hours=1)
    spk = tmp_path / "arc.bsp"
    trajectory.write_to_spk(str(spk), **write_kwargs)
    return trajectory, spk


class TestSkipSTMSidecar:
    """
    The STM sidecar is the largest artifact a trajectory writes: one matrix per
    output epoch, growing with the square of the state size, and an OD run writes
    one per iteration. The filter serves its own STM lookups from the in-memory
    history, so a run that never reads the sidecar back can skip writing it.
    """

    def test_stm_sidecar_written_by_default(self, state, epochs, sc, tmp_path):
        """The default behaviour is unchanged."""
        _written_trajectory(state, epochs, sc, tmp_path)

        assert (tmp_path / "arc_stms.json").exists()

    def test_stm_sidecar_can_be_skipped(self, state, epochs, sc, tmp_path):
        """write_stms=False leaves the SPK and the other sidecars in place."""
        _written_trajectory(state, epochs, sc, tmp_path, write_stms=False)

        assert not (tmp_path / "arc_stms.json").exists()
        assert (tmp_path / "arc.bsp").exists(), "the state history must still be written"

    def test_skipping_the_sidecar_keeps_the_stms_in_memory(self, state, epochs, sc, tmp_path):
        """Only the file is skipped; lookups still work."""
        trajectory, _ = _written_trajectory(
            state, epochs, sc, tmp_path, write_stms=False
        )

        stm = trajectory.get_STM(epoch=float(np.asarray(trajectory.STMs_timestamp)[2]))

        assert stm.shape[0] == stm.shape[1]
        assert np.isfinite(stm).all()

    @pytest.mark.parametrize("configured, expected", [(True, True), (False, False)])
    def test_output_settings_flag_reaches_the_writer(self, configured, expected):
        """OutputSettings.save_trajectory_stms is what gets passed to write_to_spk."""

        class StubSolution:
            output_settings = scb.OutputSettings(save_trajectory_stms=configured)

        formats = scb.SolutionOD._trajectory_write_formats(StubSolution())

        assert formats["write_stms"] is expected


# Multi-body SPK output
class _Owner:
    def __init__(self, name, spice_id):
        self._name = name
        self.name = name
        self.spice_id = spice_id


def _two_body_state_definition():
    """State of [sc pos, sc vel, body pos, body vel, eta_srp, gm]."""
    sc_owner = _Owner("ORBITER", -1092)
    body_owner = _Owner("Earth", 399)
    return [
        ("position", 3, "estimated", "dynamic", sc_owner, None),
        ("velocity", 3, "estimated", "dynamic", sc_owner, None),
        ("position", 3, "estimated", "dynamic", body_owner, None),
        ("velocity", 3, "estimated", "dynamic", body_owner, None),
        ("eta_srp", 1, "estimated", "static", sc_owner, None),
        ("gm", 1, "estimated", "static", body_owner, None),
    ]


class TestMultiBodySpkOutput:
    """
    A trajectory whose state carries more than one propagated body writes one SPK
    segment per body, into a single file by default or into one file per body
    with per_body_spk. Either way each body's parameters sidecar is named after
    that body: they all write _parameters.json, so a shared name left only the
    last body's parameters on disk.
    """

    def test_parameter_entries_excludes_position_and_velocity(self):
        """A second propagated body's pos/vel are not parameters of the first."""
        entries = scb.SpiceManager._parameter_entries(_two_body_state_definition())

        assert [e[0] for e in entries] == ["eta_srp", "gm"], (
            "the old rule skipped the first two entries, which labelled the second "
            "body's position and velocity as parameters"
        )

    @pytest.mark.parametrize(
        "body_id, expected", [(-1092, ["eta_srp"]), (399, ["gm"])]
    )
    def test_parameter_entries_narrow_to_one_body(self, body_id, expected):
        """A sidecar describes its own body's parameters, not every body's."""
        entries = scb.SpiceManager._parameter_entries(
            _two_body_state_definition(), body_id
        )

        assert [e[0] for e in entries] == expected

    @pytest.mark.parametrize(
        "body_id, expected", [(-1092, ["eta_srp", "range_bias"]), (399, ["gm"])]
    )
    def test_parameter_entries_unpropagated_owner_goes_to_first_body(
        self, body_id, expected
    ):
        """A station is never propagated, so its range bias joins the first body's sidecar."""
        state_definition = _two_body_state_definition() + [
            ("range_bias", 1, "estimated", "static", _Owner("DSS-14", 399014), None)
        ]
        entries = scb.SpiceManager._parameter_entries(state_definition, body_id)

        assert [e[0] for e in entries] == expected

    def test_body_tagged_path_is_unique_per_body(self):
        """The per-body path/stem is derived from the body name."""
        tagged = [
            scb.Trajectory._body_tagged_path("/out/mission.bsp", _Owner(n, i))
            for n, i in (("ORBITER", -1092), ("Earth", 399))
        ]

        assert tagged == ["/out/mission_ORBITER.bsp", "/out/mission_Earth.bsp"]
        assert len(set(tagged)) == 2

# endregion Tests #
# --------------#
