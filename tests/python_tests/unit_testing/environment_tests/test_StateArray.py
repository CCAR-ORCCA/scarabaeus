# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
#=======================================#
#  Unit Tests for the StateArray class  #
#=======================================#
import src.scarabaeus as scb
from scarabaeus.utils.Serialization import owner_to_dict
from types import SimpleNamespace
import os
import pytest
import numpy as np

km, sec = scb.Units.get_units(['km', 'sec'])
J2000   = scb.Frame('J2000')


#--------------------#
# region    Fixtures #
#--------------------#

@pytest.fixture
def epoch():
    """Epoch to define state."""
    return scb.EpochArray('2000 JAN 01 12:00:00.000', 'TDB')

@pytest.fixture
def origin():
    """Origin body to define state."""
    return scb.CelestialBody.from_constants('Earth')

@pytest.fixture
def sc_area():
    return scb.ArrayWUnits(1e-06, km**2)


@pytest.fixture
def sc_ref_coeff():
    return 1.5


# construction
@pytest.fixture
def sc(sc_area, sc_ref_coeff):
    """Spacecraft to define the cannonball model for."""
    return scb.Spacecraft("ORCCA", -1234, area=sc_area, ref_coeff=sc_ref_coeff)


@pytest.fixture
def multi_epoch_state_arr(origin, sc):
    """
        Fresh 2-epoch StateArray with position/velocity stored as (2, 3)
        time histories and a static eta_srp parameter, for exercising
        add/remove/update_state_at_epoch.
    """
    two_epochs = scb.EpochArray([700000000.0, 700003600.0], 'TDB')
    pos_hist = scb.ArrayWFrame(
        np.array([[7000.0, 0.0, 0.0], [7001.0, 0.1, 0.05]]), km, J2000
    )
    vel_hist = scb.ArrayWFrame(
        np.array([[0.0, 7.5, 0.0], [0.0, 7.55, 0.0]]), km / sec, J2000
    )
    eta_val = scb.ArrayWFrame(scb.ArrayWUnits(np.array([1.2]), None), J2000)
    return scb.StateArray(two_epochs, origin, state=scb.StateDefinition()
                                                .position(sc, pos_hist)
                                                .velocity(sc, vel_hist)
                                                .param("eta_srp", sc, eta_val, dynamics="static"))

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#


def test_initialization(epoch, origin, sc):
    """
        Verifies that object is constructed correctly.
    """
    # initialize
    pos_0   = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0   = scb.ArrayWFrame(np.array([0.0, 7.546049, 0.0]), km/sec, J2000)  
    state_dict = [
        ("position", 3, "estimated", "dynamic", sc, pos_0),
        ("velocity", 3, "estimated", "dynamic", sc, vel_0),
    ]

    state_arr = scb.StateArray(epoch, origin, state=scb.StateDefinition()
                                                .position(sc, pos_0)
                                                .velocity(sc, vel_0),)

    assert state_arr._epoch == epoch
    assert state_arr._origin == origin
    assert state_arr._state == scb.StateDefinition.from_components(state_dict).components


class TestAddToStateVector:
    def test_append_and_duplicate_raises(self, epoch, origin, sc):
        """
            Verifies that a new component can be appended, that a TypeError is raised
            for an invalid argument type, and that a ValueError is raised for duplicates.
        """
        pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
        vel_0 = scb.ArrayWFrame(np.array([0.0, 7.546049, 0.0]), km / sec, J2000)

        state_arr = scb.StateArray(epoch, origin, state=scb.StateDefinition()
                                                    .position(sc, pos_0)
                                                    .velocity(sc, vel_0),)

        # add eta_srp component via list — method returns a new StateArray
        eta_val = scb.ArrayWFrame(scb.ArrayWUnits(np.array([1.2]), None), J2000)
        new_arr = state_arr.add_to_state_vector([
            ("eta_srp", 1, "estimated", "static", sc, eta_val)
        ])
        assert len(new_arr._state) == 3
        assert new_arr._state[2][0] == "eta_srp"

        # duplicate parameter for the same body raises ValueError
        with pytest.raises(ValueError):
            state_arr.add_to_state_vector([
                ("position", 3, "estimated", "dynamic", sc, pos_0)
            ])

    def test_statedefinition_input_and_new_owner_routing(self, epoch, origin, sc, sc_area, sc_ref_coeff):
        """
            Verifies add_to_state_vector accepts a StateDefinition directly,
            rejects a non-list/StateDefinition argument, and routes
            "position"/"velocity" components (for a new owner) through their
            dedicated builder methods rather than the generic param() path.
        """
        pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
        vel_0 = scb.ArrayWFrame(np.array([0.0, 7.546049, 0.0]), km / sec, J2000)
        state_arr = scb.StateArray(epoch, origin, state=scb.StateDefinition()
                                                    .position(sc, pos_0)
                                                    .velocity(sc, vel_0),)

        # StateDefinition input, routed through .components
        eta_val = scb.ArrayWFrame(scb.ArrayWUnits(np.array([1.2]), None), J2000)
        new_def = scb.StateDefinition().param("eta_srp", sc, eta_val, dynamics="static")
        new_arr = state_arr.add_to_state_vector(new_def)
        assert new_arr._state[2][0] == "eta_srp"

        # invalid argument type raises TypeError
        with pytest.raises(TypeError, match="StateDefinition or list"):
            state_arr.add_to_state_vector("not valid")

        # position/velocity for a second owner go through their own builder methods
        sc2 = scb.Spacecraft("ORCCA2", -5678, area=sc_area, ref_coeff=sc_ref_coeff)
        pos_1 = scb.ArrayWFrame(np.array([1000.0, 0.0, 0.0]), km, J2000)
        vel_1 = scb.ArrayWFrame(np.array([0.0, 1.0, 0.0]), km / sec, J2000)
        new_arr2 = state_arr.add_to_state_vector([
            ("position", 3, "estimated", "dynamic", sc2, pos_1),
            ("velocity", 3, "estimated", "dynamic", sc2, vel_1),
        ])
        assert len(new_arr2._state) == 4
        assert [c[0] for c in new_arr2._state[2:]] == ["position", "velocity"]
        assert [c[4] for c in new_arr2._state[2:]] == [sc2, sc2]


class TestSerialization:
    def test_to_dict(self, epoch, origin, sc):
        """
            Verifies that to_dict serializes the epoch, origin, and each state
            component (name, dim, estimation, dynamics, owner, values, units,
            frame) into a plain, JSON-friendly dict.
        """
        pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
        vel_0 = scb.ArrayWFrame(np.array([0.0, 7.546049, 0.0]), km / sec, J2000)

        state_arr = scb.StateArray(epoch, origin, state=scb.StateDefinition()
                                                    .position(sc, pos_0)
                                                    .velocity(sc, vel_0),)

        assert state_arr.to_dict() == {
            "epoch": epoch.to_dict(),
            "origin": owner_to_dict(origin),
            "components": [
                {
                    "name": "position",
                    "dim": 3,
                    "estimation": "estimated",
                    "dynamics": "dynamic",
                    "owner": owner_to_dict(sc),
                    "values": [7000.0, 0.0, 0.0],
                    "units": km.to_dict(),
                    "frame": "J2000",
                },
                {
                    "name": "velocity",
                    "dim": 3,
                    "estimation": "estimated",
                    "dynamics": "dynamic",
                    "owner": owner_to_dict(sc),
                    "values": [0.0, 7.546049, 0.0],
                    "units": (km / sec).to_dict(),
                    "frame": "J2000",
                },
            ],
        }

    def test_from_dict_round_trip(self, epoch, origin, sc):
        """
            Verifies that from_dict reconstructs a StateArray whose to_dict()
            output matches the original exactly, and that owner_overrides
            substitutes the exact pre-built owner instead of reconstructing
            a duplicate from its spice_id.
        """
        pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
        vel_0 = scb.ArrayWFrame(np.array([0.0, 7.546049, 0.0]), km / sec, J2000)

        state_arr = scb.StateArray(epoch, origin, state=scb.StateDefinition()
                                                    .position(sc, pos_0)
                                                    .velocity(sc, vel_0),)

        data = state_arr.to_dict()
        rebuilt = scb.StateArray.from_dict(data, owner_overrides={sc.spice_id: sc})

        # owner_overrides substituted the exact same object rather than rebuilding one
        assert rebuilt.state[0][4] is sc
        assert rebuilt.state[1][4] is sc

        # round trip is lossless: re-serializing gives back the same dict
        assert rebuilt.to_dict() == data


#-----------------------------------#
# region  Malformed State Entries  #
#-----------------------------------#
# StateDefinition's public builder (.position/.velocity/.param) already
# validates component tuples, so the defensive checks inside
# _extract_values_array are exercised here by handing StateArray a raw
# object with a `.components` list, bypassing that builder entirely.
_FAKE_OWNER = SimpleNamespace(spice_id=42)
_POS_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
_NON_AWF_VALUE = scb.ArrayWUnits(np.array([7000.0, 0.0, 0.0]), km)
_STATIC_SIZE_MISMATCH_VAL = scb.ArrayWFrame(np.array([1.0, 2.0]), km, J2000)
_DYNAMIC_SCALAR_VAL = scb.ArrayWFrame(scb.ArrayWUnits(5.0, km), J2000)
_POS_HIST_N2 = scb.ArrayWFrame(
    np.array([[7000.0, 0.0, 0.0], [7001.0, 0.1, 0.05]]), km, J2000
)
_VEL_HIST_N3 = scb.ArrayWFrame(np.array([[0.0, 7.5, 0.0]] * 3), km / sec, J2000)

# Numeric ET construction avoids requiring the SPICE leapseconds kernel to
# already be loaded at module-collection time (unlike the calendar-string
# form the `epoch` fixture uses, which runs after the autouse kernel fixture).
_SINGLE_EPOCH = scb.EpochArray(0.0, 'TDB')
_MULTI_EPOCH = scb.EpochArray([796900869.0, 796900869.0 + 3600.0], 'TDB')
# endregion Malformed State Entries #
#-----------------------------------#


class TestExtractValuesArray:
    @pytest.mark.parametrize(
        "epoch_arg, components, exc_type, match",
        [
            pytest.param(
                # 5-tuple (owner missing): _extract_values0 runs first and only
                # checks entry[-1], which is a valid ArrayWFrame here, so this
                # slips past it and reaches _extract_values_array's own length check.
                _SINGLE_EPOCH,
                [("position", 3, "estimated", "dynamic", _POS_0)],
                ValueError, "Invalid state entry format",
                id="invalid_entry_format",
            ),
            pytest.param(
                # 7-tuple: entry[-1] is a valid ArrayWFrame (passes _extract_values0's
                # check), but entry[5] — what _extract_values_array actually reads
                # as `awf` — is not, reaching its own isinstance check.
                _SINGLE_EPOCH,
                [("position", 3, "estimated", "dynamic", _FAKE_OWNER, _NON_AWF_VALUE, _POS_0)],
                TypeError, "must be ArrayWFrame",
                id="non_arraywframe_value",
            ),
            pytest.param(
                _SINGLE_EPOCH,
                [
                    ("position", 3, "estimated", "dynamic", _FAKE_OWNER, _POS_0),
                    ("position", 3, "estimated", "dynamic", _FAKE_OWNER, _POS_0),
                ],
                ValueError, "Duplicate state component",
                id="duplicate_key",
            ),
            pytest.param(
                _SINGLE_EPOCH,
                [("eta_srp", 3, "estimated", "static", _FAKE_OWNER, _STATIC_SIZE_MISMATCH_VAL)],
                ValueError, "must have size",
                id="static_size_mismatch",
            ),
            pytest.param(
                _SINGLE_EPOCH,
                [("position", 3, "estimated", "dynamic", _FAKE_OWNER, _DYNAMIC_SCALAR_VAL)],
                ValueError, "single scalar value",
                id="dynamic_scalar_size_mismatch",
            ),
            pytest.param(
                _MULTI_EPOCH,
                [
                    ("position", 3, "estimated", "dynamic", _FAKE_OWNER, _POS_HIST_N2),
                    ("velocity", 3, "estimated", "dynamic", _FAKE_OWNER, _VEL_HIST_N3),
                ],
                ValueError, "inconsistent time lengths",
                id="inconsistent_dynamic_lengths",
            ),
            pytest.param(
                _SINGLE_EPOCH,
                [("position", 3, "estimated", "dynamic", _FAKE_OWNER, _POS_HIST_N2)],
                ValueError, "Mismatch between epoch size",
                id="epoch_size_mismatch",
            ),
        ],
    )
    def test_raises_for_malformed_entries(self, epoch_arg, components, exc_type, match, origin):
        """
            Verifies each defensive check in _extract_values_array raises the
            expected exception for a malformed state entry.
        """
        with pytest.raises(exc_type, match=match):
            scb.StateArray(epoch_arg, origin, state=SimpleNamespace(components=components))

    def test_dynamic_time_history(self, origin):
        """
            Verifies that a dynamic parameter given as an (N, dim) time history
            whose N matches epoch.size is extracted without error.
        """
        components = [("position", 3, "estimated", "dynamic", _FAKE_OWNER, _POS_HIST_N2)]
        state_arr = scb.StateArray(_MULTI_EPOCH, origin, state=SimpleNamespace(components=components))

        vals, _ = state_arr.values_array[("position", _FAKE_OWNER.spice_id)]
        assert np.allclose(np.asarray(vals), _POS_HIST_N2.quantity.values)

    def test_scalar_dynamic_time_history(self, origin):
        """
            Verifies that a dim=1 dynamic parameter given as a bare (N,) time
            history (not (N, dim)) is extracted without error, exercising the
            1-D branch of the time-length computation (N = vals_arr.size)
            rather than the (N, dim) branch (N = vals_arr.shape[0]).
        """
        scalar_hist = scb.ArrayWFrame(scb.ArrayWUnits(np.array([1.1, 1.2]), None), J2000)
        components = [("gm", 1, "estimated", "dynamic", _FAKE_OWNER, scalar_hist)]
        state_arr = scb.StateArray(_MULTI_EPOCH, origin, state=SimpleNamespace(components=components))

        vals, _ = state_arr.values_array[("gm", _FAKE_OWNER.spice_id)]
        assert np.allclose(np.asarray(vals), [1.1, 1.2])

    def test_extract_values0_raises_for_non_arraywframe(self, origin):
        """
            Verifies that a non-ArrayWFrame component value is caught by
            _extract_values0 (which runs before _extract_values_array in
            __init__, and reports a differently-worded TypeError).
        """
        components = [("position", 3, "estimated", "dynamic", _FAKE_OWNER, _NON_AWF_VALUE)]
        with pytest.raises(TypeError, match="should be of type ArrayWFrame"):
            scb.StateArray(_SINGLE_EPOCH, origin, state=SimpleNamespace(components=components))


@pytest.mark.parametrize(
    "raw_state, exc_type, match",
    [
        pytest.param(
            [("position", 3, "estimated")],
            ValueError, "Invalid state entry format",
            id="invalid_entry_format",
        ),
        pytest.param(
            [
                ("position", 3, "estimated", "dynamic", _FAKE_OWNER, _POS_0),
                ("position", 3, "estimated", "dynamic", _FAKE_OWNER, _POS_0),
            ],
            ValueError, "Duplicate state component",
            id="duplicate_key",
        ),
    ],
)
def test_make_index_rep_raises(epoch, origin, sc, raw_state, exc_type, match):
    """
        Verifies _make_index_rep's own defensive checks. Both are
        unreachable through the normal constructor because
        _extract_values_array runs first and already rejects anything
        that would trip these (its length threshold is stricter, and
        its duplicate-key check is at least as broad) — so the method
        is called directly on an instance with a mutated `_state`.
    """
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0, 7.546049, 0.0]), km / sec, J2000)
    state_arr = scb.StateArray(epoch, origin, state=scb.StateDefinition()
                                                .position(sc, pos_0)
                                                .velocity(sc, vel_0),)

    state_arr._state = raw_state
    with pytest.raises(exc_type, match=match):
        state_arr._make_index_rep()


class TestMakeDirectRepresentation:
    @pytest.mark.parametrize(
        "raw_state, y, exc_type, match",
        [
            pytest.param(
                [("position", 3, "estimated")],
                np.zeros(3),
                ValueError, "Invalid state entry format",
                id="invalid_entry_format",
            ),
            pytest.param(
                [("position", 3, "estimated", "dynamic", SimpleNamespace(), None)],
                np.zeros(3),
                ValueError, "no 'spice_id'",
                id="owner_missing_spice_id",
            ),
            pytest.param(
                [
                    ("position", 3, "estimated", "dynamic", _FAKE_OWNER, None),
                    ("velocity", 3, "estimated", "dynamic", _FAKE_OWNER, None),
                ],
                np.zeros(3),  # only enough room for the first entry
                ValueError, "Inconsistent StateArray layout",
                id="y_too_short",
            ),
        ],
    )
    def test_raises(self, epoch, origin, sc, raw_state, y, exc_type, match):
        """
            Verifies _make_direct_representation's layout-building checks
            (invalid entry format, owner missing spice_id, and y shorter
            than the layout requires). Only entry[0], entry[1], and entry[4]
            matter for layout building, so the value slots are left as None.
        """
        pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
        vel_0 = scb.ArrayWFrame(np.array([0.0, 7.546049, 0.0]), km / sec, J2000)
        state_arr = scb.StateArray(epoch, origin, state=scb.StateDefinition()
                                                    .position(sc, pos_0)
                                                    .velocity(sc, vel_0),)

        state_arr._state = raw_state
        with pytest.raises(exc_type, match=match):
            state_arr._make_direct_representation(y=y)

    def test_default_y(self, epoch, origin, sc):
        """
            Verifies that omitting y falls back to self.values0.values, i.e.
            the direct representation of the state at its initial values.
        """
        pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
        vel_0 = scb.ArrayWFrame(np.array([0.0, 7.546049, 0.0]), km / sec, J2000)
        state_arr = scb.StateArray(epoch, origin, state=scb.StateDefinition()
                                                    .position(sc, pos_0)
                                                    .velocity(sc, vel_0),)

        direct_rep = state_arr._make_direct_representation()
        assert np.allclose(direct_rep[("position", sc.spice_id)], pos_0.quantity.values)
        assert np.allclose(direct_rep[("velocity", sc.spice_id)], vel_0.quantity.values)


def test_reinitialize(epoch, origin, sc):
    """
        Verifies that reinitialize updates epoch/origin/state in place,
        accepts a raw component list (converting it to a StateDefinition),
        rejects an invalid state type, and invalidates the cached direct-
        representation layout when the state changes.
    """
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0, 7.546049, 0.0]), km / sec, J2000)
    state_arr = scb.StateArray(epoch, origin, state=scb.StateDefinition()
                                                .position(sc, pos_0)
                                                .velocity(sc, vel_0),)

    # populate the direct-rep layout cache so we can confirm it gets invalidated
    state_arr._make_direct_representation(y=np.zeros(state_arr.size))
    assert state_arr._direct_rep_layout is not None

    # epoch-only update
    new_epoch = scb.EpochArray('2000 JAN 02 12:00:00.000', 'TDB')
    state_arr.reinitialize(epoch=new_epoch)
    assert state_arr.epoch is new_epoch
    assert state_arr.origin is origin

    # origin-only update
    new_origin = scb.CelestialBody.from_constants('Sun')
    state_arr.reinitialize(origin=new_origin)
    assert state_arr.origin is new_origin

    # state update via a raw component list -> converted to StateDefinition,
    # dependent attributes recomputed, and the layout cache invalidated
    eta_val = scb.ArrayWFrame(scb.ArrayWUnits(np.array([1.2]), None), J2000)
    state_arr.reinitialize(state=[("eta_srp", 1, "estimated", "static", sc, eta_val)])
    assert state_arr.size == 1
    assert state_arr._state[0][0] == "eta_srp"
    assert state_arr._direct_rep_layout is None

    # invalid state type raises TypeError
    with pytest.raises(TypeError, match="StateDefinition instance or a list"):
        state_arr.reinitialize(state="not a valid state")


def test_find_indices(epoch, origin, sc):
    """
        Verifies find_indices returns the (start, end) slice for a known
        parameter/body pair, and None for one that isn't present.
    """
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0, 7.546049, 0.0]), km / sec, J2000)
    state_arr = scb.StateArray(epoch, origin, state=scb.StateDefinition()
                                                .position(sc, pos_0)
                                                .velocity(sc, vel_0),)

    assert state_arr.find_indices("position", sc) == (0, 3)
    assert state_arr.find_indices("velocity", sc) == (3, 6)
    assert state_arr.find_indices("eta_srp", sc) is None
    assert state_arr.find_indices("position", origin) is None


class TestAddStateAtEpoch:
    def test_append(self, multi_epoch_state_arr, sc):
        """
            Verifies that adding a state at a later epoch appends it in sorted
            order: dynamic (already time-history) parameters grow by one row,
            and a static parameter given in new_values is overwritten.

            For an already-time-history parameter, add_state_at_epoch compares
            `param_data_values.shape[1:]` against the existing history's
            trailing shape, so the new value must carry a leading axis of 1
            (e.g. (1, 3), not a bare (3,)).
        """
        new_pos = scb.ArrayWFrame(np.array([[7002.0, 0.2, 0.1]]), km, J2000)
        new_vel = scb.ArrayWFrame(np.array([[0.0, 7.6, 0.0]]), km / sec, J2000)
        new_eta = scb.ArrayWFrame(scb.ArrayWUnits(np.array([1.3]), None), J2000)

        multi_epoch_state_arr.add_state_at_epoch(700007200.0, {
            ("position", sc.spice_id): new_pos,
            ("velocity", sc.spice_id): new_vel,
            ("eta_srp", sc.spice_id): new_eta,
        })

        assert multi_epoch_state_arr.epoch.size == 3
        pos_vals, _ = multi_epoch_state_arr.values_array[("position", sc.spice_id)]
        eta_vals, _ = multi_epoch_state_arr.values_array[("eta_srp", sc.spice_id)]
        assert np.asarray(pos_vals).shape == (3, 3)
        assert np.allclose(pos_vals[-1], [7002.0, 0.2, 0.1])
        assert np.isclose(float(np.asarray(eta_vals)), 1.3)

    def test_insert_middle(self, multi_epoch_state_arr, sc):
        """
            Verifies that a new epoch earlier than the last existing one is
            inserted at the correct sorted position, not appended.
        """
        new_pos = scb.ArrayWFrame(np.array([[7000.5, 0.05, 0.025]]), km, J2000)
        new_vel = scb.ArrayWFrame(np.array([[0.0, 7.52, 0.0]]), km / sec, J2000)

        # between the fixture's two epochs: 700000000.0 and 700003600.0
        multi_epoch_state_arr.add_state_at_epoch(700001800.0, {
            ("position", sc.spice_id): new_pos,
            ("velocity", sc.spice_id): new_vel,
        })

        assert multi_epoch_state_arr.epoch.size == 3
        assert list(multi_epoch_state_arr.epoch.times.values) == [
            700000000.0, 700001800.0, 700003600.0,
        ]
        pos_vals, _ = multi_epoch_state_arr.values_array[("position", sc.spice_id)]
        assert np.allclose(pos_vals[1], [7000.5, 0.05, 0.025])

    @pytest.mark.parametrize(
        "build_call, exc_type, match",
        [
            pytest.param(
                # numeric UTC input isn't supported by SPICE's unitim conversion,
                # so a UTC-system epoch must be built from a calendar string instead.
                lambda s: (scb.EpochArray('2000 JAN 01 12:00:00.000', sys='UTC'), {}),
                ValueError, "Epoch time frame mismatch", id="epoch_system_mismatch",
            ),
            pytest.param(
                lambda s: (700007200.0, {("position", s.spice_id): 42.0}),
                TypeError, "Expected ArrayWFrame or ArrayWUnits", id="invalid_value_type",
            ),
            pytest.param(
                lambda s: (700007200.0, {
                    ("position", s.spice_id): scb.ArrayWFrame(
                        np.array([1.0, 2.0, 3.0]), km, scb.Frame('ECLIPJ2000')
                    )
                }),
                ValueError, "Frame mismatch", id="frame_mismatch",
            ),
            pytest.param(
                lambda s: (700007200.0, {
                    ("position", s.spice_id): scb.ArrayWFrame(np.array([1.0, 2.0]), km, J2000)
                }),
                ValueError, "do not match existing shape", id="shape_mismatch",
            ),
            pytest.param(
                lambda s: (700007200.0, {
                    ("position", s.spice_id): scb.ArrayWFrame(np.array([[7002.0, 0.2, 0.1]]), km, J2000)
                }),
                ValueError, "inconsistent time lengths", id="partial_update_leaves_param_short",
            ),
        ],
    )
    def test_raises(self, multi_epoch_state_arr, sc, build_call, exc_type, match):
        """
            Verifies add_state_at_epoch's defensive checks: epoch system
            mismatch, wrong value type, frame mismatch, shape mismatch, and
            leaving a dynamic parameter's time length inconsistent with the
            grown epoch count.
        """
        epoch_arg, new_values = build_call(sc)
        with pytest.raises(exc_type, match=match):
            multi_epoch_state_arr.add_state_at_epoch(epoch_arg, new_values)

    def test_no_matching_entry_raises(self, multi_epoch_state_arr):
        """
            Verifies add_state_at_epoch raises when values_array holds an
            entry with no matching state component (corrupted on purpose
            here, since a real StateArray's values_array is always built
            from the same state it's paired with).
        """
        multi_epoch_state_arr._values_array[("ghost_param", 999999)] = (np.array([0.0]), km)
        with pytest.raises(ValueError, match="No matching state entry found"):
            multi_epoch_state_arr.add_state_at_epoch(700007200.0, {})

    def test_legacy_arraywunits_entry(self, multi_epoch_state_arr, sc):
        """
            Verifies the legacy-storage branch: a values_array entry stored as
            a bare ArrayWUnits (not ArrayWFrame) is still unwrapped correctly,
            and gets re-wrapped as ArrayWFrame(..., frame=None) in the result.
        """
        eta_key = ("eta_srp", sc.spice_id)
        old_vals, units = multi_epoch_state_arr._values_array[eta_key]
        multi_epoch_state_arr._values_array[eta_key] = (scb.ArrayWUnits(old_vals, units), units)

        new_pos = scb.ArrayWFrame(np.array([[7002.0, 0.2, 0.1]]), km, J2000)
        new_vel = scb.ArrayWFrame(np.array([[0.0, 7.6, 0.0]]), km / sec, J2000)
        multi_epoch_state_arr.add_state_at_epoch(700007200.0, {
            ("position", sc.spice_id): new_pos,
            ("velocity", sc.spice_id): new_vel,
        })

        eta_vals, _ = multi_epoch_state_arr.values_array[eta_key]
        assert np.isclose(float(np.asarray(eta_vals)), 1.2)

    def test_new_value_as_arraywunits(self, multi_epoch_state_arr, sc):
        """
            Verifies that a new_values entry may be given as a bare ArrayWUnits
            (not wrapped in ArrayWFrame), which is accepted and applied.
        """
        new_pos = scb.ArrayWFrame(np.array([[7002.0, 0.2, 0.1]]), km, J2000)
        new_vel = scb.ArrayWFrame(np.array([[0.0, 7.6, 0.0]]), km / sec, J2000)
        new_eta = scb.ArrayWUnits(np.array([1.5]), None)
        multi_epoch_state_arr.add_state_at_epoch(700007200.0, {
            ("position", sc.spice_id): new_pos,
            ("velocity", sc.spice_id): new_vel,
            ("eta_srp", sc.spice_id): new_eta,
        })

        eta_vals, _ = multi_epoch_state_arr.values_array[("eta_srp", sc.spice_id)]
        assert np.isclose(float(np.asarray(eta_vals)), 1.5)

    def test_arrayframe_entry(self, multi_epoch_state_arr, sc):
        """
            Verifies the primary (ArrayWFrame) branch of the unwrap logic.
            reinitialize()'s call to _extract_values_array() always resets
            values_array entries back to raw arrays/scalars, so this branch
            (like the ArrayWUnits one) is never reached in normal use — only
            by an entry stored pre-wrapped as ArrayWFrame, as exercised here.
        """
        eta_key = ("eta_srp", sc.spice_id)
        old_vals, units = multi_epoch_state_arr._values_array[eta_key]
        multi_epoch_state_arr._values_array[eta_key] = (
            scb.ArrayWFrame(scb.ArrayWUnits(old_vals, units), J2000), units
        )

        new_pos = scb.ArrayWFrame(np.array([[7002.0, 0.2, 0.1]]), km, J2000)
        new_vel = scb.ArrayWFrame(np.array([[0.0, 7.6, 0.0]]), km / sec, J2000)
        new_eta = scb.ArrayWFrame(scb.ArrayWUnits(np.array([1.6]), None), J2000)
        multi_epoch_state_arr.add_state_at_epoch(700007200.0, {
            ("position", sc.spice_id): new_pos,
            ("velocity", sc.spice_id): new_vel,
            eta_key: new_eta,
        })

        eta_vals, _ = multi_epoch_state_arr.values_array[eta_key]
        assert np.isclose(float(np.asarray(eta_vals)), 1.6)


def _remove_state_epoch_system_mismatch(sa, s):
    return scb.EpochArray('2000 JAN 01 12:00:00.000', sys='UTC')


def _remove_state_epoch_not_found(sa, s):
    return 999999999.0


def _remove_state_no_matching_entry(sa, s):
    sa._values_array[("ghost_param", 999999)] = (np.array([0.0]), km)
    return 700000000.0


class TestRemoveStateAtEpoch:
    def test_remove(self, multi_epoch_state_arr, sc):
        """
            Verifies that removing an epoch deletes the matching time-history
            row from dynamic parameters, decrements epoch.size, and leaves a
            static parameter untouched.
        """
        multi_epoch_state_arr.remove_state_at_epoch(700000000.0)

        assert multi_epoch_state_arr.epoch.size == 1
        # a single-epoch EpochArray collapses .times.values to a 0-d scalar, so
        # use .et (the raw internal array) for a safely indexable value here.
        assert np.isclose(float(np.atleast_1d(multi_epoch_state_arr.epoch.et)[0]), 700003600.0)
        pos_vals, _ = multi_epoch_state_arr.values_array[("position", sc.spice_id)]
        assert np.allclose(np.asarray(pos_vals), [7001.0, 0.1, 0.05])
        eta_vals, _ = multi_epoch_state_arr.values_array[("eta_srp", sc.spice_id)]
        assert np.isclose(float(np.asarray(eta_vals)), 1.2)

    @pytest.mark.parametrize(
        "mutator, exc_type, match",
        [
            pytest.param(_remove_state_epoch_system_mismatch, ValueError, "Epoch time frame mismatch", id="epoch_system_mismatch"),
            pytest.param(_remove_state_epoch_not_found, ValueError, "not found in the state vector", id="epoch_not_found"),
            pytest.param(_remove_state_no_matching_entry, ValueError, "No matching state entry found", id="no_matching_entry"),
        ],
    )
    def test_raises(self, multi_epoch_state_arr, sc, mutator, exc_type, match):
        """
            Verifies remove_state_at_epoch's defensive checks: epoch system
            mismatch, epoch not present, and a values_array entry with no
            matching state component.
        """
        epoch_arg = mutator(multi_epoch_state_arr, sc)
        with pytest.raises(exc_type, match=match):
            multi_epoch_state_arr.remove_state_at_epoch(epoch_arg)

    def test_arrayframe_entry(self, multi_epoch_state_arr, sc):
        """
            Verifies the primary (ArrayWFrame) branch of the unwrap logic.
            reinitialize()'s call to _extract_values_array() always resets
            values_array entries back to raw arrays/scalars, so this branch
            (like the ArrayWUnits one) is never reached in normal use — only
            by an entry stored pre-wrapped as ArrayWFrame, as exercised here.
        """
        eta_key = ("eta_srp", sc.spice_id)
        old_vals, units = multi_epoch_state_arr._values_array[eta_key]
        multi_epoch_state_arr._values_array[eta_key] = (
            scb.ArrayWFrame(scb.ArrayWUnits(old_vals, units), J2000), units
        )

        multi_epoch_state_arr.remove_state_at_epoch(700000000.0)

        eta_vals, _ = multi_epoch_state_arr.values_array[eta_key]
        assert np.isclose(float(np.asarray(eta_vals)), 1.2)

    def test_legacy_arraywunits_entry(self, multi_epoch_state_arr, sc):
        """
            Verifies the legacy-storage branch: a values_array entry stored as
            a bare ArrayWUnits (not ArrayWFrame) is still unwrapped correctly
            (with its frame recovered from the state's template value).
        """
        eta_key = ("eta_srp", sc.spice_id)
        old_vals, units = multi_epoch_state_arr._values_array[eta_key]
        multi_epoch_state_arr._values_array[eta_key] = (scb.ArrayWUnits(old_vals, units), units)

        multi_epoch_state_arr.remove_state_at_epoch(700000000.0)

        eta_vals, _ = multi_epoch_state_arr.values_array[eta_key]
        assert np.isclose(float(np.asarray(eta_vals)), 1.2)


def _update_state_epoch_system_mismatch(sa, s):
    return scb.EpochArray('2000 JAN 01 12:00:00.000', sys='UTC'), {}


def _update_state_epoch_not_found(sa, s):
    return 999999999.0, {}


def _update_state_invalid_value_type(sa, s):
    return 700000000.0, {("position", s.spice_id): 42.0}


def _update_state_frame_mismatch(sa, s):
    other_frame = scb.Frame('ECLIPJ2000')
    return 700000000.0, {
        ("position", s.spice_id): scb.ArrayWFrame(np.array([1.0, 2.0, 3.0]), km, other_frame)
    }


def _update_state_shape_mismatch(sa, s):
    return 700000000.0, {
        ("position", s.spice_id): scb.ArrayWFrame(np.array([1.0, 2.0]), km, J2000)
    }


def _update_state_no_matching_entry(sa, s):
    sa._values_array[("ghost_param", 999999)] = (np.array([0.0]), km)
    return 700000000.0, {}


class TestUpdateStateAtEpoch:
    def test_update(self, multi_epoch_state_arr, sc):
        """
            Verifies that updating a dynamic parameter at a specific epoch
            overwrites only that time slice (accepting both the bare
            per-epoch shape and a (1, *per_epoch_shape) form), and that a
            static parameter is fully overwritten regardless of which epoch
            is named.
        """
        new_pos = scb.ArrayWFrame(np.array([7100.0, 1.0, 0.5]), km, J2000)
        multi_epoch_state_arr.update_state_at_epoch(700000000.0, {
            ("position", sc.spice_id): new_pos,
        })
        pos_vals, _ = multi_epoch_state_arr.values_array[("position", sc.spice_id)]
        assert np.allclose(pos_vals[0], [7100.0, 1.0, 0.5])
        assert np.allclose(pos_vals[1], [7001.0, 0.1, 0.05])  # untouched epoch

        # (1, *per_epoch_shape) form
        new_vel_leading_dim = scb.ArrayWFrame(np.array([[0.0, 7.99, 0.0]]), km / sec, J2000)
        multi_epoch_state_arr.update_state_at_epoch(700003600.0, {
            ("velocity", sc.spice_id): new_vel_leading_dim,
        })
        vel_vals, _ = multi_epoch_state_arr.values_array[("velocity", sc.spice_id)]
        assert np.allclose(vel_vals[1], [0.0, 7.99, 0.0])

        # static parameter: overwritten regardless of which epoch is named
        new_eta = scb.ArrayWFrame(scb.ArrayWUnits(np.array([2.0]), None), J2000)
        multi_epoch_state_arr.update_state_at_epoch(700003600.0, {
            ("eta_srp", sc.spice_id): new_eta,
        })
        eta_vals, _ = multi_epoch_state_arr.values_array[("eta_srp", sc.spice_id)]
        assert np.isclose(float(np.asarray(eta_vals)), 2.0)

    @pytest.mark.parametrize(
        "mutator, exc_type, match",
        [
            pytest.param(_update_state_epoch_system_mismatch, ValueError, "Epoch time frame mismatch", id="epoch_system_mismatch"),
            pytest.param(_update_state_epoch_not_found, ValueError, "not found in the state vector", id="epoch_not_found"),
            pytest.param(_update_state_invalid_value_type, TypeError, "Expected ArrayWFrame or ArrayWUnits", id="invalid_value_type"),
            pytest.param(_update_state_frame_mismatch, ValueError, "Frame mismatch", id="frame_mismatch"),
            pytest.param(_update_state_shape_mismatch, ValueError, "incompatible shape", id="shape_mismatch"),
            pytest.param(_update_state_no_matching_entry, ValueError, "No matching state entry found", id="no_matching_entry"),
        ],
    )
    def test_raises(self, multi_epoch_state_arr, sc, mutator, exc_type, match):
        """
            Verifies update_state_at_epoch's defensive checks: epoch system
            mismatch, epoch not present, wrong value type, frame mismatch,
            incompatible per-epoch shape, and a values_array entry with no
            matching state component.
        """
        epoch_arg, updated_values = mutator(multi_epoch_state_arr, sc)
        with pytest.raises(exc_type, match=match):
            multi_epoch_state_arr.update_state_at_epoch(epoch_arg, updated_values)

    def test_arrayframe_entry(self, multi_epoch_state_arr, sc):
        """
            Verifies the primary (ArrayWFrame) branch of the unwrap logic.
            reinitialize()'s call to _extract_values_array() always resets
            values_array entries back to raw arrays/scalars, so this branch
            (like the ArrayWUnits one) is never reached in normal use — only
            by an entry stored pre-wrapped as ArrayWFrame, as exercised here.
        """
        eta_key = ("eta_srp", sc.spice_id)
        old_vals, units = multi_epoch_state_arr._values_array[eta_key]
        multi_epoch_state_arr._values_array[eta_key] = (
            scb.ArrayWFrame(scb.ArrayWUnits(old_vals, units), J2000), units
        )

        new_eta = scb.ArrayWFrame(scb.ArrayWUnits(np.array([1.9]), None), J2000)
        multi_epoch_state_arr.update_state_at_epoch(700000000.0, {
            eta_key: new_eta,
        })

        eta_vals, _ = multi_epoch_state_arr.values_array[eta_key]
        assert np.isclose(float(np.asarray(eta_vals)), 1.9)

    def test_legacy_arraywunits_entry(self, multi_epoch_state_arr, sc):
        """
            Verifies the legacy-storage branch: a values_array entry stored as
            a bare ArrayWUnits (not ArrayWFrame) is still unwrapped correctly
            when left untouched (no update given for it at this epoch).
        """
        eta_key = ("eta_srp", sc.spice_id)
        old_vals, units = multi_epoch_state_arr._values_array[eta_key]
        multi_epoch_state_arr._values_array[eta_key] = (scb.ArrayWUnits(old_vals, units), units)

        new_pos = scb.ArrayWFrame(np.array([7100.0, 1.0, 0.5]), km, J2000)
        multi_epoch_state_arr.update_state_at_epoch(700000000.0, {
            ("position", sc.spice_id): new_pos,
        })

        eta_vals, _ = multi_epoch_state_arr.values_array[eta_key]
        assert np.isclose(float(np.asarray(eta_vals)), 1.2)

    def test_new_value_as_arraywunits(self, multi_epoch_state_arr, sc):
        """
            Verifies that an updated_values entry may be given as a bare
            ArrayWUnits (not wrapped in ArrayWFrame), which is accepted and
            applied to the static eta_srp parameter.
        """
        new_eta = scb.ArrayWUnits(np.array([2.5]), None)
        multi_epoch_state_arr.update_state_at_epoch(700000000.0, {
            ("eta_srp", sc.spice_id): new_eta,
        })

        eta_vals, _ = multi_epoch_state_arr.values_array[("eta_srp", sc.spice_id)]
        assert np.isclose(float(np.asarray(eta_vals)), 2.5)


class TestGetStateAtEpoch:
    def test_get(self, multi_epoch_state_arr, sc):
        """
            Verifies that get_state_at_epoch returns a new, single-epoch
            StateArray holding the requested time slice of each dynamic
            parameter, with a static parameter carried over unchanged.
        """
        first = multi_epoch_state_arr.get_state_at_epoch(700000000.0)
        assert first is not multi_epoch_state_arr
        assert first.epoch.size == 1
        pos_vals, _ = first.values_array[("position", sc.spice_id)]
        assert np.allclose(np.asarray(pos_vals), [7000.0, 0.0, 0.0])

        second = multi_epoch_state_arr.get_state_at_epoch(700003600.0)
        pos_vals2, _ = second.values_array[("position", sc.spice_id)]
        assert np.allclose(np.asarray(pos_vals2), [7001.0, 0.1, 0.05])
        eta_vals2, _ = second.values_array[("eta_srp", sc.spice_id)]
        assert np.isclose(float(np.asarray(eta_vals2)), 1.2)

    @pytest.mark.parametrize(
        "build_epoch, exc_type, match",
        [
            pytest.param(
                # deferred (not a bare EpochArray literal) only so this isn't built
                # before the module-scoped kernel-loading fixture has run.
                lambda: scb.EpochArray('2000 JAN 01 12:00:00.000', sys='UTC'),
                ValueError, "Epoch time frame mismatch", id="epoch_system_mismatch",
            ),
            pytest.param(
                lambda: 999999999.0,
                ValueError, "not found in the state vector", id="epoch_not_found",
            ),
        ],
    )
    def test_raises(self, multi_epoch_state_arr, build_epoch, exc_type, match):
        """
            Verifies get_state_at_epoch's defensive checks: epoch system
            mismatch and epoch not present.
        """
        with pytest.raises(exc_type, match=match):
            multi_epoch_state_arr.get_state_at_epoch(build_epoch())

    def test_values_array_missing_entry_raises(self, multi_epoch_state_arr, sc):
        """
            Verifies get_state_at_epoch raises when a state component has no
            matching values_array entry (corrupted on purpose here, since a
            real StateArray's values_array is always built from the same
            state it's paired with).
        """
        del multi_epoch_state_arr._values_array[("velocity", sc.spice_id)]
        with pytest.raises(ValueError, match="Values array missing entry"):
            multi_epoch_state_arr.get_state_at_epoch(700000000.0)

    def test_arrayframe_entry(self, multi_epoch_state_arr, sc):
        """
            Verifies the primary (ArrayWFrame) branch of the unwrap logic.
            reinitialize()'s call to _extract_values_array() always resets
            values_array entries back to raw arrays/scalars, so this branch
            (like the ArrayWUnits one) is never reached in normal use — only
            by an entry stored pre-wrapped as ArrayWFrame, as exercised here.
        """
        eta_key = ("eta_srp", sc.spice_id)
        old_vals, units = multi_epoch_state_arr._values_array[eta_key]
        multi_epoch_state_arr._values_array[eta_key] = (
            scb.ArrayWFrame(scb.ArrayWUnits(old_vals, units), J2000), units
        )

        result = multi_epoch_state_arr.get_state_at_epoch(700000000.0)

        eta_vals, _ = result.values_array[eta_key]
        assert np.isclose(float(np.asarray(eta_vals)), 1.2)

    def test_legacy_arraywunits_entry(self, multi_epoch_state_arr, sc):
        """
            Verifies the legacy-storage branch: a values_array entry stored as
            a bare ArrayWUnits (not ArrayWFrame) is still unwrapped correctly
            into the returned single-epoch StateArray.
        """
        eta_key = ("eta_srp", sc.spice_id)
        old_vals, units = multi_epoch_state_arr._values_array[eta_key]
        multi_epoch_state_arr._values_array[eta_key] = (scb.ArrayWUnits(old_vals, units), units)

        result = multi_epoch_state_arr.get_state_at_epoch(700000000.0)

        eta_vals, _ = result.values_array[eta_key]
        assert np.isclose(float(np.asarray(eta_vals)), 1.2)


def test_rebuild_state(multi_epoch_state_arr, sc):
    """
        Verifies _rebuild_state rebuilds self.state in the same order with
        each component's value swapped in from updated_values_array, and
        raises KeyError when an entry is missing for a state component.
    """
    new_pos = scb.ArrayWFrame(np.array([[1.0, 2.0, 3.0]] * 2), km, J2000)
    updated_values_array = {
        key: (new_pos if key[0] == "position" else vals, units)
        for key, (vals, units) in multi_epoch_state_arr._values_array.items()
    }

    new_state = multi_epoch_state_arr._rebuild_state(updated_values_array)
    assert [c[0] for c in new_state] == [c[0] for c in multi_epoch_state_arr._state]
    assert new_state[0][5] is new_pos

    incomplete = {k: v for k, v in updated_values_array.items() if k[0] != "velocity"}
    with pytest.raises(KeyError):
        multi_epoch_state_arr._rebuild_state(incomplete)


# NOTE: the rest of this class' methods are tested in integration tests
