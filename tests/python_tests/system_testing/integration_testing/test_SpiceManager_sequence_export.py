# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
#=====================================================================#
#  Integration Tests for SpiceManager's MissionSequence-based export  #
#=====================================================================#
import src.scarabaeus as scb
import numpy as np
import pytest
import os
import json
import warnings

km, sec, rad = scb.Units.get_units(["km", "sec", "rad"])
J2000 = scb.Frame("J2000")

# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture
def sc():
    # A name distinct from other fixtures in the suite (e.g. test_StateArray.py's
    # "ORCCA"/-1234): SPICE's name<->ID registration is process-global and
    # persists across test files, so reusing a name with a different ID here
    # would collide with it.
    return scb.Spacecraft(
        "SEQ_EXPORT_TESTSAT", -999011, area=scb.ArrayWUnits(1e-06, km**2), ref_coeff=1.5
    )


@pytest.fixture
def origin():
    return scb.CelestialBody.from_constants("EARTH")


@pytest.fixture
def state_arr(sc, origin):
    pos0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel0 = scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000)
    state_def = scb.StateDefinition().position(sc, pos0).velocity(sc, vel0)
    epoch0 = scb.EpochArray("2024 JAN 01 00:00:00", "TDB")
    return scb.StateArray(epoch0, origin, state=state_def)


@pytest.fixture
def one_leg_sequence(state_arr):
    """A MissionSequence with a single leg, built without a Propagator --
    addLeg only needs one when period/dt/epochs aren't given explicitly."""
    et0 = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")
    epochs_leg = scb.EpochArray(np.array([et0, et0 + 100.0]), sys="TDB")
    seq = scb.MissionSequence("TEST SEQUENCE")
    seq.addLeg("leg0", state_vector=state_arr, epochs=epochs_leg)
    return seq, et0


@pytest.fixture
def leg_data():
    return {
        "by_seq_idx": {
            0: {"body_ID": -999011, "origin_ID": 399, "reference_frame": "J2000"}
        }
    }


@pytest.fixture
def two_leg_sequence(sc, origin):
    """A MissionSequence with two legs. addLeg auto-inserts a boundary Node
    before the second leg, giving a ["Leg", "Node", "Leg"] type sequence --
    used to verify non-Leg segments are skipped, not mistaken for legs."""
    et0 = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")

    pos0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel0 = scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000)
    state0 = scb.StateArray(
        scb.EpochArray(et0, "TDB"), origin,
        state=scb.StateDefinition().position(sc, pos0).velocity(sc, vel0),
    )

    pos1 = scb.ArrayWFrame(np.array([7000.0, 100.0, 0.0]), km, J2000)
    vel1 = scb.ArrayWFrame(np.array([-0.1, 7.5, 0.0]), km / sec, J2000)
    state1 = scb.StateArray(
        scb.EpochArray(et0 + 100.0, "TDB"), origin,
        state=scb.StateDefinition().position(sc, pos1).velocity(sc, vel1),
    )

    seq = scb.MissionSequence("TEST SEQUENCE")
    seq.addLeg("leg0", state_vector=state0, epochs=scb.EpochArray(np.array([et0, et0 + 100.0]), sys="TDB"))
    seq.addLeg("leg1", state_vector=state1, epochs=scb.EpochArray(np.array([et0 + 100.0, et0 + 200.0]), sys="TDB"))
    return seq, et0


@pytest.fixture
def two_leg_data():
    """Leg metadata for two_leg_sequence, whose legs sit at sequence indices
    0 and 2 around the auto-inserted Node."""
    meta = {"body_ID": -999011, "origin_ID": 399, "reference_frame": "J2000"}
    return {"by_seq_idx": {0: meta, 2: meta}}


@pytest.fixture
def leg_and_finite_burn_sequence(sc, origin, state_arr):
    """A MissionSequence with a leg followed by a Finite Burn segment,
    giving a ["Leg", "Node", "Finite Burn"] type sequence -- used to verify
    that _prepare_sequence_data's "typ != 'Leg'" fallback (distinct from the
    Impulsive-Burn/Node check) also skips a Finite Burn segment."""
    et0 = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")
    epochs_leg = scb.EpochArray(np.array([et0, et0 + 100.0]), sys="TDB")

    seq = scb.MissionSequence("TEST SEQUENCE")
    seq.addLeg("leg0", state_vector=state_arr, epochs=epochs_leg)

    pos_fb = scb.ArrayWFrame(np.array([7000.0, 100.0, 0.0]), km, J2000)
    vel_fb = scb.ArrayWFrame(np.array([-0.1, 7.5, 0.0]), km / sec, J2000)
    state_fb = scb.StateArray(
        scb.EpochArray(et0 + 100.0, "TDB"), origin,
        state=scb.StateDefinition().position(sc, pos_fb).velocity(sc, vel_fb),
    )
    seq.addFiniteBurn(
        "burn0", state_vector=state_fb,
        epochs=scb.EpochArray(np.array([et0 + 100.0, et0 + 110.0]), sys="TDB"),
    )
    return seq, et0


# endregion Fixtures #
# --------------------#

# ---------------#
# region  Tests  #
# ---------------#


class TestPrepareSequenceData:
    """Tests for SpiceManager._prepare_sequence_data."""

    def test_single_leg(self, one_leg_sequence, leg_data):
        """Builds the expected "legs" payload directly from a one-leg MissionSequence."""
        seq, et0 = one_leg_sequence
        data = scb.SpiceManager._prepare_sequence_data(seq, leg_data, None)

        assert list(data.keys()) == ["leg_0"]
        leg = data["leg_0"]
        assert leg["body_ID"] == -999011
        assert leg["origin_ID"] == 399
        assert leg["reference_frame"] == "J2000"
        assert leg["name"] == "leg0"
        # position/velocity are the only state components (indices 0, 1), so the
        # parameter list (state_definition[2:]) is empty.
        assert leg["parameters_def"] == []
        assert leg["parameters_body"] == []
        np.testing.assert_allclose(leg["epochsTDB"], [et0, et0 + 100.0])

    def test_skips_non_leg_segments(self, two_leg_sequence):
        """The auto-inserted boundary Node between two legs is skipped, not
        treated as a leg."""
        seq, _ = two_leg_sequence
        assert seq.types == ["Leg", "Node", "Leg"]

        leg_data = {
            "by_seq_idx": {
                0: {"body_ID": -999011, "origin_ID": 399, "reference_frame": "J2000"},
                2: {"body_ID": -999011, "origin_ID": 399, "reference_frame": "J2000"},
            }
        }
        data = scb.SpiceManager._prepare_sequence_data(seq, leg_data, None)

        # only the two "Leg" segments (indices 0 and 2) produce entries; the
        # auto-inserted "Node" at index 1 is skipped.
        assert set(data.keys()) == {"leg_0", "leg_1"}
        assert data["leg_0"]["name"] == "leg0"
        assert data["leg_1"]["name"] == "leg1"

    def test_skips_finite_burn_segment(self, leg_and_finite_burn_sequence):
        """The "typ != 'Leg'" fallback branch must also skip a "Finite Burn"
        segment -- it isn't caught by the earlier Impulsive-Burn/Node check."""
        seq, _ = leg_and_finite_burn_sequence
        assert seq.types == ["Leg", "Node", "Finite Burn"]

        leg_data = {
            "by_seq_idx": {
                0: {"body_ID": -999011, "origin_ID": 399, "reference_frame": "J2000"},
            }
        }
        data = scb.SpiceManager._prepare_sequence_data(seq, leg_data, None)

        assert list(data.keys()) == ["leg_0"]
        assert data["leg_0"]["name"] == "leg0"

    def test_flat_leg_data_format(self, one_leg_sequence):
        """The legacy leg_data format keys parallel lists by leg_i rather than
        by sequence index, and a per-leg ArrayWFrame parameters_JSON entry
        populates parameters/unitsPower/unitsScale/parameters_frame."""
        seq, _ = one_leg_sequence
        leg_data_flat = {
            "body_ID": [-999011],
            "origin_ID": [399],
            "reference_frame": ["J2000"],
        }
        mu_units = scb.Units.get_units(["km"]) ** 3 / scb.Units.get_units(["sec"]) ** 2
        parameters_json = [scb.ArrayWFrame(4902.8, mu_units, J2000)]

        data = scb.SpiceManager._prepare_sequence_data(seq, leg_data_flat, parameters_json)

        leg = data["leg_0"]
        assert leg["parameters"] == [4902.8]
        assert leg["parameters_frame"] == "J2000"
        assert leg["unitsPower"] == [0, 3, -2, 0]

    def test_non_arraywframe_parameter(self, one_leg_sequence):
        """A parameters_JSON entry that isn't an ArrayWFrame is passed through
        as-is, with empty units/frame metadata (no dimensions to report)."""
        seq, _ = one_leg_sequence
        leg_data_flat = {
            "body_ID": [-999011],
            "origin_ID": [399],
            "reference_frame": ["J2000"],
        }
        data = scb.SpiceManager._prepare_sequence_data(seq, leg_data_flat, [4902.8])

        leg = data["leg_0"]
        assert leg["parameters"] == pytest.approx(4902.8)
        assert leg["unitsPower"] == []
        assert leg["unitsScale"] == []
        assert leg["parameters_frame"] == []

    def test_missing_leg_metadata_raises(self, one_leg_sequence):
        """Raises KeyError when leg_data['by_seq_idx'] has no entry for a
        sequence index."""
        seq, _ = one_leg_sequence
        with pytest.raises(KeyError, match="leg_data missing metadata"):
            scb.SpiceManager._prepare_sequence_data(seq, {"by_seq_idx": {}}, None)


class TestWriteSpkMetadataJson:
    """Tests for SpiceManager.write_spk_metadata_json with a sequence_definition."""

    def test_stms(self, tmp_path, sc, one_leg_sequence, leg_data):
        """write_spk_metadata_json's sequence_definition path end to end,
        including _prepare_stm_data's sequence branch (STMs sidecar also gets
        a "legs" section)."""
        seq, et0 = one_leg_sequence
        spk_name = str(tmp_path / "test_seq_meta.bsp")
        stms = np.array([np.eye(6), np.eye(6)])

        scb.SpiceManager.write_spk_metadata_json(
            SPK_name=spk_name,
            body=sc,
            origin_ID=399,
            ref_frame=J2000,
            epochs_TDB=[et0, et0 + 100.0],
            degree_poly=1,
            STMs_JSON=stms,
            sequence_definition=seq,
            leg_data=leg_data,
        )

        base = str(tmp_path / "test_seq_meta")
        assert os.path.isfile(base + "_stms.json")
        stm_data = json.load(open(base + "_stms.json"))

        assert stm_data["body_ID"] == -999011
        assert "legs" in stm_data
        assert stm_data["legs"]["leg_0"]["name"] == "leg0"
        np.testing.assert_allclose(
            stm_data["STMs"][str(et0)], np.eye(6).tolist()
        )

    def test_parameters(self, tmp_path, sc, one_leg_sequence, leg_data):
        """Providing both parameters_JSON and sequence_definition must dispatch
        _prepare_parameters_data to the sequence branch (not the flat schema)."""
        seq, et0 = one_leg_sequence
        spk_name = str(tmp_path / "test_seq_params.bsp")
        mu_units = scb.Units.get_units(["km"]) ** 3 / scb.Units.get_units(["sec"]) ** 2
        parameters_json = [scb.ArrayWFrame(4902.8, mu_units, J2000)]

        scb.SpiceManager.write_spk_metadata_json(
            SPK_name=spk_name,
            body=sc,
            origin_ID=399,
            ref_frame=J2000,
            epochs_TDB=[et0, et0 + 100.0],
            degree_poly=1,
            parameters_JSON=parameters_json,
            sequence_definition=seq,
            leg_data=leg_data,
        )

        base = str(tmp_path / "test_seq_params")
        assert os.path.isfile(base + "_parameters.json")
        params = json.load(open(base + "_parameters.json"))
        assert "legs" in params
        assert params["legs"]["leg_0"]["parameters"] == [4902.8]

    def test_multi_parameter_leg_round_trip(self, tmp_path, sc, origin, one_leg_sequence, leg_data):
        """Every (name, ArrayWFrame) component of a leg is written, each with its
        own units, and get_parameters reads all of them back -- a finite-burn
        leg's dRA/dDEC must not be lost behind its k_thrust."""
        seq, et0 = one_leg_sequence
        spk_name = str(tmp_path / "test_seq_burn.bsp")
        parameters_json = [[
            ("k_thrust1", scb.ArrayWFrame(0.9, None, None)),
            ("dRA1", scb.ArrayWFrame(0.4, rad, J2000)),
            ("dDEC1", scb.ArrayWFrame(0.3, rad, J2000)),
        ]]

        scb.SpiceManager.write_spk_metadata_json(
            SPK_name=spk_name,
            body=sc,
            origin_ID=399,
            ref_frame=J2000,
            epochs_TDB=[et0, et0 + 100.0],
            degree_poly=1,
            parameters_JSON=parameters_json,
            sequence_definition=seq,
            leg_data=leg_data,
        )

        sidecar = str(tmp_path / "test_seq_burn_parameters.json")
        written = json.load(open(sidecar))["legs"]["leg_0"]["parameters"]
        assert written[0::2] == ["k_thrust1", "dRA1", "dDEC1"]

        stacked = scb.SpiceManager.get_parameters(
            sc, et0 + 50.0, J2000, origin, sidecar_path=sidecar
        )
        np.testing.assert_allclose(np.ravel(stacked.values), [0.9, 0.4, 0.3])

        split = scb.SpiceManager.get_parameters(
            sc, et0 + 50.0, J2000, origin, sidecar_path=sidecar, per_parameter=True
        )
        assert list(split) == ["k_thrust1", "dRA1", "dDEC1"]
        assert split["k_thrust1"].units.unitless
        assert split["dRA1"].units == rad
        assert split["dDEC1"].units == rad

    def test_stm_sidecar_does_not_shadow_parameters(self, tmp_path, sc, origin, one_leg_sequence, leg_data):
        """A sequence run writes its STM sidecar with the same "legs" section
        as its parameters sidecar. Looked up by folder, each must resolve to
        its own file: no "Multiple matching" warning, and both payloads read."""
        seq, et0 = one_leg_sequence
        spk_name = str(tmp_path / "test_seq_both.bsp")

        scb.SpiceManager.write_spk_metadata_json(
            SPK_name=spk_name,
            body=sc,
            origin_ID=399,
            ref_frame=J2000,
            epochs_TDB=[et0, et0 + 100.0],
            degree_poly=1,
            parameters_JSON=[[("k_thrust1", scb.ArrayWFrame(0.9, None, None))]],
            STMs_JSON=np.array([np.eye(6), 3.0 * np.eye(6)]),
            sequence_definition=seq,
            leg_data=leg_data,
        )
        assert "legs" in json.load(open(tmp_path / "test_seq_both_stms.json"))

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            params = scb.SpiceManager.get_parameters(
                sc, et0 + 50.0, J2000, origin, sidecar_path=str(tmp_path)
            )
            stm = scb.SpiceManager.get_STMs(
                sc, et0 + 50.0, J2000, origin, sidecar_path=str(tmp_path)
            )
        np.testing.assert_allclose(np.ravel(params.values), [0.9])
        np.testing.assert_allclose(stm, 2.0 * np.eye(6))

    def test_stms_per_leg_round_trip(self, tmp_path, sc, origin, two_leg_sequence, two_leg_data):
        """One STM block per leg, as a scenario hands them over, is written per
        leg and read back from the leg that holds the epoch. The legs share
        their boundary epoch, and the second leg's STM is larger."""
        seq, et0 = two_leg_sequence
        spk_name = str(tmp_path / "test_seq_legs.bsp")
        blocks = [
            np.array([np.eye(6).ravel(), 2.0 * np.eye(6).ravel()]),
            np.array([np.eye(9).ravel(), 3.0 * np.eye(9).ravel()]),
        ]

        scb.SpiceManager.write_spk_metadata_json(
            SPK_name=spk_name,
            body=sc,
            origin_ID=399,
            ref_frame=J2000,
            epochs_TDB=[et0, et0 + 100.0, et0 + 200.0],
            degree_poly=1,
            STMs_JSON=blocks,
            sequence_definition=seq,
            leg_data=two_leg_data,
        )

        stm_data = json.load(open(tmp_path / "test_seq_legs_stms.json"))
        assert list(stm_data["STMs"]) == ["leg_0", "leg_1"]
        np.testing.assert_allclose(
            stm_data["legs"]["leg_0"]["STM_epochsTDB"], [et0, et0 + 100.0]
        )
        np.testing.assert_allclose(
            stm_data["legs"]["leg_1"]["STM_epochsTDB"], [et0 + 100.0, et0 + 200.0]
        )

        def stm_at(t):
            return scb.SpiceManager.get_STMs(
                sc, t, J2000, origin, sidecar_path=str(tmp_path)
            )

        np.testing.assert_allclose(stm_at(et0 + 50.0), 1.5 * np.eye(6))
        np.testing.assert_allclose(stm_at(et0 + 100.0), 2.0 * np.eye(6))
        np.testing.assert_allclose(stm_at(et0 + 150.0), 2.0 * np.eye(9))

    def test_stms_matching_no_layout_raise(self, tmp_path, sc, two_leg_sequence, two_leg_data):
        """Blocks that fit neither the legs' grids nor the global grid are
        refused, rather than written under the wrong epochs."""
        seq, et0 = two_leg_sequence
        blocks = [np.tile(np.eye(6).ravel(), (3, 1))] * 2

        with pytest.raises(scb.SpiceManager.STMLayoutError):
            scb.SpiceManager.write_spk_metadata_json(
                SPK_name=str(tmp_path / "test_seq_bad.bsp"),
                body=sc,
                origin_ID=399,
                ref_frame=J2000,
                epochs_TDB=[et0, et0 + 100.0, et0 + 200.0],
                degree_poly=1,
                STMs_JSON=blocks,
                sequence_definition=seq,
                leg_data=two_leg_data,
            )

# endregion Tests #
# ---------------#
