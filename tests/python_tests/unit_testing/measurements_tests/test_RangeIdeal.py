# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =======================================#
#  Unit Tests for the RangeIdeal class  #
# =======================================#
import src.scarabaeus as scb
import numpy as np
import pytest

kg, km, sec, rad, hr = scb.Units.get_units(["kg", "km", "sec", "rad", "hr"])


# --------------------#
# region    Fixtures #
# --------------------#
@pytest.fixture
def epochs():
    return scb.EpochArray.interval(
        "2028 Feb 1 00:00:00", "2028 Feb 2 00:00:00", scb.ArrayWUnits(1, hr), sys="TDB")


@pytest.fixture
def GS1():
    """
    Fixture to create a sample measurement for testing.
    """
    return scb.GroundStation("DSS-14")


@pytest.fixture
def sc():
    Orbiter_area = scb.ArrayWUnits(1e-06, km**2)
    Orbiter_cr_srp = scb.ArrayWUnits(1.5, None)
    return scb.Spacecraft(
        "Orbiter", -1000, scb.ArrayWUnits(2000.0, kg), Orbiter_area, Orbiter_cr_srp
    )


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(GS1):
    """
    Verifies that object is constructed correctly.
    """
    range_sigma = scb.ArrayWUnits(1e-3, km)
    Range_GS1 = scb.RangeIdeal(
        name="GS1 Ideal Range Model", instrument=GS1, sigma=range_sigma)

    assert Range_GS1._name == "GS1 Ideal Range Model"
    assert Range_GS1._instrument == GS1
    assert Range_GS1._sigma == range_sigma


def test_measurement_dataset_val_logic(sc, epochs, GS1):
    """Verify that the function prevents invalid input combinations."""
    meas_model = scb.RangeIdeal(
        name="test_range_ideal", instrument=GS1, sigma=scb.ArrayWUnits(0.001, km))

    # Test 1: Provide nothing
    with pytest.raises(ValueError, match="Provide exactly one of"):
        meas_model.generate_measurement_dataset("test", target=sc)

    # Test 2: Provide both
    obs_mock = (epochs, np.zeros(epochs.size), None, np.zeros(epochs.size))
    with pytest.raises(ValueError, match="Provide only one of"):
        meas_model.generate_measurement_dataset(
            "test", target=sc, epochs=epochs, observed_meas=obs_mock)


def _range_fake_get_state(trgt_bdy, epoch_time, reference_frame, obsvr_bdy, **_):
    # A fixed 3-4-5 geometry: the range is 5e4 km at every epoch
    return scb.ArrayWUnits(
        np.array([3.0e4, 4.0e4, 0.0, 0.3, 0.4, 0.0]), [km] * 3 + [km / sec] * 3
    )


def test_single_bias_applies_to_every_pass(GS1, sc, monkeypatch):
    """
    With a pass mapping set, a single range bias shifts every pass, and its
    partial is 1 on every pass.

    A size-1 bias was sliced as if it were stacked per pass, so only pass 0
    saw it: a scalar meas_bias under MeasurementScheduler(per_pass_bias=True),
    or a dim-1 range_bias in the state.
    """
    monkeypatch.setattr(
        scb.SpiceManager, "get_state", staticmethod(_range_fake_get_state)
    )
    t = [1.0e9, 1.0e9 + 60.0, 1.0e9 + 7200.0]
    grid = scb.EpochArray(np.array(t), sys="TDB")
    pass_epochs = [t[:2], t[2:]]

    model = scb.RangeIdeal("rng", GS1, meas_bias=0.5)
    _, _, _, meas = model.computed_measurements(
        sc, epoch_array=grid, pass_epochs=pass_epochs
    )
    np.testing.assert_allclose(
        np.ravel(meas.quantity.values), 5.0e4 + 0.5, rtol=0, atol=1e-9
    )

    state_def = [
        ("range_bias_GS", 1, "estimated", "parameter", GS1,
         scb.ArrayWFrame(np.zeros(1), km, scb.Frame("J2000"))),
    ]
    model = scb.RangeIdeal("rng", GS1, state_definition=state_def)
    model.set_pass_epochs(pass_epochs)
    assert [list(model._partials(sc, epoch)) for epoch in grid] == [[1.0]] * 3


def test_stacked_bias_without_pass_index(GS1, sc, monkeypatch):
    """
    A bias with one value per pass needs the pass of each epoch. Without a
    pass mapping the partials gave it one column instead of one per pass,
    silently shifting every later state component, and the forward model
    failed on a generic ArrayWUnits shape error. Both now raise
    MissingPassIndexError.
    """
    monkeypatch.setattr(
        scb.SpiceManager, "get_state", staticmethod(_range_fake_get_state)
    )
    grid = scb.EpochArray(np.array([1.0e9, 1.0e9 + 7200.0]), sys="TDB")

    model = scb.RangeIdeal(
        "rng", GS1, meas_bias=scb.ArrayWUnits(np.array([0.5, 0.7]), km)
    )
    with pytest.raises(scb.RangeIdeal.MissingPassIndexError):
        model.computed_measurements(sc, epoch_array=grid)

    state_def = [
        ("range_bias_GS", 2, "estimated", "parameter", GS1,
         scb.ArrayWFrame(np.zeros(2), km, scb.Frame("J2000"))),
    ]
    model = scb.RangeIdeal("rng", GS1, state_definition=state_def)
    with pytest.raises(scb.RangeIdeal.MissingPassIndexError):
        model._partials(sc, next(iter(grid)))


def _electronic_delay_state(station, spacecraft, values=(0.0, 0.0, 0.0), units=sec):
    """The station's uplink and downlink delays and the spacecraft's transponder delay."""
    names = ("uplink_delay_bias_DSS14", "transponder_delay_bias", "downlink_delay_bias_DSS14")
    return [(name, 1, "estimated", "parameter", owner,
             scb.ArrayWFrame(np.array([value]), units, None))
            for name, owner, value in zip(names, (station, spacecraft, station), values)]


def test_electronic_delays_lengthen_the_range_by_half_their_sum(GS1, sc, monkeypatch):
    """
    The station's delays at t1 and t3 and the spacecraft's transponder delay at
    t2 lengthen the round trip, so the range grows by c(t1 + t2 + t3)/2 at every
    epoch, whatever the pass.
    """
    monkeypatch.setattr(
        scb.SpiceManager, "get_state", staticmethod(_range_fake_get_state)
    )
    grid = scb.EpochArray(np.array([1.0e9, 1.0e9 + 7200.0]), sys="TDB")
    delays = (1.0e-6, 2.0e-6, 3.0e-6)

    model = scb.RangeIdeal(
        "rng", GS1, state_definition=_electronic_delay_state(GS1, sc, delays))
    *_, meas = model.computed_measurements(sc, epoch_array=grid)
    np.testing.assert_allclose(
        np.ravel(meas.quantity.values), 5.0e4 + 299792.458 * sum(delays) / 2.0,
        rtol=0, atol=1e-9,
    )

    # Another station's delays and another spacecraft's transponder delay
    other_station = scb.GroundStation("DSS-24")
    other_sc = scb.Spacecraft("Other", -1001, scb.ArrayWUnits(2000.0, kg))
    model = scb.RangeIdeal(
        "rng", GS1,
        state_definition=_electronic_delay_state(other_station, other_sc, delays))
    *_, meas = model.computed_measurements(sc, epoch_array=grid)
    np.testing.assert_allclose(np.ravel(meas.quantity.values), 5.0e4, rtol=0, atol=1e-9)


def test_electronic_delay_partials_are_half_the_speed_of_light(GS1, sc, monkeypatch):
    """
    Each delay this station and spacecraft own has a partial of c/2 km per
    declared unit; those of another station or spacecraft have none.
    """
    monkeypatch.setattr(
        scb.SpiceManager, "get_state", staticmethod(_range_fake_get_state)
    )
    epoch = next(iter(scb.EpochArray(np.array([1.0e9]), sys="TDB")))
    other_sc = scb.Spacecraft("Other", -1001, scb.ArrayWUnits(2000.0, kg))
    others = [
        ("downlink_delay_bias_DSS24", 1, "estimated", "parameter",
         scb.GroundStation("DSS-24"), scb.ArrayWFrame(np.zeros(1), sec, None)),
        ("transponder_delay_bias", 1, "estimated", "parameter", other_sc,
         scb.ArrayWFrame(np.zeros(1), sec, None)),
    ]
    half_c = 299792.458 / 2.0

    model = scb.RangeIdeal(
        "rng", GS1, state_definition=_electronic_delay_state(GS1, sc) + others)
    np.testing.assert_allclose(
        model._partials(sc, epoch), [half_c, half_c, half_c, 0.0, 0.0], rtol=1e-15)

    # A delay declared in milliseconds has its partial per millisecond
    msec = scb.Units.get_units(["msec"])
    model = scb.RangeIdeal(
        "rng", GS1, state_definition=_electronic_delay_state(GS1, sc, units=msec))
    np.testing.assert_allclose(model._partials(sc, epoch), [half_c * 1e-3] * 3, rtol=1e-15)
