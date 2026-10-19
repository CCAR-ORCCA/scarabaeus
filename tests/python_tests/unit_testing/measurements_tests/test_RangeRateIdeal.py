# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ===========================================#
#  Unit Tests for the RangeRateIdeal class  #
# ===========================================#
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
        "Orbiter", -1000, scb.ArrayWUnits(2000.0, kg), Orbiter_area, Orbiter_cr_srp)


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(GS1):
    """
    Verifies that object is constructed correctly.
    """
    rangerate_sigma = scb.ArrayWUnits(1e-5, km / sec)
    name = "GS1 Ideal RangeRate Model"
    RangeRate_GS1 = scb.RangeRateIdeal(name=name, instrument=GS1, sigma=rangerate_sigma)

    assert RangeRate_GS1._name == name
    assert RangeRate_GS1._instrument == GS1
    assert RangeRate_GS1._sigma == rangerate_sigma


def test_measurement_dataset_val_logic(sc, epochs, GS1):
    """Verify that the function prevents invalid input combinations."""
    meas_model = scb.RangeRateIdeal(
        name="test_range_rate_ideal",
        instrument=GS1,
        sigma=scb.ArrayWUnits(0.001, km / sec),)

    # Test 1: Provide nothing
    with pytest.raises(ValueError, match="Provide exactly one of"):
        meas_model.generate_measurement_dataset("test", target=sc)

    # Test 2: Provide both
    obs_mock = (epochs, np.zeros(epochs.size), None, np.zeros(epochs.size))
    with pytest.raises(ValueError, match="Provide only one of"):
        meas_model.generate_measurement_dataset(
            "test", target=sc, epochs=epochs, observed_meas=obs_mock)


def _range_rate_fake_get_state(trgt_bdy, epoch_time, reference_frame, obsvr_bdy, **_):
    # A fixed 3-4-5 geometry moving radially: the range rate is 0.5 km/s
    return scb.ArrayWUnits(
        np.array([3.0e4, 4.0e4, 0.0, 0.3, 0.4, 0.0]), [km] * 3 + [km / sec] * 3
    )


def test_single_bias_applies_to_every_pass(GS1, sc, monkeypatch):
    """
    With a pass mapping set, a single range-rate bias shifts every pass, and
    its partial is 1 on every pass.

    A size-1 bias was sliced as if it were stacked per pass, so only pass 0
    saw it: a scalar meas_bias under MeasurementScheduler(per_pass_bias=True),
    or a dim-1 rangerate_bias in the state.
    """
    monkeypatch.setattr(
        scb.SpiceManager, "get_state", staticmethod(_range_rate_fake_get_state)
    )
    t = [1.0e9, 1.0e9 + 60.0, 1.0e9 + 7200.0]
    grid = scb.EpochArray(np.array(t), sys="TDB")
    pass_epochs = [t[:2], t[2:]]

    model = scb.RangeRateIdeal("rr", GS1, meas_bias=1.0e-3)
    _, _, _, meas = model.computed_measurements(
        sc, epoch_array=grid, pass_epochs=pass_epochs
    )
    np.testing.assert_allclose(
        np.ravel(meas.quantity.values), 0.5 + 1.0e-3, rtol=0, atol=1e-12
    )

    state_def = [
        ("rangerate_bias_GS", 1, "estimated", "parameter", GS1,
         scb.ArrayWFrame(np.zeros(1), km / sec, scb.Frame("J2000"))),
    ]
    model = scb.RangeRateIdeal("rr", GS1, state_definition=state_def)
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
        scb.SpiceManager, "get_state", staticmethod(_range_rate_fake_get_state)
    )
    grid = scb.EpochArray(np.array([1.0e9, 1.0e9 + 7200.0]), sys="TDB")

    model = scb.RangeRateIdeal(
        "rr", GS1, meas_bias=scb.ArrayWUnits(np.array([1.0e-3, 2.0e-3]), km / sec)
    )
    with pytest.raises(scb.RangeRateIdeal.MissingPassIndexError):
        model.computed_measurements(sc, epoch_array=grid)

    state_def = [
        ("rangerate_bias_GS", 2, "estimated", "parameter", GS1,
         scb.ArrayWFrame(np.zeros(2), km / sec, scb.Frame("J2000"))),
    ]
    model = scb.RangeRateIdeal("rr", GS1, state_definition=state_def)
    with pytest.raises(scb.RangeRateIdeal.MissingPassIndexError):
        model._partials(sc, next(iter(grid)))
