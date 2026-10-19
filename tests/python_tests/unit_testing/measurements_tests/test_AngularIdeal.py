# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =========================================#
#  Unit Tests for the AngularIdeal class  #
# =========================================#
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
def test_initialization(sc):
    """
    Verifies that object is constructed correctly.
    """
    sigma = scb.ArrayWUnits([10e-6, 10e-6], rad)
    meas_model = scb.AngularIdeal( 
        name="AngularIdeal", observer=sc, sigma=sigma)  # Example instantiation

    assert meas_model._name == "AngularIdeal"
    assert meas_model._instrument == sc
    assert meas_model._sigma == sigma


def test_validation_logic(sc, epochs):
    """Verify that the function prevents invalid input combinations."""
    sigma = scb.ArrayWUnits([10e-6, 10e-6], rad)
    meas_model = scb.AngularIdeal(
        name="AngularIdeal", observer=sc, sigma=sigma)  # Example instantiation

    # Test 1: Provide nothing
    with pytest.raises(ValueError, match="Provide exactly one of"):
        meas_model.generate_measurement_dataset("test", target=sc)

    # Test 2: Provide both
    obs_mock = (epochs, np.zeros(epochs.size), None, np.zeros(epochs.size))
    with pytest.raises(ValueError, match="Provide only one of"):
        meas_model.generate_measurement_dataset(
            "test", target=sc, epochs=epochs, observed_meas=obs_mock)


# ---------------------------------------------------#
# region  Two-component dataset naming               #
# ---------------------------------------------------#
"""
A two-component observable is split into one scalar dataset per component, and
those dataset names become the residual keys in ``solution.json``. They are
built from the caller's ``dataset_name``, exactly as a scalar radiometric
dataset is, and suffixed with the model's own component labels — RA and
declination for an angular model, not the pixel-coordinate default.

Tests
-----
test_angular_components_are_ra_and_dec
    AngularIdeal measures angles on the sky, so it labels them as such.
test_pixel_models_name_the_detector_axes
    A model measuring detector coordinates names them sample and line.
test_dataset_name_reaches_the_split_datasets
    The caller's name is used, so two optical instruments cannot collide.
"""


def test_angular_components_are_ra_and_dec(GS1):
    """AngularIdeal measures angles on the sky, so it labels them as such."""
    model = scb.AngularIdeal(
        "angular", observer=GS1, sigma=scb.ArrayWUnits(np.array([1e-6, 1e-6]), rad)
    )

    assert model.component_labels == ("ra", "dec")


def test_pixel_models_name_the_detector_axes(camera):
    """A model measuring detector coordinates names them sample and line."""
    model = scb.CentroidingIdeal(
        "centroid", camera=camera, sigma=scb.ArrayWUnits(np.array([1.0, 1.0]), None)
    )

    assert model.component_labels == ("sample", "line")


@pytest.fixture
def camera():
    """Camera whose declared FOV and pixel scale agree."""
    return scb.Camera(
        name="test_cam",
        associated_body_spice_id=-1000,
        camera_frame=scb.Frame("J2000"),
        fov_angular=scb.ArrayWUnits(np.deg2rad([2.0, 2.0]), rad),
        fov_pixels=(2000, 2000),
        ifov_angle=scb.ArrayWUnits(np.deg2rad([2.0, 2.0]) / 2000.0, rad),
        focal_length=scb.ArrayWUnits(150e-06, km),
    )


def test_dataset_name_reaches_the_split_datasets(GS1, epochs):
    """The caller's name is used, so two optical instruments cannot collide."""
    moon = scb.CelestialBody.from_constants("Moon")
    model = scb.AngularIdeal(
        "NAC angular",
        observer=GS1,
        sigma=scb.ArrayWUnits(np.array([1e-6, 1e-6]), rad),
        observed_target=moon,
    )

    # Station and Moon are both in the locked kernels, so the real split runs
    # without a purpose-built spacecraft SPK.
    datasets = model.generate_measurement_dataset(
        dataset_name="NAC_AngularIdeal", target=moon, epochs=epochs[:4]
    )

    names = [d.set_name for d in datasets]
    assert names == ["NAC_AngularIdeal_ra", "NAC_AngularIdeal_dec"], (
        "the split datasets must be named from the caller's dataset_name, so "
        "two optical instruments in one run do not collide"
    )
    assert [d.measurement_type for d in datasets] == [
        "AngularIdeal ra",
        "AngularIdeal dec",
    ]


# ---------------------------------------------------#
# region  Default dataset naming                     #
# ---------------------------------------------------#
"""
The dataset name becomes the residual key in ``solution.json``. Radiometric
datasets already read as ``<station>_<measurement_type>``, so that convention is
now the default for every model rather than something the caller has to spell
out — and for a camera-based model the instrument *is* the camera, giving
``NAC_CentroidingIdeal``.

Tests
-----
test_default_name_follows_instrument_and_type
    The convention holds for radiometric and optical models alike.
test_default_name_is_used_when_none_is_supplied
    An omitted name no longer reaches the dataset as ``None`` / ``None_ra``.
test_explicit_name_still_wins
    A caller who names the dataset keeps control of the key.
"""


def test_default_name_follows_instrument_and_type(GS1, camera):
    """The convention holds for radiometric and optical models alike."""
    radiometric = scb.RangeIdeal("r", GS1, sigma=scb.ArrayWUnits(1e-3, km))
    optical = scb.CentroidingIdeal(
        "c", camera=camera, sigma=scb.ArrayWUnits(np.array([1.0, 1.0]), None)
    )

    assert radiometric.default_dataset_name == "DSS-14_RangeIdeal"
    assert optical.default_dataset_name == "test_cam_CentroidingIdeal"


def test_default_name_is_used_when_none_is_supplied(GS1, epochs):
    """An omitted name no longer reaches the dataset as ``None`` / ``None_ra``."""
    moon = scb.CelestialBody.from_constants("Moon")
    model = scb.AngularIdeal(
        "angular",
        observer=GS1,
        sigma=scb.ArrayWUnits(np.array([1e-6, 1e-6]), rad),
        observed_target=moon,
    )

    datasets = model.generate_measurement_dataset(target=moon, epochs=epochs[:3])

    assert [d.set_name for d in datasets] == [
        "DSS-14_AngularIdeal_ra",
        "DSS-14_AngularIdeal_dec",
    ]
    assert all("None" not in d.set_name for d in datasets)


def test_explicit_name_still_wins(GS1, epochs):
    """A caller who names the dataset keeps control of the key."""
    moon = scb.CelestialBody.from_constants("Moon")
    model = scb.AngularIdeal(
        "angular",
        observer=GS1,
        sigma=scb.ArrayWUnits(np.array([1e-6, 1e-6]), rad),
        observed_target=moon,
    )

    datasets = model.generate_measurement_dataset(
        "NAC_AngularIdeal", target=moon, epochs=epochs[:3]
    )

    assert [d.set_name for d in datasets] == [
        "NAC_AngularIdeal_ra",
        "NAC_AngularIdeal_dec",
    ]


# ---------------------------------------------------#
# region  Camera as observer                         #
# ---------------------------------------------------#
"""
AngularIdeal is CentroidingIdeal with the attitude taken out of the loop, so it
takes the same instrument: the observer may be a camera as well as the
spacecraft. A Camera's ``spice_id`` is already the body it rides on, so the two
forms observe from the same place — only the dataset name differs, which is the
point: the camera-named key matches the radiometric convention.

Tests
-----
test_camera_observer_names_the_dataset_after_the_camera
    A camera observer gives ``<camera>_AngularIdeal``.
test_camera_and_spacecraft_observers_agree
    Both resolve to the same body, so the observable is unchanged.
test_observer_designator_prefers_the_body_id
    The SPICE designator is the body id, which a camera name is not.
"""


def test_observer_designator_prefers_the_body_id(GS1, camera):
    """The SPICE designator is the body id, which a camera name is not."""
    from_camera = scb.AngularIdeal(
        "ang", observer=camera, sigma=scb.ArrayWUnits(np.array([1e-6, 1e-6]), rad)
    )
    from_station = scb.AngularIdeal(
        "ang", observer=GS1, sigma=scb.ArrayWUnits(np.array([1e-6, 1e-6]), rad)
    )

    assert from_camera._observer_designator == str(camera.spice_id)
    assert from_station._observer_designator == str(GS1.spice_id)


def test_camera_observer_names_the_dataset_after_the_camera(camera):
    """A camera observer gives ``<camera>_AngularIdeal``."""
    model = scb.AngularIdeal(
        "ang", observer=camera, sigma=scb.ArrayWUnits(np.array([1e-6, 1e-6]), rad)
    )

    assert model.default_dataset_name == "test_cam_AngularIdeal"


def test_camera_and_spacecraft_observers_agree(GS1, epochs):
    """Both resolve to the same body, so the observable is unchanged."""
    moon = scb.CelestialBody.from_constants("Moon")
    # A camera riding on the station: its spice_id is the station's body id.
    station_camera = scb.Camera(
        name="station_cam",
        associated_body_spice_id=GS1.spice_id,
        camera_frame=scb.Frame("J2000"),
        fov_angular=scb.ArrayWUnits(np.deg2rad([2.0, 2.0]), rad),
        fov_pixels=(2000, 2000),
        ifov_angle=scb.ArrayWUnits(np.deg2rad([2.0, 2.0]) / 2000.0, rad),
        focal_length=scb.ArrayWUnits(150e-06, km),
    )

    measured = []
    for observer in (station_camera, GS1):
        model = scb.AngularIdeal(
            "ang",
            observer=observer,
            sigma=scb.ArrayWUnits(np.array([1e-6, 1e-6]), rad),
            observed_target=moon,
        )
        _, _, _, meas = model.computed_measurements(
            target=moon, epoch_array=epochs[:3]
        )
        measured.append(np.asarray(meas.quantity.values, dtype=float))

    np.testing.assert_array_equal(measured[0], measured[1])


# ---------------------------------------------------#
# region  Position partials                          #
# ---------------------------------------------------#
"""
The relative position is ``r_target − r_observer``, so the target position
enters the (RA, Dec) partials with +H and the observer position with −H. The
analytic rows are compared with central differences of computed_measurements;
SPICE is replaced by a table of inertial positions so each body can be moved.

Tests
-----
test_position_partials_match_central_differences
    Observer and target columns, and the legacy (target-only) row.
"""

_ANGULAR_POS = {
    "-1000": np.array([1.2e5, -3.4e4, 2.1e4]),
    "Beacon": np.array([1.23e5, -3.32e4, 1.6e4]),
}


def _angular_fake_get_state(trgt_bdy, epoch_time, reference_frame, obsvr_bdy, **_):
    rel = _ANGULAR_POS[str(trgt_bdy)] - _ANGULAR_POS[str(obsvr_bdy)]
    return scb.ArrayWUnits(np.concatenate([rel, np.zeros(3)]), km)


def test_position_partials_match_central_differences(sc, monkeypatch):
    """Observer and target columns, and the legacy (target-only) row."""
    monkeypatch.setattr(
        scb.SpiceManager, "get_state", staticmethod(_angular_fake_get_state)
    )
    J2000 = scb.Frame("J2000")
    beacon = scb.Body("Beacon", spice_id=2000001)
    epochs = scb.EpochArray(np.array([1.0e9, 1.0e9 + 60.0]), sys="TDB")
    state_def = [
        ("position", 3, "estimated", "dynamic", sc, scb.ArrayWFrame(np.zeros(3), km, J2000)),
        ("velocity", 3, "estimated", "dynamic", sc, scb.ArrayWFrame(np.zeros(3), km, J2000)),
        ("position", 3, "estimated", "dynamic", beacon, scb.ArrayWFrame(np.zeros(3), km, J2000)),
    ]
    model = scb.AngularIdeal("ang", sc, state_definition=state_def)
    H = np.array(model._partials(beacon, epochs[0]))

    def ra_dec():
        _, _, _, meas = model.computed_measurements(beacon, epoch_array=epochs)
        return np.ravel(meas.quantity.values[0])

    step = 1e-2  # km
    for key, cols in (("-1000", slice(0, 3)), ("Beacon", slice(6, 9))):
        numeric = np.zeros((2, 3))
        for j in range(3):
            _ANGULAR_POS[key][j] += step
            plus = ra_dec()
            _ANGULAR_POS[key][j] -= 2 * step
            minus = ra_dec()
            _ANGULAR_POS[key][j] += step
            numeric[:, j] = (plus - minus) / (2 * step)
        np.testing.assert_allclose(H[:, cols], numeric, rtol=1e-6, atol=0.0)

    # The velocity columns are zero, and without a state definition the row
    # is the target's position + velocity, the same as the target block above
    np.testing.assert_array_equal(H[:, 3:6], 0.0)
    legacy = np.array(scb.AngularIdeal("leg", sc)._partials(beacon, epochs[0]))
    np.testing.assert_allclose(legacy[:, 0:3], H[:, 6:9], rtol=1e-12)


def test_start_and_end_epochs_build_the_grid(sc, monkeypatch):
    """
    epoch_start / epoch_end / tstep give the same rows as the equivalent array.

    The documented alternative to epoch_array used to build each epoch with
    EpochArray(..., timeFrame=...), which EpochArray does not accept, so the
    start/end form raised before computing anything.
    """
    monkeypatch.setattr(
        scb.SpiceManager, "get_state", staticmethod(_angular_fake_get_state)
    )
    beacon = scb.Body("Beacon", spice_id=2000001)
    model = scb.AngularIdeal("ang", sc)
    start = scb.EpochArray(1.0e9, sys="TDB")
    end = scb.EpochArray(1.0e9 + 180.0, sys="TDB")

    _, _, _, from_bounds = model.computed_measurements(
        beacon, epoch_start=start, epoch_end=end, tstep=60.0
    )
    grid = scb.EpochArray.interval(start, end, dt=scb.ArrayWUnits(60.0, sec), sys="TDB")
    _, _, _, from_array = model.computed_measurements(beacon, epoch_array=grid)

    assert from_bounds.quantity.values.shape[0] == len(grid.et) > 1
    np.testing.assert_array_equal(
        from_bounds.quantity.values, from_array.quantity.values
    )


def test_numeric_meas_bias_shifts_ra_and_dec(sc, monkeypatch):
    """
    A plain (b_ra, b_dec) meas_bias is taken in rad and added to each row.

    It used to raise AttributeError, because the bias was read as an
    ArrayWFrame's ``.quantity``.
    """
    monkeypatch.setattr(
        scb.SpiceManager, "get_state", staticmethod(_angular_fake_get_state)
    )
    beacon = scb.Body("Beacon", spice_id=2000001)
    grid = scb.EpochArray(np.array([1.0e9, 1.0e9 + 60.0]), sys="TDB")
    bias = [1.0e-5, -2.0e-5]

    _, _, _, plain = scb.AngularIdeal("ang", sc).computed_measurements(
        beacon, epoch_array=grid
    )
    _, _, _, biased = scb.AngularIdeal(
        "ang", sc, meas_bias=bias
    ).computed_measurements(beacon, epoch_array=grid)

    np.testing.assert_allclose(
        biased.quantity.values - plain.quantity.values,
        np.tile(bias, (2, 1)),
        rtol=0, atol=1e-15,
    )


def test_single_bias_pair_applies_to_every_image(sc, monkeypatch):
    """
    With an image mapping set, a single (b_ra, b_dec) pair shifts every image,
    and its partial is the identity on every image.

    A two-element bias was sliced as if it were stacked per image, so only
    image 0 saw it, whether it came from meas_bias or a dim-2 angular_bias.
    """
    monkeypatch.setattr(
        scb.SpiceManager, "get_state", staticmethod(_angular_fake_get_state)
    )
    J2000 = scb.Frame("J2000")
    beacon = scb.Body("Beacon", spice_id=2000001)
    t = [1.0e9, 1.0e9 + 60.0]
    grid = scb.EpochArray(np.array(t), sys="TDB")
    bias = [1.0e-5, -2.0e-5]

    _, _, _, plain = scb.AngularIdeal("ang", sc).computed_measurements(
        beacon, epoch_array=grid
    )
    _, _, _, biased = scb.AngularIdeal(
        "ang", sc, meas_bias=bias
    ).computed_measurements(beacon, epoch_array=grid, image_epochs=t)
    np.testing.assert_allclose(
        biased.quantity.values - plain.quantity.values,
        np.tile(bias, (2, 1)),
        rtol=0, atol=1e-15,
    )

    state_def = [
        ("angular_bias_A", 2, "estimated", "parameter", sc, scb.ArrayWFrame(np.zeros(2), rad, J2000)),
    ]
    model = scb.AngularIdeal("ang", sc, state_definition=state_def)
    model.set_image_epochs(t)
    for epoch in grid:
        assert model._partials(beacon, epoch) == [[1.0, 0.0], [0.0, 1.0]]


def test_partials_stacked_bias_without_image_index(sc, monkeypatch):
    """
    A stacked per-image angular bias with no image mapping is refused.

    It used to emit two columns for a 2*N-wide component, shifting every later
    component's partials into the wrong state column. CentroidingIdeal already
    refused this case; AngularIdeal now does too.
    """
    monkeypatch.setattr(
        scb.SpiceManager, "get_state", staticmethod(_angular_fake_get_state)
    )
    J2000 = scb.Frame("J2000")
    beacon = scb.Body("Beacon", spice_id=2000001)
    epoch = scb.EpochArray(np.array([1.0e9]), sys="TDB")[0]
    state_def = [
        ("position", 3, "estimated", "dynamic", sc, scb.ArrayWFrame(np.zeros(3), km, J2000)),
        ("angular_bias_A", 6, "estimated", "parameter", sc, scb.ArrayWFrame(np.zeros(6), rad, J2000)),
        ("position", 3, "estimated", "dynamic", beacon, scb.ArrayWFrame(np.zeros(3), km, J2000)),
    ]
    model = scb.AngularIdeal("stacked", sc, state_definition=state_def)

    with pytest.raises(
        scb.AngularIdeal.MissingImageIndexError, match="one \\(b_ra, b_dec\\) bias per image"
    ):
        model._partials(beacon, epoch)

    # With the mapping supplied the row is the full state width, the bias
    # indicator lands in this image's slot, and the target block follows it
    model.set_image_index({float(epoch.times.values): 1})
    ra, dec = model._partials(beacon, epoch)
    assert len(ra) == len(dec) == 12  # 3 sc pos + 6 bias + 3 beacon pos
    assert ra[3:9] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert dec[3:9] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    np.testing.assert_allclose(ra[9:12], -np.asarray(ra[0:3]))


def test_forward_model_stacked_bias_without_image_index(sc, monkeypatch):
    """
    The forward model refuses a stacked per-image bias with no image mapping,
    as the partials do.

    It used to apply image 0's pair to every epoch without a word.
    """
    monkeypatch.setattr(
        scb.SpiceManager, "get_state", staticmethod(_angular_fake_get_state)
    )
    beacon = scb.Body("Beacon", spice_id=2000001)
    t = [1.0e9, 1.0e9 + 60.0]
    grid = scb.EpochArray(np.array(t), sys="TDB")
    stacked = [1.0e-5, -2.0e-5, 3.0e-5, -4.0e-5]
    model = scb.AngularIdeal("ang", sc, meas_bias=stacked)

    with pytest.raises(
        scb.AngularIdeal.MissingImageIndexError, match="one \\(b_ra, b_dec\\) bias per image"
    ):
        model.computed_measurements(beacon, epoch_array=grid)

    # With the mapping each image takes its own pair
    _, _, _, plain = scb.AngularIdeal("ang", sc).computed_measurements(
        beacon, epoch_array=grid
    )
    _, _, _, biased = model.computed_measurements(beacon, epoch_array=grid, image_epochs=t)
    np.testing.assert_allclose(
        biased.quantity.values - plain.quantity.values,
        np.reshape(stacked, (2, 2)),
        rtol=0, atol=1e-15,
    )
