# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =============================================#
#  Unit Tests for the CentroidingIdeal class  #
# =============================================#
import src.scarabaeus as scb

import pytest
import numpy as np

J2000, ITRF93, ECLIPJ2000, IAUEARTH = scb.Frame.generate_common_frames()
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


@pytest.fixture
def camera(sc):
    """
    Fixture to create a sample camera instrument for testing.
    """
    cam_fov_angular = scb.ArrayWUnits(np.array([np.deg2rad(2.5), np.deg2rad(2.5)]), rad)
    cam_fov_pixels = (2000, 2000)
    cam_ifov_angle = scb.ArrayWUnits(np.array([25 * 10**-6, 20 * 25**-6]), rad)
    cam_focal_length = scb.ArrayWUnits(150 * 10**-6, km)

    return scb.Camera(
        name="ORCCA_camera",
        associated_body_spice_id=sc.spice_id,
        camera_frame=J2000,
        fov_angular=cam_fov_angular,
        fov_pixels=cam_fov_pixels,
        ifov_angle=cam_ifov_angle,
        focal_length=cam_focal_length,)


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(camera):
    """
    Verifies that object is constructed correctly.
    """
    centroiding_sigma = scb.ArrayWUnits(np.array([1, 1]), None)

    Centroid = scb.CentroidingIdeal(
        name="Opnav model", camera=camera, sigma=centroiding_sigma)

    assert Centroid.name == "Opnav model"
    assert Centroid._camera == camera
    assert Centroid._sigma == centroiding_sigma


def test_measurement_dataset_val_logic(sc, epochs, camera):
    """Verify that the function prevents invalid input combinations."""
    sigma = scb.ArrayWUnits(np.array([1, 1]), None)
    meas_model = scb.CentroidingIdeal(
        name="CentroidIdeal", camera=camera, sigma=sigma)  # Example instantiation

    # Test 1: Provide nothing
    with pytest.raises(ValueError, match="Provide exactly one of"):
        meas_model.generate_measurement_dataset("test", target=sc)

    # Test 2: Provide both
    obs_mock = (epochs, np.zeros(epochs.size), None, np.zeros(epochs.size))
    with pytest.raises(ValueError, match="Provide only one of"):
        meas_model.generate_measurement_dataset(
            "test", target=sc, epochs=epochs, observed_meas=obs_mock)


def test_set_image_epochs_and_index(camera):
    """
    Verifies both image-index helpers.

    set_image_epochs builds an epoch→index dict from an ordered list and
    reverts to None when passed None.  set_image_index assigns a pre-built
    mapping directly and also reverts to None.
    """
    sigma = scb.ArrayWUnits(np.array([1, 1]), None)
    meas = scb.CentroidingIdeal("test", camera=camera, sigma=sigma)

    # set_image_epochs: list → dict
    meas.set_image_epochs([10.0, 20.0, 30.0])
    assert meas._epoch_to_image_index == {10.0: 0, 20.0: 1, 30.0: 2}

    # set_image_epochs: None → cleared
    meas.set_image_epochs(None)
    assert meas._epoch_to_image_index is None

    # set_image_index: pre-built dict assigned by reference
    mapping = {100.0: 0, 200.0: 1}
    meas.set_image_index(mapping)
    assert meas._epoch_to_image_index is mapping

    # set_image_index: None → cleared
    meas.set_image_index(None)
    assert meas._epoch_to_image_index is None


def test_computed_measurements(mocker, camera, sc, epochs):
    """
    Covers multiple branches of computed_measurements in one test.

    mocker (from pytest-mock) temporarily replaces _get_rel_cam_pos with a
    stub that returns a fixed position vector, so no SPICE kernels are needed.

    Branches exercised:
    - noiseless path: output shape is (N, 2)
    - noisy path: output shape is still (N, 2)
    - z == 0 branch: u and v are forced to 0
    - image_epochs kwarg: calls set_image_epochs before iterating
    """
    sigma = scb.ArrayWUnits(np.array([1.0, 1.0]), None)
    meas = scb.CentroidingIdeal("test", camera=camera, sigma=sigma)

    # Normal position (z != 0) — noiseless
    mocker.patch.object(
        meas, "_get_rel_cam_pos",
        return_value=scb.ArrayWUnits(np.array([100.0, 50.0, 500.0]), km),
    )
    _, _, _, result = meas.computed_measurements(target=sc, epoch_array=epochs)
    assert result.quantity.values.shape == (epochs.size, 2)

    # Noisy path
    _, _, _, result_noisy = meas.computed_measurements(
        target=sc, epoch_array=epochs, noisy=True
    )
    assert result_noisy.quantity.values.shape == (epochs.size, 2)

    # z == 0 branch → u = v = 0
    mocker.patch.object(
        meas, "_get_rel_cam_pos",
        return_value=scb.ArrayWUnits(np.array([1.0, 1.0, 0.0]), km),
    )
    _, _, _, result_z0 = meas.computed_measurements(target=sc, epoch_array=epochs)
    assert result_z0.quantity.values == pytest.approx(0.0)

    # image_epochs kwarg wires up the epoch→index mapping
    epoch_vals = [float(t) for t in epochs.times.values]
    meas.computed_measurements(
        target=sc, epoch_array=epochs, image_epochs=epoch_vals
    )
    assert meas._epoch_to_image_index == {t: i for i, t in enumerate(epoch_vals)}


class TestPartials:
    """Covers _partials via direct call (internal method, tested for coverage)."""

    def test_legacy_and_state_definition_paths(self, mocker, camera, sc, epochs):
        """
        mocker replaces _get_rel_cam_pos and get_rotation_to_camera so no SPICE
        kernels are required.

        Branches exercised:
        - state_definition=None (legacy): returns [row_u, row_v] each of length 6
        - explicit position+velocity state_definition: loop branches for each
          component type, still produces two 6-element rows
        """
        mock_rel_pos = scb.ArrayWUnits(np.array([100.0, 50.0, 500.0]), km)

        # Legacy path (no state_definition)
        meas_legacy = scb.CentroidingIdeal("test_legacy", camera=camera)
        mocker.patch.object(meas_legacy, "_get_rel_cam_pos", return_value=mock_rel_pos)
        mocker.patch.object(camera, "get_rotation_to_camera", return_value=np.eye(3))

        rows = meas_legacy._partials(target=sc, epoch=epochs[0])
        assert len(rows) == 2
        assert len(rows[0]) == 6   # 3 pos + 3 vel
        assert len(rows[1]) == 6

        # State-definition path (position + velocity entries)
        pos_awf = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
        vel_awf = scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000)
        state_def = [
            ("position", 3, "estimated", "dynamic", sc, pos_awf),
            ("velocity", 3, "estimated", "dynamic", sc, vel_awf),
        ]
        meas_sd = scb.CentroidingIdeal("test_sd", camera=camera, state_definition=state_def)
        mocker.patch.object(meas_sd, "_get_rel_cam_pos", return_value=mock_rel_pos)

        rows_sd = meas_sd._partials(target=sc, epoch=epochs[0])
        assert len(rows_sd) == 2
        assert len(rows_sd[0]) == 6
        assert len(rows_sd[1]) == 6


def test_camera_frame_accepts_string_and_frame():
    """A camera frame given as a name and as a Frame must behave identically.

    Previously the two disagreed: a string broke computed_measurements at the
    ArrayWFrame construction, while a Frame made get_rotation_to_camera fall
    through to the identity attitude instead of asking SPICE.
    """
    import spiceypy as spice

    et = 7.5e8
    truth = spice.pxform("J2000", "IAU_MARS", et)

    def build(frame, tag):
        return scb.Camera(
            name=f"frame_test_cam_{tag}",
            associated_body_spice_id=-1000,
            camera_frame=frame,
            fov_angular=scb.ArrayWUnits(np.array([np.deg2rad(2.5)] * 2), rad),
            fov_pixels=(2000, 2000),
            ifov_angle=scb.ArrayWUnits(np.array([25e-6, 25e-6]), rad),
            focal_length=scb.ArrayWUnits(150 * 10**-6, km),
        )

    from_name = build("IAU_MARS", "name")
    from_frame = build(scb.Frame("IAU_MARS"), "obj")

    for cam in (from_name, from_frame):
        # The name is canonical either way, and it is the real SPICE rotation
        assert cam.camera_frame == "IAU_MARS"
        assert cam.camera_frame_object.name == "IAU_MARS"
        np.testing.assert_allclose(cam.get_rotation_to_camera(et), truth, atol=1e-14)

    # An unknown frame still falls back to the attitude matrix, as documented
    unknown = build("NOT_A_SPICE_FRAME", "unknown")
    unknown.attitude_matrix = np.diag([1.0, -1.0, -1.0])
    np.testing.assert_allclose(
        unknown.get_rotation_to_camera(et), np.diag([1.0, -1.0, -1.0])
    )


def test_partials_stacked_bias_without_image_index(camera, epochs, mocker):
    """A stacked per-image bias with no image mapping must be refused.

    It used to emit two columns for a 2*N-wide component, silently shifting
    every later component's partials into the wrong state column.
    """
    mock_rel_pos = scb.ArrayWUnits(np.array([100.0, 50.0, 500.0]), km)
    sc_body = scb.Body("Mars", spice_id=499)

    state_def = [
        ("position", 3, "estimated", "dynamic", camera,
         scb.ArrayWFrame(np.zeros(3), km, J2000)),
        ("centroid_bias_C", 6, "estimated", "parameter", camera,
         scb.ArrayWFrame(np.zeros(6), km / km, J2000)),
        ("cam_pp_offset_C", 2, "estimated", "parameter", camera,
         scb.ArrayWFrame(np.zeros(2), km / km, J2000)),
    ]
    meas = scb.CentroidingIdeal("stacked", camera=camera, state_definition=state_def)
    mocker.patch.object(meas, "_get_rel_cam_pos", return_value=mock_rel_pos)

    meas.set_image_index(None)
    with pytest.raises(scb.CentroidingIdeal.MissingImageIndexError, match="one \\(bu, bv\\) bias per image"):
        meas._partials(target=sc_body, epoch=epochs[0])

    # With the mapping supplied the row is the full state width and the bias
    # indicator lands in this image's slot
    meas.set_image_index({float(epochs[0].times.values): 1})
    rows = meas._partials(target=sc_body, epoch=epochs[0])
    assert len(rows[0]) == 11  # 3 pos + 6 bias + 2 pp_offset
    assert rows[0][3 + 2] == 1.0 and rows[1][3 + 3] == 1.0
    # ... and the principal-point columns are still where they belong
    assert rows[0][9] == 1.0 and rows[1][10] == 1.0



def test_single_bias_pair_applies_to_every_image(camera, epochs, mocker):
    """
    With an image mapping set, a dim-2 centroid_bias shifts every image, and
    its partial is the identity on every image.

    It was sliced as if it were stacked per image, so only image 0 saw it.
    """
    import types

    mock_rel_pos = scb.ArrayWUnits(np.array([100.0, 50.0, 500.0]), km)
    target = scb.Body("Mars", spice_id=499)
    grid = epochs[:2]
    t = [float(epoch.times.values) for epoch in grid]
    bias = np.array([0.5, -0.25])
    state_def = [
        ("centroid_bias_C", 2, "estimated", "parameter", camera,
         scb.ArrayWFrame(bias, km / km, J2000)),
    ]

    plain = scb.CentroidingIdeal("plain", camera=camera)
    mocker.patch.object(plain, "_get_rel_cam_pos", return_value=mock_rel_pos)
    _, _, _, unbiased = plain.computed_measurements(target, epoch_array=grid)

    meas = scb.CentroidingIdeal("biased", camera=camera, state_definition=state_def)
    mocker.patch.object(meas, "_get_rel_cam_pos", return_value=mock_rel_pos)
    meas._reference_state_vector = types.SimpleNamespace(state=state_def)
    _, _, _, biased = meas.computed_measurements(target, epoch_array=grid, image_epochs=t)
    np.testing.assert_allclose(
        biased.quantity.values - unbiased.quantity.values,
        np.tile(bias, (2, 1)),
        rtol=0, atol=1e-12,
    )

    for epoch in grid:
        assert meas._partials(target=target, epoch=epoch) == [[1.0, 0.0], [0.0, 1.0]]



def test_forward_model_stacked_bias_without_image_index(camera, epochs, mocker):
    """
    The forward model refuses a stacked per-image bias with no image mapping,
    as the partials do.

    It used to apply image 0's pair to every epoch without a word.
    """
    import types

    mock_rel_pos = scb.ArrayWUnits(np.array([100.0, 50.0, 500.0]), km)
    target = scb.Body("Mars", spice_id=499)
    grid = epochs[:2]
    t = [float(epoch.times.values) for epoch in grid]
    stacked = np.array([0.5, -0.25, 1.5, -1.25])
    state_def = [
        ("centroid_bias_C", 4, "estimated", "parameter", camera,
         scb.ArrayWFrame(stacked, km / km, J2000)),
    ]
    meas = scb.CentroidingIdeal("stacked", camera=camera, state_definition=state_def)
    mocker.patch.object(meas, "_get_rel_cam_pos", return_value=mock_rel_pos)
    meas._reference_state_vector = types.SimpleNamespace(state=state_def)

    with pytest.raises(scb.CentroidingIdeal.MissingImageIndexError, match="one \\(bu, bv\\) bias per image"):
        meas.computed_measurements(target, epoch_array=grid)

    # With the mapping each image takes its own pair
    plain = scb.CentroidingIdeal("plain", camera=camera)
    mocker.patch.object(plain, "_get_rel_cam_pos", return_value=mock_rel_pos)
    _, _, _, unbiased = plain.computed_measurements(target, epoch_array=grid)
    _, _, _, biased = meas.computed_measurements(target, epoch_array=grid, image_epochs=t)
    np.testing.assert_allclose(
        biased.quantity.values - unbiased.quantity.values,
        np.reshape(stacked, (2, 2)),
        rtol=0, atol=1e-12,
    )


_CENTROID_POS = {
    "-1000": np.array([1.2e5, -3.4e4, 2.1e4]),
    "Beacon": np.array([1.23e5, -3.32e4, 1.6e4]),
}


def _centroid_fake_get_state(trgt_bdy, epoch_time, reference_frame, obsvr_bdy, **_):
    # The camera name is not in the table, so _get_rel_cam_pos takes its
    # inertial fallback and rotates by get_rotation_to_camera
    rel = _CENTROID_POS[str(trgt_bdy)] - _CENTROID_POS[str(obsvr_bdy)]
    return scb.ArrayWUnits(np.concatenate([rel, np.zeros(3)]), km)


def test_position_partials_match_central_differences(camera, sc, monkeypatch):
    """The position columns are the derivatives of (u, v).

    The camera-frame vector is R_ci @ (r_target − r_camera_body), so the imaged
    body takes +H and the camera's spacecraft −H; the two used to be swapped.
    SPICE is replaced by a table of inertial positions so each body can be moved.
    """
    monkeypatch.setattr(
        scb.SpiceManager, "get_state", staticmethod(_centroid_fake_get_state)
    )
    # Boresight a few mrad off the line of sight, so (u, v) is off-centre
    los = _CENTROID_POS["Beacon"] - _CENTROID_POS["-1000"]
    z = los / np.linalg.norm(los) + np.array([2e-3, -1e-3, 1.5e-3])
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    R_ci = np.vstack([x, np.cross(z, x), z])
    monkeypatch.setattr(camera, "get_rotation_to_camera", lambda *_: R_ci)

    beacon = scb.Body("Beacon", spice_id=2000001)
    epochs = scb.EpochArray(np.array([1.0e9, 1.0e9 + 60.0]), sys="TDB")
    state_def = [
        ("position", 3, "estimated", "dynamic", sc, scb.ArrayWFrame(np.zeros(3), km, J2000)),
        ("velocity", 3, "estimated", "dynamic", sc, scb.ArrayWFrame(np.zeros(3), km, J2000)),
        ("position", 3, "estimated", "dynamic", beacon, scb.ArrayWFrame(np.zeros(3), km, J2000)),
    ]
    model = scb.CentroidingIdeal("cen", camera=camera, state_definition=state_def)
    H = np.array(model._partials(beacon, epochs[0]))

    def pixels():
        _, _, _, meas = model.computed_measurements(beacon, epoch_array=epochs)
        return np.ravel(meas.quantity.values[0])

    step = 1e-2  # km
    for key, cols in (("-1000", slice(0, 3)), ("Beacon", slice(6, 9))):
        numeric = np.zeros((2, 3))
        for j in range(3):
            _CENTROID_POS[key][j] += step
            plus = pixels()
            _CENTROID_POS[key][j] -= 2 * step
            minus = pixels()
            _CENTROID_POS[key][j] += step
            numeric[:, j] = (plus - minus) / (2 * step)
        np.testing.assert_allclose(H[:, cols], numeric, rtol=1e-6, atol=0.0)

    # Without a state definition the row is the target's, as in the state path
    legacy = np.array(scb.CentroidingIdeal("leg", camera=camera)._partials(beacon, epochs[0]))
    np.testing.assert_allclose(legacy[:, 0:3], H[:, 6:9], rtol=1e-12)
