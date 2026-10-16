# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ================================================#
#  Integration Tests for Measurement Models      #
# ================================================#
import json
import os
import numpy as np
import pytest
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import spiceypy as spice

# These tests write measurement files into the repo's data folder rather than a
# temp folder, and clean up after themselves, because write_observed_measurements
# and generate_measurement_dataset both write there by default.

kg, km, sec, rad = scb.Units.get_units(["kg", "km", "sec", "rad"])
J2000 = scb.Frame("J2000")


# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture(scope="module")
def orbiter_traj(sc, epochs, tmp_path_factory):
    """
    Generates a trajectory for the test spacecraft by propagating an initial state using a simple force model.
    The trajectory is saved as a SPICE kernel file for use in measurement generation and testing.
    """

    # define origin
    origin = scb.CelestialBody.from_constants("EARTH")

    # compute initial state
    a = 6778  # km, ~400km altitude LEO
    mu = origin.grav_param.values  # 398600.435436096 km^3/s^2
    v = np.sqrt((mu / a))
    pos0 = scb.ArrayWFrame(
        awu(
            np.array([a, 0, 0]),
            km,
        ),
        J2000,
    )
    vel0 = scb.ArrayWFrame(
        awu(
            np.array([0, v, 0]),
            km / sec,
        ),
        J2000,
    )
    x0 = scb.StateArray(
        epochs[0],
        origin,
        state=scb.StateDefinition().position(sc, pos0).velocity(sc, vel0),
    )

    # propagate
    cannon = scb.ForceModelTranslation(primary_body=sc, cannonball_SRP=True)
    prop = scb.Propagator(
        primary_body=sc, state_vector=x0, tspan=epochs, force_models=cannon
    )
    prop.propagate()

    tmp_bsp = str(tmp_path_factory.mktemp("meas_integ") / "meas_sim_true.bsp")
    orbiter_traj = scb.Trajectory(state_array=prop.propagated_state_array)
    orbiter_traj.write_to_spk(tmp_bsp)

    return orbiter_traj


@pytest.fixture(scope="module")
def observed_meas(epochs):
    def _factory(unit, obs_value=100.0, n_components=1, outlier_indices=(0,)):
        shape = (epochs.size,) if n_components == 1 else (epochs.size, n_components)
        fake_values = np.ones(shape) * obs_value
        obs_awf = scb.ArrayWFrame(awu(fake_values, unit), J2000)
        outlier_flags = np.zeros(epochs.size)
        for idx in outlier_indices:
            outlier_flags[idx] = 1
        return (epochs, np.arange(epochs.size), obs_awf, outlier_flags)

    return _factory


class TestMeasurements:
    """Verifies that each measurement model correctly writes simulated
    observables to disk and reads them back."""

    def test_range_ideal(self, sc, GS1, epochs, orbiter_traj):
        """
        Tests the RangeIdeal measurement model's ability to write observed measurements to a JSON file and read them back correctly.
        """

        file_name = "test_range_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:
            # Test the write_observed_measurements function
            meas_model = scb.RangeIdeal(
                name="rangeIdeal", instrument=GS1, sigma=awu(1e-3, km)
            )  # Example instantiation
            meas_model.write_observed_measurements(
                target=sc,
                epoch_array=epochs,
                frame=J2000,
                noisy=True,
                file_name=file_name,
            )
            assert os.path.exists(
                expected_path
            ), f"Measurement JSON not found at {expected_path}"

            with open(expected_path, "r") as f:
                content = json.load(f)

            # ensure correct name and measurement type
            assert content["name"] == file_name
            assert "meas_ideal" in content

            # ensure epochs exist and are sized correctly
            inner_data = content["meas_ideal"]
            assert len(inner_data["meas"]) == epochs.size
            assert "year" in inner_data
            assert "sec" in inner_data
            assert len(inner_data["year"]) == epochs.size
            assert "doy" in inner_data

            # ensure additional measurement parameters exist
            assert "spice_id" in inner_data
            assert "outlier_flag" in inner_data

            # Test the observed_measurements function
            meas_time_et, meas_sec, meas_obs, meas_outliers, _aux_dict = (
                meas_model.observed_measurements(
                    file_name=expected_path,
                    meas_name="meas_ideal",
                    units=km,
                )
            )

            # ensure correct types are returned
            assert isinstance(
                meas_time_et, scb.EpochArray
            ), "Time was not converted to EpochArray"
            assert isinstance(
                meas_obs, scb.ArrayWFrame
            ), "Observations were not converted to ArrayWFrame"

            # ensure returned data is sized correctly and matches input epochs
            assert meas_time_et.size == epochs.size
            assert len(meas_sec) == epochs.size
            assert len(meas_outliers) == epochs.size
            np.testing.assert_allclose(
                meas_time_et.times.values, epochs.times.values, rtol=1e-12
            )
            assert meas_obs.quantity.units == km

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)

    def test_range_rate_ideal(self, sc, GS1, epochs, orbiter_traj):
        """
        Test the RangeRateIdeal measurement model's ability to write observed measurements to a JSON file and read them back correctly.
        """

        file_name = "test_range_rate_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:
            # Test the write_observed_measurements function
            meas_model = scb.RangeRateIdeal(
                name="rangeIdeal", instrument=GS1, sigma=awu(1e-3, km / sec)
            )  # Example instantiation
            meas_model.write_observed_measurements(
                target=sc,
                epoch_array=epochs,
                frame=J2000,
                noisy=True,
                file_name=file_name,
            )
            assert os.path.exists(
                expected_path
            ), f"Measurement JSON not found at {expected_path}"

            with open(expected_path, "r") as f:
                content = json.load(f)

            # ensure correct name and measurement type
            assert content["name"] == file_name
            assert "meas_ideal" in content

            # ensure epochs exist and are sized correctly
            inner_data = content["meas_ideal"]
            assert len(inner_data["meas"]) == epochs.size
            assert "year" in inner_data
            assert "sec" in inner_data
            assert len(inner_data["year"]) == epochs.size
            assert "doy" in inner_data

            # ensure additional measurement parameters exist
            assert "spice_id" in inner_data
            assert "outlier_flag" in inner_data

            # Test the observed_measurements function
            meas_time_et, meas_sec, meas_obs, meas_outliers, _aux_dict = (
                meas_model.observed_measurements(
                    file_name=expected_path,
                    meas_name="meas_ideal",
                    units=km,
                )
            )

            # ensure correct types are returned
            assert isinstance(
                meas_time_et, scb.EpochArray
            ), "Time was not converted to EpochArray"
            assert isinstance(
                meas_obs, scb.ArrayWFrame
            ), "Observations were not converted to ArrayWFrame"

            # ensure returned data is sized correctly and matches input epochs
            assert meas_time_et.size == epochs.size
            assert len(meas_sec) == epochs.size
            assert len(meas_outliers) == epochs.size
            np.testing.assert_allclose(
                meas_time_et.times.values, epochs.times.values, rtol=1e-12
            )
            assert meas_obs.quantity.units == km

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)

    def test_angular_ideal(self, sc, epochs, orbiter_traj):
        """
        Test the AngularIdeal measurement model's ability to write observed measurements to a JSON file and read them back correctly.
        """

        file_name = "test_angular_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:
            # Test the write_observed_measurements function
            meas_model = scb.AngularIdeal(
                name="AngularIdeal", observer=sc, sigma=awu([10e-6, 10e-6], rad)
            )  # Example instantiation
            meas_model.write_observed_measurements(
                target=sc,
                epoch_array=epochs,
                frame=J2000,
                noisy=True,
                file_name=file_name,
            )
            assert os.path.exists(
                expected_path
            ), f"Measurement JSON not found at {expected_path}"

            with open(expected_path, "r") as f:
                content = json.load(f)

            # ensure correct name and measurement type
            assert content["name"] == file_name
            assert "meas_ideal" in content

            # ensure epochs exist and are sized correctly
            inner_data = content["meas_ideal"]
            assert len(inner_data["meas"]) == epochs.size
            assert "year" in inner_data
            assert "sec" in inner_data
            assert len(inner_data["year"]) == epochs.size
            assert "doy" in inner_data

            # ensure additional measurement parameters exist
            assert "spice_id" in inner_data
            assert "outlier_flag" in inner_data

            # Test the observed_measurements function
            meas_time_et, meas_sec, meas_obs, meas_outliers, _aux_dict = (
                meas_model.observed_measurements(
                    file_name=expected_path,
                    meas_name="meas_ideal",
                    units=km,
                )
            )

            # ensure correct types are returned
            assert isinstance(
                meas_time_et, scb.EpochArray
            ), "Time was not converted to EpochArray"
            assert isinstance(
                meas_obs, scb.ArrayWFrame
            ), "Observations were not converted to ArrayWFrame"

            # ensure returned data is sized correctly and matches input epochs
            assert meas_time_et.size == epochs.size
            assert len(meas_sec) == epochs.size
            assert len(meas_outliers) == epochs.size
            np.testing.assert_allclose(
                meas_time_et.times.values, epochs.times.values, rtol=1e-12
            )
            assert meas_obs.quantity.units == km

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)

    def test_centroid_ideal(self, sc, epochs, orbiter_traj, camera):
        """
        Test the CentroidingIdeal measurement model's ability to write observed measurements to a JSON file and read them back correctly.
        """

        file_name = "test_centroid_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:
            # Test the write_observed_measurements function
            meas_model = scb.CentroidingIdeal(
                name="CentroidIdeal",
                camera=camera,
                sigma=scb.ArrayWUnits(np.array([1, 1]), None),
            )  # Example instantiation
            meas_model.write_observed_measurements(
                target=sc,
                epoch_array=epochs,
                frame=J2000,
                noisy=True,
                file_name=file_name,
            )
            assert os.path.exists(
                expected_path
            ), f"Measurement JSON not found at {expected_path}"

            with open(expected_path, "r") as f:
                content = json.load(f)

            # ensure correct name and measurement type
            assert content["name"] == file_name
            assert "meas_ideal" in content

            # ensure epochs exist and are sized correctly
            inner_data = content["meas_ideal"]
            assert len(inner_data["meas"]) == epochs.size
            assert "year" in inner_data
            assert "sec" in inner_data
            assert len(inner_data["year"]) == epochs.size
            assert "doy" in inner_data
            assert "spice_id" in inner_data
            assert "outlier_flag" in inner_data

            # Test the observed_measurements function
            meas_time_et, meas_sec, meas_obs, meas_outliers, _aux_dict = (
                meas_model.observed_measurements(
                    file_name=expected_path,
                    meas_name="meas_ideal",
                    units=km,
                )
            )

            # ensure correct types are returned
            assert isinstance(
                meas_time_et, scb.EpochArray
            ), "Time was not converted to EpochArray"
            assert isinstance(
                meas_obs, scb.ArrayWFrame
            ), "Observations were not converted to ArrayWFrame"

            # ensure returned data is sized correctly and matches input epochs
            assert meas_time_et.size == epochs.size
            assert len(meas_sec) == epochs.size
            assert len(meas_outliers) == epochs.size
            np.testing.assert_allclose(
                meas_time_et.times.values, epochs.times.values, rtol=1e-12
            )
            assert meas_obs.quantity.units == km

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)

    def test_diff_oneway_range_ideal(self, sc, GS1, GS2, epochs, orbiter_traj):
        """
        Test the DiffOneWayRangeIdeal measurement model's ability to write observed measurements to a JSON file and read them back correctly.
        """

        file_name = "test_diff_oneway_range_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:
            # Test the write_observed_measurements function
            meas_model = scb.DiffOneWayRangeIdeal(
                name="GS1 DOR",
                instrument=GS1,
                sigma=scb.ArrayWUnits(1e-3, sec),
                ground_station_2=GS2,
            )  # Example instantiation
            meas_model.write_observed_measurements(
                target=sc,
                epoch_array=epochs,
                frame=J2000,
                noisy=True,
                file_name=file_name,
            )
            assert os.path.exists(
                expected_path
            ), f"Measurement JSON not found at {expected_path}"

            with open(expected_path, "r") as f:
                content = json.load(f)

            # ensure correct name and measurement type
            assert content["name"] == file_name
            assert "meas_ideal" in content

            # ensure epochs exist and are sized correctly
            inner_data = content["meas_ideal"]
            assert len(inner_data["meas"]) == epochs.size
            assert "year" in inner_data
            assert "sec" in inner_data
            assert len(inner_data["year"]) == epochs.size
            assert "doy" in inner_data

            # ensure additional measurement parameters exist
            assert "spice_id" in inner_data
            assert "outlier_flag" in inner_data

            # Test the observed_measurements function
            meas_time_et, meas_sec, meas_obs, meas_outliers, _aux_dict = (
                meas_model.observed_measurements(
                    file_name=expected_path,
                    meas_name="meas_ideal",
                    units=km,
                )
            )

            # ensure correct types are returned
            assert isinstance(
                meas_time_et, scb.EpochArray
            ), "Time was not converted to EpochArray"
            assert isinstance(
                meas_obs, scb.ArrayWFrame
            ), "Observations were not converted to ArrayWFrame"

            # ensure returned data is sized correctly and matches input epochs
            assert meas_time_et.size == epochs.size
            assert len(meas_sec) == epochs.size
            assert len(meas_outliers) == epochs.size
            np.testing.assert_allclose(
                meas_time_et.times.values, epochs.times.values, rtol=1e-12
            )
            assert meas_obs.quantity.units == km

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)

    def test_doppler_ideal(self, sc, epochs, orbiter_traj, GS1, antenna):
        """
        Test the DopplerIdeal measurement model's ability to write observed measurements to a JSON file and read them back correctly.
        """

        file_name = "test_centroid_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:
            # Test the write_observed_measurements function
            meas_model = scb.DopplerIdeal(
                name="dopplerIdeal",
                instrument=GS1,
                sigma=scb.ArrayWUnits(1e-3, sec**-1),
                antenna_name=antenna.name,
                doppler_transmit_frequency=awu(8.8 * 10**9, sec**-1),
            )  # Example instantiation
            meas_model.write_observed_measurements(
                target=sc,
                epoch_array=epochs,
                frame=J2000,
                noisy=True,
                file_name=file_name,
            )
            assert os.path.exists(
                expected_path
            ), f"Measurement JSON not found at {expected_path}"

            with open(expected_path, "r") as f:
                content = json.load(f)

            # ensure correct name and measurement type
            assert content["name"] == file_name
            assert "meas_ideal" in content

            # ensure epochs exist and are sized correctly
            inner_data = content["meas_ideal"]
            assert len(inner_data["meas"]) == epochs.size
            assert "year" in inner_data
            assert "sec" in inner_data
            assert len(inner_data["year"]) == epochs.size
            assert "doy" in inner_data

            # ensure additional measurement parameters exist
            assert "spice_id" in inner_data
            assert "outlier_flag" in inner_data

            # Test the observed_measurements function
            meas_time_et, meas_sec, meas_obs, meas_outliers, _aux_dict = (
                meas_model.observed_measurements(
                    file_name=expected_path,
                    meas_name="meas_ideal",
                    units=km,
                )
            )

            # ensure correct types are returned
            assert isinstance(
                meas_time_et, scb.EpochArray
            ), "Time was not converted to EpochArray"
            assert isinstance(
                meas_obs, scb.ArrayWFrame
            ), "Observations were not converted to ArrayWFrame"

            # ensure returned data is sized correctly and matches input epochs
            assert meas_time_et.size == epochs.size
            assert len(meas_sec) == epochs.size
            assert len(meas_outliers) == epochs.size
            np.testing.assert_allclose(
                meas_time_et.times.values, epochs.times.values, rtol=1e-12
            )
            assert meas_obs.quantity.units == km

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)


class TestMeasurementDataSetGenerationRangeIdeal:
    """Verifies end-to-end MeasurementDataSet generation for RangeIdeal."""

    def test_epochs_only_mode(self, sc, epochs, GS1, orbiter_traj):
        """Test generating a dataset for future epochs (simulation mode)."""

        file_name = "test_range_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:

            meas_model = scb.RangeIdeal(
                name="test_range_ideal",
                instrument=GS1,
                sigma=scb.ArrayWUnits(0.001, km),
            )

            # Generate dataset using only epochs (no observed measurements)
            mds = meas_model.generate_measurement_dataset(
                dataset_name="SimDataset", target=sc, epochs=epochs, frame=J2000
            )
            data_dict = mds.data
            computed_vals = data_dict["computed"]
            obs_vals = data_dict["observed"]
            res_vals = data_dict["residuals"]
            partials = data_dict["partials"]
            sigmas = data_dict["sigma"]

            # Verify dataset size and contents
            assert mds.set_name == "SimDataset"
            assert np.all(
                obs_vals == 0
            ), "Observed values should be zero in epochs-only mode"
            assert np.all(res_vals == 0), "Residuals should be zero in epochs-only mode"
            assert len(computed_vals) == epochs.size
            assert len(partials) == epochs.size
            assert len(sigmas) == epochs.size

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)

    def test_observed_mode(self, sc, epochs, GS1, orbiter_traj, observed_meas):
        """Test generating a dataset with real observations (filtering mode)."""

        file_name = "test_range_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:

            meas_model = scb.RangeIdeal(
                name="test_range_ideal",
                instrument=GS1,
                sigma=scb.ArrayWUnits(0.001, km),
            )

            observed_meas = observed_meas(km)
            mds = meas_model.generate_measurement_dataset(
                dataset_name="ObsDataset",
                target=sc,
                observed_meas=observed_meas,
                frame=J2000,
            )
            data_dict = mds.data
            res_vals = data_dict["residuals"]
            obs_vals = data_dict["observed"]
            outlier_vals = data_dict["outlier_flag"]

            # Verify residuals are non-zero, observations match, and outlier flags are intact.
            assert not np.all(res_vals == 0), "Residuals were not calculated correctly!"
            np.testing.assert_allclose(obs_vals, np.ones(epochs.size) * 100.0)
            assert (
                outlier_vals[0] == 1
            ), "The outlier flag was lost during dataset generation"
            assert outlier_vals[1] == 0

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)


class TestMeasurementDataSetGenerationRangeRateIdeal:
    """Verifies end-to-end MeasurementDataSet generation for RangeRateIdeal."""

    def test_epochs_only_mode(self, sc, epochs, GS1, orbiter_traj):
        """Test generating a dataset for future epochs (simulation mode)."""

        file_name = "test_range_rate_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:

            meas_model = scb.RangeRateIdeal(
                name="test_range_rate_ideal",
                instrument=GS1,
                sigma=scb.ArrayWUnits(0.001, km / sec),
            )

            # Generate dataset using only epochs (no observed measurements)
            mds = meas_model.generate_measurement_dataset(
                dataset_name="SimDataset", target=sc, epochs=epochs, frame=J2000
            )
            data_dict = mds.data
            computed_vals = data_dict["computed"]
            obs_vals = data_dict["observed"]
            res_vals = data_dict["residuals"]
            partials = data_dict["partials"]
            sigmas = data_dict["sigma"]

            # Verify dataset size and contents
            assert mds.set_name == "SimDataset"
            assert np.all(
                obs_vals == 0
            ), "Observed values should be zero in epochs-only mode"
            assert np.all(res_vals == 0), "Residuals should be zero in epochs-only mode"
            assert len(computed_vals) == epochs.size
            assert len(partials) == epochs.size
            assert len(sigmas) == epochs.size

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)

    def test_observed_mode(self, sc, epochs, GS1, orbiter_traj, observed_meas):
        """Test generating a dataset with real observations (filtering mode)."""

        file_name = "test_range_rate_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:

            meas_model = scb.RangeRateIdeal(
                name=file_name, instrument=GS1, sigma=scb.ArrayWUnits(0.001, km / sec)
            )

            observed_meas = observed_meas(km / sec)
            # Create measurement dataset using observed measurements
            mds = meas_model.generate_measurement_dataset(
                dataset_name="ObsDataset",
                target=sc,
                observed_meas=observed_meas,
                frame=J2000,
            )
            data_dict = mds.data
            res_vals = data_dict["residuals"]
            obs_vals = data_dict["observed"]
            outlier_vals = data_dict["outlier_flag"]

            # Verify residuals are non-zero, observations match, and outlier flags are intact.
            assert not np.all(res_vals == 0), "Residuals were not calculated correctly!"
            np.testing.assert_allclose(obs_vals, np.ones(epochs.size) * 100.0)
            assert (
                outlier_vals[0] == 1
            ), "The outlier flag was lost during dataset generation"
            assert outlier_vals[1] == 0

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)


class TestMeasurementDataSetGenerationAngularIdeal:
    """Verifies end-to-end MeasurementDataSet generation for AngularIdeal."""

    def test_epochs_only_mode(self, sc, epochs, orbiter_traj):
        """Test generating a dataset for future epochs (simulation mode)."""

        file_name = "test_range_rate_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:

            meas_model = scb.AngularIdeal(
                name="AngularIdeal", observer=sc, sigma=awu([10e-6, 10e-6], rad)
            )  # Example instantiation

            # Generate dataset using only epochs (no observed measurements)
            mds = meas_model.generate_measurement_dataset(
                dataset_name="SimDataset", target=sc, epochs=epochs, frame=J2000
            )
            data_dict = mds[0].data
            computed_vals = data_dict["computed"]
            obs_vals = data_dict["observed"]
            res_vals = data_dict["residuals"]
            partials = data_dict["partials"]
            sigmas = data_dict["sigma"]

            # Verify dataset size and contents
            assert np.all(
                obs_vals == 0
            ), "Observed values should be zero in epochs-only mode"
            assert np.all(res_vals == 0), "Residuals should be zero in epochs-only mode"
            assert len(computed_vals) == epochs.size
            assert len(partials) == epochs.size
            assert len(sigmas) == epochs.size

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)

    def test_observed_mode(self, sc, epochs, orbiter_traj, observed_meas):
        """Test generating a dataset with real observations (filtering mode)."""

        file_name = "test_range_rate_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:

            meas_model = scb.AngularIdeal(
                name="AngularIdeal", observer=sc, sigma=awu([10e-6, 10e-6], rad)
            )  # Example instantiation

            observed_meas = observed_meas(rad, obs_value=0.5, n_components=2)
            # Create measurement dataset using observed measurements
            mds = meas_model.generate_measurement_dataset(
                dataset_name="ObsDatasetAngular",
                target=sc,
                observed_meas=observed_meas,
                frame=J2000,
            )
            data_dict = mds[0].data
            res_vals = data_dict["residuals"]

            # Verify mds is a list of MeasurementDataSet objects (one per angle component)
            assert isinstance(mds, list)
            assert len(mds) == 2

            # Verify residuals are non-zero, observations match, and outlier flags are intact.
            assert not np.all(res_vals == 0), "Residuals were not calculated correctly!"
            ra_mds = mds[0]
            assert "observed" in ra_mds.data
            assert ra_mds.data["observed"].shape == (epochs.size,)
            np.testing.assert_allclose(ra_mds.data["observed"], 0.5)

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)


class TestMeasurementDataSetGenerationCentroidingIdeal:
    """Verifies end-to-end MeasurementDataSet generation for CentroidingIdeal."""

    def test_epochs_only_mode(self, sc, epochs, orbiter_traj, camera):
        """Test generating a dataset for future epochs (simulation mode)."""

        file_name = "test_range_rate_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:

            meas_model = scb.CentroidingIdeal(
                name="CentroidIdeal",
                camera=camera,
                sigma=scb.ArrayWUnits(np.array([1, 1]), None),
            )  # Example instantiation

            # Generate dataset using only epochs (no observed measurements)
            mds = meas_model.generate_measurement_dataset(
                dataset_name="SimDataset", target=sc, epochs=epochs, frame=J2000
            )
            data_dict = mds[0].data
            computed_vals = data_dict["computed"]
            obs_vals = data_dict["observed"]
            res_vals = data_dict["residuals"]
            partials = data_dict["partials"]
            sigmas = data_dict["sigma"]

            # Verify dataset size and contents
            assert np.all(
                obs_vals == 0
            ), "Observed values should be zero in epochs-only mode"
            assert np.all(res_vals == 0), "Residuals should be zero in epochs-only mode"
            assert len(computed_vals) == epochs.size
            assert len(partials) == epochs.size
            assert len(sigmas) == epochs.size

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)

    def test_observed_mode(self, sc, epochs, orbiter_traj, observed_meas):
        """Test generating a dataset with real observations (filtering mode)."""

        file_name = "test_range_rate_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:

            meas_model = scb.AngularIdeal(
                name="AngularIdeal", observer=sc, sigma=awu([10e-6, 10e-6], rad)
            )  # Example instantiation

            observed_meas = observed_meas(rad, obs_value=0.5, n_components=2)
            # Create measurement dataset using observed measurements
            mds = meas_model.generate_measurement_dataset(
                dataset_name="ObsDatasetAngular",
                target=sc,
                observed_meas=observed_meas,
                frame=J2000,
            )
            data_dict = mds[0].data
            res_vals = data_dict["residuals"]

            # Verify mds is a list of MeasurementDataSet objects (one per angle component)
            assert isinstance(mds, list)
            assert len(mds) == 2

            # Verify residuals are non-zero, observations match, and outlier flags are intact.
            assert not np.all(res_vals == 0), "Residuals were not calculated correctly!"
            ra_mds = mds[0]
            assert "observed" in ra_mds.data
            assert ra_mds.data["observed"].shape == (epochs.size,)
            np.testing.assert_allclose(ra_mds.data["observed"], 0.5)

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)


class TestMeasurementDataSetGenerationDiffOneWayRangeIdeal:
    """Verifies end-to-end MeasurementDataSet generation for DiffOneWayRangeIdeal."""

    def test_epochs_only_mode(self, sc, epochs, orbiter_traj, GS1, GS2):
        """Test generating a dataset for future epochs (simulation mode)."""

        file_name = "test_range_rate_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:

            meas_model = scb.DiffOneWayRangeIdeal(
                name="GS1 DOR",
                instrument=GS1,
                sigma=scb.ArrayWUnits(1e-3, sec),
                ground_station_2=GS2,
            )  # Example instantiation

            # Generate dataset using only epochs (no observed measurements)
            mds = meas_model.generate_measurement_dataset(
                dataset_name="SimDataset", target=sc, epochs=epochs, frame=J2000
            )
            data_dict = mds.data
            computed_vals = data_dict["computed"]
            obs_vals = data_dict["observed"]
            res_vals = data_dict["residuals"]
            partials = data_dict["partials"]
            sigmas = data_dict["sigma"]

            # Verify dataset size and contents
            assert np.all(
                obs_vals == 0
            ), "Observed values should be zero in epochs-only mode"
            assert np.all(res_vals == 0), "Residuals should be zero in epochs-only mode"
            assert len(computed_vals) == epochs.size
            assert len(partials) == epochs.size
            assert len(sigmas) == epochs.size

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)

    def test_observed_mode(self, sc, epochs, GS1, GS2, orbiter_traj, observed_meas):
        """Test generating a dataset with real observations (filtering mode)."""

        file_name = "test_range_rate_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:

            meas_model = scb.DiffOneWayRangeIdeal(
                name="GS1 DOR",
                instrument=GS1,
                sigma=scb.ArrayWUnits(1e-3, sec),
                ground_station_2=GS2,
            )  # Example instantiation

            observed_meas = observed_meas(sec, obs_value=0.5, n_components=2)
            # Create measurement dataset using observed measurements
            mds = meas_model.generate_measurement_dataset(
                dataset_name="ObsDatasetAngular",
                target=sc,
                observed_meas=observed_meas,
                frame=J2000,
            )
            data_dict = mds.data
            res_vals = data_dict["residuals"]
            obs_vals = data_dict["observed"]
            outlier_vals = data_dict["outlier_flag"]

            # Verify residuals are non-zero, observations match, and outlier flags are intact.
            assert not np.all(res_vals == 0), "Residuals were not calculated correctly!"
            np.testing.assert_allclose(obs_vals, np.ones((epochs.size, 2)) * 0.5)
            assert (
                outlier_vals[0] == 1
            ), "The outlier flag was lost during dataset generation"
            assert outlier_vals[1] == 0

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)


class TestMeasurementDataSetGenerationDopplerIdeal:
    """Verifies end-to-end MeasurementDataSet generation for DopplerIdeal."""

    def test_epochs_only_mode(self, sc, epochs, orbiter_traj, GS1, antenna):
        """Test generating a dataset for future epochs (simulation mode)."""

        file_name = "test_range_rate_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:

            meas_model = scb.DopplerIdeal(
                name="dopplerIdeal",
                instrument=GS1,
                sigma=scb.ArrayWUnits(1e-3, sec**-1),
                antenna_name=antenna.name,
                doppler_transmit_frequency=awu(8.8 * 10**9, sec**-1),
            )  # Example instantiation

            # Generate dataset using only epochs (no observed measurements)
            mds = meas_model.generate_measurement_dataset(
                dataset_name="SimDataset", target=sc, epochs=epochs, frame=J2000
            )
            data_dict = mds.data
            computed_vals = data_dict["computed"]
            obs_vals = data_dict["observed"]
            res_vals = data_dict["residuals"]
            partials = data_dict["partials"]
            sigmas = data_dict["sigma"]

            # Verify dataset size and contents
            assert np.all(
                obs_vals == 0
            ), "Observed values should be zero in epochs-only mode"
            assert np.all(res_vals == 0), "Residuals should be zero in epochs-only mode"
            assert len(computed_vals) == epochs.size
            assert len(partials) == epochs.size
            assert len(sigmas) == epochs.size

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)

    def test_observed_mode(self, sc, epochs, GS1, antenna, orbiter_traj, observed_meas):
        """Test generating a dataset with real observations (filtering mode)."""

        file_name = "test_range_rate_ideal"
        # The function prepends 'data/measurements/radiometric/'
        expected_path = os.path.join(
            "data", "measurements", "radiometric", f"{file_name}.json"
        )

        try:

            meas_model = scb.DopplerIdeal(
                name="dopplerIdeal",
                instrument=GS1,
                sigma=scb.ArrayWUnits(1e-3, sec**-1),
                antenna_name=antenna.name,
                doppler_transmit_frequency=awu(8.8 * 10**9, sec**-1),
            )  # Example instantiation

            observed_meas = observed_meas(sec**-1, obs_value=0.5, n_components=2)
            # Create measurement dataset using observed measurements
            mds = meas_model.generate_measurement_dataset(
                dataset_name="ObsDatasetAngular",
                target=sc,
                observed_meas=observed_meas,
                frame=J2000,
            )
            data_dict = mds.data
            res_vals = data_dict["residuals"]
            obs_vals = data_dict["observed"]
            outlier_vals = data_dict["outlier_flag"]

            # Verify residuals are non-zero, observations match, and outlier flags are intact.
            assert not np.all(res_vals == 0), "Residuals were not calculated correctly!"
            np.testing.assert_allclose(obs_vals, np.ones((epochs.size, 2)) * 0.5)
            assert (
                outlier_vals[0] == 1
            ), "The outlier flag was lost during dataset generation"
            assert outlier_vals[1] == 0

        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)

class TestRLTComputation:
    def test_compute_RLT_lt(self, GS1, epochs, orbiter_traj):
        meas = scb.Measurement(name="Test", instrument=GS1)
        gs_id = str(GS1.spice_id)

        # Inside the coverage of the orbiter_traj SPK (-1000)
        t0 = float(np.ravel(epochs.et)[10])
        light_time = 0.02
        t1, t2, t3 = t0, t0 + light_time, t0 + 2 * light_time

        ssb_to_gs_t1 = spice.spkezr(gs_id, t1, "J2000", "NONE", "SSB")[0]
        sun_to_sc_t2 = spice.spkezr("-1000", t2, "J2000", "NONE", "10")[0]
        sun_to_gs_t1 = spice.spkezr(gs_id, t1, "J2000", "NONE", "10")[0]

        delta = np.zeros(6)

        # ── Case 1: RLT-SUN only ──
        dt_sun, r_vec_sun = meas._compute_RLT_lt(
            components="RLT-SUN",
            r1_r2_vec=sun_to_gs_t1[:3],
            r2_r3_vec=sun_to_sc_t2[:3],
        )
        assert dt_sun > 0
        assert r_vec_sun.shape == (3,)
        # Shapiro delay of the Sun over a near-Earth path, ~ (2 GM_S / c^3) r12 / AU
        r12 = np.linalg.norm(sun_to_sc_t2[:3] - sun_to_gs_t1[:3])
        c, mu_S = scb.constants.c.values, scb.constants.SUN.GM.values
        assert np.isclose(
            dt_sun, 2 * mu_S / c**3 * r12 / np.linalg.norm(sun_to_gs_t1[:3]), rtol=1e-2
        )

        # ── Case 2: RLT-BODIES only ──
        dt_bodies, r_vec_bodies = meas._compute_RLT_lt(
            components="RLT-BODIES",
            t2_t1=t1,
            t3_t2=t2,
            sc_spice_id="-1000",
            ssb_to_gs_t_31_RLT=ssb_to_gs_t1,
            way="up",
            delta=delta,
        )
        assert dt_bodies > 0
        assert np.all(r_vec_bodies == 0)
        assert 1e-12 < dt_bodies < 1e-4

        # ── Case 3: RLT combined == sum of parts ──
        dt_rlt, _ = meas._compute_RLT_lt(
            components="RLT",
            r1_r2_vec=sun_to_gs_t1,
            r2_r3_vec=sun_to_sc_t2,
            t2_t1=t1,
            t3_t2=t2,
            sc_spice_id="-1000",
            ssb_to_gs_t_31_RLT=ssb_to_gs_t1,
            way="up",
            delta=delta,
        )
        assert np.isclose(dt_rlt, dt_sun + dt_bodies, rtol=1e-10)
        assert dt_rlt > dt_sun
        assert dt_rlt > dt_bodies

        # ── Case 4: invalid component → zero ──
        dt_none, _ = meas._compute_RLT_lt(components="INVALID")
        assert dt_none == 0

        # ── Case 5: identical vectors → r32_r21 == 0, should not crash ──
        dt_zero, _ = meas._compute_RLT_lt(
            components="RLT-SUN",
            r1_r2_vec=sun_to_gs_t1,
            r2_r3_vec=sun_to_gs_t1,
        )
        assert np.isfinite(dt_zero)

    def test_compute_precise_RTLT(self, GS1, sc, epochs, orbiter_traj):
        meas = scb.Measurement(name="Test", instrument=GS1)
        gs_id = str(GS1.spice_id)

        # Inside the coverage of the orbiter_traj SPK (-1000)
        t3_et = float(np.ravel(epochs.et)[10]) + 7.35

        electronic_delays_dict = {
            "rcv_tt_dly": 0.0,
            "trn_tt_dly": 0.0,
            "array_delay": 0.0,
            "dl_zheight_corr": 0.0,
            "ul_zheight_corr": 0.0,
            "scft_transpd_delay_s": 0.0,
        }

        solar_corona_dict = {
            "ul_freq": 0.0,
            "M2": 0.0,
        }

        delta_partials = scb.ArrayWUnits(np.zeros(6), km)  # adjust to match your type

        rtlt, rtlt_parts, t2, t1 = meas._compute_precise_RTLT(
            sc_spice_id="-1000",
            gs_spice_id=gs_id,
            tt_et=t3_et,
            electronic_delays_dict=electronic_delays_dict,
            solar_corona_dict=solar_corona_dict,
            delta_partials=delta_partials,
            gs_delta_awf=None,
        )

        # ── Output structure ──
        assert rtlt_parts.shape == (17, 1)
        assert np.isclose(rtlt, np.sum(rtlt_parts), rtol=1e-10)  # rtlt == sum of parts

        # ── Time ordering: t1 < t2 < t3 ──
        assert t1 < t2 < t3_et, "Signal must travel forward in time"

        lt_dwn = rtlt_parts[0][0]  # CN down-leg
        lt_up = rtlt_parts[2][0]  # CN up-leg
        # t2 is the reception epoch minus the down-leg terms. The internal
        # reception epoch is the station TAI reading converted with the
        # station-dependent ET-TAI, a few microseconds from tt_et
        assert np.isclose(t3_et - t2, lt_dwn + rtlt_parts[1][0], rtol=0, atol=1e-5)
        # Two ulps of a 2029 ET epoch
        assert np.isclose(t2 - t1, lt_up + rtlt_parts[3][0], rtol=0, atol=2.5e-7)

        # The Newtonian legs are the station-spacecraft distances over c (a LEO
        # spacecraft, so the light times are a few hundredths of a second)
        c = scb.constants.c.values
        r_dwn = np.linalg.norm(
            spice.spkezr("-1000", t2, "J2000", "NONE", "SSB")[0][:3]
            - spice.spkezr(gs_id, t3_et, "J2000", "NONE", "SSB")[0][:3]
        )
        r_up = np.linalg.norm(
            spice.spkezr("-1000", t2, "J2000", "NONE", "SSB")[0][:3]
            - spice.spkezr(gs_id, t1, "J2000", "NONE", "SSB")[0][:3]
        )
        assert np.isclose(lt_dwn, r_dwn / c, rtol=0, atol=1e-6)
        assert np.isclose(lt_up, r_up / c, rtol=0, atol=1e-6)
        assert 0.0 < rtlt < 0.2, f"RTLT out of physical range: {rtlt}"

        # ── GR delays are small but non-zero ──
        dt_GR_dwn = rtlt_parts[1][0]
        dt_GR_up = rtlt_parts[3][0]
        assert dt_GR_dwn > 0, "GR down-leg delay must be positive"
        assert dt_GR_up > 0, "GR up-leg delay must be positive"
        assert dt_GR_dwn < lt_dwn, "GR delay must be smaller than Newtonian"
        assert dt_GR_up < lt_up, "GR delay must be smaller than Newtonian"

        # ── Electronic delays zero ──
        assert rtlt_parts[15][0] == 0.0  # dt_tau_D
        assert rtlt_parts[16][0] == 0.0  # dt_tau_U

        # ── Solar corona zero (we passed zeros) ──
        assert rtlt_parts[12][0] == 0.0  # dt_SC_dwn
        assert rtlt_parts[14][0] == 0.0  # dt_SC_up
