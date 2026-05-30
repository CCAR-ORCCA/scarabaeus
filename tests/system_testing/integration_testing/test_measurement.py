# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import json
import os
import numpy as np
import pytest
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu

# TODO: measurement files should be written to the temp folder, not the hard coded data folder.
#      This need to be addressed in the future, but for now we will just clean up after ourselves.
#      This is tied to write_observed_measurements and generate_measurement_dataset, which both write to the data folder.

kg, km, sec, rad = scb.Units.get_units(["kg", "km", "sec", "rad"])
J2000 = scb.Frame("J2000")


# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture(scope="module")
def orbiter_traj(sc, epochs):
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

    # propgate
    cannon = scb.ForceModelTranslation(primary_body=sc, cannonball_SRP=True)
    prop = scb.Propagator(
        primary_body=sc, state_vector=x0, tspan=epochs, force_models=cannon
    )
    prop.propagate()

    orbiter_traj = scb.Trajectory(
        "meas_sim_true.bsp", state_array=prop.propagated_state_array
    )

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
            meas_time_et, meas_sec, meas_obs, meas_outliers = (
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
            meas_time_et, meas_sec, meas_obs, meas_outliers = (
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
            meas_time_et, meas_sec, meas_obs, meas_outliers = (
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
            meas_time_et, meas_sec, meas_obs, meas_outliers = (
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
            meas_time_et, meas_sec, meas_obs, meas_outliers = (
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
            meas_time_et, meas_sec, meas_obs, meas_outliers = (
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
