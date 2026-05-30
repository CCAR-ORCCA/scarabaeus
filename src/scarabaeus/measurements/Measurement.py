# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import scarabaeus as scb
from scarabaeus import (
    constants,
    Units,
    Frame,
    ArrayWUnits,
    ArrayWFrame,
    EpochArray,
    Body,
    Spacecraft,
    Instrument,
    SpiceManager,
    MeasurementDataSet,
    StateArray,
    StateDefinition,
)

import scarabaeus.utils.NumpyWrapper as np
from scipy.optimize import root_scalar
import spiceypy as spice
from tqdm import tqdm
import json
import os
from typing import Tuple

# ------------------#
#  Generate Units  #
# ------------------#
J2000, ITRF93, ECLIPJ2000, IAUEARTH = scb.Frame.generate_common_frames()
km, sec, rad = Units.get_units(["km", "sec", "rad"])

# Generate common units and frames
spc_pos_units = km
spc_time_units = sec
spc_angle_units = rad
spc_vel_units = spc_pos_units / spc_time_units
spc_ad_units = spc_pos_units / spc_pos_units

J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


# --------------------#
#  Class Definition  #
# --------------------#
class Measurement:
    """ Abstract base class for all Scarabaeus measurement models (radiometric and optical).

    Parameters
    ----------
    name : str
        The name of the Measurement object.
    instrument : Instrument
        The instrument object linked to the measurement model (e.g. an Antenna, a Camera, etc.)

    Raises
    ------
    ValueError
        If observed and computed measurements do not share the same reference frame and
        therefore cannot be differenced to form residuals.
    ValueError
        If a single measurement model generates more than 2 components per epoch,
        which is not supported in the current version.

    Notes
    -----
    This is the abstract interface that all subclasses implement.  Each concrete
    subclass must provide:

    * ``computed_measurements`` — evaluate the forward model on a trajectory.
    * ``observed_measurements`` — load observed data from a JSON file.
    * ``compute_partials`` — assemble the :math:`H` row (measurement partial).
    * ``generate_measurement_dataset`` — produce a
      :class:`~scarabaeus.MeasurementDataSet` ready for a filter.

    No measurement-specific formula is defined here; see subclass docstrings
    for the mathematical model of each observable.

    See Also
    --------
    scarabaeus.RangeIdeal : Ideal two-way range measurement model.
    scarabaeus.RangeRateIdeal : Ideal two-way range-rate measurement model.
    scarabaeus.DopplerIdeal : Ideal Doppler frequency-shift measurement model.
    scarabaeus.DiffOneWayRangeIdeal : Ideal differential one-way range measurement model.
    scarabaeus.CentroidingIdeal : Ideal optical centroiding measurement model.
    scarabaeus.DopplerReal : Real DSN Doppler measurement model with RTLT corrections.
    scarabaeus.SequentialRangingReal : Real DSN sequential ranging measurement model.
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(self, name: str, instrument: Instrument):
        self._name = name
        self._instrument = instrument

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def name(self) -> str:
        """
        The name of the model.
        """
        return self._name

    @property
    def instrument(self) -> Instrument:
        """
        The instrument.
        """
        return self._instrument

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#

    def update_reference_state(self, state_vector: StateArray):
        """
        Call this once per iteration (before generate_measurement_dataset).
        The model will pull:
          - meas_bias_ideal_* (as ArrayWUnits)
          - gs_delta_location_ECEF_* (as ArrayWFrame)
        for this instrument (matching spice_id).
        """
        self._reference_state_vector = state_vector
        # Expose the full state layout so _partials() can build an H column for
        # every estimated parameter (e.g. gm, Cnm, pole) beyond pos/vel.
        if state_vector is not None and self._sequence_definition is None:
            self._state_definition = state_vector.state

    def _extract_gs_delta_location(self) -> ArrayWFrame | None:
        """
        Return GS delta location for THIS station, as stored in the state, if present.
        """
        if self._reference_state_vector is None:
            return None

        state_allowed = StateDefinition._generate_allowed_state_dictionary()

        for state_entry in self._reference_state_vector.state:
            # (state_type, size, estimate_flag, component_type, body, value)
            component = state_entry[0]
            body = state_entry[4]
            value = state_entry[5]  # ArrayWFrame

            m = state_allowed["gs_delta_location_ECEF"][2].match(component)
            if m is None:
                continue

            body_sid = getattr(body, "spice_id", None)
            if body_sid == self.instrument.spice_id:
                return value  # ArrayWFrame (units + frame included)

        return None

    def _compute_visibility_mask(
        self,
        target: Spacecraft,
        epoch_array: EpochArray,
        elevation_mask: float = 10.0,
    ):
        """
        Returns a boolean array marking visible epochs, or ``None`` if no check applies.

        The base implementation duck-types on the instrument:

        * ``GroundStation`` (has ``station_visibility``) — elevation-angle check against
          ``elevation_mask``.
        * ``Camera`` (has ``target_in_fov``) — pinhole-projection FOV check; ``elevation_mask``
          is not used.

        Subclasses may override this method if they need custom visibility logic.
        """
        if hasattr(self.instrument, "station_visibility"):
            return self.instrument.station_visibility(
                target, epoch_array, elevation_mask=elevation_mask
            )
        if hasattr(self.instrument, "target_in_fov"):
            return self.instrument.target_in_fov(target, epoch_array)
        return None

    def write_observed_measurements(
        self,
        target: Spacecraft,
        epoch_array: EpochArray = None,
        epoch_start: EpochArray = None,
        epoch_end: EpochArray = None,
        tstep: float = 1,
        frame: Frame = J2000,
        noisy: bool = False,
        file_name: str = "ideal_measurement",
        check_visibility: bool = False,
        elevation_mask: float = 10.0,
        folder_path_override: str = None,
    ) -> None:
        """
        Generates synthetic measurements and write them as a .json file. The input of this method
        encapsulate the ones needed for the "computed_meas" method in each measurement model class.

        Parameters
        ----------
        target : Spacecraft
            The target spacecraft for which the range measurement is to be computed.

        epoch_array : EpochArray, optional
            An array of epochs (times) at which the range measurements should be computed. If
            provided, overrides ``epoch_start``, ``epoch_end``, and ``tstep``.

        epoch_start : EpochArray, optional
            The starting epoch for the range measurement computations. Required if ``epoch_array``
            is not provided.

        epoch_end : EpochArray, optional
            The ending epoch for the range measurement computations. Required if ``epoch_array``
            is not provided.

        tstep : float, optional
            The time step, in seconds, between consecutive range measurements. If `epoch_array`
            is not provided. Defaults to ``1`` second.

        frame : Frame , optional
            The reference frame in which the range computation is performed. Defaults to ``None``.

        noisy : bool , optional
            Whether to add noise to the computed range measurement. Defaults to ``False``.

        file_name : str, optional
            The filename of the JSON in which the measurement is saved, Defaults to ``'ideal_measurement'``.

        check_visibility : bool, optional
            When ``True``, epochs where the instrument cannot observe the target are removed
            before generating measurements. The specific check is instrument-dependent:
            ``GroundStation`` uses an elevation-angle test; ``Camera`` uses a FOV projection
            test. Defaults to ``False``.

        elevation_mask : float, optional
            Minimum elevation angle in degrees used by the ``GroundStation`` visibility check.
            Ignored for ``Camera`` instruments. Only relevant when ``check_visibility=True``.
            Defaults to ``10.0``.

        folder_path_override : str, optional
            If provided, the JSON file is saved directly to this path instead of the default
            ``data/measurements/radiometric`` or ``data/measurements/optical`` folder. The
            directory is created automatically if it does not exist. Defaults to ``None``.

        Returns
        -------
        None
        """
        ## Apply instrument-appropriate visibility mask
        if check_visibility:
            mask = self._compute_visibility_mask(target, epoch_array, elevation_mask)
            if mask is not None:
                epoch_array = epoch_array[mask]
                if epoch_array.size == 0:
                    raise RuntimeError(
                        "No epochs remain after applying visibility mask. "
                        "The target is never visible from this instrument."
                    )

        ## Generate the measurement using the computed model
        t1, t2, t3, meas = self.computed_measurements(
            target,
            epoch_array,
            epoch_start,
            epoch_end,
            tstep,
            frame,
            noisy,
        )

        ## Create an obs dictionary
        n_meas = epoch_array.size
        # Convert from EpochArray to Y-DOY-SEC format
        if epoch_array.system == "TDB":
            year, doy, sec = SpiceManager.et2YDS(epoch_array.times.values)

        # Generate measurement dictionary
        dict_meas = {
            "year": np.array(year).tolist(),
            "doy": np.array(doy).tolist(),
            "sec": np.array(sec).tolist(),
            "meas": np.array(meas.quantity.values).tolist(),
            "spice_id": np.array(self.instrument.spice_id).tolist(),
            "outlier_flag": (np.zeros((1, n_meas))[0].tolist()),
        }

        # Generate DB dictionary
        dict_DB = {
            "name": file_name,
            "meas_ideal": dict_meas,  # Ideal measurement
        }

        ## Determine save path
        if folder_path_override is not None:
            save_dir = folder_path_override
        else:
            optical_keywords = ["centroiding", "landmarks", "opnav", "optical", "beacon"]
            save_folder = (
                "measurements/optical"
                if any(kw in file_name.lower() for kw in optical_keywords)
                else "measurements/radiometric"
            )
            save_dir = os.path.join("data", save_folder)

        ## Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)

        ## Save dictionary as json file
        with open(os.path.join(save_dir, file_name + ".json"), "w") as json_file:
            json.dump(dict_DB, json_file, indent=4)

    def observed_measurements(
        self,
        file_name,
        meas_name: str = "meas_ideal",
        units: Units = spc_ad_units,
        frame: Frame = J2000,
    ) -> Tuple[EpochArray, np.ndarray, ArrayWFrame]:
        """
        Reads measurements from a .json file.

        Parameters
        ----------
        file_name : str
            The filename of the .json file containig the measurement information.

        meas_name : str, optional
            The name of the measurement data to access from the dictionary.
            Defaults to ``'meas_ideal'``.

        units : Units, optional
            Units to be used to write the output AWU. Defaults to ``unitless``.

        frame : Frame, optional
            Frame to be used to write the output AWF. Defaults to a J2000
            Frame object.

        Returns
        -------
        meas_time_et, meas_sec, meas_obs : Tuple[EpochArray, numpy.ndarray, ArrayWFrame]
            A tuple with the following values corresponding to
            their respective indices:

            * ``[0]`` = meas_time_et : EpochArray
                The time in ephemeris time.
            * ``[1]`` = meas_Sec : numpy.ndarray
                The times in seconds.
            * ``[2]`` = meas_obs : ArrayWFrame
                An AWF with the quantities in AWU.
            * ``[3]`` = meas_outliers : numpy.ndarray
                The np.array of measurements outliers

        Notes
        -----
        The writing of the json assumes or requires units and frames.
        """
        ## Open the measurement .json
        with open(os.path.join(file_name), "r") as json_file:
            trk_data = json.load(json_file)

        # Get the indices of non-outlier measurements
        idx_meas = np.array(
            [i for i, x in enumerate(trk_data[meas_name]["outlier_flag"]) if x == 0]
        )

        # Convert the YDS format into ET format
        meas_time_et = []
        for idx in idx_meas:
            meas_time_et.append(
                SpiceManager.YDS2et(
                    [trk_data[meas_name]["year"][idx]],
                    [trk_data[meas_name]["doy"][idx]],
                    [trk_data[meas_name]["sec"][idx]],
                )[0]
            )

        # Prepare output for list saving: time(et), obs, time(sec)
        meas_obs = np.array(trk_data[meas_name]["meas"])[idx_meas]
        meas_sec = np.array(trk_data[meas_name]["sec"])[idx_meas]
        meas_outliers = np.array(trk_data[meas_name]["outlier_flag"])[idx_meas]

        # Transform the output in EpochArray and AWF
        meas_time_et = EpochArray(
            ArrayWUnits(np.array(meas_time_et), spc_time_units).values, sys="TDB"
        )
        if units is None:
            meas_obs = ArrayWFrame(ArrayWUnits(meas_obs, None), frame)
        else:
            meas_obs = ArrayWFrame(meas_obs, units, frame)

        return meas_time_et, meas_sec, meas_obs, meas_outliers

    def compute_partials(
        self,
        target: Spacecraft,
        epoch_array: EpochArray,
        frame: Frame = J2000,
    ) -> list:
        """
        Stacks together measurement _partials for an epoch array at different epochs.

        Parameters
        ----------
        target : Spacecraft
            The target spacecraft.

        epoch_array : EpochArray
            The epochs.

        frame : Frame, optional
            The reference frame. Defaults to a J2000 Frame object.

        Returns
        -------
        _partials : list
            A list with all the _partials evaluated at different epochs in the ``epoch_array``.
        """
        # Generate _partials
        _partials = []
        print("")
        print("Generating _partials computed measurements...")
        update, close = Measurement._progress_monitor(
            num_total=epoch_array.size, desc="Partials"
        )
        if self._sequence_definition is not None:
            for i, epoch in enumerate(epoch_array):
                # Find the corresponding epoch array and update the state definition
                for idx_epochs_vec, epoch_array_leg in enumerate(
                    self._sequence_definition.epochs_vec
                ):
                    epoch_time_value = epoch.times.values
                    t_vals = epoch_array_leg.times.values
                    # Nodes have scalar epoch arrays — skip them
                    if (
                        not isinstance(t_vals, (list, np.ndarray))
                        or np.asarray(t_vals).ndim == 0
                    ):
                        continue
                    t_start = t_vals[0]
                    t_end = t_vals[-1]
                    if epoch_time_value == t_end and idx_epochs_vec != (
                        len(self._sequence_definition.epochs_vec) - 1
                    ):
                        # Special case for the last time in the array
                        self._state_definition = self._sequence_definition.models[
                            idx_epochs_vec + 2
                        ].state_definition
                        break
                    elif t_start <= epoch_time_value <= t_end:
                        # Update state definition for the found epoch
                        self._state_definition = self._sequence_definition.models[
                            idx_epochs_vec
                        ].state_definition
                        break  # Exit once the correct epoch array is found

                # Compute measurement _partials after updating the state definition
                _partials.append(self._partials(target, epoch, frame))
                update(i)

            close()
        else:
            for i, epoch in enumerate(epoch_array):
                _partials.append(self._partials(target, epoch, frame))
                update(i)
            close()
        return _partials

    def residuals(
        self, observed_meas: ArrayWFrame, computed_meas: ArrayWFrame
    ) -> ArrayWFrame:
        """
        Generates the measurement model's residuals given observed and computed ArrayWFrames.

        Parameters
        ----------
        observed_meas : ArrayWFrame
            The observed measurements values (O).

        computed_meas : ArrayWFrame
            The computed measurements values (C).

        Returns
        -------
        residuals : ArrayWFrame
            AWF with the residual O-C.
        """
        # Check if the observed and computed measurements share the same reference frame
        if observed_meas.frame.name != computed_meas.frame.name:
            raise ValueError(
                "Observed and Computed measurements don't share the same reference frames."
            )
        if (
            observed_meas.quantity.units.dimensions
            != computed_meas.quantity.units.dimensions
        ):
            bad_unts_err = (
                "Observed and Computed measurements do not share units. "
                f"Observed units: {observed_meas.quantity.units.name}. "
                f"Computed units: {computed_meas.quantity.units.name}."
            )
            raise ValueError(bad_unts_err)

        # Compute residuals
        observed_val = observed_meas.quantity.values
        computed_val = computed_meas.quantity.values
        common_units = observed_meas.quantity.units
        common_frame = observed_meas.frame
        residuals = []
        for i, (o, c) in enumerate(zip(observed_val, computed_val, strict=True)):
            residuals.append(o - c)
        residuals_AWF = ArrayWFrame(np.array(residuals), common_units, common_frame)

        return residuals_AWF

    def generate_measurement_dataset(
        self,
        dataset_name: str,
        target: Body,
        observed_meas: tuple | list | None = None,
        epochs: EpochArray | None = None,
        frame: Frame = J2000,
        noisy: bool = False,
    ) -> list[MeasurementDataSet]:
        """
        Generates a MeasurementDataSet object that can be used by filters downstream.

        Parameters
        ----------
        dataset_name : str
            The name of the MeasurementDataSet.

        target : Spacecraft
            The target spacecraft.

        epoch_list : EpochArray, optional
            The epochs. Defaults to ``None``.

        epoch_start : EpochArray, optional
            The starting epoch. Defaults to ``None``.

        epoch_end : EpochArray, optional
            The end epoch. Defaults to ``None``.

        tstep : int, optional
            The integration timestep. Defaults to ``1``.

        observed_measurements : list, optional
            The observed measurements. Defaults to ``None``.

        frame : Frame, optional
            The reference frame. Defaults to a J2000 Frame object.

        noisy : bool, optional
            Indicates if noise is added to the measurements or not. Defaults to ``False``.

        Returns
        -------
        mds_list : list[MeasurementDataSet]
            A list of MeasurementDataSet objects representing the measurements with their key properties to be used by a filter.

        Raises
        ------
        If the observed_meas list is only made by 3 elements, it throws an error because it needs the 4th element in the list for the indices of the outlier_flag

        Notes
        -----
        The MeasurementDataSet output is generated in 6 steps:

        #. Computed measurements
        #. Partials
        #. Residuals
        #. Sigmas
        #. Outlier flag
        #. Pack everything in a list
        #. Pack the list in a MeasurementDataSet object
        """

        # ----------------------------
        # Select epochs source
        # ----------------------------
        if observed_meas is None and epochs is None:
            raise ValueError("Provide exactly one of: observed_meas OR epochs.")

        if observed_meas is not None and epochs is not None:
            raise ValueError("Provide only one of: observed_meas OR epochs (not both).")

        # observed_meas convention: (epoch_array, meas_sec, meas_obs, meas_outliers)
        if observed_meas is not None:
            if len(observed_meas) < 4:
                raise ValueError(
                    f"Expected observed_meas with 4 entries (t_et, sec, obs, outlier_flag), got {len(observed_meas)}."
                )
            epoch_array = observed_meas[0]
            outlier_flag = np.array(observed_meas[3])
        else:
            epoch_array = epochs
            outlier_flag = np.zeros(epoch_array.size)  # no outliers by definition

        # ----------------------------
        # 1) Computed measurements
        # ----------------------------
        print()
        print("=" * 80)
        print(
            f'Generating computed measurements for the dataset "{dataset_name}"'.center(
                80
            )
        )
        print("=" * 80)

        t1, t2, t3, _meas = self.computed_measurements(
            target=target,
            epoch_array=epoch_array,
            frame=frame,
            noisy=noisy,
        )

        # number of components per epoch
        if len(_meas.quantity.shape) == 2:
            n_components = _meas.quantity.shape[1]
        else:
            n_components = 1

        # ----------------------------
        # 2) Partials
        # ----------------------------
        print("")
        print("=" * 80)
        print(f'Generating _partials for the dataset "{dataset_name}"'.center(80))
        print("=" * 80)

        _measurements_partials = self.compute_partials(
            target=target, epoch_array=epoch_array
        )
        partials_list = [np.asarray(p) for p in _measurements_partials]

        # ----------------------------
        # 4) Sigmas
        # ----------------------------
        _sigmas = np.array([self.sigma.values for _ in range(epoch_array.size)])

        # ----------------------------
        # 3) Observed + residuals
        # ----------------------------
        if observed_meas is not None:
            obs_awf = observed_meas[2]
            _residuals = self.residuals(observed_meas=obs_awf, computed_meas=_meas)

            obs_vals = obs_awf.quantity.values
            res_vals = _residuals.quantity.values.copy()
        else:
            # epochs-only mode: fabricate observed/residuals arrays with correct shape
            if n_components == 1:
                obs_vals = np.zeros(epoch_array.size)
                res_vals = np.zeros(epoch_array.size)
            else:
                obs_vals = np.zeros((epoch_array.size, n_components))
                res_vals = np.zeros((epoch_array.size, n_components))

        _outlier_flag = np.array(outlier_flag)

        # ----------------------------
        # 6) Names
        # ----------------------------
        names = [
            "t1",
            "t2",
            "t3",
            "computed",
            "observed",
            "residuals",
            "partials",
            "sigma",
            "outlier_flag",
        ]

        # ----------------------------
        # 7) Outlier gating
        # In epochs-only mode threshold/bias should stay None -> no gating
        # ----------------------------
        threshold = None
        bias = None

        # if you ever enable it, guard it:
        # if observed_meas is None:
        #     threshold = None
        #     bias = None

        if threshold is None or bias is None:
            residuals_w_outliers = res_vals
            partials_w_outliers = partials_list
        else:
            gate = threshold + bias
            idx_outliers = np.where(np.abs(res_vals) > gate)[0]
            res_vals[idx_outliers] = 0.0
            residuals_w_outliers = res_vals
            partials_w_outliers = [
                np.zeros_like(p) if i in idx_outliers else p
                for i, p in enumerate(partials_list)
            ]

        # ----------------------------
        # Pack output
        # ----------------------------
        dataset_list_output = []

        if n_components == 1:
            # robust extraction of time arrays
            try:
                t1_vals = t1[:].times.values
            except (TypeError, AttributeError):
                t1_vals = t1[:]
            try:
                t2_vals = t2[:].times.values
            except (TypeError, AttributeError):
                t2_vals = t2[:]
            try:
                t3_vals = t3[:].times.values
            except (TypeError, AttributeError):
                t3_vals = t3[:]

            values = [
                t1_vals,
                t2_vals,
                t3_vals,
                _meas.quantity[:].values,
                obs_vals,
                residuals_w_outliers,
                partials_w_outliers,
                _sigmas[:],
                _outlier_flag[:],
            ]

            _data_set = MeasurementDataSet(
                set_name=dataset_name,
                measurements=values,
                names=names,
                instrument=self.instrument,
                target=target,
                epochs_t3=epoch_array,
                measurement_type=self._measurement_type,
            )
            return _data_set

        elif n_components == 2:
            # For 2-components keep your split datasets, but use obs_vals/res_vals that may be zeros
            values_1 = [
                t1[:].times.values,
                t2[:].times.values,
                t3[:].times.values,
                _meas.quantity[:, 0].values,
                obs_vals[:, 0],
                (
                    residuals_w_outliers[:, 0]
                    if residuals_w_outliers.ndim == 2
                    else residuals_w_outliers
                ),
                [element[0] for element in partials_w_outliers],
                _sigmas[:, 0] if _sigmas.ndim == 2 else _sigmas,
                _outlier_flag[:],
            ]
            values_2 = [
                t1[:].times.values,
                t2[:].times.values,
                t3[:].times.values,
                _meas.quantity[:, 1].values,
                obs_vals[:, 1],
                (
                    residuals_w_outliers[:, 1]
                    if residuals_w_outliers.ndim == 2
                    else residuals_w_outliers
                ),
                [element[1] for element in partials_w_outliers],
                _sigmas[:, 1] if _sigmas.ndim == 2 else _sigmas,
                _outlier_flag[:],
            ]

            _data_set_1 = MeasurementDataSet(
                set_name="Opnav_u",
                measurements=values_1,
                names=names,
                instrument=self.instrument,
                target=target,
                epochs_t3=epoch_array,
                measurement_type=self._measurement_type + " u",
            )
            _data_set_2 = MeasurementDataSet(
                set_name="Opnav_v",
                measurements=values_2,
                names=names,
                instrument=self.instrument,
                target=target,
                epochs_t3=epoch_array,
                measurement_type=self._measurement_type + " v",
            )
            return [_data_set_1, _data_set_2]

        else:
            raise ValueError(
                "The number of components (>2) is not supported in current SCB version."
            )

    def _build_perturbation_vector(self, eps_override: dict) -> ArrayWUnits:
        """
        Constructs a 6-element perturbation vector [dx, dy, dz, dvx, dvy, dvz]
        as an ArrayWUnits from the given eps_override dictionary.

        Parameters
        ----------
        eps_override : dict
            Dictionary with keys like 'dx', 'dvy', etc., each mapped to an ArrayWUnits.

        Returns
        -------
        delta : ArrayWUnits
            1x6 perturbation vector with units.
        """
        component_labels = ["dx", "dy", "dz", "dvx", "dvy", "dvz"]
        values = np.zeros(6)
        units = [None] * 6

        if eps_override is not None:
            for i, label in enumerate(component_labels):
                if label in eps_override:
                    eps = eps_override[label]
                    values[i] = eps.values
                    units[i] = eps.units

        # Fill missing units with defaults to ensure consistency
        km, sec = Units.get_units(["km", "sec"])
        default_units = [km, km, km, km / sec, km / sec, km / sec]
        units = [u if u is not None else default_units[i] for i, u in enumerate(units)]

        return ArrayWUnits(values.reshape(1, 6), units)

    @staticmethod
    def _compute_CN_lt(
        receiver_pos: np.ndarray,
        transmitter_id: str,
        t_rcv: np.ndarray,
        delta: np.ndarray,
        way: str = "up",
    ) -> np.ndarray:
        """
        Compute converged Newtonian one-way light time between a receiver and a transmiter at reception epoch t_rcv

        Parameters
        ----------
        receiver_pos : np.ndarray
            Position of the receiving body (e.g., ground station (t3) or spacecraft (t2)).

        transmitter_id : str
            Name of the emitting body (e.g., ground station (t2) or spacecraft(t3)).

        t_rcv : np.ndarray
            Epoch(s) at which the signal is received [ET seconds past J2000].

        delta : np.ndarray
            Perturbation vector for finite differencing [km, km/s].

        way: str
            Signal path, can be "up" or "down"

        Returns
        -------
        light_time : np.ndarray
            One-way geometric light time [s].

        Note
        -------
        All computations are internally defaulted to be executed in J2000
        """
        # Define constants to use in this method
        c = constants.c.values  # Speed of light in vacuum [km/sec]

        # The receiver is kept fixed (the np.ndarray of its position is passed), while the position of the transmitter is changed (back in time) to find the transmittions time and interval between transmission and reception.

        # Initial guess
        transmitter_pos0 = spice.spkezr(transmitter_id, t_rcv, "J2000", "NONE", "SSB")[
            0
        ]
        range0 = np.linalg.norm(transmitter_pos0[:3] - receiver_pos)
        tof_0 = range0 / c

        # Define the residual function
        def _tof_residual(tof):
            transmitter_pos_it = (
                spice.spkezr(transmitter_id, t_rcv - tof, "J2000", "NONE", "SSB")[0][:3]
                + delta[:3]
            )
            range_ = np.linalg.norm(transmitter_pos_it[:3] - receiver_pos + delta[:3])
            return (range_ / c) - tof

        # Solve using a root finder
        tof_final = root_scalar(
            _tof_residual, method="brentq", bracket=[0, 2 * tof_0], xtol=1e-15
        )
        if not tof_final.converged:
            raise RuntimeError("Light-time correction root-finding did not converge.")

        return tof_final.root

    @staticmethod
    def _compute_RLT_SUN_lt(r1_r2: float, r2_r3: float, r12_r23: float) -> float:
        """
        Compute Relativistic lightime delay due to the Sun. This effects takes into account the following General Relativity (GR) effects:
        1) Curvature of the lightpath due to gravity
        2) Reduction in the coordinate of the speed of light
        The value computed in this method is equal to the second term of [Moyer 2000], eq.8-55

        Parameters
        ----------
        r1_r2 : float
            Position vector. For way == 'dwn', r1_r2 is the norm of the position vector from the Sun to the spacecraft at reflection time t2.
            For way == 'up', r1_r2 is the norm of the position vector from the Sun to the ground-station at reception time t3.

        r2_r3 : float
            Position vector. For way == 'dwn', r2_r3 is the norm of the position vector from the Sun to the ground-station at reception time t3.
            For way == 'up', r2_r3 is the norm of the position vector from the Sun to the spacecraft at reflection time t2.

        r12_r23 : float
            Norm of the position vector difference between r2 and r1 or r3 and r2. See Moyer 2000, eq.8-57 and eq.8-58

        Returns
        -------
        dt_GR : float
            One-Way GR lightime delay [s]
        """
        # Define constants to use in this method
        c = constants.c.values  # Speed of light in vacuum [km/sec]
        mu_S = constants.SUN.GM.values  # GM of the Sun
        gamma = constants.gamma_GR.values  # Gamma value for GR

        # Compute different terms in Moyer 2000, eq.8-55
        sun_term = (1 + gamma) * mu_S / c**2
        numerator = r1_r2 + r2_r3 + r12_r23 + sun_term
        denominator = r1_r2 + r2_r3 - r12_r23 + sun_term

        # Generate GR lightime delay
        dt_GR = ((1 + gamma) * mu_S / c**3) * np.log(numerator / denominator)

        return dt_GR

    @staticmethod
    def _get_bodies_for_RLT(tx_h: float, tx_l: float, ssb_to_gs_tx_RLTframe: np.ndarray):
        """
        Generate the quantities of interest of the partecipating bodies for the GR delays

        Parameters
        ----------
        tx_h : float
            Integer part of the time tx at which to compute the geometric vectors

        tx_l : float
            Fractional part of the time tx at which to compute the geometric vectors

        ssb_to_gs_tx_RLTframe : np.ndarray
            State vector (pos,vel) from SSB to GS at tx in the SSB space-time relativistic frame of reference

        Returns
        -------
        earth_to_gs_tx : np.ndarray
            state vector (pos,vel) from Earth to the GS at tx in J2000

        moon_to_gs_tx : np.ndarray
            state vector (pos,vel) from the Moon to the GS at tx in J2000

        mars_to_gs_tx : np.ndarray
            state vector (pos,vel) from Mars to the GS at tx in J2000

        jupiter_to_gs_tx : np.ndarray
            state vector (pos,vel) from Jupiter to the GS at tx in J2000

        saturn_to_gs_tx : np.ndarray
            state vector (pos,vel) from Saturn to the GS at tx in J2000
        """
        ssb_to_earth = SpiceManager.get_state_precise(
            "399", tx_h, "J2000", "SSB", "None", tx_l
        ).values
        ssb_to_moon = SpiceManager.get_state_precise(
            "301", tx_h, "J2000", "SSB", "None", tx_l
        ).values
        ssb_to_mars = SpiceManager.get_state_precise(
            "499", tx_h, "J2000", "SSB", "None", tx_l
        ).values
        ssb_to_jupiter = SpiceManager.get_state_precise(
            "599", tx_h, "J2000", "SSB", "None", tx_l
        ).values
        ssb_to_saturn = SpiceManager.get_state_precise(
            "699", tx_h, "J2000", "SSB", "None", tx_l
        ).values

        earth_to_gs_tx = ssb_to_gs_tx_RLTframe - ssb_to_earth
        moon_to_gs_tx = ssb_to_gs_tx_RLTframe - ssb_to_moon
        mars_to_gs_tx = ssb_to_gs_tx_RLTframe - ssb_to_mars
        jupiter_to_gs_tx = ssb_to_gs_tx_RLTframe - ssb_to_jupiter
        saturn_to_gs_tx = ssb_to_gs_tx_RLTframe - ssb_to_saturn

        return (
            earth_to_gs_tx,
            moon_to_gs_tx,
            mars_to_gs_tx,
            jupiter_to_gs_tx,
            saturn_to_gs_tx,
        )

    @staticmethod
    def _compute_RLT_BODIES_lt(
        t2_t1: float,
        t3_t2: float,
        body_id,
        ssb_to_gs_t_31_RLT: np.ndarray,
        way: str = None,
        delta: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute Relativistic lightime delay due to different bodies.
        The value computed in this method is equal to the third term of [Moyer 2000], eq.8-55.
        The second term corresponds to the one computed with "_compute_RLT_SUN_lt".
        The first term is the Newtonian component, that can be computed with "_compute_CN_lt".

        Parameters
        ----------
        t2_t1 : float
            Reflection time t2 (for way == 'dwn') or transmittion time t1 (for way == 'up')

        t3_t2 : float
            Reception time t3 (for way == 'dwn') or reflection time t2 (for way == 'up')

        ssb_to_gs_t_31_RLT : np.ndarray
            State vector (pos,vel) from SSB to GS at t3 (for way == 'dwn') or t1 (for way == 'up') in a SSB space-time relativistic frame of reference

        ssb_to_gs_tx_RLTframe : np.ndarray
            State vector (pos,vel) from SSB to GS at tx in SSB space-time relativistic frame of reference

        way : str
            Signal path, can be "up" for 1-2 or "dwn" for 2-3

        delta : np.ndarray
            Perturbation vector for finite differencing [km, km/s]. This is used to compute the GR delay due to bodies other than the Sun.

        Returns
        -------
        dt_GR_bodies : float
            GR lightime delay due to bodies other than the Sun, as per Moyer 2000, eq.8-55

        """
        # Define constants
        gamma = constants.gamma_GR.values  # Gamma for GR
        c = constants.c.values  # Speed of light in vacuum [km/sec]
        mu_earth = constants.EARTH.GM.values  # GM of Earth
        mu_moon = constants.MOON.GM.values  # GM of The Moon
        mu_mars = constants.MARS.GM.values  # GM of Mars
        mu_jupiter = constants.JUPITER.GM.values  # GM of Jupiter
        mu_saturn = constants.SATURN.GM.values  # GM of Saturn

        def _body_delay_term(mu_b, r1_vec, r2_vec):
            r1 = np.linalg.norm(r1_vec[:3])
            r2 = np.linalg.norm(r2_vec[:3])
            r12 = np.linalg.norm(r2_vec[:3] - r1_vec[:3])
            numerator = r1 + r2 + r12
            denominator = r1 + r2 - r12
            return ((1 + gamma) * mu_b / c**3) * np.log(numerator / denominator)

        dt_GR_bodies = 0
        if way == "dwn":
            t2_h, t2_l = EpochArray._split_fractional_part(t2_t1)
            t3_h, t3_l = EpochArray._split_fractional_part(t3_t2)

            # Get body-to-spacecraft vectors at t2
            earth_to_sc_t2 = (
                SpiceManager.get_state_precise(
                    body_id, t2_h, "J2000", "399", "None", t2_l
                ).values
                + delta
            )
            moon_to_sc_t2 = (
                SpiceManager.get_state_precise(
                    body_id, t2_h, "J2000", "301", "None", t2_l
                ).values
                + delta
            )
            mars_to_sc_t2 = (
                SpiceManager.get_state_precise(
                    body_id, t2_h, "J2000", "499", "None", t2_l
                ).values
                + delta
            )
            jupiter_to_sc_t2 = (
                SpiceManager.get_state_precise(
                    body_id, t2_h, "J2000", "599", "None", t2_l
                ).values
                + delta
            )
            saturn_to_sc_t2 = (
                SpiceManager.get_state_precise(
                    body_id, t2_h, "J2000", "699", "None", t2_l
                ).values
                + delta
            )

            # Get body-to-gs vectors at t3
            (
                earth_to_gs_t3,
                moon_to_gs_t3,
                mars_to_gs_t3,
                jupiter_to_gs_t3,
                saturn_to_gs_t3,
            ) = Measurement._get_bodies_for_RLT(t3_h, t3_l, ssb_to_gs_t_31_RLT)

            # Compute GR delay from each body
            earth_RLT_dt = _body_delay_term(mu_earth, earth_to_sc_t2, earth_to_gs_t3)
            moon_RLT_dt = _body_delay_term(mu_moon, moon_to_sc_t2, moon_to_gs_t3)
            mars_RLT_dt = _body_delay_term(mu_mars, mars_to_sc_t2, mars_to_gs_t3)
            jupiter_RLT_dt = _body_delay_term(
                mu_jupiter, jupiter_to_sc_t2, jupiter_to_gs_t3
            )
            saturn_RLT_dt = _body_delay_term(mu_saturn, saturn_to_sc_t2, saturn_to_gs_t3)

        elif way == "up":
            t1_h, t1_l = EpochArray._split_fractional_part(t2_t1)
            t2_h, t2_l = EpochArray._split_fractional_part(t3_t2)

            # Get body-to-spacecraft vectors at t2
            earth_to_sc_t2 = (
                SpiceManager.get_state_precise(
                    body_id, t2_h, "J2000", "399", "None", t2_l
                ).values
                + delta
            )
            moon_to_sc_t2 = (
                SpiceManager.get_state_precise(
                    body_id, t2_h, "J2000", "301", "None", t2_l
                ).values
                + delta
            )
            mars_to_sc_t2 = (
                SpiceManager.get_state_precise(
                    body_id, t2_h, "J2000", "499", "None", t2_l
                ).values
                + delta
            )
            jupiter_to_sc_t2 = (
                SpiceManager.get_state_precise(
                    body_id, t2_h, "J2000", "599", "None", t2_l
                ).values
                + delta
            )
            saturn_to_sc_t2 = (
                SpiceManager.get_state_precise(
                    body_id, t2_h, "J2000", "699", "None", t2_l
                ).values
                + delta
            )

            # Get body-to-gs vectors at t1
            (
                earth_to_gs_t1,
                moon_to_gs_t1,
                mars_to_gs_t1,
                jupiter_to_gs_t1,
                saturn_to_gs_t1,
            ) = Measurement._get_bodies_for_RLT(t1_h, t1_l, ssb_to_gs_t_31_RLT)

            # Compute GR delay from each body
            earth_RLT_dt = _body_delay_term(mu_earth, earth_to_sc_t2, earth_to_gs_t1)
            moon_RLT_dt = _body_delay_term(mu_moon, moon_to_sc_t2, moon_to_gs_t1)
            mars_RLT_dt = _body_delay_term(mu_mars, mars_to_sc_t2, mars_to_gs_t1)
            jupiter_RLT_dt = _body_delay_term(
                mu_jupiter, jupiter_to_sc_t2, jupiter_to_gs_t1
            )
            saturn_RLT_dt = _body_delay_term(mu_saturn, saturn_to_sc_t2, saturn_to_gs_t1)

        # Stack delays toghether into a single value
        dt_GR_bodies = (
            earth_RLT_dt + moon_RLT_dt + mars_RLT_dt + jupiter_RLT_dt + saturn_RLT_dt
        )
        return dt_GR_bodies

    @staticmethod
    def _compute_RLT_lt(
        components: str = "RLT",
        r1_r2_vec: np.ndarray = None,
        r2_r3_vec: np.ndarray = None,
        t2_t1: float = None,
        t3_t2: float = None,
        sc_spice_id: str = None,
        ssb_to_gs_t_31_RLT: np.ndarray = None,
        way: str = None,
        delta: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute Relativistic lightime delay due to the Sun and to different bodies.
        The value computed in this method is equal to the second and third terms of [Moyer 2000], eq.8-55.
        The second term corresponds to the one computed with "_compute_RLT_SUN_lt".
        The third term corresponds to the one computed with "_compute_RLT_BODIES_lt".

        Parameters
        ----------
        components : str
            input to control whether 'SUN',

        r1_r2_vec : np.ndarray
            Position vector. For way == 'dwn', r1_r2_vec is the position vector from the body to the spacecraft at reflection time t2.
            For way == 'up', r1_r2_vec is the position vector from the body to the ground-station at reception time t3.

        r2_r3_vec : np.ndarray
            Position vector. For way == 'dwn', r2_r3_vec is the position vector from the body to the ground-station at reception time t3.
            For way == 'up', r2_r3_vec is the position vector from the body to the spacecraft at reflection time t2.

        t2_t1: float
            Reflection time t2 (for way == 'dwn') or transmission time t1 (for way == 'up')

        t3_t2: float
            Reception time t3 (for way == 'dwn') or reflection time t2 (for way == 'up')

        sc_spice_id: str
            Spice ID of the spacecraft

        ssb_to_gs_t_31_RLT : np.ndarray
            State vector (pos,vel) from SSB to GS at t3 (for way == 'dwn') or t1 (for way == 'up') in a SSB space-time relativistic frame of reference

        way : str
            Signal path, can be "up" for 1-2 or "dwn" for 2-3

        delta : np.ndarray
            Perturbation vector for finite differencing [km, km/s]. This is used to compute the GR delay due to bodies other than the Sun.

        Returns
        -------
        dt_GR_bodies : float
            GR lightime delay due to bodies and the Sun, as per Moyer 2000, eq.8-55

        r32_r21_vec: np.ndarray
            r32 vector (for way == 'dwn') or r21 vector (for way == 'up'). Used to determine convergence in the Moyer's 32-steps algorithm for precise RTLT.

        Note
        -------
        for way == 'dwn':
            r2_r1_vec = r2_vec = sun_to_sc_t2
            r3_r2_vec = r3_vec = sun_to_gs_t3
            r23_r12_vec = r32_vec = gs_to_sc (t3-t2), not defined in time

        for way == 'up':
            r2_r1_vec = r1_vec = sun_to_gs_t1
            r3_r2_vec = r2_vec = sun_to_sc_t2
            r23_r12_vec = r12_vec = sc_to_gs (t2-t1), not defined in time

        """
        # Sun-Relativistic (RLT) component of equation 8.55 (Part 2 of 3)
        if components == "RLT-SUN" or components == "RLT":
            r23_r12_vec = r2_r3_vec - r1_r2_vec
            r32_r21 = np.linalg.norm(r23_r12_vec[:3])
            r2_r1 = np.linalg.norm(r1_r2_vec[:3])
            r3_r2 = np.linalg.norm(r2_r3_vec[:3])
            # Compute RLT effect (downlink)
            dt_GR_SUN = Measurement._compute_RLT_SUN_lt(r2_r1, r3_r2, r32_r21)
            # Step (15 - part 1)
            if r32_r21 == 0:
                r32_r21 = np.finfo(float).eps
        else:
            dt_GR_SUN = 0
            r23_r12_vec = np.zeros(3)
            r32_r21 = np.finfo(float).eps

        # Body-Relativistic (RLT) component of equation 8.55 (Part 3 of 3)
        if components == "RLT-BODIES" or components == "RLT":
            dt_GR_BODIES = Measurement._compute_RLT_BODIES_lt(
                t2_t1, t3_t2, sc_spice_id, ssb_to_gs_t_31_RLT, way, delta
            )
        else:
            dt_GR_BODIES = 0

        # Stack relativistic components toghether
        if components == "RLT-SUN":
            dt_GR = dt_GR_SUN
        elif components == "RLT-BODIES":
            dt_GR = dt_GR_BODIES
        elif components == "RLT":
            dt_GR = dt_GR_SUN + dt_GR_BODIES
        else:
            dt_GR = 0

        return dt_GR, r23_r12_vec

    @staticmethod
    def _compute_delta_et_tai(t, gs_spice_id, method: int = 1) -> np.ndarray:
        """
        Method to compute the reltivistic time difference between Ephemeris Time (ET) and International Atomic Time (TAI) at a ground station on Earth.

        Parameters
        ----------
        t : float
            ephemeris time when to compute the correction term

        gs_spice_id : str
            spice id string of the ground station

        method : int
            Method to be used amongst 1 (simplified), 2 (Vector expression, with GS in J2000), 3 (Vector expression, with GS in J2000 SSB relativisti space-time reference)

        Returns
        -------
        dt : float
            time difference in seconds between ET and TAI at a given epoch

        """
        # Define constants to use in this method
        c = constants.c.values  # Speed of light in vacuum [km/sec]
        mu_S = constants.SUN.GM.values  # GM of the Sun [km^3/sec^2]
        mu_J = constants.JUPITER.GM.values  # GM of Jupiter [km^3/sec^2]
        mu_Sa = constants.SATURN.GM.values  # GM of Saturn [km^3/sec^2]

        if method == 1:
            # Simplified method (valid after 2000)
            M = 6.239996 + 1.99096871e-7 * t  # Mean anomaly [rad]
            E = M + 0.01671 * np.sin(M)  # Eccentric anomaly [rad]
            dt = 32.184 + 1.657e-3 * np.sin(E)  # ET - TAI [s]
        elif method == 2 or method == 3:  # Vector expression relativistic correction
            # X_i_j is state of point i relative to point j
            # C (SSB), S (SUN), B (EARTH-MOON Barycenter), E (EARTH)
            # M (MOON), J (JUPITER), Sa (SATURN), A (Atomic clock on Earth which reads TAI)
            X_B_S = spice.spkezr("3", t, "J2000", "None", "Sun")[
                0
            ]  # Earth-Moon Barycenter relative to Sun
            X_B_C = spice.spkezr("3", t, "J2000", "None", "SSB")[
                0
            ]  # Earth-Moon Barycenter relative to SSB
            X_E_B = spice.spkezr("399", t, "J2000", "None", "3")[
                0
            ]  # Earth relative to Earth-Moon Barycenter
            X_E_C = spice.spkezr("399", t, "J2000", "None", "SSB")[
                0
            ]  # Earth relative to SSB
            if method == 2:
                X_A_E = spice.spkezr(gs_spice_id, t, "J2000", "None", "399")[
                    0
                ]  # Earth to GS atomic clock
            elif method == 3:
                X_A_E = Measurement._compute_GS_in_SSB_space_time_RLT(
                    gs_spice_id, t
                )  # Earth to GS atomic clock, in SSB Space-Time relativistic frame
            X_J_S = spice.spkezr("599", t, "J2000", "None", "Sun")[
                0
            ]  # Jupiter relative to Sun
            X_Sa_S = spice.spkezr("699", t, "J2000", "None", "Sun")[
                0
            ]  # Saturn relative to Sun
            X_S_C = spice.spkezr("10", t, "J2000", "None", "SSB")[
                0
            ]  # Sun relative to SSB

            f1 = 32.184
            f2 = (2 / (c**2)) * np.dot(X_B_S[3:6], X_B_S[0:3])
            f3 = (1 / (c**2)) * np.dot(X_B_C[3:6], X_E_B[0:3])
            f4 = (1 / (c**2)) * np.dot(X_E_C[3:6], X_A_E[0:3])
            f5 = (1 / (c**2) * (mu_J / (mu_S + mu_J))) * np.dot(X_J_S[3:6], X_J_S[0:3])
            f6 = (1 / (c**2) * (mu_Sa / (mu_S + mu_Sa))) * np.dot(
                X_Sa_S[3:6], X_Sa_S[0:3]
            )
            f7 = (1 / (c**2)) * np.dot(X_S_C[3:6], X_B_S[0:3])
            dt = f1 + f2 + f3 + f4 + f5 + f6 + f7

        else:
            raise ValueError(
                "Invalid method selected. Choose 1 (Simplified), 2 (Vector, J2000), 3 (Vector, J2000 SSB GR-ST))."
            )

        return dt

    @staticmethod
    def _compute_precise_RTLT(
        sc_spice_id: str,
        gs_spice_id: str,
        tt_et: float,
        electronic_delays_dict: dict,
        solar_corona_dict: dict,
        delta_partials: ArrayWUnits,
        gs_delta_awf: ArrayWFrame | None,
    ) -> np.ndarray:
        """
        Method to compute the precise Round-Trip Light-Time (RTLT) between a gs-based DSN antenna and a spacecraft.
        This method computes the RTLT for a 2-way path between:
            - A DSN antenna transmitting a signal at t1
            - A spacecraft reflecting a signal at t2
            - A DSN antenna receiving a signal at t3

           implement the necessary time delays between transmittion from a gs at (t1) and reception at the same gs at (t3).

        Parameters
        ----------
        sc_spice_id : str
            spice id string of the spacecraft

        gs_spice_id : str
            spice id string of the ground station

        tt_et : float
            time-tag in ephemeris time of the reception time at the gs (t3)

        electronic_delays_dict : dict
            dictionary of electronic delays

        solar_corona_dict : dict
            dictionary of solar corona parameters

        delta: ArrayWUnits
            delta vector for position and velocity. Used to generate numeric _partials

        Returns
        -------
        rtlt : float
            Sum of the rtlt delays components as a single delay value in second between t1 and t3

        rtlt_parts : np.array(17,1)
            rtlt decomposed in its components, expressed in seconds

        gs_delta_awf: ArrayWFrame | None
            If not None, this is the delta vector for the ground station in the SSB space-time relativistic frame of reference.

        Notes
        -----
        See also the method SpiceManager.get_state_precise() which needs integer and fractional time for precise
        ephemeris retrivial

        [1]: "TRK-2-34, DSN Tracking System Data Archival Format", DSN No. 820-013, TRK-2-34, Rev. N

        """
        # Define constants to be used within this method
        c = constants.c.values  # Speed of light in vacuum [km/sec]
        # Define debug flag for CN method
        CN_flag = False  # Internal flag used to debug and use Converged or Un-Converged computation for Newtonian lightime

        ### Moyer 32-steps algorithm ###
        # For reference, see Moyer 2000, Sec. 8.3.6

        ### down-leg portion (PART 1 of 2): from t3 (reception at the ground-station) to t2 (transmission from spacecraft) ###

        # Step (1) - Transform t3(ST) to t3(TAI)
        # We go from ET to TAI because we enter this function not in sec but in time, se we need to convert back to ST
        tt_tai_i = spice.unitim(tt_et, "ET", "TAI") + (
            -electronic_delays_dict["rcv_tt_dly"]
            + electronic_delays_dict["trn_tt_dly"]
            - electronic_delays_dict["array_delay"]
        )  # As per [1], Notes 2,3, and 4
        # tt_tai_i = (spice.unitim(tt_et, "ET", "TAI") - electronic_delays_dict["rcv_tt_dly"]) # NOTE: For debug purposes, 2.0
        # tt_tai_i = spice.unitim(tt_et, 'ET', 'TAI') # NOTE: For debug purposes, 1.0

        # Step (2) - Transform t3(TAI) to t3(ET)
        tt_et, dt_et_tai_t3, r_stn_BC = Measurement._et_tai_debug(
            tt_tai_i, gs_spice_id
        )  # Method 1, DEBUG
        t3 = tt_et
        t3_h, t3_l = EpochArray._split_fractional_part(
            t3
        )  # Separate integer and fractional components of t3 for precise state retrivial

        # Step (3) - Skipped, done with Spice + RLT-corrected GS location

        # Step (4) - Compute quantities of interest for B's and S
        """
        Barycentric Frame Only: for each body B and the sun S (the ones considered in eq. 8-55 for RLT delays), 
        calculate vector and scalar distances from the body to the received at t3.
        The quantities "body_to_gs_t3" are computeed at t3 within the "compute_RTL_BODIES_lt" method.
        """
        ssb_to_earth_t3 = SpiceManager.get_state_precise(
            "Earth", t3_h, "J2000", "SSB", "None", t3_l
        ).values
        ssb_to_sun_t3 = SpiceManager.get_state_precise(
            "10", t3_h, "J2000", "SSB", "None", t3_l
        ).values
        ssb_to_gs_t3_RLTframe = (
            np.concatenate((r_stn_BC, np.zeros(3))) + ssb_to_earth_t3
        )
        sun_to_gs_t3 = ssb_to_gs_t3_RLTframe - ssb_to_sun_t3
        # tt_et = tt_et # Method 2, DEBUG
        # r_stn_BC = SpiceManager.get_state(gs_spice_id,tt_et,'J2000','Earth','None') # Method 2, DEBUG
        # r_stn_BC = r_stn_BC[0:3].values # Method 2, DEBUG

        # Step (5) - Skipped, computation performed in Barycentric Frame

        # Step (6) - setup IC for iteration loop on down-leg lightime
        lt_23 = 0  # prediction of the down-leg lightime for the first iteration
        t2 = t3 - lt_23  # first estimate of t2
        t2_h, t2_l = EpochArray._split_fractional_part(t2)

        # Steps (7, 8, 9, 10) - Skipped for free spacecraft, with 2W assumption (sranging or doppler), or because steps are substituted by Spice interpolation
        delta_t2 = 10
        max_it_counter = 0
        while (
            abs(delta_t2) > 1e-12 and max_it_counter < 10
        ):  # step(16) - Stay within the while loop if the absolute value of delta_t2 is less than 0.1s
            max_it_counter = max_it_counter + 1
            # Step (11) - get the spacecraft sstate in SSB at t2
            ssb_to_sc_t2 = (
                SpiceManager.get_state_precise(
                    sc_spice_id, t2_h, "J2000", "SSB", "None", t2_l
                ).values
                + delta_partials.values[0]
            )
            # Step (12) - Compute quantities of interest for B's and S. Same as step (4): quantities of interest fo the bodies are computed inside the "_compute_RLT_BODIES_lt" method
            sun_to_sc_t2 = (
                SpiceManager.get_state_precise(
                    sc_spice_id, t2_h, "J2000", "10", "None", t2_l
                ).values
                + delta_partials.values[0]
            )
            # Step (13) - Skipped, computation performed in Barycentric Frame and not geocentric

            # Step (14) - Generate r_23_vec, r_23_vec_dot and r_23 and compute RLT_23. This computation is split in two part and not performed originally with the 3 terms from Moyer 2000 eq. 8-55.
            # Converged newtonian (CN) component of Moyer 2000 eq. 8-55s (1st term)
            if CN_flag:
                # Note: this term is different than the one in Moyer 2000 eq. 8-55, becuase it's already internally converged.
                dt_CN_dwn = Measurement._compute_CN_lt(
                    ssb_to_gs_t3_RLTframe[:3],
                    sc_spice_id,
                    t3,
                    delta_partials.values[0],
                    "down",
                )
            else:
                dt_CN_dwn = np.linalg.norm(sun_to_sc_t2[:3] - sun_to_gs_t3[:3]) / c
                # Sun and Body Relativistic (RLT) components of Moyer 2000 eq. 8-55 (2nd and 3rd terms)
            dt_GR_dwn, r23_vec = Measurement._compute_RLT_lt(
                "RLT",
                sun_to_sc_t2,
                sun_to_gs_t3,
                t2,
                t3,
                sc_spice_id,
                ssb_to_gs_t3_RLTframe,
                "dwn",
                delta_partials.values[0],
            )
            r23 = np.linalg.norm(r23_vec[:3])
            # Get the velocity of the spacecraft at reflection time t2 in SSB space-time frame
            r_dot_2_C_t2 = ssb_to_sc_t2[3:6]
            # Compute p_dot from Moyer 2000 eq. 8-60
            p_dot_23 = np.dot(r23_vec[:3] / r23, r_dot_2_C_t2)
            # Step (15) - calculate the linear differential correction delta_t2, Moyer 2000 eq. 8-72
            delta_t2 = (t3 - t2 - dt_CN_dwn - dt_GR_dwn) / (
                1 - p_dot_23 / c
            )  # with i = 2, j = 3. Note: dt_CN_dwn is converged with respecto the un-converged term in 8-72
            t2 += delta_t2  # update t2
            t2_h, t2_l = EpochArray._split_fractional_part(
                t2
            )  # update t2 (integer and fractional parts)

        # Step (17) - Map everything calculated at the last estimate of the converged value for t2(ET). From Moyer, this is substituted by spice calls
        ssb_to_sc_t2 = (
            SpiceManager.get_state_precise(
                sc_spice_id, t2_h, "J2000", "SSB", "None", t2_l
            ).values
            + delta_partials.values[0]
        )
        sun_to_sc_t2 = (
            SpiceManager.get_state_precise(
                sc_spice_id, t2_h, "J2000", "10", "None", t2_l
            ).values
            + delta_partials.values[0]
        )

        # Step (18) - Using the mapping quantities from Step (17), repeat steps 11 to 14
        if CN_flag:
            dt_CN_dwn = Measurement._compute_CN_lt(
                ssb_to_gs_t3_RLTframe[:3],
                sc_spice_id,
                t3,
                delta_partials.values[0],
                "down",
            )
        else:
            dt_CN_dwn = np.linalg.norm(sun_to_sc_t2[:3] - sun_to_gs_t3[:3]) / c
        dt_GR_dwn, r23_vec = Measurement._compute_RLT_lt(
            "RLT",
            sun_to_sc_t2,
            sun_to_gs_t3,
            t2,
            t3,
            sc_spice_id,
            ssb_to_gs_t3_RLTframe,
            "dwn",
            delta_partials.values[0],
        )

        # Step (19) - Skipped since 2W (sranging or doppler) assumption
        ### up-leg portion (PART 2 of 2): from t2 (reception from spacecraft) to t1 (transmission from ground station) ###

        # Step (20) - Get the first estimate for t1 using the converged delta time from t3 to t2
        r2_vec = ssb_to_sc_t2
        r3_vec = ssb_to_gs_t3_RLTframe
        r23_vec = r3_vec - r2_vec
        r23 = np.linalg.norm(r23_vec[:3])
        r_dot_E = np.dot(
            r23_vec[:3] / r23, ssb_to_earth_t3[3:6]
        )  # Moyer 2000, eq. 8-79
        lt_12 = (dt_CN_dwn + dt_GR_dwn) * (1 - 2 * r_dot_E / c)
        t1 = t2 - lt_12  # first estimate of t1
        t1_h, t1_l = EpochArray._split_fractional_part(t1)

        # Step (21) - Skipped, done with Spice + RLT-corrected GS location
        # Step (22) - Skipped, the transmitter is not an Earth satellite

        delta_t1 = 10
        max_it_counter = 0
        while (
            abs(delta_t1) > 1e-12 and max_it_counter < 10
        ):  # step(29) - Stay within the while loop if the absolute value of delta_t1 is less than 0.1s
            max_it_counter = max_it_counter + 1
            # Steps (23, 24) - Compute ground-station state in SSB space-time relativistic frame at t1
            earth_to_gs_t1_RLTframe = Measurement._compute_GS_in_SSB_space_time_RLT(
                gs_spice_id, t1
            )
            # Step (25) - Compute quantities of interest for B's and S. Same as step (4): quantities of interest fo the bodies are computed inside the "_compute_RLT_BODIES_lt" method
            ssb_to_earth_t1 = SpiceManager.get_state_precise(
                "Earth", t1_h, "J2000", "SSB", "None", t1_l
            ).values
            ssb_to_gs_t1_RLTframe = ssb_to_earth_t1 + earth_to_gs_t1_RLTframe
            sun_to_gs_t1 = SpiceManager.get_state_precise(
                gs_spice_id, t1_h, "J2000", "10", "None", t1_l
            ).values
            # Step (26) - Skipped, coputation in Barycentric already perforned at step 25
            # Step (27) - Generate r_12_vec, r_12_vec_dot and r_12 and compute RLT_12. This computation is split in two part and not performed originally with the 3 terms from Moyer 2000 eq. 8-55.
            # Converged newtonian (CN) component of Moyer 2000 eq. 8-55s (1st term)
            if CN_flag:
                # Note: this term is different than the one in Moyer 2000 eq. 8-55, becuase it's already internally converged.
                dt_CN_up = Measurement._compute_CN_lt(
                    ssb_to_sc_t2[:3], gs_spice_id, t2, delta_partials.values[0], "up"
                )
            else:
                dt_CN_up = np.linalg.norm(sun_to_sc_t2[:3] - sun_to_gs_t1[:3]) / c
            # Sun and Body Relativistic (RLT) components of Moyer 2000 eq. 8-55 (2nd and 3rd terms)
            dt_GR_up, r12_vec = Measurement._compute_RLT_lt(
                "RLT",
                sun_to_gs_t1,
                sun_to_sc_t2,
                t1,
                t2,
                sc_spice_id,
                ssb_to_gs_t1_RLTframe,
                "up",
                delta_partials.values[0],
            )
            r12 = np.linalg.norm(r12_vec[:3])
            # Get the velocity of the spacecraft at reflection time t2 in SSB space-time frame
            r_dot_1_C_t1 = ssb_to_gs_t1_RLTframe[3:6]
            # Compute p_dot from Moyer 2000 eq. 8-60
            p_dot_12 = np.dot(r12_vec[:3] / r12, r_dot_1_C_t1)
            # Step (28) - calculate the linear differential correction delta_t1, Moyer 2000 eq. 8-72
            delta_t1 = (t2 - t1 - dt_CN_up - dt_GR_up) / (1 - p_dot_12 / c)
            t1 += delta_t1  # update t1
            t1_h, t1_l = EpochArray._split_fractional_part(
                t1
            )  # update t2 (integer and fractional parts)

        # Step (30) - Map everything calculated at the last estimate of the converged value for t1(ET). From Moyer, this is substituted by spice calls
        ssb_to_earth_t1 = SpiceManager.get_state_precise(
            "Earth", t1_h, "J2000", "SSB", "None", t1_l
        ).values
        sun_to_gs_t1 = SpiceManager.get_state_precise(
            gs_spice_id, t1_h, "J2000", "10", "None", t1_l
        ).values

        # Step (31) - Using the mapping quantities from Step (30), repeat steps 24 to 27
        earth_to_gs_t1_RLTframe = Measurement._compute_GS_in_SSB_space_time_RLT(
            gs_spice_id, t1
        )
        ssb_to_gs_t1_RLTframe = ssb_to_earth_t1 + earth_to_gs_t1_RLTframe
        if CN_flag:
            dt_CN_up = Measurement._compute_CN_lt(
                ssb_to_sc_t2[:3], gs_spice_id, t2, delta_partials.values[0], "up"
            )
        else:
            dt_CN_up = np.linalg.norm(sun_to_sc_t2[:3] - sun_to_gs_t1[:3]) / c
            dt_GR_up, r12_vec = Measurement._compute_RLT_lt(
                "RLT",
                sun_to_gs_t1,
                sun_to_sc_t2,
                t1,
                t2,
                sc_spice_id,
                ssb_to_gs_t1_RLTframe,
                "up",
                delta_partials.values[0],
            )

        # Step (32) - convert t1(ET) to t1(TAI) via Moyer eq. 2-23
        dt_et_tai_t1 = Measurement._compute_delta_et_tai(t1, gs_spice_id, 3)
        dt_et_tai_t3 = Measurement._compute_delta_et_tai(t3, gs_spice_id, 3)

        t1_tai = (
            t1 - dt_et_tai_t1
        )  # unused, but can be exported outside for debugging purposes

        # STEP (33) - Extra, compute extra effects on rtlt delay
        # Solar corona effects
        if solar_corona_dict["ul_freq"] != 0 and solar_corona_dict["M2"] != 0:
            dt_SC_dwn = scb.MediaCorrections._compute_SC_lt(
                sun_to_sc_t2,
                sun_to_gs_t3,
                solar_corona_dict["ul_freq"] * solar_corona_dict["M2"],
            )
            dt_SC_up = scb.MediaCorrections._compute_SC_lt(
                sun_to_sc_t2, sun_to_gs_t1, solar_corona_dict["ul_freq"]
            )
        else:
            dt_SC_dwn = 0
            dt_SC_up = 0

        # Antenna effects
        dt_A_t3 = 0
        dt_A_t1 = 0

        # Transmitter and receiver delays
        dt_tau_D = (
            electronic_delays_dict["rcv_tt_dly"]
            + electronic_delays_dict["dl_zheight_corr"]
            + electronic_delays_dict["scft_transpd_delay_s"]
        )  # Downlink delays
        dt_tau_U = (
            electronic_delays_dict["trn_tt_dly"]
            + electronic_delays_dict["ul_zheight_corr"]
        )  # Uplink delays

        ### OUTPUT handling ####
        rtlt_parts = np.zeros((17, 1))
        # Compute RTLT toghether
        # From Moyer 2000, pg. 11-11, eq.11-7
        rtlt_parts[0] = dt_CN_dwn  # Converged Newtonian, down-leg
        rtlt_parts[1] = dt_GR_dwn  # Relativistic, down-leg
        rtlt_parts[2] = dt_CN_up  # Converged Newtonian, up-leg
        rtlt_parts[3] = dt_GR_up  # Relativistic, up-leg
        rtlt_parts[4] = -dt_et_tai_t3  # ET-TAI at t3
        rtlt_parts[5] = dt_et_tai_t1  # ET-TAI at t1
        rtlt_parts[6] = 0  # UTC-TAI at t3, currently not modeled
        rtlt_parts[7] = 0  # UTC-TAI at t1, currently not modeled
        rtlt_parts[8] = 0  # UTC-ST at t3, currently not modeled
        rtlt_parts[9] = 0  # UTC-ST at t1, currently not modeled
        rtlt_parts[10] = 0  # Rc, solve-for bias GS location
        rtlt_parts[11] = dt_A_t3  # A at t3
        rtlt_parts[12] = dt_SC_dwn  # Solar Corona, down-leg
        rtlt_parts[13] = dt_A_t1  # A at t1
        rtlt_parts[14] = dt_SC_up  # Solar Corona, up-leg
        rtlt_parts[15] = dt_tau_D  # Downlink delay
        rtlt_parts[16] = dt_tau_U  # Uplink delay

        return np.sum(rtlt_parts), rtlt_parts, t2, t1

    @staticmethod
    def _compute_precise_RTLT_light(
        sc_spice_id: str,
        gs_spice_id: str,
        tt_et: float,
        electronic_delays_dict: dict,
        solar_corona_dict: dict,
        delta_partials: ArrayWUnits,
    ) -> np.ndarray:
        """
        Method to compute the precise Round-Trip Light-Time (RTLT) between a gs-based DSN antenna and a spacecraft.
        Light variant, not following the Moyer 32 steps algorithm in Moyer 2000, Sec. 8.3.6

        This method computes the RTLT for a 2-way path between:
            - A DSN antenna transmitting a signal at t1
            - A spacecraft reflecting a signal at t2
            - A DSN antenna receiving a signal at t3

           implement the necessary time delays between transmittion from a gs at (t1) and reception at the same gs at (t3).

        Parameters
        ----------
        sc_spice_id : str
            spice id string of the spacecraft

        gs_spice_id : str
            spice id string of the ground station

        tt_et : float
            time-tag in ephemeris time of the reception time at the gs (t3)

        electronic_delays_dict : dict
            dictionary of electronic delays

        solar_corona_dict : dict
            dictionary of solar corona parameters

        delta: ArrayWUnits
            delta vector for position and velocity. Used to generate numeric _partials

        Returns
        -------
        rtlt : float
            Sum of the rtlt delays components as a single delay value in second between t1 and t3

        rtlt_parts : np.array(17,1)
            rtlt decomposed in its components, expressed in seconds

        Notes
        -----
        See also the method SpiceManager.get_state_precise() which needs integer and fractional time for precise
        ephemeris retrivial

        [1]: "TRK-2-34, DSN Tracking System Data Archival Format", DSN No. 820-013, TRK-2-34, Rev. N

        """
        # Define constants to be used within this method
        c = constants.c.values  # Speed of light in vacuum [km/sec]

        tt_tai_i = spice.unitim(tt_et, "ET", "TAI") + (
            -electronic_delays_dict["rcv_tt_dly"]
            + electronic_delays_dict["trn_tt_dly"]
            - electronic_delays_dict["array_delay"]
        )  # As per [1], Notes 2,3, and 4

        # Transform t3(TAI) to t3(ET)
        tt_et, dt_et_tai_t3, r_stn_BC = Measurement._et_tai_debug(tt_tai_i, gs_spice_id)
        t3 = tt_et
        t3_h, t3_l = EpochArray._split_fractional_part(
            t3
        )  # Separate integer and fractional components of t3 for precise state retrivial

        # Get partecipating bodies at t3
        ssb_to_earth_t3 = SpiceManager.get_state_precise(
            "Earth", t3_h, "J2000", "SSB", "None", t3_l
        ).values
        ssb_to_sun_t3 = SpiceManager.get_state_precise(
            "10", t3_h, "J2000", "SSB", "None", t3_l
        ).values
        ssb_to_gs_t3_RLTframe = (
            np.concatenate((r_stn_BC, np.zeros(3))) + ssb_to_earth_t3
        )
        sun_to_gs_t3 = ssb_to_gs_t3_RLTframe - ssb_to_sun_t3

        # CN
        dt_CN_dwn = Measurement._compute_CN_lt(
            ssb_to_gs_t3_RLTframe[:3], sc_spice_id, t3, delta_partials.values[0], "down"
        )
        t2 = t3 - dt_CN_dwn
        t2_h, t2_l = EpochArray._split_fractional_part(t2)
        # GR
        sun_to_sc_t2 = (
            SpiceManager.get_state_precise(
                sc_spice_id, t2_h, "J2000", "10", "None", t2_l
            ).values
            + delta_partials.values[0]
        )
        dt_GR_dwn, _ = Measurement._compute_RLT_lt(
            "RLT",
            sun_to_sc_t2,
            sun_to_gs_t3,
            t2,
            t3,
            sc_spice_id,
            ssb_to_gs_t3_RLTframe,
            "dwn",
            delta_partials.values[0],
        )
        # SC
        if solar_corona_dict["ul_freq"] != 0 and solar_corona_dict["M2"] != 0:
            dt_SC_dwn = scb.MediaCorrections._compute_SC_lt(
                sun_to_sc_t2,
                sun_to_gs_t3,
                solar_corona_dict["ul_freq"] * solar_corona_dict["M2"],
            )
        else:
            dt_SC_dwn = 0
        # Antenna and electronic delays
        dt_A_t3 = 0
        dt_tau_D = (
            electronic_delays_dict["rcv_tt_dly"]
            + electronic_delays_dict["dl_zheight_corr"]
            + electronic_delays_dict["scft_transpd_delay_s"]
        )  # Downlink delays

        t2 = t3 - dt_CN_dwn - dt_GR_dwn - dt_SC_dwn - dt_A_t3 - dt_tau_D
        t2_h, t2_l = EpochArray._split_fractional_part(t2)

        # Get partecipating bodies at t2
        ssb_to_earth_t2 = SpiceManager.get_state_precise(
            "Earth", t2_h, "J2000", "SSB", "None", t2_l
        ).values
        ssb_to_sun_t2 = SpiceManager.get_state_precise(
            "10", t2_h, "J2000", "SSB", "None", t2_l
        ).values
        ssb_to_gs_t2_RLTframe = (
            np.concatenate((r_stn_BC, np.zeros(3))) + ssb_to_earth_t2
        )
        sun_to_gs_t2 = ssb_to_gs_t2_RLTframe - ssb_to_sun_t2
        ssb_to_sc_t2 = (
            SpiceManager.get_state_precise(
                sc_spice_id, t2_h, "J2000", "SSB", "None", t2_l
            ).values
            + delta_partials.values[0]
        )

        # CN
        dt_CN_up = Measurement._compute_CN_lt(
            ssb_to_sc_t2[:3], gs_spice_id, t2, delta_partials.values[0], "up"
        )
        t1 = t2 - dt_CN_up
        t1_h, t1_l = EpochArray._split_fractional_part(t1)
        # GR
        earth_to_gs_t1_RLTframe = Measurement._compute_GS_in_SSB_space_time_RLT(
            gs_spice_id, t1
        )
        ssb_to_earth_t1 = SpiceManager.get_state_precise(
            "Earth", t1_h, "J2000", "SSB", "None", t1_l
        ).values
        ssb_to_gs_t1_RLTframe = ssb_to_earth_t1 + earth_to_gs_t1_RLTframe
        sun_to_gs_t1 = SpiceManager.get_state_precise(
            gs_spice_id, t1_h, "J2000", "10", "None", t1_l
        ).values
        dt_GR_up, r12_vec = Measurement._compute_RLT_lt(
            "RLT",
            sun_to_gs_t1,
            sun_to_sc_t2,
            t1,
            t2,
            sc_spice_id,
            ssb_to_gs_t1_RLTframe,
            "up",
            delta_partials.values[0],
        )
        # SC
        if solar_corona_dict["ul_freq"] != 0 and solar_corona_dict["M2"] != 0:
            dt_SC_up = scb.MediaCorrections._compute_SC_lt(
                sun_to_sc_t2, sun_to_gs_t1, solar_corona_dict["ul_freq"]
            )
        else:
            dt_SC_up = 0
        # Antenna and electronic delays
        dt_A_t1 = 0
        dt_tau_U = (
            electronic_delays_dict["trn_tt_dly"]
            + electronic_delays_dict["ul_zheight_corr"]
        )  # Uplink delays

        t1 = t2 - dt_CN_up - dt_SC_up - dt_SC_up - dt_A_t1 - dt_tau_U
        t1_h, t1_l = EpochArray._split_fractional_part(t1)

        # ET-TAI at t1 and t3
        dt_et_tai_t1 = Measurement._compute_delta_et_tai(t1, gs_spice_id, 3)
        dt_et_tai_t3 = Measurement._compute_delta_et_tai(t3, gs_spice_id, 3)

        ### OUTPUT handling ####
        rtlt_parts = np.zeros((17, 1))
        # Compute RTLT toghether
        # From Moyer 2000, pg. 11-11, eq.11-7
        rtlt_parts[0] = dt_CN_dwn  # Converged Newtonian, down-leg
        rtlt_parts[1] = dt_GR_dwn  # Relativistic, down-leg
        rtlt_parts[2] = dt_CN_up  # Converged Newtonian, up-leg
        rtlt_parts[3] = dt_GR_up  # Relativistic, up-leg
        rtlt_parts[4] = -dt_et_tai_t3  # ET-TAI at t3
        rtlt_parts[5] = dt_et_tai_t1  # ET-TAI at t1
        rtlt_parts[6] = 0  # UTC-TAI at t3, currently not modeled
        rtlt_parts[7] = 0  # UTC-TAI at t1, currently not modeled
        rtlt_parts[8] = 0  # UTC-ST at t3, currently not modeled
        rtlt_parts[9] = 0  # UTC-ST at t1, currently not modeled
        rtlt_parts[10] = 0  # Rc, solve-for bias GS location
        rtlt_parts[11] = dt_A_t3  # A at t3
        rtlt_parts[12] = dt_SC_dwn  # Solar Corona, down-leg
        rtlt_parts[13] = dt_A_t1  # A at t1
        rtlt_parts[14] = dt_SC_up  # Solar Corona, up-leg
        rtlt_parts[15] = dt_tau_D  # Downlink delay
        rtlt_parts[16] = dt_tau_U  # Uplink delay

        return np.sum(rtlt_parts), rtlt_parts, t2, t1

    @staticmethod
    def _et_tai_debug(t: float, gs_spice_id: str) -> np.ndarray:
        """
        DEBUG Method to compute the time difference between Ephemeris Time (ET) and International Atomic Time (TAI) at a ground station on Earth.

        Parameters
        ----------
        t : float
            ephemeris time at which to compute the time correction

        gs_spice_id : str
            spice id string of the ground station

        Returns
        -------
        t_et : float
            adjusted ephemeris times

        dt: float
            et-tai correction

        X_A_E: np.ndarray
            Ground station state

        """
        # Constants
        mu_S = constants.SUN.GM.values
        mu_J = constants.JUPITER.GM.values
        mu_Sa = constants.SATURN.GM.values
        c = constants.c.values  # Speed of light in vacuum [km/sec]
        gamma_GR = constants.gamma_GR.values  # gamma value for general relativity
        Ltilde_GR = constants.L_tilde_GR.values  # Ltilde value for general relativity

        # Step 1: Compute approximate ET-TAI
        M = 6.239996 + 1.99096871e-7 * t
        E = M + 0.01671 * np.sin(M)
        dt = 32.184 + 1.657e-3 * np.sin(E)
        t_et = t + dt

        et_h, et_l = EpochArray._split_fractional_part(t_et)
        X_B_S = SpiceManager.get_state_precise("3", et_h, "J2000", "Sun", "None", et_l)
        X_B_C = SpiceManager.get_state_precise("3", et_h, "J2000", "SSB", "None", et_l)
        X_E_B = SpiceManager.get_state_precise(
            "Earth", et_h, "J2000", "3", "None", et_l
        )
        X_E_C = SpiceManager.get_state_precise(
            "Earth", et_h, "J2000", "SSB", "None", et_l
        )
        X_J_S = SpiceManager.get_state_precise(
            "599", et_h, "J2000", "Sun", "None", et_l
        )
        X_Sa_S = SpiceManager.get_state_precise(
            "699", et_h, "J2000", "Sun", "None", et_l
        )
        X_S_C = SpiceManager.get_state_precise(
            "Sun", et_h, "J2000", "SSB", "None", et_l
        )

        # Compute X_A_E
        r_E_S = SpiceManager.get_state_precise(
            "Earth", et_h, "J2000", "Sun", "None", et_l
        )
        r_E_SSB = SpiceManager.get_state_precise(
            "Earth", et_h, "J2000", "SSB", "None", et_l
        )
        r_E_mag = np.linalg.norm(r_E_S[:3].values)
        UE = mu_S / r_E_mag
        VE = r_E_SSB[3:6].values
        it_to_j2 = spice.sxform("ITRF93", "J2000", t_et)
        rGC = np.dot(
            it_to_j2,
            SpiceManager.get_state_precise(
                gs_spice_id, et_h, "ITRF93", "EARTH", "None", et_l
            ).values,
        )
        rBC = (1 - Ltilde_GR - (gamma_GR * UE / c**2)) * rGC[:3] - (
            1 / (2 * c**2)
        ) * np.dot(VE, rGC[:3]) * VE
        X_A_E = rBC

        f1 = 32.184  # [s]
        f2 = 2 / (c**2) * (np.dot(X_B_S[3:6].values, X_B_S[0:3].values))  #
        f3 = 1 / (c**2) * (np.dot(X_B_C[3:6].values, X_E_B[0:3].values))  #
        f4 = 1 / (c**2) * (np.dot(X_E_C[3:6].values, X_A_E[0:3]))  #
        f5 = (
            1
            / (c**2)
            * (mu_J / (mu_S + mu_J))
            * (np.dot(X_J_S[3:6].values, X_J_S[0:3].values))
        )  #
        f6 = (
            1
            / (c**2)
            * (mu_Sa / (mu_S + mu_Sa))
            * (np.dot(X_Sa_S[3:6].values, X_Sa_S[0:3].values))
        )  #
        f7 = 1 / (c**2) * (np.dot(X_S_C[3:6].values, X_B_S[0:3].values))  #
        dt = f1 + f2 + f3 + f4 + f5 + f6 + f7

        # Update ET
        t_et = t + dt
        t_et_2 = t_et
        et_h, et_l = EpochArray._split_fractional_part(t_et_2)

        # Update Epehemeris again
        X_B_S = SpiceManager.get_state_precise("3", et_h, "J2000", "Sun", "None", et_l)
        X_B_C = SpiceManager.get_state_precise("3", et_h, "J2000", "SSB", "None", et_l)
        X_E_B = SpiceManager.get_state_precise(
            "Earth", et_h, "J2000", "3", "None", et_l
        )
        X_E_C = SpiceManager.get_state_precise(
            "Earth", et_h, "J2000", "SSB", "None", et_l
        )
        X_J_S = SpiceManager.get_state_precise(
            "599", et_h, "J2000", "Sun", "None", et_l
        )
        X_Sa_S = SpiceManager.get_state_precise(
            "699", et_h, "J2000", "Sun", "None", et_l
        )
        X_S_C = SpiceManager.get_state_precise(
            "Sun", et_h, "J2000", "SSB", "None", et_l
        )
        # Compute X_A_E
        r_E_S = SpiceManager.get_state_precise(
            "EARTH", et_h, "J2000", "Sun", "None", et_l
        )
        r_E_SSB = SpiceManager.get_state_precise(
            "EARTH", et_h, "J2000", "SSB", "None", et_l
        )
        r_E_mag = np.linalg.norm(r_E_S[:3].values)
        UE = mu_S / r_E_mag
        VE = r_E_SSB[3:6].values
        it_to_j2 = spice.sxform("ITRF93", "J2000", t_et)
        rGC = np.dot(
            it_to_j2,
            SpiceManager.get_state_precise(
                gs_spice_id, et_h, "ITRF93", "Earth", "None", et_l
            ).values,
        )
        rBC = (1 - Ltilde_GR - (gamma_GR * UE / c**2)) * rGC[:3] - (
            1 / (2 * c**2)
        ) * np.dot(VE, rGC[:3]) * VE
        X_A_E = rBC

        f1 = 32.184  # [s]
        f2 = 2 / (c**2) * (np.dot(X_B_S[3:6].values, X_B_S[0:3].values))  #
        f3 = 1 / (c**2) * (np.dot(X_B_C[3:6].values, X_E_B[0:3].values))  #
        f4 = 1 / (c**2) * (np.dot(X_E_C[3:6].values, X_A_E[0:3]))  #
        f5 = (
            1
            / (c**2)
            * (mu_J / (mu_S + mu_J))
            * (np.dot(X_J_S[3:6].values, X_J_S[0:3].values))
        )  #
        f6 = (
            1
            / (c**2)
            * (mu_Sa / (mu_S + mu_Sa))
            * (np.dot(X_Sa_S[3:6].values, X_Sa_S[0:3].values))
        )  #
        f7 = 1 / (c**2) * (np.dot(X_S_C[3:6].values, X_B_S[0:3].values))  #

        dt = f1 + f2 + f3 + f4 + f5 + f6 + f7
        t_et = t + dt

        return t_et, dt, X_A_E

    @staticmethod
    def _compute_GS_in_SSB_space_time_RLT(gs_spice_id: str, t: float) -> np.ndarray:
        """
        Method to compute GS state vector in a Solar System Barycentric space-time relativistic frame of reference.

        Parameters
        ----------
        gs_spice_id : str
            spice id string of the ground station

        t : float
            time in ephemeris time

        Returns
        -------
        rBC_state : np.ndarray
            State vector (velocity padded to zeros) ot the ground station

        """
        # Define constants
        mu_sun = constants.SUN.GM.values
        Ltilde_GR = constants.L_tilde_GR.values
        gamma_GR = constants.gamma_GR.values
        c = constants.c.values

        # Compute sun-earth and ssb-earth states
        sun_to_earth_t = SpiceManager.get_state(
            "Earth", t, "J2000", "Sun", "None"
        ).values
        ssb_to_earth_t = SpiceManager.get_state(
            "Earth", t, "J2000", "SSB", "None"
        ).values
        # Compute UE and VE
        UE = mu_sun / np.linalg.norm(sun_to_earth_t[:3])
        VE = ssb_to_earth_t[3:6]
        # Get the state of the station in ITRF93 relative to Earth at time t
        earth_to_gs_t1_ITRF = SpiceManager.get_state(
            gs_spice_id, t, "ITRF93", "Earth", "None"
        ).values
        # Get the state transformation matrix from ITRF93 to J2000 at time t
        T_from_ITRF93_to_J2000 = spice.sxform("ITRF93", "J2000", t)
        # Convert GS state from ITRF93 to J2000: this returns GS state in geocentric coordinates
        rGC = T_from_ITRF93_to_J2000 @ earth_to_gs_t1_ITRF
        # Apply relativistic correction to transform GS position in SSB space-time relativistic frame (Moyer 2000, eq. 4-10)
        rBC = (1 - Ltilde_GR - (gamma_GR * UE / c**2)) * rGC[:3] - (
            1 / (2 * c**2)
        ) * np.dot(VE, rGC[:3]) * VE
        # Padd rBC for easier math
        rBC_state = np.concatenate([rBC, np.zeros(3)])
        return rBC_state

    @staticmethod
    def _generate_electronic_delays_dictionary(aux_dict, idx):
        """
        Generate a dictionary of electronic delays as per [1]

        Parameters
        ----------
        aux_dict : dict
            Dictionary of auxiliary parameters for the computed measurements

        ul_freq_i : float
            frequency to use in the seconds-to-RU conversion factor

        Returns
        -------
        sranging_calib_corr : float
            Value of the 2-way sequential ranging correction term in [1], Note 39.

        Notes
        -----
        [1]: "TRK-2-34, DSN Tracking System Data Archival Format", DSN No. 820-013, TRK-2-34, Rev. N

        """

        # Conver the invalid delays to 0 [s] delays

        # Receive and transmit time-tag delays
        rcv_tt_dly_i = aux_dict["rcv_time_tag_delay"][idx]  # [s]
        trn_tt_dly_i = aux_dict["transmit_time_tag_delay"][idx]  # [s]
        if rcv_tt_dly_i == -1:
            rcv_tt_dly_i = 0
        if trn_tt_dly_i == -1:
            trn_tt_dly_i = 0
        # Uplink and Downlink zheight corrections
        ul_zheight_corr_i = aux_dict["ul_zheight_corr"][idx]  # [s]
        dl_zheight_corr_i = aux_dict["dl_zheight_corr"][idx]  # [s]
        if ul_zheight_corr_i == -99.0:  # Invalid value for ul_zheight_corr (see [1])
            ul_zheight_corr_i = 0
        if dl_zheight_corr_i == -99.0:  # Invalid value for dl_zheight_corr (see [1])
            dl_zheight_corr_i = 0
        # Array delay
        if isinstance(aux_dict.get("array_delay"), (list, tuple, np.ndarray)):
            array_delay_i = aux_dict["array_delay"][idx]  # [s]
            if array_delay_i in (-99, -99.0, -1, -1.0):
                array_delay_i = 0

        # Compact the electronic delays in a dictionary
        electronic_delays_dict = {
            "ul_freq": aux_dict["ul_freq"][
                idx
            ],  # uplink frequency at time tag [Hz], see [1] 3-70
            "rcv_tt_dly": rcv_tt_dly_i,  # transmitter timetag delay [s], see [1] 3-11
            "trn_tt_dly": trn_tt_dly_i,  # receiver timetag delay [s], see [1] 3-10
            "ul_zheight_corr": ul_zheight_corr_i,  # uplink zheight correction [s], see [1] 3-10
            "dl_zheight_corr": dl_zheight_corr_i,  # downlink zheight correction [s], see [1] 3-11
            "ul_stn_cal": aux_dict["ul_stn_cal"][
                idx
            ],  # uplink GS calibration [RU], see [1] 3-68
            "dl_stn_cal": aux_dict["dl_stn_cal"][
                idx
            ],  # downlink GS calibration [RU], see [1] 3-68
            "scft_transpd_delay": aux_dict[
                "scft_transpd_delay"
            ],  # spacecraft transponder delay correction [RU], see [1] 3-12
            "scft_transpd_delay_s": aux_dict[
                "scft_transpd_delay_s"
            ],  # spacecraft transponder delay correction [s]
            "array_delay": array_delay_i,  # time delay added to the path delay due to arraying equipment [s], see [1] 3-11
        }
        return electronic_delays_dict

    @staticmethod
    def _progress_monitor(num_total: int, desc: str = "Processing", unit: str = "obs"):
        """
        Returns a tqdm-based progress updater function.

        Parameters
        ----------
        num_total : int
            Total number of items to process.
        desc : str
            Description for the tqdm bar.
        unit : str
            Unit label (e.g., 'obs', 'steps').

        Returns
        -------
        update_fn : Callable[[int], None]
            A function to call with the current index.
        close_fn : Callable[[], None]
            A function to finalize and close the bar.
        """
        pbar = tqdm(
            total=num_total,
            desc=desc,
            unit=unit,
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}]",
        )

        def update(current_index: int):
            pbar.n = current_index + 1
            pbar.refresh()

        def close():
            pbar.n = num_total
            pbar.refresh()
            pbar.close()

        return update, close
