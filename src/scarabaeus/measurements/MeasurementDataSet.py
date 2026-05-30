# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import EpochArray, Body, Instrument, GroundStation
import numpy as np 

#-----------------------#
#    Class Definition   #
#-----------------------#
class MeasurementDataSet:
    """ Groups together measurement data from different measurement models. Used by the active filter.

    Generates a list of packaged data from a given measurement. At each epoch at which the filter
    has a valid measurement, provides access to its computed and observed value, residual, numerical
    value of the partial, and the standard deviation of the measurement.

    Parameters
    ----------
    set_name : str
        The name of the dataset.
    measurements : list
        List with AWU, np.array, and lists of measurements data
        (e.g. observed, computed, residuals, partials, sigmas).
    names : list[str]
        List of values in the measurement list
        (``["computed", "observed", "residuals", "partials", "sigma"]``).
    instrument : Instrument
        Instrument used to generate these measurements (e.g. antenna, camera, etc.)
    target : Body
        Target body of the measurement model.
    epochs_t3 : EpochArray
        Epochs of the measurements.
    measurement_type : str, optional
        The type of measurements. Can be given as:

        * range
        * range real
        * rangerate
        * doppler
        * doppler real
        * opnav
        * dor

        Defaults to ``None``.
    station_two : GroundStation, optional
        Optional Ground station object for three-way and DOR measurements. Defaults to ``None``.

    Notes
    -----
    The dataset is populated by :meth:`~scarabaeus.Measurement.generate_measurement_dataset`
    and stores all quantities needed by a filter at each measurement epoch:
    computed value, observed value, prefit residual, measurement partials
    (:math:`H` row), and measurement noise standard deviation
    (:math:`\\sigma`).  The ``outlier_flag`` column (value 0/1) allows
    downstream editing without removing data permanently.

    See Also
    --------
    scarabaeus.Measurement : Abstract base class for all measurement models.
    """
    #--------------------#
    # region Constructor #
    #--------------------#
    def __init__(self, set_name : str, measurements : list, names : list[str], instrument : Instrument, target : Body,
                 epochs_t3 : EpochArray, measurement_type : str = None, station_two : GroundStation = None):

        # Set
        self._set_name         = set_name
        self._data             = dict(zip(names, measurements))
        self._instrument       = instrument
        self._target           = target
        self._epochs_t3           = epochs_t3
        self._measurement_type = measurement_type
        self._station_two      = station_two
    
    #----------------------#
    # region    Properties #
    #----------------------#
    @property
    def set_name(self) -> str:
        """The name of the dataset."""
        return self._set_name

    @property
    def data(self) -> list[str]:
        """The measurement data dictionary (computed, observed, residuals, partials, sigmas)."""
        return self._data

    @property
    def instrument(self) -> Instrument:
        """The instrument that generated the measurements."""
        return self._instrument

    @property
    def target(self) -> Body:
        """Target body of the measurement model."""
        return self._target

    @property
    def epochs_t3(self) -> EpochArray:
        """The reception epochs linked to the measurements."""
        return self._epochs_t3

    @epochs_t3.setter
    def epochs_t3(self, value: EpochArray):
        """
        Allow assigning a new EpochArray.
        """
        self._epochs_t3 = value
        
    @property
    def measurement_type(self):
        """The measurement type string tag (e.g. ``"range"``, ``"doppler"``)."""
        return self._measurement_type

    @property
    def station_two(self) -> GroundStation:
        """The second ground station, for three-way and DOR measurement types."""
        return self._station_two

    # endregion Properties #
    #----------------------#

    #----------------#
    # region Methods #
    #----------------#
    def add_data(self, data : list, name : str):
        """
        Adds data to an existing dataset.

        Parameters
        ----------
        data : list
            Extra data to add to an existing MeasurementDataset object.

        name : str
            The name of the dataset. 
        """
        self.data[name] = data

    
    def flag_outliers_by_resid(
        self, 
        mean_outlier: float | None = None, 
        sigma_outlier: float | None = None
    ) -> None:
        """
        Flag outliers in the MeasurementDataset based on residual values.
        
        Identifies measurements whose residuals fall outside the interval 
        [mean - sigma, mean + sigma] and marks them as outliers.
        
        Parameters
        ----------
        mean_outlier : float, optional
            Center of the acceptable residual interval. If None, uses the mean 
            of current residuals. Default is None.
        sigma_outlier : float, optional
            Half-width of the acceptable residual interval. If None, uses 5 times 
            the standard deviation of current residuals. Default is None.
        
        Returns
        -------
        None
            Modifies self.data['outlier_flag'] in place.
        
        Raises
        ------
        KeyError
            If 'outlier_flag' column is missing from self.data.
        KeyError
            If 'residuals' column is missing from self.data.
        
        Notes
        -----
        - By default (when both parameters are None), uses 5-sigma criterion
        - Outliers are flagged with value 1 in the 'outlier_flag' column
        - Existing outlier flags are preserved and combined with new detections
        
        Examples
        --------
        >>> # Flag outliers using default 5-sigma criterion
        >>> dataset.flag_outliers_by_resid()
        
        >>> # Flag outliers using custom interval
        >>> dataset.flag_outliers_by_resid(mean_outlier=0.0, sigma_outlier=3.0)
        """
        # Validate required columns exist
        if "outlier_flag" not in self.data:
            raise KeyError(
                "Expected column 'outlier_flag' in self.data, but it is missing. "
                "Initialize this column before calling flag_outliers_by_resid()."
            )
        
        if "residuals" not in self.data:
            raise KeyError(
                "Expected column 'residuals' in self.data, but it is missing."
            )
        
        residuals = self.data['residuals'].values
        
        # Set default parameters: 5-sigma around current mean
        if mean_outlier is None:
            mean_outlier = np.mean(residuals)
        if sigma_outlier is None:
            sigma_outlier = 5.0 * np.std(residuals)
        print("Automatically flagging outliers using 5-sigma criterion around current mean.")
        
        # Identify outliers outside [mean - sigma, mean + sigma]
        lower_bound = mean_outlier - sigma_outlier
        upper_bound = mean_outlier + sigma_outlier
        mask = (residuals < lower_bound) | (residuals > upper_bound)
        
        # Flag outliers (using .loc for safer assignment)
        self.data.loc[mask, 'outlier_flag'] = 1

    def remove_outliers(self):
        """
        Remove from the MeasurementDataSet object the data and epochs_t3 associated to measurements with outlier_flag equal to 1.

        Raises
        ------
        KeyError
            If ``outlier_flag`` key is missing from ``self.data``.
        """

        if "outlier_flag" not in self.data:
            raise KeyError("Expected self.data['outlier_flag'], but the key is missing.")
        
        # Determine remove and keep indeces
        remove_idx = np.where(self.data['outlier_flag']==1)[0]
        keep_idx = np.where(self.data['outlier_flag']==0)[0]

        # Remove the outlier data from the MeasurementDataSet.data
        if isinstance(self.data, dict):
            for key, arr in self.data.items():
                if key == 'partials':
                    self.data[key] = np.asarray(arr)[keep_idx]
                else:
                    self.data[key] = arr[keep_idx]
        # Remove the outlier data from the MeasurementDataSet.epochs_t3
        self.epochs_t3 = self.epochs_t3[keep_idx]
        # Print the outlier removal
        print(f"Removed {len(remove_idx)} outliers from the MeasurementDataSet; "
            f"{len(keep_idx)} points remain.")

        return keep_idx, remove_idx