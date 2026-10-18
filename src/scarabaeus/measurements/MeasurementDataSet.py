# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import EpochArray, Body, Instrument, GroundStation

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

    See Also
    --------
    scarabaeus.Measurement : Abstract base class for all measurement models.

    Notes
    -----
    The dataset is populated by :meth:`~scarabaeus.Measurement.generate_measurement_dataset`
    and stores all quantities needed by a filter at each measurement epoch:
    computed value, observed value, prefit residual, measurement partials
    (:math:`H` row), and measurement noise standard deviation
    (:math:`\\sigma`).  The ``outlier_flag`` column (value 0/1) allows
    downstream editing without removing data permanently.
    """
    #--------------------#
    # region Constructor #
    #--------------------#
    def __init__(self, set_name : str, measurements : list, names : list[str], instrument : Instrument, target : Body,
                 epochs_t3 : EpochArray, measurement_type : str = None, station_two : GroundStation = None,
                 units=None, frame=None):

        # Set
        self._set_name         = set_name
        self._data             = dict(zip(names, measurements))
        self._instrument       = instrument
        self._target           = target
        self._epochs_t3           = epochs_t3
        self._measurement_type = measurement_type
        self._station_two      = station_two
        # Frame / units of the measurement values, captured once at construction
        # (the filter computes on raw numbers; this metadata is only for I/O).
        self._units            = units
        self._frame            = frame
    
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
    def units(self):
        """Units of the measurement values (an ``ArrayWUnits`` units object), or ``None``."""
        return self._units

    @property
    def frame(self):
        """Frame the measurements were generated in (a ``Frame`` object), or ``None``."""
        return self._frame

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
