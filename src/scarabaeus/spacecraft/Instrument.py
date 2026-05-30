# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import ArrayWUnits, Spacecraft


# -----------------------#
#    Class Definition   #
# -----------------------#
class Instrument(Spacecraft):
    """ Represents a generic instrument object. Provides functionalities related to spacecraft
    instrumentation.

    Parameters
    ----------
    name : str
        The name of the instrument.

    spice_id : int
        The SPICE ID associated with the instrument.

    measurement_type : str
        The type of measurement the instrument performs.

    Notes
    -----
    Instrument is a lightweight subclass of :class:`~scarabaeus.Spacecraft`
    that adds a ``measurement_type`` tag.  Concrete subclasses
    (:class:`~scarabaeus.Antenna`, :class:`~scarabaeus.Camera`) extend it
    with sensor-specific geometry.

    See Also
    --------
    scarabaeus.Antenna : Radiometric instrument (Doppler, range).
    scarabaeus.Camera : Optical instrument (centroiding, OpNav).
    scarabaeus.Spacecraft : Parent class providing SPICE body registration.
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self, name: str, spice_id: int | str = None, measurement_type: str = None
    ):
        # Assign properties
        self._measurement_type = measurement_type

        # Intialize parent class
        super().__init__(name, spice_id)

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def measurement_type(self):
        """String tag describing the type of measurement this instrument produces (e.g. ``"opnav"``).  ``None`` if unset."""
        return self._measurement_type

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#

    def field_of_view(self):
        """
        Return the field of view of the instrument.

        Returns
        -------
        None
            Not yet implemented in the base class; subclasses override this
            method to return instrument-specific FOV data.
        """
        pass
