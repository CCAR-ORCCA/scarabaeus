# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    Body,
    Frame,
    Units,
    ArrayWUnits,
    EpochArray,
    SpiceManager,
    nPlateModel,
    DynamicalUtils,
)
import scarabaeus.utils.NumpyWrapper as np
import spiceypy as spice
import warnings

# ------------------#
#  Generate Units  #
# ------------------#
m, km, sec = Units.get_units(["m", "km", "sec"])


# --------------------#
#  Class Definition  #
# --------------------#
class Spacecraft(Body):
    """ Represents a spacecraft body.

    Parameters
    ----------
    name : str
        The name of the spacecraft.

    spice_id : int | str
        The NAIF ID code of the object. May be given as an integer ID or as a string.

    tot_mass : ArrayWUnits or SCBPolynomial, optional
        The total (wet) mass of the spacecraft, passed to
        :class:`~scarabaeus.Body` as its ``mass``: an
        :class:`~scarabaeus.ArrayWUnits` for a constant mass or an
        :class:`~scarabaeus.SCBPolynomial` of time. Recommend expressed in
        units of kilograms. Defaults to ``None``.

    area : ArrayWUnits, optional
        The cross-sectional area of the spacecraft, used by
        :class:`~scarabaeus.CannonballSRP`. Recommend expressed in units of
        :math:`m^2`. Defaults to ``None``.

    ref_coeff : float, optional
        The reflectivity coefficient of the spacecraft, used by
        :class:`~scarabaeus.CannonballSRP`. Defaults to ``None``.

    n_plate_model : nPlateModel, optional
        The n-plate model that describes the shape and physical parameters of
        the spacecraft, used by :class:`~scarabaeus.nPlateSRP`. Defaults to
        ``None``.

    thruster : Actuator, optional
        The thruster attached to the spacecraft. Defaults to ``None``.

    instrument_list : list[Instrument], optional
        The Instrument objects attached to the spacecraft (cameras, antennas).
        Defaults to ``[]``.

    body_fixed_frame_id : int, optional
        The SPICE ID of the spacecraft-fixed frame. Defaults to ``None``.

    attitudeMode : str, optional
        How the surface-force models (:class:`~scarabaeus.nPlateSRP`,
        :class:`~scarabaeus.ThermalRadiation`,
        :class:`~scarabaeus.AntennaRadiationPressure`) orient the
        spacecraft-fixed frame. One of
        :attr:`DynamicalUtils.ATTITUDE_MODES <scarabaeus.DynamicalUtils.ATTITUDE_MODES>`:

        * ``"CK"``: from the loaded C-kernel, whose frame ID is
          ``spice_id * 1000`` (:meth:`get_attitude`).
        * ``"inertialFixedAttitude"``: the constant rotation
          ``fixedInertialAttitude``.
        * ``"nadirPointingSun"``: the :math:`-X` axis points at the Sun and
          :math:`+Z` along the heliocentric orbit normal, from the heliocentric
          position and velocity
          (:meth:`~scarabaeus.DynamicalUtils.nadirPointingSun`).

        Any other value raises
        :class:`~scarabaeus.DynamicalUtils.InvalidAttitudeModeError` when one
        of those models first needs the attitude. Models that need none
        (gravity, :class:`~scarabaeus.CannonballSRP`, an
        :class:`~scarabaeus.nPlateModel` whose plates all sit in their own
        SPICE frames) accept the default. Defaults to ``None``.

    fixedInertialAttitude : numpy.ndarray, optional
        The ``(3, 3)`` spacecraft-fixed to inertial rotation, required by
        ``attitudeMode="inertialFixedAttitude"``. Defaults to ``None``.

    nadirPointingAxis : numpy.ndarray, optional
        The spacecraft-fixed axis meant to point at the Sun. It is stored and
        serialized only: the ``"nadirPointingSun"`` attitude always points
        :math:`-X` at the Sun. Defaults to ``None``.

    See Also
    --------
    scarabaeus.nPlateModel : Geometric surface model for SRP computation.
    scarabaeus.DynamicalUtils : Turns ``attitudeMode`` into the spacecraft-fixed to inertial rotation.
    scarabaeus.Instrument : Generic instrument class (cameras, antennas).
    scarabaeus.Body : Parent class providing SPICE ephemeris object management.

    Notes
    -----
    Spacecraft is the primary vehicle representation in Scarabaeus.  It holds
    physical properties needed for force-model evaluation (mass, area,
    reflectivity), the optional :class:`~scarabaeus.nPlateModel` for
    high-fidelity SRP modeling and the attitude the surface-force models
    orient it with.  Instruments are registered via :meth:`add_instrument`.

    A propagation with maneuvers integrates the mass and replaces
    ``tot_mass`` with a polynomial fit of the mass history, which
    :meth:`~scarabaeus.Body.mass` then evaluates;
    :meth:`~scarabaeus.Body.reset_mass_history` restores ``tot_mass``.
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        name: str,
        spice_id: int | str,
        tot_mass: ArrayWUnits = None,
        area: ArrayWUnits = None,
        ref_coeff: float = None,
        n_plate_model=None,
        thruster=None,
        instrument_list: list = [],
        body_fixed_frame_id=None,
        attitudeMode=None,
        fixedInertialAttitude=None,
        nadirPointingAxis=None,
    ):
        # parent class initialization
        super().__init__(name, spice_id, tot_mass)

        # define class properties
        self.area = area
        self._ref_coeff = ref_coeff
        self._n_plate_model = n_plate_model
        self._instrument_list = instrument_list
        self._thruster = thruster
        self.body_fixed_frame_id = body_fixed_frame_id
        self.attitudeMode = attitudeMode
        self.fixedInertialAttitude = fixedInertialAttitude
        self.nadirPointingAxis = nadirPointingAxis

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def name(self) -> str:
        """
        The name of the spacecraft.
        """
        return self._name

    @property
    def area(self) -> ArrayWUnits:
        """
        The cross-sectional area of the spacecraft. Expressed in units of square meters.
        """
        return self._area

    @area.setter
    def area(self, input_val: ArrayWUnits):
        ## input_validation and conditioning
        # must be an AWU

        # if input_val is not None:
        #     if not isinstance(input_val, ArrayWUnits):
        #         not_unit_err = ('Argument [area] must be given as an ArrayWUnits object '
        #                         f'with Dimensions of square length. Received {type(input_val)}.')
        #         raise TypeError(not_unit_err)

        #     # given area must be some unit of square length
        #     if input_val.units.dimensions.name != 'Area':
        #         not_area_err = ('Argument [area] must be given as an ArrayWUnits object '
        #                         'with Dimensions of square length. Received '
        #                         f'Dimensions of {input_val.units.dimensions}.')
        #         raise ValueError(not_area_err)

        #     # ensure units are square meters
        #     if input_val.units != m**2:
        #         input_val = input_val.convert_to(m**2)

        # # set to 0 m^2 if given None
        # else:
        #     input_val = ArrayWUnits(0, m**2)

        ## valid input -> assign property
        self._area = input_val

    @property
    def ref_coeff(self) -> float:
        """
        The reflectivity coefficient of the spacecraft.
        """
        return self._ref_coeff

    @property
    def n_plate_model(self) -> nPlateModel:
        """
        The N-Plate model representing the spacecraft's geometry.
        """
        return self._n_plate_model

    @property
    def thruster(self):
        """Thruster actuator attached to this spacecraft, or ``None`` if not set."""
        return self._thruster

    @property
    def instrument_list(self) -> list:
        """
        The list of instruments attached to the spacecraft.
        """
        return self._instrument_list

    @property
    def body_fixed_frame_id(self) -> int:
        """
        The SPICE ID assigned to the spacecraft's body-fixed frame.
        """
        return self._body_fixed_frame_id

    @body_fixed_frame_id.setter
    def body_fixed_frame_id(self, input_val):
        # given ID must be an integer
        if (not isinstance(input_val, int)) and (input_val is not None):
            not_int_err = (
                "Argument [body_fixed_frame_id] must be given as an integer. "
                f"Received {type(input_val)}."
            )
            raise TypeError(not_int_err)

        # valid input -> assign property
        self._body_fixed_frame_id = input_val

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    def __repr__(self):
        """
        Overloading the printable representation operator for a Spacecraft object.
        """
        return f"{self.spice_id} {self.name}"

    def get_attitude(
        self, from_frame: Frame, epoch: EpochArray, tol: int = 0
    ) -> np.ndarray:
        """
        Computes the rotation from a given inertial frame to the spacecraft-fixed
        frame, from the C-kernel of frame ID ``spice_id * 1000``.

        Parameters
        ----------
        from_frame : Frame
            The intertial reference frame for which the spacecraft-fixed frame will be
            transformed from.

        epoch : EpochArray
            The epoch at which the transformation will be performed. Given in ephemeris
            time (seconds past J2000).

        tol : int, optional
            The time tolerance for retrieving attitude and angular velocity from the
            specified spacecraft clock time, given in sclock ticks. Defaults to ``0``.

        Returns
        -------
        quat : numpy.ndarray
            The ``(4,)`` SPICE quaternion (``spice.m2q``) of the rotation from
            ``from_frame`` to spacecraft-fixed coordinates.

        Notes
        -----
        The ``epoch`` argument is converted from ephemeris time in seconds to continuous encoded
        spacecraft clock ticks.
        """
        # ------------------#
        # input validation #
        # ------------------#
        # must be correct object types
        if not isinstance(from_frame, Frame):
            not_frame_err = (
                "Argument [from_frame] must be given as a Frame object. "
                f"Received {type(from_frame)}."
            )
            raise TypeError(not_frame_err)

        # if not isinstance(epoch, EpochArray):
        #     not_EA_err = ('Argument [epoch] must be given as an EpochArray object. '
        #                   f'Received {type(epoch)}.')
        #     raise TypeError(not_EA_err)

        if not isinstance(tol, int):
            not_int_err = (
                "Argument [tol] must be given as an integer. " f"Received {type(tol)}."
            )
            raise TypeError(not_int_err)

        # --------------#
        # get attitude #
        # --------------#
        # call SpiceManager
        dcm, _, _ = SpiceManager.get_attitude(self.spice_id, epoch, from_frame, tol)
        return spice.m2q(dcm)

    def inertial_to_body_DCM_nadirPointingSun(
        self,
        origin_name,
        frame,
        sun2body_pos,  # ArrayWUnits, body wrt SUN position
        current_state: dict,  # may contain keys like ('velocity', spice_id) or 'velocity'
        epoch,
        body,
    ):
        """
        Computes the Direction Cosine Matrix (DCM) that points the body frame's
        -X axis toward the Sun (nadir-to-sun), using inertial quantities.

        .. deprecated:: 2026.0.0
            Use :meth:`~scarabaeus.DynamicalUtils.nadirPointingSun`, which
            returns the spacecraft-fixed to inertial DCM (the transpose of this
            one) and, on request, its position and velocity derivatives. This
            method now delegates to it.

        Parameters
        ----------
        origin_name : str
            Name of the origin at which `current_state` is defined (e.g., 'EARTH', 'SUN').
        frame : Frame or str
            Frame in which `current_state` is expressed (e.g., J2000).
        sun2body_pos : ArrayWUnits
            Position of the body w.r.t. SUN (km).
        current_state : dict
            State of the body w.r.t. the origin, keyed either by
            ``('velocity', spice_id)`` or by ``'velocity'``, with an array or
            ``ArrayWUnits`` value (and similarly for ``'position'`` if ever
            needed).
        epoch : float
            Epoch (TDB seconds past J2000) at which the state is computed.
        body : CelestialBody or Spacecraft
            The body whose state we are extracting from `current_state` (used for spice_id lookup).

        Returns
        -------
        DCM_inertial_to_body : (3,3) ndarray
            DCM that rotates inertial vectors into the body-fixed frame realizing the nadir-to-sun pointing.

        Raises
        ------
        DynamicalUtils.MissingVelocityError
            If `current_state` holds no velocity for `body` (a ``KeyError``).
        """
        warnings.warn(
            "Spacecraft.inertial_to_body_DCM_nadirPointingSun is deprecated; "
            "use DynamicalUtils.nadirPointingSun, which returns the "
            "spacecraft-fixed to inertial DCM (the transpose of this one).",
            DeprecationWarning,
            stacklevel=2,
        )
        vel = np.asarray(
            DynamicalUtils.velocity_from_state(current_state, body), dtype=float
        )
        sun2body_vel = (
            DynamicalUtils.sun_to_origin_velocity(origin_name, frame, epoch) + vel
        )
        dcm_body2inertial, _, _ = DynamicalUtils.nadirPointingSun(
            np.asarray(sun2body_pos.values, dtype=float), sun2body_vel
        )
        return dcm_body2inertial.T

    def add_instrument(self, instruments) -> None:
        """
        Adds an instrument or instruments to the spacecraft.

        Parameters
        ----------
        instruments : list of Instrument
            The instruments to be attached to the spacecraft.

        Returns
        -------
        None
        """
        for instr in instruments:
            self._instrument_list.append(instr)

    def get_dependent_spice_ids(self):
        """
        Collects the SPICE ID's of all attached frames and instruments into a
        dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        ids : dict
            A dictionary containing the keys:

            * ``'instruments'`` : list
              The SPICE ID's of all attached :class:`Instrument` objects.
            * ``'plates'`` : list
              The SPICE ID's of all C-frames for CK type panels defined in the
              :class:`nPlateModel` if one has been assigned to the Spacecraft.
        """
        # first go through instrument list
        instr_ids = []
        for instr in self.instrument_list:
            instr_ids.append(instr.spice_id)

        # now check n plate model for any CK objects if s/c has one
        plate_ids = []
        if self.n_plate_model:
            ck_id_inds = [
                i for i, e in enumerate(self.n_plate_model.plate_types) if e == "CK"
            ]
            for i in ck_id_inds:
                plate_ids.append(self.n_plate_model.plate_ids[i])

        # combine into dict and return
        ids = {"instruments": instr_ids, "plates": plate_ids}

        return ids
