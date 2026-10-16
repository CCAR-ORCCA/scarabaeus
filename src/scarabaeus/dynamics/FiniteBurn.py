# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    DynamicModel,
    Units,
    Frame,
    ArrayWUnits,
    Body,
    Spacecraft,
    StateArray,
    Maneuver,
    CelestialBody,
)
from autograd.numpy.numpy_boxes import ArrayBox
import scarabaeus.utils.NumpyWrapper as np
import re

# ------------------#
#  Generate Units  #
# ------------------#
km, sec, rad, N, m, kN, kg = Units.get_units(["km", "sec", "rad", "N", "m", "kN", "kg"])


# --------------------#
#  Class Definition  #
# --------------------#
class FiniteBurn(DynamicModel):
    """ Models the acceleration induced by a finite-duration burn with estimable
    thrust magnitude and pointing-bias parameters.

    This model represents a continuous thrust arc in which the spacecraft
    acceleration is computed from the commanded thrust magnitude, the spacecraft
    mass, and the thrust direction. In addition to the nominal maneuver
    polynomials, the model supports three estimable parameters read from the
    propagated state ``current_state``:

    * ``k_thrust`` (unitless): multiplicative scale factor on thrust magnitude.
    * ``dRA`` (rad): additive right-ascension bias applied to the nominal thrust
      direction.
    * ``dDEC`` (rad): additive declination bias applied to the nominal thrust
      direction.

    The acceleration is computed as

    .. math::

        \\mathbf{a}(t)
        = \\frac{k_{\\mathrm{thrust}}\\,T(t)}{m(t)}\\,\\hat{\\mathbf{u}}(t)

    where :math:`T(t)` is the thrust magnitude in kN and :math:`m(t)` is
    spacecraft mass in kg; dividing kN by kg gives km/s^2 directly.
    The unit direction :math:`\\hat{\\mathbf{u}}(t)` is formed
    from biased angles

    .. math::

        \\alpha(t) = \\alpha_0(t) + d\\alpha,\\qquad
        \\delta(t) = \\delta_0(t) + d\\delta,

    with

    .. math::

        \\hat{\\mathbf{u}}(\\alpha,\\delta) =
        \\begin{bmatrix}
        \\cos\\delta\\cos\\alpha \\\\
        \\cos\\delta\\sin\\alpha \\\\
        \\sin\\delta
        \\end{bmatrix}.

    The nominal angles :math:`(\\alpha_0,\\delta_0)` are obtained in one of two
    ways:

    1. If the maneuver object provides ``maneuver.ra`` and ``maneuver.dec``
       polynomials, they are evaluated directly.
    2. Otherwise, the direction is derived from the evaluated direction-component
       polynomials ``maneuver.ux``, ``maneuver.uy``, ``maneuver.uz`` by
       normalization and conversion to :math:`(\\alpha_0,\\delta_0)`.

    Parameters
    ----------
    spacecraft : Spacecraft
        Spacecraft object providing the mass model ``spacecraft.mass(epoch)``
        used in the thrust-to-acceleration conversion.
    base_frame : Frame
        Reference frame in which the burn acceleration is returned.
    maneuver : Maneuver
        Maneuver definition providing thrust magnitude and direction
        polynomials. Required fields are ``thrust`` and either the ``ra`` and
        ``dec`` polynomials, or the ``ux``, ``uy``, ``uz`` direction-component
        polynomials.

    See Also
    --------
    scarabaeus.CannonballSRP : Reads an estimable SRP scale factor from the state the same way.
    scarabaeus.Maneuver : Provides maneuver polynomials for thrust magnitude and direction.
    scarabaeus.Spacecraft : Provides mass model required for thrust-to-acceleration conversion.
    scarabaeus.ImpulsiveBurn : Instantaneous delta-v model.

    Notes
    -----
    The estimable parameters are read from the propagated state on every
    evaluation, keyed by ``(name, spice_id)`` of the owner. Expected state
    entries are of the form:

    .. code-block:: python

        ("k_thrust", 1, "estimated", "static", owner, ArrayWUnits(1.0, None))
        ("dRA",      1, "estimated", "static", owner, ArrayWUnits(0.0, "rad"))
        ("dDEC",     1, "estimated", "static", owner, ArrayWUnits(0.0, "rad"))

    The parameter resolution priority is (see :meth:`_get_param`):
    (1) ``current_state[(name, spice_id)]`` if present, (2) a value a caller
    pushed into :attr:`k_thrust_map`, :attr:`dra_map` or :attr:`ddec_map`,
    (3) an indexed entry such as ``("k_thrust1", spice_id)`` in
    ``current_state``, (4) a conservative fallback (``k_thrust=1``, ``dRA=0``,
    ``dDEC=0``).

    The model provides analytic partial derivatives:
    :math:`\\partial\\mathbf{a}/\\partial k_{\\mathrm{thrust}}` of shape (3, 1)
    and :math:`\\partial\\mathbf{a}/\\partial[d\\alpha,\\ d\\delta]` of shape
    (3, 2). The acceleration is position-independent, so
    :meth:`_compute_partial_by_position` returns zeros.

    The RA/DEC bias parameterization becomes ill-conditioned near the poles
    (:math:`\\cos\\delta \\to 0`). If near-polar thrusting is expected, a local
    tangent-plane perturbation may be preferable.

    Examples
    --------
    .. code-block:: python

        import scarabaeus as scb

        # Define estimable parameters in the state
        state = [
            ("position", 3, "estimated", "dynamic", sc, scb.ArrayWUnits([7000, 0, 0], "km")),
            ("velocity", 3, "estimated", "dynamic", sc, scb.ArrayWUnits([0, 7.5, 0], "km/s")),
            ("k_thrust", 1, "estimated", "static",  sc, scb.ArrayWUnits(1.0, None)),
            ("dRA",      1, "estimated", "static",  sc, scb.ArrayWUnits(0.0, "rad")),
            ("dDEC",     1, "estimated", "static",  sc, scb.ArrayWUnits(0.0, "rad")),
        ]
        state_vector = scb.StateArray(epoch=epochs[0], frame="J2000", origin=scb.CelestialBody("Earth"), state=state)

        # Create the model
        fb = FiniteBurn(spacecraft=sc, base_frame="J2000", maneuver=man)

        # During OD/filtering, override values through current_state:
        current_state = {
            ("k_thrust", sc.spice_id): 0.98,
            ("dRA",      sc.spice_id): 2e-4,
            ("dDEC",     sc.spice_id): -1e-4,
        }
        a = fb.compute_acceleration(position=r_sc, current_state=current_state, epoch=t, body=sc)
    """

    # --------------------#
    # region Constructor #
    # --------------------#

    def __init__(
        self,
        spacecraft: Spacecraft,
        base_frame: Frame,
        maneuver: Maneuver,
    ):
        # take in arguments
        self.maneuver = maneuver
        self._ref_frame = base_frame
        self._spacecraft = spacecraft

        # Estimable burn parameters are read from the propagated state on every
        # evaluation (see _get_param). These maps stay as an override hook for
        # callers that push a value in directly.
        self._k_thrust_map: dict[int, float] = {}
        self._dra_map: dict[int, float] = {}
        self._ddec_map: dict[int, float] = {}

    # -------------------------#
    # region ====> Properties #
    # -------------------------#

    @property
    def ref_frame(self) -> Frame:
        """
        Reference frame in which the finite-burn acceleration is expressed.
        """
        return self._ref_frame

    @property
    def spacecraft(self) -> Spacecraft:
        """
        Spacecraft providing the mass model used to convert thrust to acceleration.
        """
        return self._spacecraft

    @property
    def k_thrust_map(self) -> dict[int, float]:
        """
        Caller-supplied thrust scale-factor overrides (unitless), keyed by owner
        ``spice_id``.

        Empty by default; an entry here is used only when ``current_state`` holds
        no ``("k_thrust", spice_id)`` entry (see :meth:`_get_param`).
        """
        return self._k_thrust_map

    @property
    def dra_map(self) -> dict[int, float]:
        """
        Caller-supplied right-ascension bias overrides, in rad, keyed by owner
        ``spice_id``.

        Empty by default; an entry here is used only when ``current_state`` holds
        no ``("dRA", spice_id)`` entry (see :meth:`_get_param`).
        """
        return self._dra_map

    @property
    def ddec_map(self) -> dict[int, float]:
        """
        Caller-supplied declination bias overrides, in rad, keyed by owner
        ``spice_id``.

        Empty by default; an entry here is used only when ``current_state`` holds
        no ``("dDEC", spice_id)`` entry (see :meth:`_get_param`).
        """
        return self._ddec_map

    @property
    def has_direction_angles(self) -> bool:
        """
        True if maneuver provides ra/dec polynomials directly; otherwise direction is derived from ux/uy/uz.
        """
        return hasattr(self.maneuver, "ra") and hasattr(self.maneuver, "dec")

    @property
    def estimable_parameter_names(self) -> tuple[str, ...]:
        """
        Names of estimable parameters handled by this model.
        """
        return ("k_thrust", "dRA", "dDEC")

    # endregion => Properties  #
    # -------------------------#

    # ----------------#
    # region Methods #
    # ----------------#

    def _base_param_name(self, name: str) -> str:
        """
        Strip a trailing maneuver index from an estimable-parameter name.

        Parameters
        ----------
        name : str
            Parameter name, possibly indexed, e.g. ``"dRA2"``.

        Returns
        -------
        str
            The name with surrounding whitespace and trailing digits removed.

        Examples
        --------
        .. code-block:: python

            fb._base_param_name("k_thrust")   # "k_thrust"
            fb._base_param_name("k_thrust1")  # "k_thrust"
            fb._base_param_name("dRA2")       # "dRA"
            fb._base_param_name("dDEC3")      # "dDEC"
        """
        return re.sub(r"\d+$", "", str(name).strip())

    def _get_param(
        self, current_state: dict, body: Body, name: str, default: float
    ) -> float:
        """
        Resolve an estimable burn parameter for one owner.

        The lookup order is:

        1. ``current_state[(name, spice_id)]``, so an estimated burn parameter
           takes effect;
        2. a value a caller pushed into :attr:`k_thrust_map`, :attr:`dra_map` or
           :attr:`ddec_map`;
        3. the first ``current_state`` entry of the same owner whose name with its
           maneuver index stripped equals ``name``, e.g. ``("dRA2", spice_id)``;
        4. ``default``.

        Parameters
        ----------
        current_state : dict
            The propagated state, keyed by ``(name, spice_id)``.
        body : Body
            Owner of the parameter. A body without a ``spice_id`` always gets
            ``default``.
        name : str
            Base parameter name: ``"k_thrust"``, ``"dRA"`` or ``"dDEC"``.
        default : float
            Value returned when no entry is found.

        Returns
        -------
        float or ArrayBox
            The parameter value, or the traced box when autograd is tracing (see
            :meth:`~scarabaeus.DynamicModel._scalar_from_state`).
        """
        spice_id = getattr(body, "spice_id", None)

        if spice_id is not None:
            # 1) propagated state
            key = (name, spice_id)
            if key in current_state:
                val = current_state[key]
                return self._scalar_from_state(val)

            # 2) caller-supplied override
            if name == "k_thrust" and spice_id in self._k_thrust_map:
                return self._k_thrust_map[spice_id]
            if name == "dRA" and spice_id in self._dra_map:
                return self._dra_map[spice_id]
            if name == "dDEC" and spice_id in self._ddec_map:
                return self._ddec_map[spice_id]

            # also allow indexed names in current_state, e.g. k_thrust1, dRA2, dDEC3
            for (k_name, k_sid), val in current_state.items():
                if k_sid != spice_id:
                    continue
                if self._base_param_name(k_name) != name:
                    continue
                return self._scalar_from_state(val)

        # 3) fallback
        return default

    def _evaluate_polynomial(self, values, t):
        """
        Evaluates a 6th-order polynomial or returns a scalar value with units.

        Parameters
        ----------
        values : ArrayWUnits
            An object containing the polynomial coefficients or a scalar value, along with associated units.
            If `values.values` is a scalar (float or int), it is returned as is.
            If `values.values` is a sequence (list, tuple, np.ndarray, or ArrayBox) of length 7, it is interpreted as
            the coefficients of a 6th-order polynomial in descending order of degree.
        t : float or int
            The value at which to evaluate the polynomial.

        Returns
        -------
        ArrayWUnits
            The evaluated polynomial (or scalar value) with the same units as the input.

        Raises
        ------
        ValueError
            If `values.values` is a sequence but does not contain exactly 7 elements.
        """
        match values.size:
            case 1:
                # Scalar input — return directly.
                return values
            case 7:
                # Evaluate the 6th-order polynomial: y = c6*t^6 + c5*t^5 + ... + c0
                # values.values is in descending order of degree (index 0 = t^6
                # coefficient, index 6 = constant term), so values.values[i]
                # pairs directly with exponent (6 - i) -- no reversal needed.
                # Coefficients carry units [base_unit/s^i]; multiplication by t^i [s^i]
                # cancels the time dependence, so the result always carries the base unit.
                y = sum(values.values[i] * t ** (6 - i) for i in range(7))
                base_unit = (
                    values.units[-1]
                    if isinstance(values.units, (list, np.ndarray))
                    else values.units
                )
                return ArrayWUnits(y, base_unit)
            case _:
                raise ValueError(
                    f"Expected scalar or 6th order polynomial. Received size {values.size}."
                )

    def _u_hat_from_ra_dec(self, ra: float, dec: float) -> np.ndarray:
        """
        Compute the unit vector in Cartesian coordinates from right ascension and declination.

        Parameters
        ----------
        ra : float
            Right ascension angle in radians.
        dec : float
            Declination angle in radians.

        Returns
        -------
        np.ndarray
            A 3-element numpy array representing the unit vector [x, y, z] in Cartesian coordinates.
        """
        cdec = np.cos(dec)
        return np.array([cdec * np.cos(ra), cdec * np.sin(ra), np.sin(dec)])

    def _d_u_hat_d_ra(self, ra: float, dec: float) -> np.ndarray:
        """
        Compute the partial derivative of the unit vector with respect to right ascension (ra).

        Given the right ascension (ra) and declination (dec) angles, this method returns the derivative
        of the unit vector in Cartesian coordinates with respect to ra.

        Parameters
        ----------
        ra : float
            Right ascension angle in radians.
        dec : float
            Declination angle in radians.

        Returns
        -------
        np.ndarray
            A 3-element array representing the derivative of the unit vector with respect to ra.
        """
        cdec = np.cos(dec)
        return np.array([-cdec * np.sin(ra), cdec * np.cos(ra), 0.0])

    def _d_u_hat_d_dec(self, ra: float, dec: float) -> np.ndarray:
        """
        Compute the partial derivative of the unit vector with respect to declination.

        Given right ascension (ra) and declination (dec), this method returns the derivative
        of the unit vector in spherical coordinates with respect to the declination angle.

        Parameters
        ----------
        ra : float
            Right ascension angle in radians.
        dec : float
            Declination angle in radians.

        Returns
        -------
        np.ndarray
            A 3-element array representing the derivative of the unit vector with respect to declination.
        """
        sdec = np.sin(dec)
        cdec = np.cos(dec)
        return np.array([-sdec * np.cos(ra), -sdec * np.sin(ra), cdec])

    def _nominal_ra_dec(self, epoch: float) -> tuple[float, float]:
        """
        Nominal thrust RA/DEC at ``epoch``, memoized per epoch.

        :meth:`compute_acceleration` and the thrust partials all request the nominal
        pointing at the same epoch within one RHS step, so the (epoch-only)
        polynomial evaluation is computed once and reused. Delegates the actual
        computation to :meth:`_compute_nominal_ra_dec`.

        Parameters
        ----------
        epoch : float
            Epoch at which to evaluate the nominal thrust direction, in ephemeris
            time (seconds past J2000).

        Returns
        -------
        tuple of float
            ``(ra, dec)`` nominal right ascension and declination in radians.
        """
        if self.__dict__.get("_radec_epoch") != epoch:
            self._radec_cache = self._compute_nominal_ra_dec(epoch)
            self._radec_epoch = epoch
        return self._radec_cache

    def _thrust_at(self, epoch: float) -> ArrayWUnits:
        """
        Nominal thrust magnitude at ``epoch``, memoized per epoch.

        Evaluated by :meth:`compute_acceleration` and the thrust partials at the
        same epoch within one RHS step, so the thrust polynomial is evaluated once
        and reused.

        Parameters
        ----------
        epoch : float
            Epoch at which to evaluate the thrust polynomial, in ephemeris time
            (seconds past J2000).

        Returns
        -------
        ArrayWUnits
            Nominal thrust magnitude at ``epoch`` (force units, e.g. Newtons).
        """
        if self.__dict__.get("_thrust_epoch") != epoch:
            self._thrust_cache = self._evaluate_polynomial(self.maneuver.thrust, epoch)
            self._thrust_epoch = epoch
        return self._thrust_cache

    def _compute_nominal_ra_dec(self, epoch: float) -> tuple[float, float]:
        """
        Evaluate the nominal thrust direction as right ascension / declination.

        The nominal pointing is taken directly from the ``maneuver.ra`` / ``maneuver.dec``
        polynomials when present; otherwise it is derived by evaluating the
        ``maneuver.(ux, uy, uz)`` direction polynomials, normalizing, and converting
        the unit direction to angles.

        Parameters
        ----------
        epoch : float
            Epoch at which to evaluate the nominal thrust direction, in ephemeris
            time (seconds past J2000).

        Returns
        -------
        tuple of float
            ``(ra, dec)`` nominal right ascension and declination in radians.

        Raises
        ------
        ValueError
            If the derived direction vector ``(ux, uy, uz)`` has zero norm, so that
            RA/DEC cannot be defined.
        """
        if hasattr(self.maneuver, "ra") and hasattr(self.maneuver, "dec"):
            ra0 = self._evaluate_polynomial(self.maneuver.ra, epoch).values
            dec0 = self._evaluate_polynomial(self.maneuver.dec, epoch).values
            return ra0, dec0

        # fallback: derive from ux,uy,uz
        ux = float(self._evaluate_polynomial(self.maneuver.ux, epoch).values)
        uy = float(self._evaluate_polynomial(self.maneuver.uy, epoch).values)
        uz = float(self._evaluate_polynomial(self.maneuver.uz, epoch).values)
        u = np.array([ux, uy, uz])

        # Normalize to be safe (polynomials may drift slightly)
        un = np.linalg.norm(u)
        if un == 0:
            raise ValueError(
                "FiniteBurn: direction vector norm is zero; cannot derive RA/DEC."
            )
        u = u / un

        ra0 = np.arctan2(u[1], u[0])
        dec0 = np.arcsin(np.clip(u[2], -1.0, 1.0))
        return float(ra0), float(dec0)

    def compute_acceleration(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Compute the acceleration vector due to finite burn thrust.

        Parameters
        ----------
        position : np.ndarray
            Position vector (not directly used in calculation).
        current_state : dict
            Dictionary containing estimable parameters including thrust scaling factor
            and thrust direction offsets.
        epoch : float
            Epoch at which to evaluate thrust and spacecraft mass.
        body : CelestialBody
            Celestial body reference for parameter lookup.

        Returns
        -------
        np.ndarray
            Acceleration vector in km/s^2 due to thrust.

        Notes
        -----
        The method computes thrust acceleration using a cannonball-style model with
        estimable parameters:
        - k_thrust: thrust magnitude scaling factor (default 1.0)
        - dRA: right ascension offset (default 0.0 rad)
        - dDEC: declination offset (default 0.0 rad)

        The thrust direction is specified in RA/DEC coordinates and combined with
        the nominal thrust direction to obtain the final acceleration vector.
        """

        # thrust magnitude
        T = self._thrust_at(epoch)  # N
        m_sc = self._spacecraft.mass(epoch)  # kg

        # estimable params (Cannonball-style)
        k_thrust = self._get_param(
            current_state, body, "k_thrust", default=1.0
        )  # unitless
        dRA = self._get_param(current_state, body, "dRA", default=0.0)  # rad
        dDEC = self._get_param(current_state, body, "dDEC", default=0.0)  # rad

        ra0, dec0 = self._nominal_ra_dec(epoch)
        ra = ra0 + dRA
        dec = dec0 + dDEC

        u_hat = self._u_hat_from_ra_dec(ra, dec)

        a_norm = (k_thrust * T.values) / m_sc.values  # kN/kg = km/s^2
        return a_norm * u_hat  # km/s^2

    def _compute_partial_by_position(
        self, position, current_state, epoch, body
    ) -> np.ndarray:
        """
        Compute the partial of the burn acceleration with respect to position.

        Finite-burn thrust does not depend on spacecraft position, so this
        partial is identically zero.

        Parameters
        ----------
        position : np.ndarray
            Current position vector (3,) in km. Unused.
        current_state : dict
            Current state parameters. Unused.
        epoch : float
            Current epoch in TDB seconds. Unused.
        body : object
            Target body. Unused.

        Returns
        -------
        np.ndarray
            (3, 3) zero matrix.
        """

        return np.zeros((3, 3))

    def _compute_partial_by_k_thrust(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: Body,
    ) -> np.ndarray:
        """
        Compute the partial of the burn acceleration with respect to ``k_thrust``.

        .. math::

            \\frac{\\partial\\mathbf{a}}{\\partial k_{\\mathrm{thrust}}}
            = \\frac{T(t)}{m(t)}\\,\\hat{\\mathbf{u}}(\\alpha_0 + d\\alpha,\\,
            \\delta_0 + d\\delta)

        Parameters
        ----------
        position : np.ndarray
            Current position vector of the spacecraft. Unused.
        current_state : dict
            The propagated state, from which ``dRA`` and ``dDEC`` are read.
        epoch : float
            Epoch at which the thrust and spacecraft mass are evaluated, in TDB
            seconds.
        body : Body
            Owner of the burn parameters.

        Returns
        -------
        np.ndarray
            (3, 1) partial, in km/s^2.
        """
        T = float(self._thrust_at(epoch).values)  # kN
        m_sc = float(self._spacecraft.mass(epoch).values)

        ra0, dec0 = self._nominal_ra_dec(epoch)
        dRA = self._get_param(current_state, body, "dRA", default=0.0)
        dDEC = self._get_param(current_state, body, "dDEC", default=0.0)

        u_hat = self._u_hat_from_ra_dec(ra0 + dRA, dec0 + dDEC)
        return ((T / m_sc) * u_hat).reshape(3, 1)

    def _compute_partial_by_ra_dec_bias(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: Body,
    ) -> np.ndarray:
        """
        Compute the partials of the burn acceleration with respect to the RA and
        DEC biases.

        .. math::

            \\frac{\\partial\\mathbf{a}}{\\partial[d\\alpha,\\ d\\delta]}
            = \\frac{k_{\\mathrm{thrust}}\\,T(t)}{m(t)}
            \\left[\\frac{\\partial\\hat{\\mathbf{u}}}{\\partial\\alpha},\\
            \\frac{\\partial\\hat{\\mathbf{u}}}{\\partial\\delta}\\right]

        evaluated at the biased angles :math:`\\alpha_0 + d\\alpha` and
        :math:`\\delta_0 + d\\delta`.

        Parameters
        ----------
        position : np.ndarray
            Current position vector of the spacecraft. Unused.
        current_state : dict
            The propagated state, from which ``k_thrust``, ``dRA`` and ``dDEC`` are
            read.
        epoch : float
            Epoch at which the thrust, mass and nominal angles are evaluated, in
            TDB seconds.
        body : Body
            Owner of the burn parameters.

        Returns
        -------
        np.ndarray
            (3, 2) Jacobian, in km/s^2/rad: the first column is the partial with
            respect to ``dRA``, the second with respect to ``dDEC``.
        """

        T = float(self._thrust_at(epoch).values)  # kN
        m_sc = float(self._spacecraft.mass(epoch).values)

        k_thrust = self._get_param(current_state, body, "k_thrust", default=1.0)
        dRA = self._get_param(current_state, body, "dRA", default=0.0)
        dDEC = self._get_param(current_state, body, "dDEC", default=0.0)

        ra0, dec0 = self._nominal_ra_dec(epoch)
        ra = ra0 + dRA
        dec = dec0 + dDEC

        a_scale = k_thrust * T / m_sc

        du_dra = self._d_u_hat_d_ra(ra, dec)
        du_ddec = self._d_u_hat_d_dec(ra, dec)

        J = a_scale * np.column_stack([du_dra, du_ddec])  # (3,2)
        return J

    # Keep your existing ones if you still use them elsewhere:
    def _compute_partial_by_thrust(
        self, position, current_state, epoch, body
    ) -> np.ndarray:
        """
        Compute the partial of the burn acceleration with respect to the thrust
        magnitude :math:`T`.

        .. math::

            \\frac{\\partial\\mathbf{a}}{\\partial T}
            = \\frac{k_{\\mathrm{thrust}}}{m(t)}\\,\\hat{\\mathbf{u}}(\\alpha_0 + d\\alpha,\\,
            \\delta_0 + d\\delta)

        Parameters
        ----------
        position : np.ndarray
            Current position vector of the spacecraft. Unused.
        current_state : dict
            The propagated state, from which ``k_thrust``, ``dRA`` and ``dDEC`` are
            read.
        epoch : float
            Epoch at which the mass and nominal angles are evaluated, in TDB
            seconds.
        body : Body
            Owner of the burn parameters.

        Returns
        -------
        np.ndarray
            (3,) partial, in km/s^2 per kN (that is, 1/kg).
        """

        ra0, dec0 = self._nominal_ra_dec(epoch)
        dRA = self._get_param(current_state, body, "dRA", default=0.0)
        dDEC = self._get_param(current_state, body, "dDEC", default=0.0)
        u_hat = self._u_hat_from_ra_dec(ra0 + dRA, dec0 + dDEC)

        m_sc = float(self._spacecraft.mass(epoch).values)
        k_thrust = self._get_param(current_state, body, "k_thrust", default=1.0)
        return (k_thrust / m_sc) * u_hat  # kN/kg = km/s^2, (3,)

    def _compute_partial_by_direction(
        self, position, current_state, epoch, body
    ) -> np.ndarray:
        """
        Compute the partials of the burn acceleration with respect to the thrust
        direction components.

        Treating the three components of :math:`\\hat{\\mathbf{u}}` as independent
        gives :math:`\\partial\\mathbf{a}/\\partial\\hat{\\mathbf{u}} =
        (k_{\\mathrm{thrust}}\\,T/m)\\,\\mathbf{I}_3`.

        Parameters
        ----------
        position : np.ndarray
            Current position vector of the spacecraft. Unused.
        current_state : dict
            The propagated state, from which ``k_thrust`` is read.
        epoch : float
            Epoch at which the thrust and mass are evaluated, in TDB seconds.
        body : Body
            Owner of the burn parameters.

        Returns
        -------
        np.ndarray
            (3, 3) diagonal matrix, in km/s^2.
        """
        T = float(self._thrust_at(epoch).values)
        m_sc = float(self._spacecraft.mass(epoch).values)
        k_thrust = self._get_param(current_state, body, "k_thrust", default=1.0)
        a_norm = (k_thrust * T) / m_sc
        return a_norm * np.eye(3)
