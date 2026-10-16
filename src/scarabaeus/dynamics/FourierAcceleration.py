# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    Body,
    EpochArray,
    Spacecraft,
    Frame,
    DynamicModel,
    SpiceManager,
    DynamicalUtils,
)
from scarabaeus import constants

import scarabaeus.utils.NumpyWrapper as np



class FourierAcceleration(DynamicModel):
    """Models an empirical acceleration as a truncated Fourier series.

    Each axis of a chosen frame carries its own series in a phase angle
    :math:`\\theta`,

    .. math::

        a_j(\\theta) = A_{j,0}
            + \\sum_{k=1}^{N} \\left[
                A_{j,k}\\cos(k\\theta) + B_{j,k}\\sin(k\\theta)
            \\right],
        \\qquad j \\in \\{1, 2, 3\\}

    and the resulting frame-local vector is rotated into the propagation frame,

    .. math::

        \\mathbf a_F = \\mathbf T \\, \\mathbf a^{(f)}
                     = \\mathbf T \\, \\mathbf M(\\theta) \\, \\mathbf p .

    Degree :math:`N = 0` leaves only the constant terms, so the model degenerates
    to a **constant empirical acceleration** in the chosen frame — the usual
    "constant along-track acceleration" is ``degree=0``, ``frame="RTN"``.

    Because the acceleration is *linear* in the coefficients, the partial that
    the filter needs is the design matrix itself,

    .. math::

        \\frac{\\partial \\mathbf a_F}{\\partial \\mathbf p}
            = \\mathbf T \\, \\mathbf M(\\theta),

    which is exact rather than a difference approximation.

    Parameters
    ----------
    origin : Body
        Propagation origin.
    spacecraft : Spacecraft
        Spacecraft the empirical acceleration acts on.
    base_frame : Frame
        Reference frame of the propagator.
    degree : int, optional
        Highest harmonic :math:`N` retained. ``0`` (the default) gives a
        constant acceleration.
    frame : str, optional
        ``"inertial"``, ``"body_fixed"``, or ``"RTN"``. Defaults to
        ``"inertial"``.
    angle : str, optional
        ``"time"``, ``"argument_of_latitude"``, or ``"solar_longitude"``.
        Defaults to ``"time"``. Ignored when ``degree == 0``, which has no
        phase dependence.
    period : float, optional
        Repeat period of the series in seconds, for ``angle="time"``. Required
        when ``degree >= 1`` and the phase is time-based.
    t0 : EpochArray or float, optional
        Epoch of zero phase for ``angle="time"``, in ephemeris seconds (TDB) or
        as an ``EpochArray``. In a :class:`~scarabaeus.MissionSequence` this
        must be the same epoch for every leg, since the legs share one
        coefficient state and a moving origin would give the phase a different
        meaning in each of them.
    body_fixed_frame : str or Frame, optional
        Frame the coefficients are expressed in, for ``frame="body_fixed"``.
    relative_body : str or Body, optional
        Body the orbit is described about, when that is not the propagation
        origin — a lunar RTN frame while propagating an Earth-centred arc, say.
        Governs the RTN triad *and* the orbital phase angles together: they
        describe the same orbit, so anchoring them to different bodies would
        give a frame and a phase that disagree. Defaults to the propagation
        origin.
    coefficients : np.ndarray, optional
        Default coefficient vector, used when the propagated state carries no
        ``a_fourier`` entry. This is what makes the model usable as a plain
        force model — a known constant or periodic acceleration applied without
        estimating anything.

    See Also
    --------
    scarabaeus.FirstOrderGaussMarkov : Stochastic empirical acceleration.
    scarabaeus.PiecewiseFirstOrderGaussMarkov : Batch-segmented stochastic variant.
    scarabaeus.StateArray : State vector used to initialize the model.
    scarabaeus.DynamicModel : Abstract base class for all force model components.

    Notes
    -----
    The ``frame`` argument selects the triad the coefficients are expressed in:

    ``"inertial"``
        Coefficients are expressed directly in the propagation frame,
        :math:`\\mathbf T = \\mathbf I`.
    ``"body_fixed"``
        Coefficients are expressed in a rotating body-fixed frame and rotated
        out through SPICE, :math:`\\mathbf T = \\mathbf R_{bf\\to rf}(t)`. Use
        this for an acceleration tied to the central body's geography or
        illumination, such as an outgassing jet on a rotating comet.
    ``"RTN"``
        Coefficients are expressed in the spacecraft's local orbital triad
        (radial / along-track / cross-track, also called RSW or RIC),

        .. math::

            \\hat{\\mathbf R} = \\frac{\\mathbf r}{r}, \\quad
            \\hat{\\mathbf N} = \\frac{\\mathbf r \\times \\mathbf v}
                                     {\\|\\mathbf r \\times \\mathbf v\\|},
            \\quad
            \\hat{\\mathbf T} = \\hat{\\mathbf N} \\times \\hat{\\mathbf R},

        with :math:`\\mathbf T = [\\hat{\\mathbf R}\\;\\hat{\\mathbf T}\\;
        \\hat{\\mathbf N}]`. This triad depends on the state, so it contributes
        to :math:`\\partial \\mathbf a / \\partial \\mathbf r` and
        :math:`\\partial \\mathbf a / \\partial \\mathbf v`.

    The ``angle`` argument selects the phase :math:`\\theta`:

    ``"time"``
        :math:`\\theta = 2\\pi (t - t_0) / T`, with ``period`` the repeat time
        :math:`T` in seconds. Independent of the state, so it adds nothing to
        the position and velocity partials. Required only when ``degree >= 1``.
    ``"argument_of_latitude"``
        :math:`\\theta = u`, the angle from the ascending node to the spacecraft
        measured in the orbit plane. The natural variable for a per-revolution
        empirical acceleration. Undefined for an exactly equatorial orbit, where
        the node direction vanishes.
    ``"solar_longitude"``
        :math:`\\theta = \\lambda_s`, the longitude of the **Sun** in the
        coefficient frame. This is the phase McMahon and Scheeres expand solar
        radiation pressure in [3]_: in a body-fixed frame it is exactly their
        :math:`\\lambda_s`, whose frequency is the body's rotation relative to
        the Sun; in RTN it is the Sun-relative in-plane angle the CODE/Bern GPS
        model writes as :math:`u - u_0`. Nothing is supplied by the caller — the
        geometry sets the frequency. Undefined when the Sun lies along the
        frame's third axis.

    The difference between the last two is where the phase is measured *from*.
    A constant offset is absorbed by the coefficients, so over a short arc the
    two are equivalent; the Sun's direction drifts by about a degree a day,
    though, so on a long arc only ``"solar_longitude"`` keeps a Sun-driven
    effect at a fixed phase.

    A series in true anomaly :math:`\\nu` is deliberately not offered: since
    :math:`u = \\nu + \\omega`, a Fourier series in :math:`u` spans exactly the
    same function space for a fixed argument of periapsis, and :math:`u` stays
    well defined on the near-circular orbits where empirical accelerations are
    most often used, whereas :math:`\\nu` does not.

    The estimated state is one stacked ``("a_fourier", spice_id)`` entry of
    length :math:`3(2N+1)`, ordered axis-major and, within an axis, by harmonic:

    .. code-block:: text

        [ A_1,0  A_1,1  B_1,1  ...  A_1,N  B_1,N |   <- first frame axis
          A_2,0  A_2,1  B_2,1  ...  A_2,N  B_2,N |   <- second frame axis
          A_3,0  A_3,1  B_3,1  ...  A_3,N  B_3,N ]   <- third frame axis

    so ``degree=0`` is simply ``[a_1, a_2, a_3]``. :meth:`coefficient_labels`
    returns the matching names.

    The state vector must contain an entry of the form

    - ``('a_fourier', 3*(2*degree+1), 'estimated', 'static', owner, value)``

    for the coefficients to be estimated. The coefficients are constants of the
    dynamics, so they are declared ``'static'``: unlike the Gauss-Markov
    models, nothing about them is integrated.

    References
    ----------
    .. [1] Montenbruck, O., & Gill, E. (2000). *Satellite Orbits*, Sec. 3.7
       (empirical accelerations in the radial / along-track / cross-track frame).
    .. [2] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
       Orbit Determination*, Ch. 4 (estimating dynamic model parameters).
    .. [3] McMahon, J. W., & Scheeres, D. J. (2010). New Solar Radiation
       Pressure Force Model for Navigation. *Journal of Guidance, Control, and
       Dynamics*, 33(5). See also McMahon (2011), PhD thesis, University of
       Colorado, Eq. (2.28), where the force is expanded as
       :math:`\\sum_n [A_n(\\delta_s)\\cos n\\lambda_s +
       B_n(\\delta_s)\\sin n\\lambda_s]`.

    Examples
    --------
    A once-per-revolution along-track acceleration, estimated:

    .. code-block:: python

        import scarabaeus as scb

        state = [
            ("position", 3, "estimated", "dynamic", sc, [7000, 0, 0]),
            ("velocity", 3, "estimated", "dynamic", sc, [0, 7.5, 0]),
            ("a_fourier", 9, "estimated", "static", sc, [0.0] * 9),
        ]

        force_model = scb.ForceModelTranslation(
            primary_body=sc,
            fourier_acceleration=True,
            fourier_degree=1,
            fourier_frame="RTN",
            fourier_angle="argument_of_latitude",
        )
    """

    #: Frame pole, the direction the ascending node is measured about.
    _K_HAT = np.array([0.0, 0.0, 1.0])

    #: Cross-product matrix of ``_K_HAT``. Used on the RHS path, so it is
    #: built once rather than per evaluation.
    _SKEW_K = DynamicalUtils.skew(_K_HAT)

    #  Configuration errors, each raised where the setting is first used. The
    #  frame settings and the RTN triad raise the DynamicalUtils errors.
    class InvalidAngleError(ValueError):
        """Raised when ``angle`` is not one of the supported phase angles."""

    class InvalidDegreeError(ValueError):
        """Raised when ``degree`` is negative or not an integer."""

    class MissingPeriodError(ValueError):
        """Raised when a time-based phase is requested without a ``period``."""

    class MissingEpochError(ValueError):
        """Raised when a time-based or solar phase has no ``t0`` or epoch."""

    class InvalidEpochError(TypeError):
        """Raised when ``t0`` is neither an ``EpochArray`` nor a real number."""

    class CoefficientSizeError(ValueError):
        """Raised when the ``a_fourier`` state does not match the model degree."""

    class DegenerateGeometryError(ValueError):
        """Raised when the orbit geometry leaves the phase angle undefined."""

    def __init__(
        self,
        origin: Body,
        spacecraft: Spacecraft,
        base_frame: Frame,
        degree: int = 0,
        frame: str = "inertial",
        angle: str = "time",
        period: float | None = None,
        t0: EpochArray | float | None = None,
        body_fixed_frame=None,
        relative_body=None,
        coefficients: np.ndarray | None = None,
    ):
        if int(degree) != degree or degree < 0:
            raise self.InvalidDegreeError(
                f"degree must be a non-negative integer, got {degree!r}. "
                "Use degree=0 for a constant acceleration."
            )
        self._degree = int(degree)
        self._n_terms = 2 * self._degree + 1

        # The relative body governs both the RTN triad and the orbital phase
        # angles: they describe the same orbit, so anchoring them to different
        # bodies would give a frame and a phase that disagree.
        self._frame, self._body_fixed_frame, self._relative_body = (
            DynamicalUtils.resolve_frame(frame, body_fixed_frame, relative_body)
        )
        # Per-epoch SPICE lookups of the frame, kept by DynamicalUtils.
        self._frame_memo = {}

        angle_key = str(angle).strip().lower()
        if angle_key not in ("time", "argument_of_latitude", "solar_longitude"):
            raise self.InvalidAngleError(
                "angle must be 'time', 'argument_of_latitude' or "
                f"'solar_longitude', got {angle!r}. A series in true anomaly is "
                "covered by 'argument_of_latitude', which spans the same "
                "functions and stays defined on a circular orbit."
            )
        self._angle = angle_key

        # Only a series with at least one harmonic has a phase at all, so the
        # phase settings are validated for degree >= 1 and otherwise ignored.
        if self._degree >= 1 and self._angle == "time":
            if period is None:
                raise self.MissingPeriodError(
                    "a time-based Fourier phase needs 'period', the repeat time "
                    "of the series in seconds — typically the orbital period. "
                    "Pass angle='argument_of_latitude' to key the series to the "
                    "orbit geometry instead."
                )
            if float(period) <= 0.0:
                raise self.MissingPeriodError(
                    f"period must be > 0 seconds, got {period!r}."
                )
            if t0 is None:
                raise self.MissingEpochError(
                    "a time-based Fourier phase needs 't0', the epoch of zero "
                    "phase. In a MissionSequence pass the same arc-start epoch "
                    "to every leg: the legs share one coefficient state, so a "
                    "moving phase origin would give each leg a different series."
                )

        self._origin = origin
        self._origin_name = origin.name
        self._spacecraft = spacecraft
        self._ref_frame = base_frame

        self._period = float(period) if period is not None else None
        self._omega = 2.0 * np.pi / self._period if self._period else 0.0
        self._t0 = self._epoch_to_et(t0) if t0 is not None else None

        self._default_coefficients = (
            self._validate_coefficients(np.asarray(coefficients, dtype=float).ravel())
            if coefficients is not None
            else None
        )

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _epoch_to_et(t0: EpochArray | float) -> float:
        """
        Normalise ``t0`` to raw ephemeris seconds (TDB).

        Both forms are accepted because the callers disagree: a script
        configures the model with an ``EpochArray``, while
        :class:`~scarabaeus.Propagator` binds its force models with the raw ET
        float it hands the integrator. The phase is measured against that raw
        ET, so an ``EpochArray`` is read through ``.et`` rather than ``.times``,
        which follows the array's display system.

        Parameters
        ----------
        t0 : EpochArray or float
            Epoch of zero phase.

        Returns
        -------
        float
            Epoch in ephemeris seconds past J2000 (TDB).

        Raises
        ------
        InvalidEpochError
            If ``t0`` is neither an ``EpochArray`` nor a real number.
        """
        if isinstance(t0, EpochArray):
            return float(np.asarray(t0.et).ravel()[0])
        if isinstance(t0, (int, float, np.floating, np.integer)):
            return float(t0)
        raise FourierAcceleration.InvalidEpochError(
            "t0 must be an EpochArray or ephemeris seconds (TDB) as a float; "
            f"received {type(t0).__name__}."
        )

    def _validate_coefficients(self, values: np.ndarray) -> np.ndarray:
        """
        Check a coefficient vector against the configured degree.

        Parameters
        ----------
        values : np.ndarray
            Flat coefficient vector.

        Returns
        -------
        np.ndarray
            The same vector, validated.

        Raises
        ------
        CoefficientSizeError
            If the length is not ``3 * (2 * degree + 1)``.
        """
        expected = 3 * self._n_terms
        if values.size != expected:
            raise self.CoefficientSizeError(
                f"a Fourier acceleration of degree {self._degree} has "
                f"{expected} coefficients (3 axes x {self._n_terms} terms), got "
                f"{values.size}. Either size the 'a_fourier' state to "
                f"{expected} or set fourier_degree to "
                f"{(values.size // 3 - 1) // 2 if values.size % 3 == 0 else '?'}."
            )
        return values

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def origin(self) -> Body:
        """Celestial body serving as origin of the state vector."""
        return self._origin

    @property
    def origin_name(self) -> str:
        """Name of the origin body."""
        return self._origin_name

    @property
    def ref_frame(self):
        """Reference frame used by the model."""
        return self._ref_frame

    @property
    def degree(self) -> int:
        """Highest harmonic retained; ``0`` is a constant acceleration."""
        return self._degree

    @property
    def n_terms(self) -> int:
        """Number of terms per axis, ``2 * degree + 1``."""
        return self._n_terms

    @property
    def n_coefficients(self) -> int:
        """Length of the stacked ``a_fourier`` state, ``3 * (2 * degree + 1)``."""
        return 3 * self._n_terms

    @property
    def frame(self) -> str:
        """Frame the coefficients are expressed in."""
        return self._frame

    @property
    def angle(self) -> str:
        """Phase angle driving the series."""
        return self._angle

    @property
    def period(self) -> float | None:
        """Repeat period of a time-based phase, in seconds."""
        return self._period

    @property
    def t0(self) -> float | None:
        """Epoch of zero phase for a time-based series, in ephemeris seconds."""
        return self._t0

    @property
    def relative_body(self) -> str | None:
        """Body the RTN triad and the orbital phase are referred to, if any."""
        return self._relative_body

    @property
    def body_fixed_frame(self) -> str | None:
        """Body-fixed frame the coefficients are expressed in, if any."""
        return self._body_fixed_frame

    @property
    def default_coefficients(self):
        """Coefficients applied when the state carries no ``a_fourier`` entry."""
        return self._default_coefficients

    @default_coefficients.setter
    def default_coefficients(self, values):
        """Set the coefficients used when the state carries no ``a_fourier``."""
        self._default_coefficients = (
            self._validate_coefficients(np.asarray(values, dtype=float).ravel())
            if values is not None
            else None
        )

    @property
    def depends_on_velocity(self) -> bool:
        """
        Whether the acceleration is a function of velocity.

        True for the RTN frame, whose triad is built from the velocity, and for
        the argument-of-latitude phase, which is measured in the orbit plane.
        The force model reads this to decide whether a
        :math:`\\partial \\mathbf a / \\partial \\mathbf v` block is needed at
        all.
        """
        return self._frame == "RTN" or self._angle in (
            "argument_of_latitude",
            "solar_longitude",
        )

    @property
    def _phase_depends_on_state(self) -> bool:
        """Whether the phase angle is a function of the propagated state."""
        return self._degree >= 1 and self._angle in (
            "argument_of_latitude",
            "solar_longitude",
        )

    def coefficient_labels(self) -> list[str]:
        """
        Names of the stacked coefficients, in state order.

        Returns
        -------
        list of str
            ``3 * (2 * degree + 1)`` labels such as ``['A0_R', 'A1_R', 'B1_R',
            'A0_T', ...]`` for the RTN frame, matching the layout of the
            ``a_fourier`` state.
        """
        labels = []
        for axis in DynamicalUtils.ACCELERATION_FRAME_AXES[self._frame]:
            labels.append(f"A0_{axis}")
            for k in range(1, self._degree + 1):
                labels.append(f"A{k}_{axis}")
                labels.append(f"B{k}_{axis}")
        return labels

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------
    def _get_coefficients(self, current_state: dict, body: Body):
        """
        Resolve the Fourier coefficients for the given body.

        Priority:
        1. ``current_state[('a_fourier', spice_id)]``
        2. the model's ``default_coefficients``
        3. ``None``, meaning the model contributes nothing

        Parameters
        ----------
        current_state : dict
            Dictionary containing the propagated state.
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        np.ndarray or None
            Coefficient vector of length ``3 * (2 * degree + 1)``.
        """
        spice_id = getattr(body, "spice_id", None)
        if spice_id is None:
            return self._default_coefficients

        key = ("a_fourier", spice_id)
        if key in current_state:
            val = current_state[key]
            values = np.atleast_1d(
                val.values if hasattr(val, "values") else val
            ).astype(float)
            return self._validate_coefficients(values.ravel())

        return self._default_coefficients

    # ------------------------------------------------------------------
    # Series construction
    # ------------------------------------------------------------------
    def _design_row(self, theta: float) -> np.ndarray:
        """
        Basis of the truncated series at one phase.

        .. math::

            \\mathbf m(\\theta) = [\\,1,\\;
                \\cos\\theta,\\;\\sin\\theta,\\;\\ldots,\\;
                \\cos N\\theta,\\;\\sin N\\theta\\,]

        Parameters
        ----------
        theta : float
            Phase angle in radians.

        Returns
        -------
        np.ndarray
            Vector of length ``2 * degree + 1``.
        """
        row = np.ones(self._n_terms)
        if self._degree >= 1:
            k_theta = np.arange(1, self._degree + 1) * theta
            row[1::2] = np.cos(k_theta)
            row[2::2] = np.sin(k_theta)
        return row

    def _design_row_derivative(self, theta: float) -> np.ndarray:
        """
        Derivative of :meth:`_design_row` with respect to the phase.

        .. math::

            \\frac{d\\mathbf m}{d\\theta} = [\\,0,\\;
                -\\sin\\theta,\\;\\cos\\theta,\\;\\ldots,\\;
                -N\\sin N\\theta,\\;N\\cos N\\theta\\,]

        Parameters
        ----------
        theta : float
            Phase angle in radians.

        Returns
        -------
        np.ndarray
            Vector of length ``2 * degree + 1``.
        """
        row = np.zeros(self._n_terms)
        if self._degree >= 1:
            k = np.arange(1, self._degree + 1)
            k_theta = k * theta
            row[1::2] = -k * np.sin(k_theta)
            row[2::2] = k * np.cos(k_theta)
        return row

    def _phase(self, position, current_state: dict, epoch, body) -> float:
        """
        Phase angle of the series at the current state and epoch.

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position 3-vector, km.
        current_state : dict
            Dictionary containing the propagated state.
        epoch : float
            Current epoch in ephemeris seconds (TDB).
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        float
            Phase angle in radians. Always zero for ``degree == 0``, which has
            no phase dependence.
        """
        if self._degree == 0:
            return 0.0
        if self._angle == "time":
            if epoch is None:
                raise self.MissingEpochError(
                    "a time-based Fourier phase needs the evaluation epoch, but "
                    "none was supplied."
                )
            return self._omega * (float(epoch) - self._t0)
        if self._angle == "solar_longitude":
            return self._solar_longitude(position, current_state, epoch, body)[0]
        return self._argument_of_latitude(
            position, current_state, epoch, body
        )[0]

    def _phase_with_partials(self, position, current_state: dict, epoch, body):
        """
        Phase angle and its gradients with respect to position and velocity.

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position 3-vector, km.
        current_state : dict
            Dictionary containing the propagated state.
        epoch : float
            Current epoch in ephemeris seconds (TDB).
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        tuple
            ``(theta, dtheta_dr, dtheta_dv)``. The gradients are ``None`` for a
            time-based phase, which is not a function of the state.
        """
        if self._angle == "argument_of_latitude":
            return self._argument_of_latitude(position, current_state, epoch, body)
        if self._angle == "solar_longitude":
            return self._solar_longitude(position, current_state, epoch, body)
        return self._phase(position, current_state, epoch, body), None, None

    def _sun_direction(self, position, epoch):
        """
        Unit vector from the spacecraft to the Sun, in the propagation frame.

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position 3-vector from the propagation origin, km.
        epoch : float
            Current epoch in ephemeris seconds (TDB).

        Returns
        -------
        tuple
            ``(u_hat, distance)``, the unit vector and its magnitude in km.
        """
        if epoch is None:
            raise self.MissingEpochError(
                "an epoch is required to locate the Sun for a solar-longitude "
                "Fourier phase."
            )

        cache = self.__dict__.get("_sun_cache")
        if cache is None or cache[0] != epoch:
            self._sun_cache = (
                epoch,
                np.asarray(
                    SpiceManager.get_pos_values(
                        constants.SUN.spice_name,
                        epoch,
                        DynamicalUtils.frame_name(self._ref_frame),
                        self._origin_name,
                    ),
                    dtype=float,
                ).ravel(),
            )
            cache = self._sun_cache

        d = cache[1] - np.asarray(position, dtype=float).ravel()
        distance = np.linalg.norm(d)
        if distance == 0.0:
            raise self.DegenerateGeometryError(
                "the solar longitude is undefined for a spacecraft at the Sun."
            )
        return d / distance, distance

    def _solar_longitude(self, position, current_state: dict, epoch, body):
        """
        Solar longitude in the coefficient frame, and its state partials.

        This is the phase McMahon and Scheeres expand the solar radiation
        pressure force in [1]_: the longitude of the Sun as seen in the frame
        the coefficients live in,

        .. math::

            \\hat{\\mathbf u} = \\cos\\delta_s \\cos\\lambda_s\\,
                \\hat{\\mathbf e}_1
              + \\cos\\delta_s \\sin\\lambda_s\\, \\hat{\\mathbf e}_2
              + \\sin\\delta_s\\, \\hat{\\mathbf e}_3,
            \\qquad
            \\lambda_s = \\operatorname{atan2}
                (\\hat{\\mathbf u}\\cdot\\hat{\\mathbf e}_2,\\;
                 \\hat{\\mathbf u}\\cdot\\hat{\\mathbf e}_1).

        In a body-fixed frame this is exactly their :math:`\\lambda_s`, whose
        fundamental frequency is the body's rotation relative to the Sun rather
        than anything the caller supplies. In RTN it is the Sun-relative
        in-plane angle that the CODE/Bern GPS model writes as
        :math:`u - u_0`, up to a sign that the estimated coefficients absorb.

        Unlike ``argument_of_latitude``, the phase origin follows the Sun, so
        the coefficients keep their meaning as the Sun-relative geometry drifts
        over a long arc.

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position 3-vector, km.
        current_state : dict
            Dictionary containing the propagated state.
        epoch : float
            Current epoch in ephemeris seconds (TDB).
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        tuple
            ``(lambda_s, dlambda_dr, dlambda_dv)``, the angle in radians and the
            two gradients as row 3-vectors.

        Raises
        ------
        DegenerateGeometryError
            If the Sun lies along the frame's third axis, where the longitude
            about that axis is undefined.

        References
        ----------
        .. [1] McMahon, J. W. (2011). *An Analytical Theory for the Perturbative
           Effect of Solar Radiation Pressure on Natural and Artificial
           Satellites*, PhD thesis, University of Colorado, Eq. (2.28).
        """
        u_hat, distance = self._sun_direction(position, epoch)

        # Frame axes, and how they move with the state. Only RTN has axes that
        # are a function of the spacecraft state at all.
        if self._frame == "RTN":
            basis, jac_r, jac_v = DynamicalUtils.rtn_triad(
                self, position, current_state, epoch, body, derivatives=True
            )
            e1, e2, e3 = basis[:, 0], basis[:, 1], basis[:, 2]
            de1_dr, de2_dr = jac_r[0], jac_r[1]
            de1_dv, de2_dv = jac_v[0], jac_v[1]
        else:
            T = DynamicalUtils.frame_matrix(self, position, current_state, epoch, body)
            basis = np.eye(3) if T is None else T
            e1, e2, e3 = basis[:, 0], basis[:, 1], basis[:, 2]
            de1_dr = de2_dr = de1_dv = de2_dv = np.zeros((3, 3))

        cos_l = float(u_hat @ e1)
        sin_l = float(u_hat @ e2)

        # cos^2(delta_s): zero when the Sun is along the frame's third axis,
        # where no longitude about that axis exists.
        denom = cos_l * cos_l + sin_l * sin_l
        if denom < 1e-12:
            raise self.DegenerateGeometryError(
                "the solar longitude is undefined when the Sun lies along the "
                f"third axis of the '{self._frame}' frame "
                f"(cos(solar latitude) = {np.sqrt(max(denom, 0.0)):.3e}). Use "
                "angle='time', or express the series in another frame."
            )

        # d(u_hat)/dr: the Sun's position is an ephemeris quantity, so only the
        # spacecraft end of the vector moves, with a minus sign.
        J_u_r = -(np.eye(3) - np.outer(u_hat, u_hat)) / distance

        dcos_dr = e1 @ J_u_r + u_hat @ de1_dr
        dsin_dr = e2 @ J_u_r + u_hat @ de2_dr
        dcos_dv = u_hat @ de1_dv
        dsin_dv = u_hat @ de2_dv

        lam = float(np.arctan2(sin_l, cos_l))
        dlam_dr = (cos_l * dsin_dr - sin_l * dcos_dr) / denom
        dlam_dv = (cos_l * dsin_dv - sin_l * dcos_dv) / denom

        return lam, dlam_dr, dlam_dv

    def _argument_of_latitude(self, position, current_state: dict, epoch, body):
        """
        Argument of latitude and its partials with respect to the state.

        The angle is read off the in-plane basis
        :math:`(\\hat{\\mathbf n}, \\hat{\\mathbf m})`, where
        :math:`\\hat{\\mathbf n}` points at the ascending node and
        :math:`\\hat{\\mathbf m} = \\hat{\\mathbf h} \\times \\hat{\\mathbf n}`
        leads it by a quarter revolution:

        .. math::

            u = \\operatorname{atan2}
                (\\hat{\\mathbf r}\\cdot\\hat{\\mathbf m},\\;
                 \\hat{\\mathbf r}\\cdot\\hat{\\mathbf n}).

        Differentiating with :math:`c = \\cos u`, :math:`s = \\sin u` gives
        :math:`du = (c\\,ds - s\\,dc)/(c^2+s^2)`, and every term below is the
        chain rule applied to the unit vectors that build :math:`c` and
        :math:`s`.

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position 3-vector, km.
        current_state : dict
            Dictionary containing the propagated state.
        epoch : float
            Current epoch in ephemeris seconds (TDB).
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        tuple
            ``(u, du_dr, du_dv)`` with the angle in radians and the two
            gradients as row 3-vectors, in 1/km and s/km.

        Raises
        ------
        DegenerateGeometryError
            If the orbit is exactly equatorial or the angular momentum vanishes,
            leaving the node direction undefined.
        """
        r, v = DynamicalUtils.relative_state(self, position, current_state, epoch, body)

        r_norm = np.linalg.norm(r)
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)
        if r_norm == 0.0 or h_norm == 0.0:
            raise self.DegenerateGeometryError(
                "the argument of latitude is undefined for a state with zero "
                "radius or zero angular momentum."
            )
        r_hat = r / r_norm
        h_hat = h / h_norm

        # Node direction: K x h, whose length is sin(i) once h is normalised.
        # An equatorial orbit has no ascending node to measure the angle from.
        q = np.cross(self._K_HAT, h_hat)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            raise self.DegenerateGeometryError(
                "the argument of latitude is undefined for an equatorial orbit: "
                "the ascending node it is measured from does not exist "
                f"(sin(i) = {q_norm:.3e}). Use angle='time' instead, or express "
                "the series in the inertial frame."
            )
        n_hat = q / q_norm
        m_hat = np.cross(h_hat, n_hat)

        cos_u = float(r_hat @ n_hat)
        sin_u = float(r_hat @ m_hat)
        u = float(np.arctan2(sin_u, cos_u))

        # Jacobians of the unit vectors. P_x = (I - x x^T)/|x| projects out the
        # component along x, which is what normalising removes.
        eye = np.eye(3)
        J_rhat_r = (eye - np.outer(r_hat, r_hat)) / r_norm
        P_h = (eye - np.outer(h_hat, h_hat)) / h_norm
        J_hhat_r = -P_h @ DynamicalUtils.skew(v)
        J_hhat_v = P_h @ DynamicalUtils.skew(r)

        P_q = (eye - np.outer(n_hat, n_hat)) / q_norm
        J_nhat_r = P_q @ self._SKEW_K @ J_hhat_r
        J_nhat_v = P_q @ self._SKEW_K @ J_hhat_v

        # d(a x b) = -[b]x da + [a]x db, with a = h_hat and b = n_hat.
        J_mhat_r = -DynamicalUtils.skew(n_hat) @ J_hhat_r + DynamicalUtils.skew(h_hat) @ J_nhat_r
        J_mhat_v = -DynamicalUtils.skew(n_hat) @ J_hhat_v + DynamicalUtils.skew(h_hat) @ J_nhat_v

        dc_dr = n_hat @ J_rhat_r + r_hat @ J_nhat_r
        dc_dv = r_hat @ J_nhat_v
        ds_dr = m_hat @ J_rhat_r + r_hat @ J_mhat_r
        ds_dv = r_hat @ J_mhat_v

        # Analytically c^2 + s^2 = 1, since r_hat lies in the orbit plane; the
        # division keeps that exact under accumulated round-off.
        denom = cos_u * cos_u + sin_u * sin_u
        du_dr = (cos_u * ds_dr - sin_u * dc_dr) / denom
        du_dv = (cos_u * ds_dv - sin_u * dc_dv) / denom

        return u, du_dr, du_dv

    # ------------------------------------------------------------------
    # Acceleration and partials
    # ------------------------------------------------------------------
    def compute_acceleration(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: Body,
    ) -> np.ndarray:
        """
        Empirical Fourier acceleration applied to the body.

        .. math::

            \\mathbf a_F = \\mathbf T \\, \\mathbf M(\\theta) \\, \\mathbf p

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position 3-vector, km.
        current_state : dict
            Dictionary containing the propagated state.
        epoch : float
            Current epoch in ephemeris seconds (TDB).
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        np.ndarray
            Acceleration 3-vector in km/s^2. Zero when neither the state nor
            the model configuration supplies coefficients.
        """
        p = self._get_coefficients(current_state, body)
        if p is None:
            return np.zeros(3)

        theta = self._phase(position, current_state, epoch, body)
        a_frame = p.reshape(3, self._n_terms) @ self._design_row(theta)

        T = DynamicalUtils.frame_matrix(self, position, current_state, epoch, body)
        return a_frame if T is None else T @ a_frame

    def _compute_partial_by_position(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: EpochArray,
        body: Body,
    ) -> np.ndarray:
        """
        Partial derivative of the acceleration with respect to position.

        Two mechanisms make the acceleration position-dependent, and only the
        active ones contribute:

        .. math::

            \\frac{\\partial \\mathbf a_F}{\\partial \\mathbf r}
            = \\underbrace{\\sum_{j} c_j
                \\frac{\\partial \\hat{\\mathbf e}_j}{\\partial \\mathbf r}}
                _{\\text{frame triad}}
            + \\underbrace{\\mathbf T \\,
                \\frac{d\\mathbf M}{d\\theta} \\mathbf p \\,
                \\left(\\frac{\\partial \\theta}
                             {\\partial \\mathbf r}\\right)^{\\!T}}
                _{\\text{phase}}

        An inertial or body-fixed frame with a time-based phase therefore has an
        exactly zero block, as an empirical acceleration should.

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position 3-vector, km.
        current_state : dict
            Dictionary containing the propagated state.
        epoch : float
            Current epoch in ephemeris seconds (TDB).
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        np.ndarray
            3x3 Jacobian in 1/s^2.
        """
        return self._state_partials(position, current_state, epoch, body)[0]

    def _compute_partial_by_velocity(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: EpochArray,
        body: Body,
    ) -> np.ndarray:
        """
        Partial derivative of the acceleration with respect to velocity.

        Non-zero only for the RTN frame, whose along-track and cross-track
        directions are built from the velocity, and for the
        argument-of-latitude phase.

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position 3-vector, km.
        current_state : dict
            Dictionary containing the propagated state.
        epoch : float
            Current epoch in ephemeris seconds (TDB).
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        np.ndarray
            3x3 Jacobian in 1/s.
        """
        return self._state_partials(position, current_state, epoch, body)[1]

    def _state_partials(self, position, current_state: dict, epoch, body):
        """
        Position and velocity Jacobians, sharing their common intermediates.

        The triad and the phase both need the angular momentum and its
        derivatives, so they are computed once here and the two blocks returned
        together.

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position 3-vector, km.
        current_state : dict
            Dictionary containing the propagated state.
        epoch : float
            Current epoch in ephemeris seconds (TDB).
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        tuple of np.ndarray
            ``(da_dr, da_dv)``, each 3x3.
        """
        p = self._get_coefficients(current_state, body)
        if p is None:
            return np.zeros((3, 3)), np.zeros((3, 3))

        # Nothing in an inertial or body-fixed frame with a time-based phase is
        # a function of the state, so the common case short-circuits here.
        if self._frame != "RTN" and not self._phase_depends_on_state:
            return np.zeros((3, 3)), np.zeros((3, 3))

        coefficients = p.reshape(3, self._n_terms)

        if self._phase_depends_on_state:
            theta, dtheta_dr, dtheta_dv = self._phase_with_partials(
                position, current_state, epoch, body
            )
        else:
            theta = self._phase(position, current_state, epoch, body)
            dtheta_dr = dtheta_dv = None

        c = coefficients @ self._design_row(theta)

        if self._frame == "RTN":
            T, D_r, D_v = DynamicalUtils.rtn_triad(
                self, position, current_state, epoch, body, derivatives=True
            )
            da_dr = DynamicalUtils.body_vector_jacobian(c, D_r)
            da_dv = DynamicalUtils.body_vector_jacobian(c, D_v)
        else:
            T = DynamicalUtils.frame_matrix(self, position, current_state, epoch, body)
            da_dr = np.zeros((3, 3))
            da_dv = np.zeros((3, 3))

        if self._phase_depends_on_state:
            dc_dtheta = coefficients @ self._design_row_derivative(theta)
            if T is not None:
                dc_dtheta = T @ dc_dtheta
            da_dr = da_dr + np.outer(dc_dtheta, dtheta_dr)
            da_dv = da_dv + np.outer(dc_dtheta, dtheta_dv)

        return da_dr, da_dv

    def _compute_partial_by_coefficients(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: EpochArray,
        body: Body,
    ) -> np.ndarray:
        """
        Partial derivative of the acceleration with respect to the coefficients.

        The acceleration is linear in the coefficients, so the partial is the
        design matrix itself,

        .. math::

            \\frac{\\partial \\mathbf a_F}{\\partial \\mathbf p}
                = \\mathbf T \\, \\mathbf M(\\theta),
            \\qquad
            \\mathbf M(\\theta) = \\mathbf I_3 \\otimes \\mathbf m(\\theta)^T,

        which is exact at every epoch and needs no linearisation.

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position 3-vector, km.
        current_state : dict
            Dictionary containing the propagated state.
        epoch : float
            Current epoch in ephemeris seconds (TDB).
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        np.ndarray
            ``3 x 3*(2*degree+1)`` dimensionless matrix, in the coefficient
            order documented by :meth:`coefficient_labels`.
        """
        theta = self._phase(position, current_state, epoch, body)
        M = np.kron(np.eye(3), self._design_row(theta).reshape(1, self._n_terms))

        T = DynamicalUtils.frame_matrix(self, position, current_state, epoch, body)
        return M if T is None else T @ M
