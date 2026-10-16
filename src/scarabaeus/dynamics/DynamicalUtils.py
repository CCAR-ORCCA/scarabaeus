# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Frame, SpiceManager
import scarabaeus.utils.NumpyWrapper as np
import spiceypy

J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


# --------------------#
#  Class Definition  #
# --------------------#
class DynamicalUtils:
    """ Shared geometry of the dynamics models: the spacecraft attitude of the
    surface-force models and the component frames of the empirical
    accelerations.

    **Spacecraft attitude.** Force models whose surfaces or boresights are
    fixed in the spacecraft frame (:class:`~scarabaeus.nPlateSRP`,
    :class:`~scarabaeus.ThermalRadiation`,
    :class:`~scarabaeus.AntennaRadiationPressure`) need the spacecraft-fixed to
    inertial rotation :math:`C`, so that a spacecraft-fixed vector
    :math:`\\mathbf{e}_b` reaches the inertial frame as
    :math:`\\mathbf{e} = C\\mathbf{e}_b`. The supported
    ``Spacecraft.attitudeMode`` values, :attr:`ATTITUDE_MODES`, are:

    * ``"CK"``: :math:`C` from the loaded C-kernel (:meth:`ck_dcm`).
    * ``"inertialFixedAttitude"``: :math:`C` is
      ``Spacecraft.fixedInertialAttitude`` (:meth:`fixed_dcm`).
    * ``"nadirPointingSun"``: the axes follow the heliocentric position
      :math:`\\mathbf{s}` and velocity :math:`\\mathbf{w}`
      (:meth:`nadirPointingSun`),

      .. math::

          \\hat{\\mathbf{x}} = \\frac{\\mathbf{s}}{\\lVert\\mathbf{s}\\rVert},\\quad
          \\hat{\\mathbf{z}} =
            \\frac{\\hat{\\mathbf{x}}\\times\\hat{\\mathbf{w}}}
                 {\\lVert\\hat{\\mathbf{x}}\\times\\hat{\\mathbf{w}}\\rVert},\\quad
          \\hat{\\mathbf{y}} = \\hat{\\mathbf{z}}\\times\\hat{\\mathbf{x}},\\quad
          C = \\left[\\hat{\\mathbf{x}}\\;\\hat{\\mathbf{y}}\\;\\hat{\\mathbf{z}}\\right]

      so the spacecraft :math:`-X` axis points at the Sun. It replaces the
      deprecated
      :meth:`~scarabaeus.Spacecraft.inertial_to_body_DCM_nadirPointingSun`,
      which returns :math:`C^T`. Only this mode depends on the state, and on
      the velocity as well as the position.

    The last nadir attitude evaluated with derivatives is kept here, since the
    position and velocity Jacobians of every surface model are taken at one
    state.

    **Acceleration frames.** An empirical acceleration
    (:class:`~scarabaeus.FirstOrderGaussMarkov`,
    :class:`~scarabaeus.PiecewiseFirstOrderGaussMarkov`,
    :class:`~scarabaeus.PiecewisePolynomialAcceleration`,
    :class:`~scarabaeus.FourierAcceleration`) can express its components
    :math:`\\mathbf{a}^{(f)}` in a chosen frame. They reach the propagation
    frame as :math:`\\mathbf{a} = T\\,\\mathbf{a}^{(f)}` (:meth:`from_frame`).
    The ``frame`` options are the keys of
    :attr:`ACCELERATION_FRAME_AXES`:

    * ``"inertial"``: components in the propagation frame, :math:`T = I`.
    * ``"body_fixed"``: components in a rotating body-fixed frame, rotated out
      through SPICE, :math:`T = R_{bf\\to rf}(t)`
      (:meth:`body_fixed_rotation`).
    * ``"RTN"``: components in the radial / along-track / cross-track triad
      of the orbit (also called RSW or RIC),
      :math:`T = \\left[\\hat{\\mathbf{R}}\\;\\hat{\\mathbf{T}}\\;
      \\hat{\\mathbf{N}}\\right]`, from :meth:`rtn`, which is built and
      differentiated like the nadir attitude. The triad is built from the
      state, so it adds the position and velocity partials
      :meth:`frame_partials` returns. With a ``relative_body`` it describes
      the orbit about that body rather than the propagation origin
      (:meth:`relative_state`).

    A model resolves its settings once in its constructor with
    :meth:`resolve_frame`, and exposes them as the public ``frame``,
    ``body_fixed_frame`` and ``relative_body`` properties next to
    ``ref_frame`` and ``origin_name``, which is all the methods here read. A
    model that also carries a ``_frame_memo`` dict gets the body-fixed
    rotation and the relative-body offset memoized for the epoch of the RHS
    step in progress: the acceleration and every partial are evaluated at the
    same epoch, so each SPICE lookup happens once per step rather than per
    call.

    All methods are static or class methods and work on raw NumPy arrays; no
    instantiation is required::

        mode = scb.DynamicalUtils.attitude_mode(spacecraft)
        C, D_r, D_v = scb.DynamicalUtils.nadirPointingSun(s, w, derivatives=True)
        frame, bff, rel = scb.DynamicalUtils.resolve_frame("rtn")
        a = scb.DynamicalUtils.from_frame(model, a_rtn, r, state, epoch, sc)
        da_dr, da_dv = scb.DynamicalUtils.frame_partials(
            model, a_rtn, r, state, epoch, sc
        )

    See Also
    --------
    scarabaeus.nPlateSRP : Solar radiation pressure on a plate model.
    scarabaeus.ThermalRadiation : Thermal re-radiation recoil.
    scarabaeus.AntennaRadiationPressure : Antenna and LIDAR recoil.
    scarabaeus.Spacecraft : Carries ``attitudeMode`` and ``fixedInertialAttitude``.
    scarabaeus.FirstOrderGaussMarkov : Gauss-Markov empirical acceleration.
    scarabaeus.FourierAcceleration : Fourier-series empirical acceleration.
    """

    # (inputs, (C, D_r, D_v)) of the last untraced nadir evaluation with
    # derivatives; the arrays are read-only since every caller shares them.
    _nadir_memo: tuple = (None, None)

    ATTITUDE_MODES = ("CK", "nadirPointingSun", "inertialFixedAttitude")
    """``Spacecraft.attitudeMode`` values the surface-force models support.
    Each is turned into the spacecraft-fixed to inertial rotation: from the
    loaded C-kernel, from the heliocentric position and velocity, or from
    ``Spacecraft.fixedInertialAttitude``."""

    NADIR_ATTITUDE_MODE = "nadirPointingSun"
    """The entry of :attr:`ATTITUDE_MODES` whose rotation depends on the
    spacecraft state, on its velocity as well as its position."""

    ACCELERATION_FRAME_AXES = {
        "inertial": ("x", "y", "z"),
        "body_fixed": ("x", "y", "z"),
        "RTN": ("R", "T", "N"),
    }
    """Frames an empirical acceleration can be expressed in, with the axis
    labels of each. The keys are the ``frame`` options :meth:`resolve_frame`
    accepts; the labels name the components in reports and plot legends. The
    inertial and body-fixed frames are Cartesian triads, the RTN one is radial
    / along-track / cross-track."""

    # ------------------#
    # region   Errors  #
    # ------------------#
    class InvalidAttitudeModeError(ValueError):
        """The spacecraft attitude mode is not supported.

        Surface normals and boresights are fixed in the spacecraft frame, so a
        spacecraft-to-inertial rotation is needed. It is available for the
        ``"CK"``, ``"nadirPointingSun"`` and ``"inertialFixedAttitude"``
        modes only.
        """

    class InvalidFixedAttitudeError(ValueError):
        """``fixedInertialAttitude`` is not a 3x3 array.

        An ``inertialFixedAttitude`` spacecraft must carry the
        spacecraft-fixed to inertial rotation as a 3x3 ``np.ndarray``.
        """

    class MissingVelocityError(KeyError):
        """The spacecraft velocity is not in the propagated state.

        The ``nadirPointingSun`` attitude uses the heliocentric velocity to
        fix the spacecraft Y axis, and the RTN triad uses the orbital angular
        momentum, so the state must hold ``("velocity", spice_id)`` or a plain
        ``"velocity"`` entry.
        """

    class DegenerateGeometryError(ValueError):
        """The RTN triad is undefined for the given state.

        The along-track and cross-track directions follow the orbital angular
        momentum :math:`\\mathbf{r}\\times\\mathbf{v}`, which vanishes for a
        zero radius or a purely radial trajectory.
        """

    class InvalidFrameError(ValueError):
        """``frame`` is not one of the supported frames.

        The options are the keys of :attr:`DynamicalUtils.ACCELERATION_FRAME_AXES`,
        matched without regard to case.
        """

    class MissingBodyFixedFrameError(ValueError):
        """``frame='body_fixed'`` was requested without a frame name.

        The body-fixed frame the components are expressed in (for example
        ``'IAU_BENNU'``) is what SPICE rotates them out of.
        """

    class MissingEpochError(ValueError):
        """An ephemeris lookup was needed but no epoch was supplied.

        The body-fixed rotation and the offset of a ``relative_body`` are both
        evaluated at the epoch of the acceleration.
        """

    # endregion Errors #
    # ------------------#

    # -------------------------#
    # region Attitude Methods #
    # -------------------------#
    @staticmethod
    def _raw(value):
        """
        Strip autograd boxes from ``value``.

        Parameters
        ----------
        value : float or numpy.ndarray or ArrayBox
            Possibly traced value.

        Returns
        -------
        float or numpy.ndarray
            The underlying plain value.
        """
        while isinstance(value, np.autograd_types):
            value = value._value
        return value

    @staticmethod
    def skew(v: np.ndarray) -> np.ndarray:
        """
        Cross-product matrix :math:`[\\mathbf{v}]_\\times`, so that
        :math:`[\\mathbf{v}]_\\times \\mathbf{u} = \\mathbf{v} \\times \\mathbf{u}`.

        Parameters
        ----------
        v : numpy.ndarray
            ``(3,)`` vector.

        Returns
        -------
        numpy.ndarray
            ``(3, 3)`` skew-symmetric matrix.
        """
        return np.array(
            [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
        )

    @staticmethod
    def cross(a, b):
        """
        Cross product of two ``(3,)`` vectors, written out by component.

        ``np.cross`` spends most of its time on axis handling for inputs this
        small; this is several times faster and stays traceable by autograd.

        Parameters
        ----------
        a, b : numpy.ndarray or ArrayBox
            ``(3,)`` vectors.

        Returns
        -------
        numpy.ndarray or ArrayBox
            ``(3,)`` vector :math:`\\mathbf{a}\\times\\mathbf{b}`.
        """
        return np.array(
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        )

    @classmethod
    def attitude_mode(cls, body) -> str:
        """
        The spacecraft attitude mode, checked against
        :attr:`ATTITUDE_MODES`.

        Parameters
        ----------
        body : Spacecraft
            The spacecraft.

        Returns
        -------
        str
            ``body.attitudeMode``.

        Raises
        ------
        InvalidAttitudeModeError
            If the mode is not one of :attr:`ATTITUDE_MODES`.
        """
        mode = getattr(body, "attitudeMode", None)
        if mode not in cls.ATTITUDE_MODES:
            raise cls.InvalidAttitudeModeError(
                f"Invalid attitudeMode '{mode}'. "
                f"Must be one of: {cls.ATTITUDE_MODES}."
            )
        return mode

    @staticmethod
    def depends_on_velocity(body) -> bool:
        """
        Whether the attitude of ``body`` depends on its velocity.

        True only for ``"nadirPointingSun"``, whose Y axis follows the
        heliocentric velocity. The mode is not validated here.

        Parameters
        ----------
        body : Spacecraft
            The spacecraft.

        Returns
        -------
        bool
        """
        return getattr(body, "attitudeMode", None) == DynamicalUtils.NADIR_ATTITUDE_MODE

    @classmethod
    def fixed_dcm(cls, body) -> np.ndarray:
        """
        Spacecraft-fixed to inertial rotation of an ``inertialFixedAttitude``
        spacecraft.

        Parameters
        ----------
        body : Spacecraft
            The spacecraft.

        Returns
        -------
        numpy.ndarray
            ``(3, 3)`` float copy of ``body.fixedInertialAttitude``.

        Raises
        ------
        InvalidFixedAttitudeError
            If ``fixedInertialAttitude`` is not a 3x3 ``np.ndarray``.
        """
        dcm = getattr(body, "fixedInertialAttitude", None)
        if not (isinstance(dcm, np.ndarray) and dcm.shape == (3, 3)):
            raise cls.InvalidFixedAttitudeError(
                "fixedInertialAttitude must be a 3x3 numpy array holding the "
                "spacecraft-fixed to inertial rotation. Received "
                f"{type(dcm).__name__}."
            )
        # dtype pinned to float: an integer matrix would otherwise stay int64,
        # which the Rust kernels reject.
        return np.asarray(dcm, dtype=float)

    @staticmethod
    def ck_dcm(body, epoch: float, frame: Frame = None) -> np.ndarray:
        """
        Spacecraft-fixed to inertial rotation from the C-kernel.

        Parameters
        ----------
        body : Spacecraft
            Spacecraft whose C-kernel attitude is queried.
        epoch : float
            TDB epoch (seconds past J2000).
        frame : Frame, optional
            Inertial frame. Anything that is not a :class:`~scarabaeus.Frame`,
            including ``None``, means J2000.

        Returns
        -------
        numpy.ndarray
            ``(3, 3)`` spacecraft-fixed to inertial DCM.
        """
        frame = frame if isinstance(frame, Frame) else J2000
        # get_attitude returns the inertial->body quaternion
        quat_inertial2body = body.get_attitude(frame, epoch)
        return spiceypy.q2m(quat_inertial2body).T

    @classmethod
    def velocity_from_state(cls, current_state: dict, body):
        """
        Spacecraft velocity relative to the origin, read from the state.

        Parameters
        ----------
        current_state : dict
            The propagated state.
        body : Spacecraft
            The spacecraft.

        Returns
        -------
        numpy.ndarray or ArrayBox
            ``(3,)`` velocity in km/s, still traced when the state is.

        Raises
        ------
        MissingVelocityError
            If the state holds neither ``("velocity", spice_id)`` nor
            ``"velocity"``.
        """
        sid = getattr(body, "spice_id", None)
        if ("velocity", sid) in current_state:
            vel = current_state[("velocity", sid)]
        elif "velocity" in current_state:
            vel = current_state["velocity"]
        else:
            raise cls.MissingVelocityError(
                "The nadirPointingSun attitude or the RTN frame needs the "
                f"spacecraft velocity, but the state has no ('velocity', {sid}) "
                "or 'velocity' entry."
            )
        return vel.values if hasattr(vel, "values") else vel

    @staticmethod
    def sun_to_origin_velocity(origin_spice_name: str, frame, epoch: float) -> np.ndarray:
        """
        Sun->origin velocity from SPICE.

        Parameters
        ----------
        origin_spice_name : str
            SPICE name of the propagation origin.
        frame : Frame or str
            Inertial frame of the result.
        epoch : float
            TDB epoch (seconds past J2000).

        Returns
        -------
        numpy.ndarray
            ``(3,)`` Sun->origin velocity in km/s.
        """
        frame_name = getattr(frame, "name", str(frame))
        return np.asarray(
            SpiceManager.get_vel(
                trgt_bdy=origin_spice_name,
                epoch_time=epoch,
                reference_frame=frame_name,
                obsvr_bdy="SUN",
            ).values,
            dtype=float,
        )

    @classmethod
    def nadirPointingSun(
        cls, sun2body, sun2body_vel, derivatives: bool = False
    ) -> tuple:
        """
        Spacecraft-fixed to inertial rotation of the ``nadirPointingSun``
        attitude and, optionally, its derivatives.

        With :math:`C = [\\hat{\\mathbf{x}}\\;\\hat{\\mathbf{y}}\\;\\hat{\\mathbf{z}}]`
        (see the class docstring) the derivatives are returned per axis,
        ``D[j]`` being the Jacobian of axis :math:`j`. With
        :math:`\\hat{\\mathbf{w}}` the unit heliocentric velocity and
        :math:`\\mathbf{z}_0 = \\hat{\\mathbf{x}}\\times\\hat{\\mathbf{w}}`:

        .. math::

            \\frac{\\partial\\hat{\\mathbf{x}}}{\\partial\\mathbf{r}}
                = \\frac{I - \\hat{\\mathbf{x}}\\hat{\\mathbf{x}}^T}
                        {\\lVert\\mathbf{s}\\rVert},\\quad
            \\frac{\\partial\\hat{\\mathbf{z}}}{\\partial\\mathbf{r}}
                = -\\frac{I - \\hat{\\mathbf{z}}\\hat{\\mathbf{z}}^T}
                         {\\lVert\\mathbf{z}_0\\rVert}
                  [\\hat{\\mathbf{w}}]_\\times
                  \\frac{\\partial\\hat{\\mathbf{x}}}{\\partial\\mathbf{r}},\\quad
            \\frac{\\partial\\hat{\\mathbf{y}}}{\\partial\\mathbf{r}}
                = [\\hat{\\mathbf{z}}]_\\times
                  \\frac{\\partial\\hat{\\mathbf{x}}}{\\partial\\mathbf{r}}
                - [\\hat{\\mathbf{x}}]_\\times
                  \\frac{\\partial\\hat{\\mathbf{z}}}{\\partial\\mathbf{r}}

        and, since :math:`\\hat{\\mathbf{x}}` does not depend on the velocity,

        .. math::

            \\frac{\\partial\\hat{\\mathbf{z}}}{\\partial\\mathbf{v}}
                = \\frac{I - \\hat{\\mathbf{z}}\\hat{\\mathbf{z}}^T}
                        {\\lVert\\mathbf{z}_0\\rVert}
                  [\\hat{\\mathbf{x}}]_\\times
                  \\frac{I - \\hat{\\mathbf{w}}\\hat{\\mathbf{w}}^T}
                        {\\lVert\\mathbf{w}\\rVert},\\quad
            \\frac{\\partial\\hat{\\mathbf{y}}}{\\partial\\mathbf{v}}
                = -[\\hat{\\mathbf{x}}]_\\times
                   \\frac{\\partial\\hat{\\mathbf{z}}}{\\partial\\mathbf{v}}

        The Sun->origin terms do not depend on the spacecraft state, so these
        are the derivatives with respect to the spacecraft position
        :math:`\\mathbf{r}` and velocity :math:`\\mathbf{v}` relative to the
        origin. A spacecraft-fixed vector then has
        :math:`\\partial\\mathbf{e}/\\partial\\mathbf{r} = \\sum_j e_{b,j} D_r[j]`
        (:meth:`body_vector_jacobian`).

        Parameters
        ----------
        sun2body : numpy.ndarray or ArrayBox
            ``(3,)`` Sun->spacecraft position, km.
        sun2body_vel : numpy.ndarray or ArrayBox
            ``(3,)`` Sun->spacecraft velocity, km/s.
        derivatives : bool, optional
            Also return the derivatives. Defaults to ``False``.

        Returns
        -------
        tuple
            ``(C, D_r, D_v)``: ``C`` is the ``(3, 3)`` spacecraft-fixed to
            inertial DCM, traced when the inputs are; ``D_r`` and ``D_v`` are
            plain ``(3, 3, 3)`` arrays, or ``None`` when ``derivatives`` is
            false.
        """
        traced = isinstance(sun2body, np.autograd_types) or isinstance(
            sun2body_vel, np.autograd_types
        )
        if derivatives and not traced:
            key = (
                np.asarray(sun2body, dtype=float).tobytes(),
                np.asarray(sun2body_vel, dtype=float).tobytes(),
            )
            memo_key, memo = cls._nadir_memo
            if memo_key == key:
                return memo

        s_norm = np.sqrt(np.dot(sun2body, sun2body))
        x_b = sun2body / s_norm
        w_norm = np.sqrt(np.dot(sun2body_vel, sun2body_vel))
        w_hat = sun2body_vel / w_norm
        z_raw = cls.cross(x_b, w_hat)
        z_norm = np.sqrt(np.dot(z_raw, z_raw))
        z_b = z_raw / z_norm
        y_b = cls.cross(z_b, x_b)
        # Stacked, not assigned into, so a traced position stays traced.
        dcm = np.stack([x_b, y_b, z_b], axis=1)

        if not derivatives:
            return dcm, None, None

        # Derivatives are for the analytic partials: plain values only.
        x, w, z = cls._raw(x_b), cls._raw(w_hat), cls._raw(z_b)
        s_n, w_n, z_n = cls._raw(s_norm), cls._raw(w_norm), cls._raw(z_norm)
        eye = np.eye(3)
        P_z = (eye - np.outer(z, z)) / z_n
        x_skew, z_skew = cls.skew(x), cls.skew(z)

        dx_dr = (eye - np.outer(x, x)) / s_n
        dz_dr = P_z @ (-cls.skew(w) @ dx_dr)
        dy_dr = z_skew @ dx_dr - x_skew @ dz_dr
        D_r = np.stack([dx_dr, dy_dr, dz_dr])

        dw_dv = (eye - np.outer(w, w)) / w_n
        dz_dv = P_z @ (x_skew @ dw_dv)
        dy_dv = -x_skew @ dz_dv
        D_v = np.stack([np.zeros((3, 3)), dy_dv, dz_dv])

        if not traced:
            for array in (dcm, D_r, D_v):
                array.flags.writeable = False
            cls._nadir_memo = (key, (dcm, D_r, D_v))
        return dcm, D_r, D_v

    @classmethod
    def rtn(cls, position, velocity, derivatives: bool = False) -> tuple:
        """
        Radial / along-track / cross-track (RTN) to inertial rotation of an
        orbit and, optionally, its derivatives.

        With the angular momentum :math:`\\mathbf{h} = \\mathbf{r}\\times\\mathbf{v}`,

        .. math::

            \\hat{\\mathbf{R}} = \\frac{\\mathbf{r}}{\\lVert\\mathbf{r}\\rVert},\\quad
            \\hat{\\mathbf{N}} = \\frac{\\mathbf{h}}{\\lVert\\mathbf{h}\\rVert},\\quad
            \\hat{\\mathbf{T}} = \\hat{\\mathbf{N}}\\times\\hat{\\mathbf{R}},\\quad
            C = \\left[\\hat{\\mathbf{R}}\\;\\hat{\\mathbf{T}}\\;\\hat{\\mathbf{N}}\\right]

        (the triad is also called RSW or RIC). The derivatives are returned per
        axis, as for :meth:`nadirPointingSun`:

        .. math::

            \\frac{\\partial\\hat{\\mathbf{R}}}{\\partial\\mathbf{r}}
                = \\frac{I - \\hat{\\mathbf{R}}\\hat{\\mathbf{R}}^T}
                        {\\lVert\\mathbf{r}\\rVert},\\quad
            \\frac{\\partial\\hat{\\mathbf{N}}}{\\partial\\mathbf{r}}
                = -P_h[\\mathbf{v}]_\\times,\\quad
            \\frac{\\partial\\hat{\\mathbf{N}}}{\\partial\\mathbf{v}}
                = P_h[\\mathbf{r}]_\\times,\\quad
            P_h = \\frac{I - \\hat{\\mathbf{N}}\\hat{\\mathbf{N}}^T}{\\lVert\\mathbf{h}\\rVert}

        and :math:`d\\hat{\\mathbf{T}} = -[\\hat{\\mathbf{R}}]_\\times d\\hat{\\mathbf{N}}
        + [\\hat{\\mathbf{N}}]_\\times d\\hat{\\mathbf{R}}`. The radial axis does not
        depend on the velocity, so ``D_v[0]`` is zero.

        Parameters
        ----------
        position : numpy.ndarray or ArrayBox
            ``(3,)`` position about the body the orbit is described about, km.
        velocity : numpy.ndarray or ArrayBox
            ``(3,)`` velocity about the same body, km/s.
        derivatives : bool, optional
            Also return the derivatives. Defaults to ``False``.

        Returns
        -------
        tuple
            ``(C, D_r, D_v)``: ``C`` is the ``(3, 3)`` RTN to inertial DCM,
            traced when the inputs are; ``D_r`` and ``D_v`` are plain
            ``(3, 3, 3)`` arrays, or ``None`` when ``derivatives`` is false.

        Raises
        ------
        DegenerateGeometryError
            If the radius or the angular momentum vanishes.
        """
        r_norm = np.sqrt(np.dot(position, position))
        h = cls.cross(position, velocity)
        h_norm = np.sqrt(np.dot(h, h))
        if r_norm == 0.0 or h_norm == 0.0:
            raise cls.DegenerateGeometryError(
                "the RTN triad is undefined for a state with zero radius or "
                "zero angular momentum (a purely radial trajectory has no "
                "along-track direction)."
            )
        r_hat = position / r_norm
        n_hat = h / h_norm
        t_hat = cls.cross(n_hat, r_hat)
        # Stacked, not assigned into, so a traced position stays traced.
        dcm = np.stack([r_hat, t_hat, n_hat], axis=1)

        if not derivatives:
            return dcm, None, None

        # Derivatives are for the analytic partials: plain values only.
        r, v = cls._raw(position), cls._raw(velocity)
        R, N = cls._raw(r_hat), cls._raw(n_hat)
        eye = np.eye(3)
        P_h = (eye - np.outer(N, N)) / cls._raw(h_norm)
        R_skew, N_skew = cls.skew(R), cls.skew(N)

        dR_dr = (eye - np.outer(R, R)) / cls._raw(r_norm)
        dN_dr = -P_h @ cls.skew(v)
        dT_dr = -R_skew @ dN_dr + N_skew @ dR_dr
        D_r = np.stack([dR_dr, dT_dr, dN_dr])

        dN_dv = P_h @ cls.skew(r)
        dT_dv = -R_skew @ dN_dv
        D_v = np.stack([np.zeros((3, 3)), dT_dv, dN_dv])
        return dcm, D_r, D_v

    @staticmethod
    def body_vector_jacobian(vectors_b: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        Jacobian of spacecraft-fixed vectors mapped to the inertial frame,
        :math:`\\partial(C\\mathbf{e}_b)/\\partial\\mathbf{q} = \\sum_j e_{b,j} D[j]`.

        Parameters
        ----------
        vectors_b : numpy.ndarray
            ``(3,)`` vector, or ``(..., 3)`` stack of vectors, in the
            spacecraft frame.
        D : numpy.ndarray
            ``(3, 3, 3)`` axis derivatives from :meth:`nadirPointingSun` or
            :meth:`rtn`.

        Returns
        -------
        numpy.ndarray
            ``(3, 3)`` Jacobian, or ``(..., 3, 3)`` for a stack.
        """
        return np.einsum("...j,jab->...ab", vectors_b, D)

    # endregion Attitude Methods #
    # -------------------------#

    # ----------------------#
    # region Frame Methods #
    # ----------------------#
    @staticmethod
    def frame_name(frame):
        """
        Reduce a Frame or string to the frame name SPICE expects.

        Parameters
        ----------
        frame : Frame or str or None
            Frame to resolve.

        Returns
        -------
        str or None
            The frame name, or ``None`` when nothing was supplied.
        """
        if frame is None:
            return None
        return str(getattr(frame, "name", frame))

    @classmethod
    def resolve_frame(
        cls, frame="inertial", body_fixed_frame=None, relative_body=None
    ) -> tuple:
        """
        Validate the frame settings of a frame-expressed acceleration.

        Parameters
        ----------
        frame : str, optional
            ``"inertial"``, ``"body_fixed"`` or ``"RTN"``, in any case.
            Defaults to ``"inertial"``.
        body_fixed_frame : str or Frame, optional
            Frame the components are expressed in, for ``frame="body_fixed"``
            (for example ``'IAU_BENNU'``).
        relative_body : str or Body, optional
            Body the RTN triad is built about, when that is not the
            propagation origin: a lunar RTN frame while propagating an
            Earth-centred arc. Defaults to the propagation origin.

        Returns
        -------
        tuple
            ``(frame, body_fixed_frame, relative_body)``: the frame key as
            spelled in :attr:`ACCELERATION_FRAME_AXES`, and the two
            names, ``None`` where unset.

        Raises
        ------
        InvalidFrameError
            If ``frame`` is not a key of :attr:`ACCELERATION_FRAME_AXES`.
        MissingBodyFixedFrameError
            If ``frame="body_fixed"`` is requested without ``body_fixed_frame``.
        """
        axes = cls.ACCELERATION_FRAME_AXES
        # Matched without regard to case, so "rtn" reads as "RTN".
        frame_key = {key.lower(): key for key in axes}.get(str(frame).strip().lower())
        if frame_key is None:
            raise cls.InvalidFrameError(
                f"frame must be one of {sorted(axes)}, got {frame!r}."
            )
        if frame_key == "body_fixed" and body_fixed_frame is None:
            raise cls.MissingBodyFixedFrameError(
                "frame='body_fixed' needs 'body_fixed_frame', the frame the "
                "components are expressed in (for example 'IAU_BENNU')."
            )
        relative_name = (
            getattr(relative_body, "name", relative_body)
            if relative_body is not None
            else None
        )
        return frame_key, cls.frame_name(body_fixed_frame), relative_name

    @classmethod
    def relative_state(cls, model, position, current_state: dict, epoch, body) -> tuple:
        """
        Spacecraft position and velocity about the body the orbit is described
        about.

        Identical to the propagated state when ``model.relative_body`` is
        unset. When it is set, the body's own state is subtracted, so the RTN
        triad describes the orbit about *that* body.

        The offset comes from the ephemeris and so does not depend on the
        spacecraft state: the partials of anything built from the relative
        state are therefore unchanged in form, since
        :math:`\\partial \\mathbf{r}_{rel}/\\partial \\mathbf{r} = I`.

        Parameters
        ----------
        model : DynamicModel
            Frame-expressed acceleration; reads ``relative_body``,
            ``ref_frame`` and ``origin_name``.
        position : numpy.ndarray
            Spacecraft position 3-vector from the propagation origin, km.
        current_state : dict
            The propagated state.
        epoch : float
            Current epoch in ephemeris seconds (TDB).
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        tuple of numpy.ndarray
            ``(r, v)`` relative to ``relative_body``, in km and km/s.

        Raises
        ------
        MissingEpochError
            If ``relative_body`` is set and no epoch is supplied.
        MissingVelocityError
            If the state carries no velocity for ``body``.
        """
        r = getattr(position, "values", position)
        v = cls.velocity_from_state(current_state, body)
        # A state traced by autograd stays a box, so the derivative survives.
        if not isinstance(r, np.autograd_types):
            r = np.asarray(r, dtype=float)
        if not isinstance(v, np.autograd_types):
            v = np.asarray(v, dtype=float)
        r, v = np.ravel(r), np.ravel(v)

        relative_body = model.relative_body
        if relative_body is None:
            return r, v

        if epoch is None:
            raise cls.MissingEpochError(
                f"an epoch is required to locate '{relative_body}'; the RTN "
                "frame is referred to that body, not to the propagation origin."
            )

        memo = getattr(model, "_frame_memo", None)
        hit = memo.get("offset") if memo is not None else None
        if hit is None or hit[0] != epoch:
            state = SpiceManager.get_state(
                relative_body,
                epoch,
                cls.frame_name(model.ref_frame),
                model.origin_name,
            )
            offset = np.asarray(getattr(state, "values", state), dtype=float)
            hit = (epoch, offset.ravel())
            if memo is not None:
                memo["offset"] = hit

        offset = hit[1]
        return r - offset[:3], v - offset[3:6]

    @classmethod
    def body_fixed_rotation(cls, model, epoch) -> np.ndarray:
        """
        Body-fixed to propagation-frame rotation :math:`R_{bf\\to rf}(t)`.

        Parameters
        ----------
        model : DynamicModel
            Frame-expressed acceleration; reads ``body_fixed_frame`` and
            ``ref_frame``.
        epoch : float
            Epoch of the rotation, in ephemeris seconds (TDB).

        Returns
        -------
        numpy.ndarray
            ``(3, 3)`` rotation matrix.

        Raises
        ------
        MissingEpochError
            If no epoch is supplied.
        """
        if epoch is None:
            raise cls.MissingEpochError(
                f"an epoch is required to rotate a body-fixed "
                f"{type(model).__name__} out of '{model.body_fixed_frame}'."
            )

        memo = getattr(model, "_frame_memo", None)
        hit = memo.get("rotation") if memo is not None else None
        if hit is None or hit[0] != epoch:
            hit = (
                epoch,
                SpiceManager.get_xfrm(
                    model.body_fixed_frame, cls.frame_name(model.ref_frame), epoch
                ),
            )
            if memo is not None:
                memo["rotation"] = hit
        return hit[1]

    @classmethod
    def rtn_triad(
        cls,
        model,
        position,
        current_state: dict,
        epoch,
        body,
        derivatives: bool = False,
    ) -> tuple:
        """
        RTN triad of the orbit about the model's relative body.

        :meth:`relative_state` followed by :meth:`rtn`.

        Parameters
        ----------
        model : DynamicModel
            Frame-expressed acceleration.
        position : numpy.ndarray
            Spacecraft position 3-vector, km.
        current_state : dict
            The propagated state.
        epoch : float
            Current epoch in ephemeris seconds (TDB).
        body : Body
            Body the acceleration is evaluated for.
        derivatives : bool, optional
            Also return the position and velocity derivatives of the axes.

        Returns
        -------
        tuple
            ``(T, D_r, D_v)``: the ``(3, 3)`` matrix
            :math:`[\\hat{\\mathbf{R}}\\;\\hat{\\mathbf{T}}\\;\\hat{\\mathbf{N}}]`
            and, with ``derivatives``, the ``(3, 3, 3)`` stacks whose ``[j]``
            is the Jacobian of axis ``j`` (``None`` otherwise).

        Raises
        ------
        DegenerateGeometryError
            If the radius or the angular momentum vanishes.
        """
        r, v = cls.relative_state(model, position, current_state, epoch, body)
        return cls.rtn(r, v, derivatives)

    @classmethod
    def frame_matrix(cls, model, position, current_state: dict, epoch, body):
        """
        Rotation from the component frame into the propagation frame.

        Parameters
        ----------
        model : DynamicModel
            Frame-expressed acceleration.
        position : numpy.ndarray
            Spacecraft position 3-vector, km.
        current_state : dict
            The propagated state.
        epoch : float
            Current epoch in ephemeris seconds (TDB).
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        numpy.ndarray or None
            ``(3, 3)`` rotation, or ``None`` for the inertial frame, where the
            components already live in the propagation frame.
        """
        frame = model.frame
        if frame == "inertial":
            return None
        if frame == "body_fixed":
            return cls.body_fixed_rotation(model, epoch)
        return cls.rtn_triad(model, position, current_state, epoch, body)[0]

    @classmethod
    def from_frame(cls, model, values, position, current_state: dict, epoch, body):
        """
        Rotate frame-local components into the propagation frame.

        Applied to the acceleration, :math:`\\mathbf{a} = T\\,\\mathbf{a}^{(f)}`,
        and to its partial with respect to a parameter the frame does not
        depend on, :math:`\\partial\\mathbf{a}/\\partial\\mathbf{p} =
        T\\,\\partial\\mathbf{a}^{(f)}/\\partial\\mathbf{p}`.

        Parameters
        ----------
        model : DynamicModel
            Frame-expressed acceleration.
        values : numpy.ndarray
            3-vector or ``(3, n)`` matrix of frame-local components.
        position : numpy.ndarray
            Spacecraft position 3-vector, km.
        current_state : dict
            The propagated state.
        epoch : float
            Current epoch in ephemeris seconds (TDB).
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        numpy.ndarray
            ``values`` expressed in the propagation frame.
        """
        T = cls.frame_matrix(model, position, current_state, epoch, body)
        return values if T is None else T @ values

    @classmethod
    def frame_partials(
        cls, model, a_frame, position, current_state: dict, epoch, body
    ) -> tuple:
        """
        Position and velocity partials of :math:`T\\,\\mathbf{a}^{(f)}` for
        frame-local components that do not depend on the state.

        Only the RTN triad is built from the state, so

        .. math::

            \\frac{\\partial \\mathbf{a}}{\\partial \\mathbf{r}}
                = \\sum_j a^{(f)}_j \\,
                  \\frac{\\partial \\hat{\\mathbf{e}}_j}{\\partial \\mathbf{r}},
            \\qquad
            \\frac{\\partial \\mathbf{a}}{\\partial \\mathbf{v}}
                = \\sum_j a^{(f)}_j \\,
                  \\frac{\\partial \\hat{\\mathbf{e}}_j}{\\partial \\mathbf{v}},

        with :math:`\\hat{\\mathbf{e}}_j` the RTN axes. The inertial and
        body-fixed rotations depend on the epoch alone, so both blocks are
        zero there.

        Parameters
        ----------
        model : DynamicModel
            Frame-expressed acceleration.
        a_frame : numpy.ndarray
            Frame-local acceleration 3-vector, km/s^2.
        position : numpy.ndarray
            Spacecraft position 3-vector, km.
        current_state : dict
            The propagated state.
        epoch : float
            Current epoch in ephemeris seconds (TDB).
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        tuple of numpy.ndarray
            ``(da_dr, da_dv)``, each ``(3, 3)``.
        """
        if model.frame != "RTN":
            return np.zeros((3, 3)), np.zeros((3, 3))
        _, D_r, D_v = cls.rtn_triad(
            model, position, current_state, epoch, body, derivatives=True
        )
        return (
            cls.body_vector_jacobian(a_frame, D_r),
            cls.body_vector_jacobian(a_frame, D_v),
        )

    # endregion Frame Methods #
    # ----------------------#
