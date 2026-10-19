# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    CelestialBody,
    DynamicModel,
    Frame,
    Units,
    ArrayWUnits,
    EpochArray,
    Body,
    SpiceManager,
    StateArray,
    Spacecraft,
    nPlateModel,
    DynamicalUtils,
)
from scarabaeus import constants
from scarabaeus.utils.EphemerisCache import build_ephemeris_cache
import scarabaeus.utils.NumpyWrapper as np

try:
    from scarabaeus.rust import NPlateSRPCore as _NPlateRustCore
except ImportError:
    _NPlateRustCore = None

_SUN_SPICE_ID = 10

# ------------------#
#  Generate Units  #
# ------------------#
kg, km, sec, N, unitless = Units.get_units(["kg", "km", "sec", "N", "unitless"])
J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


# --------------------#
#  Class Definition  #
# --------------------#
class nPlateSRP(DynamicModel):
    """ Models the acceleration due to Solar Radiation Pressure (SRP) using an N-plate model.

        The spacecraft is decomposed into :math:`N` flat plates, each with its own
        area, surface normal, specular reflectivity :math:`\\rho_s`, and diffuse
        reflectivity :math:`\\rho_d`.  For each illuminated plate :math:`i`
        (i.e., :math:`\\hat{\\mathbf{n}}_i \\cdot \\hat{\\mathbf{s}} > 0`), the
        SRP acceleration contribution is:

        .. math::

            \\mathbf{a}_i =
                -\\frac{\\nu\\, P_\\odot}{m}
                \\left(\\frac{r_\\odot}{r}\\right)^2
                A_i
                \\left(\\hat{\\mathbf{n}}_i \\cdot \\hat{\\mathbf{r}}_\\odot\\right)
                \\left[
                    (1 - \\rho_s)\\,\\hat{\\mathbf{r}}_\\odot
                    + 2\\rho_s
                    \\left(\\hat{\\mathbf{n}}_i \\cdot \\hat{\\mathbf{r}}_\\odot\\right)
                    \\hat{\\mathbf{n}}_i
                    + \\tfrac{2}{3}\\rho_d\\,\\hat{\\mathbf{n}}_i
                \\right]

        where :math:`\\nu` is the shadow factor (0 in eclipse, 1 in full Sun),
        :math:`P_\\odot` is the solar radiation pressure at 1 AU,
        :math:`r_\\odot / r` is the inverse-square scaling with heliocentric
        distance, and :math:`\\hat{\\mathbf{r}}_\\odot` is the unit vector from
        the spacecraft to the Sun (direction of incoming radiation); the dot
        product :math:`\\hat{\\mathbf{n}}_i \\cdot \\hat{\\mathbf{r}}_\\odot > 0`
        selects illuminated plates, and the leading minus sign yields acceleration
        directed away from the Sun.  The total acceleration is:

        .. math::

            \\mathbf{a}_\\text{SRP} = \\eta_\\text{SRP} \\sum_{i=1}^{N} \\mathbf{a}_i

        where :math:`\\eta_\\text{SRP}` is an estimable scale factor (default 1)
        retrieved from the state vector via a ``("eta_srp", spice_id)`` entry
        and can be solved for in the OD process.

        A double-sided plate (``nPlateModel.double_sided``) lit from behind
        (:math:`\\hat{\\mathbf{n}}_i \\cdot \\hat{\\mathbf{r}}_\\odot < 0`)
        contributes as a plate with normal :math:`-\\hat{\\mathbf{n}}_i`.

        Plates in the spacecraft body frame (Fixed plates without a frame) are
        oriented by the spacecraft ``attitudeMode`` (see
        :class:`~scarabaeus.DynamicalUtils`): a C-kernel, an inertially fixed
        DCM, or ``nadirPointingSun``. The nadir attitude follows the
        spacecraft state, so its position partial carries the attitude terms
        and its velocity partial is non-zero (``depends_on_velocity``).
        CK plates and Fixed plates in a named SPICE frame are oriented by
        SPICE alone, so their contribution does not depend on the
        ``attitudeMode``; a model with no body-frame plate needs none. The two
        groups can be mixed under any attitude (a nadir-pointed bus with
        solar arrays on their own CK, say): each is summed in its own frame.

        Parameters
        ----------
        origin : CelestialBody
            Propagation origin. Used for the Sun-relative geometry; the estimable
            ``eta_srp`` is read from the propagated state at evaluation time.

        spacecraft : Spacecraft
            The spacecraft containing physical and plate model properties.

        Raises
        ------
        ValueError
            If any plate area in the n-plate model is not expressed in km^2.

        See Also
        --------
        scarabaeus.StateArray : Represents state vectors with origins and reference frames.
        scarabaeus.Spacecraft : Defines spacecraft properties including mass and area.
        scarabaeus.Propagator : Uses this model when ``nPlate_SRP = True``.
        scarabaeus.CannonballSRP : Lower fidelity model for calculating SRP.

        Notes
        -----
        The Sun’s position is retrieved dynamically from SPICE data.

        References
        ----------
        .. [1] P. W. Kenneally, “High geometric fidelity solar radiation pressure modeling via graphics processing unit,”
           Master’s thesis, Colorado Boulder, 2016.
    """

    # ------------------#
    # region   Errors  #
    # ------------------#
    # Custom exceptions: named on the model and raised on the
    # line that detects the failure, so tracebacks point at the real cause
    # rather than a generic Exception/RuntimeError.
    class RustCoreUnavailableError(Exception):
        """The rust N-plate SRP core is unavailable.

        Raised by the compute and partial methods when ``_rust_cores`` could not
        be built because the extension is not importable. The Rust kernels are
        the only implementation of the acceleration and its position partial.
        """

    class AutogradTracingError(Exception):
        """Autograd is tracing the acceleration through the Rust kernels.

        The autograd NumPy backend (``UseAutoGrad=TRUE``) is fine on its own —
        the model runs on plain arrays under it. Only a traced position or
        eta_srp cannot pass into Rust. The partials are verified by the Rust
        dual-number differentiation instead
        (``NPlateSRPCore.compute_partials_ad``), which is what
        ``check_force_model_partials=TRUE`` runs.
        """

    # endregion Errors #
    # ------------------#

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        origin: CelestialBody,
        spacecraft: Spacecraft,
        base_frame: Frame,
        t0: float = None,
        tf: float = None,
    ):
        # Origin and reference frame
        self._origin = origin
        self._origin_name = self._origin.name
        self._frame = base_frame
        self._spacecraft = spacecraft
        # Precompute the SPICE name for the origin body so compute_acceleration does not
        # call eval() on every ODE function evaluation.
        self._origin_spice_name = getattr(constants, self._origin_name.upper()).spice_name

        # Body-specific parameters
        self._area = ArrayWUnits.get_value_in_target_units(self._spacecraft.area, km**2)
        self._ref_coeff = ArrayWUnits.get_value_in_target_units(
            self._spacecraft.ref_coeff, unitless
        )

        # Plate-specific parameters
        self._plate_model = self.spacecraft.n_plate_model

        # Constants
        self._P_SRP = ArrayWUnits.get_value_in_target_units(
            constants.srp_constant2, kg * km * sec**-2
        )

        # eta_srp is read from the propagated state on every evaluation (see
        # _get_eta_srp). The map is kept only so callers that poke a value in
        # directly — the partial-derivative diagnostics do — still work.
        self._eta_srp_map: dict[int, float] = {}

        # Initialization check on the correct units for nPlate areas
        for ii in range(len(self.plate_model.names)):
            if self.plate_model.areas[ii].units != km**2:
                raise ValueError(f"units of nPlate areas not compliant.")

        # Precompute constant plate geometry as raw numpy arrays so the per-RHS
        # loop doesn't re-extract .values / re-index the plate model each step.
        _nplates = len(self.plate_model.names)
        self._plate_areas = np.array(
            [self.plate_model.areas[ii].values for ii in range(_nplates)], dtype=float
        )  # km^2
        self._plate_diff_ref = np.array(
            [self.plate_model.diff_ref_coeffs[ii] for ii in range(_nplates)], dtype=float
        )
        self._plate_spec_ref = np.array(
            [self.plate_model.spec_ref_coeffs[ii] for ii in range(_nplates)], dtype=float
        )
        # Plates whose normals are given in the body frame; the others (CK and
        # named-frame plates) come back from the plate model in base_frame.
        # Each group runs in its own rust core (_build_rust_cores).
        self._body_plates = np.array(self.plate_model.body_frame_plates, dtype=bool)

        # Sun-position source. When the origin IS the Sun the vector is
        # identically zero. Otherwise, if the propagation span is known and
        # the frame is the SPK-native J2000, cache the Chebyshev records so
        # _sun_to_origin avoids the per-step spkpos; SPICE stays the fallback.
        self._origin_spice_id = getattr(self._origin, "spice_id", None)
        self._origin_is_sun = self._origin_spice_id == _SUN_SPICE_ID
        self._sun_eph = None
        self._sun_eph_span = None
        frame_name = getattr(base_frame, "name", str(base_frame))
        self._frame_name = frame_name
        if (
            t0 is not None
            and tf is not None
            and self._origin_spice_id is not None
            and not self._origin_is_sun
            and frame_name.upper() == "J2000"
        ):
            self._sun_eph_span = (t0 - 86400.0, tf + 86400.0)
            self._sun_eph = self._build_sun_ephemeris()

        # Plate sums and the SC-fixed <-> inertial mapping run in rust — the
        # only compute path. Unavailable only when the extension is missing;
        # the compute methods then raise (_require_rust_core). Built under the
        # autograd backend too: the propagation evaluates the model on plain
        # arrays.
        self._rust_cores = self._build_rust_cores()

    def _build_rust_cores(self):
        """
        One rust core per non-empty plate group, or ``None`` without the
        extension.

        The body-frame plates are turned by the attitude. The CK and
        named-frame plates come back from the plate model already in
        ``base_frame``, so they are evaluated with the identity: they do not
        turn with an attitude that follows the state (``nadirPointingSun``),
        and need no ``attitudeMode`` at all. The kernel is rotation
        covariant, so the two group sums add up to the acceleration.

        Returns
        -------
        tuple or None
            ``(plates, body_frame, core)`` per group: the plate mask, whether
            the group is in the body frame, and its ``NPlateSRPCore``.
        """
        if _NPlateRustCore is None:
            return None
        double_sided = np.array(self.plate_model.double_sided, dtype=bool)
        cores = []
        for plates, body_frame in (
            (self._body_plates, True),
            (~self._body_plates, False),
        ):
            if not plates.any():
                continue
            core = _NPlateRustCore(
                self._P_SRP,
                self._plate_areas[plates],
                self._plate_diff_ref[plates],
                self._plate_spec_ref[plates],
                double_sided=[bool(flag) for flag in double_sided[plates]],
            )
            cores.append((plates, body_frame, core))
        return tuple(cores)

    def __getstate__(self):
        # Compiled extension objects are not picklable; rebuilt in __setstate__.
        state = self.__dict__.copy()
        state["_rust_cores"] = None
        state["_sun_eph"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._rust_cores = self._build_rust_cores()
        if self.__dict__.get("_sun_eph_span") is not None:
            self._sun_eph = self._build_sun_ephemeris()

    def _build_sun_ephemeris(self):
        """
        Cache the origin and Sun Chebyshev records over ``_sun_eph_span``.

        Returns
        -------
        EphemerisCache or None
            The cache, or ``None`` when the loaded SPKs do not cover the span;
            the Sun position then falls back to SPICE.
        """
        try:
            return build_ephemeris_cache(
                [self._origin_spice_id, _SUN_SPICE_ID], *self._sun_eph_span
            )
        except Exception:
            return None  # no/partial SPK coverage -> SPICE fallback

    def _require_rust_core(self, *inputs):
        """
        Guard for the compute methods: the rust core is the only
        implementation of the N-plate kernels.

        The autograd NumPy backend alone is not an error — only an input that
        autograd is actually tracing, since it cannot enter Rust.

        Parameters
        ----------
        *inputs
            The differentiable inputs handed to Rust (position, eta_srp).

        Raises
        ------
        RustCoreUnavailableError
            If the rust cores were not built.
        AutogradTracingError
            If autograd is tracing any of ``inputs``.
        """
        if self._rust_cores is None:
            raise self.RustCoreUnavailableError(
                "nPlateSRP requires the rust extension. Build it with "
                "`maturin develop --release` from the repository root."
            )
        if any(isinstance(x, np.autograd_types) for x in inputs):
            raise self.AutogradTracingError(
                "nPlateSRP evaluates its plate sums in Rust, which autograd "
                "cannot trace. Verify its partials with "
                "check_force_model_partials=TRUE, which uses the Rust "
                "dual-number differentiation (compute_partials_ad)."
            )

    # -------------------------#
    # region ====> Properties #
    # -------------------------#
    @property
    def plate_model(self) -> nPlateModel:
        """ The associated nPlateModel used to calculate SRP. """
        return self._plate_model

    @property
    def spacecraft(self) -> Spacecraft:
        """
        The spacecraft associated with the SRP model.
        """
        return self._spacecraft

    @property
    def depends_on_velocity(self) -> bool:
        """ Whether the acceleration depends on the spacecraft velocity.

            True only for the ``nadirPointingSun`` attitude, where the
            spacecraft Y axis follows the heliocentric velocity, and only when
            some plate is in the body frame.
        """
        return bool(self._body_plates.any()) and DynamicalUtils.depends_on_velocity(
            self._spacecraft
        )

    @property
    def origin(self) -> Body:
        """
        The celestial body that serves as the origin for the state vector.
        """
        return self._origin

    @property
    def origin_name(self) -> str:
        """
        The name of the origin body.
        """
        return self._origin_name

    @property
    def frame(self) -> str:
        """
        The reference frame in which SRP calculations are performed.
        """
        return self._frame

    @property
    def area(self) -> float:
        """
        The total surface area of the spacecraft. Expressed in units of square meters.
        """
        return self._area

    @property
    def ref_coeff(self) -> float:
        """
        The reflectivity coefficient of the spacecraft. Unitless.
        """
        return self._ref_coeff

    @property
    def P_SRP(self) -> float:
        """
        The precomputed SRP constant for scaling factors. Expressed in units of :math:`\\frac{N}{m^2}`.
        """
        return self._P_SRP

    @property
    def eta_srp(self) -> float:
        """
        The SRP scale factor. Unitless.
        """
        return self._eta_srp

    @eta_srp.setter
    def eta_srp(self, input_val):
        if input_val is None:
            input_val = 0

        self._eta_srp = input_val

    # endregion => Properties  #
    # -------------------------#

    # ----------------#
    # region Methods #
    # ----------------#

    def _get_eta_srp(self, current_state: dict, body: Body) -> float:
        """
        Resolve eta_srp for the given body.

        Priority:
        1. The propagated state, ``current_state[("eta_srp", spice_id)]``
        2. Any value pushed into ``self._eta_srp_map`` by a caller
        3. Fallback: 1.0

        The state comes first so that an estimated eta_srp takes effect, which
        also matches how every other model resolves its parameters.
        """
        spice_id = getattr(body, "spice_id", None)

        if spice_id is not None:
            key = ("eta_srp", spice_id)
            if key in current_state:
                val = current_state[key]
                return self._scalar_from_state(val)

            if spice_id in self._eta_srp_map:
                return self._eta_srp_map[spice_id]

        return 1.0

    def inertial_to_body_DCM(self, epoch: EpochArray, tol: int = 50) -> float:
        """
        Computes the Direction Cosine Matrix (DCM) to transform vectors from the inertial
        frame to the spacecraft-fixed frame.

        Parameters
        ----------
        epoch : EpochArray
            The current epoch given in ephemeris time (seconds past J2000).

        tol : int, optional
            The time tolerance for retrieving attitude and angular velocity from the
            specified spacecraft clock time. Defaults to ``50``.

        Returns
        -------
        dcm : np.ndarray
            The DCM for transforming inertial vectors to spacecraft-fixed coordinates.

        Notes
        -----
        The ``epoch`` parameter is converted from ephemeris time to continuous encoded spacecraft clock "ticks". Due to
        this, the time tolerance should be given as sclock ticks and not seconds.
        """
        # Default DCM is the identity matrix
        default_DCM = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Attempt to get the attitude from the SPICE Manager
        try:
            # Attempt to get the attitude from the SPICE Manager
            dcm, _, _ = SpiceManager.get_attitude(
                self.spacecraft.spice_id, epoch, self.frame, tol
            )
            return dcm

        except Exception:
            return default_DCM

    def _sun_to_origin(self, epoch: float) -> np.ndarray:
        """
        Sun->origin position (km, raw numpy), memoized per epoch.

        ``compute_acceleration`` and ``_compute_partial_by_position`` both need it
        at the same epoch within one RHS step, so the SPICE lookup is cached and
        reused (the origin SPICE name is already resolved at construction).

        Parameters
        ----------
        epoch : float
            TDB epoch (seconds past J2000).

        Returns
        -------
        numpy.ndarray
            ``(3,)`` Sun->origin position in km.
        """
        if self.__dict__.get("_s2o_epoch") != epoch:
            self._s2o_cache = self._lookup_sun_to_origin(epoch)
            self._s2o_epoch = epoch
        return self._s2o_cache

    def _lookup_sun_to_origin(self, epoch: float) -> np.ndarray:
        """
        Uncached Sun->origin lookup: identically zero for a heliocentric
        origin; Chebyshev ephemeris cache when available (falling back to
        SPICE for epochs outside the cached span); SPICE otherwise.
        """
        if self._origin_is_sun:
            return np.zeros(3)
        if self._sun_eph is not None:
            try:
                return np.asarray(
                    self._sun_eph.get_relative_position(
                        self._origin_spice_id, _SUN_SPICE_ID, epoch
                    ),
                    dtype=float,
                )
            except Exception:
                pass  # epoch outside the cached span -> SPICE fallback
        return SpiceManager.get_pos_values(
            self._origin_spice_name, epoch, self._frame, "SUN"
        )

    def _sun_to_origin_velocity(self, epoch: float) -> np.ndarray:
        """
        Sun->origin velocity (km/s, raw numpy), memoized per epoch. Only the
        ``nadirPointingSun`` attitude needs it.

        Parameters
        ----------
        epoch : float
            TDB epoch (seconds past J2000).

        Returns
        -------
        numpy.ndarray
            ``(3,)`` Sun->origin velocity in km/s.
        """
        if self.__dict__.get("_s2o_vel_epoch") != epoch:
            if self._origin_is_sun:
                self._s2o_vel_cache = np.zeros(3)
            else:
                self._s2o_vel_cache = DynamicalUtils.sun_to_origin_velocity(
                    self._origin_spice_name, self._frame, epoch
                )
            self._s2o_vel_epoch = epoch
        return self._s2o_vel_cache

    def _attitude_dcm_ck(self, body, epoch: float) -> np.ndarray:
        """
        Body->inertial DCM for CK attitude at ``epoch``, memoized per epoch.

        ``compute_acceleration`` and ``_compute_partial_by_position`` both query
        the CK attitude kernel at the same epoch within one RHS step; the lookup
        depends only on ``epoch``, so the kernel query and the ``q2m``
        conversion of :meth:`~scarabaeus.DynamicalUtils.ck_dcm` run once.

        Parameters
        ----------
        body : Spacecraft
            Spacecraft whose CK-driven attitude is queried.
        epoch : float
            TDB epoch (seconds past J2000).

        Returns
        -------
        numpy.ndarray
            ``(3, 3)`` direction-cosine matrix transforming spacecraft-fixed
            vectors into ``base_frame``, the frame of the Sun vector and of
            the CK and named-frame plate normals.
        """
        if self.__dict__.get("_att_dcm_epoch") != epoch:
            self._att_dcm_cache = DynamicalUtils.ck_dcm(body, epoch, self._frame)
            self._att_dcm_epoch = epoch
        return self._att_dcm_cache

    def _all_normals(self, epoch: float) -> np.ndarray:
        """
        Plate outward normals at ``epoch``: the body-frame plates in the body
        frame, the CK and named-frame plates in ``base_frame``.

        Like the attitude DCM, the CK and named-frame normals are requested by
        both :meth:`compute_acceleration` and
        :meth:`_compute_partial_by_position` at the same epoch within one RHS
        step, so the (epoch-only) SPICE lookup is cached and reused.

        Parameters
        ----------
        epoch : float
            TDB epoch (seconds past J2000).

        Returns
        -------
        numpy.ndarray
            ``(N, 3)`` array of the ``N`` plate outward-normal vectors at ``epoch``.
        """
        if self.__dict__.get("_normals_epoch") != epoch:
            # dtype is pinned to float: a config whose normals are written as
            # integers (e.g. [1, 0, 0]) otherwise yields an int64 array, which
            # the Rust core rejects since it takes f64. Models with a CK plate
            # hide this, because the rotated CK normals make the array float.
            self._normals_cache = np.asarray(
                self.plate_model.get_all_normals(
                    epoch, self.spacecraft.spice_id, inertial_frame=self._frame_name
                ),
                dtype=float,
            )
            self._normals_epoch = epoch
        return self._normals_cache

    def _mass_value(self, epoch: float) -> float:
        """
        Spacecraft mass at ``epoch`` (raw kg float), memoized per epoch.

        :meth:`compute_acceleration` and :meth:`_compute_partial_by_position` both
        need the spacecraft mass at the same epoch within one RHS step (the partial
        reuses the same value for every plate), so the polynomial mass evaluation and
        its ``ArrayWUnits`` wrap are performed once and reused.

        Parameters
        ----------
        epoch : float
            Epoch at which to evaluate the spacecraft mass, in ephemeris time
            (seconds past J2000).

        Returns
        -------
        float
            Spacecraft mass at ``epoch`` in kilograms.
        """
        if self.__dict__.get("_mass_epoch") != epoch:
            self._mass_cache = self._spacecraft.mass(epoch).values
            self._mass_epoch = epoch
        return self._mass_cache

    def _attitude(
        self, sun2body: np.ndarray, current_state: dict, epoch: float, body: Spacecraft,
        derivatives: bool = False,
    ) -> tuple:
        """
        Spacecraft-fixed to inertial rotation and, for the nadir attitude,
        its derivatives, from :class:`~scarabaeus.DynamicalUtils`.

        The derivatives are returned per axis, ``D[j]`` being the Jacobian of
        axis :math:`j` (see
        :meth:`~scarabaeus.DynamicalUtils.nadirPointingSun`), which is the
        layout the rust Jacobians take.

        Parameters
        ----------
        sun2body : numpy.ndarray
            ``(3,)`` Sun->spacecraft position (inertial), km.
        current_state : dict
            The propagated state.
        epoch : float
            TDB epoch (seconds past J2000).
        body : Spacecraft
            The spacecraft for which SRP acceleration is calculated.
        derivatives : bool, optional
            Also return the nadir derivatives. Defaults to ``False``.

        Returns
        -------
        tuple
            ``(C, D_r, D_v)``: ``C`` is the ``(3, 3)`` spacecraft-fixed to
            inertial DCM; ``D_r`` and ``D_v`` are ``(3, 3, 3)`` arrays, or
            ``None`` when the attitude does not depend on the state or
            ``derivatives`` is false.

        Raises
        ------
        DynamicalUtils.InvalidAttitudeModeError
            If ``body.attitudeMode`` is not supported.
        DynamicalUtils.InvalidFixedAttitudeError
            If an ``inertialFixedAttitude`` body carries no 3x3
            ``fixedInertialAttitude``.
        DynamicalUtils.MissingVelocityError
            If a ``nadirPointingSun`` body has no velocity in the state.
        """
        attitude_mode = DynamicalUtils.attitude_mode(body)

        if attitude_mode == "CK":
            return self._attitude_dcm_ck(body, epoch), None, None

        if attitude_mode == "inertialFixedAttitude":
            return DynamicalUtils.fixed_dcm(body), None, None

        # Nadir: -X at the Sun, Y along the heliocentric velocity.
        velocity = DynamicalUtils.velocity_from_state(current_state, body)
        sun2body_vel = self._sun_to_origin_velocity(epoch) + velocity
        return DynamicalUtils.nadirPointingSun(sun2body, sun2body_vel, derivatives)

    def _rust_terms(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Spacecraft,
        derivatives: bool = False,
    ) -> list:
        """
        Resolve the per-evaluation inputs of the rust kernels, one term per
        plate group (:meth:`_build_rust_cores`), in their argument order:
        Sun->body position (inertial, km), DCM into ``base_frame``, eta_srp,
        spacecraft mass (kg), and the group's plate normals; plus the DCM
        derivatives the Jacobians take when the attitude follows the state.

        The body-frame group takes the attitude DCM and its body-frame
        normals. The CK and named-frame group takes the identity and its
        normals in ``base_frame``, with no DCM derivatives: those plates are
        oriented by SPICE, not by the attitude.

        Shared by :meth:`compute_acceleration`, the partials and the
        dual-number partials check, so all of them evaluate the same geometry.

        Parameters
        ----------
        position : np.ndarray
            3x1 vector describing the spacecraft's position. Assumed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch in seconds.

        body : Spacecraft
            The spacecraft for which SRP acceleration is calculated.

        derivatives : bool, optional
            Also return the nadir DCM derivatives. Defaults to ``False``.

        Returns
        -------
        list
            ``(core, inputs, D_r, D_v)`` per plate group: ``inputs`` is
            ``(sun2body, DCM, eta_srp, mass, normals)``; ``D_r`` and ``D_v``
            are as in :meth:`_attitude` (``None`` for the CK and named-frame
            group).

        Raises
        ------
        RustCoreUnavailableError
            If the rust extension is not available.
        AutogradTracingError
            If autograd is tracing ``position`` or eta_srp.
        DynamicalUtils.InvalidAttitudeModeError
            If ``body.attitudeMode`` is not supported.
        DynamicalUtils.InvalidFixedAttitudeError
            If an ``inertialFixedAttitude`` body carries no 3x3
            ``fixedInertialAttitude``.
        DynamicalUtils.MissingVelocityError
            If a ``nadirPointingSun`` body has no velocity in the state.
        """
        # Sun-to-Body vector (Sun-to-Origin memoized per epoch)
        sun2body = self._sun_to_origin(epoch) + position

        # Get the SRP scale factor
        eta_srp = self._get_eta_srp(current_state, body)
        self._require_rust_core(position, eta_srp)

        terms = []
        for plates, body_frame, core in self._rust_cores:
            if body_frame:
                # Spacecraft-fixed -> inertial DCM for the attitude mode.
                # dtype pinned to float: the rust core takes f64 (an integer
                # fixedInertialAttitude would otherwise be rejected).
                DCM, D_r, D_v = self._attitude(
                    sun2body, current_state, epoch, body, derivatives
                )
                DCM = np.asarray(DCM, dtype=float)
            else:
                # Normals already in base_frame: the kernel is rotation
                # covariant, so the identity gives their acceleration.
                DCM, D_r, D_v = np.eye(3), None, None

            # Plate normals and spacecraft mass (kg), both memoized per epoch.
            # Pass the raw ET float to avoid creating an EpochArray on every
            # evaluation.
            normals = self._all_normals(epoch)[plates]
            m_t = self._mass_value(epoch)
            terms.append((core, (sun2body, DCM, eta_srp, m_t, normals), D_r, D_v))
        return terms

    def compute_acceleration(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Spacecraft
    ) -> np.ndarray:
        """
        Computes the acceleration due to solar radiation pressure (SRP) based on the
        n-plate model.

        Calculates the contribution of each spacecraft plate to the total
        acceleration, considering reflection, absorption, and the relative orientation
        of the spacecraft with respect to the Sun. The plate sums and the
        SC-fixed -> inertial mapping run in rust (``NPlateSRPCore``), one core
        per plate group (body frame; CK and named frame), summed.

        Parameters
        ----------
        position : np.ndarray
            3x1 vector describing the spacecraft's position. Assumed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch in seconds.

        body : Spacecraft
            The spacecraft for which SRP acceleration is calculated.

        Returns
        -------
        total_srp_acc : np.ndarrays
            3x1 vector describing the total SRP acceleration. Assumed in units
            of :math:`\\frac{km}{s^2}`.

        Raises
        ------
        RustCoreUnavailableError
            If the rust extension is not available.
        AutogradTracingError
            If autograd is tracing ``position`` or eta_srp.
        """
        acceleration = np.zeros(3)
        for core, inputs, _, _ in self._rust_terms(position, current_state, epoch, body):
            acceleration = acceleration + np.asarray(core.compute_acceleration(*inputs))
        return acceleration

    def _compute_partial_by_position(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Spacecraft
    ) -> np.ndarray:
        """
        Computes the partial derivatives of the SRP acceleration with respect to position.

        Determines how the SRP acceleration varies as a function of
        the spacecraft's position relative to the Sun. Analytical, in rust
        (``NPlateSRPCore.compute_jacobian``): with s = DCM^T (-sun2body) the
        SC-fixed body->Sun vector and T the SC-fixed acceleration,
        da/dr = -DCM @ dT/ds @ DCM^T for an attitude that does not follow the
        position (CK, inertially fixed). The ``nadirPointingSun`` attitude
        turns with the position, so its DCM derivative adds
        sum_j T_j dDCM[:, j]/dr + DCM @ dT/ds @ (ds/dr through the attitude).
        Body-frame normals are held fixed in the body, CK and named-frame
        normals in ``base_frame``; the mass is held fixed.

        Parameters
        ----------
        position : np.ndarray
            3x1 vector describing the spacecraft's position. Assumed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch in seconds.

        body : Spacecraft
            The spacecraft for which SRP acceleration is calculated.

        Returns
        -------
        partial_a_by_pos : np.ndarrays
            The 3x3 Jacobian matrix of SRP acceleration with respect to position. Assumed in units
            of :math:`\\frac{1}{s^2}`.

        Raises
        ------
        RustCoreUnavailableError
            If the rust extension is not available.
        AutogradTracingError
            If autograd is tracing ``position`` or eta_srp.
        """
        jacobian = np.zeros((3, 3))
        for core, inputs, D_r, _ in self._rust_terms(
            position, current_state, epoch, body, derivatives=True
        ):
            jacobian = jacobian + np.asarray(
                core.compute_jacobian(*inputs, D_r)
            ).reshape((3, 3))
        return jacobian

    def _compute_partial_by_velocity(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Spacecraft
    ) -> np.ndarray:
        """
        Computes the partial derivatives of the SRP acceleration with respect to velocity.

        Non-zero only for the body-frame plates under the
        ``nadirPointingSun`` attitude, whose Y axis follows the heliocentric
        velocity: analytical, in rust
        (``NPlateSRPCore.compute_jacobian_by_velocity``), from the DCM
        derivative with respect to the velocity. Body-frame normals are held
        fixed in the body, CK and named-frame normals in ``base_frame``; the
        mass is held fixed.

        Parameters
        ----------
        position : np.ndarray
            3x1 vector describing the spacecraft's position. Assumed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch in seconds.

        body : Spacecraft
            The spacecraft for which SRP acceleration is calculated.

        Returns
        -------
        partial_a_by_vel : np.ndarray
            The 3x3 Jacobian matrix of SRP acceleration with respect to velocity. Assumed in units
            of :math:`\\frac{1}{s}`.

        Raises
        ------
        RustCoreUnavailableError
            If the rust extension is not available.
        AutogradTracingError
            If autograd is tracing ``position`` or eta_srp.
        """
        jacobian = np.zeros((3, 3))
        if not self.depends_on_velocity:
            return jacobian
        for core, inputs, _, D_v in self._rust_terms(
            position, current_state, epoch, body, derivatives=True
        ):
            if D_v is None:
                # SPICE-oriented plates do not turn with the velocity
                continue
            jacobian = jacobian + np.asarray(
                core.compute_jacobian_by_velocity(*inputs, D_v)
            ).reshape((3, 3))
        return jacobian

    def _compute_partial_by_eta(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Spacecraft
    ) -> np.ndarray:
        """
        Computes the partial derivative of SRP acceleration with respect to the SRP scale factor (`eta_srp`).

        This method determines how changes in the SRP scale factor affect the acceleration.

        Parameters
        ----------
        position : np.ndarray
            3x1 vector describing the spacecraft's position. Assumed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch in seconds.

        body : Spacecraft
            The spacecraft for which SRP acceleration is calculated.

        Returns
        -------
        partial_by_eta : np.ndarray
            3x1 vector describing the partial derivative of SRP acceleration with respect to ``eta_srp``. Assumed in units
            of :math:`\\frac{km}{s^2}`.
        """
        # Get the SRP scale factor
        eta_srp = self._get_eta_srp(current_state, body)

        # Compute the partial derivative
        partial_a_by_eta = (
            self.compute_acceleration(position, current_state, epoch, body) / eta_srp
        )

        # Return the transposed result
        return partial_a_by_eta.transpose()

    def _angle_between(
        self,
        vec_1: np.ndarray,
        vec_2: np.ndarray,
    ) -> float:
        """Compute the angle between two vectors

        Parameters
        ----------
        vec_1 : np.ndarray
            First vector.

        vec_2 : np.ndarray
            Second vector.

        Returns
        -------
        angle : float
            Angle between the two vectors in radians
        """
        # Compute the angle between two vectors
        cross = np.linalg.norm(np.cross(vec_1, vec_2))
        dot = np.dot(vec_1, vec_2)

        return np.arctan2(cross, dot)
