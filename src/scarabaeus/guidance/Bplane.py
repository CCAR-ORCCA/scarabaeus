# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from __future__ import annotations

import os
import pathlib
import warnings
from typing import Optional, Union

import spiceypy as spice
from scipy.optimize import minimize

from scarabaeus import (
    ArrayWUnits,
    constants,
    EpochArray,
    SpiceManager,
    Units,
    Trajectory,
)  # assumed interface

import scarabaeus.utils.NumpyWrapper as np

# ------------------#
#  Generate Units  #
# ------------------#
m, km, sec = Units.get_units(["m", "km", "sec"])

# Type alias for SPICE body identifiers
BodyID = Union[int, str]


# --------------------#
#  Class Definition  #
# --------------------#
class Bplane:
    """ B-plane and B-vector definition and targeting.

    The B-plane is a plane passing through the centre of a target body,
    perpendicular to the incoming hyperbolic asymptote of the approach
    trajectory.  The B-vector is the vector from the target-body centre
    to the point where the asymptote pierces that plane, and its components
    ``B·T`` and ``B·R`` are the primary targeting parameters used in
    planetary-flyby and encounter design.

    **B-plane geometry**

    The B-plane is defined at a reference epoch by three right-handed
    orthonormal vectors:

    .. math::

        \\hat{S} &= \\frac{\\mathbf{v}_\\infty}{\\|\\mathbf{v}_\\infty\\|}
        \\quad\\text{(incoming asymptote direction)}\\\\
        \\hat{T} &= \\frac{\\hat{k} \\times \\hat{S}}{\\|\\hat{k} \\times \\hat{S}\\|}
        \\quad\\text{(in B-plane, perpendicular to S)}\\\\
        \\hat{R} &= \\hat{S} \\times \\hat{T}

    where :math:`\\hat{k}` is the ecliptic north pole and
    :math:`\\mathbf{v}_\\infty` is the hyperbolic excess velocity vector.

    The **B-vector** is the perpendicular from the target-body centre to the
    incoming asymptote:

    .. math::

        \\mathbf{B} = B_T\\,\\hat{T} + B_R\\,\\hat{R},
        \\qquad
        |\\mathbf{B}| = \\sqrt{B_T^2 + B_R^2}

    For a hyperbolic flyby with gravitational parameter :math:`\\mu`, the
    periapsis distance is

    .. math::

        r_p = \\frac{\\mu}{v_\\infty^2}
              \\left(\\sqrt{1 + \\frac{v_\\infty^2\\,|\\mathbf{B}|^2}{\\mu^2}} - 1\\right)

    **Targeting**

    :meth:`target` minimizes :math:`\\|\\mathbf{b}_{target} - \\mathbf{b}_{ref}\\|`
    by solving for the :math:`\\Delta v` correction at a specified maneuver epoch,
    where :math:`\\mathbf{b}_{target} = [B\\cdot T_{target},\\;B\\cdot R_{target},\\;\\mathrm{LTOF}_{target}]`.

    Parameters
    ----------
    epoch : EpochArray | datetime.datetime | float
        Reference epoch for the B-plane frame.  Floats are interpreted as
        J2000 ephemeris seconds (ET); ``datetime`` objects are converted
        automatically.
    bplane_name : str
        Name of the B-plane SPICE frame (e.g. ``'BPLANE_EUROPA'``).
    bplane_spice_id : int or str
        SPICE numeric frame ID for the B-plane frame.
    sc_name : str
        Human-readable spacecraft name (used for labelling only).
    sc_spice_id : int or str
        SPICE body ID (name string or integer) for the spacecraft.
    target_name : str
        Human-readable target-body name; must match a key in
        ``scarabaeus.Constants`` that exposes a ``mu`` entry.
    target_spice_id : int or str
        SPICE body ID (name string or integer) for the target body.
    fk_file : str or pathlib.Path
        Path to the frames-kernel (.tf) file.  If *new_bplane* is ``True``
        the file is created here; otherwise it is loaded from this path.
    v_hat : bool, optional
        When ``True`` (default) the secondary axis of the B-plane frame is
        aligned with the +Y direction; ``False`` flips it to −Y.
    new_bplane : bool, optional
        ``True`` (default) creates and furnishes a new FK file.
        ``False`` loads an existing kernel from *fk_file*.
    scenario_kernel_dir : str or pathlib.Path, optional
        Directory into which the FK file is written.  Defaults to
        ``<cwd>/data/Kernels``.

    Attributes
    ----------
    bt_ref, br_ref : ArrayWUnits
        Reference B-plane targeting parameters ``B·T`` and ``B·R`` [km].
    ltof_ref : ArrayWUnits
        Linearised time-of-flight from the reference epoch to close
        approach [s].
    time_close_approach_ref : ArrayWUnits
        Estimated close-approach epoch expressed as ET seconds.
    bpar_ref : ArrayWUnits
        Concatenated reference vector ``[B·T, B·R, ltof]``.

    See Also
    --------
    scarabaeus.Trajectory : Trajectory container used as input to compute the B-plane.
    scarabaeus.EpochArray : Epoch representation for the B-plane reference epoch.
    """

    # -----------------------------------
    #           Constructor
    # -----------------------------------
    def __init__(
        self,
        epoch: Union[EpochArray, float],
        bplane_name: str,
        bplane_spice_id: BodyID,
        sc_name: str,
        sc_spice_id: BodyID,
        target_name: str,
        target_spice_id: BodyID,
        fk_file: Union[str, pathlib.Path],
        v_hat: bool = True,
        new_bplane: bool = True,
        scenario_kernel_dir: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        # ------------------------------------------------------------------
        # Normalise epoch to EpochArray regardless of input type
        # ------------------------------------------------------------------
        self._epoch = self._coerce_epoch(epoch)

        # ------------------------------------------------------------------
        # Normalise body IDs (accept name strings or integer IDs)
        # ------------------------------------------------------------------
        self._bplane_name = bplane_name
        self._bplane_spice_id = int(bplane_spice_id)
        self._sc_name = sc_name
        self._sc_spice_id = self._resolve_body_id(sc_spice_id)
        self._target_name = target_name
        self._target_spice_id = self._resolve_body_id(target_spice_id)
        self._fk_file = pathlib.Path(fk_file)
        self._v_hat = v_hat
        self._new_bplane = new_bplane

        # Kernel output directory
        if scenario_kernel_dir is not None:
            self._kernel_dir = pathlib.Path(scenario_kernel_dir)
        else:
            self._kernel_dir = pathlib.Path(os.getcwd()) / "data" / "Kernels"
        self._kernel_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Create / load the B-plane frame kernel
        # ------------------------------------------------------------------
        self._create_kernel()

        # ------------------------------------------------------------------
        # Rotation matrices J2000 <-> B-plane
        # ------------------------------------------------------------------
        et = self._epoch.times.values
        self._rot_to_j2000 = ArrayWUnits(
            SpiceManager.get_xfrm(bplane_name, "J2000", et), None
        )
        self._rot_to_bplane = ArrayWUnits(
            SpiceManager.get_xfrm("J2000", bplane_name, et), None
        )

        # ------------------------------------------------------------------
        # Reference B-plane parameters at the given epoch
        # ------------------------------------------------------------------
        sc_rel_state_j2000 = SpiceManager.get_state(
            trgt_bdy=str(self._sc_spice_id),
            epoch_time=et,
            reference_frame="J2000",
            obsvr_bdy=str(self._target_spice_id),
        )
        bpar = self.compute_hyperbolic_parameters(sc_rel_state_j2000)
        self.bpar_ref = bpar
        self._bt_ref = bpar[0]  # [km]
        self._br_ref = bpar[1]  # [km]
        self._ltof_ref = bpar[2]  # [s]

        # Estimated close-approach epoch (ET)
        self._time_close_approach_ref = ArrayWUnits(
            np.array(et + self._ltof_ref.values), sec
        )

    # =========================================================================
    # region ───── Properties ─────────────────────────────────────────────────
    # =========================================================================

    @property
    def bplane_name(self) -> str:
        """SPICE frame name of the B-plane."""
        return self._bplane_name

    @property
    def bplane_spice_id(self) -> int:
        """SPICE numeric frame ID of the B-plane."""
        return self._bplane_spice_id

    @property
    def sc_name(self) -> str:
        """Human-readable spacecraft name."""
        return self._sc_name

    @property
    def sc_spice_id(self) -> BodyID:
        """Resolved SPICE body identifier for the spacecraft."""
        return self._sc_spice_id

    @property
    def target_name(self) -> str:
        """Human-readable target-body name."""
        return self._target_name

    @property
    def target_spice_id(self) -> BodyID:
        """Resolved SPICE body identifier for the target."""
        return self._target_spice_id

    @property
    def fk_file(self) -> pathlib.Path:
        """Absolute path to the B-plane frames kernel file."""
        return self._fk_file

    @property
    def v_hat(self) -> bool:
        """Secondary-axis orientation flag (``True`` → +Y, ``False`` → −Y)."""
        return self._v_hat

    @property
    def epoch(self) -> EpochArray:
        """Reference epoch as an ``EpochArray``."""
        return self._epoch

    @property
    def new_bplane(self) -> bool:
        """Whether a new FK file was created by this instance."""
        return self._new_bplane

    @property
    def rot_to_j2000(self) -> ArrayWUnits:
        """Rotation matrix (DCM) from the B-plane frame to J2000."""
        return self._rot_to_j2000

    @property
    def rot_to_bplane(self) -> ArrayWUnits:
        """Rotation matrix (DCM) from J2000 to the B-plane frame."""
        return self._rot_to_bplane

    # Keep legacy aliases so existing call-sites are not broken
    @property
    def rotmat_to_j2000(self) -> ArrayWUnits:
        """Alias for :attr:`rot_to_j2000`."""
        return self._rot_to_j2000

    @property
    def rotmat_to_bplane(self) -> ArrayWUnits:
        """Alias for :attr:`rot_to_bplane`."""
        return self._rot_to_bplane

    @property
    def bt_ref(self) -> ArrayWUnits:
        """Reference ``B·T`` component of the B-vector [km]."""
        return self._bt_ref

    @property
    def br_ref(self) -> ArrayWUnits:
        """Reference ``B·R`` component of the B-vector [km]."""
        return self._br_ref

    @property
    def ltof_ref(self) -> ArrayWUnits:
        """Reference linearised time-of-flight [s]."""
        return self._ltof_ref

    @property
    def time_close_approach_ref(self) -> ArrayWUnits:
        """Estimated close-approach epoch [ET seconds]."""
        return self._time_close_approach_ref

    # endregion ── Properties ─────────────────────────────────────────────────

    # =========================================================================
    # region ───── Private helpers ────────────────────────────────────────────
    # =========================================================================

    @staticmethod
    def _coerce_epoch(epoch) -> EpochArray:
        """
        Accept an ``EpochArray``, a ``datetime``, or a float (ET seconds) and
        always return an ``EpochArray``.
        """
        if isinstance(epoch, EpochArray):
            return epoch
        # datetime → ET float via SPICE
        import datetime as _dt

        if isinstance(epoch, _dt.datetime):
            et = spice.str2et(epoch.strftime("%Y-%b-%d %H:%M:%S.%f UTC"))
            return EpochArray(et)
        # bare float treated as ET seconds
        if isinstance(epoch, (int, float)):
            return EpochArray(float(epoch))
        raise TypeError(
            f"epoch must be EpochArray, datetime, or float (ET); got {type(epoch)}"
        )

    @staticmethod
    def _resolve_body_id(body: BodyID) -> BodyID:
        """
        Accept either a SPICE body-name string or an integer ID and return
        whichever form SPICE prefers (string name when available, else int).
        Raises ``ValueError`` if the body cannot be resolved.
        """
        if isinstance(body, str):
            try:
                spice.bodn2c(body)  # validates the name exists
            except spice.utils.exceptions.SpiceyError:
                raise ValueError(f"SPICE cannot resolve body name '{body}'.")
            return body
        if isinstance(body, int):
            try:
                name = spice.bodc2n(body)
                return name  # prefer the canonical name
            except spice.utils.exceptions.SpiceyError:
                return body  # fall back to the raw integer
        raise TypeError(f"Body ID must be str or int; got {type(body)}")

    @staticmethod
    def _check_frame_name(name: str) -> None:
        """Raise ``RuntimeError`` if the frame *name* is already in the pool."""
        if spice.namfrm(name) != 0:
            raise RuntimeError(f"Frame with name '{name}' already exists in pool.")

    @staticmethod
    def _check_frame_id(frame_id: int) -> None:
        """
        Raise ``RuntimeError`` if the numeric frame *frame_id* is already
        known to SPICE.

        Replaces the original ``spice.cidfrm`` call which is designed for
        *body*-fixed frames; ``spice.frinfo`` is the correct routine for
        querying an arbitrary frame by its integer ID.
        """
        try:
            _cent, _fclass, _clssid = spice.frinfo(frame_id)
            # frinfo succeeded → frame already registered
            raise RuntimeError(
                f"Frame ID {frame_id} is already registered in the SPICE pool."
            )
        except spice.utils.exceptions.NotFoundError:
            pass  # expected: frame does not exist yet

    # endregion ── Private helpers ────────────────────────────────────────────

    # =========================================================================
    # region ───── Public methods ─────────────────────────────────────────────
    # =========================================================================

    # -------------------------------------------------------------------------
    # Kernel creation
    # -------------------------------------------------------------------------

    def _create_kernel(self) -> None:
        """
        Create (or load) the B-plane SPICE frames kernel.

        If *new_bplane* is ``False`` the kernel at :attr:`fk_file` is loaded
        into the SPICE pool (if not already there).  If *new_bplane* is
        ``True`` a new ``.tf`` file is written to :attr:`fk_file` and then
        furnished.

        Raises
        ------
        RuntimeError
            If the target file already exists on disk, or if the frame name /
            frame ID is already registered in the SPICE pool.
        """
        fk_path = self._kernel_dir / self._fk_file

        if not self._new_bplane:
            # Load existing kernel if not already in pool
            if not SpiceManager.check_kernel_status_in_pool(str(fk_path)):
                SpiceManager.load_kernel(str(fk_path))
            return

        # ── Sanity checks before writing ──────────────────────────────────
        if fk_path.exists():
            raise RuntimeError(f"Frames kernel file already exists: {fk_path}")
        self._check_frame_name(self._bplane_name)
        self._check_frame_id(self._bplane_spice_id)

        # ── Write FK file ─────────────────────────────────────────────────
        epoch_str = spice.timout(
            self._epoch.times.values,
            "@YYYY-MON-DD/HR:MN:SC.######::TDB",
        )
        sec_axis = "Y" if self._v_hat else "-Y"
        bid = self._bplane_spice_id

        lines = [
            "\\begindata\n",
            f"\nFRAME_{self._bplane_name}   = {bid}\n",
            f"FRAME_{bid}_NAME            = '{self._bplane_name}'\n",
            f"FRAME_{bid}_CLASS           = 5\n",
            f"FRAME_{bid}_CLASS_ID        = {bid}\n",
            f"FRAME_{bid}_CENTER          = '{self._target_spice_id}'\n",
            f"FRAME_{bid}_RELATIVE        = 'J2000'\n",
            f"FRAME_{bid}_DEF_STYLE       = 'PARAMETERIZED'\n",
            f"FRAME_{bid}_FAMILY          = 'TWO-VECTOR'\n",
            "\n",
            f"FRAME_{bid}_PRI_AXIS        = 'Z'\n",
            f"FRAME_{bid}_PRI_VECTOR_DEF  = 'OBSERVER_TARGET_VELOCITY'\n",
            f"FRAME_{bid}_PRI_OBSERVER    = '{self._target_spice_id}'\n",
            f"FRAME_{bid}_PRI_TARGET      = '{self._sc_spice_id}'\n",
            f"FRAME_{bid}_PRI_ABCORR      = 'NONE'\n",
            f"FRAME_{bid}_PRI_FRAME       = 'J2000'\n",
            "\n",
            f"FRAME_{bid}_SEC_AXIS        = '{sec_axis}'\n",
            f"FRAME_{bid}_SEC_VECTOR_DEF  = 'CONSTANT'\n",
            f"FRAME_{bid}_SEC_SPEC        = 'RECTANGULAR'\n",
            f"FRAME_{bid}_SEC_VECTOR      = ( 0, 0, 1 )\n",
            f"FRAME_{bid}_SEC_FRAME       = 'J2000'\n",
            f"FRAME_{bid}_FREEZE_EPOCH    = {epoch_str}\n",
            "\n\\begintext\n\nEnd of FK File.\n",
        ]

        fk_path.write_text("".join(lines))

        # Furnish the newly written kernel
        spice.furnsh(str(fk_path))

    # -------------------------------------------------------------------------
    # B-plane parameter computation
    # -------------------------------------------------------------------------

    def compute_parameters(self, sc_rel_state_j2000: ArrayWUnits) -> ArrayWUnits:
        """
        Compute the linear B-plane parameters from a spacecraft state.

        Projects the relative state onto the B-plane and returns the
        intersection coordinates together with the linearised time-of-flight.

        Parameters
        ----------
        sc_rel_state_j2000 : ArrayWUnits, shape (6,)
            Relative Cartesian state ``[x, y, z, vx, vy, vz]`` of the
            spacecraft with respect to the target body, expressed in J2000.

        Returns
        -------
        bpar : ArrayWUnits, shape (3,)
            ``[B·T, B·R, ltof]`` where B·T and B·R are in km and ltof is in
            seconds.
        """
        R = self._rot_to_bplane.values

        rel_pos_bp = R @ sc_rel_state_j2000[0:3].values
        rel_vel_bp = R @ sc_rel_state_j2000[3:6].values

        # Linear propagation to the B-plane (z = 0)
        ltof_val = rel_pos_bp[2] / rel_vel_bp[2]
        sc_bp_pos = rel_pos_bp - ltof_val * rel_vel_bp

        bt = ArrayWUnits(np.array([sc_bp_pos[0]]), km)
        br = ArrayWUnits(np.array([sc_bp_pos[1]]), km)
        ltof = ArrayWUnits(np.array([ltof_val]), sec)

        return bt.append(br).append(ltof)

    def compute_hyperbolic_parameters(
        self, sc_rel_state_j2000: ArrayWUnits
    ) -> ArrayWUnits:
        """
        Compute classical hyperbolic B-plane parameters from a Cartesian state.

        This routine evaluates the full conic geometry of the approach
        trajectory and derives the B-vector components ``B·T`` and ``B·R``
        from the asymptote direction and the orbit's semi-latus rectum.  For
        elliptic trajectories (eccentricity < 1) the B-plane is undefined and
        the corresponding output entries are set to ``NaN``.

        Parameters
        ----------
        sc_rel_state_j2000 : ArrayWUnits, shape (6,) or (N, 6)
            Relative state(s) in J2000 with units convertible to km / km·s⁻¹.

        Returns
        -------
        ArrayWUnits, shape (3,) or (N, 3)
            ``[B·T [km], B·R [km], ltof [s]]``.

        Notes
        -----
        The B-plane convention follows Kizner (1961):

        * **S** – unit vector along the incoming asymptote.
        * **T** – perpendicular to S lying in the target equatorial plane
          (constructed as ``T = [-Sy, Sx, 0] / |[-Sy, Sx, 0]|``).
        * **R = S × T**.
        * **B** = semi-minor axis × (S × ĥ).

        Warnings
        --------
        The T-vector construction becomes degenerate when S is nearly
        parallel to the z-axis (i.e. the approach asymptote nearly coincides
        with the north pole direction).
        - The computation of ``beta = arccos(1/ecc)`` is only valid for
        hyperbolic or parabolic cases.
        - The construction of ``Tvec`` as [-Sy, Sx, 0] becomes degenerate
        when the projection of ``Svec`` onto the xy-plane is near zero.
        - If ``ecc == 1`` exactly, the semimajor/seminor-axis expressions
        may become numerically delicate.
        - ``gm`` is retrieved from :class:`~scarabaeus.units.constants` using 
          ``self.target_name``.
        """
        if self.target_name.upper().endswith('BARYCENTER'):
            cnstnts_name = self.target_name.upper().removesuffix('_BARYCENTER').removesuffix(' BARYCENTER').removesuffix('BARYCENTER')
            gm = getattr(constants, cnstnts_name).GM.values
        else:
            gm = getattr(constants, self.target_name).GM.values

        states = sc_rel_state_j2000.convert_to(
            [km, km, km, km / sec, km / sec, km / sec]
        ).values

        scalar_input = states.ndim == 1
        if scalar_input:
            states = states[np.newaxis, :]  # (1, 6)

        r_vec = states[:, :3]
        v_vec = states[:, 3:]
        rmag = np.linalg.norm(r_vec, axis=1)
        vmag = np.linalg.norm(v_vec, axis=1)

        h_vec = np.cross(r_vec, v_vec, axis=1)
        hmag = np.linalg.norm(h_vec, axis=1)
        hhat = h_vec / hmag[:, None]

        rdotv = np.sum(r_vec * v_vec, axis=1)
        e_vec = ((vmag**2 - gm / rmag)[:, None] * r_vec - rdotv[:, None] * v_vec) / gm
        ecc = np.linalg.norm(e_vec, axis=1)
        c3 = vmag**2 - 2.0 * gm / rmag  # noqa: F841  (available for callers)

        n = len(ecc)
        bdott = np.full(n, np.nan)
        bdotr = np.full(n, np.nan)

        hyp = ecc >= 1.0
        if np.any(hyp):
            ei = e_vec[hyp]
            ei_mag = ecc[hyp]
            hi = hhat[hyp]

            beta = np.arccos(np.clip(1.0 / ei_mag, -1.0, 1.0))

            # h × e_hat  (used to complete the S-vector triad)
            hcrosse = np.cross(hi, ei / ei_mag[:, None], axis=1)
            hcrosse_mag = np.linalg.norm(hcrosse, axis=1)

            S = (
                np.cos(beta)[:, None] * ei / ei_mag[:, None]
                + np.sin(beta)[:, None] * hcrosse / hcrosse_mag[:, None]
            )

            # T perpendicular to S, lying in the xy-plane
            T_raw = np.column_stack([-S[:, 1], S[:, 0], np.zeros(len(S))])
            Tmag = np.linalg.norm(T_raw, axis=1)
            if np.any(Tmag < 1e-12):
                warnings.warn(
                    "T-vector near-degenerate: S is nearly aligned with z-axis.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            T = T_raw / np.where(Tmag[:, None] > 1e-12, Tmag[:, None], 1.0)

            R = np.cross(S, T, axis=1)

            # B = b * (S × h)
            Bhat = np.cross(S, hi, axis=1)
            sma = 1.0 / (2.0 / rmag[hyp] - vmag[hyp] ** 2 / gm)
            smm = sma * np.sqrt(ei_mag**2 - 1.0)  # semi-minor axis
            B = Bhat * smm[:, None]

            bdotr[hyp] = np.sum(B * R, axis=1)
            bdott[hyp] = np.sum(B * T, axis=1)

        ltof = self.compute_parameters(sc_rel_state_j2000)[2]  # ArrayWUnits [s]

        if scalar_input:
            out = ArrayWUnits([bdott[0], bdotr[0]], [km, km])
        else:
            out = ArrayWUnits(np.column_stack([bdott, bdotr]), [km, km])

        return out.append(ltof)

    # -------------------------------------------------------------------------
    # Jacobian
    # -------------------------------------------------------------------------

    def compute_jacobian(self, sc_rel_state_j2000: ArrayWUnits) -> ArrayWUnits:
        """
        Finite-difference Jacobian of the B-plane parameters w.r.t. velocity.

        Computes the 3×3 matrix  ``d[B·T, B·R, ltof] / d[vx, vy, vz]``  via
        forward finite differences with perturbation size chosen as
        ``sqrt(eps_machine) * max(1, |v_i|)``.

        Parameters
        ----------
        sc_rel_state_j2000 : ArrayWUnits, shape (6,)
            Current relative state in J2000.

        Returns
        -------
        jacobian : ArrayWUnits, shape (3, 3)
            Jacobian matrix with units ``[km/(km/s), km/(km/s), s/(km/s)]``
            i.e. ``[s, s, s²/km]`` per column.
        """
        vel = sc_rel_state_j2000[3:6].values
        n_in = len(vel)
        n_out = len(self.compute_hyperbolic_parameters(sc_rel_state_j2000).values)
        jac = np.zeros((n_out, n_in))

        eps = np.sqrt(np.finfo(np.float64).eps) * np.maximum(1.0, np.abs(vel))
        sc_units = [km, km, km, km / sec, km / sec, km / sec]

        bpar0 = self.compute_hyperbolic_parameters(sc_rel_state_j2000)

        for i in range(n_in):
            perturbed = sc_rel_state_j2000[:].values.copy()
            perturbed[3 + i] += eps[i]
            bpar_plus = self.compute_hyperbolic_parameters(ArrayWUnits(perturbed, sc_units))
            jac[:, i] = (bpar_plus.values - bpar0.values) / eps[i]

        jac_units = np.array(
            [
                [sec, sec, sec**2 / km],
                [sec, sec, sec**2 / km],
                [sec, sec, sec**2 / km],
            ]
        )
        return ArrayWUnits(jac, jac_units)

    # -------------------------------------------------------------------------
    # Linear targeting (Newton / pseudo-Newton iteration)
    # -------------------------------------------------------------------------

    def linear_targeting(
        self,
        sc_rel_state_j2000: ArrayWUnits,
        err: float = 1e-8,
        max_iterations: int = 100,
    ) -> ArrayWUnits:
        """
        Compute a trajectory-correction manoeuvre (TCM) via linear targeting.

        Iterates a Newton-style correction loop using the local Jacobian until
        the residual in B-plane parameter space drops below *err*.

        Parameters
        ----------
        sc_rel_state_j2000 : ArrayWUnits, shape (6,)
            Perturbed relative state in J2000 at the manoeuvre epoch.
        err : float, optional
            Convergence threshold on the L2 norm of the B-plane residual.
        max_iterations : int, optional
            Maximum number of Newton iterations before raising an error.

        Returns
        -------
        DV_tot : ArrayWUnits, shape (3,)
            Velocity increment vector ``[ΔVx, ΔVy, ΔVz]`` in km/s.

        Raises
        ------
        RuntimeError
            If the iteration does not converge within *max_iterations* steps.
        """
        sc_units = [km, km, km, km / sec, km / sec, km / sec]
        bpar_ref_val = np.array(
            [
                self._bt_ref.values,
                self._br_ref.values,
                self._ltof_ref.values,
            ]
        )
        bpar_ref = ArrayWUnits(bpar_ref_val, [km, km, sec])

        state_targ = sc_rel_state_j2000[:].values.copy()
        count = 0

        while True:
            bpar_cur = self.compute_hyperbolic_parameters(ArrayWUnits(state_targ, sc_units))
            err_vec = bpar_ref - bpar_cur
            jac = self.compute_jacobian(ArrayWUnits(state_targ, sc_units))
            DV_step = np.linalg.solve(jac.values, err_vec.values)

            state_targ[3:6] += DV_step
            count += 1

            # Converged when both the residual and the correction step are small
            if np.linalg.norm(err_vec.values) <= err or np.linalg.norm(DV_step) <= err:
                break
            if count >= max_iterations:
                raise RuntimeError(
                    f"linear_targeting did not converge in {max_iterations} iterations. "
                    f"Residual = {np.linalg.norm(err_vec.values):.3e}"
                )

        print(
            f"linear_targeting converged in {count} iterations. "
            f"Residual = {np.linalg.norm((bpar_ref - self.compute_hyperbolic_parameters(ArrayWUnits(state_targ, sc_units))).values):.3e}"
        )
        return ArrayWUnits(state_targ[3:6] - sc_rel_state_j2000[3:6].values, km / sec)

    # -------------------------------------------------------------------------
    # Nonlinear targeting (Nelder-Mead)
    # -------------------------------------------------------------------------

    def nonlinear_targeting(
        self,
        sc_rel_state_j2000: ArrayWUnits,
        err: float = 1e-10,
    ) -> ArrayWUnits:
        """
        Compute a TCM via nonlinear optimisation (Nelder-Mead).

        Minimises the squared L2 norm of the B-plane residual
        ``‖[B·T, B·R, ltof]_ref − [B·T, B·R, ltof]‖²`` over the three
        velocity components at the manoeuvre epoch.

        Parameters
        ----------
        sc_rel_state_j2000 : ArrayWUnits, shape (6,)
            Perturbed relative state in J2000 at the manoeuvre epoch.
        err : float, optional
            Tolerance passed to ``scipy.optimize.minimize``.

        Returns
        -------
        DV_tot : ArrayWUnits, shape (3,)
            Velocity increment vector ``[ΔVx, ΔVy, ΔVz]`` in km/s.
        """
        sc_units = [km, km, km, km / sec, km / sec, km / sec]
        bpar_ref_val = np.array(
            [
                self._bt_ref.values,
                self._br_ref.values,
                self._ltof_ref.values,
            ]
        )
        bpar_ref = ArrayWUnits(bpar_ref_val, [km, km, sec])
        pos0 = sc_rel_state_j2000[0:3].values

        def objective(v_vec: np.ndarray) -> float:
            state = ArrayWUnits(np.concatenate([pos0, v_vec]), sc_units)
            err_vec = bpar_ref - self.compute_hyperbolic_parameters(state)
            return float(np.linalg.norm(err_vec.values) ** 2)

        result = minimize(
            objective,
            x0=sc_rel_state_j2000[3:6].values.copy(),
            method="Nelder-Mead",
            tol=err,
            options={"disp": True},
        )
        return ArrayWUnits(result.x - sc_rel_state_j2000[3:6].values, km / sec)

    # -------------------------------------------------------------------------
    # Piecewise / trajectory-based targeting
    # -------------------------------------------------------------------------

    def piecewise_targeting(
        self,
        trajectory: "Trajectory",
        tcm_epochs: list,
        method: str = "linear",
        err: float = 1e-8,
        max_iterations: int = 100,
        spk_output_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> list[ArrayWUnits]:
        """
        Compute TCMs at multiple epochs along a trajectory and (optionally)
        generate a corrected SPK ephemeris file.

        The algorithm proceeds segment-by-segment:

        1. Evaluate the current relative state at *tcm_epoch[i]*.
        2. Solve for the ΔV that corrects the B-plane parameters to the
           reference values using the chosen *method*.
        3. Apply the ΔV to the state, propagate to the next TCM epoch (or to
           close approach for the last segment), and write the resulting arc to
           the SPK file.

        Parameters
        ----------
        trajectory : Trajectory
            A ``scarabaeus.Trajectory`` object that exposes at minimum:

            * ``get_state(epoch)`` → ``ArrayWUnits`` relative state in J2000.
            * ``propagate(state, epoch_start, epoch_end)``
              → updated ``Trajectory`` segment.
            * ``write_spk(filepath)`` → generate a SPICE SPK kernel.

        tcm_epochs : list of EpochArray | datetime | float
            Ordered sequence of epochs at which TCMs are to be computed.
            Must be in chronological order.

        method : {'linear', 'nonlinear'}, optional
            Targeting algorithm.  ``'linear'`` uses the Newton iteration
            (:meth:`linear_targeting`); ``'nonlinear'`` uses Nelder-Mead
            (:meth:`nonlinear_targeting`).

        err : float, optional
            Convergence tolerance forwarded to the chosen targeting method.

        max_iterations : int, optional
            Maximum Newton iterations (linear method only).

        spk_output_path : str or pathlib.Path, optional
            If provided, a SPICE SPK file covering the full corrected
            trajectory is written to this path after all TCMs have been
            applied.

        Returns
        -------
        dv_list : list of ArrayWUnits
            ΔV vectors ``[ΔVx, ΔVy, ΔVz]`` [km/s] at each TCM epoch.

        Raises
        ------
        ValueError
            If *method* is not ``'linear'`` or ``'nonlinear'``.
        """
        if method not in ("linear", "nonlinear"):
            raise ValueError(f"method must be 'linear' or 'nonlinear'; got '{method}'")

        sc_units = [km, km, km, km / sec, km / sec, km / sec]
        target_fn = (
            self.linear_targeting if method == "linear" else self.nonlinear_targeting
        )
        target_kwargs = {"err": err}
        if method == "linear":
            target_kwargs["max_iterations"] = max_iterations

        dv_list: list[ArrayWUnits] = []
        segments: list = []  # propagated Trajectory segments for SPK writing

        for i, raw_epoch in enumerate(tcm_epochs):
            tcm_epoch = self._coerce_epoch(raw_epoch)

            # Current relative state at this TCM epoch
            sc_rel_state = trajectory.get_state(tcm_epoch)

            # Solve for ΔV
            dv = target_fn(sc_rel_state, **target_kwargs)
            dv_list.append(dv)

            # Apply ΔV
            corrected_vel = ArrayWUnits(sc_rel_state[3:6].values + dv.values, km / sec)
            corrected_state = ArrayWUnits(
                np.concatenate([sc_rel_state[0:3].values, corrected_vel.values]),
                sc_units,
            )

            # Propagate to the next TCM epoch or to close approach
            if i + 1 < len(tcm_epochs):
                next_epoch = self._coerce_epoch(tcm_epochs[i + 1])
            else:
                next_epoch = self._time_close_approach_ref

            segment = trajectory.propagate(corrected_state, tcm_epoch, next_epoch)
            segments.append(segment)

        # Write combined SPK if requested
        if spk_output_path is not None:
            spk_path = pathlib.Path(spk_output_path)
            # Trajectory is expected to accept a list of segments and merge them
            trajectory.write_spk(segments, str(spk_path))
            print(f"Corrected SPK written to: {spk_path}")

        return dv_list

    # -------------------------------------------------------------------------
    # Covariance mapping
    # -------------------------------------------------------------------------

    def target_covariance(
        self,
        sc_cov: np.ndarray,
        sc_rel_state_j2000: ArrayWUnits,
    ) -> ArrayWUnits:
        """
        Map a spacecraft state covariance to B-plane parameter covariance.

        Uses the analytical linearisation of the B-plane projection to
        propagate a 6×6 Cartesian covariance (in J2000) into the 3×3 space
        ``[B·T, B·R, ltof]``.

        Assumes no cross-correlation between the B-plane position components
        and the linearised time-of-flight.

        Parameters
        ----------
        sc_cov : array-like, shape (6, 6)
            Full covariance matrix of the spacecraft state in J2000
            (position–velocity).
        sc_rel_state_j2000 : ArrayWUnits, shape (6,)
            Relative Cartesian state in J2000 at the close-approach epoch.

        Returns
        -------
        P_bplane : ArrayWUnits, shape (3, 3)
            B-plane parameter covariance with units
            ``[[km², km², km·s], [km², km², km·s], [km·s, km·s, s²]]``.
        """
        return self._map_covariance(sc_cov, sc_rel_state_j2000)

    def target_spacecraft_covariance(
        self,
        sc_targ_cov: np.ndarray,
        sc_rel_state_j2000: ArrayWUnits,
    ) -> ArrayWUnits:
        """
        Map a joint spacecraft-and-target covariance to B-plane parameter covariance.

        Accounts for uncertainty in both the spacecraft and the target body
        by first forming the relative covariance via the 6×12 subtraction
        matrix, then propagating as in :meth:`target_covariance`.

        Parameters
        ----------
        sc_targ_cov : array-like, shape (12, 12)
            Joint covariance matrix of the spacecraft and target states in
            J2000, ordered as ``[sc_pos, sc_vel, tgt_pos, tgt_vel]``.
        sc_rel_state_j2000 : ArrayWUnits, shape (6,)
            Relative Cartesian state ``sc − target`` in J2000 at close approach.

        Returns
        -------
        P_bplane : ArrayWUnits, shape (3, 3)
            B-plane parameter covariance (same units as
            :meth:`target_covariance`).
        """
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))
        T = np.block(
            [
                [I3, Z3, -I3, Z3],
                [Z3, I3, Z3, -I3],
            ]
        )
        rel_cov = T @ sc_targ_cov @ T.T
        return self._map_covariance(rel_cov, sc_rel_state_j2000)

    def propagate_covariance_to_close_approach(
        self,
        sc_cov: np.ndarray,
        sc_rel_state_j2000: ArrayWUnits,
        process_noise: Optional[np.ndarray] = None,
    ) -> ArrayWUnits:
        """
        Propagate a state covariance to the close-approach epoch.

        .. note::
            Not yet implemented.  A correct implementation requires a state
            transition matrix (STM) Φ from the manoeuvre epoch to close
            approach, so that::

                P_ca = Φ @ P @ Φ.T + Q

            where Q is the process-noise covariance.  The mapped B-plane
            covariance is then obtained via :meth:`_map_covariance` evaluated
            at the close-approach state.  Implement once the ``Trajectory``
            class exposes ``get_stm(epoch_start, epoch_end) -> ndarray``.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "propagate_covariance_to_close_approach is not yet implemented. "
            "A state transition matrix (STM) from the manoeuvre epoch to close "
            "approach is required. Implement once Trajectory.get_stm() is available."
        )

    # -------------------------------------------------------------------------
    # Private covariance helper (shared by the two public wrappers)
    # -------------------------------------------------------------------------

    def _map_covariance(
        self,
        rel_cov_6x6: np.ndarray,
        sc_rel_state_j2000: ArrayWUnits,
    ) -> ArrayWUnits:
        """
        Core routine: map a 6×6 relative Cartesian covariance to B-plane space.

        Parameters
        ----------
        rel_cov_6x6 : ndarray, shape (6, 6)
            Relative state covariance in J2000.
        sc_rel_state_j2000 : ArrayWUnits, shape (6,)
            Relative state in J2000.

        Returns
        -------
        ArrayWUnits, shape (3, 3)
            B-plane parameter covariance.
        """
        R3 = self._rot_to_bplane.values
        Z3 = np.zeros((3, 3))
        M = np.block([[R3, Z3], [Z3, R3]])

        # Rotate full covariance into B-plane frame
        P_bp = M @ rel_cov_6x6 @ M.T

        # S-hat sub-covariance: (z_pos, z_vel) entries
        P_shat = P_bp[np.ix_([2, 5], [2, 5])]

        rel_pos_bp = R3 @ sc_rel_state_j2000[0:3].values
        rel_vel_bp = R3 @ sc_rel_state_j2000[3:6].values

        vz = rel_vel_bp[2]
        pz = rel_pos_bp[2] - (rel_pos_bp[2] / vz) * vz  # = 0 by construction
        # ltof Jacobian: dltof/d[pz, vz]
        J_ltof = np.array([1.0 / vz, -rel_pos_bp[2] / vz**2])
        P_ltof = float(J_ltof @ P_shat @ J_ltof)

        P_vals = np.block(
            [
                [P_bp[0:2, 0:2], np.zeros((2, 1))],
                [np.zeros((1, 2)), np.array([[P_ltof]])],
            ]
        )
        P_units = np.array(
            [
                [km**2, km**2, km * sec],
                [km**2, km**2, km * sec],
                [km * sec, km * sec, sec**2],
            ]
        )
        return ArrayWUnits(P_vals, P_units)

    # endregion ── Public methods ──────────────────────────────────────────────
