# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    Units,
    ArrayWUnits,
    ForceModel,
    CelestialBody,
    Spacecraft,
    PointMassGravity,
    CannonballSRP,
    SphericalHarmonicsGravity,
    nPlateSRP,
    ThreeBodyGravity,
    FiniteBurn,
    ImpulsiveBurn,
    YarkovskyEffect,
    FirstOrderGaussMarkov,
    PiecewiseFirstOrderGaussMarkov,
    Frame,
    Maneuver,
)

import numpy as np
import re
import os
from autograd import jacobian
from autograd.numpy.numpy_boxes import ArrayBox
import copy

from scarabaeus.timeAndFrame.EpochArray import EpochArray


kg, km, sec, N = Units.get_units(["kg", "km", "sec", "N"])
J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


class ForceModelTranslation(ForceModel):
    """ High-level translational dynamics force model.

    This class aggregates and manages all **translational** (i.e., orbital)
    force contributions acting on a spacecraft — such as point-mass gravity,
    spherical harmonics, solar radiation pressure, and thrust — into a single,
    configurable interface. It acts as the physical backbone of trajectory
    propagation and estimation in Scarabaeus.

    The total translational acceleration is the ordered sum of all active
    sub-models:

    .. math::

        \\mathbf{a}_\\text{total} =
            \\mathbf{a}_\\text{grav}
            + \\mathbf{a}_\\text{3b}
            + \\mathbf{a}_\\text{SRP}
            + \\mathbf{a}_\\text{thrust}
            + \\mathbf{a}_\\text{Yark}
            + \\mathbf{a}_\\text{FOGM}
            + \\cdots

    where each term is non-zero only when the corresponding flag is enabled
    at construction time.

    Unlike low-level force model components (e.g., :class:`~scarabaeus.PointMassGravity`,
    :class:`~scarabaeus.CannonballSRP`), this class serves as a *meta-model*:
    it does not compute accelerations itself, but dynamically instantiates and
    combines sub-models according to the selected configuration flags.

    Parameters
    ----------
    primary_body : Spacecraft
        The primary spacecraft object to which the forces are applied. Contains
        physical properties such as mass, area, and reflectivity.
    t0 : EpochArray or float, optional
        The initial epoch for the force model.
    third_bodies : list[str] or list[CelestialBody], optional
        List of point-mass bodies to include when ``third_body=True``.
    cannonball_SRP : bool, optional
        Enable a classical cannonball solar radiation pressure model.
        Ignored if ``nplate_SRP=True``.
    nplate_SRP : bool, optional
        Enable a multi-facet (n-plate) SRP model. Takes precedence over
        ``cannonball_SRP``.
    sph_harm : bool, optional
        Enable a spherical harmonics gravity field expansion.
    sph_harm_body : CelestialBody, optional
        The celestial body for which the spherical harmonics are defined.
    sph_harm_order : int, optional
        The maximum harmonic order (m) to include.
    sph_harm_cs_file : str, optional
        Path to the coefficient file containing Cnm/Snm terms.
    sph_harm_norm_flag : bool, optional
        Whether the coefficients are normalized (True) or unnormalized (False).
    sph_degree_max : int, optional
        The maximum spherical harmonic degree (n) to include. Default is 100.
    finite_burn : bool, optional
        Enable continuous-thrust (finite burn) modeling.
    maneuver : Maneuver, optional
        Maneuver definition object for finite-burn or impulsive dynamics.
    yarkovsky_effect : bool, optional
        Enable Yarkovsky effect modeling.

    Notes
    -----
    The constructor defines which physical models are *active* and stores all
    metadata (files, degrees, body references, etc.) needed for later use.
    Actual sub-model objects are not built until :meth:`initialize_force_models`
    is called, which binds the model to a specific :class:`~scarabaeus.StateArray`,
    reference frame, and epoch.

    When both SRP flags are ``False``, SRP effects are excluded entirely.
    If spherical harmonics are not activated, a baseline point-mass gravity
    model is always included.

    See Also
    --------
    scarabaeus.ForceModel : Abstract parent class defining the shared interface.
    scarabaeus.ForceModelRotation : Rotational placeholder (attitude dynamics stub).
    scarabaeus.PointMassGravity : Single-body Newtonian gravity.
    scarabaeus.SphericalHarmonicsGravity : Gravity field expansion model.
    scarabaeus.CannonballSRP : Cannonball solar radiation pressure model.
    scarabaeus.nPlateSRP : Multi-plate solar radiation pressure model.
    scarabaeus.FiniteBurn : Continuous finite-duration thrust model.
    scarabaeus.ImpulsiveBurn : Instantaneous delta-v model.

    References
    ----------
    Vallado, D. A. *Fundamentals of Astrodynamics and Applications*, 4th Ed.,
    Microcosm Press, 2013.
    """

    def __init__(
        self,
        primary_body: Spacecraft,
        t0: EpochArray | float = None,
        # Gravity
        third_bodies: list | None = None,
        # SRP
        cannonball_SRP: bool = False,
        nplate_SRP: bool = False,
        # Spherical Harmonics
        sph_harm: bool = False,
        sph_harm_body: CelestialBody | None = None,
        sph_harm_order: int | None = None,
        sph_harm_cs_file: str | None = None,
        sph_harm_norm_flag: bool | None = None,
        sph_degree_max: int = 100,
        # Thrust / maneuvers
        finite_burn: bool = False,
        maneuver: Maneuver | None = None,
        # Yarkovsky Effect
        yarkovsky_effect: bool = False,
        # First-Order Gauss-Markov
        first_order_gauss_markov: bool = False,
        fogm_beta: np.ndarray | float | None = None,
        # Piecewise First-Order Gauss-Markov
        piecewise_first_order_gauss_markov: bool = False,
        pfogm_beta: np.ndarray | float | None = None,
        pfogm_batch_length: float | None = None,
        pfogm_n_batches: int | None = None,
    ) -> None:
        super().__init__()

        dynamics_mode_list: list[str] = []

        # Spherical Harmonics
        if sph_harm:
            missing = [
                name
                for name, val in {
                    "sph_harm_body": sph_harm_body,
                    "sph_harm_order": sph_harm_order,
                    "sph_harm_cs_file": sph_harm_cs_file,
                    "sph_harm_norm_flag": sph_harm_norm_flag,
                }.items()
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"Spherical harmonics enabled but missing: {', '.join(missing)}"
                )
            dynamics_mode_list.append("SphericalHarmonicsGravity")

        # Third-body gravity
        if third_bodies is not None and len(third_bodies) > 0:
            dynamics_mode_list.append("ThreeBodyGravity")

        # SRP (n-plate has precedence)
        if nplate_SRP:
            dynamics_mode_list.append("nPlateSRP")
        elif cannonball_SRP:
            dynamics_mode_list.append("CannonballSRP")

        # Finite burn
        if finite_burn:
            dynamics_mode_list.append("FiniteBurn")

        # Yarkovsky Effect
        if yarkovsky_effect:
            dynamics_mode_list.append("YarkovskyEffect")

        # First-Order Gauss-Markov, non-piecewise and piecewise
        if first_order_gauss_markov and piecewise_first_order_gauss_markov:
            raise ValueError(
                "Cannot enable both 'FirstOrderGaussMarkov' and "
                "'PiecewiseFirstOrderGaussMarkov' at the same time."
            )
        if first_order_gauss_markov:
            dynamics_mode_list.append("FirstOrderGaussMarkov")
        if piecewise_first_order_gauss_markov:
            dynamics_mode_list.append("PiecewiseFirstOrderGaussMarkov")

        # Always ensure baseline point-mass gravity if no SH
        if "SphericalHarmonicsGravity" not in dynamics_mode_list:
            dynamics_mode_list.append("PointMassGravity")

        # Store config
        self._dynamics_mode_list = dynamics_mode_list
        self._third_bodies = third_bodies
        self._primary_body = primary_body

        self._sph_harm_order = sph_harm_order if sph_harm else 0
        self._sph_harm_cs_file = sph_harm_cs_file if sph_harm else None
        self._sph_harm_body = sph_harm_body if sph_harm else None
        self._sph_harm_norm_flag = sph_harm_norm_flag if sph_harm else False
        self._sph_harm_degree_max = sph_degree_max if sph_harm else 0

        self._maneuver = maneuver
        self._first_order_gauss_markov = first_order_gauss_markov
        self._fogm_beta = fogm_beta
        self._piecewise_first_order_gauss_markov = piecewise_first_order_gauss_markov
        self._pfogm_beta = pfogm_beta
        self._pfogm_batch_length = pfogm_batch_length
        self._pfogm_n_batches = pfogm_n_batches
        self._t0 = t0

    # Properties specific to translation
    @property
    def dynamics_mode_list(self) -> list[str]:
        """Ordered list of active force model names (strings) built during construction."""
        return self._dynamics_mode_list

    @property
    def third_bodies(self) -> list[str]:
        """List of point-mass bodies included in third-body gravity, or ``None``."""
        return self._third_bodies

    @property
    def primary_body(self) -> Spacecraft:
        """Primary spacecraft object to which all forces are applied."""
        return self._primary_body

    @property
    def sph_harm_order(self) -> int:
        """Maximum harmonic order (m) used in the spherical harmonics model."""
        return self._sph_harm_order

    @property
    def sph_harm_cs_file(self) -> str | None:
        """Path to the Cnm/Snm coefficient file, or ``None`` when SH is inactive."""
        return self._sph_harm_cs_file

    @property
    def sph_harm_body(self) -> CelestialBody | None:
        """Central body for which spherical harmonics are defined, or ``None``."""
        return self._sph_harm_body

    @property
    def sph_harm_norm_flag(self) -> bool:
        """``True`` if coefficients are normalized; ``False`` for unnormalized."""
        return self._sph_harm_norm_flag

    @property
    def sph_degree_max(self) -> int:
        """Maximum harmonic degree (n) included in the expansion."""
        return self._sph_harm_degree_max

    @property
    def fogm_beta(self) -> float | None:
        """Time-correlation coefficient(s) for the first-order Gauss-Markov model."""
        return self._fogm_beta

    @property
    def pfogm_beta(self) -> float | None:
        """Time-correlation coefficient(s) for the piecewise first-order Gauss-Markov model."""
        return self._pfogm_beta

    @property
    def pfogm_batch_length(self) -> float | None:
        """Duration of each piecewise FOGM batch in seconds."""
        return self._pfogm_batch_length

    @property
    def pfogm_n_batches(self) -> int | None:
        """Number of batches in the piecewise FOGM model."""
        return self._pfogm_n_batches

    @property
    def maneuver(self) -> Maneuver | None:
        """Maneuver definition used by the finite-burn sub-model, or ``None``."""
        return self._maneuver

    @maneuver.setter
    def maneuver(self, value: Maneuver | None):
        """
        Replace the maneuver object on the live finite-burn sub-model.

        Parameters
        ----------
        value : Maneuver or None
            New maneuver to use. Passing ``None`` effectively disables finite-burn
            dynamics until a valid ``Maneuver`` is assigned.
        """
        self._maneuver = value

    # Internal model wiring
    def _initialize_force_models(self) -> dict:
        """
        Instantiate all active translational force models.

        Requires `_state_vector`, `_base_frame`, and `_t0` to be set.
        """
        if self._state_vector is None:
            raise RuntimeError(
                "state_vector not set. Call initialize_force_models() first."
            )

        sv = self._state_vector
        bf = self._base_frame
        t0 = self._t0

        force_models: dict = {}

        if "PointMassGravity" in self._dynamics_mode_list:
            force_models["PointMassGravity"] = PointMassGravity(sv, bf)

        if "ThreeBodyGravity" in self._dynamics_mode_list:
            force_models["ThreeBodyGravity"] = ThreeBodyGravity(
                sv, self._third_bodies, bf
            )

        if "SphericalHarmonicsGravity" in self._dynamics_mode_list:
            force_models["SphericalHarmonicsGravity"] = SphericalHarmonicsGravity(
                self._sph_harm_order,
                self._sph_harm_cs_file,
                self._sph_harm_body,
                sv,
                self._sph_harm_norm_flag,
                self._sph_harm_degree_max,
                bf,
            )

        if "CannonballSRP" in self._dynamics_mode_list:
            force_models["CannonballSRP"] = CannonballSRP(sv, self._primary_body, bf)

        if "nPlateSRP" in self._dynamics_mode_list:
            force_models["nPlateSRP"] = nPlateSRP(sv, self._primary_body, bf)

        if "FiniteBurn" in self._dynamics_mode_list:
            force_models["FiniteBurn"] = FiniteBurn(
                self._primary_body,
                bf,
                self._maneuver,
                sv,
            )

        if "ImpulsiveBurn" in self._dynamics_mode_list:
            dv_key = next(
                (
                    key
                    for key in ["dv_man", "dv_man1", "dv_man2"]
                    if hasattr(sv, "index_rep") and key in sv.index_rep
                ),
                None,
            )
            if dv_key:
                dv_value = next(
                    (item for item in sv.state if item[0] == dv_key),
                    None,
                )[5]
                force_models["ImpulsiveBurn"] = ImpulsiveBurn(
                    delta_v=dv_value,
                    t_burn=t0,
                    base_frame=bf,
                )

        if "YarkovskyEffect" in self._dynamics_mode_list:
            force_models["YarkovskyEffect"] = YarkovskyEffect(
                state_vector=self._state_vector,
                base_frame=self.base_frame,
                param_name="A2_yark",  # or whatever you used
            )

        if "FirstOrderGaussMarkov" in self._dynamics_mode_list:
            force_models["FirstOrderGaussMarkov"] = FirstOrderGaussMarkov(
                sv,
                self._primary_body,
                bf,
                default_beta=self._fogm_beta,
            )
        if "PiecewiseFirstOrderGaussMarkov" in self._dynamics_mode_list:
            force_models["PiecewiseFirstOrderGaussMarkov"] = (
                PiecewiseFirstOrderGaussMarkov(
                    sv,
                    self._primary_body,
                    bf,
                    t0=self._t0,
                    default_beta=self._pfogm_beta,
                    batch_length=self._pfogm_batch_length,
                    n_batches=self._pfogm_n_batches,
                )
            )
            if t0 == None:
                raise ValueError(
                    "t0 must be provided for PiecewiseFirstOrderGaussMarkov model."
                )

        return force_models

    def _find_body_by_spice_id(self, sid: int):
        """
        Return the owner object in the state vector whose ``spice_id`` matches *sid*.

        Parameters
        ----------
        sid : int
            SPICE integer ID to search for.

        Returns
        -------
        object or None
            The first matching owner, or ``None`` if not found.
        """
        # Look through the state vector template for owners
        for name, _, _, _, owner, _ in self._state_vector.state:
            if getattr(owner, "spice_id", None) == sid:
                return owner
        return None

    def _norm(self, s: str) -> str:
        """Return *s* stripped and lower-cased for case-insensitive key matching."""
        return str(s).strip().lower()

    def _body_tag(self, body) -> str:
        """
        Return the ``@<spice_id>`` string tag for *body*.

        Parameters
        ----------
        body : object
            Object with an optional ``spice_id`` attribute.

        Returns
        -------
        str
            ``"@<sid>"`` if ``spice_id`` is present, else ``"@none"``.
        """
        sid = getattr(body, "spice_id", None)
        return f"@{sid}" if sid is not None else "@none"

    def _collect_parameter_dynamics(
        self,
        name: str,
        position: ArrayWUnits,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ):
        """
        Retrieves the dynamic equation associated with a specified parameter.

        Acts as a lookup for the governing equations of various dynamic
        parameters, returning the appropriate function based on the provided name.

        Parameters
        ----------
        name : str
            The name of the dynamic parameter.

        position : ArrayWUnits
            3x1 vector describing the position of the affected object. Expressed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch.

        body : CelestialBody
            The celestial body influencing the dynamics.

        Returns
        -------
        param_dynamics : Callable
            The function defining the parameter's dynamics, or raises an error
            if not found.
        """

        dynamics_parameters = {}

        if "FirstOrderGaussMarkov" in self.force_models:
            gm = self.force_models["FirstOrderGaussMarkov"]

            a_dot = gm._compute_acceleration_state_derivative(
                position, current_state, epoch, body
            ).reshape(3)

            dynamics_parameters["a_fogm_dot"] = lambda: ArrayWUnits(
                a_dot,
                km * sec**-3,
            )

        if "PiecewiseFirstOrderGaussMarkov" in self.force_models:
            pwgm = self.force_models["PiecewiseFirstOrderGaussMarkov"]

            a_dot = pwgm._compute_acceleration_state_derivative(
                position, current_state, epoch, body
            ).reshape(3 * pwgm.n_batches)

            dynamics_parameters["a_pfogm_dot"] = lambda: ArrayWUnits(
                a_dot,
                km * sec**-3,
            )

        try:
            return dynamics_parameters[name]()
        except Exception as e:
            raise Exception("No ODE found for: {}".format(e))

    def _collect_partials(
        self, name: str, position: ArrayWUnits, current_state: dict, epoch, body
    ):
        """
        Retrieves the function computing the partial derivative of a specified dynamic parameter.

        Provides a mapping between parameter names and their corresponding
        derivative computation functions.

        Parameters
        ----------
        name : str
            The key representing the desired partial derivative computation.

        position : ArrayWUnits
            3x1 vector describing the position of the affected object. Expressed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch.

        body : CelestialBody
            The celestial body influencing the dynamics.

        Returns
        -------
        partials : Callable or None
            The function computing the partial derivative if found,
            otherwise None.
        """
        # Regex to parse the key
        _KEY_RE = re.compile(
            r"^(?P<lhs>\w+)@(?P<lhs_id>-?\d+)_by_(?P<rhs>\w+)@(?P<rhs_id>-?\d+)$"
        )
        m = _KEY_RE.match(str(name).strip())
        if not m:
            # backward compatibility: try old naming if you want
            return None

        lhs = m.group("lhs").lower()
        lhs_id = int(m.group("lhs_id"))
        rhs = m.group("rhs").lower()
        rhs_id = int(m.group("rhs_id"))

        # Strip maneuver numbering suffixes: k_thrust1 -> k_thrust, dRA2 -> dra, etc.
        lhs_base = re.sub(r"\d+$", "", lhs)
        rhs_base = re.sub(r"\d+$", "", rhs)

        # Sanity: this force model instance is for self._primary_body,
        # but you may still compute partials for whichever `body` is passed in.
        if getattr(body, "spice_id", None) != lhs_id:
            # asked for a partial of a different row-body than the one supplied
            return None

        # dv/dv block
        if lhs_base == "velocity" and rhs_base == "velocity" and rhs_id == lhs_id:
            return self._compute_total_velocity_partial(
                position, current_state, epoch, body
            )

        # da/d(position@something)
        if lhs_base == "acceleration" and rhs_base == "position":
            # If wrt is SAME body -> standard partial
            if rhs_id == lhs_id:
                return self._compute_total_acceleration_partial(
                    position, current_state, epoch, body
                )

            # If wrt is a propagated third body (CelestialBody position in state),
            # return cross partial from ThreeBodyGravity if applicable.
            # We need an object for the wrt body:
            wrt_body = self._find_body_by_spice_id(rhs_id)  # implement once (below)
            if wrt_body is None:
                return None

            # Only ThreeBodyGravity contributes cross-pos dependence here
            if "ThreeBodyGravity" in self.force_models:
                threeb = self.force_models["ThreeBodyGravity"]
                # map wrt body to Constants key (typically name upper)
                pm_key = getattr(wrt_body, "name", "").upper()
                if pm_key in getattr(threeb, "pnt_masses", []):
                    d = threeb._compute_partial_by_third_body_position(
                        position=position.values,
                        current_state=current_state,
                        epoch=epoch,
                        body=body,
                        point_mass_name=pm_key,
                    )
                    if d is None:
                        return None
                    return ArrayWUnits(d, sec**-2)

            return None

        # da/d(eta_srp@same spacecraft)
        if lhs_base == "acceleration" and rhs_base == "eta_srp":
            if rhs_id == lhs_id:
                return self._compute_acceleration_partial_by_eta(
                    position, current_state, epoch, body
                )
            return None

        # da/d(k_thrust@same spacecraft)
        if lhs_base == "acceleration" and rhs_base == "k_thrust":
            if rhs_id == lhs_id:
                if "FiniteBurn" not in self.force_models:
                    return None
                fb = self.force_models["FiniteBurn"]

                d = fb._compute_partial_by_k_thrust(
                    position=position.values,
                    current_state=current_state,
                    epoch=epoch,
                    body=body,
                )
                return ArrayWUnits(d, sec**-2)  # (3,1)
            return None

        # da/d(dRA@same spacecraft) and da/d(dDEC@same spacecraft)
        if lhs_base == "acceleration" and rhs_base in ("dra", "ddec"):
            if rhs_id == lhs_id:
                if "FiniteBurn" not in self.force_models:
                    return None
                fb = self.force_models["FiniteBurn"]

                J = fb._compute_partial_by_ra_dec_bias(
                    position=position.values,
                    current_state=current_state,
                    epoch=epoch,
                    body=body,
                )  # (3,2)

                # Column pick: [dRA, dDEC]
                col = 0 if rhs_base == "dra" else 1
                d = J[:, col].reshape(3, 1)

                return ArrayWUnits(d, sec**-2)  # (3,1)
            return None

        # da/d(a_fogm@same spacecraft)
        if lhs_base == "acceleration" and rhs_base == "a_fogm":
            if rhs_id == lhs_id:
                if "FirstOrderGaussMarkov" not in self.force_models:
                    return None
                gm = self.force_models["FirstOrderGaussMarkov"]
                return gm._compute_partial_by_acceleration_state(
                    position, current_state, epoch, body
                )
            return None

        # da/d(a_pfogm@same spacecraft)
        if lhs_base == "acceleration" and rhs_base == "a_pfogm":
            if rhs_id == lhs_id:
                if "PiecewiseFirstOrderGaussMarkov" not in self.force_models:
                    return None
                gm = self.force_models["PiecewiseFirstOrderGaussMarkov"]
                return gm._compute_partial_by_acceleration_state(
                    position, current_state, epoch, body
                )
            return None

        # da_fogm / d(a_fogm)
        if lhs_base == "a_fogm_dot" and rhs_base == "a_fogm" and rhs_id == lhs_id:
            if "FirstOrderGaussMarkov" not in self.force_models:
                return None
            gm = self.force_models["FirstOrderGaussMarkov"]
            J = gm._compute_partial_of_acceleration_state_by_acceleration_state(
                position, current_state, epoch, body
            )
            return ArrayWUnits(J, sec**-1)

        # da_fogm / d(a_pfogm)
        if lhs_base == "a_fogm_dot" and rhs_base == "a_pfogm" and rhs_id == lhs_id:
            if "PiecewiseFirstOrderGaussMarkov" not in self.force_models:
                return None
            gm = self.force_models["PiecewiseFirstOrderGaussMarkov"]
            J = gm._compute_partial_of_acceleration_state_by_acceleration_state(
                position, current_state, epoch, body
            )
            return ArrayWUnits(J, sec**-1)

        # da_fogm / d(beta_fogm)
        if lhs_base == "a_fogm_dot" and rhs_base == "beta_fogm" and rhs_id == lhs_id:
            if "FirstOrderGaussMarkov" not in self.force_models:
                return None
            gm = self.force_models["FirstOrderGaussMarkov"]
            J = gm._compute_partial_of_acceleration_state_by_beta(
                current_state, epoch, body
            )
            return ArrayWUnits(J, km * sec**-2)

        # da_fogm / d(beta_pfogm)
        if lhs_base == "a_fogm_dot" and rhs_base == "beta_pfogm" and rhs_id == lhs_id:
            if "PiecewiseFirstOrderGaussMarkov" not in self.force_models:
                return None
            gm = self.force_models["PiecewiseFirstOrderGaussMarkov"]
            J = gm._compute_partial_of_acceleration_state_by_beta(
                current_state, epoch, body
            )
            return ArrayWUnits(J, km * sec**-2)

        # -------------------------------------------------------------------
        # da/d(gm@<body_id>)  — SH body GM, central-body GM, or third-body GM
        # -------------------------------------------------------------------
        if lhs_base == "acceleration" and rhs_base == "gm":
            # SH body GM (when SphericalHarmonicsGravity is active — no PointMassGravity
            # for the same body in that case)
            if "SphericalHarmonicsGravity" in self.force_models:
                sh = self.force_models["SphericalHarmonicsGravity"]
                if rhs_id == getattr(sh, "_sph_body_spice_id", None):
                    d = sh._compute_partial_by_mu(position.values, current_state, epoch, body)
                    return ArrayWUnits(d, km**-2)
            # Central body (PointMassGravity)
            if "PointMassGravity" in self.force_models:
                pm = self.force_models["PointMassGravity"]
                if rhs_id == getattr(pm, "_origin_spice_id", None):
                    d = pm._compute_partial_by_gm(position, current_state, epoch, body)
                    return d  # already ArrayWUnits with proper units
            # Third body (ThreeBodyGravity)
            if "ThreeBodyGravity" in self.force_models:
                tb = self.force_models["ThreeBodyGravity"]
                for pm_name, pm_sid in tb._pnt_mass_spice_ids.items():
                    if pm_sid == rhs_id:
                        d = tb._compute_partial_by_gm(
                            position.values, current_state, epoch, body, pm_name
                        )
                        return ArrayWUnits(d, km**-2)
            return None

        # -------------------------------------------------------------------
        # da/d(cnm_n_m@<body_id>)  and  da/d(snm_n_m@<body_id>)
        # -------------------------------------------------------------------
        if lhs_base == "acceleration" and "SphericalHarmonicsGravity" in self.force_models:
            sh = self.force_models["SphericalHarmonicsGravity"]
            if rhs_id == getattr(sh, "_sph_body_spice_id", None):
                cnm_match = re.match(r"cnm_(\d+)_(\d+)$", rhs)
                if cnm_match:
                    nn, mm = int(cnm_match.group(1)), int(cnm_match.group(2))
                    if nn <= sh.order and mm <= nn:
                        partials_C = sh._compute_partial_by_C(position.values, epoch)
                        d = partials_C[:, nn, mm].reshape(3, 1)
                        return ArrayWUnits(d, km * sec**-2)
                    return None
                snm_match = re.match(r"snm_(\d+)_(\d+)$", rhs)
                if snm_match:
                    nn, mm = int(snm_match.group(1)), int(snm_match.group(2))
                    if nn <= sh.order and mm <= nn and mm > 0:
                        partials_S = sh._compute_partial_by_S(position.values, epoch)
                        d = partials_S[:, nn, mm].reshape(3, 1)
                        return ArrayWUnits(d, km * sec**-2)
                    return None

        # -------------------------------------------------------------------
        # da/d(pole_ra@<body_id>), da/d(pole_dec@<body_id>),
        # da/d(pole_pm@<body_id>)
        # -------------------------------------------------------------------
        if lhs_base == "acceleration" and rhs_base in ("pole_ra", "pole_dec", "pole_pm"):
            if "SphericalHarmonicsGravity" not in self.force_models:
                return None
            sh = self.force_models["SphericalHarmonicsGravity"]
            if rhs_id != getattr(sh, "_sph_body_spice_id", None):
                return None
            if rhs_base == "pole_ra":
                d = sh._compute_partial_by_pole_RA(position.values, current_state, epoch, body)
                return ArrayWUnits(d, km * sec**-2)
            if rhs_base == "pole_dec":
                d = sh._compute_partial_by_pole_DEC(position.values, current_state, epoch, body)
                return ArrayWUnits(d, km * sec**-2)
            if rhs_base == "pole_pm":
                d = sh._compute_partial_by_pole_PM(position.values, current_state, epoch, body)
                # units: km/s^2 / (deg/day) — keep as km*s^-2 for filter consistency
                return ArrayWUnits(d, km * sec**-2)
            return None

        return None

    def _compute_total_acceleration(
        self,
        position: ArrayWUnits,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Computes the total acceleration from all selected dynamical models.

        Iterates through the active force models and computes the
        cumulative acceleration acting on the spacecraft.

        Parameters
        ----------
        position : ArrayWUnits
            3x1 vector describing the position of the spacecraft. Expressed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch.

        body : CelestialBody
            The celestial body influencing the dynamics.

        Returns
        -------
        total_acceleration : np.ndarray
            3x1 vector describing the total acceleration of the spacecraft. Expressed in units
            of :math:`\\frac{km}{s^2}`.
        """

        # Initialize total acceleration
        total_acceleration = np.zeros(3)

        # Get the position vector
        position_vec = position.values

        # Stack toghether individual acceleration components from the list of dynamics model
        for model in self._dynamics_mode_list:
            model_acceleration = self.force_models[model].compute_acceleration(
                position_vec, current_state, epoch, body
            )
            # Add accelerations
            total_acceleration += model_acceleration

        # return total_acceleration
        return total_acceleration

    def _compute_total_velocity_partial(
        self, position: ArrayWUnits, current_state: dict, epoch, body
    ) -> ArrayWUnits:
        """
        Computes the Jacobian of velocity with respect to itself.

        Since velocity is directly mapped to itself in the state transition, the
        partial derivative results in an identity matrix.

        Parameters
        ----------
        position : ArrayWUnits
            3x1 vector describing the position of the spacecraft. Expressed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch.

        body : CelestialBody
            The celestial body influencing the dynamics.

        Returns
        -------
        total_partial_awu : ArrayWUnits
            3x3 identity matrix representing dv/dv. Unitless.
        """

        return ArrayWUnits(np.eye(3), None)

    def _compute_total_acceleration_partial(
        self, position: ArrayWUnits, current_state: dict, epoch, body
    ) -> ArrayWUnits:
        """
        Computes the total acceleration Jacobian with respect to position.

        Sums the contributions of all active force models to determine
        how the acceleration changes with respect to position.

        Parameters
        ----------
        position : ArrayWUnits
            3x1 vector describing the position of the spacecraft. Expressed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch.

        body : CelestialBody
            The celestial body influencing the dynamics.

        Returns
        -------
        total_partial_awu : ArrayWUnits
            The 3x3 Jacobian matrix of acceleration with respect to position. Expressed in units
            of :math:`\\frac{1}{s^2}`.
        """

        # Initialize the partial acceleration
        partial_a_by_pos = np.zeros((3, 3))

        # Get the position vector
        position_vec = position.values

        # Stack toghether individual partials components from the list of dynamics model
        for model in self._dynamics_mode_list:
            if model in self.force_models:
                model_partial = self.force_models[model]._compute_partial_by_position(
                    position_vec, current_state, epoch, body
                )
                # Add partials
                partial_a_by_pos += model_partial

        # Add units to partials
        total_partial_awu = ArrayWUnits(partial_a_by_pos, sec**-2)

        return total_partial_awu

    def _compute_acceleration_partial_by_dv(
        self, position: ArrayWUnits, current_state: dict, epoch, body
    ) -> ArrayWUnits:
        """
        Computes the partial derivative of acceleration with respect to the applied delta-v.

        Relevant for impulsive maneuvers where the spacecraft undergoes
        a sudden change in velocity.

        Parameters
        ----------
        position : ArrayWUnits
            3x1 vector describing the position of the spacecraft. Expressed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch.

        body : CelestialBody
            The celestial body influencing the dynamics.

        Returns
        -------
        total_partial_awu : ArrayWUnits
            The 3x3 partial derivative of acceleration with respect to delta-v. Expressed in units
            of :math:`\\frac{1}{s}`.
        """
        # Get the position vector
        position_vec = position.values

        # Compute partial of acceleration by DV
        partial_a_by_dv = self.force_models["ImpulsiveBurn"]._compute_partial_by_dv(
            position_vec, current_state, epoch, body
        )
        partial_a_by_dv = ArrayWUnits(partial_a_by_dv, sec**-1)

        return partial_a_by_dv

    def _compute_acceleration_partial_by_eta(
        self, position: ArrayWUnits, current_state: dict, epoch, body
    ) -> ArrayWUnits:
        """
        Computes the partial derivative of acceleration with respect to the solar radiation
        pressure (SRP) scale factor.

        Accounts for different SRP models (cannonball or n-plate) and
        returns the corresponding derivative.

        Parameters
        ----------
        position : ArrayWUnits
            3x1 vector describing the position of the spacecraft. Expressed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch.

        body : CelestialBody
            The celestial body influencing the dynamics.

        Returns
        -------
        partial_by_eta : ArrayWUnits
            The 3x3 partial derivative of acceleration with respect to SRP. Expressed in units
            of :math:`\\frac{km}{s^2}`.
        """
        # Get the position vector
        position_vec = position.values

        # Check for "CannonballSRP" in dynamics mode list and compute partial_a_by_eta
        if "CannonballSRP" in self._dynamics_mode_list:
            partial_a_by_eta = self.force_models[
                "CannonballSRP"
            ]._compute_partial_by_eta(position_vec, current_state, epoch, body)
        # Check for "nPlateSRP" in dynamics mode list
        if "nPlateSRP" in self._dynamics_mode_list:
            partial_a_by_eta = self.force_models["nPlateSRP"]._compute_partial_by_eta(
                position_vec, current_state, epoch, body
            )

        partial_a_by_eta_awu = ArrayWUnits(partial_a_by_eta.reshape(3, 1), km * sec**-2)

        return partial_a_by_eta_awu

    def _check_all_partials(self, position, full_state, epoch, body):
        """
        Numerically/algorithmically validate force-model partial derivatives against
        the corresponding analytical implementations.

        This routine:
        1) For each dynamics mode in `self._dynamics_mode_list`, computes an
            algorithmic Jacobian of acceleration w.r.t. position using autograd
            (only when `UseAutoGrad=TRUE`).
        2) Computes analytical partials via each force model's
            `compute_partial_by_*` methods.
        3) Writes a CSV-style line per partial comparison to `partials_FMoutput.txt`:

            epoch,model,d(acc_<dep>)/d(<indep>),analytical,algo,error

        Notes / preserved behavior:
        - The environment-variable gate is unchanged:
                if "UseAutoGrad" in os.environ.keys() and os.environ["UseAutoGrad"] == "TRUE"
        - All wrappers keep the same signatures, internal side effects, and
            `pos.values` vs `pos._values` usage as in your original code.
        - Output file name, formatting, and the error definition are unchanged:
                error = (analytical - algo) / algo
        - This is intentionally verbose; it is a diagnostic utility.

        Parameters
        ----------
        position : ArrayWUnits-like
            Must provide `.values` (and for some branches, `._values`).
        full_state : dict-like
            State container used by the force models. Mutated in SRP wrappers.
        epoch : any
            Epoch passed through to the force model.
        body : object
            Must provide `spice_id` used for SRP parameter mapping.

        Returns
        -------
        None
            Writes diagnostics to `partials_FMoutput.txt`.
        """
        # Keep file name identical and minimize repeated string literals.
        out_path = "partials_FMoutput.txt"

        # Small local helper: identical format/output, but avoids copy-paste.
        def _write_line(model, dvar, ivar, analytical, algo):
            error = (analytical - algo) / algo
            with open(out_path, mode="a", encoding="utf-8") as f:
                f.write(
                    "{epoch},{model},d(acc_{dvar})/d({ivar}),{a},{b},{c}\n".format(
                        dvar=dvar,
                        ivar=ivar,
                        a=analytical,
                        b=algo,
                        c=error,
                        epoch=epoch,
                        model=model,
                    )
                )

        # ------------------------------------------------------------------
        # Wrappers for autograd (must remain inner functions to close over `mode`)
        # ------------------------------------------------------------------
        def wrap_acc(posvals, full_state, epoch, body):
            acc = self.force_models[mode].compute_acceleration(
                posvals, full_state, epoch, body
            )
            return acc

        def wrap_gm(gmval, pos, full_state, epoch, body):
            self.force_models[mode].origin_mu = gmval
            acc = self.force_models[mode].compute_acceleration(
                pos.values, full_state, epoch, body
            )
            return acc

        def wrap_eta(etaval, units, pos, full_state, epoch, body):
            self.force_models[mode]._eta_srp_map[body.spice_id] = etaval
            full_state[("eta_srp", body.spice_id)] = np.array(etaval)
            acc = self.force_models[mode].compute_acceleration(
                pos.values, full_state, epoch, body
            )
            return acc

        def wrap_C(Cval, units, pos, full_state, epoch, body):
            self.force_models[mode].Cnm = Cval
            acc = self.force_models[mode].compute_acceleration(
                pos._values, full_state, epoch, body
            )
            return acc

        def wrap_S(Sval, units, pos, full_state, epoch, body):
            self.force_models[mode].Snm = Sval
            acc = self.force_models[mode].compute_acceleration(
                pos._values, full_state, epoch, body
            )
            return acc

        def wrap_fbthrust(thrustval, units, pos, full_state, epoch, body):
            M1 = Maneuver()
            M1.thrust = ArrayWUnits(thrustval, N)
            M1.mass_flow = ArrayWUnits(3e-4, kg / sec)
            M1.start_time = ArrayWUnits(12 * 60 * 60, sec)
            M1.end_time = ArrayWUnits(13 * 60 * 60, sec)
            M1.ux = ArrayWUnits(1/np.sqrt(6), None)
            M1.uy = ArrayWUnits(3/np.sqrt(6), None)
            M1.uz = ArrayWUnits(2/np.sqrt(6), None)
            M1.ra = ArrayWUnits(33.0, None)
            M1.dec = ArrayWUnits(-12.0, None)

            self.force_models[mode].maneuver = M1
            acc = self.force_models[mode].compute_acceleration(
                pos, full_state, epoch, body
            )
            return acc

        def wrap_fbra(raval, pos, full_state, epoch, body):
            M1 = Maneuver()
            M1.thrust = ArrayWUnits(0.5, N)
            M1.mass_flow = ArrayWUnits(3e-4, kg / sec)
            M1.start_time = ArrayWUnits(2 * 60 * 60, sec)
            M1.end_time = ArrayWUnits(3 * 60 * 60, sec)
            M1.ux = ArrayWUnits(1/np.sqrt(6), None)
            M1.uy = ArrayWUnits(3/np.sqrt(6), None)
            M1.uz = ArrayWUnits(2/np.sqrt(6), None)
            M1.ra = ArrayWUnits(raval, None)
            M1.dec = ArrayWUnits(-12.0, None)

            # self.force_models[mode].maneuver = ArrayWUnits(ra, None)
            self.force_models[mode].maneuver = M1
            acc = self.force_models[mode].compute_acceleration(
                pos, full_state, epoch, body
            )
            return acc

        def wrap_fbdec(decval, pos, full_state, epoch, body):
            M1 = Maneuver()
            M1.thrust = ArrayWUnits(0.5, N)
            M1.mass_flow = ArrayWUnits(3e-4, kg / sec)
            M1.start_time = ArrayWUnits(2 * 60 * 60, sec)
            M1.end_time = ArrayWUnits(3 * 60 * 60, sec)
            M1.ux = ArrayWUnits(1/np.sqrt(6), None)
            M1.uy = ArrayWUnits(3/np.sqrt(6), None)
            M1.uz = ArrayWUnits(2/np.sqrt(6), None)
            M1.ra = ArrayWUnits(33.0, None)
            M1.dec = ArrayWUnits(decval, None)

            self.force_models[mode].maneuver = M1
            acc = self.force_models[mode].compute_acceleration(
                pos, full_state, epoch, body
            )
            return acc

        def wrap_GMaccel(accelval, pos, full_state, epoch, body):
            full_state[("a_fogm",body._spice_id)] = accelval
            acc = self.force_models[mode].compute_acceleration(
                pos, full_state, epoch, body
            )
            return acc

        def wrap_GMdriftaccel(accelval, pos, full_state, epoch, body):
            full_state[("a_fogm",body._spice_id)] = accelval
            a_dot = self.force_models[mode]._compute_acceleration_state_derivative(
                pos, full_state, epoch, body).reshape(3)

            return a_dot

        def wrap_beta(betaval, pos, full_state, epoch, body):
            self.force_models[mode].default_beta = betaval
            a_dot = self.force_models[mode]._compute_acceleration_state_derivative(
                pos, full_state, epoch, body).reshape(3)

            return a_dot

        def wrap_PGMaccel(accelval, pos, full_state, epoch, body):
            full_state[("a_pfogm",body._spice_id)] = accelval
            acc = self.force_models[mode].compute_acceleration(
                pos, full_state, epoch, body
            )
            return acc

        def wrap_PGMdriftaccel(accelval, pos, full_state, epoch, body):
            full_state[("a_pfogm",body._spice_id)] = accelval
            a_dot = self.force_models[mode]._compute_acceleration_state_derivative(
                pos, full_state, epoch, body).reshape(3 * self.force_models[mode].n_batches)

            return a_dot

        def wrap_Pbeta(betaval, pos, full_state, epoch, body):
            self.force_models[mode].default_beta = betaval
            a_dot = self.force_models[mode]._compute_acceleration_state_derivative(
                pos, full_state, epoch, body).reshape(3 * self.force_models[mode].n_batches)

            return a_dot
        
        for mode in self._dynamics_mode_list:
            # ------------------------------------------------------------------
            # Algorithmic derivatives (autograd only)
            # ------------------------------------------------------------------
            if (
                "UseAutoGrad" in os.environ.keys()
                and os.environ["UseAutoGrad"] == "TRUE"
            ):
                # dacc/d(position)
                grad = jacobian(wrap_acc)
                dacc_dstate = grad(position.values, full_state, epoch, body)

                # ------------------------------------------------------------------
                # Analytical partial: dacc/d(position)
                # ------------------------------------------------------------------
                partial_a = self.force_models[mode]._compute_partial_by_position(
                    position.values, full_state, epoch, body
                )

                # Compare position partials (same loops, now using helper writer)
                for independent_dx, independent in enumerate(["x", "y", "z"]):
                    for dependent_dx, dependent in enumerate(["x", "y", "z"]):
                        analytical = partial_a[dependent_dx][independent_dx]
                        algo = dacc_dstate[dependent_dx][independent_dx]
                        _write_line(mode, dependent, independent, analytical, algo)

                # Additional per-model derivatives (as in original)
                if mode == "PointMassGravity":
                    original_mu = copy.deepcopy(self.force_models[mode].origin_mu)
                    grad = jacobian(wrap_gm)
                    dacc_dgm = grad(
                        self.force_models[mode].origin_mu,
                        position,
                        full_state,
                        epoch,
                        body,
                    )
                    self.force_models[mode].origin_mu = original_mu

                elif mode == "CannonballSRP" or mode == "nPlateSRP":
                    original_eta = copy.deepcopy(full_state[("eta_srp", body.spice_id)])
                    eta_srp = ArrayWUnits(original_eta, None)
                    grad = jacobian(wrap_eta)
                    dacc_deta = grad(
                        eta_srp.values, eta_srp.units, position, full_state, epoch, body
                    )
                    self.force_models[mode]._eta_srp_map[body.spice_id] = original_eta
                    full_state[("eta_srp", body.spice_id)] = original_eta

                elif mode == "SphericalHarmonicsGravity":
                    original_Cnm = copy.deepcopy(self.force_models[mode].Cnm)
                    grad = jacobian(wrap_C)
                    dacc_dC = grad(
                        self.force_models[mode].Cnm,
                        None,
                        position,
                        full_state,
                        epoch,
                        body,
                    )
                    self.force_models[mode].Cnm = original_Cnm

                    original_Snm = copy.deepcopy(self.force_models[mode].Snm)
                    grad = jacobian(wrap_S)
                    dacc_dS = grad(
                        self.force_models[mode].Snm,
                        None,
                        position,
                        full_state,
                        epoch,
                        body,
                    )
                    self.force_models[mode].Snm = original_Snm

                elif mode == "FiniteBurn":

                    original_thrust = copy.deepcopy(
                        self.force_models[mode].maneuver.thrust
                    )

                    grad = jacobian(wrap_fbthrust)
                    dacc_dthrust = grad(
                        self.force_models[mode].maneuver.thrust[0].values,
                        self.force_models[mode].maneuver.thrust.units,
                        position,
                        full_state,
                        epoch,
                        body,
                    )
                    self.force_models[mode].maneuver.thrust = original_thrust

                    original_ra = copy.deepcopy(
                        self.force_models[mode].maneuver.ra
                    )

                    grad = jacobian(wrap_fbra)
                    dacc_dra = grad(
                        self.force_models[mode].maneuver.ra.values,
                        position,
                        full_state,
                        epoch,
                        body,
                    )
                    self.force_models[mode].maneuver.ra = original_ra

                    original_dec = copy.deepcopy(
                        self.force_models[mode].maneuver.dec
                    )

                    grad = jacobian(wrap_fbdec)
                    dacc_ddec = grad(
                        self.force_models[mode].maneuver.dec.values,
                        position,
                        full_state,
                        epoch,
                        body,
                    )
                    self.force_models[mode].maneuver.dec = original_dec

                elif mode == "FirstOrderGaussMarkov":
                    original_afogm = copy.deepcopy(full_state[("a_fogm",body._spice_id)])
                    grad = jacobian(wrap_GMaccel)
                    dacc_dGMaccel = grad(
                        full_state[("a_fogm",body._spice_id)],
                        position,
                        full_state,
                        epoch,
                        body,
                    )
                    full_state[("a_fogm",body._spice_id)] = original_afogm

                    grad = jacobian(wrap_GMdriftaccel)
                    dacc_dGMdriftaccel = grad(
                        full_state[("a_fogm",body._spice_id)],
                        position,
                        full_state,
                        epoch,
                        body,
                    )
                    full_state[("a_fogm",body._spice_id)] = original_afogm

                    original_beta = copy.deepcopy(self.force_models[mode].default_beta)
                    grad = jacobian(wrap_beta)
                    dacc_dbeta = grad(
                        self.force_models[mode].default_beta,
                        position,
                        full_state,
                        epoch,
                        body,
                    )
                    self.force_models[mode].default_beta = original_beta

                elif mode == "PiecewiseFirstOrderGaussMarkov":
                    original_apfogm = copy.deepcopy(full_state[("a_pfogm",body._spice_id)])
                    grad = jacobian(wrap_PGMaccel)
                    dacc_dPGMaccel = grad(
                        full_state[("a_pfogm",body._spice_id)],
                        position,
                        full_state,
                        epoch,
                        body,
                    )
                    full_state[("a_pfogm",body._spice_id)] = original_apfogm

                    grad = jacobian(wrap_PGMdriftaccel)
                    dacc_dPGMdriftaccel = grad(
                        full_state[("a_pfogm",body._spice_id)],
                        position,
                        full_state,
                        epoch,
                        body,
                    )
                    full_state[("a_pfogm",body._spice_id)] = original_apfogm

                    original_beta = copy.deepcopy(self.force_models[mode].default_beta)
                    grad = jacobian(wrap_Pbeta)
                    dacc_dPbeta = grad(
                        self.force_models[mode].default_beta,
                        position,
                        full_state,
                        epoch,
                        body,
                    )
                    self.force_models[mode].default_beta = original_beta

                else:
                    continue

            # ------------------------------------------------------------------
            # Model-specific comparisons
            # ------------------------------------------------------------------
            if mode == "PointMassGravity":
                partial_a_by_gm = self.force_models[mode]._compute_partial_by_gm(
                    position, full_state, epoch, body
                )

                for independent_dx, independent in enumerate(["gm"]):
                    for dependent_dx, dependent in enumerate(["x", "y", "z"]):
                        analytical = partial_a_by_gm.values[dependent_dx][
                            independent_dx
                        ]
                        algo = dacc_dgm[dependent_dx]
                        _write_line(mode, dependent, independent, analytical, algo)

            elif mode == "CannonballSRP":
                partial_a_by_eta = self.force_models[mode]._compute_partial_by_eta(
                    position.values, full_state, epoch, body
                )

                for independent_dx, independent in enumerate(["eta"]):
                    for dependent_dx, dependent in enumerate(["x", "y", "z"]):
                        analytical = partial_a_by_eta[dependent_dx]
                        algo = dacc_deta[dependent_dx]
                        _write_line(mode, dependent, independent, analytical, algo)

            elif mode == "nPlateSRP":
                partial_a_by_eta = self.force_models[mode]._compute_partial_by_eta(
                    position.values, full_state, epoch, body
                )

                for independent_dx, independent in enumerate(["eta"]):
                    for dependent_dx, dependent in enumerate(["x", "y", "z"]):
                        analytical = partial_a_by_eta[dependent_dx]
                        algo = dacc_deta[dependent_dx]
                        _write_line(mode, dependent, independent, analytical, algo)

            elif mode == "SphericalHarmonicsGravity":
                partial_a_by_C = self.force_models[mode]._compute_partial_by_C(
                    position.values, epoch
                )

                for n in range(0, self.sph_harm_order + 1):
                    for m in range(0, self.sph_harm_order + 1):
                        independent = "C" + str(n) + "," + str(m)

                        for dependent_dx, dependent in enumerate(["x", "y", "z"]):
                            analytical = partial_a_by_C[dependent_dx][n][m]
                            algo = dacc_dC[dependent_dx][n][m]
                            _write_line(mode, dependent, independent, analytical, algo)

                partial_a_by_S = self.force_models[mode]._compute_partial_by_S(
                    position.values, epoch
                )

                for n in range(0, self.sph_harm_order + 1):
                    for m in range(0, self.sph_harm_order + 1):
                        independent = "S" + str(n) + "," + str(m)

                        for dependent_dx, dependent in enumerate(["x", "y", "z"]):
                            analytical = partial_a_by_S[dependent_dx][n][m]
                            algo = dacc_dS[dependent_dx][n][m]
                            _write_line(mode, dependent, independent, analytical, algo)

            elif mode == "FiniteBurn":
                partial_a_by_thrust = self.force_models[mode]._compute_partial_by_thrust(
                    position.values, full_state, epoch, body
                )

                for independent_dx, independent in enumerate(["thrust"]):
                    for dependent_dx, dependent in enumerate(["x", "y", "z"]):
                        analytical = partial_a_by_thrust[dependent_dx]
                        algo = dacc_dthrust[dependent_dx]
                        _write_line(mode, dependent, independent, analytical, algo)

                partial_a_by_ra_dec = self.force_models[mode]._compute_partial_by_ra_dec_bias(
                    position.values, full_state, epoch, body
                )

                for independent_dx, independent in enumerate(["ra"]):
                    for dependent_dx, dependent in enumerate(["x", "y", "z"]):
                        analytical = partial_a_by_ra_dec[dependent_dx][independent_dx]
                        algo = dacc_dra[dependent_dx]
                        _write_line(mode, dependent, independent, analytical, algo)
                
                for independent_dx, independent in enumerate(["dec"]):
                    for dependent_dx, dependent in enumerate(["x", "y", "z"]):
                        analytical = partial_a_by_ra_dec[dependent_dx][1]
                        algo = dacc_ddec[dependent_dx]
                        _write_line(mode, dependent, independent, analytical, algo)

            elif mode == "FirstOrderGaussMarkov":
                partial_a_by_accel = self.force_models[mode]._compute_partial_by_acceleration_state(
                    position.values, full_state, epoch, body
                )

                for independent_dx, independent in enumerate(["afogm_x", "afogm_y", "afogm_z"]):
                    for dependent_dx, dependent in enumerate(["x", "y", "z"]):
                        analytical = partial_a_by_accel[dependent_dx][independent_dx]
                        algo = dacc_dGMaccel[dependent_dx][independent_dx]
                        _write_line(mode, dependent, independent, analytical, algo)

                partial_a_by_accel = self.force_models[mode]._compute_partial_of_acceleration_state_by_acceleration_state(
                    position.values, full_state, epoch, body
                )

                for independent_dx, independent in enumerate(["afogm_x", "afogm_y", "afogm_z"]):
                    for dependent_dx, dependent in enumerate(["afogm_x", "afogm_y", "afogm_z"]):
                        analytical = partial_a_by_accel[dependent_dx][independent_dx]
                        algo = dacc_dGMdriftaccel[dependent_dx][independent_dx]
                        _write_line(mode, dependent, independent, analytical, algo)

                partial_a_by_beta = self.force_models[mode]._compute_partial_of_acceleration_state_by_beta(
                    position.values, full_state, epoch, body
                )

                for independent_dx, independent in enumerate(["beta_afogm_x", "beta_afogm_y", "beta_afogm_z"]):
                    for dependent_dx, dependent in enumerate(["afogm_x", "afogm_y", "afogm_z"]):
                        if dependent == "afogm_x":
                            analytical = partial_a_by_beta[dependent_dx][0]
                            algo = dacc_dbeta[dependent_dx][0]
                            _write_line(mode, dependent, independent, analytical, algo)
                        elif dependent == "afogm_y":
                            analytical = partial_a_by_beta[dependent_dx][1]
                            algo = dacc_dbeta[dependent_dx][1]
                            _write_line(mode, dependent, independent, analytical, algo)
                        elif dependent == "afogm_z":
                            analytical = partial_a_by_beta[dependent_dx][2]
                            algo = dacc_dbeta[dependent_dx][2]
                            _write_line(mode, dependent, independent, analytical, algo)

            elif mode == "PiecewiseFirstOrderGaussMarkov":
                partial_a_by_accel = self.force_models[mode]._compute_partial_by_acceleration_state(
                    position.values, full_state, epoch, body
                )

                for independent_dx, independent in enumerate(["apfogm_x", "apfogm_y", "apfogm_z"]):
                    for dependent_dx, dependent in enumerate(["x", "y", "z"]):
                        analytical = partial_a_by_accel[dependent_dx][independent_dx]
                        algo = dacc_dPGMaccel[dependent_dx][independent_dx]
                        _write_line(mode, dependent, independent, analytical, algo)

                partial_a_by_accel = self.force_models[mode]._compute_partial_of_acceleration_state_by_acceleration_state(
                    position.values, full_state, epoch, body
                )

                for independent_dx, independent in enumerate(["apfogm_x", "apfogm_y", "apfogm_z"]):
                    for dependent_dx, dependent in enumerate(["apfogm_x", "apfogm_y", "apfogm_z"]):
                        analytical = partial_a_by_accel[dependent_dx][independent_dx]
                        algo = dacc_dPGMdriftaccel[dependent_dx][independent_dx]
                        _write_line(mode, dependent, independent, analytical, algo)

                if self.force_models[mode]._get_active_batch_index(epoch) < self.force_models[mode].n_batches:
                    pass

                partial_a_by_beta = self.force_models[mode]._compute_partial_of_acceleration_state_by_beta(
                    position.values, full_state, epoch, body
                )

                for independent_dx, independent in enumerate(["beta_apfogm_x", "beta_apfogm_y", "beta_apfogm_z"]):
                    for dependent_dx, dependent in enumerate(["apfogm_x", "apfogm_y", "apfogm_z"]):
                        if dependent == "apfogm_x":
                            analytical = partial_a_by_beta[dependent_dx][0]
                            algo = dacc_dPbeta[dependent_dx][0]
                            _write_line(mode, dependent, independent, analytical, algo)
                        elif dependent == "apfogm_y":
                            analytical = partial_a_by_beta[dependent_dx][1]
                            algo = dacc_dPbeta[dependent_dx][1]
                            _write_line(mode, dependent, independent, analytical, algo)
                        elif dependent == "apfogm_z":
                            analytical = partial_a_by_beta[dependent_dx][2]
                            algo = dacc_dPbeta[dependent_dx][2]
                            _write_line(mode, dependent, independent, analytical, algo)
            else:
                continue
