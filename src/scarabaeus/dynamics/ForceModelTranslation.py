# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    Units,
    ArrayWUnits,
    CustomDynamicModel,
    ForceModel,
    CelestialBody,
    Spacecraft,
    PointMassGravity,
    CannonballSRP,
    SphericalHarmonicsGravity,
    PolyhedralGravity,
    nPlateSRP,
    ThreeBodyGravity,
    FiniteBurn,
    ImpulsiveBurn,
    SmoothedImpulsiveBurn,
    YarkovskyEffect,
    ThermalRadiation,
    AntennaRadiationPressure,
    FirstOrderGaussMarkov,
    PiecewiseFirstOrderGaussMarkov,
    FourierAcceleration,
    PiecewisePolynomialAcceleration,
    Frame,
    Maneuver,
)

import numpy as np
import re
import os
import warnings
from autograd import jacobian
from autograd.numpy.numpy_boxes import ArrayBox
import copy
from pathlib import Path

from scarabaeus.timeAndFrame.EpochArray import EpochArray
from scarabaeus.utils.Serialization import (
    owner_to_dict,
    owner_from_dict,
    scalar_to_dict,
    scalar_from_dict,
    dump as _dump_serialized,
    load as _load_serialized,
)

kg, km, sec, N = Units.get_units(["kg", "km", "sec", "N"])
J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()

# Recurring partial-derivative units, precomputed once. Building these via
# ``sec**-2`` etc. inside the per-call partial path is a measurable cost during
# propagation (Units.__pow__/__mul__ run thousands of times), so cache them here.
_U_S2_INV = sec**-2  # acceleration w.r.t. position / dv-rate / thrust-scale
_U_S_INV = sec**-1  # velocity w.r.t. velocity / Gauss-Markov decay
_U_KM_S2 = km * sec**-2  # acceleration w.r.t. SH coefficients / pole / beta
_U_KM2_INV = km**-2  # acceleration w.r.t. GM
_U_KM_S3 = km * sec**-3  # acceleration w.r.t. velocity-like terms

# d(velocity)/d(velocity) is a constant unitless 3x3 identity; build it once
# rather than per RHS evaluation during propagation.
_EYE3_AWU = ArrayWUnits(np.eye(3), None)

# Compiled once: parses a partial key "lhs@id_by_rhs@id" (see _collect_partials).
_PARTIAL_KEY_RE = re.compile(
    r"^(?P<lhs>\w+)@(?P<lhs_id>-?\d+)_by_(?P<rhs>\w+)@(?P<rhs_id>-?\d+)$"
)
_TRAILING_DIGITS_RE = re.compile(r"\d+$")

#: Scale factors of the spacecraft-emission models and the mode each scales.
_EMISSION_SCALE_FACTORS = {
    "eta_thermal": "ThermalRadiation",
    "eta_antenna": "AntennaRadiationPressure",
}

#: Mode-list key for spherical harmonics. A force model may carry several
#: fields at once — Jupiter plus its moons, say — so the first is registered
#: under this bare name (what a single-field model has always been called) and
#: any further one under ``SphericalHarmonicsGravity[<BODY>]``. Keeping them as
#: separate entries means the accumulation and verification loops, which are
#: generic over the mode list, pick every field up without knowing there is
#: more than one.
_SPH_MODE = "SphericalHarmonicsGravity"


def _sph_mode_key(index: int, body_name: str) -> str:
    """Build the mode-list key of one spherical-harmonics field.

    Parameters
    ----------
    index : int
        0-based position of the field among the model's harmonics fields.
    body_name : str
        Name of the body carrying the field.

    Returns
    -------
    str
        ``"SphericalHarmonicsGravity"`` for the first field, otherwise
        ``"SphericalHarmonicsGravity[<BODY>]"`` with the body name upper-cased.
    """
    return _SPH_MODE if index == 0 else f"{_SPH_MODE}[{str(body_name).upper()}]"


def _is_sph_mode(mode: str) -> bool:
    """Check whether a mode-list entry names a spherical-harmonics field.

    Parameters
    ----------
    mode : str
        Mode-list entry.

    Returns
    -------
    bool
        ``True`` for ``"SphericalHarmonicsGravity"`` and for any
        ``"SphericalHarmonicsGravity[<BODY>]"`` key.
    """
    return mode == _SPH_MODE or mode.startswith(f"{_SPH_MODE}[")


def _beta_to_dict(beta):
    """
    Convert a Gauss-Markov inverse correlation time to a JSON-friendly value.

    ``beta`` is stored as whatever the caller passed (a scalar or a 3-vector,
    array or list), but a raw ``ndarray`` would come back as a list from JSON
    and as an array from pickle, so a saved dict would not compare equal to the
    one it was written from. Everything array-like is flattened to a list here,
    which is also what :class:`FirstOrderGaussMarkov` re-coerces on the way in.

    Parameters
    ----------
    beta : float or array_like or None
        Inverse correlation time, in 1/s.

    Returns
    -------
    float or list of float or None
        ``None`` unchanged, a scalar as a ``float``, anything else as a flat
        list of floats.
    """
    if beta is None:
        return None
    if np.isscalar(beta):
        return float(beta)
    return np.asarray(beta, dtype=float).ravel().tolist()


class ForceModelTranslation(ForceModel):
    """High-level translational dynamics force model.

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
        The initial epoch for the force model. Sub-models that are bound to a
        propagation leg (ephemeris caching windows, impulsive burns) use the
        leg's own start epoch instead, supplied by
        :class:`~scarabaeus.Propagator` when it binds the model. This value is
        kept as configured and is what anchors the batch grid of
        :class:`~scarabaeus.PiecewiseFirstOrderGaussMarkov` and
        :class:`~scarabaeus.PiecewisePolynomialAcceleration`, which is required
        for those models and must be the same epoch for every leg of a
        :class:`~scarabaeus.MissionSequence`.
    third_bodies : list of :class:`~scarabaeus.body.CelestialBody`, optional
        List of bodies to be used in computing third body accelerations.
        SPICE-defined body positions are queried by EphemerisCache.
    cannonball_SRP : bool, optional
        Enable a classical cannonball solar radiation pressure model.
        Ignored if ``nplate_SRP=True``.
    nplate_SRP : bool, optional
        Enable a multi-facet (n-plate) SRP model. Takes precedence over
        ``cannonball_SRP``.
    polyhedral_gravity : bool, optional
        Enable constant-density polyhedral (shape-model) gravity for the
        propagation origin. Replaces the default point mass, since a shape
        model already carries the body's whole field.
    polyhedron_shape : str or Path or object or np.ndarray, optional
        The body's shape: a path to a Wavefront ``.obj`` or an ASCII
        vertex/facet table, any object exposing ``vertices`` and ``faces``, a
        ``(vertices, faces)`` pair, or a vertex array paired with
        ``polyhedron_faces``.
    polyhedron_faces : np.ndarray, optional
        Facet indices, needed only when ``polyhedron_shape`` is a bare vertex
        array.
    polyhedron_density : float or ArrayWUnits, optional
        Bulk density in kg/km^3. Supply this or ``polyhedron_gm``.
    polyhedron_gm : float or ArrayWUnits, optional
        Gravitational parameter in km^3/s^2; the density follows from the mesh
        volume.
    polyhedron_body_fixed_frame : str or Frame, optional
        Frame the mesh is expressed in, rotated through SPICE each evaluation.
    polyhedron_scale : float, optional
        Multiplier taking the shape file's length unit to km.
    sph_harm : bool, optional
        Enable a spherical harmonics gravity field expansion. On its own it
        only turns harmonics on; the fields may be named either by the
        singular keywords below or by ``sph_harm_bodies``.
    sph_harm_body : CelestialBody, optional
        The celestial body for which the spherical harmonics are defined.
        Optional when the fields are listed in ``sph_harm_bodies`` instead.
    sph_harm_order : int, optional
        The maximum harmonic order (m) to include.
    sph_harm_cs_file : str, optional
        Path to the coefficient file containing Cnm/Snm terms.
    sph_harm_norm_flag : bool, optional
        Whether the coefficients are normalized (True) or unnormalized (False).
    sph_degree_max : int, optional
        The maximum spherical harmonic degree (n) to include. Default is 100.
    sph_harm_bodies : list of dict, optional
        Harmonic fields, for flying somewhere several bodies have one —
        Jupiter and its moons, say. Each entry is a dict with ``body``,
        ``order`` and ``cs_file``, optionally ``norm_flag`` (default True) and
        ``degree_max`` (default 100). List every field here, central body
        included, and leave the singular keywords out::

            sph_harm=True,
            sph_harm_bodies=[
                {"body": jupiter, "order": 8, "cs_file": jup_cs},
                {"body": io,      "order": 2, "cs_file": io_cs},
            ]

        A field named by the singular keywords is simply the first field, so
        the two spellings compose and an existing single-field model can gain a
        second body without its first being rewritten. One field per body: two
        fields for the same body would count that body's gravity twice and are
        refused. Each field's Cnm/Snm, GM and pole partials are keyed by its
        own body, so several fields can be estimated at once.
    finite_burn : bool, optional
        Enable continuous-thrust (finite burn) modeling.
    maneuver : Maneuver, optional
        Maneuver definition object for finite-burn or impulsive dynamics.
    yarkovsky_effect : bool, optional
        Enable Yarkovsky effect modeling.
    thermal_radiation : bool, optional
        Enable the recoil of the heat the spacecraft radiates, scaled by the
        estimable ``eta_thermal``. Requires ``thermal_model``.
    thermal_model : dict or str, optional
        Temperature tables and radiating surfaces, or the path of a JSON file
        holding them; see :class:`~scarabaeus.ThermalRadiation`.
    antenna_radiation_pressure : bool, optional
        Enable the recoil of the power the spacecraft transmits (radio antennas,
        LIDAR), scaled by the estimable ``eta_antenna``. Requires
        ``antenna_emitters``.
    antenna_emitters : list of dict, optional
        The emitters; see :class:`~scarabaeus.AntennaRadiationPressure`.
    piecewise_polynomial : bool, optional
        Enable a deterministic empirical acceleration that is a polynomial in
        time over each batch of a fixed grid anchored at ``t0``; see
        :class:`~scarabaeus.PiecewisePolynomialAcceleration`. Its coefficients
        are estimated through a static ``a_ppoly`` state entry. May be combined
        with the Gauss-Markov and Fourier models.
    ppoly_order : int, optional
        Polynomial order within a batch, required when ``piecewise_polynomial``
        is on: ``0`` for a zero-order hold (one constant acceleration per
        batch, ``3 * ppoly_n_batches`` coefficients) or ``1`` for a first-order
        hold (linear per batch and continuous across batches,
        ``3 * (ppoly_n_batches + 1)`` node values).
    ppoly_batch_length : float, optional
        Duration of each batch, in seconds.
    ppoly_n_batches : int, optional
        Number of batches.
    ppoly_coefficients : np.ndarray, optional
        Coefficients applied when the state carries no ``a_ppoly`` entry, to use
        the model as a known force without estimating it.
    fogm_frame, pfogm_frame, ppoly_frame : str, optional
        Frame the components of the first-order Gauss-Markov, piecewise
        Gauss-Markov and piecewise polynomial accelerations are expressed in,
        as ``fourier_frame`` is for the Fourier one: ``"inertial"`` (the
        propagation frame, the default), ``"RTN"`` (radial / along-track /
        cross-track, the RTN triad) or ``"body_fixed"``.
    fogm_body_fixed_frame, pfogm_body_fixed_frame, ppoly_body_fixed_frame : str or Frame, optional
        Frame the components are expressed in, for a ``"body_fixed"`` frame
        (for example ``'IAU_BENNU'``).
    fogm_relative_body, pfogm_relative_body, ppoly_relative_body : str or Body, optional
        Body the RTN triad is built about, when that is not the propagation
        origin.
    custom_models : list of CustomDynamicModel, optional
        User-defined force models built in a script. Their accelerations are
        summed with the built-in ones, and the
        :class:`~scarabaeus.CustomParameter` objects they declare are
        propagated, differentiated, and estimated like any built-in solve-for
        parameter — so a new force needs no change to the library source.

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
    scarabaeus.ThermalRadiation : Thermal re-radiation recoil.
    scarabaeus.AntennaRadiationPressure : Antenna and LIDAR recoil.
    scarabaeus.FiniteBurn : Continuous finite-duration thrust model.
    scarabaeus.ImpulsiveBurn : Instantaneous delta-v model.
    scarabaeus.PiecewisePolynomialAcceleration : Piecewise-in-time empirical acceleration.
    scarabaeus.CustomDynamicModel : User-defined force model added from a script.

    Examples
    --------
    Create a force model with third body perturbations.

    >>> import scarabaeus as scb

    >>> sc = scb.Spacecraft('EXAMPLE SC', -123)     # example spacecraft

    >>> sun, venus, moon, mars = scb.CelestialBody.from_constants(['sun', 'venus', 'moon', 'mars'])
    >>> third_bodies = [sun, venus, moon, mars.barycenter()]
    >>> fm = scb.ForceModelTranslation(primary_body = sc, third_bodies = third_bodies)

    Add a user-defined force with its own estimated parameter, without touching
    the library source.

    >>> k = scb.CustomParameter("thermal_k", default=1.0)
    >>> def thermal(position, params):
    ...     r = np.linalg.norm(position)
    ...     return params["thermal_k"][0] * 1e-12 * position / r
    >>> fm = scb.ForceModelTranslation(
    ...     primary_body = sc,
    ...     custom_models = [scb.CustomDynamicModel("Thermal", thermal, parameters=[k])],
    ... )
    """

    # ------------------#
    # region   Errors  #
    # ------------------#
    # Custom exceptions: named on the model and raised on the
    # line that detects the failure, so tracebacks point at the real cause
    # rather than a generic Exception/RuntimeError.
    class UnknownThirdBodyError(Exception):
        """A third body was named but could not be resolved.

        Raised by :meth:`_coerce_third_bodies` when an entry of ``third_bodies``
        is a name that Scarabaeus' constants library does not define, or a type
        that is neither a name nor a
        :class:`~scarabaeus.CelestialBody`.
        """

    # endregion Errors #
    # ------------------#

    @staticmethod
    def _coerce_third_bodies(third_bodies: list | None) -> list | None:
        """
        Resolve third-body names to :class:`~scarabaeus.CelestialBody` objects.

        ``third_bodies`` accepts either objects or plain names such as
        ``"MERCURY"``. Names are resolved through
        :meth:`~scarabaeus.CelestialBody.from_constants`, i.e. the same
        constants library the name-based API used before third bodies became
        objects, so a name resolves to exactly the body (and GM) it always did.

        Parameters
        ----------
        third_bodies : list or None
            Mixed list of :class:`~scarabaeus.CelestialBody` and/or names.

        Returns
        -------
        list or None
            The input with every name replaced by its ``CelestialBody``.
            ``None`` and empty lists pass through untouched.

        Raises
        ------
        UnknownThirdBodyError
            If a name is not in the constants library, or an entry is neither a
            name nor a ``CelestialBody``.

        Notes
        -----
        A plain name resolves to the body centre (e.g. ``"MERCURY"`` gives 199). A
        name ending in ``BARYCENTER`` — with a space, an underscore or nothing
        in between — resolves to that system's barycentre instead (e.g.
        ``"MARS BARYCENTER"``, ``"MARS_BARYCENTER"`` and ``"MARSBARYCENTER"``
        all give 4), which is what the name-based API did before third bodies
        became objects. Use barycentres when the loaded ephemerides do not cover
        the body centres.
        """
        if not third_bodies:
            return third_bodies

        coerced = []
        for body in third_bodies:
            if isinstance(body, CelestialBody):
                coerced.append(body)
            elif isinstance(body, str):
                coerced.append(ForceModelTranslation._body_from_name(body))
            else:
                raise ForceModelTranslation.UnknownThirdBodyError(
                    f"Third body entry {body!r} is a {type(body).__name__}; "
                    "third_bodies takes CelestialBody objects or their names."
                )
        return coerced

    @staticmethod
    def _body_from_name(name: str) -> CelestialBody:
        """
        Resolve a third-body name to a :class:`~scarabaeus.CelestialBody`.

        Accepts the constants-library names, and additionally any of the
        ``BARYCENTER`` spellings, which map to the system barycentre.

        Parameters
        ----------
        name : str
            Body name, e.g. ``"VENUS"``, ``"MARS BARYCENTER"``.

        Returns
        -------
        CelestialBody
            The resolved body.

        Raises
        ------
        UnknownThirdBodyError
            If the name is not in the constants library.
        """
        cleaned = name.strip().upper()

        # Barycentre spellings: "<BODY> BARYCENTER" / "<BODY>_BARYCENTER" /
        # "<BODY>BARYCENTER". Strip the suffix, resolve the body, take its
        # barycentre.
        wants_barycenter = cleaned.endswith("BARYCENTER")
        if wants_barycenter:
            cleaned = (
                cleaned.removesuffix("BARYCENTER").removesuffix("_").removesuffix(" ")
            ).strip()

        try:
            resolved = CelestialBody.from_constants(cleaned)
        except Exception as exc:
            raise ForceModelTranslation.UnknownThirdBodyError(
                f"Third body '{name}' is not defined in Scarabaeus' constants "
                f"library (resolved to '{cleaned}'), so it cannot be turned into a "
                "CelestialBody. Pass a CelestialBody built with "
                "CelestialBody.from_constants(), or use a name the constants "
                "library knows."
            ) from exc

        if wants_barycenter:
            try:
                return resolved.barycenter()
            except Exception as exc:
                raise ForceModelTranslation.UnknownThirdBodyError(
                    f"Third body '{name}' resolved to '{cleaned}', but that body has "
                    "no barycentre defined. Drop the BARYCENTER suffix to use the "
                    "body centre."
                ) from exc
        return resolved

    class SelfGravityError(ValueError):
        """A force model was asked for a body's acceleration under its own
        gravity field.

        Raised when ``sph_harm_body`` names the same body as ``primary_body``:
        the harmonic expansion would be evaluated at zero separation, and the
        physical answer — a body exerts no net force on itself — is not what
        that configuration computes.
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
        sph_harm_bodies: list | None = None,
        # Thrust / maneuvers
        finite_burn: bool = False,
        maneuver: Maneuver | None = None,
        # Smoothed impulsive maneuvers
        smoothed_maneuvers: list | None = None,
        dv_smoothing_factor: float = 1.0,
        # Yarkovsky Effect
        yarkovsky_effect: bool = False,
        # Spacecraft emission (thermal re-radiation, antenna/LIDAR recoil)
        thermal_radiation: bool = False,
        thermal_model=None,
        antenna_radiation_pressure: bool = False,
        antenna_emitters: list | None = None,
        # First-Order Gauss-Markov
        first_order_gauss_markov: bool = False,
        fogm_beta: np.ndarray | float | None = None,
        fogm_frame: str = "inertial",
        fogm_body_fixed_frame=None,
        fogm_relative_body=None,
        # Piecewise First-Order Gauss-Markov
        piecewise_first_order_gauss_markov: bool = False,
        pfogm_beta: np.ndarray | float | None = None,
        pfogm_batch_length: float | None = None,
        pfogm_n_batches: int | None = None,
        pfogm_frame: str = "inertial",
        pfogm_body_fixed_frame=None,
        pfogm_relative_body=None,
        # Fourier (deterministic empirical) acceleration
        fourier_acceleration: bool = False,
        fourier_degree: int = 0,
        fourier_frame: str = "inertial",
        fourier_angle: str = "time",
        fourier_period: float | None = None,
        fourier_body_fixed_frame=None,
        fourier_relative_body=None,
        fourier_coefficients: np.ndarray | None = None,
        # Piecewise polynomial (deterministic empirical) acceleration
        piecewise_polynomial: bool = False,
        ppoly_order: int | None = None,
        ppoly_batch_length: float | None = None,
        ppoly_n_batches: int | None = None,
        ppoly_coefficients: np.ndarray | None = None,
        ppoly_frame: str = "inertial",
        ppoly_body_fixed_frame=None,
        ppoly_relative_body=None,
        # Polyhedral (shape-model) gravity
        polyhedral_gravity: bool = False,
        polyhedron_shape=None,
        polyhedron_faces: np.ndarray | None = None,
        polyhedron_density=None,
        polyhedron_gm=None,
        polyhedron_body_fixed_frame=None,
        polyhedron_scale: float = 1.0,
        # User-defined dynamics
        custom_models: list | None = None,
    ) -> None:
        super().__init__()

        # Third bodies may be given as CelestialBody objects or as plain names.
        # Names are resolved here so everything downstream sees objects.
        third_bodies = self._coerce_third_bodies(third_bodies)

        dynamics_mode_list: list[str] = []

        # Spherical Harmonics. The single-field keywords and the multi-field
        # `sph_harm_bodies` list are two spellings of the same thing, so both
        # are folded into one list of specs and nothing downstream has to know
        # which spelling was used.
        self._sph_harm_specs = self._build_sph_specs(
            sph_harm,
            sph_harm_body,
            sph_harm_order,
            sph_harm_cs_file,
            sph_harm_norm_flag,
            sph_degree_max,
            sph_harm_bodies,
        )
        for index, spec in enumerate(self._sph_harm_specs):
            dynamics_mode_list.append(_sph_mode_key(index, self._sph_spec_name(spec)))

        # Polyhedral (shape-model) gravity
        if polyhedral_gravity:
            if polyhedron_shape is None:
                raise ValueError(
                    "Polyhedral gravity enabled but missing: polyhedron_shape. "
                    "Pass a shape file ('.obj' or a vertex/facet table), a mesh "
                    "object, a (vertices, faces) pair, or a vertex array with "
                    "polyhedron_faces."
                )
            dynamics_mode_list.append("PolyhedralGravity")

        # Third-body gravity
        self._validate_third_bodies(third_bodies)
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

        # Smoothed impulsive maneuvers (tanh pulses, no leg splitting)
        if smoothed_maneuvers:
            dynamics_mode_list.append("SmoothedImpulsiveBurn")

        # Yarkovsky Effect
        if yarkovsky_effect:
            dynamics_mode_list.append("YarkovskyEffect")

        # Spacecraft emission. Validated here rather than at binding, so a
        # malformed table or emitter is reported where it was configured.
        self._thermal_model = None
        if thermal_radiation:
            if thermal_model is None:
                raise self.MissingEmissionModelError(
                    "thermal_radiation=True needs thermal_model: the temperature "
                    "tables and radiating surfaces, as a dict or a JSON file."
                )
            self._thermal_model = ThermalRadiation.model_from_dict(thermal_model)
            dynamics_mode_list.append("ThermalRadiation")

        self._antenna_emitters = None
        if antenna_radiation_pressure:
            if antenna_emitters is None:
                raise self.MissingEmissionModelError(
                    "antenna_radiation_pressure=True needs antenna_emitters: the "
                    "list of emitters with their power, beamwidth and boresight."
                )
            self._antenna_emitters = AntennaRadiationPressure.emitters_from_dict(
                antenna_emitters
            )
            dynamics_mode_list.append("AntennaRadiationPressure")

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

        # Fourier empirical acceleration. Independent of the Gauss-Markov
        # models: those describe a stochastic process, this one a deterministic
        # series with estimable coefficients, and an arc may carry both.
        if fourier_acceleration:
            dynamics_mode_list.append("FourierAcceleration")

        # Piecewise polynomial empirical acceleration. The order sizes the
        # coefficient state, so it has no default and is checked here, before
        # a state vector is built against it.
        if piecewise_polynomial:
            PiecewisePolynomialAcceleration._validate_order(ppoly_order)
            PiecewisePolynomialAcceleration._validate_grid(
                ppoly_batch_length, ppoly_n_batches
            )
            dynamics_mode_list.append("PiecewisePolynomialAcceleration")

        # User-defined dynamics
        self._custom_models = self._validate_custom_models(
            custom_models, dynamics_mode_list
        )
        dynamics_mode_list.extend(self._custom_models)

        # Always ensure baseline point-mass gravity unless a full-field model
        # already represents the central body. Whether it does cannot be
        # decided here — the origin arrives with the state vector — so this
        # suppression is provisional and _initialize_force_models revisits it
        # once the origin is known.
        if (
            not any(_is_sph_mode(mode) for mode in dynamics_mode_list)
            and "PolyhedralGravity" not in dynamics_mode_list
        ):
            dynamics_mode_list.append("PointMassGravity")

        # Store config
        self._dynamics_mode_list = dynamics_mode_list

        # already coerced to CelestialBody objects at the top of __init__
        self._third_bodies = third_bodies

        # self._third_bodies = third_bodies
        self._primary_body = primary_body

        # Cache of parsed partial keys: name -> (lhs_base, lhs_id, rhs, rhs_base,
        # rhs_id) or False when the name does not match. The parse depends only on
        # the key string, so it is computed once per unique name rather than on
        # every RHS evaluation during propagation. See :meth:`_collect_partials`.
        self._partial_key_cache = {}

        # Scalar mirrors of the first field, so the long-standing single-field
        # attributes and properties keep meaning what they always did.
        first = self._sph_harm_specs[0] if self._sph_harm_specs else {}
        self._sph_harm_order = first.get("order", 0)
        self._sph_harm_cs_file = first.get("cs_file")
        self._sph_harm_body = first.get("body")
        self._sph_harm_norm_flag = first.get("norm_flag", False)
        self._sph_harm_degree_max = first.get("degree_max", 0)

        self._polyhedron_shape = polyhedron_shape
        self._polyhedron_faces = polyhedron_faces
        self._polyhedron_density = polyhedron_density
        self._polyhedron_gm = polyhedron_gm
        self._polyhedron_body_fixed_frame = polyhedron_body_fixed_frame
        self._polyhedron_scale = polyhedron_scale

        self._maneuver = maneuver
        self._smoothed_maneuvers = (
            list(smoothed_maneuvers) if smoothed_maneuvers else None
        )
        self._dv_smoothing_factor = dv_smoothing_factor
        self._first_order_gauss_markov = first_order_gauss_markov
        self._fogm_beta = fogm_beta
        self._fogm_frame = fogm_frame
        self._fogm_body_fixed_frame = fogm_body_fixed_frame
        self._fogm_relative_body = fogm_relative_body
        self._piecewise_first_order_gauss_markov = piecewise_first_order_gauss_markov
        self._pfogm_beta = pfogm_beta
        self._pfogm_batch_length = pfogm_batch_length
        self._pfogm_n_batches = pfogm_n_batches
        self._pfogm_frame = pfogm_frame
        self._pfogm_body_fixed_frame = pfogm_body_fixed_frame
        self._pfogm_relative_body = pfogm_relative_body

        self._fourier_acceleration = fourier_acceleration
        self._fourier_degree = fourier_degree
        self._fourier_frame = fourier_frame
        self._fourier_angle = fourier_angle
        self._fourier_period = fourier_period
        self._fourier_body_fixed_frame = fourier_body_fixed_frame
        self._fourier_relative_body = fourier_relative_body
        self._fourier_coefficients = fourier_coefficients

        self._piecewise_polynomial = piecewise_polynomial
        self._ppoly_order = ppoly_order
        self._ppoly_batch_length = ppoly_batch_length
        self._ppoly_n_batches = ppoly_n_batches
        self._ppoly_coefficients = ppoly_coefficients
        self._ppoly_frame = ppoly_frame
        self._ppoly_body_fixed_frame = ppoly_body_fixed_frame
        self._ppoly_relative_body = ppoly_relative_body

        # The RTN triad and the argument-of-latitude phase are both built from
        # the velocity, so an empirical model using either needs the da/dv block
        # that otherwise only velocity-dependent custom models request.
        if fourier_acceleration and (
            str(fourier_frame).strip().lower() == "rtn"
            or str(fourier_angle).strip().lower()
            in ("argument_of_latitude", "solar_longitude")
        ):
            self._velocity_partial_active = True
        if any(
            enabled and str(frame).strip().lower() == "rtn"
            for enabled, frame in (
                (first_order_gauss_markov, fogm_frame),
                (piecewise_first_order_gauss_markov, pfogm_frame),
                (piecewise_polynomial, ppoly_frame),
            )
        ):
            self._velocity_partial_active = True

        self._t0 = t0

        # Epoch as configured by the caller. `initialize_force_models` rebinds
        # `_t0` to the propagation leg being set up, which is what the ephemeris
        # caches and the burn models want; this copy survives that rebind so
        # model settings anchored to a fixed epoch (the PFOGM and piecewise
        # polynomial batch grids, the Fourier phase origin) keep the epoch they
        # were configured with.
        self._t0_config = t0

    # ------------------------------------------------------------------
    # Third bodies
    # ------------------------------------------------------------------
    class ThirdBodyTypeError(TypeError):
        """Raised when ``third_bodies`` holds anything but a CelestialBody."""

    class MissingBatchEpochError(ValueError):
        """Raised when a batch-grid model (piecewise Gauss-Markov or piecewise
        polynomial) is requested without a ``t0``."""

    class MissingEmissionModelError(ValueError):
        """Raised when thermal or antenna recoil is enabled without its
        ``thermal_model`` or ``antenna_emitters``."""

    @staticmethod
    def _validate_third_bodies(third_bodies) -> None:
        """
        Require every third body to be a :class:`~scarabaeus.CelestialBody`.

        Names are deliberately **not** accepted. A body carries its own GM,
        radius and body-fixed frame, so resolving a bare string here would hide
        where those numbers came from and would silently pick the planet centre
        for names that denote a system barycenter. Building the object in the
        script keeps that choice explicit and visible.

        Parameters
        ----------
        third_bodies : list or None
            Value supplied to the ``third_bodies`` constructor argument.

        Raises
        ------
        ForceModelTranslation.ThirdBodyTypeError
            If any entry is not a ``CelestialBody``. The message names the
            offending entry and shows the call that builds it.
        """
        if third_bodies is None:
            return

        for body in third_bodies:
            if isinstance(body, CelestialBody):
                continue

            if isinstance(body, str):
                # 'MARSBARYCENTER' / 'JUPITER BARYCENTER' denote the system
                # barycenter, which the constants library does not expose
                # directly — it is reached through the planet's .barycenter().
                stripped = re.sub(r"[\s_]*BARYCENTER$", "", body.strip().upper())
                if stripped != body.strip().upper():
                    suggestion = (
                        f'CelestialBody.from_constants("{stripped}").barycenter()'
                    )
                else:
                    suggestion = f'CelestialBody.from_constants("{body}")'

                raise ForceModelTranslation.ThirdBodyTypeError(
                    f"Third body {body!r} was given as a name. 'third_bodies' "
                    "takes CelestialBody objects so that the GM, radius and "
                    f"frame in use are explicit. Pass {suggestion} instead."
                )

            raise ForceModelTranslation.ThirdBodyTypeError(
                "Entries of 'third_bodies' must be CelestialBody objects, got "
                f"{type(body).__name__}. Build them with "
                "CelestialBody.from_constants(name), and add .barycenter() for "
                "a system barycenter."
            )

    @staticmethod
    def _third_body_to_dict(body: CelestialBody) -> dict:
        """
        Serialize one third body to a JSON-friendly dict.

        Parameters
        ----------
        body : CelestialBody
            Third body to serialize.

        Returns
        -------
        dict
            ``{"name": ..., "barycenter": bool}``. ``name`` is the constants
            name of the planet, so a barycenter records its planet rather than
            its own display name.
        """
        is_barycenter = re.search(r"\bbarycenter\b", body.name, re.IGNORECASE)
        name = re.sub(r"\s*barycenter\s*$", "", body.name, flags=re.IGNORECASE)

        return {"name": name.strip(), "barycenter": bool(is_barycenter)}

    @staticmethod
    def _third_body_from_dict(entry) -> CelestialBody:
        """
        Rebuild one third body from :meth:`_third_body_to_dict`.

        Parameters
        ----------
        entry : dict or str
            Serialized body. A bare string is accepted for files written before
            the barycenter flag existed.

        Returns
        -------
        CelestialBody
            The reconstructed body.
        """
        if isinstance(entry, str):
            entry = {"name": entry, "barycenter": False}

        body = CelestialBody.from_constants(entry["name"])

        return body.barycenter() if entry.get("barycenter") else body

    # ------------------------------------------------------------------
    # User-defined dynamics
    # ------------------------------------------------------------------
    class CustomModelTypeError(TypeError):
        """Raised when ``custom_models`` holds something other than a CustomDynamicModel."""

    class DuplicateCustomModelError(ValueError):
        """Raised when two active force models share the same name."""

    class DuplicateCustomDynamicsError(ValueError):
        """Raised when two custom models define the dynamics of the same parameter."""

    def _validate_custom_models(self, custom_models, dynamics_mode_list) -> dict:
        """
        Validate the user-supplied models and index the parameters they declare.

        Builds the lookup tables consumed by :meth:`_collect_partials` and
        :meth:`_collect_parameter_dynamics`. Partial keys are lower-cased by the
        propagator, so parameters are indexed by their lower-cased name.

        Parameters
        ----------
        custom_models : CustomDynamicModel, list of CustomDynamicModel, or None
            Models passed to the constructor. A single model is accepted
            without wrapping it in a list.
        dynamics_mode_list : list of str
            Names of the built-in models already enabled, checked for collisions.

        Returns
        -------
        dict
            Mapping ``model name -> CustomDynamicModel``, in declaration order.

        Raises
        ------
        ForceModelTranslation.CustomModelTypeError
            If an entry is not a :class:`~scarabaeus.CustomDynamicModel`.
        ForceModelTranslation.DuplicateCustomModelError
            If a model name repeats or shadows a built-in model.
        ForceModelTranslation.DuplicateCustomDynamicsError
            If two models supply the equation of motion of the same parameter.
        """
        # Maps consulted on every RHS evaluation; always defined so the hot path
        # can test them without a hasattr/getattr guard. The velocity flag is
        # raised by any model that declares ``depends_on_velocity`` — custom
        # models here, built-in ones once the rest of the config is read.
        self._custom_param_owners: dict = {}
        self._custom_param_dot_owners: dict = {}
        self._velocity_partial_active = False

        if custom_models is None:
            return {}

        if isinstance(custom_models, CustomDynamicModel):
            custom_models = [custom_models]

        models: dict = {}
        for model in custom_models:
            if not isinstance(model, CustomDynamicModel):
                raise self.CustomModelTypeError(
                    "Entries of 'custom_models' must be CustomDynamicModel "
                    f"instances, got {type(model).__name__}."
                )
            if model.name in models or model.name in dynamics_mode_list:
                raise self.DuplicateCustomModelError(
                    f"Custom model name '{model.name}' is already in use by "
                    "another active force model. Give each model a unique name."
                )
            models[model.name] = model

            if model.depends_on_velocity:
                self._velocity_partial_active = True

            for parameter in model.parameters.values():
                key = parameter.name.lower()
                self._custom_param_owners.setdefault(key, []).append((model, parameter))

                if not parameter.is_dynamic:
                    continue

                dot_key = f"{key}_dot"
                if dot_key in self._custom_param_dot_owners:
                    other = self._custom_param_dot_owners[dot_key][0]
                    raise self.DuplicateCustomDynamicsError(
                        f"Custom parameter '{parameter.name}' has its dynamics "
                        f"defined by both '{other.name}' and '{model.name}'. "
                        "Only one model may integrate a given parameter."
                    )
                self._custom_param_dot_owners[dot_key] = (model, parameter)

        return models

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """
        Serialize the force-model configuration to a JSON-friendly dict.

        Captures the constructor options needed to rebuild this model.
        ``primary_body`` is intentionally omitted: it is owned by the Propagator
        and re-supplied to :meth:`from_dict` (a file written by :meth:`save`
        carries it alongside). ``t0`` is the epoch the model was configured with,
        which anchors the PFOGM batch grid and the Fourier phase; binding to a leg
        does not change it. Maneuver objects and custom models cannot be serialized
        (both hold user callables); only their presence is recorded.

        Returns
        -------
        dict
            The constructor options, readable by :meth:`from_dict`.
        """
        dml = self._dynamics_mode_list
        return {
            "t0": self._t0_to_dict(self._t0_config),
            # Name plus a barycenter flag: a system barycenter is not in the
            # constants library under its own name, so the flag is what lets
            # from_dict rebuild it through CelestialBody.barycenter().
            "third_bodies": [
                self._third_body_to_dict(tb) for tb in (self._third_bodies or [])
            ],
            "cannonball_SRP": "CannonballSRP" in dml,
            "nplate_SRP": "nPlateSRP" in dml,
            "sph_harm": any(_is_sph_mode(mode) for mode in dml),
            "polyhedral_gravity": "PolyhedralGravity" in dml,
            "polyhedron": self._polyhedron_to_dict(),
            "sph_harm_body": (
                getattr(self._sph_harm_body, "name", None)
                if self._sph_harm_body is not None
                else None
            ),
            "sph_harm_order": self._sph_harm_order,
            "sph_harm_cs_file": self._sph_harm_cs_file,
            "sph_harm_norm_flag": self._sph_harm_norm_flag,
            "sph_degree_max": self._sph_harm_degree_max,
            # Every field, so a multi-field model round-trips. The scalar keys
            # above stay as the first field's, for readers that predate this.
            "sph_harm_bodies": [
                {
                    "body": self._sph_spec_name(spec),
                    "order": spec["order"],
                    "cs_file": spec["cs_file"],
                    "norm_flag": spec["norm_flag"],
                    "degree_max": spec["degree_max"],
                }
                for spec in self._sph_harm_specs
            ],
            "finite_burn": "FiniteBurn" in dml,
            "yarkovsky_effect": "YarkovskyEffect" in dml,
            "thermal_radiation": "ThermalRadiation" in dml,
            "thermal_model": (
                ThermalRadiation.model_to_dict(self._thermal_model)
                if self._thermal_model is not None
                else None
            ),
            "antenna_radiation_pressure": "AntennaRadiationPressure" in dml,
            "antenna_emitters": (
                AntennaRadiationPressure.emitters_to_dict(self._antenna_emitters)
                if self._antenna_emitters is not None
                else None
            ),
            "first_order_gauss_markov": self._first_order_gauss_markov,
            "fogm_beta": _beta_to_dict(self._fogm_beta),
            "fogm_frame": self._fogm_frame,
            "fogm_body_fixed_frame": (
                getattr(self._fogm_body_fixed_frame, "name", self._fogm_body_fixed_frame)
                if self._fogm_body_fixed_frame is not None
                else None
            ),
            "fogm_relative_body": (
                getattr(self._fogm_relative_body, "name", self._fogm_relative_body)
                if self._fogm_relative_body is not None
                else None
            ),
            "piecewise_first_order_gauss_markov": self._piecewise_first_order_gauss_markov,
            "pfogm_beta": _beta_to_dict(self._pfogm_beta),
            "pfogm_batch_length": self._pfogm_batch_length,
            "pfogm_n_batches": self._pfogm_n_batches,
            "pfogm_frame": self._pfogm_frame,
            "pfogm_body_fixed_frame": (
                getattr(self._pfogm_body_fixed_frame, "name", self._pfogm_body_fixed_frame)
                if self._pfogm_body_fixed_frame is not None
                else None
            ),
            "pfogm_relative_body": (
                getattr(self._pfogm_relative_body, "name", self._pfogm_relative_body)
                if self._pfogm_relative_body is not None
                else None
            ),
            "fourier_acceleration": self._fourier_acceleration,
            "fourier_degree": self._fourier_degree,
            "fourier_frame": self._fourier_frame,
            "fourier_angle": self._fourier_angle,
            "fourier_period": self._fourier_period,
            "fourier_body_fixed_frame": (
                getattr(
                    self._fourier_body_fixed_frame,
                    "name",
                    self._fourier_body_fixed_frame,
                )
                if self._fourier_body_fixed_frame is not None
                else None
            ),
            "fourier_relative_body": (
                getattr(
                    self._fourier_relative_body,
                    "name",
                    self._fourier_relative_body,
                )
                if self._fourier_relative_body is not None
                else None
            ),
            "fourier_coefficients": (
                np.asarray(self._fourier_coefficients, dtype=float).ravel().tolist()
                if self._fourier_coefficients is not None
                else None
            ),
            "piecewise_polynomial": self._piecewise_polynomial,
            "ppoly_order": self._ppoly_order,
            "ppoly_batch_length": self._ppoly_batch_length,
            "ppoly_n_batches": self._ppoly_n_batches,
            "ppoly_coefficients": (
                np.asarray(self._ppoly_coefficients, dtype=float).ravel().tolist()
                if self._ppoly_coefficients is not None
                else None
            ),
            "ppoly_frame": self._ppoly_frame,
            "ppoly_body_fixed_frame": (
                getattr(self._ppoly_body_fixed_frame, "name", self._ppoly_body_fixed_frame)
                if self._ppoly_body_fixed_frame is not None
                else None
            ),
            "ppoly_relative_body": (
                getattr(self._ppoly_relative_body, "name", self._ppoly_relative_body)
                if self._ppoly_relative_body is not None
                else None
            ),
            "has_maneuver": self._maneuver is not None,
            "n_smoothed_maneuvers": len(self._smoothed_maneuvers or []),
            "dv_smoothing_factor": self._dv_smoothing_factor,
            "custom_models": list(self._custom_models),
        }

    @classmethod
    def from_dict(cls, data: dict, primary_body, t0=None) -> "ForceModelTranslation":
        """
        Reconstruct a ForceModelTranslation from :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Serialized force-model config.
        primary_body : Spacecraft
            The body this model acts on (owned by the Propagator).
        t0 : EpochArray or float, optional
            Configured epoch. Defaults to ``None``, which uses the one recorded in
            ``data``.

        Returns
        -------
        ForceModelTranslation
            The rebuilt model, not yet initialized (see
            :meth:`initialize_force_models`).

        Warns
        -----
        UserWarning
            For each part of the configuration that cannot be rebuilt from the
            file and is left out: a finite-burn maneuver, smoothed impulsive
            maneuvers, custom models, or a polyhedral gravity model whose shape
            was not recorded.
        """
        if data.get("has_maneuver"):
            warnings.warn(
                "ForceModelTranslation.from_dict cannot reconstruct a Maneuver "
                "object; the maneuver was omitted.",
                UserWarning,
            )

        if data.get("n_smoothed_maneuvers"):
            warnings.warn(
                "ForceModelTranslation.from_dict cannot reconstruct the "
                f"{data['n_smoothed_maneuvers']} ImpulsiveBurn object(s) behind "
                "the smoothed maneuvers; rebuild them in the script and pass "
                "smoothed_maneuvers= again.",
                UserWarning,
            )

        if data.get("custom_models"):
            warnings.warn(
                "ForceModelTranslation.from_dict cannot reconstruct the custom "
                "models "
                f"{', '.join(data['custom_models'])} (they hold user callables); "
                "rebuild them in the script and pass custom_models= again.",
                UserWarning,
            )

        # Files written before the shape was recorded carry only the flag.
        if data.get("polyhedral_gravity") and not data.get("polyhedron"):
            warnings.warn(
                "ForceModelTranslation.from_dict cannot reconstruct the "
                "polyhedral gravity model: this file records no shape. Pass "
                "polyhedral_gravity=True with polyhedron_shape= again.",
                UserWarning,
            )

        if t0 is None:
            t0 = cls._t0_from_dict(data.get("t0"))

        sph_harm = data.get("sph_harm", False)
        sph_body = None
        if sph_harm and data.get("sph_harm_body"):
            from scarabaeus.body.CelestialBody import CelestialBody

            sph_body = CelestialBody.from_constants(data["sph_harm_body"])

        return cls(
            primary_body=primary_body,
            t0=t0,
            third_bodies=(
                [cls._third_body_from_dict(tb) for tb in data["third_bodies"]]
                if data.get("third_bodies")
                else None
            ),
            cannonball_SRP=data.get("cannonball_SRP", False),
            nplate_SRP=data.get("nplate_SRP", False),
            sph_harm=sph_harm,
            sph_harm_body=sph_body,
            sph_harm_order=(data.get("sph_harm_order") or None),
            sph_harm_cs_file=data.get("sph_harm_cs_file"),
            sph_harm_norm_flag=data.get("sph_harm_norm_flag"),
            # The first field is already covered by the scalar keys above, so
            # only the rest are replayed through the list form.
            sph_harm_bodies=[
                {**entry, "body": CelestialBody.from_constants(entry["body"])}
                for entry in (data.get("sph_harm_bodies") or [])[1:]
            ],
            sph_degree_max=(data.get("sph_degree_max") or 100),
            finite_burn=data.get("finite_burn", False),
            yarkovsky_effect=data.get("yarkovsky_effect", False),
            thermal_radiation=data.get("thermal_radiation", False),
            thermal_model=data.get("thermal_model"),
            antenna_radiation_pressure=data.get("antenna_radiation_pressure", False),
            antenna_emitters=data.get("antenna_emitters"),
            first_order_gauss_markov=data.get("first_order_gauss_markov", False),
            fogm_beta=data.get("fogm_beta"),
            fogm_frame=data.get("fogm_frame", "inertial"),
            fogm_body_fixed_frame=data.get("fogm_body_fixed_frame"),
            fogm_relative_body=data.get("fogm_relative_body"),
            piecewise_first_order_gauss_markov=data.get(
                "piecewise_first_order_gauss_markov", False
            ),
            pfogm_beta=data.get("pfogm_beta"),
            pfogm_batch_length=data.get("pfogm_batch_length"),
            pfogm_n_batches=data.get("pfogm_n_batches"),
            pfogm_frame=data.get("pfogm_frame", "inertial"),
            pfogm_body_fixed_frame=data.get("pfogm_body_fixed_frame"),
            pfogm_relative_body=data.get("pfogm_relative_body"),
            fourier_acceleration=data.get("fourier_acceleration", False),
            fourier_degree=data.get("fourier_degree", 0),
            fourier_frame=data.get("fourier_frame", "inertial"),
            fourier_angle=data.get("fourier_angle", "time"),
            fourier_period=data.get("fourier_period"),
            fourier_body_fixed_frame=data.get("fourier_body_fixed_frame"),
            fourier_relative_body=data.get("fourier_relative_body"),
            fourier_coefficients=data.get("fourier_coefficients"),
            piecewise_polynomial=data.get("piecewise_polynomial", False),
            ppoly_order=data.get("ppoly_order"),
            ppoly_batch_length=data.get("ppoly_batch_length"),
            ppoly_n_batches=data.get("ppoly_n_batches"),
            ppoly_coefficients=data.get("ppoly_coefficients"),
            ppoly_frame=data.get("ppoly_frame", "inertial"),
            ppoly_body_fixed_frame=data.get("ppoly_body_fixed_frame"),
            ppoly_relative_body=data.get("ppoly_relative_body"),
            **cls._polyhedron_from_dict(data.get("polyhedron")),
        )

    @staticmethod
    def _t0_to_dict(t0):
        """Serialize a configured ``t0``.

        Parameters
        ----------
        t0 : EpochArray or float or None
            The configured epoch, as an ``EpochArray`` or a float of ET seconds.

        Returns
        -------
        dict or float or None
            ``{"kind": "epoch_array", "data": ...}`` for an ``EpochArray``, the ET
            as a ``float``, or ``None``.
        """
        if isinstance(t0, EpochArray):
            return {"kind": "epoch_array", "data": t0.to_dict()}
        return None if t0 is None else float(t0)

    @staticmethod
    def _t0_from_dict(d):
        """Rebuild a configured ``t0`` written by :meth:`_t0_to_dict`.

        Parameters
        ----------
        d : dict or float or None
            The serialized epoch.

        Returns
        -------
        EpochArray or float or None
            An ``EpochArray`` for a dict, otherwise ``d`` unchanged.
        """
        if isinstance(d, dict):
            return EpochArray.from_dict(d["data"])
        return d

    def _polyhedron_to_dict(self) -> dict | None:
        """
        Serialize the polyhedral-gravity inputs.

        A shape file is recorded by its path, as ``sph_harm_cs_file`` is; any
        other form (mesh object, vertex/facet arrays) is reduced to its unscaled
        vertices and facets, with ``scale`` kept separately.

        Returns
        -------
        dict or None
            The shape (``path``, or ``vertices`` and ``faces``) with ``density``,
            ``gm``, ``body_fixed_frame`` and ``scale``; ``None`` when the
            polyhedral model is off.
        """
        if "PolyhedralGravity" not in self._dynamics_mode_list:
            return None

        shape = self._polyhedron_shape
        if isinstance(shape, str) or hasattr(shape, "__fspath__"):
            shape_data = {"path": str(shape)}
        else:
            vertices, faces = PolyhedralGravity._resolve_shape(
                shape, self._polyhedron_faces, 1.0
            )
            shape_data = {"vertices": vertices.tolist(), "faces": faces.tolist()}

        return {
            **shape_data,
            "density": scalar_to_dict(self._polyhedron_density),
            "gm": scalar_to_dict(self._polyhedron_gm),
            "body_fixed_frame": PolyhedralGravity._resolve_frame_name(
                self._polyhedron_body_fixed_frame
            ),
            "scale": self._polyhedron_scale,
        }

    @staticmethod
    def _polyhedron_from_dict(d) -> dict:
        """Rebuild the polyhedral-gravity constructor keywords.

        Parameters
        ----------
        d : dict or None
            The dict written by :meth:`_polyhedron_to_dict`.

        Returns
        -------
        dict
            ``polyhedral_gravity=True`` with the ``polyhedron_*`` keywords, ready
            to unpack into the constructor; empty when ``d`` is empty or ``None``.
        """
        if not d:
            return {}
        if "path" in d:
            shape = d["path"]
        else:
            shape = (np.asarray(d["vertices"], dtype=float), np.asarray(d["faces"]))
        return {
            "polyhedral_gravity": True,
            "polyhedron_shape": shape,
            "polyhedron_density": scalar_from_dict(d.get("density")),
            "polyhedron_gm": scalar_from_dict(d.get("gm")),
            "polyhedron_body_fixed_frame": d.get("body_fixed_frame"),
            "polyhedron_scale": d.get("scale", 1.0),
        }

    class NotAForceModelFileError(ValueError):
        """Raised by :meth:`load` when the file was not written by :meth:`save`."""

    def save(self, filepath, fmt: str = None) -> "Path":
        """
        Save this force model's configuration to a JSON or pickle file.

        The file holds :meth:`to_dict` plus the primary body, so :meth:`load`
        can rebuild the model on its own, without a Propagator.

        Parameters
        ----------
        filepath : str or Path
            Destination file.
        fmt : {"json", "pickle"}, optional
            On-disk format. Defaults to ``None``, which infers it from the file
            extension: ``.pkl`` or ``.pickle`` gives pickle, anything else JSON.

        Returns
        -------
        Path
            The resolved path written.

        Raises
        ------
        UnknownSerializationFormatError
            If ``fmt`` is neither ``"json"`` nor ``"pickle"``.
        """
        return _dump_serialized(
            {
                "class": type(self).__name__,
                "primary_body": owner_to_dict(self._primary_body),
                "config": self.to_dict(),
            },
            filepath,
            fmt=fmt,
        )

    @classmethod
    def load(
        cls, filepath, primary_body=None, fmt: str = None
    ) -> "ForceModelTranslation":
        """
        Reconstruct a ForceModelTranslation from a file written by :meth:`save`.

        Parameters
        ----------
        filepath : str or Path
            File written by :meth:`save`.
        primary_body : Spacecraft, optional
            Body the model acts on. Pass the spacecraft that owns your state
            components so the model and the state share one instance. Defaults to
            ``None``, which rebuilds a new one from the file.
        fmt : {"json", "pickle"}, optional
            Input format. Defaults to ``None``, which infers it from the file
            extension.

        Returns
        -------
        ForceModelTranslation
            The rebuilt model, not yet initialized.

        Raises
        ------
        NotAForceModelFileError
            If the file was not written by this class's :meth:`save`, for example
            a Propagator file.
        UnknownSerializationFormatError
            If ``fmt`` is neither ``"json"`` nor ``"pickle"``.
        """
        data = _load_serialized(filepath, fmt=fmt)
        if not isinstance(data, dict) or data.get("class") != cls.__name__:
            raise cls.NotAForceModelFileError(
                f"{filepath} was not written by {cls.__name__}.save(). A "
                "Propagator file loads with Propagator.load(); its force models "
                "are then in propagator.force_models."
            )
        if primary_body is None:
            primary_body = owner_from_dict(data["primary_body"])
        return cls.from_dict(data["config"], primary_body=primary_body)

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
    def fogm_frame(self) -> str:
        """Frame the first-order Gauss-Markov components are expressed in."""
        return self._fogm_frame

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
    def pfogm_frame(self) -> str:
        """Frame the piecewise FOGM batch components are expressed in."""
        return self._pfogm_frame

    @property
    def fourier_degree(self) -> int:
        """Highest harmonic of the Fourier empirical acceleration."""
        return self._fourier_degree

    @property
    def fourier_frame(self) -> str:
        """Frame the Fourier coefficients are expressed in."""
        return self._fourier_frame

    @property
    def fourier_angle(self) -> str:
        """Phase angle driving the Fourier series."""
        return self._fourier_angle

    @property
    def fourier_period(self) -> float | None:
        """Repeat period of a time-based Fourier phase, in seconds."""
        return self._fourier_period

    @property
    def ppoly_order(self) -> int | None:
        """Polynomial order of the piecewise polynomial acceleration (0 or 1)."""
        return self._ppoly_order

    @property
    def ppoly_batch_length(self) -> float | None:
        """Duration of one batch of the piecewise polynomial acceleration, in seconds."""
        return self._ppoly_batch_length

    @property
    def ppoly_n_batches(self) -> int | None:
        """Number of batches of the piecewise polynomial acceleration."""
        return self._ppoly_n_batches

    @property
    def ppoly_frame(self) -> str:
        """Frame the piecewise polynomial coefficients are expressed in."""
        return self._ppoly_frame

    @property
    def custom_models(self) -> dict:
        """User-defined force models keyed by name, in declaration order."""
        return self._custom_models

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

    # ------------------------------------------------------------------
    #  Force listing
    # ------------------------------------------------------------------
    def list_forces(self) -> list[str]:
        """
        Readable labels of the forces this model applies to ``primary_body``.

        One label per entry of :attr:`dynamics_mode_list`, in that order. A
        label is the sub-model's class name followed, in parentheses, by what
        tells one configuration of that model from another: the origin of the
        point-mass term, the bodies the third-body term attracts with, the
        field body and truncation of a spherical-harmonics term, the name of a
        custom model, and so on.

        Returns
        -------
        list of str
            For example ``['ThreeBodyGravity(SUN, MOON)', 'CannonballSRP',
            'PointMassGravity(EARTH)']``.

        See Also
        --------
        scarabaeus.Propagator.list_forces : The same listing for every integrated object.
        scarabaeus.MissionSequence.list_forces : The same listing, leg by leg.

        Notes
        -----
        Before :meth:`~scarabaeus.ForceModel.initialize_force_models` has run
        (i.e. before the model is handed to a :class:`~scarabaeus.Propagator`)
        the labels describe the configuration as passed to the constructor;
        details that only the propagation fixes, such as the origin, are left
        out. Once bound, the labels come from the live sub-models and can
        differ from the configuration:

        * ``PointMassGravity`` is added at binding when spherical-harmonics
          fields are configured but none is centred on the propagation origin,
          so the origin keeps its monopole.
        * ``ThreeBodyGravity`` lists the bodies that actually attract: a third
          body that also carries a harmonics field is dropped (the field
          already includes its monopole), and every celestial body integrated
          in the state is added whether or not it was in ``third_bodies``.
        * A spherical-harmonics truncation is clamped to the maximum degree of
          its coefficient file.
        """
        live = self._force_models or {}
        return [
            self._describe_force(mode, live.get(mode))
            for mode in self._dynamics_mode_list
        ]

    def _describe_force(self, mode: str, model=None) -> str:
        """
        The :meth:`list_forces` label of one mode.

        Parameters
        ----------
        mode : str
            An entry of :attr:`dynamics_mode_list`.
        model : DynamicModel, optional
            The live sub-model for that mode, once the force models are
            initialized. Details are read from it in preference to the
            constructor settings, because it is what acts during propagation.

        Returns
        -------
        str
            ``"<ClassName>"`` or ``"<ClassName>(<details>)"``.
        """
        details = None

        if mode == "PointMassGravity":
            if model is not None:
                details = str(model.origin_name).upper()

        elif mode == "ThreeBodyGravity":
            if model is not None:
                # What attracts: the ephemeris bodies plus every integrated
                # celestial body other than the primary itself (`_prop_bodies`
                # includes it; the model skips the self term at run time).
                # Listed in the configured order, then any integrated body that
                # was not configured.
                primary_sid = getattr(self._primary_body, "spice_id", None)
                integrated = [
                    name for name, sid, _ in model._prop_bodies if sid != primary_sid
                ]
                attracting = {name for name, _, _ in model._cached_bodies}
                attracting.update(integrated)
                names = [
                    name
                    for name in (str(pm).upper() for pm in model.pnt_masses)
                    if name in attracting
                ]
                names += [name for name in integrated if name not in names]
            else:
                names = [
                    str(body.name).upper()
                    for body in (self._point_masses_less_sph_bodies() or [])
                ]
            details = ", ".join(names)

        elif _is_sph_mode(mode):
            # Specs and mode keys are built in the same order, so the key's
            # rank among the harmonics keys is the spec's index.
            spec = self._sph_harm_specs[self._sph_mode_keys().index(mode)]
            order = model.order if model is not None else spec["order"]
            details = f"{self._sph_spec_name(spec).upper()}, {order}x{order}"
            mode = _SPH_MODE

        elif mode == "PolyhedralGravity":
            if model is not None:
                details = f"{str(model.origin_name).upper()}, {model.n_facets} facets"

        elif mode == "SmoothedImpulsiveBurn":
            details = f"n_maneuvers={len(self._smoothed_maneuvers or [])}"

        elif mode == "FirstOrderGaussMarkov":
            details = self._frame_label(self._fogm_frame)

        elif mode == "PiecewiseFirstOrderGaussMarkov":
            parts = []
            if self._pfogm_n_batches is not None:
                parts.append(f"n_batches={self._pfogm_n_batches}")
            if self._frame_label(self._pfogm_frame):
                parts.append(self._frame_label(self._pfogm_frame))
            details = ", ".join(parts)

        elif mode == "ThermalRadiation":
            details = f"n_surfaces={len(self._thermal_model['surfaces'])}"

        elif mode == "AntennaRadiationPressure":
            details = ", ".join(e["name"] for e in self._antenna_emitters)

        elif mode == "FourierAcceleration":
            details = (
                f"degree={self._fourier_degree}, frame={self._fourier_frame}, "
                f"angle={self._fourier_angle}"
            )

        elif mode == "PiecewisePolynomialAcceleration":
            details = ", ".join(
                part
                for part in (
                    f"order={self._ppoly_order}",
                    f"n_batches={self._ppoly_n_batches}",
                    self._frame_label(self._ppoly_frame),
                )
                if part
            )

        elif mode in self._custom_models:
            details = mode
            mode = "CustomDynamicModel"

        return f"{mode}({details})" if details else mode

    @staticmethod
    def _frame_label(frame) -> str:
        """
        ``"frame=<name>"`` for an empirical model's component frame, or an
        empty string for the default inertial one, so a model that keeps the
        default keeps the label it always had.
        """
        frame = str(frame).strip()
        return "" if frame.lower() == "inertial" else f"frame={frame}"

    # ------------------------------------------------------------------
    #  Spherical-harmonics fields
    # ------------------------------------------------------------------
    class SphericalHarmonicsSpecError(ValueError):
        """A spherical-harmonics field is missing a setting or names a body
        that another field already covers."""

    #: Keys accepted for one entry of ``sph_harm_bodies``, and the constructor
    #: keyword each mirrors.
    _SPH_SPEC_KEYS = {
        "body": "sph_harm_body",
        "order": "sph_harm_order",
        "cs_file": "sph_harm_cs_file",
        "norm_flag": "sph_harm_norm_flag",
        "degree_max": "sph_degree_max",
    }

    @staticmethod
    def _sph_spec_name(spec: dict) -> str:
        """Return the body name of one harmonics field.

        Parameters
        ----------
        spec : dict
            A harmonics field spec whose ``"body"`` is either a body object or a
            serialized body dict.

        Returns
        -------
        str
            The body's ``spice_name`` (dict form) or ``name`` (object form); an
            empty string when neither is present.
        """
        body = spec.get("body")
        if isinstance(body, dict):
            return str(body.get("spice_name", ""))
        return str(getattr(body, "name", ""))

    @classmethod
    def _build_sph_specs(
        cls,
        sph_harm,
        sph_harm_body,
        sph_harm_order,
        sph_harm_cs_file,
        sph_harm_norm_flag,
        sph_degree_max,
        sph_harm_bodies,
    ) -> list:
        """
        Fold both spellings of the harmonics settings into one list of fields.

        ``sph_harm=True`` with the singular keywords describes one field;
        ``sph_harm_bodies`` describes any number. Passing both is fine — the
        singular one is simply the first field — which is what lets an existing
        single-field model gain a second body without being rewritten.

        Parameters
        ----------
        sph_harm : bool
            Whether the singular keywords describe a field.
        sph_harm_body, sph_harm_order, sph_harm_cs_file, sph_harm_norm_flag : optional
            The singular field's settings.
        sph_degree_max : int
            Degree cap for the singular field.
        sph_harm_bodies : list of dict, optional
            Further fields, each a dict with ``body``, ``order``, ``cs_file``
            and optionally ``norm_flag`` (default True) and ``degree_max``
            (default 100).

        Returns
        -------
        list of dict
            One normalised spec per field, in the order given.

        Raises
        ------
        SphericalHarmonicsSpecError
            If a field is missing a required setting, carries an unknown key,
            or names a body another field already covers.
        """
        specs: list = []
        # The singular keywords describe a field only when they actually name
        # one. `sph_harm=True` on its own just means harmonics are on, leaving
        # every field — the central body included — to `sph_harm_bodies`.
        # Order or cs_file without a body is an incomplete field rather than no
        # field, so it is still built and reported below.
        if sph_harm and any(
            value is not None
            for value in (sph_harm_body, sph_harm_order, sph_harm_cs_file)
        ):
            specs.append(
                {
                    "body": sph_harm_body,
                    "order": sph_harm_order,
                    "cs_file": sph_harm_cs_file,
                    "norm_flag": sph_harm_norm_flag,
                    "degree_max": sph_degree_max,
                }
            )

        for entry in sph_harm_bodies or []:
            if not isinstance(entry, dict):
                raise cls.SphericalHarmonicsSpecError(
                    "each entry of sph_harm_bodies must be a dict with keys "
                    f"{sorted(cls._SPH_SPEC_KEYS)}, got {type(entry).__name__}."
                )
            unknown = set(entry) - set(cls._SPH_SPEC_KEYS)
            if unknown:
                raise cls.SphericalHarmonicsSpecError(
                    f"unknown key(s) {sorted(unknown)} in a sph_harm_bodies "
                    f"entry; accepted keys are {sorted(cls._SPH_SPEC_KEYS)}."
                )
            specs.append(
                {
                    "body": entry.get("body"),
                    "order": entry.get("order"),
                    "cs_file": entry.get("cs_file"),
                    "norm_flag": entry.get("norm_flag", True),
                    "degree_max": entry.get("degree_max", 100),
                }
            )

        if sph_harm and not specs:
            raise cls.SphericalHarmonicsSpecError(
                "sph_harm=True but no harmonic field was described. Either name "
                "one with sph_harm_body/sph_harm_order/sph_harm_cs_file, or list "
                "the fields in sph_harm_bodies."
            )

        seen: dict = {}
        for spec in specs:
            missing = [
                cls._SPH_SPEC_KEYS[key]
                for key in ("body", "order", "cs_file", "norm_flag")
                if spec.get(key) is None
            ]
            if missing:
                name = cls._sph_spec_name(spec) or "<unnamed>"
                raise cls.SphericalHarmonicsSpecError(
                    f"spherical harmonics for '{name}' is missing: "
                    f"{', '.join(missing)}."
                )
            # Two fields for one body would double-count that body's gravity.
            name = cls._sph_spec_name(spec).upper()
            if name in seen:
                raise cls.SphericalHarmonicsSpecError(
                    f"'{name}' is given a spherical-harmonics field twice; a "
                    "body's gravity would be counted once per field. Merge the "
                    "two into a single field."
                )
            seen[name] = spec
        return specs

    def _sph_mode_keys(self) -> list:
        """Return the mode-list keys of this model's harmonics fields.

        Returns
        -------
        list of str
            The keys, in mode-list order.
        """
        return [mode for mode in self._dynamics_mode_list if _is_sph_mode(mode)]

    def _sph_models(self) -> list:
        """Return the live harmonics models.

        Returns
        -------
        list of SphericalHarmonicsGravity
            The instantiated harmonics models, in mode-list order. Fields not yet
            instantiated are skipped.
        """
        force_models = self.force_models
        return [
            force_models[mode] for mode in self._sph_mode_keys() if mode in force_models
        ]

    def _point_masses_less_sph_bodies(self) -> list:
        """
        ``third_bodies`` without any body that also carries a harmonics field.

        A harmonics field includes its body's monopole (C00 is forced to 1), so
        a body appearing in both lists is attracted twice. The harmonics model
        is the higher-fidelity of the two, so it keeps the body and the
        point-mass list gives it up.

        Returns
        -------
        list
            The point masses to hand to :class:`~scarabaeus.ThreeBodyGravity`.
        """
        if not self._third_bodies or not self._sph_harm_specs:
            return self._third_bodies
        return [
            pm
            for pm in self._third_bodies
            if not any(
                self._names_match(self._sph_spec_name(spec), pm)
                for spec in self._sph_harm_specs
            )
        ]

    def _sph_model_for_body(self, spice_id):
        """Return the harmonics model centred on a body.

        Parameters
        ----------
        spice_id : int
            SPICE ID of the body.

        Returns
        -------
        SphericalHarmonicsGravity or None
            The first live harmonics model whose body has this SPICE ID, or
            ``None``.
        """
        for model in self._sph_models():
            if getattr(model, "_sph_body_spice_id", None) == spice_id:
                return model
        return None

    # Internal model wiring
    @staticmethod
    def _names_match(name, body) -> bool:
        """
        Whether ``name`` and ``body`` refer to the same body.

        Parameters
        ----------
        name : str
            A body name, typically from :meth:`_sph_spec_name`.
        body : CelestialBody, Spacecraft or dict
            The body to compare against.

        Returns
        -------
        bool
            True when both name the same body, False when either is unnamed.

        Notes
        -----
        The comparison is by name because that is what
        :class:`~scarabaeus.SphericalHarmonicsGravity` compares for its own
        ``_sph_body_is_origin`` flag; matching it here keeps the two from
        disagreeing about whether a field is centred on the origin.
        """
        if isinstance(body, dict):
            other = body.get("spice_name")
        else:
            other = getattr(body, "name", None)
        if not name or not other:
            return False
        return str(name).upper() == str(other).upper()

    def _initialize_force_models(self) -> dict:
        """
        Instantiate all active translational force models.

        Requires ``_state_vector``, ``_base_frame`` and ``_t0`` to be set, which
        :meth:`initialize_force_models` does.

        Returns
        -------
        dict
            The force-model objects keyed by mode-list name.

        Raises
        ------
        RuntimeError
            If ``_state_vector`` is not set.
        SelfGravityError
            If a spherical-harmonics field is centred on the primary body itself.
        MissingBatchEpochError
            If the piecewise first-order Gauss-Markov or piecewise polynomial
            model is on and no ``t0`` was configured or set to anchor its batch
            grid.
        """
        if self._state_vector is None:
            raise RuntimeError(
                "state_vector not set. Call initialize_force_models() first."
            )

        sv = self._state_vector
        bf = self._base_frame
        t0 = self._t0
        tf = self._tf
        origin = sv.origin

        # Third bodies that are integrated in the state rather than read from
        # the ephemeris. This is the one piece of information the sub-models
        # cannot derive on their own, so it is resolved here, where the state
        # is available, and handed over explicitly.
        propagated_bodies = [
            owner
            for name, _, _, _, owner, _ in sv.state
            if name == "position" and isinstance(owner, CelestialBody)
        ]

        # A body does not accelerate under its own gravity: the field of a mass
        # distribution exerts no net force on that distribution. Asking for it
        # evaluates the harmonic expansion at zero separation, which is
        # singular — and before the position came from the state it was
        # evaluated at the state/ephemeris offset instead, giving a finite but
        # enormous spurious acceleration that silently wrecked the body's
        # trajectory. Reported rather than zeroed, because a silent no-op would
        # leave a model the user believes they enabled doing nothing.
        for spec in self._sph_harm_specs:
            if self._names_match(self._sph_spec_name(spec), self._primary_body):
                raise self.SelfGravityError(
                    f"spherical-harmonics body "
                    f"'{getattr(self._primary_body, 'name', self._primary_body)}' "
                    "is also this force model's primary_body, which asks for the "
                    "body's acceleration under its own gravity field. A body "
                    "exerts no net force on itself. Put this harmonics "
                    "configuration on the force model of the body that should "
                    "*feel* the field (typically the spacecraft), not on the "
                    "field body's own force model."
                )

        # A full-field model only stands in for the central body when it is a
        # model *of* the central body. A spherical-harmonics model of some
        # other body is a perturbation, so the origin still needs its monopole:
        # without this the propagation is left with no central body at all —
        # Earth harmonics in a heliocentric frame deleted the Sun, an
        # acceleration five orders of magnitude larger than what replaced it.
        # PolyhedralGravity is built on `origin` itself, so it never applies.
        if (
            self._sph_mode_keys()
            and "PointMassGravity" not in self._dynamics_mode_list
            and not any(
                self._names_match(self._sph_spec_name(spec), origin)
                for spec in self._sph_harm_specs
            )
        ):
            self._dynamics_mode_list.append("PointMassGravity")

        force_models: dict = {}

        if "PointMassGravity" in self._dynamics_mode_list:
            force_models["PointMassGravity"] = PointMassGravity(origin, bf)

        if "ThreeBodyGravity" in self._dynamics_mode_list:
            # A body with a harmonics field is already pulled on by that field,
            # whose C00 term is the full monopole. Leaving it in the point-mass
            # list too counted its attraction twice, exactly once over: with
            # Earth harmonics configured and Earth also a third body, the
            # spurious term measured a ratio of 1.0000 against Earth's monopole.
            # SphericalHarmonicsGravity carries the indirect (frame) half of the
            # perturbation for a non-origin field, so nothing is lost by
            # dropping the body here.
            force_models["ThreeBodyGravity"] = ThreeBodyGravity(
                origin=origin,
                point_masses=self._point_masses_less_sph_bodies(),
                base_frame=bf,
                t0=t0,
                tf=tf,
                propagated_bodies=propagated_bodies,
            )

        for index, spec in enumerate(self._sph_harm_specs):
            key = _sph_mode_key(index, self._sph_spec_name(spec))
            force_models[key] = SphericalHarmonicsGravity(
                spec["order"],
                spec["cs_file"],
                spec["body"],
                origin,
                spec["norm_flag"],
                spec["degree_max"],
                bf,
                propagated_bodies=propagated_bodies,
            )

        if "PolyhedralGravity" in self._dynamics_mode_list:
            force_models["PolyhedralGravity"] = PolyhedralGravity(
                origin,
                bf,
                self._polyhedron_shape,
                self._polyhedron_faces,
                density=self._polyhedron_density,
                gm=self._polyhedron_gm,
                body_fixed_frame=self._polyhedron_body_fixed_frame,
                scale=self._polyhedron_scale,
            )

        if "CannonballSRP" in self._dynamics_mode_list:
            force_models["CannonballSRP"] = CannonballSRP(
                origin, self._primary_body, bf, t0=t0, tf=tf
            )

        if "nPlateSRP" in self._dynamics_mode_list:
            force_models["nPlateSRP"] = nPlateSRP(
                origin, self._primary_body, bf, t0=t0, tf=tf
            )

        if "ThermalRadiation" in self._dynamics_mode_list:
            force_models["ThermalRadiation"] = ThermalRadiation(
                origin, self._primary_body, bf, self._thermal_model, t0=t0, tf=tf
            )

        if "AntennaRadiationPressure" in self._dynamics_mode_list:
            force_models["AntennaRadiationPressure"] = AntennaRadiationPressure(
                origin, self._primary_body, bf, self._antenna_emitters, t0=t0, tf=tf
            )

        if "FiniteBurn" in self._dynamics_mode_list:
            force_models["FiniteBurn"] = FiniteBurn(
                self._primary_body,
                bf,
                self._maneuver,
            )

        if "SmoothedImpulsiveBurn" in self._dynamics_mode_list:
            force_models["SmoothedImpulsiveBurn"] = SmoothedImpulsiveBurn(
                spacecraft=self._primary_body,
                maneuvers=self._smoothed_maneuvers,
                base_frame=bf,
                t0=t0,
                smoothing_factor=self._dv_smoothing_factor,
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
                base_frame=self.base_frame,
                param_name="A2_yark",
            )

        if "FirstOrderGaussMarkov" in self._dynamics_mode_list:
            force_models["FirstOrderGaussMarkov"] = FirstOrderGaussMarkov(
                origin,
                self._primary_body,
                bf,
                default_beta=self._fogm_beta,
                frame=self._fogm_frame,
                body_fixed_frame=self._fogm_body_fixed_frame,
                relative_body=self._fogm_relative_body,
            )
        if "PiecewiseFirstOrderGaussMarkov" in self._dynamics_mode_list:
            # Batch grid origin: the epoch this model was configured with, which
            # binding to a leg must not move. Falling back to the leg's own start
            # would restart batch 0 at every leg of a MissionSequence, so batch k
            # would mean a different time window in each leg while the legs share
            # one `a_pfogm` state.
            pfogm_t0 = self._t0_config if self._t0_config is not None else t0
            if pfogm_t0 is None:
                raise self.MissingBatchEpochError(
                    "t0 must be provided for the PiecewiseFirstOrderGaussMarkov "
                    "model: it anchors the batch grid. Pass the epoch the batches "
                    "start from to ForceModelTranslation(t0=...) — in a "
                    "MissionSequence, the same arc-start epoch for every leg."
                )
            force_models["PiecewiseFirstOrderGaussMarkov"] = (
                PiecewiseFirstOrderGaussMarkov(
                    origin,
                    self._primary_body,
                    bf,
                    t0=pfogm_t0,
                    default_beta=self._pfogm_beta,
                    batch_length=self._pfogm_batch_length,
                    n_batches=self._pfogm_n_batches,
                    frame=self._pfogm_frame,
                    body_fixed_frame=self._pfogm_body_fixed_frame,
                    relative_body=self._pfogm_relative_body,
                )
            )

        if "FourierAcceleration" in self._dynamics_mode_list:
            # Phase origin: the epoch this model was configured with, which
            # binding to a leg must not move. Falling back to the leg's own
            # start would restart the phase in every leg of a MissionSequence,
            # so a given coefficient would describe a different series in each
            # of them while the legs share one `a_fourier` state.
            force_models["FourierAcceleration"] = FourierAcceleration(
                origin,
                self._primary_body,
                bf,
                degree=self._fourier_degree,
                frame=self._fourier_frame,
                angle=self._fourier_angle,
                period=self._fourier_period,
                t0=self._t0_config if self._t0_config is not None else t0,
                body_fixed_frame=self._fourier_body_fixed_frame,
                relative_body=self._fourier_relative_body,
                coefficients=self._fourier_coefficients,
            )

        if "PiecewisePolynomialAcceleration" in self._dynamics_mode_list:
            # Same anchoring rule as the PFOGM grid above: the legs of a
            # MissionSequence share one `a_ppoly` state, so batch k must be the
            # same time window in every leg.
            ppoly_t0 = self._t0_config if self._t0_config is not None else t0
            if ppoly_t0 is None:
                raise self.MissingBatchEpochError(
                    "t0 must be provided for the PiecewisePolynomialAcceleration "
                    "model: it anchors the batch grid. Pass the epoch the batches "
                    "start from to ForceModelTranslation(t0=...) — in a "
                    "MissionSequence, the same arc-start epoch for every leg."
                )
            force_models["PiecewisePolynomialAcceleration"] = (
                PiecewisePolynomialAcceleration(
                    origin,
                    self._primary_body,
                    bf,
                    t0=ppoly_t0,
                    batch_length=self._ppoly_batch_length,
                    n_batches=self._ppoly_n_batches,
                    order=self._ppoly_order,
                    coefficients=self._ppoly_coefficients,
                    frame=self._ppoly_frame,
                    body_fixed_frame=self._ppoly_body_fixed_frame,
                    relative_body=self._ppoly_relative_body,
                )
            )

        # User-defined models: already constructed by the script, so they only
        # need to be bound to this propagation.
        for name, model in self._custom_models.items():
            model._bind(
                origin=origin,
                base_frame=bf,
                primary_body=self._primary_body,
                t0=t0,
                tf=tf,
            )
            force_models[name] = model

        # Radiation models flown in nadirPointingSun take their attitude
        # from the velocity, so they need the da/dv block as well. Read here,
        # at binding, so an attitude set after construction is still seen.
        if any(
            getattr(fm, "depends_on_velocity", False) for fm in force_models.values()
        ):
            self._velocity_partial_active = True

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
        """Normalize a name for case-insensitive key matching.

        Parameters
        ----------
        s : str
            The name.

        Returns
        -------
        str
            ``s`` stripped and lower-cased.
        """
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

        # User-defined parameter dynamics. Resolved first and by exact name:
        # custom names can never collide with a built-in one.
        owner = self._custom_param_dot_owners.get(str(name).lower())
        if owner is not None:
            model, parameter = owner
            rate = model._compute_parameter_dynamics(
                parameter.name, position.values, current_state, epoch, body
            )

            return ArrayWUnits.from_trusted(rate, parameter.rate_units)

        dynamics_parameters = {}

        if "FirstOrderGaussMarkov" in self.force_models:
            gm = self.force_models["FirstOrderGaussMarkov"]

            a_dot = gm._compute_acceleration_state_derivative(
                position, current_state, epoch, body
            ).reshape(3)

            dynamics_parameters["a_fogm_dot"] = lambda: ArrayWUnits(
                a_dot,
                _U_KM_S3,
            )

        if "PiecewiseFirstOrderGaussMarkov" in self.force_models:
            pwgm = self.force_models["PiecewiseFirstOrderGaussMarkov"]

            a_dot = pwgm._compute_acceleration_state_derivative(
                position, current_state, epoch, body
            ).reshape(3 * pwgm.n_batches)

            dynamics_parameters["a_pfogm_dot"] = lambda: ArrayWUnits(
                a_dot,
                _U_KM_S3,
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
        # Parse the key once per unique name (the parse is epoch-independent).
        parsed = self._partial_key_cache.get(name)
        if parsed is None:
            m = _PARTIAL_KEY_RE.match(str(name).strip())
            if not m:
                # backward compatibility: try old naming if you want
                self._partial_key_cache[name] = False
                return None
            lhs = m.group("lhs").lower()
            rhs = m.group("rhs").lower()
            # Strip maneuver numbering suffixes: k_thrust1 -> k_thrust, dRA2 -> dra.
            parsed = (
                lhs,
                int(m.group("lhs_id")),
                rhs,
                int(m.group("rhs_id")),
                _TRAILING_DIGITS_RE.sub("", lhs),
                _TRAILING_DIGITS_RE.sub("", rhs),
            )
            self._partial_key_cache[name] = parsed
        elif parsed is False:
            return None

        lhs, lhs_id, rhs, rhs_id, lhs_base, rhs_base = parsed

        # Sanity: this force model instance is for self._primary_body,
        # but you may still compute partials for whichever `body` is passed in.
        if getattr(body, "spice_id", None) != lhs_id:
            # asked for a partial of a different row-body than the one supplied
            return None

        # User-defined dynamics. Checked first and by exact (lower-cased) name:
        # custom parameter names can never shadow a built-in one, since
        # StateDefinition.register_custom_parameter rejects those.
        if rhs in self._custom_param_owners or lhs in self._custom_param_dot_owners:
            return self._collect_custom_partials(
                lhs, lhs_id, rhs, rhs_id, position, current_state, epoch, body
            )

        # dv/dv block
        if lhs_base == "velocity" and rhs_base == "velocity" and rhs_id == lhs_id:
            return self._compute_total_velocity_partial(
                position, current_state, epoch, body
            )

        # da/d(velocity@same body) — non-zero only for models that declare
        # ``depends_on_velocity``: custom forces such as drag, a velocity-framed
        # Fourier model, or radiation models in nadirPointingSun.
        if lhs_base == "acceleration" and rhs_base == "velocity":
            if rhs_id == lhs_id and self._velocity_partial_active:
                return self._compute_total_acceleration_partial_by_velocity(
                    position, current_state, epoch, body
                )
            return None

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

            # Every model whose acceleration depends on that body's position
            # contributes; a body can be both a point mass and the harmonics
            # body, so the contributions are summed rather than returned from
            # the first match.
            wrt_key = getattr(wrt_body, "name", "").upper()
            total = None

            if "ThreeBodyGravity" in self.force_models:
                threeb = self.force_models["ThreeBodyGravity"]
                if wrt_key in getattr(threeb, "pnt_masses", []):
                    d = threeb._compute_partial_by_third_body_position(
                        position=position.values,
                        current_state=current_state,
                        epoch=epoch,
                        body=body,
                        point_mass_name=wrt_key,
                    )
                    if d is not None:
                        total = d if total is None else total + d

            # The harmonics field is evaluated relative to its own body, so
            # moving that body changes this acceleration. Missing, this leaves
            # an estimated flyby target invisible to the dynamics that its
            # own shape drives.
            for sph in self._sph_models():
                if str(sph._sph_body_name()).upper() != wrt_key:
                    continue
                d = sph._compute_partial_by_sph_body_position(
                    position.values, current_state, epoch, body
                )
                if d is not None:
                    total = d if total is None else total + d

            if total is None:
                return None
            return ArrayWUnits.from_trusted(total, _U_S2_INV)

        # da/d(eta_srp@same spacecraft)
        if lhs_base == "acceleration" and rhs_base == "eta_srp":
            if rhs_id == lhs_id:
                return self._compute_acceleration_partial_by_eta(
                    position, current_state, epoch, body
                )
            return None

        # da/d(eta_thermal@same spacecraft) and da/d(eta_antenna@same spacecraft)
        if lhs_base == "acceleration" and rhs_base in _EMISSION_SCALE_FACTORS:
            mode = _EMISSION_SCALE_FACTORS[rhs_base]
            if rhs_id != lhs_id or mode not in self.force_models:
                return None
            d = self.force_models[mode]._compute_partial_by_eta(
                position.values, current_state, epoch, body
            )
            return ArrayWUnits.from_trusted(np.reshape(d, (3, 1)), _U_KM_S2)

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
                return ArrayWUnits.from_trusted(d, _U_S2_INV)  # (3,1)
            return None

        # da/d(dv_smooth@same spacecraft) and da/d(dt_dv@same spacecraft).
        # The trailing digits stripped into rhs_base are the maneuver index, so
        # they are read back off the full name here to pick the right pulse.
        if lhs_base == "acceleration" and rhs_base in ("dv_smooth", "dt_dv"):
            if rhs_id != lhs_id:
                return None
            if "SmoothedImpulsiveBurn" not in self.force_models:
                return None
            sib = self.force_models["SmoothedImpulsiveBurn"]

            suffix = rhs[len(rhs_base) :]
            index = int(suffix) if suffix else 1

            if rhs_base == "dv_smooth":
                d = sib._compute_partial_by_dv_smooth(
                    position=position.values,
                    current_state=current_state,
                    epoch=epoch,
                    body=body,
                    index=index,
                )
                return ArrayWUnits.from_trusted(d, _U_S_INV)  # (3,3)

            d = sib._compute_partial_by_dt_dv(
                position=position.values,
                current_state=current_state,
                epoch=epoch,
                body=body,
                index=index,
            )
            return ArrayWUnits.from_trusted(d, _U_KM_S3)  # (3,1)

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

                return ArrayWUnits.from_trusted(d, _U_S2_INV)  # (3,1)
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

        # da/d(a_fourier@same spacecraft)
        if lhs_base == "acceleration" and rhs_base == "a_fourier":
            if rhs_id == lhs_id:
                if "FourierAcceleration" not in self.force_models:
                    return None
                fa = self.force_models["FourierAcceleration"]
                return fa._compute_partial_by_coefficients(
                    position.values, current_state, epoch, body
                )
            return None

        # da/d(a_ppoly@same spacecraft)
        if lhs_base == "acceleration" and rhs_base == "a_ppoly":
            if rhs_id == lhs_id:
                if "PiecewisePolynomialAcceleration" not in self.force_models:
                    return None
                pp = self.force_models["PiecewisePolynomialAcceleration"]
                return pp._compute_partial_by_coefficients(
                    position.values, current_state, epoch, body
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
            return ArrayWUnits.from_trusted(J, _U_S_INV)

        # da_fogm / d(a_pfogm)
        if lhs_base == "a_fogm_dot" and rhs_base == "a_pfogm" and rhs_id == lhs_id:
            if "PiecewiseFirstOrderGaussMarkov" not in self.force_models:
                return None
            gm = self.force_models["PiecewiseFirstOrderGaussMarkov"]
            J = gm._compute_partial_of_acceleration_state_by_acceleration_state(
                position, current_state, epoch, body
            )
            return ArrayWUnits.from_trusted(J, _U_S_INV)

        # da_fogm / d(beta_fogm)
        if lhs_base == "a_fogm_dot" and rhs_base == "beta_fogm" and rhs_id == lhs_id:
            if "FirstOrderGaussMarkov" not in self.force_models:
                return None
            gm = self.force_models["FirstOrderGaussMarkov"]
            J = gm._compute_partial_of_acceleration_state_by_beta(
                position, current_state, epoch, body
            )
            return ArrayWUnits.from_trusted(J, _U_KM_S2)

        # da_fogm / d(beta_pfogm)
        if lhs_base == "a_fogm_dot" and rhs_base == "beta_pfogm" and rhs_id == lhs_id:
            if "PiecewiseFirstOrderGaussMarkov" not in self.force_models:
                return None
            gm = self.force_models["PiecewiseFirstOrderGaussMarkov"]
            J = gm._compute_partial_of_acceleration_state_by_beta(
                position, current_state, epoch, body
            )
            return ArrayWUnits.from_trusted(J, _U_KM_S2)

        # -------------------------------------------------------------------
        # da/d(gm@<body_id>)  — SH body GM, central-body GM, or third-body GM
        # -------------------------------------------------------------------
        if lhs_base == "acceleration" and rhs_base == "gm":
            # SH body GM (when SphericalHarmonicsGravity is active — no PointMassGravity
            # for the same body in that case)
            if "SphericalHarmonicsGravity" in self.force_models:
                sh = self.force_models["SphericalHarmonicsGravity"]
                if rhs_id == getattr(sh, "_sph_body_spice_id", None):
                    d = sh._compute_partial_by_mu(
                        position.values, current_state, epoch, body
                    )
                    return ArrayWUnits.from_trusted(d, _U_KM2_INV)
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
                        return ArrayWUnits.from_trusted(d, _U_KM2_INV)
            return None

        # -------------------------------------------------------------------
        # da/d(cnm_n_m@<body_id>)  and  da/d(snm_n_m@<body_id>)
        # -------------------------------------------------------------------
        if lhs_base == "acceleration" and self._sph_mode_keys():
            sh = self._sph_model_for_body(rhs_id)
            if sh is not None:
                cnm_match = re.match(r"cnm_(\d+)_(\d+)$", rhs)
                if cnm_match:
                    nn, mm = int(cnm_match.group(1)), int(cnm_match.group(2))
                    if nn <= sh.order and mm <= nn:
                        partials_C = sh._compute_partial_by_C(
                            position.values, current_state, epoch
                        )
                        d = partials_C[:, nn, mm].reshape(3, 1)
                        return ArrayWUnits.from_trusted(d, _U_KM_S2)
                    return None
                snm_match = re.match(r"snm_(\d+)_(\d+)$", rhs)
                if snm_match:
                    nn, mm = int(snm_match.group(1)), int(snm_match.group(2))
                    if nn <= sh.order and mm <= nn and mm > 0:
                        partials_S = sh._compute_partial_by_S(
                            position.values, current_state, epoch
                        )
                        d = partials_S[:, nn, mm].reshape(3, 1)
                        return ArrayWUnits.from_trusted(d, _U_KM_S2)
                    return None
                # da/d(love_k_n_m@<body_id>): the solid tide reaches the
                # trajectory only through the field's coefficients, and each
                # k_nm drives its own (n, m) alone and linearly, so this
                # contracts partials the field already computes.
                love_match = re.match(r"love_k_(\d+)_(\d+)$", rhs)
                if love_match:
                    nn, mm = int(love_match.group(1)), int(love_match.group(2))
                    d = sh._compute_partial_by_love_k(
                        nn, mm, position.values, current_state, epoch
                    )
                    return ArrayWUnits.from_trusted(d, _U_KM_S2)

        # -------------------------------------------------------------------
        # da/d(pole_ra@<body_id>), da/d(pole_dec@<body_id>),
        # da/d(pole_pm@<body_id>), da/d(pole_w0@<body_id>)
        # -------------------------------------------------------------------
        if lhs_base == "acceleration" and rhs_base in (
            "pole_ra",
            "pole_dec",
            "pole_pm",
            "pole_w0",
        ):
            sh = self._sph_model_for_body(rhs_id)
            if sh is None:
                return None
            if rhs_base == "pole_ra":
                d = sh._compute_partial_by_pole_RA(
                    position.values, current_state, epoch, body
                )
                return ArrayWUnits.from_trusted(d, _U_KM_S2)
            if rhs_base == "pole_dec":
                d = sh._compute_partial_by_pole_DEC(
                    position.values, current_state, epoch, body
                )
                return ArrayWUnits.from_trusted(d, _U_KM_S2)
            if rhs_base == "pole_pm":
                d = sh._compute_partial_by_pole_PM(
                    position.values, current_state, epoch, body
                )
                # units: km/s^2 / (deg/day) — keep as km*s^-2 for filter consistency
                return ArrayWUnits.from_trusted(d, _U_KM_S2)
            if rhs_base == "pole_w0":
                d = sh._compute_partial_by_pole_W0(
                    position.values, current_state, epoch, body
                )
                return ArrayWUnits.from_trusted(d, _U_KM_S2)
            return None

        return None

    def _collect_custom_partials(
        self, lhs, lhs_id, rhs, rhs_id, position, current_state, epoch, body
    ):
        """
        Resolve a partial-derivative block that involves a custom parameter.

        Handles the three shapes a :class:`~scarabaeus.CustomParameter` can take
        part in:

        * :math:`\\partial \\mathbf a / \\partial p` — summed over every custom
          model declaring *p*, since several forces may share one parameter;
        * :math:`\\partial \\dot p / \\partial \\mathbf r` and
          :math:`\\partial \\dot p / \\partial \\mathbf v` for a dynamic
          parameter;
        * :math:`\\partial \\dot p / \\partial q`, where *q* is any parameter of
          the model that integrates *p*.

        Parameters
        ----------
        lhs : str
            Lower-cased left-hand side of the partial key (the derivative row).
        lhs_id : int
            SPICE ID of the row body.
        rhs : str
            Lower-cased right-hand side of the partial key (the state column).
        rhs_id : int
            SPICE ID of the column body.
        position : ArrayWUnits
            3x1 position of the body, expressed in kilometers.
        current_state : dict
            Dictionary containing the current state parameters.
        epoch : float
            Current simulation epoch.
        body : CelestialBody
            The celestial body influencing the dynamics.

        Returns
        -------
        ArrayWUnits or None
            The partial block, or ``None`` when this force model does not
            contribute to the requested entry.
        """
        # All custom parameters belong to the body the force model acts on.
        if rhs_id != lhs_id:
            return None

        position_vec = position.values

        # d(acceleration) / d(custom parameter)
        if lhs == "acceleration":
            owners = self._custom_param_owners.get(rhs)
            if not owners:
                return None

            block = None
            for model, parameter in owners:
                contribution = model._compute_partial_by_parameter(
                    parameter.name, position_vec, current_state, epoch, body
                )
                block = contribution if block is None else block + contribution

            return ArrayWUnits.from_trusted(
                block, owners[0][1].acceleration_partial_units
            )

        # d(custom parameter rate) / d(anything)
        owner = self._custom_param_dot_owners.get(lhs)
        if owner is None:
            return None

        model, parameter = owner

        if rhs in ("position", "velocity"):
            with_respect_to = rhs
            units = parameter.rate_partial_units(rhs)
        else:
            differentiated = next(
                (
                    other
                    for other in model.parameters.values()
                    if other.name.lower() == rhs
                ),
                None,
            )
            if differentiated is None:
                return None
            with_respect_to = differentiated.name
            units = parameter.rate_units
            if differentiated.units is not None:
                units = units / differentiated.units

        block = model._compute_parameter_dynamics_partial(
            parameter.name,
            with_respect_to,
            position_vec,
            current_state,
            epoch,
            body,
        )

        return ArrayWUnits.from_trusted(block, units)

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
        # Get the position vector
        position_vec = position.values

        # Initialize total acceleration
        total_acceleration = np.zeros(3)

        # Get the position vector
        position_vec = position.values

        # Stack toghether individual acceleration components from the list of dynamics model
        # (bind force_models once to avoid the re-validating property per access)
        fms = self.force_models
        for model in self._dynamics_mode_list:
            # Add accelerations
            total_acceleration += fms[model].compute_acceleration(
                position_vec, current_state, epoch, body
            )

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
        # Constant identity, precomputed once at module load (hot path).
        return _EYE3_AWU

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
        # (bind force_models once to avoid the re-validating property per access)
        fms = self.force_models
        for model in self._dynamics_mode_list:
            fm = fms.get(model)
            if fm is not None:
                # Add partials
                partial_a_by_pos += fm._compute_partial_by_position(
                    position_vec, current_state, epoch, body
                )

        # Add units to partials
        total_partial_awu = ArrayWUnits.from_trusted(partial_a_by_pos, _U_S2_INV)

        return total_partial_awu

    def _compute_total_acceleration_partial_by_velocity(
        self, position: ArrayWUnits, current_state: dict, epoch, body
    ) -> ArrayWUnits:
        """
        Computes the total acceleration Jacobian with respect to velocity.

        Only models that declare ``depends_on_velocity`` contribute: a
        velocity-dependent :class:`~scarabaeus.CustomDynamicModel` such as drag,
        a :class:`~scarabaeus.FourierAcceleration` whose frame or phase is
        built from the velocity, or a radiation model (N-plate SRP, thermal,
        antenna) whose spacecraft flies ``nadirPointingSun``, where the
        attitude follows the velocity. Every other translational model leaves
        this block at zero, and it is requested only when at least one model
        raises the flag.

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
            The 3x3 Jacobian of acceleration with respect to velocity. Expressed
            in units of :math:`\\frac{1}{s}`.
        """
        partial_a_by_vel = np.zeros((3, 3))
        position_vec = position.values

        fms = self.force_models
        for name in self._dynamics_mode_list:
            fm = fms.get(name)
            if fm is None or not getattr(fm, "depends_on_velocity", False):
                continue
            partial_a_by_vel += fm._compute_partial_by_velocity(
                position_vec, current_state, epoch, body
            )

        return ArrayWUnits.from_trusted(partial_a_by_vel, _U_S_INV)

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
        partial_a_by_dv = ArrayWUnits.from_trusted(partial_a_by_dv, _U_S_INV)

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

        partial_a_by_eta_awu = ArrayWUnits.from_trusted(
            partial_a_by_eta.reshape(3, 1), _U_KM_S2
        )

        return partial_a_by_eta_awu

    @staticmethod
    def _check_sph_partials(sh, position, full_state, epoch, body, write_line):
        """
        Verify the SphericalHarmonicsGravity partials without Python autograd.

        The harmonics kernels run in Rust, which autograd cannot trace. Every
        reference value is instead forward-mode automatic differentiation (dual
        numbers) inside Rust, of the same kernels that compute the acceleration:

        - position: each body-fixed position component seeded in turn.
        - GM: GM seeded, the indirect term included when the harmonics body is
          not the propagation origin.
        - Cnm, Snm: one coefficient seeded per acceleration pass.
        - pole_ra, pole_dec, pole_pm (orientation or "EOP" parameters): the pole
          angle seeded through the IAU pole rotation, which Rust rebuilds as
          :meth:`~scarabaeus.SphericalHarmonicsGravity._get_R_b2rf_effective`
          does. pole_w0 reuses the pole_pm derivative divided by :math:`d`,
          since both enter only through :math:`W = W_0 + \\mathrm{PM}\\,d`.

        GM, the coefficients and the pole are differentiated at the values the
        acceleration uses, ``full_state`` overrides included. Runs whenever the
        Rust core is present, with or without ``UseAutoGrad``.

        Parameters
        ----------
        sh : SphericalHarmonicsGravity
            The harmonics model to check.
        position : ArrayWUnits
            Position of the propagated body, in km.
        full_state : dict
            The propagated state, keyed by ``(name, spice_id)``.
        epoch : float
            Epoch, in TDB seconds past J2000.
        body : Body
            The propagated body.
        write_line : callable
            Called as ``write_line(model, dependent, independent, analytical,
            algorithmic)`` once per (dependent, independent) pair.
        """
        import math as _m
        import spiceypy as _sp

        MODEL = "SphericalHarmonicsGravity"
        posv = position.values
        xyz = ["x", "y", "z"]
        sid = sh._sph_body_spice_id
        core = sh._rust_core

        def scalar(value):
            return float(value.values if hasattr(value, "values") else value)

        # --- position, rotated into the reference frame ---
        R_b2rf = sh._get_R_b2rf_effective(full_state, epoch)
        r_bf = sh._rf_transform(posv, epoch, R_b2rf, full_state)
        jac_ad_bf = np.asarray(
            core.compute_jacobian_ad_bf(r_bf, sh.mu, sh.Cnm, sh.Snm)
        ).reshape(3, 3)
        algo_pos = sh._jacobian_pos_transform_rf(jac_ad_bf, epoch, R_b2rf)
        an_pos = sh._compute_partial_by_position(posv, full_state, epoch, body)
        for j, ind in enumerate(xyz):
            for i, dep in enumerate(xyz):
                write_line(MODEL, dep, ind, an_pos[i][j], algo_pos[i][j])

        # GM and coefficients as compute_acceleration applies them
        mu = float(sh.mu)
        if sid is not None and ("gm", sid) in full_state:
            mu = scalar(full_state[("gm", sid)])
        Cnm, Snm = sh._apply_cs_from_state(full_state)

        # --- GM ---
        if sid is not None:
            r_origin = (
                None
                if sh._sph_body_is_origin
                else sh._sph_body_position(full_state, epoch)
            )
            algo = core.compute_partial_by_mu_ad_rf(
                r_bf, R_b2rf, mu, Cnm, Snm, r_origin
            )
            an = np.asarray(
                sh._compute_partial_by_mu(posv, full_state, epoch, body)
            ).reshape(3)
            for i, dep in enumerate(xyz):
                write_line(MODEL, dep, "gm", an[i], algo[i])

        # --- pole RA/DEC/PM (orientation params) ---
        if sid is not None and sh._iau_ra_coeffs is not None:
            d = epoch / _sp.spd()
            T = d / 36525.0
            ra_c, dec_c, pm_c = sh._iau_ra_coeffs, sh._iau_dec_coeffs, sh._iau_pm_coeffs
            keys = [("pole_ra", sid), ("pole_dec", sid), ("pole_pm", sid)]
            # PCK values where the pole is not estimated; all three keys are
            # set so the analytical side runs on the same IAU rotation.
            pole_state = {
                keys[0]: _m.radians(ra_c[0] + ra_c[1] * T + ra_c[2] * T**2),
                keys[1]: _m.radians(dec_c[0] + dec_c[1] * T + dec_c[2] * T**2),
                keys[2]: float(pm_c[1]),
                **full_state,
            }
            ra, dec, pm = (scalar(pole_state[k]) for k in keys)
            # An estimated W0 is held in rad; Rust takes it in deg.
            w0 = (
                _m.degrees(scalar(full_state[("pole_w0", sid)]))
                if ("pole_w0", sid) in full_state
                else float(pm_c[0])
            )

            ref = sh._ref_frame.name
            M_j2rf = (
                np.eye(3)
                if ref.upper() == "J2000"
                else sh._spice_xfrm_cached("J2000", ref, epoch)
            )
            r_rel = (
                posv
                if sh._sph_body_is_origin
                else posv - sh._sph_body_position(pole_state, epoch)
            )
            _, algo_pole = core.compute_partial_by_pole_ad_rf(
                r_rel, M_j2rf, ra, dec, w0, pm, d, mu, Cnm, Snm
            )
            for (label, _), fn, algo in zip(
                keys,
                (
                    sh._compute_partial_by_pole_RA,
                    sh._compute_partial_by_pole_DEC,
                    sh._compute_partial_by_pole_PM,
                ),
                algo_pole,
            ):
                an = np.asarray(fn(posv, pole_state, epoch, body)).reshape(3)
                for i, dep in enumerate(xyz):
                    write_line(MODEL, dep, label, an[i], algo[i])
            # W0 and PM enter only through W = W0 + PM d, so the PM derivative
            # over d (deg) gives the W0 one per rad.
            if d != 0.0:
                algo_w0 = np.asarray(algo_pole[2]) / (d * _m.pi / 180.0)
                an = np.asarray(
                    sh._compute_partial_by_pole_W0(posv, pole_state, epoch, body)
                ).reshape(3)
                for i, dep in enumerate(xyz):
                    write_line(MODEL, dep, "pole_w0", an[i], algo_w0[i])

        # --- Cnm / Snm coefficients ---
        # NOTE: one acceleration pass per coefficient; this is a debug harness.
        ad_C_bf, ad_S_bf = core.compute_partial_by_cs_ad_bf(r_bf, mu, Cnm, Snm)
        an_C = np.asarray(sh._compute_partial_by_C(posv, full_state, epoch))
        an_S = np.asarray(sh._compute_partial_by_S(posv, full_state, epoch))
        algo_C = sh._jacobian_C_transform_rf(np.asarray(ad_C_bf), epoch, R_b2rf)
        algo_S = sh._jacobian_C_transform_rf(np.asarray(ad_S_bf), epoch, R_b2rf)
        for an, algo, tag, m_start in ((an_C, algo_C, "C", 0), (an_S, algo_S, "S", 1)):
            for n in range(0, sh.order + 1):
                for m in range(m_start, n + 1):
                    ind = f"{tag}{n},{m}"
                    for i, dep in enumerate(xyz):
                        write_line(MODEL, dep, ind, an[i][n][m], algo[i][n][m])

    @staticmethod
    def _check_nplate_partials(npl, position, full_state, epoch, body, write_line):
        """
        Verify the nPlateSRP partials without Python autograd.

        The plate sums and the SC-fixed to inertial mapping run in Rust, which
        autograd cannot trace. The reference values are forward-mode automatic
        differentiation (dual numbers) inside Rust, of the same kernel that
        computes the acceleration (``NPlateSRPCore.compute_partials_ad``):

        - position: each inertial Sun-to-body component seeded in turn, through
          the DCM mapping, so the rotation of the analytical Jacobian is checked
          as well; for ``nadirPointingSun`` the DCM is seeded with
          its position derivative too.
        - velocity (``nadirPointingSun`` only, the one attitude that
          follows the velocity): the DCM seeded with its velocity derivative.
        - eta_srp: eta seeded.

        Plate normals, mass and the Sun position are held fixed, as in the
        analytical partials. The nadir DCM derivatives
        (:meth:`~scarabaeus.DynamicalUtils.nadirPointingSun`) are inputs to
        both sides, so this checks how the kernel uses them. The dual numbers
        are summed over the plate groups (body frame; CK and named frame), as
        the analytical partials are. Runs whenever the Rust cores are present,
        with or without ``UseAutoGrad``.

        Parameters
        ----------
        npl : nPlateSRP
            The n-plate SRP model to check.
        position : ArrayWUnits
            Position of the propagated body, in km.
        full_state : dict
            The propagated state, keyed by ``(name, spice_id)``.
        epoch : float
            Epoch, in TDB seconds past J2000.
        body : Body
            The propagated body.
        write_line : callable
            Called as ``write_line(model, dependent, independent, analytical,
            algorithmic)`` once per (dependent, independent) pair.
        """
        MODEL = "nPlateSRP"
        posv = position.values
        xyz = ["x", "y", "z"]

        jac_ad, jac_v_ad, eta_ad = np.zeros(9), np.zeros(9), np.zeros(3)
        for core, inputs, D_r, D_v in npl._rust_terms(
            posv, full_state, epoch, body, derivatives=True
        ):
            jac, jac_v, eta = core.compute_partials_ad(*inputs, D_r, D_v)
            jac_ad = jac_ad + np.asarray(jac)
            jac_v_ad = jac_v_ad + np.asarray(jac_v)
            eta_ad = eta_ad + np.asarray(eta)
        algo_pos = jac_ad.reshape(3, 3)

        # --- position ---
        an_pos = npl._compute_partial_by_position(posv, full_state, epoch, body)
        for j, ind in enumerate(xyz):
            for i, dep in enumerate(xyz):
                write_line(MODEL, dep, ind, an_pos[i][j], algo_pos[i][j])

        # --- velocity (the nadir attitude follows it) ---
        if npl.depends_on_velocity:
            algo_vel = jac_v_ad.reshape(3, 3)
            an_vel = npl._compute_partial_by_velocity(posv, full_state, epoch, body)
            for j, ind in enumerate(xyz):
                for i, dep in enumerate(xyz):
                    write_line(MODEL, dep, f"v{ind}", an_vel[i][j], algo_vel[i][j])

        # --- eta_srp ---
        an_eta = np.asarray(
            npl._compute_partial_by_eta(posv, full_state, epoch, body)
        ).reshape(3)
        for i, dep in enumerate(xyz):
            write_line(MODEL, dep, "eta", an_eta[i], eta_ad[i])

    @staticmethod
    def _check_emission_partials(fm, position, full_state, epoch, body, write_line):
        """
        Verify the thermal or antenna recoil partials against autograd.

        The position, velocity and scale-factor partials are each compared with
        the autograd Jacobian of ``compute_acceleration``, seeded through the
        state as the propagator would pass them. The velocity lines are written
        only for the Sun-pointed attitude, the one attitude that depends on it.

        Parameters
        ----------
        fm : ThermalRadiation or AntennaRadiationPressure
            The bound model.
        position : ArrayWUnits
            Spacecraft position, km.
        full_state : dict
            The propagated state; left unchanged.
        epoch : float
            TDB epoch (seconds past J2000).
        body : Spacecraft
            The emitting spacecraft.
        write_line : callable
            ``write_line(model, dependent, independent, analytical, algo)``.
        """
        model = type(fm).__name__
        sid = getattr(body, "spice_id", None)
        pos = np.asarray(position.values, dtype=float)
        xyz = ("x", "y", "z")

        def acc_by_position(r):
            return fm.compute_acceleration(r, full_state, epoch, body)

        def acc_by_velocity(v):
            return fm.compute_acceleration(
                pos, {**full_state, ("velocity", sid): v}, epoch, body
            )

        def acc_by_eta(eta):
            return fm.compute_acceleration(
                pos, {**full_state, (fm.param_name, sid): eta}, epoch, body
            )

        algo_pos = jacobian(acc_by_position)(pos)
        an_pos = fm._compute_partial_by_position(pos, full_state, epoch, body)
        for j, ind in enumerate(xyz):
            for i, dep in enumerate(xyz):
                write_line(model, dep, ind, an_pos[i][j], algo_pos[i][j])

        if fm.depends_on_velocity:
            vel = full_state.get(("velocity", sid), full_state.get("velocity"))
            vel = np.asarray(getattr(vel, "values", vel), dtype=float)
            algo_vel = jacobian(acc_by_velocity)(vel)
            an_vel = fm._compute_partial_by_velocity(pos, full_state, epoch, body)
            for j, ind in enumerate(xyz):
                for i, dep in enumerate(xyz):
                    write_line(model, dep, f"v{ind}", an_vel[i][j], algo_vel[i][j])

        eta = float(fm._get_scale(full_state, body))
        algo_eta = jacobian(acc_by_eta)(eta)
        an_eta = fm._compute_partial_by_eta(pos, full_state, epoch, body)
        for i, dep in enumerate(xyz):
            write_line(model, dep, "eta", an_eta[i], algo_eta[i])

    def _check_all_partials(self, position, full_state, epoch, body):
        """
        Validate every force model's analytical partials against algorithmic
        derivatives.

        For each mode in the dynamics mode list:

        1. Spherical harmonics and n-plate SRP are checked against Rust
           dual-number derivatives (:meth:`_check_sph_partials`,
           :meth:`_check_nplate_partials`), and custom models run their own
           check; all three run with or without ``UseAutoGrad``.
        2. Every other model is differentiated with autograd, but only when the
           environment variable ``UseAutoGrad`` is ``"TRUE"``; otherwise the mode
           is skipped with a warning. Thermal and antenna recoil also compare
           their velocity and scale-factor partials
           (:meth:`_check_emission_partials`).
        3. The analytical partials come from each model's
           ``_compute_partial_by_*`` methods.
        4. One line per comparison is appended to ``partials_FMoutput.txt`` in
           the current directory, in the format
           ``epoch,model,d(acc_<dep>)/d(<indep>),analytical,algo,error`` with
           ``error = (analytical - algo) / algo``.

        This is a diagnostic utility called by the Propagator when partial
        checking is on.

        Parameters
        ----------
        position : ArrayWUnits
            Position of the propagated body, in km.
        full_state : dict
            The propagated state, keyed by ``(name, spice_id)``. Some entries are
            overwritten while differentiating and restored afterwards.
        epoch : float
            Epoch, in TDB seconds past J2000.
        body : Body
            The propagated body; its ``spice_id`` keys the SRP parameters.

        Warns
        -----
        UserWarning
            For each autograd-checked mode skipped because ``UseAutoGrad`` is not
            ``"TRUE"``.

        Notes
        -----
        The autograd wrappers temporarily overwrite model attributes
        (``origin_mu``, the SRP ``eta`` map, the maneuver ``thrust``, ``ra`` and
        ``dec``, the Gauss-Markov ``default_beta``) and state entries while
        differentiating, and restore them afterwards.
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

        def wrap_third_body_gm(gmval, spice_id, pos, full_state, epoch, body):
            state = {**full_state, ("gm", spice_id): gmval}
            acc = self.force_models[mode].compute_acceleration(
                pos.values, state, epoch, body
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

        def wrap_fbthrust(thrustval, units, pos, full_state, epoch, body):
            M1 = Maneuver()
            M1.thrust = ArrayWUnits(thrustval, N)
            M1.mass_flow = ArrayWUnits(3e-4, kg / sec)
            M1.start_time = ArrayWUnits(12 * 60 * 60, sec)
            M1.end_time = ArrayWUnits(13 * 60 * 60, sec)
            M1.ux = ArrayWUnits(1 / np.sqrt(6), None)
            M1.uy = ArrayWUnits(3 / np.sqrt(6), None)
            M1.uz = ArrayWUnits(2 / np.sqrt(6), None)
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
            M1.ux = ArrayWUnits(1 / np.sqrt(6), None)
            M1.uy = ArrayWUnits(3 / np.sqrt(6), None)
            M1.uz = ArrayWUnits(2 / np.sqrt(6), None)
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
            M1.ux = ArrayWUnits(1 / np.sqrt(6), None)
            M1.uy = ArrayWUnits(3 / np.sqrt(6), None)
            M1.uz = ArrayWUnits(2 / np.sqrt(6), None)
            M1.ra = ArrayWUnits(33.0, None)
            M1.dec = ArrayWUnits(decval, None)

            self.force_models[mode].maneuver = M1
            acc = self.force_models[mode].compute_acceleration(
                pos, full_state, epoch, body
            )
            return acc

        def wrap_GMaccel(accelval, pos, full_state, epoch, body):
            full_state[("a_fogm", body._spice_id)] = accelval
            acc = self.force_models[mode].compute_acceleration(
                pos, full_state, epoch, body
            )
            return acc

        def wrap_GMdriftaccel(accelval, pos, full_state, epoch, body):
            full_state[("a_fogm", body._spice_id)] = accelval
            a_dot = (
                self.force_models[mode]
                ._compute_acceleration_state_derivative(pos, full_state, epoch, body)
                .reshape(3)
            )

            return a_dot

        def wrap_beta(betaval, pos, full_state, epoch, body):
            self.force_models[mode].default_beta = betaval
            a_dot = (
                self.force_models[mode]
                ._compute_acceleration_state_derivative(pos, full_state, epoch, body)
                .reshape(3)
            )

            return a_dot

        def wrap_PGMaccel(accelval, pos, full_state, epoch, body):
            full_state[("a_pfogm", body._spice_id)] = accelval
            acc = self.force_models[mode].compute_acceleration(
                pos, full_state, epoch, body
            )
            return acc

        def wrap_PGMdriftaccel(accelval, pos, full_state, epoch, body):
            full_state[("a_pfogm", body._spice_id)] = accelval
            a_dot = (
                self.force_models[mode]
                ._compute_acceleration_state_derivative(pos, full_state, epoch, body)
                .reshape(3 * self.force_models[mode].n_batches)
            )

            return a_dot

        def wrap_Pbeta(betaval, pos, full_state, epoch, body):
            self.force_models[mode].default_beta = betaval
            a_dot = (
                self.force_models[mode]
                ._compute_acceleration_state_derivative(pos, full_state, epoch, body)
                .reshape(3 * self.force_models[mode].n_batches)
            )

            return a_dot

        for mode in self._dynamics_mode_list:
            # ------------------------------------------------------------------
            # Spherical harmonics runs its kernels in Rust, which Python autograd
            # cannot trace. Verify its position Jacobian against the Rust
            # dual-number automatic differentiation instead — available whenever
            # the Rust core is built, with or without UseAutoGrad.
            # ------------------------------------------------------------------
            if _is_sph_mode(mode):
                sh = self.force_models[mode]
                if getattr(sh, "_rust_core", None) is not None:
                    self._check_sph_partials(
                        sh, position, full_state, epoch, body, _write_line
                    )
                continue

            # Same for the N-plate SRP model: plate sums and attitude mapping
            # run in Rust, verified against its dual-number derivatives.
            if mode == "nPlateSRP":
                npl = self.force_models[mode]
                if getattr(npl, "_rust_cores", None) is not None:
                    self._check_nplate_partials(
                        npl, position, full_state, epoch, body, _write_line
                    )
                continue

            # Thermal and antenna recoil check their position, velocity and
            # scale-factor partials together, so the velocity block of the
            # Sun-pointed attitude is covered too.
            if mode in _EMISSION_SCALE_FACTORS.values():
                if os.environ.get("UseAutoGrad") == "TRUE":
                    self._check_emission_partials(
                        self.force_models[mode],
                        position,
                        full_state,
                        epoch,
                        body,
                        _write_line,
                    )
                else:
                    warnings.warn(
                        f"{mode} partials are not checked: they are verified "
                        "with autograd, which needs UseAutoGrad=TRUE."
                    )
                continue

            # ------------------------------------------------------------------
            # User-defined models own their verification: autograd cannot be
            # pointed at an arbitrary user callable safely, and the model
            # already knows which of its Jacobians are analytical (the rest are
            # finite differences, which there is no point comparing to
            # themselves).
            # ------------------------------------------------------------------
            if mode in self._custom_models:
                self._custom_models[mode]._check_partials(
                    position, full_state, epoch, body, _write_line
                )
                continue

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

                elif mode == "ThreeBodyGravity":
                    # GM of each attracting body, propagated or cached
                    tb = self.force_models[mode]
                    grad = jacobian(wrap_third_body_gm)
                    for name, sid, gm in tb._prop_bodies + tb._cached_bodies:
                        gmval = tb._scalar_from_state(full_state.get(("gm", sid), gm))
                        dacc_dtbgm = grad(gmval, sid, position, full_state, epoch, body)
                        partial_a_by_gm = tb._compute_partial_by_gm(
                            position.values, full_state, epoch, body, name
                        )
                        for dependent_dx, dependent in enumerate(["x", "y", "z"]):
                            analytical = partial_a_by_gm[dependent_dx][0]
                            algo = dacc_dtbgm[dependent_dx]
                            _write_line(mode, dependent, f"gm@{sid}", analytical, algo)

                elif mode == "CannonballSRP":
                    original_eta = copy.deepcopy(full_state[("eta_srp", body.spice_id)])
                    eta_srp = ArrayWUnits(original_eta, None)
                    grad = jacobian(wrap_eta)
                    dacc_deta = grad(
                        eta_srp.values, eta_srp.units, position, full_state, epoch, body
                    )
                    self.force_models[mode]._eta_srp_map[body.spice_id] = original_eta
                    full_state[("eta_srp", body.spice_id)] = original_eta

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

                    original_ra = copy.deepcopy(self.force_models[mode].maneuver.ra)

                    grad = jacobian(wrap_fbra)
                    dacc_dra = grad(
                        self.force_models[mode].maneuver.ra.values,
                        position,
                        full_state,
                        epoch,
                        body,
                    )
                    self.force_models[mode].maneuver.ra = original_ra

                    original_dec = copy.deepcopy(self.force_models[mode].maneuver.dec)

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
                    original_afogm = copy.deepcopy(
                        full_state[("a_fogm", body._spice_id)]
                    )
                    grad = jacobian(wrap_GMaccel)
                    dacc_dGMaccel = grad(
                        full_state[("a_fogm", body._spice_id)],
                        position,
                        full_state,
                        epoch,
                        body,
                    )
                    full_state[("a_fogm", body._spice_id)] = original_afogm

                    grad = jacobian(wrap_GMdriftaccel)
                    dacc_dGMdriftaccel = grad(
                        full_state[("a_fogm", body._spice_id)],
                        position,
                        full_state,
                        epoch,
                        body,
                    )
                    full_state[("a_fogm", body._spice_id)] = original_afogm

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
                    original_apfogm = copy.deepcopy(
                        full_state[("a_pfogm", body._spice_id)]
                    )
                    grad = jacobian(wrap_PGMaccel)
                    dacc_dPGMaccel = grad(
                        full_state[("a_pfogm", body._spice_id)],
                        position,
                        full_state,
                        epoch,
                        body,
                    )
                    full_state[("a_pfogm", body._spice_id)] = original_apfogm

                    grad = jacobian(wrap_PGMdriftaccel)
                    dacc_dPGMdriftaccel = grad(
                        full_state[("a_pfogm", body._spice_id)],
                        position,
                        full_state,
                        epoch,
                        body,
                    )
                    full_state[("a_pfogm", body._spice_id)] = original_apfogm

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

            else:
                # the comparisons below need the autograd reference values
                warnings.warn(
                    f"{mode} partials are not checked: they are verified with "
                    "autograd, which needs UseAutoGrad=TRUE."
                )
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

            elif mode == "FiniteBurn":
                partial_a_by_thrust = self.force_models[
                    mode
                ]._compute_partial_by_thrust(position.values, full_state, epoch, body)

                for independent_dx, independent in enumerate(["thrust"]):
                    for dependent_dx, dependent in enumerate(["x", "y", "z"]):
                        analytical = partial_a_by_thrust[dependent_dx]
                        algo = dacc_dthrust[dependent_dx]
                        _write_line(mode, dependent, independent, analytical, algo)

                partial_a_by_ra_dec = self.force_models[
                    mode
                ]._compute_partial_by_ra_dec_bias(
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
                partial_a_by_accel = self.force_models[
                    mode
                ]._compute_partial_by_acceleration_state(
                    position.values, full_state, epoch, body
                )

                for independent_dx, independent in enumerate(
                    ["afogm_x", "afogm_y", "afogm_z"]
                ):
                    for dependent_dx, dependent in enumerate(["x", "y", "z"]):
                        analytical = partial_a_by_accel[dependent_dx][independent_dx]
                        algo = dacc_dGMaccel[dependent_dx][independent_dx]
                        _write_line(mode, dependent, independent, analytical, algo)

                partial_a_by_accel = self.force_models[
                    mode
                ]._compute_partial_of_acceleration_state_by_acceleration_state(
                    position.values, full_state, epoch, body
                )

                for independent_dx, independent in enumerate(
                    ["afogm_x", "afogm_y", "afogm_z"]
                ):
                    for dependent_dx, dependent in enumerate(
                        ["afogm_x", "afogm_y", "afogm_z"]
                    ):
                        analytical = partial_a_by_accel[dependent_dx][independent_dx]
                        algo = dacc_dGMdriftaccel[dependent_dx][independent_dx]
                        _write_line(mode, dependent, independent, analytical, algo)

                partial_a_by_beta = self.force_models[
                    mode
                ]._compute_partial_of_acceleration_state_by_beta(
                    position.values, full_state, epoch, body
                )

                for independent_dx, independent in enumerate(
                    ["beta_afogm_x", "beta_afogm_y", "beta_afogm_z"]
                ):
                    for dependent_dx, dependent in enumerate(
                        ["afogm_x", "afogm_y", "afogm_z"]
                    ):
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
                partial_a_by_accel = self.force_models[
                    mode
                ]._compute_partial_by_acceleration_state(
                    position.values, full_state, epoch, body
                )

                for independent_dx, independent in enumerate(
                    ["apfogm_x", "apfogm_y", "apfogm_z"]
                ):
                    for dependent_dx, dependent in enumerate(["x", "y", "z"]):
                        analytical = partial_a_by_accel[dependent_dx][independent_dx]
                        algo = dacc_dPGMaccel[dependent_dx][independent_dx]
                        _write_line(mode, dependent, independent, analytical, algo)

                partial_a_by_accel = self.force_models[
                    mode
                ]._compute_partial_of_acceleration_state_by_acceleration_state(
                    position.values, full_state, epoch, body
                )

                for independent_dx, independent in enumerate(
                    ["apfogm_x", "apfogm_y", "apfogm_z"]
                ):
                    for dependent_dx, dependent in enumerate(
                        ["apfogm_x", "apfogm_y", "apfogm_z"]
                    ):
                        analytical = partial_a_by_accel[dependent_dx][independent_dx]
                        algo = dacc_dPGMdriftaccel[dependent_dx][independent_dx]
                        _write_line(mode, dependent, independent, analytical, algo)

                if (
                    self.force_models[mode]._get_active_batch_index(epoch)
                    < self.force_models[mode].n_batches
                ):
                    pass

                partial_a_by_beta = self.force_models[
                    mode
                ]._compute_partial_of_acceleration_state_by_beta(
                    position.values, full_state, epoch, body
                )

                for independent_dx, independent in enumerate(
                    ["beta_apfogm_x", "beta_apfogm_y", "beta_apfogm_z"]
                ):
                    for dependent_dx, dependent in enumerate(
                        ["apfogm_x", "apfogm_y", "apfogm_z"]
                    ):
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
