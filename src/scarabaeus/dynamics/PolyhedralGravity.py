# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    DynamicModel,
    Frame,
    CelestialBody,
    SpiceManager,
    ArrayWUnits,
    Units,
    Constants,
)

import scarabaeus.utils.NumpyWrapper as np

try:
    from scarabaeus.rust import PolyhedronGravityCore as _PolyhedronRustCore
except ImportError:
    _PolyhedronRustCore = None

kg, km, sec = Units.get_units(["kg", "km", "sec"])


#######################################################################
class PolyhedralGravity(DynamicModel):
    """Models the gravitational acceleration of a constant-density polyhedron.

    For a small, irregular body the shape *is* the gravity field: a
    spherical-harmonic expansion diverges inside the circumscribing sphere,
    which for an elongated asteroid is most of the space a spacecraft actually
    occupies. The polyhedral formulation of Werner & Scheeres [1]_ instead
    integrates the potential of a homogeneous polyhedron in closed form, one
    solid-angle term per triangular facet and one line term per edge:

    .. math::

        U = \\frac{1}{2} G \\rho \\sum_{e \\in edges}
                \\mathbf{r}_e \\cdot \\mathbf{E}_e \\mathbf{r}_e L_e
          - \\frac{1}{2} G \\rho \\sum_{f \\in facets}
                \\mathbf{r}_f \\cdot \\mathbf{F}_f \\mathbf{r}_f \\omega_f

    .. math::

        \\nabla U = -G \\rho \\sum_e \\mathbf{E}_e \\mathbf{r}_e L_e
                  + G \\rho \\sum_f \\mathbf{F}_f \\mathbf{r}_f \\omega_f

    .. math::

        \\nabla\\nabla U = G \\rho \\sum_e \\mathbf{E}_e L_e
                        - G \\rho \\sum_f \\mathbf{F}_f \\omega_f

    where :math:`L_e = \\ln\\frac{a+b+e}{a+b-e}` is the edge factor,
    :math:`\\omega_f` the signed solid angle the facet subtends at the field
    point, and :math:`\\mathbf{E}_e` / :math:`\\mathbf{F}_f` the edge and facet
    dyads. The result is exact everywhere outside the body — no truncation
    degree, no divergence — and :math:`\\nabla^2 U = -4\\pi G \\rho` inside it,
    which the model also exposes as a containment test.

    The summations run in Rust (:class:`~scarabaeus.rust.PolyhedronGravityCore`);
    Python keeps the shape bookkeeping, the body-fixed to inertial rotation and
    the estimated-density handling.

    Parameters
    ----------
    origin : CelestialBody
        Body the propagation is centred on, and the body the shape belongs to.
    base_frame : Frame
        Reference frame the propagator integrates in.
    shape : str or Path or object or np.ndarray
        The body's shape, in whichever form is at hand:

        * a path to a **Wavefront** ``.obj`` file, or to an ASCII
          **vertex/facet table** (``.tab``, ``.txt``, ``.dat``) of the kind
          the PDS distributes for small bodies;
        * any object exposing ``vertices`` and ``faces`` (a ``trimesh.Trimesh``
          and friends drop straight in);
        * a ``(vertices, faces)`` pair;
        * an ``(N, 3)`` vertex array, with ``faces`` given separately.
    faces : np.ndarray, optional
        ``(M, 3)`` zero-based vertex indices, one row per triangular facet,
        wound counter-clockwise as seen from outside the body. Only needed
        when ``shape`` is a bare vertex array.
    density : float or ArrayWUnits, optional
        Bulk density in kg/km^3. Supply this or ``gm``, not both.
    gm : float or ArrayWUnits, optional
        Gravitational parameter in km^3/s^2. The density follows from the mesh
        volume, which is the usual way a shape model is published: a measured
        GM plus a shape, with the density inferred.
    body_fixed_frame : str or Frame, optional
        Frame the vertices are expressed in. When given, positions are rotated
        into it through SPICE at each evaluation, so a rotating body is handled
        properly. Defaults to ``None``, meaning the mesh is already expressed in
        ``base_frame`` and no rotation is applied.
    scale : float, optional
        Multiplier taking the file's length unit to kilometers. Shape models
        are published in whatever unit the team preferred — pass ``1e-3`` for a
        mesh in meters. Defaults to ``1.0``.

    Raises
    ------
    ShapeModelError
        If the mesh is unusable, or if neither/both of ``density`` and ``gm``
        are supplied.
    RustCoreUnavailableError
        If the compiled extension is missing, or autograd tracing is active.

    See Also
    --------
    scarabaeus.SphericalHarmonicsGravity : Harmonic expansion, for bodies whose
        field is wanted outside the Brillouin sphere.
    scarabaeus.PointMassGravity : Single-body Newtonian gravity.

    Notes
    -----
    The mesh must be a closed, consistently wound surface: every edge is shared
    by exactly two facets. The constructor checks this and reports the offending
    edge, since a hole or a duplicated facet otherwise produces a plausible but
    wrong field.

    A negative :meth:`volume` means the facets are wound inwards; flip the last
    two indices of every facet.

    References
    ----------
    .. [1] Werner, R. A., & Scheeres, D. J. (1997). Exterior gravitation of a
       polyhedron derived and compared with harmonic and mascon gravitation
       representations of asteroid 4769 Castalia. *Celestial Mechanics and
       Dynamical Astronomy*, 65(3), 313-344.

    Examples
    --------
    .. code-block:: python

        # straight from a shape file
        gravity = PolyhedralGravity(eros, J2000, "eros_shape.obj", gm=4.463e-4)

        # from anything mesh-shaped already in memory
        gravity = PolyhedralGravity(eros, J2000, trimesh.load("eros.obj"),
                                    density=2670e9)

        # or from explicit arrays
        gravity = PolyhedralGravity(eros, J2000, vertices, faces, density=2670e9)
    """

    # ------------------#
    # region   Errors  #
    # ------------------#
    class ShapeModelError(ValueError):
        """Raised when the supplied shape model cannot define a gravity field."""

    class RustCoreUnavailableError(Exception):
        """Raised when the compiled polyhedral kernel is not available."""

    # endregion Errors #
    # ------------------#

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        origin: CelestialBody,
        base_frame: Frame,
        shape=None,
        faces: np.ndarray = None,
        density=None,
        gm=None,
        body_fixed_frame=None,
        scale: float = 1.0,
    ):
        self._origin = origin
        self._origin_name = getattr(origin, "name", None)
        self._origin_spice_id = getattr(origin, "spice_id", None)
        self._ref_frame = base_frame

        vertices, faces = self._resolve_shape(shape, faces, scale)
        self._shape_source = self._describe_shape(shape)

        self._vertices = np.array(vertices, dtype=float)
        self._faces = np.array(faces, dtype=np.int64)

        if self._vertices.ndim != 2 or self._vertices.shape[1] != 3:
            raise self.ShapeModelError(
                f"vertices must have shape (N, 3); received {self._vertices.shape}."
            )
        if self._faces.ndim != 2 or self._faces.shape[1] != 3:
            raise self.ShapeModelError(
                "faces must have shape (M, 3) — every facet is a triangle; "
                f"received {self._faces.shape}."
            )

        self._body_fixed_frame = self._resolve_frame_name(body_fixed_frame)

        # Density and GM are two views of the same number once the mesh fixes
        # the volume, so exactly one of them defines the field.
        if (density is None) == (gm is None):
            raise self.ShapeModelError(
                "supply exactly one of 'density' (kg/km^3) or 'gm' (km^3/s^2); "
                f"received density={density!r}, gm={gm!r}."
            )

        self._G = float(Constants.G.values)
        self._rust_core = self._build_rust_core(
            self._resolve_density(density, gm)
        )

    # endregion Constructor #
    # --------------------#

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def origin_name(self) -> str:
        """Name of the body the shape belongs to."""
        return self._origin_name

    @property
    def ref_frame(self) -> Frame:
        """Reference frame the acceleration is returned in."""
        return self._ref_frame

    @property
    def body_fixed_frame(self):
        """Frame the mesh is expressed in, or ``None`` when it is ``ref_frame``."""
        return self._body_fixed_frame

    @property
    def vertices(self) -> np.ndarray:
        """``(N, 3)`` body-fixed vertex coordinates in km."""
        return self._vertices

    @property
    def faces(self) -> np.ndarray:
        """``(M, 3)`` zero-based facet vertex indices."""
        return self._faces

    @property
    def density(self) -> float:
        """Bulk density in kg/km^3."""
        return self._require_rust_core().density

    @density.setter
    def density(self, input_val):
        """Set the bulk density, in kg/km^3."""
        self._require_rust_core().density = self._as_float(input_val, kg / km**3)

    @property
    def gm(self) -> float:
        """Gravitational parameter in km^3/s^2, from the density and mesh volume."""
        return self._G * self._require_rust_core().mass()

    @property
    def n_facets(self) -> int:
        """Number of triangular facets."""
        return self._require_rust_core().n_facets

    @property
    def n_edges(self) -> int:
        """Number of unique edges."""
        return self._require_rust_core().n_edges

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    def volume(self) -> float:
        """
        Enclosed volume of the shape, in km^3.

        Signed: a negative value means the facets are wound inwards, which is
        the cheapest check that a mesh follows the outward convention.

        Returns
        -------
        float
            Enclosed volume, km^3.
        """
        return self._require_rust_core().volume()

    def center_of_mass(self) -> np.ndarray:
        """
        Centre of mass of the shape, in its own frame, km.

        Worth checking on any shape model before trusting a comparison against
        a point mass: a mesh whose centroid is off the origin carries a dipole
        term, and its field will not approach ``GM/r^2`` about the origin at
        any distance.

        Returns
        -------
        np.ndarray
            3-vector centroid in km.
        """
        return np.array(self._require_rust_core().center_of_mass())

    def compute_potential(self, position: np.ndarray, epoch: float = None) -> float:
        """
        Gravitational potential at a field point.

        Parameters
        ----------
        position : np.ndarray
            3-vector in ``ref_frame``, km.
        epoch : float, optional
            Epoch used for the body-fixed rotation. Required only when
            ``body_fixed_frame`` is set.

        Returns
        -------
        float
            Potential in km^2/s^2, positive by the convention
            :math:`\\nabla U = \\mathbf{a}`.
        """
        r_bf = self._to_body_fixed(np.asarray(position, dtype=float).ravel(), epoch)
        return self._require_rust_core().compute_potential(r_bf)

    def contains(self, position: np.ndarray, epoch: float = None) -> bool:
        """
        Whether a point lies inside the body.

        Uses the summed solid angle, which is exact for a closed mesh: zero
        outside, :math:`4\\pi` inside.

        Parameters
        ----------
        position : np.ndarray
            3-vector in ``ref_frame``, km.
        epoch : float, optional
            Epoch used for the body-fixed rotation.

        Returns
        -------
        bool
            ``True`` when the point is inside the shape.
        """
        r_bf = self._to_body_fixed(np.asarray(position, dtype=float).ravel(), epoch)
        return self._require_rust_core().contains(r_bf)

    def compute_acceleration(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Computes the gravitational acceleration of the polyhedral body.

        Parameters
        ----------
        position : np.ndarray
            3x1 spacecraft position relative to the central body, in km,
            expressed in the reference frame.
        current_state : dict
            Dictionary containing the current state parameters. A
            ``("rho_poly", spice_id)`` entry overrides the bulk density, so the
            density can be estimated.
        epoch : float
            Current simulation epoch in seconds.
        body : CelestialBody
            Included for interface consistency; not used.

        Returns
        -------
        np.ndarray
            3x1 acceleration in km/s^2, in the reference frame.
        """
        core = self._require_rust_core()

        density_save = core.density
        self._apply_density_from_state(core, current_state)
        try:
            r_bf = self._to_body_fixed(np.asarray(position, dtype=float).ravel(), epoch)
            acc_bf = np.array(core.compute_acceleration(r_bf))
        finally:
            core.density = density_save

        return self._to_ref_frame(acc_bf, epoch)

    def _compute_partial_by_position(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Computes the Jacobian of the acceleration with respect to position.

        This is the gravity-gradient matrix :math:`\\nabla\\nabla U`, rotated
        into the reference frame as :math:`R\\,\\nabla\\nabla U\\,R^T`.

        Parameters
        ----------
        position : np.ndarray
            3x1 spacecraft position relative to the central body, in km.
        current_state : dict
            Dictionary containing the current state parameters.
        epoch : float
            Current simulation epoch in seconds.
        body : CelestialBody
            Included for interface consistency; not used.

        Returns
        -------
        np.ndarray
            3x3 Jacobian in 1/s^2, in the reference frame.
        """
        core = self._require_rust_core()

        density_save = core.density
        self._apply_density_from_state(core, current_state)
        try:
            r_bf = self._to_body_fixed(np.asarray(position, dtype=float).ravel(), epoch)
            jac_bf = np.array(core.compute_jacobian(r_bf)).reshape(3, 3)
        finally:
            core.density = density_save

        R = self._R_bf2rf(epoch)
        if R is None:
            return jac_bf
        return R @ jac_bf @ R.T

    def _compute_partial_by_rho_poly(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Computes the partial of the acceleration with respect to the density.

        The potential is linear in :math:`\\rho`, so the partial is just the
        acceleration divided by the density it was evaluated at — exact, with
        no extra field evaluation beyond the one call.

        Parameters
        ----------
        position : np.ndarray
            3x1 spacecraft position relative to the central body, in km.
        current_state : dict
            Dictionary containing the current state parameters.
        epoch : float
            Current simulation epoch in seconds.
        body : CelestialBody
            Included for interface consistency; not used.

        Returns
        -------
        np.ndarray
            3x1 partial in km/s^2 per kg/km^3.
        """
        core = self._require_rust_core()
        density = self._density_from_state(current_state, default=core.density)
        if density == 0.0:
            raise self.ShapeModelError(
                "cannot take the density partial at zero density."
            )

        acc = self.compute_acceleration(position, current_state, epoch, body)
        return acc / density

    # endregion Methods #
    # ----------------#

    # ------------------#
    # region Internals #
    # ------------------#
    @classmethod
    def _resolve_shape(cls, shape, faces, scale):
        """
        Reduce whatever was handed in to a ``(vertices, faces)`` pair.

        A shape arrives in whatever form the caller already has: a path to a
        published shape file, a mesh object from another library, a pair of
        arrays, or a bare vertex array with the facets alongside. All four
        reduce to the same two arrays here, so the rest of the model never has
        to care which it was.

        Parameters
        ----------
        shape : str or Path or object or array_like or None
            The shape in any supported form.
        faces : array_like or None
            Facet indices, when ``shape`` carries only vertices.
        scale : float
            Multiplier taking the shape's length unit to kilometers.

        Returns
        -------
        vertices, faces : np.ndarray
            ``(N, 3)`` coordinates in km and ``(M, 3)`` zero-based indices.

        Raises
        ------
        ShapeModelError
            If nothing usable was supplied.
        """
        if shape is None:
            raise cls.ShapeModelError(
                "no shape supplied: pass a shape file, a mesh object, a "
                "(vertices, faces) pair, or a vertex array with 'faces'."
            )

        # a path to a shape file
        if isinstance(shape, (str, bytes)) or hasattr(shape, "__fspath__"):
            vertices, faces = cls._read_shape_file(shape)

        # a mesh object from another library (trimesh and friends)
        elif hasattr(shape, "vertices") and hasattr(shape, "faces"):
            vertices, faces = np.asarray(shape.vertices), np.asarray(shape.faces)

        # a (vertices, faces) pair
        elif isinstance(shape, (tuple, list)) and len(shape) == 2:
            vertices, faces = np.asarray(shape[0]), np.asarray(shape[1])

        # a bare vertex array, with the facets given separately
        else:
            if faces is None:
                raise cls.ShapeModelError(
                    "a bare vertex array needs its facets: pass 'faces' as well, "
                    "or hand over a shape file or mesh object instead."
                )
            vertices = np.asarray(shape)

        vertices = np.asarray(vertices, dtype=float)
        if scale != 1.0:
            vertices = vertices * float(scale)
        return vertices, np.asarray(faces)

    @classmethod
    def _read_shape_file(cls, path):
        """
        Read a shape model off disk, dispatching on the file extension.

        Parameters
        ----------
        path : str or Path
            Shape file to read.

        Returns
        -------
        vertices, faces : np.ndarray

        Raises
        ------
        ShapeModelError
            If the file is missing, empty, or in an unsupported format.
        """
        import os

        path = os.fspath(path)
        if not os.path.exists(path):
            raise cls.ShapeModelError(f"shape file not found: '{path}'.")

        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".obj":
            vertices, faces = cls._read_obj(path)
        elif suffix in (".tab", ".txt", ".dat", ""):
            vertices, faces = cls._read_vertex_facet_table(path)
        else:
            raise cls.ShapeModelError(
                f"unsupported shape format '{suffix}' for '{path}'. Supported: "
                "'.obj' (Wavefront) and '.tab'/'.txt'/'.dat' (vertex/facet "
                "table). For anything else, load it with a mesh library and "
                "pass the object, or pass (vertices, faces) directly."
            )

        if len(vertices) == 0 or len(faces) == 0:
            raise cls.ShapeModelError(
                f"'{path}' yielded {len(vertices)} vertices and {len(faces)} "
                "facets; it does not look like a shape model."
            )
        return vertices, faces

    @classmethod
    def _read_obj(cls, path):
        """
        Read a Wavefront ``.obj`` mesh.

        Only ``v`` and ``f`` records matter for gravity; normals, texture
        coordinates, groups and materials are ignored. Face records may carry
        ``v/vt/vn`` triplets, indices may be negative (relative to the end of
        the vertex list, as the format allows), and faces with more than three
        corners are fan-triangulated.

        Parameters
        ----------
        path : str
            File to read.

        Returns
        -------
        vertices, faces : np.ndarray
        """
        vertices = []
        faces = []

        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line_number, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                tag, _, rest = line.partition(" ")
                if tag == "v":
                    parts = rest.split()
                    if len(parts) < 3:
                        raise cls.ShapeModelError(
                            f"{path}:{line_number}: vertex needs three "
                            f"coordinates, found {len(parts)}."
                        )
                    vertices.append([float(x) for x in parts[:3]])

                elif tag == "f":
                    corners = []
                    for token in rest.split():
                        # 'v', 'v/vt', 'v//vn' and 'v/vt/vn' all start with v
                        index = int(token.split("/")[0])
                        # OBJ is 1-based, and negative indices count back from
                        # the vertices seen so far
                        corners.append(
                            len(vertices) + index if index < 0 else index - 1
                        )
                    if len(corners) < 3:
                        raise cls.ShapeModelError(
                            f"{path}:{line_number}: facet needs at least three "
                            f"corners, found {len(corners)}."
                        )
                    # fan-triangulate anything with more than three corners
                    for k in range(1, len(corners) - 1):
                        faces.append([corners[0], corners[k], corners[k + 1]])

        return np.array(vertices, dtype=float), np.array(faces, dtype=np.int64)

    @classmethod
    def _read_vertex_facet_table(cls, path):
        """
        Read an ASCII vertex/facet table.

        The layout small-body shape models are usually distributed in: a header
        line giving the vertex and facet counts, then one line per vertex and
        one per facet. Both blocks may carry a leading running index, which is
        detected from the column count rather than assumed.

        Parameters
        ----------
        path : str
            File to read.

        Returns
        -------
        vertices, faces : np.ndarray

        Raises
        ------
        ShapeModelError
            If the header or the row counts do not line up.
        """
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            rows = [
                line.split()
                for line in handle
                if line.strip() and not line.lstrip().startswith(("#", "!"))
            ]

        if not rows or len(rows[0]) != 2:
            raise cls.ShapeModelError(
                f"'{path}': expected a header line with the vertex and facet "
                "counts, e.g. '1708 3412'."
            )

        n_vertices, n_facets = (int(x) for x in rows[0])
        if len(rows) < 1 + n_vertices + n_facets:
            raise cls.ShapeModelError(
                f"'{path}': header declares {n_vertices} vertices and "
                f"{n_facets} facets, but the file holds {len(rows) - 1} rows."
            )

        vertex_rows = rows[1 : 1 + n_vertices]
        facet_rows = rows[1 + n_vertices : 1 + n_vertices + n_facets]

        # a leading running index is optional in both blocks
        vertices = np.array(
            [[float(x) for x in row[-3:]] for row in vertex_rows], dtype=float
        )
        faces = np.array(
            [[int(x) for x in row[-3:]] for row in facet_rows], dtype=np.int64
        )

        # these tables are 1-based; only shift when nothing already sits at 0
        if faces.min() == 1:
            faces = faces - 1

        return vertices, faces

    @staticmethod
    def _describe_shape(shape) -> str:
        """
        Short label for where the mesh came from, for repr and errors.

        Parameters
        ----------
        shape : object
            Whatever was passed as the shape.

        Returns
        -------
        str
            A human-readable origin, e.g. a file name or 'arrays'.
        """
        if isinstance(shape, (str, bytes)) or hasattr(shape, "__fspath__"):
            import os

            return os.path.basename(os.fspath(shape))
        if hasattr(shape, "vertices") and hasattr(shape, "faces"):
            return type(shape).__name__
        return "arrays"

    def _build_rust_core(self, density: float):
        """
        Build the Rust kernel, or ``None`` when it cannot be used.

        Parameters
        ----------
        density : float
            Bulk density, kg/km^3.

        Returns
        -------
        PolyhedronGravityCore or None
            The kernel, or ``None`` when the extension is missing or autograd
            is tracing (the compute methods then raise).
        """
        if _PolyhedronRustCore is None or np.autograd_types:
            return None
        try:
            return _PolyhedronRustCore(
                self._vertices, self._faces, density, self._G
            )
        except ValueError as exc:
            raise self.ShapeModelError(str(exc)) from exc

    def _require_rust_core(self):
        """
        Return the Rust kernel, raising if it is unavailable.

        Returns
        -------
        PolyhedronGravityCore

        Raises
        ------
        RustCoreUnavailableError
            If the extension is missing or autograd tracing is active.
        """
        if self._rust_core is None:
            raise self.RustCoreUnavailableError(
                "PolyhedralGravity requires the compiled Scarabaeus Rust extension, "
                "and it cannot run under autograd tracing (UseAutoGrad=TRUE). "
                "Build it with 'maturin develop --release' from the repository root."
            )
        return self._rust_core

    def _resolve_density(self, density, gm) -> float:
        """
        Reduce the density/GM pair to a density in kg/km^3.

        Parameters
        ----------
        density : float or ArrayWUnits or None
            Bulk density.
        gm : float or ArrayWUnits or None
            Gravitational parameter; converted using the mesh volume.

        Returns
        -------
        float
            Bulk density, kg/km^3.
        """
        if density is not None:
            return self._as_float(density, kg / km**3)

        # GM -> density needs the volume, which needs a built core; build a
        # unit-density one first and rescale.
        gm_value = self._as_float(gm, km**3 / sec**2)
        probe = self._build_rust_core(1.0)
        if probe is None:
            # No kernel to measure the volume with; the compute methods will
            # raise on first use, so any placeholder does.
            return 0.0

        volume = probe.volume()
        if volume <= 0.0:
            raise self.ShapeModelError(
                f"mesh encloses a non-positive volume ({volume:.6g} km^3): the "
                "facets are likely wound inwards. Flip the last two indices of "
                "every facet."
            )
        return gm_value / (self._G * volume)

    def _apply_density_from_state(self, core, current_state: dict) -> None:
        """
        Override the kernel's density from the state vector, when estimated.

        Parameters
        ----------
        core : PolyhedronGravityCore
            Kernel to update in place.
        current_state : dict
            Current state parameters.
        """
        density = self._density_from_state(current_state, default=None)
        if density is not None:
            core.density = density

    def _density_from_state(self, current_state: dict, default=None):
        """
        Read an estimated ``rho_poly`` out of the state vector.

        Parameters
        ----------
        current_state : dict
            Current state parameters.
        default : optional
            Value returned when the state carries no density.

        Returns
        -------
        float or None
            The estimated density in kg/km^3, or ``default``.
        """
        if not current_state or self._origin_spice_id is None:
            return default
        value = current_state.get(("rho_poly", self._origin_spice_id))
        if value is None:
            return default
        return float(getattr(value, "values", value))

    def _to_body_fixed(self, position: np.ndarray, epoch) -> np.ndarray:
        """
        Rotate a reference-frame position into the mesh's frame.

        Parameters
        ----------
        position : np.ndarray
            3-vector in the reference frame, km.
        epoch : float or None
            Epoch of the rotation.

        Returns
        -------
        np.ndarray
            3-vector in the body-fixed frame, km.
        """
        R = self._R_bf2rf(epoch)
        if R is None:
            return position
        return R.T @ position

    def _to_ref_frame(self, vector_bf: np.ndarray, epoch) -> np.ndarray:
        """
        Rotate a body-fixed vector back into the reference frame.

        Parameters
        ----------
        vector_bf : np.ndarray
            3-vector in the body-fixed frame.
        epoch : float or None
            Epoch of the rotation.

        Returns
        -------
        np.ndarray
            3-vector in the reference frame.
        """
        R = self._R_bf2rf(epoch)
        if R is None:
            return vector_bf
        return R @ vector_bf

    def _R_bf2rf(self, epoch):
        """
        Body-fixed to reference-frame rotation, memoised per epoch.

        ``compute_acceleration`` and the partials are called at the same epoch
        within one RHS step, so the SPICE lookup happens once per step.

        Parameters
        ----------
        epoch : float or None
            Epoch of the rotation.

        Returns
        -------
        np.ndarray or None
            3x3 rotation, or ``None`` when the mesh is already expressed in the
            reference frame.
        """
        if self._body_fixed_frame is None:
            return None
        if epoch is None:
            raise self.ShapeModelError(
                "an epoch is required to rotate out of "
                f"'{self._body_fixed_frame}'; the mesh is not expressed in the "
                "propagation frame."
            )

        cache = self.__dict__.get("_R_cache")
        if cache is not None and cache[0] == epoch:
            return cache[1]

        R = SpiceManager.get_xfrm(
            self._body_fixed_frame, self._resolve_frame_name(self._ref_frame), epoch
        )
        self._R_cache = (epoch, R)
        return R

    @staticmethod
    def _resolve_frame_name(frame):
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
        return getattr(frame, "name", frame)

    @staticmethod
    def _as_float(value, target_units) -> float:
        """
        Convert a possibly unit-carrying quantity to a float in target units.

        Parameters
        ----------
        value : float or ArrayWUnits
            Quantity to convert.
        target_units : Units
            Units the returned number is expressed in.

        Returns
        -------
        float
            The value in ``target_units``.
        """
        if isinstance(value, ArrayWUnits):
            return float(
                ArrayWUnits.get_value_in_target_units(value, target_units)
            )
        return float(value)

    def __getstate__(self):
        """Copy the attributes, leaving out the Rust kernel.

        ``copy.deepcopy`` and ``pickle`` both call this. ``SolutionOD``
        deep-copies the propagator, and so this model, in
        ``_integrate_reference`` and ``propagate_covariance``. The compiled
        kernel cannot be copied, so it is dropped here, with its density kept
        as ``_density_snapshot``, and rebuilt in :meth:`__setstate__`. The
        cached body-fixed rotation ``_R_cache`` is dropped too and recomputed
        at the next evaluation.
        """
        state = self.__dict__.copy()
        state["_rust_core"] = None
        state["_density_snapshot"] = (
            self._rust_core.density if self._rust_core is not None else None
        )
        state.pop("_R_cache", None)
        return state

    def __setstate__(self, state):
        """Restore the copied attributes and rebuild the Rust kernel with
        the saved density."""
        density = state.pop("_density_snapshot", None)
        self.__dict__.update(state)
        self._rust_core = (
            self._build_rust_core(density) if density is not None else None
        )

    def __repr__(self):
        """Concise summary of the shape and its density."""
        core = self._rust_core
        if core is None:
            return (
                f"PolyhedralGravity | {self._origin_name} | {self._shape_source} | "
                f"{len(self._faces)} facets | kernel unavailable"
            )
        return (
            f"PolyhedralGravity | {self._origin_name} | {self._shape_source} | "
            f"{core.n_facets} facets | {core.n_edges} edges | "
            f"rho: {core.density:.6g} kg/km^3 | GM: {self.gm:.6g} km^3/s^2"
        )

    # endregion Internals #
    # ------------------#
