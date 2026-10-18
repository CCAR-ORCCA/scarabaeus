# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import json
import os

import scarabaeus.utils.NumpyWrapper as np
from scarabaeus import ArrayWUnits, Units, Constants

try:
    from scarabaeus.rust import PolyhedronHarmonics as _PolyhedronHarmonics
except ImportError:
    _PolyhedronHarmonics = None

kg, km, sec = Units.get_units(["kg", "km", "sec"])


# --------------------#
#  Class Definition  #
# --------------------#
class ShapeHarmonics:
    r"""Convert a polyhedral shape model into spherical harmonic coefficients.

    A shape model and a harmonic expansion are two descriptions of the same
    field, each convenient somewhere different: the polyhedron is exact right
    down to the surface but costs one term per facet, while the expansion is
    cheap and truncatable but only converges outside the sphere that
    circumscribes the body. Turning the first into the second is what lets a
    shape model be flown with :class:`~scarabaeus.SphericalHarmonicsGravity`.

    The coefficients follow Werner's exact recursion over the tetrahedra formed
    by each facet with the origin, so for a constant-density polyhedron they are
    exact at every degree — the only error is the truncation you choose and the
    fidelity of the mesh itself.

    For a body of constant density and volume :math:`V`, the fully normalized
    coefficients of reference radius :math:`R` are [1]_

    .. math::

        \bar{C}_{nm} + i\,\bar{S}_{nm} = \frac{1}{(2n+1)\,V} \int_V
            \left(\frac{r}{R}\right)^{n} \bar{P}_{nm}(\sin\phi)\,
            e^{i m \lambda}\, dV.

    The integrand is a homogeneous polynomial of degree :math:`n` in
    :math:`(x, y, z)/R`, built by a recursion on :math:`(n, m)` (see
    :class:`scarabaeus.rust.PolyhedronHarmonics`). The polyhedron is split into
    the tetrahedra :math:`(\mathbf{0}, \mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2)`
    that each facet forms with the origin. Inside one,
    :math:`\mathbf{r} = \alpha\mathbf{v}_0 + \beta\mathbf{v}_1 + \gamma\mathbf{v}_2`,
    so the integral becomes a sum of monomials in :math:`(\alpha, \beta, \gamma)`.
    Each of these integrates exactly over the unit simplex:

    .. math::

        \int_V f\, dV = \sum_{\text{facets}} \det[\mathbf{v}_0\ \mathbf{v}_1\ \mathbf{v}_2]
            \int_{\alpha+\beta+\gamma \le 1} f\, d\alpha\, d\beta\, d\gamma,
        \qquad
        \int_{\alpha+\beta+\gamma \le 1} \alpha^i \beta^j \gamma^k\,
            d\alpha\, d\beta\, d\gamma = \frac{i!\, j!\, k!}{(i+j+k+3)!}.

    The output is written in the JSON layout Scarabaeus reads through
    ``sph_harm_cs_file``, so a shape file goes to a flyable gravity model in one
    call.

    See Also
    --------
    scarabaeus.PolyhedralGravity : The exact shape-model field.
    scarabaeus.SphericalHarmonicsGravity : Consumer of the file written here.

    Notes
    -----
    Coefficients are normalised so that :math:`C_{00} = 1`, which is what the
    loader expects, and the mesh is centred on its centre of mass by default so
    the degree-1 terms vanish as they should.

    A harmonic expansion **diverges inside the Brillouin sphere** — the sphere
    of radius ``ref_radius`` about the origin. For an elongated body that is a
    large region, and it is exactly where the polyhedral model should be used
    instead. The reference radius defaults to the circumscribing radius so the
    boundary is explicit in the file.

    References
    ----------
    .. [1] Werner, R. A. (1997). Spherical harmonic coefficients for the
       potential of a constant-density polyhedron. *Computers & Geosciences*,
       23(10), 1071-1077.

    Examples
    --------
    .. code-block:: python

        # shape file in, coefficient file out
        ShapeHarmonics.to_coefficient_file(
            "bennu.obj", "bennu_deg16.json", degree=16,
            gm=4.891e-9, body_name="Bennu",
        )

        # then fly it
        force_model = ForceModelTranslation(
            primary_body=sc, sph_harm=True, sph_harm_body=bennu,
            sph_harm_order=16, sph_harm_cs_file="bennu_deg16.json",
            sph_harm_norm_flag=True,
        )
    """

    # ------------------#
    # region   Errors  #
    # ------------------#
    class ShapeHarmonicsError(ValueError):
        """Raised when a shape cannot be reduced to harmonic coefficients."""

    class RustCoreUnavailableError(Exception):
        """Raised when the compiled harmonics kernel is not available."""

    # endregion Errors #
    # ------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    @classmethod
    def from_shape(
        cls,
        shape,
        degree: int,
        faces=None,
        ref_radius=None,
        normalized: bool = True,
        center: bool = True,
        scale: float = 1.0,
    ):
        """
        Compute the harmonic coefficients of a constant-density shape model.

        Parameters
        ----------
        shape : str or Path or object or np.ndarray
            The body's shape, in any form
            :class:`~scarabaeus.PolyhedralGravity` accepts: a Wavefront
            ``.obj`` or ASCII vertex/facet table path, a mesh object exposing
            ``vertices``/``faces``, a ``(vertices, faces)`` pair, or a vertex
            array with ``faces`` supplied separately.
        degree : int
            Maximum degree and order to compute.
        faces : np.ndarray, optional
            Facet indices, when ``shape`` is a bare vertex array.
        ref_radius : float or ArrayWUnits, optional
            Normalising radius in km. Defaults to the circumscribing radius of
            the mesh, which is the Brillouin sphere the expansion is valid
            outside of.
        normalized : bool, optional
            Fully normalised coefficients when ``True`` (the default and what
            Scarabaeus prefers), raw when ``False``.
        center : bool, optional
            Shift the mesh onto its centre of mass first. Defaults to ``True``;
            without it the degree-1 terms carry the offset instead of vanishing.
        scale : float, optional
            Multiplier taking the shape's length unit to km.

        Returns
        -------
        Cnm, Snm : np.ndarray
            ``(degree + 1, degree + 1)`` coefficient arrays, normalised to
            ``C[0, 0] = 1``; only entries with ``n >= m`` are populated.
        ref_radius : float
            The reference radius the coefficients are expressed against, km.

        Raises
        ------
        ShapeHarmonicsError
            If the degree is negative or the mesh encloses no volume.
        RustCoreUnavailableError
            If the compiled extension is missing.
        """
        if _PolyhedronHarmonics is None:
            raise cls.RustCoreUnavailableError(
                "ShapeHarmonics requires the compiled Scarabaeus Rust extension. "
                "Build it with 'maturin develop --release' from the repository root."
            )
        if degree < 0:
            raise cls.ShapeHarmonicsError(f"degree must be >= 0; received {degree}.")

        vertices, faces = cls._resolve_shape(shape, faces, scale)

        if center:
            vertices = vertices - cls.center_of_mass(vertices, faces)

        if ref_radius is None:
            ref_radius = float(np.max(np.linalg.norm(vertices, axis=1)))
        else:
            ref_radius = cls._as_float(ref_radius, km)
        if ref_radius <= 0.0:
            raise cls.ShapeHarmonicsError(
                f"ref_radius must be positive; received {ref_radius}."
            )

        flat_c, flat_s = _PolyhedronHarmonics(
            np.ascontiguousarray(vertices, dtype=float),
            np.ascontiguousarray(faces, dtype=np.int64),
        ).compute(int(degree), float(ref_radius), bool(normalized))
        side = degree + 1
        Cnm = np.array(flat_c).reshape(side, side)
        Snm = np.array(flat_s).reshape(side, side)

        # The recursion carries the body's volume through every term, so the
        # zeroth coefficient is that volume and dividing by it gives the
        # conventional C00 = 1.
        volume = Cnm[0, 0]
        if volume == 0.0:
            raise cls.ShapeHarmonicsError(
                "the mesh encloses no volume, so the coefficients cannot be "
                "normalised. Check that it is closed and consistently wound."
            )
        if volume < 0.0:
            raise cls.ShapeHarmonicsError(
                f"the mesh encloses a negative volume ({volume:.6g} km^3): its "
                "facets are wound inwards. Flip the last two indices of every "
                "facet."
            )

        return Cnm / volume, Snm / volume, ref_radius

    @classmethod
    def to_coefficient_file(
        cls,
        shape,
        path: str,
        degree: int,
        gm=None,
        density=None,
        faces=None,
        ref_radius=None,
        normalized: bool = True,
        center: bool = True,
        scale: float = 1.0,
        body_name: str = None,
        source: str = None,
    ) -> dict:
        """
        Write a shape model's harmonic coefficients as a Scarabaeus file.

        The one call that takes a published shape model to something
        :class:`~scarabaeus.SphericalHarmonicsGravity` can load.

        Parameters
        ----------
        shape : str or Path or object or np.ndarray
            The body's shape, in any form :meth:`from_shape` accepts.
        path : str
            Destination ``.json`` file.
        degree : int
            Maximum degree and order to write.
        gm : float or ArrayWUnits, optional
            Gravitational parameter in km^3/s^2. Supply this or ``density``:
            the coefficients are scale-free, but the file records the field
            strength the model will be flown with.
        density : float or ArrayWUnits, optional
            Bulk density in kg/km^3, from which GM follows via the mesh volume.
        faces : np.ndarray, optional
            Facet indices, when ``shape`` is a bare vertex array.
        ref_radius : float or ArrayWUnits, optional
            Reference radius in km; defaults to the circumscribing radius.
        normalized : bool, optional
            Write fully normalised coefficients. Defaults to ``True``.
        center : bool, optional
            Centre the mesh on its centre of mass first. Defaults to ``True``.
        scale : float, optional
            Multiplier taking the shape's length unit to km.
        body_name : str, optional
            Name recorded in the file. Defaults to the shape file's stem.
        source : str, optional
            Provenance note recorded in the file.

        Returns
        -------
        dict
            The record written, so the caller can inspect it without re-reading
            the file.

        Raises
        ------
        ShapeHarmonicsError
            If neither or both of ``gm`` and ``density`` are supplied, or the
            shape cannot be reduced to coefficients.
        """
        if (gm is None) == (density is None):
            raise cls.ShapeHarmonicsError(
                "supply exactly one of 'gm' (km^3/s^2) or 'density' (kg/km^3); "
                f"received gm={gm!r}, density={density!r}."
            )

        vertices, resolved_faces = cls._resolve_shape(shape, faces, scale)
        if center:
            vertices = vertices - cls.center_of_mass(vertices, resolved_faces)

        Cnm, Snm, radius = cls.from_shape(
            (vertices, resolved_faces),
            degree,
            ref_radius=ref_radius,
            normalized=normalized,
            center=False,  # already centred above, so the volume matches
            scale=1.0,
        )

        if gm is not None:
            gm_value = cls._as_float(gm, km**3 / sec**2)
        else:
            volume = cls.volume(vertices, resolved_faces)
            gm_value = (
                float(Constants.G.values) * cls._as_float(density, kg / km**3) * volume
            )

        if body_name is None:
            body_name = cls._describe_shape(shape)

        record = {
            "BodyName": body_name,
            "ReferenceRadius": radius,
            "GravitationalParameter": gm_value,
            "Normalized": bool(normalized),
            "MaxDegree": int(degree),
            "Coefficients": [
                {
                    "degree": int(n),
                    "order": int(m),
                    "Cnm": float(Cnm[n, m]),
                    "Snm": float(Snm[n, m]),
                }
                for n in range(degree + 1)
                for m in range(n + 1)
            ],
            "source": [
                {
                    "srs": source
                    or (
                        "Constant-density polyhedron, Werner (1997) recursion, "
                        f"from {cls._describe_shape(shape)} "
                        f"({len(resolved_faces)} facets)"
                    )
                }
            ],
        }

        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(record, handle, indent=2)

        return record

    @staticmethod
    def volume(vertices, faces) -> float:
        """
        Signed volume enclosed by a mesh, km^3.

        Parameters
        ----------
        vertices : np.ndarray
            ``(N, 3)`` coordinates.
        faces : np.ndarray
            ``(M, 3)`` facet indices.

        Returns
        -------
        float
            Enclosed volume; negative when the facets are wound inwards.
        """
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=int)
        a = vertices[faces[:, 0]]
        b = vertices[faces[:, 1]]
        c = vertices[faces[:, 2]]
        return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)

    @classmethod
    def center_of_mass(cls, vertices, faces) -> np.ndarray:
        """
        Centre of mass of a constant-density mesh, km.

        Parameters
        ----------
        vertices : np.ndarray
            ``(N, 3)`` coordinates.
        faces : np.ndarray
            ``(M, 3)`` facet indices.

        Returns
        -------
        np.ndarray
            3-vector centroid.

        Raises
        ------
        ShapeHarmonicsError
            If the mesh encloses no volume.
        """
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=int)
        a = vertices[faces[:, 0]]
        b = vertices[faces[:, 1]]
        c = vertices[faces[:, 2]]

        tetra_volumes = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
        total = float(np.sum(tetra_volumes))
        if total == 0.0:
            raise cls.ShapeHarmonicsError(
                "the mesh encloses no volume, so it has no centre of mass."
            )

        centroids = (a + b + c) / 4.0  # the fourth corner is the origin
        return np.sum(tetra_volumes[:, None] * centroids, axis=0) / total

    @staticmethod
    def degree_power(Cnm, Snm) -> np.ndarray:
        """
        RMS power per degree, the usual way to pick a truncation.

        .. math::

            P_n = \\sqrt{\\sum_{m=0}^{n} \\left(\\bar C_{nm}^2 +
                  \\bar S_{nm}^2\\right)}

        Plotting this against degree shows where the field stops carrying
        information, which is the degree worth keeping.

        Parameters
        ----------
        Cnm, Snm : np.ndarray
            Coefficient arrays.

        Returns
        -------
        np.ndarray
            ``P_n`` for each degree.
        """
        Cnm = np.asarray(Cnm, dtype=float)
        Snm = np.asarray(Snm, dtype=float)
        return np.sqrt(np.sum(Cnm**2 + Snm**2, axis=1))

    # endregion Methods #
    # ----------------#

    # ------------------#
    # region Internals #
    # ------------------#
    @classmethod
    def _resolve_shape(cls, shape, faces, scale):
        """
        Reduce a shape in any accepted form to ``(vertices, faces)``.

        Delegates to :class:`~scarabaeus.PolyhedralGravity` so both entry
        points read exactly the same files and mesh objects.

        Parameters
        ----------
        shape : object
            Shape in any supported form.
        faces : array_like or None
            Facet indices, when ``shape`` carries only vertices.
        scale : float
            Multiplier taking the shape's length unit to km.

        Returns
        -------
        vertices, faces : np.ndarray
        """
        from scarabaeus import PolyhedralGravity

        try:
            return PolyhedralGravity._resolve_shape(shape, faces, scale)
        except PolyhedralGravity.ShapeModelError as exc:
            raise cls.ShapeHarmonicsError(str(exc)) from exc

    @staticmethod
    def _describe_shape(shape) -> str:
        """
        Short label for where a mesh came from, for the file's provenance.

        Parameters
        ----------
        shape : object
            Whatever was passed as the shape.

        Returns
        -------
        str
            A file stem, a type name, or 'arrays'.
        """
        if isinstance(shape, (str, bytes)) or hasattr(shape, "__fspath__"):
            return os.path.splitext(os.path.basename(os.fspath(shape)))[0]
        if hasattr(shape, "vertices") and hasattr(shape, "faces"):
            return type(shape).__name__
        return "arrays"

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
        """
        if isinstance(value, ArrayWUnits):
            return float(ArrayWUnits.get_value_in_target_units(value, target_units))
        return float(value)

    # endregion Internals #
    # ------------------#
