# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import scarabaeus.utils.NumpyWrapper as np 
import numpy.typing as npt

# class definition
class OrbitalElements:
    """ Performs conversions between Cartesian state vectors and classical orbital elements.

        Provides static methods to convert between position/velocity vectors
        ``(r, v)`` and Keplerian orbital elements :math:`(p, e, i, \\Omega, \\omega, \\nu)`, as
        well as related quantities such as mean motion, period, and eccentric
        anomaly.

        Two element sets appear here. :meth:`rv2coe` returns the osculating set
        built on the semi-latus rectum and the true anomaly. Everything to do
        with partials — :meth:`coe2rv`, :meth:`rv2mean_elements`, the two
        Jacobians and :meth:`covariance_to_elements` — uses Montenbruck &
        Gill's :math:`(a, e, i, \\Omega, \\omega, M)` instead, listed in
        :attr:`MEAN_ELEMENT_NAMES`, because that is the set their closed-form
        partials are written for. The two describe the same orbit; only the
        parameterisation differs.

        See Also
        --------
        scarabaeus.SolutionOD.state_covariance_to_elements : Applies the covariance transformation to an OD solution.

        References
        ----------
        .. [1] Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications
           (4th ed.). Microcosm Press. ISBN 978-1-881883-18-0.
        .. [2] Montenbruck, O., & Gill, E. (2000). Satellite Orbits: Models, Methods
           and Applications. Springer. ISBN 978-3-540-67280-7.
    """
    def rv2coe(self, grav_param : float, r : npt.NDArray, v : npt.NDArray, threshHold : float = 1e-8):
        """ Convert state vector to classical orbital elements.

            Parameters
            ----------
            grav_param : float
                Standard gravitational parameter in :math:`\\frac{km^3}{s^2}`.

            r : np.ndarray
                Position vector in kilometers.

            v : np.ndarray
                Velocity vector in kilometers per second.

            tol : float, optional
                Tolerance for eccentricity and inclination checks. Defaults to ``1e-8``.

            Returns
            -------
            p : float
                Semi-latus rectum of parameter in kilometers.

            ecc: float
                Eccentricity.

            inc: float
                Inclination in radians.

            raan: float
                Right ascension of the ascending node in radians.

            argp: float
                Argument of Perigee in radians.

            nu: float
                True Anomaly in radians.
            
            Notes
            -----
            All angles are in radians; distances in km; velocities in km/s
        """
        h   = np.cross(r, v)
        n   = np.cross([0, 0, 1], h)
        e   = ((v @ v - grav_param / np.linalg.norm(r)) * r - (r @ v) * v) / grav_param
        ecc = np.linalg.norm(e)
        p   = (h @ h) / grav_param
        inc = np.arccos(h[2] / np.linalg.norm(h))

        circular = ecc < threshHold
        equatorial = abs(inc) < threshHold

        if equatorial and not circular:
            raan = 0
            argp = np.arctan2(e[1], e[0]) % (2 * np.pi)  # Longitude of periapsis
            nu = np.arctan2((h @ np.cross(e, r)) / np.linalg.norm(h), r @ e)
        elif not equatorial and circular:
            raan = np.arctan2(n[1], n[0]) % (2 * np.pi)
            argp = 0
            # Argument of latitude
            nu = np.arctan2((r @ np.cross(h, n)) / np.linalg.norm(h), r @ n)
        elif equatorial and circular:
            raan = 0
            argp = 0
            nu = np.arctan2(r[1], r[0]) % (2 * np.pi)  # True longitude
        else:
            a = p / (1 - (ecc**2))
            ka = grav_param * a
            if a > 0:
                e_se = (r @ v) / np.sqrt(ka)
                e_ce = np.linalg.norm(r) * (v @ v) / grav_param - 1
                E = np.arctan2(e_se, e_ce) # eccentric anomaly
                nu = 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(E / 2))
            else:
                e_sh = (r @ v) / np.sqrt(-ka)
                e_ch = np.linalg.norm(r) * (np.linalg.norm(v) ** 2) / grav_param - 1
                F  = np.log((e_ch + e_sh) / (e_ch - e_sh)) / 2 # Hyperbolic anomaly.
                nu = 2 * np.arctan(np.sqrt((ecc + 1) / (ecc - 1)) * np.tanh(F / 2))

            raan = np.arctan2(n[1], n[0]) % (2 * np.pi)
            px = r @ n
            py = (r @ np.cross(h, n)) / np.linalg.norm(h)
            argp = (np.arctan2(py, px) - nu) % (2 * np.pi)

        nu = (nu + np.pi) % (2 * np.pi) - np.pi

        return p, ecc, inc, raan, argp, nu

    # ------------------------------------------------------------------
    #  Mean elements, the state they build, and the Jacobian between them
    # ------------------------------------------------------------------
    class NonEllipticOrbitError(ValueError):
        """Raised when an operation defined only for a closed orbit is asked
        for a parabolic or hyperbolic one."""

    class KeplerConvergenceError(RuntimeError):
        """Raised when Kepler's equation does not converge."""

    class InvalidCovarianceError(ValueError):
        """Raised when a covariance does not have the shape a Cartesian state
        covariance must have."""

    #: Element order used by :meth:`rv2mean_elements`, :meth:`coe2rv` and the
    #: Jacobians: semi-major axis, eccentricity, inclination, RAAN, argument of
    #: periapsis, mean anomaly. This is Montenbruck & Gill's set — ``a`` rather
    #: than the semi-latus rectum and ``M`` rather than the true anomaly — and
    #: is the one the covariance transformation is expressed in.
    MEAN_ELEMENT_NAMES = ("a", "ecc", "inc", "raan", "argp", "M")

    @staticmethod
    def solve_kepler(mean_anomaly, ecc, tol: float = 1e-13, max_iter: int = 60):
        """
        Solve ``M = E - e sin E`` for the eccentric anomaly.

        Parameters
        ----------
        mean_anomaly : float
            Mean anomaly in radians.
        ecc : float
            Eccentricity, ``0 <= e < 1``.
        tol : float, optional
            Convergence tolerance on the Newton step. Defaults to ``1e-13``.
        max_iter : int, optional
            Iteration cap. Defaults to ``60``.

        Returns
        -------
        float
            Eccentric anomaly in radians.

        Raises
        ------
        NonEllipticOrbitError
            If ``ecc`` is outside ``[0, 1)``. The equation still has real roots
            there, so without this guard an open orbit would return a number
            that is not an eccentric anomaly.
        KeplerConvergenceError
            If Newton's method does not converge within ``max_iter``.
        """
        if not (0.0 <= ecc < 1.0):
            raise OrbitalElements.NonEllipticOrbitError(
                f"Kepler's equation in this form needs 0 <= e < 1, got e={ecc!r}."
            )
        M = float(mean_anomaly)
        M = (M + np.pi) % (2.0 * np.pi) - np.pi
        # Starting guess that stays inside the basin of attraction even at
        # high eccentricity.
        E = M if ecc < 0.8 else np.pi * np.sign(M) if M != 0.0 else 0.0
        for _ in range(max_iter):
            step = (E - ecc * np.sin(E) - M) / (1.0 - ecc * np.cos(E))
            E -= step
            if abs(step) < tol:
                return float(E)
        raise OrbitalElements.KeplerConvergenceError(
            f"Kepler's equation did not converge for e={ecc!r}, M={mean_anomaly!r} "
            f"in {max_iter} iterations (last step {step:.3e})."
        )

    @staticmethod
    def _perifocal_axes(inc, raan, argp):
        """
        Inertial directions of the perifocal triad.

        Returns
        -------
        tuple of numpy.ndarray
            ``(P, Q, W)`` — periapsis direction, its in-plane quadrature, and
            the orbit normal, as unit vectors in the reference frame.
        """
        cO, sO = np.cos(raan), np.sin(raan)
        ci, si = np.cos(inc), np.sin(inc)
        cw, sw = np.cos(argp), np.sin(argp)
        P = np.array([cO * cw - sO * sw * ci,
                      sO * cw + cO * sw * ci,
                      sw * si])
        Q = np.array([-cO * sw - sO * cw * ci,
                      -sO * sw + cO * cw * ci,
                      cw * si])
        W = np.array([sO * si, -cO * si, ci])
        return P, Q, W

    @classmethod
    def coe2rv(cls, grav_param, a, ecc, inc, raan, argp, mean_anomaly):
        """
        Build a Cartesian state from mean Keplerian elements.

        The inverse of :meth:`rv2mean_elements`, and the function the
        elements-to-state Jacobian differentiates.

        Parameters
        ----------
        grav_param : float
            Gravitational parameter in km^3/s^2.
        a : float
            Semi-major axis in km. Must be positive.
        ecc : float
            Eccentricity, ``0 <= e < 1``.
        inc, raan, argp, mean_anomaly : float
            Inclination, RAAN, argument of periapsis and mean anomaly, radians.

        Returns
        -------
        tuple of numpy.ndarray
            Position (km) and velocity (km/s).

        Raises
        ------
        NonEllipticOrbitError
            If ``a <= 0`` or ``ecc`` is outside ``[0, 1)``.
        """
        if a <= 0.0 or not (0.0 <= ecc < 1.0):
            raise cls.NonEllipticOrbitError(
                "coe2rv is defined for a closed orbit: needs a > 0 and "
                f"0 <= e < 1, got a={a!r}, e={ecc!r}."
            )

        E = cls.solve_kepler(mean_anomaly, ecc)
        cE, sE = np.cos(E), np.sin(E)
        beta = np.sqrt(1.0 - ecc * ecc)
        radius = a * (1.0 - ecc * cE)

        P, Q, _ = cls._perifocal_axes(inc, raan, argp)
        position = a * (cE - ecc) * P + a * beta * sE * Q
        speed = np.sqrt(grav_param * a) / radius
        velocity = speed * (-sE * P + beta * cE * Q)
        return position, velocity

    @classmethod
    def rv2mean_elements(cls, grav_param, r, v):
        """
        Convert a Cartesian state to the mean elements the Jacobian uses.

        Same geometry as :meth:`rv2coe`, re-expressed in Montenbruck & Gill's
        set: semi-major axis instead of the semi-latus rectum, and mean anomaly
        instead of true anomaly.

        Parameters
        ----------
        grav_param : float
            Gravitational parameter in km^3/s^2.
        r, v : numpy.ndarray
            Position (km) and velocity (km/s).

        Returns
        -------
        numpy.ndarray
            ``(a, ecc, inc, raan, argp, M)`` in km and radians.

        Raises
        ------
        NonEllipticOrbitError
            If the state is not on a closed orbit.
        """
        p, ecc, inc, raan, argp, nu = cls().rv2coe(grav_param, np.asarray(r, dtype=float),
                                                   np.asarray(v, dtype=float))
        if ecc >= 1.0:
            raise cls.NonEllipticOrbitError(
                f"mean elements are defined for a closed orbit; e = {ecc!r}."
            )
        a = p / (1.0 - ecc * ecc)
        # true -> eccentric -> mean
        E = 2.0 * np.arctan2(np.sqrt(1.0 - ecc) * np.sin(nu / 2.0),
                             np.sqrt(1.0 + ecc) * np.cos(nu / 2.0))
        M = (E - ecc * np.sin(E)) % (2.0 * np.pi)
        return np.array([a, ecc, inc, raan, argp, M])

    @classmethod
    def state_by_elements_jacobian(
        cls, grav_param, a, ecc, inc, raan, argp, mean_anomaly
    ):
        """
        ``d(r, v) / d(a, e, i, RAAN, argp, M)`` in closed form.

        Follows Montenbruck & Gill: the three orientation angles enter only
        through rigid rotations, so their partials are cross products with the
        corresponding rotation axis — the inertial pole for the RAAN, the line
        of nodes for the inclination, and the orbit normal for the argument of
        periapsis. The in-plane elements are differentiated through the
        eccentric anomaly, using ``dE/dM = a/r`` and ``dE/de = a sin(E)/r``
        from Kepler's equation.

        Parameters
        ----------
        grav_param : float
            Gravitational parameter in km^3/s^2.
        a, ecc, inc, raan, argp, mean_anomaly : float
            Mean elements, km and radians.

        Returns
        -------
        numpy.ndarray
            ``(6, 6)`` Jacobian; rows are ``(rx, ry, rz, vx, vy, vz)`` and
            columns follow :attr:`MEAN_ELEMENT_NAMES`.
        """
        E = cls.solve_kepler(mean_anomaly, ecc)
        cE, sE = np.cos(E), np.sin(E)
        beta = np.sqrt(1.0 - ecc * ecc)
        radius = a * (1.0 - ecc * cE)

        P, Q, W = cls._perifocal_axes(inc, raan, argp)
        position = a * (cE - ecc) * P + a * beta * sE * Q
        speed = np.sqrt(grav_param * a) / radius
        velocity = speed * (-sE * P + beta * cE * Q)

        # Line of nodes: the axis the inclination rotates about.
        node = np.array([np.cos(raan), np.sin(raan), 0.0])
        pole = np.array([0.0, 0.0, 1.0])

        jac = np.zeros((6, 6))

        # a: r scales with a; v scales with 1/sqrt(a).
        jac[0:3, 0] = position / a
        jac[3:6, 0] = -velocity / (2.0 * a)

        # e at fixed M, with dE/de = a sin(E) / r.
        dE_de = a * sE / radius
        dr_de = (a * (-sE * dE_de - 1.0) * P
                 + a * (-ecc / beta * sE + beta * cE * dE_de) * Q)
        dradius_de = a * (-cE + ecc * sE * dE_de)
        dspeed_de = -speed * dradius_de / radius
        dv_de = (dspeed_de * (-sE * P + beta * cE * Q)
                 + speed * (-cE * dE_de * P
                            + (-ecc / beta * cE - beta * sE * dE_de) * Q))
        jac[0:3, 1] = dr_de
        jac[3:6, 1] = dv_de

        # Orientation angles: rigid rotations about their own axes.
        jac[0:3, 2] = np.cross(node, position)
        jac[3:6, 2] = np.cross(node, velocity)
        jac[0:3, 3] = np.cross(pole, position)
        jac[3:6, 3] = np.cross(pole, velocity)
        jac[0:3, 4] = np.cross(W, position)
        jac[3:6, 4] = np.cross(W, velocity)

        # M, with dE/dM = a / r.
        dE_dM = a / radius
        dr_dM = (a * (-sE * P + beta * cE * Q)) * dE_dM
        dradius_dM = a * ecc * sE * dE_dM
        dspeed_dM = -speed * dradius_dM / radius
        dv_dM = (dspeed_dM * (-sE * P + beta * cE * Q)
                 + speed * (-cE * P - beta * sE * Q) * dE_dM)
        jac[0:3, 5] = dr_dM
        jac[3:6, 5] = dv_dM

        return jac

    @classmethod
    def elements_by_state_jacobian(cls, grav_param, r, v):
        """
        ``d(a, e, i, RAAN, argp, M) / d(r, v)``.

        The inverse of :meth:`state_by_elements_jacobian`, which is how
        Montenbruck & Gill obtain it: the state-by-element partials have closed
        forms, and the map is a bijection wherever the elements are defined.

        Parameters
        ----------
        grav_param : float
            Gravitational parameter in km^3/s^2.
        r, v : numpy.ndarray
            Position (km) and velocity (km/s).

        Returns
        -------
        numpy.ndarray
            ``(6, 6)`` Jacobian; rows follow :attr:`MEAN_ELEMENT_NAMES` and
            columns are ``(rx, ry, rz, vx, vy, vz)``.

        Raises
        ------
        NonEllipticOrbitError
            If the state is not on a closed orbit.
        """
        elements = cls.rv2mean_elements(grav_param, r, v)
        return np.linalg.inv(
            cls.state_by_elements_jacobian(grav_param, *elements)
        )

    @classmethod
    def covariance_to_elements(cls, grav_param, r, v, covariance):
        """
        Map a Cartesian state and its covariance into mean elements.

        Linear (first-order) propagation of uncertainty,

        .. math::

            P_{oe} = J \\, P_{rv} \\, J^{T},
            \\qquad J = \\frac{\\partial(a,e,i,\\Omega,\\omega,M)}
                                {\\partial(\\mathbf r, \\mathbf v)}

        Parameters
        ----------
        grav_param : float
            Gravitational parameter in km^3/s^2.
        r, v : numpy.ndarray
            Position (km) and velocity (km/s).
        covariance : numpy.ndarray
            ``(6, 6)`` Cartesian covariance, ordered ``(r, v)``. A larger
            matrix — a state carrying estimated parameters — is accepted and
            its leading 6x6 block is used.

        Returns
        -------
        elements : numpy.ndarray
            ``(a, e, i, RAAN, argp, M)``, km and radians.
        covariance_elements : numpy.ndarray
            ``(6, 6)`` covariance in those elements, symmetrised.
        jacobian : numpy.ndarray
            The ``(6, 6)`` Jacobian that was applied.

        Raises
        ------
        InvalidCovarianceError
            If ``covariance`` is not square, or smaller than 6x6.
        NonEllipticOrbitError
            If the state is not on a closed orbit.

        Notes
        -----
        This is a linearisation: it is accurate while the Cartesian
        uncertainty is small against the curvature of the element map. A
        near-circular or near-equatorial orbit makes ``argp``/``RAAN`` weakly
        defined and the Jacobian correspondingly large, so read those variances
        with care.
        """
        cartesian = np.asarray(covariance, dtype=float)
        if cartesian.ndim != 2 or cartesian.shape[0] != cartesian.shape[1]:
            raise cls.InvalidCovarianceError(
                "covariance must be a square matrix, got shape "
                f"{cartesian.shape}."
            )
        if cartesian.shape[0] < 6:
            raise cls.InvalidCovarianceError(
                "covariance must be at least 6x6 to cover position and "
                f"velocity, got shape {cartesian.shape}."
            )
        cartesian = cartesian[:6, :6]

        elements = cls.rv2mean_elements(grav_param, r, v)
        jacobian = np.linalg.inv(
            cls.state_by_elements_jacobian(grav_param, *elements)
        )
        mapped = jacobian @ cartesian @ jacobian.T
        # Round-off makes the product very slightly asymmetric.
        return elements, 0.5 * (mapped + mapped.T), jacobian
