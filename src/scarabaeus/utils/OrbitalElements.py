# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import scarabaeus.utils.NumpyWrapper as np 
import numpy.typing as npt

# class definition
class OrbitalElements:
    """ Performs conversions between Cartesian state vectors and classical orbital elements.

        Provides static methods to convert between position/velocity vectors
        ``(r, v)`` and Keplerian orbital elements ``(p, e, i, Ω, ω, ν)``, as
        well as related quantities such as mean motion, period, and eccentric
        anomaly.
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
