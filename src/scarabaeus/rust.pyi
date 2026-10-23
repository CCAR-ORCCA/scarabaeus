# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""
Scarabaeus Rust
---------------
Rust implementation of computationally expensive tasks within Scarabaeus.
"""

from typing import Callable, List, Optional
import numpy as np

class IAS15:
    r"""Implicit integrator with Adaptive timeStepping, 15th order (IAS15).

    This is a Rust implementation of the IAS15 algorithm [1]_ that interfaces
    with Python.

    IAS15 integrates second-order systems
    :math:`\ddot{\mathbf{y}} = \mathbf{F}(t, \mathbf{y}, \dot{\mathbf{y}})`:
    the first ``n_state / 2`` rows of the state are positions, the rest
    velocities, and ``dynamics`` returns the derivative of the full state.
    Over a step :math:`\Delta t`, with :math:`h = (t - t_0)/\Delta t \in [0, 1]`,
    the force is expanded as

    .. math::

        \mathbf{F}(h) = \mathbf{F}_0 + \sum_{k=0}^{6} \mathbf{b}_k\, h^{k+1},

    and integrated term by term [2]_:

    .. math::

        \dot{\mathbf{y}}(h) &= \dot{\mathbf{y}}_0 + h\,\Delta t
            \left(\mathbf{F}_0 + \sum_{k=0}^{6} \frac{\mathbf{b}_k\, h^{k+1}}{k+2}\right), \\
        \mathbf{y}(h) &= \mathbf{y}_0 + h\,\Delta t\,\dot{\mathbf{y}}_0
            + \frac{(h\,\Delta t)^2}{2}\left(\mathbf{F}_0
            + \sum_{k=0}^{6} \frac{2\,\mathbf{b}_k\, h^{k+1}}{(k+2)(k+3)}\right).

    The :math:`\mathbf{b}_k` come from a predictor-corrector iteration over
    the seven Gauss-Radau nodes :math:`h_1, \dots, h_7`, with compensated
    summation. It stops when :math:`\max|\delta\mathbf{b}_6| / \max|\mathbf{F}|`
    falls below :math:`10^{-16}`, stops decreasing, or after 12 iterations.
    The last term sets the next step:

    .. math::

        \tilde{\epsilon} = \frac{\max_k |b_{6,k}|}{\max_k |F_k|}, \qquad
        \Delta t_{new} = \Delta t \left(\frac{\epsilon}{\tilde{\epsilon}}\right)^{1/7}.

    A step with :math:`|\Delta t_{new}/\Delta t|` below ``SAFETY_FACTOR`` is
    rejected and retried with :math:`\Delta t_{new}`. Growth is capped at
    :math:`\Delta t /` ``SAFETY_FACTOR`` and the step never drops below
    ``min_dt``.

    Parameters
    ----------
    dt : float
        Time step given in seconds
    min_dt : float
        Minimum allowed time step in seconds.
    dynamics : callable
        Dynamics function describing the equations of motion of the object
        to be integrated.
    n_state : int, optional
        The number of rows in the propagated state vector. Defaults to ``6``.

    Attributes
    ----------
    SAFETY_FACTOR : float
        Factor to ensure time steps taken within the integrator are not too small. Default value is ``0.25``.
    EPSILON : float
        Precision parameter of the step-size control, :math:`\epsilon` below. Default value is ``1e-10``.
    events : list[IntegrationEvent]
        The IntegrationEvent objects that IAS15 will search for during the integration.
    logged_events : list[(float, list)]
        Detected events, saved as a tuple of the form (time, state).
    event_tol : float
        Allowable error to consider an event has occured. Default value is ``1e-3``.

    References
    ----------
    .. [1] H. Rein and D. S. Spiegel, "IAS15: a fast, adaptive, high-order
        integrator for gravitational dynamics, accurate to machine
        precision over a billion orbits," Monthly Notices of the Royal
        Astronomical Society, vol. 446, no. 2, pp. 1424-1437, 2015.
        Available: https://doi.org/10.1093/mnras/stu2164
    .. [2] E. Everhart, "An efficient integrator that uses Gauss-Radau
        spacings," in *Dynamics of Comets: Their Origin and Evolution*,
        A. Carusi and G. B. Valsecchi, Eds., Reidel, 1985, pp. 185-202.
    """

    def __init__(
        self,
        dt: float,
        min_dt: float,
        dynamics: Callable[[float, list[float]], list[float]],
        n_state: int = 6,
    ) -> None: ...

    # attribute types
    SAFETY_FACTOR: float
    EPSILON: float

    def integrate(
        self, interval: np.ndarray | tuple[float, float], y0: list
    ) -> tuple[list[float], list[float]]:
        """Perform integration with the propagator.

        Parameters
        ----------
        interval : numpy.ndarray or tuple of two floats
            The interval to integrate over. If a numpy.ndarray is given,
            solve for states at each time within the array. If a tuple of
            of floats is given, solve for states at times determined by the
            predictor-corrector along the interval and exactly at the start
            and end times.
        y0 : list
            Initial state vector.

        Returns
        -------
        t : ndarray of shape (n_points)
            Time points.
        y : ndarray of shape (n, n_points)
            Solution states at times t.
        """
        ...

    def add_event(self, event: IntegrationEvent) -> None:
        """Add an event to search for during integration.

        Parameters
        ----------
        event : IntegrationEvent
            Event to search for during integration.
        """
        ...

    def set_event_tol(self, tolerance: float) -> None:
        """Set the event detection tolerance.

        Parameters
        ----------
        tolerance : float
            The new event tolerance.
        """
        ...

class IntegrationEvent:
    r"""Defines an event based off time, state, or both.

    After each accepted :class:`IAS15` step, with :math:`g_{k-1}` and
    :math:`g_k` the values of ``event_finder`` at the previous and the current
    step, the event triggers on

    .. math::

        g_{k-1} < 0 \le g_k \quad \text{(increasing)}, \qquad
        g_{k-1} > 0 \ge g_k \quad \text{(decreasing)},

    as selected by ``find_increasing`` and ``find_decreasing``. The crossing
    time is not refined inside the step, so it is resolved to the step size.

    Parameters
    ----------
    event_finder : Callable[[float, list], float]
        Event function :math:`g(t, \mathbf{y})`; the event is a zero crossing.
    find_increasing : bool
        Flag to detect the event when it crosses from negative to positive.
    find_decreasing : bool
        Flag to detect the even when it crosses from positive to negative.
    is_terminal : bool, optional
        Flag to decide if integration should stop when the event is
        detected. Defaults to ``False``.
    """

    def __init__(
        self,
        event_finder: Callable[[float, list], float],
        find_increasing: bool,
        find_decreasing: bool,
        is_terminal: bool = False,
    ) -> None: ...

class EphemerisCache:
    r"""Chebyshev ephemeris for a set of bodies, read out of the loaded SPK kernel
    or kernels. Positions only. Build it with
    ``scarabaeus.utils.EphemerisCache.build_ephemeris_cache``.

    A record covers :math:`[t_{mid} - t_{rad},\ t_{mid} + t_{rad}]`. With
    :math:`\tau = (t - t_{mid})/t_{rad} \in [-1, 1]` and :math:`c_k` the
    coefficients of one axis, the coordinate is the Chebyshev series

    .. math::

        x(t) = \sum_{k=0}^{N} c_k\, T_k(\tau),

    summed by Clenshaw's recurrence [1]_:

    .. math::

        b_k = c_k + 2\tau\, b_{k+1} - b_{k+2}, \quad b_{N+1} = b_{N+2} = 0,
        \qquad x(t) = c_0 + \tau\, b_1 - b_2.

    A body is chained through its centers down to the solar-system
    barycenter, :math:`\mathbf{r}_b = \mathbf{r}_{b/c(b)} + \mathbf{r}_{c(b)}`,
    and a relative position is the difference of two such positions,
    :math:`\mathbf{r}_{t/o} = \mathbf{r}_t - \mathbf{r}_o`.

    Parameters
    ----------
    body_coeffs : list of list of tuple
        Chebyshev records ``(t_mid, t_rad, x, y, z)`` for each body across the
        cached interval, where ``x``, ``y`` and ``z`` are coefficient lists.
    body_ids : list of int
        SPICE IDs in the same order as ``body_coeffs``.
    center_ids : list of int
        SPICE IDs of each body's center. Necessary for chaining between frames
        for bodies that are not the primary, e.g. Phobos.

    References
    ----------
    .. [1] C. W. Clenshaw, "A note on the summation of Chebyshev series,"
        *Mathematical Tables and Other Aids to Computation*, vol. 9, no. 51,
        pp. 118-120, 1955.
    """

    ids: List[int]
    """SPICE IDs of the cached bodies, center bodies included."""

    def __init__(self, body_coeffs, body_ids, center_ids) -> None: ...
    def get_relative_position(
        self, target_id: int, observer_id: int, t: float
    ) -> List[float]:
        """Position of ``target_id`` relative to ``observer_id`` at ``t``, in km,
        in the SPK native frame (J2000). Both bodies must be in the cache.
        """
        ...

    def get_relative_positions(
        self, target_ids: List[int], observer_id: int, t: float
    ) -> np.ndarray:
        """Positions of ``target_ids`` relative to ``observer_id`` at ``t``, in
        km, in the SPK native frame (J2000), as an ``(n, 3)`` array with one row
        per target. All bodies must be in the cache.
        """
        ...

class SphericalHarmonicsCore:
    r"""Body-fixed spherical-harmonics kernels (normalized coefficients only).

    Computational core of ``SphericalHarmonicsGravity``: the normalized
    Cunningham B_nm recurrence, the acceleration sum, the position Jacobian,
    and the partials with respect to the C_nm / S_nm coefficients. SPICE
    rotations, ephemerides, and estimated-parameter handling stay on the
    Python side, which passes in the body-fixed position and the effective
    mu / C_nm / S_nm for each evaluation.

    With :math:`R` = ``r_ref`` and fully normalized coefficients, the
    potential is [1]_

    .. math::

        U = \frac{\mu}{r} \sum_{n=0}^{N} \sum_{m=0}^{n}
            \left(\frac{R}{r}\right)^{n} \bar{P}_{nm}(\sin\phi)
            \left[\bar{C}_{nm} \cos m\lambda + \bar{S}_{nm} \sin m\lambda\right]
          = \frac{\mu}{R} \sum_{n=0}^{N} \sum_{m=0}^{n}
            \left(\bar{C}_{nm} V_{nm} + \bar{S}_{nm} W_{nm}\right),

    in terms of the solid harmonics

    .. math::

        B_{nm} = V_{nm} + i\,W_{nm}
               = \left(\frac{R}{r}\right)^{n+1} \bar{P}_{nm}(\sin\phi)\, e^{i m \lambda}.

    These are built from :math:`B_{00} = R/r` with Cartesian recurrences [2]_,
    so there are no angles and no singularity at the poles:

    .. math::

        B_{mm} &= \sqrt{\frac{(1+\delta_{1m})(2m+1)}{2m}}\,
            \frac{R}{r}\,\frac{x + i y}{r}\, B_{m-1,m-1}, \\
        B_{nm} &= \sqrt{\frac{4n^2-1}{n^2-m^2}}\, \frac{R}{r}\,\frac{z}{r}\, B_{n-1,m}
            - \sqrt{\frac{(2n+1)\left[(n-1)^2-m^2\right]}{(2n-3)(n^2-m^2)}}
            \left(\frac{R}{r}\right)^2 B_{n-2,m}.

    The body-fixed acceleration is [3]_

    .. math::

        \ddot{x} &= \frac{\mu}{R^2} \sum_{n,m} \left[
            -g_1 \left(\bar{C}_{nm} V_{n+1,m+1} + \bar{S}_{nm} W_{n+1,m+1}\right)
            + g_2 \left(\bar{C}_{nm} V_{n+1,m-1} + \bar{S}_{nm} W_{n+1,m-1}\right)\right], \\
        \ddot{y} &= \frac{\mu}{R^2} \sum_{n,m} \left[
            g_1 \left(-\bar{C}_{nm} W_{n+1,m+1} + \bar{S}_{nm} V_{n+1,m+1}\right)
            + g_2 \left(-\bar{C}_{nm} W_{n+1,m-1} + \bar{S}_{nm} V_{n+1,m-1}\right)\right], \\
        \ddot{z} &= \frac{\mu}{R^2} \sum_{n,m}
            g_3 \left(\bar{C}_{nm} V_{n+1,m} + \bar{S}_{nm} W_{n+1,m}\right),

    where the :math:`g_2` terms are absent for :math:`m = 0` and

    .. math::

        g_1 &= \frac{1+\delta_{0m}}{2}
            \sqrt{\frac{(2-\delta_{0m})(2n+1)(n+m+2)(n+m+1)}{2(2n+3)}}, \\
        g_2 &= \frac{1}{2}
            \sqrt{\frac{(2-\delta_{0m})(2n+1)(n-m+2)(n-m+1)}{(2-\delta_{1m})(2n+3)}}, \\
        g_3 &= -\sqrt{\frac{(2n+1)(n+m+1)(n-m+1)}{2n+3}}.

    The acceleration is linear in :math:`\bar{C}_{nm}` and :math:`\bar{S}_{nm}`,
    so their partials are the bracketed factors. The position Jacobian
    differentiates once more, which carries the recurrence to degree
    :math:`N + 2`. The ``_rf`` methods rotate the body-fixed results with
    the body-fixed to reference rotation :math:`\mathbf{R}`:
    :math:`\mathbf{a}_{rf} = \mathbf{R}\,\mathbf{a}` and
    :math:`\mathbf{J}_{rf} = \mathbf{R}\,\mathbf{J}\,\mathbf{R}^T`.

    Parameters
    ----------
    order : int
        Maximum degree/order of the expansion.
    r_ref : float
        Reference radius of the body, in km.

    References
    ----------
    .. [1] W. M. Kaula, *Theory of Satellite Geodesy*. Waltham, MA:
        Blaisdell, 1966.
    .. [2] L. E. Cunningham, "On the computation of the spherical harmonic
        terms needed during the numerical integration of the orbital motion
        of an artificial satellite," *Celestial Mechanics*, vol. 2, no. 2,
        pp. 207-216, 1970.
    .. [3] O. Montenbruck and E. Gill, *Satellite Orbits: Models, Methods and
        Applications*. Berlin: Springer, 2000, Sec. 3.2.
    """

    def __init__(self, order: int, r_ref: float) -> None: ...
    def compute_bnm(self, r_bf: list) -> tuple[np.ndarray, np.ndarray]:
        """Normalized exterior B_nm matrices for a body-fixed position [km].

        Returns
        -------
        (bnm_real, bnm_imag) : tuple of np.ndarray
            Each of shape (order+3, order+3).
        """
        ...

    def compute_acceleration_bf(
        self, r_bf: list, mu: float, cnm: np.ndarray, snm: np.ndarray
    ) -> tuple[List[float], np.ndarray, np.ndarray]:
        """Acceleration in the body-fixed frame, in km/s^2.

        Returns
        -------
        (acc, bnm_real, bnm_imag) : tuple
            ``acc`` as ``[ax, ay, az]`` plus the B_nm matrices.
        """
        ...

    def compute_accel_and_jacobian_bf(
        self,
        r_bf: list,
        mu: float,
        cnm: np.ndarray,
        snm: np.ndarray,
        mu_jac: float | None = None,
        cnm_jac: np.ndarray | None = None,
        snm_jac: np.ndarray | None = None,
    ) -> tuple[List[float], List[float], np.ndarray, np.ndarray]:
        """Fused acceleration + da/dr on a single B_nm pass, body-fixed frame.

        The jacobian uses ``mu_jac``/``cnm_jac``/``snm_jac`` when given
        (nominal coefficients while the acceleration uses estimated ones),
        otherwise the acceleration's values.

        Returns
        -------
        (acc, jac, bnm_real, bnm_imag) : tuple
            ``acc`` as ``[ax, ay, az]``; ``jac`` flattened row-major.
        """
        ...

    def compute_jacobian_bf(
        self, r_bf: list, mu: float, cnm: np.ndarray, snm: np.ndarray
    ) -> tuple[List[float], np.ndarray, np.ndarray]:
        """da/dr in the body-fixed frame, in 1/s^2.

        Returns
        -------
        (jac, bnm_real, bnm_imag) : tuple
            ``jac`` flattened row-major as
            ``[d00, d01, d02, d10, d11, d12, d20, d21, d22]``.
        """
        ...

    def compute_acceleration_rf(
        self,
        r_bf: list,
        r_b2rf: np.ndarray,
        mu: float,
        cnm: np.ndarray,
        snm: np.ndarray,
    ) -> tuple[List[float], np.ndarray, np.ndarray]:
        """Acceleration rotated into the reference frame (``R @ a``), km/s^2.

        ``r_b2rf`` is the body-fixed -> reference-frame DCM (3x3).
        Returns ``(acc_rf, bnm_real, bnm_imag)``.
        """
        ...

    def compute_accel_and_jacobian_rf(
        self,
        r_bf: list,
        r_b2rf: np.ndarray,
        mu: float,
        cnm: np.ndarray,
        snm: np.ndarray,
        mu_jac: float | None = None,
        cnm_jac: np.ndarray | None = None,
        snm_jac: np.ndarray | None = None,
    ) -> tuple[List[float], List[float], np.ndarray, np.ndarray]:
        """Fused acceleration + da/dr, both rotated into the reference frame
        (``R @ a`` and ``R @ J @ R^T``). Returns
        ``(acc_rf, jac_rf, bnm_real, bnm_imag)``.
        """
        ...

    def compute_jacobian_rf(
        self,
        r_bf: list,
        r_b2rf: np.ndarray,
        mu: float,
        cnm: np.ndarray,
        snm: np.ndarray,
    ) -> tuple[List[float], np.ndarray, np.ndarray]:
        """da/dr rotated into the reference frame (``R @ J @ R^T``), 1/s^2.
        Returns ``(jac_rf, bnm_real, bnm_imag)``.
        """
        ...

    def compute_jacobian_ad_bf(
        self, r_bf: list, mu: float, cnm: np.ndarray, snm: np.ndarray
    ) -> List[float]:
        """da/dr in the body-fixed frame via forward-mode automatic
        differentiation (dual numbers) of the acceleration.

        Independent of ``compute_jacobian_bf``; the two agree to machine
        precision, which verifies the analytical position-partial formulas.
        Returns the flattened row-major 3x3 Jacobian.
        """
        ...

    def compute_partial_by_mu_ad_rf(
        self,
        r_bf: list,
        r_b2rf: np.ndarray,
        mu: float,
        cnm: np.ndarray,
        snm: np.ndarray,
        r_origin_sph: list | None = None,
    ) -> List[float]:
        """da/dGM in the reference frame via forward-mode automatic
        differentiation (dual numbers), in km/s^2 per km^3/s^2.

        With ``r_origin_sph`` (the harmonics body relative to the propagation
        origin, when they differ) the indirect term ``-GM r_o / |r_o|^3`` is
        differentiated too. Independent of the analytical ``a / GM`` partial.
        """
        ...

    def compute_partial_by_cs_ad_bf(
        self, r_bf: list, mu: float, cnm: np.ndarray, snm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """da/dC_nm and da/dS_nm in the body-fixed frame via forward-mode
        automatic differentiation (dual numbers), in km/s^2.

        Independent of ``compute_partial_by_c_bf`` / ``compute_partial_by_s_bf``.

        Returns
        -------
        (d_c, d_s) : tuple of np.ndarray
            Each of shape (3, order+1, order+1); S_n0 left at zero.
        """
        ...

    def compute_partial_by_pole_ad_rf(
        self,
        r_rel_rf: list,
        m_j2rf: np.ndarray,
        ra: float,
        dec: float,
        w0: float,
        pm_rate: float,
        d_days: float,
        mu: float,
        cnm: np.ndarray,
        snm: np.ndarray,
    ) -> tuple[List[float], List[List[float]]]:
        """Pole partials in the reference frame via forward-mode automatic
        differentiation (dual numbers) of the rotated acceleration, the IAU pole
        rotation ``m_j2rf @ Rz(-(90+RA)) @ Rx(-(90-DEC)) @ Rz(-W)`` included,
        with ``W = w0 + pm_rate * d_days`` [deg].

        Returns
        -------
        (a_rf, partials) : tuple
            ``a_rf`` in km/s^2 and ``[da/dRA, da/dDEC, da/dPM]``, in km/s^2 per
            rad, per rad, and per deg/day.
        """
        ...

    def compute_partial_by_c_bf(
        self, r_bf: list, mu: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """da/dC_nm in the body-fixed frame, in km/s^2.

        Returns
        -------
        (partials, bnm_real, bnm_imag) : tuple of np.ndarray
            ``partials`` of shape (3, order+1, order+1).
        """
        ...

    def compute_partial_by_s_bf(
        self, r_bf: list, mu: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """da/dS_nm in the body-fixed frame, in km/s^2.

        Returns
        -------
        (partials, bnm_real, bnm_imag) : tuple of np.ndarray
            ``partials`` of shape (3, order+1, order+1).
        """
        ...

class NPlateSRPCore:
    r"""N-plate SRP kernels.

    Computational core of ``nPlateSRP``: the per-plate illumination sums
    for the acceleration and the position / velocity Jacobians, plus the
    spacecraft-fixed <-> inertial mapping. Attitude resolution (and its
    state derivatives), the Sun ephemeris and eta_srp lookup stay on the
    Python side, which passes the results in.

    Let :math:`\mathbf{C}` be the SC-fixed to inertial DCM (``dcm_b2i``),
    :math:`\mathbf{s} = -\mathbf{C}^T \mathbf{r}_{\odot \to sc}` the SC-fixed
    spacecraft-to-Sun vector, :math:`r = |\mathbf{s}|`,
    :math:`\hat{\mathbf{s}} = \mathbf{s}/r` and
    :math:`\cos\theta_i = \hat{\mathbf{n}}_i \cdot \hat{\mathbf{s}}`. Plate
    :math:`i` has area :math:`A_i`, specular and diffuse reflectivities
    :math:`\rho_{s,i}` and :math:`\rho_{d,i}`. Summed over the lit plates,
    the SC-fixed acceleration [1]_ is

    .. math::

        \mathbf{T} = -\frac{\eta\, P_{SRP}}{m\, r^2}
            \sum_{\cos\theta_i > 0} A_i \cos\theta_i \left[
            (1 - \rho_{s,i})\, \hat{\mathbf{s}}
            + \left(\tfrac{2}{3}\rho_{d,i} + 2\rho_{s,i}\cos\theta_i\right)
            \hat{\mathbf{n}}_i \right],

    and the inertial one is :math:`\mathbf{a} = \mathbf{C}\,\mathbf{T}`.
    With :math:`p_i = \mathbf{n}_i \cdot \mathbf{s}`, :math:`c_i = \eta P_{SRP} A_i / m`,
    :math:`a_i = -c_i(1-\rho_{s,i})`, :math:`b_i = -\tfrac{2}{3}c_i\rho_{d,i}`
    and :math:`d_i = -2c_i\rho_{s,i}`, the kernel writes it as
    :math:`\mathbf{T} = \sum a_i p_i\, \mathbf{s}/r^4 + \sum b_i p_i\, \mathbf{n}_i/r^3
    + \sum d_i p_i^2\, \mathbf{n}_i/r^4`, whose gradient is

    .. math::

        \frac{\partial \mathbf{T}}{\partial \mathbf{s}} &=
            \frac{\left(\sum a_i p_i\right) \mathbf{I}
                + \mathbf{s} \left(\sum a_i \mathbf{n}_i\right)^T}{r^4}
            - \frac{4 \left(\sum a_i p_i\right) \mathbf{s}\mathbf{s}^T}{r^6}
            + \frac{\sum b_i \mathbf{n}_i \mathbf{n}_i^T}{r^3}
            - \frac{3 \left(\sum b_i p_i \mathbf{n}_i\right) \mathbf{s}^T}{r^5} \\
            &\quad + \frac{2 \sum d_i p_i \mathbf{n}_i \mathbf{n}_i^T}{r^4}
            - \frac{4 \left(\sum d_i p_i^2 \mathbf{n}_i\right) \mathbf{s}^T}{r^6}.

    The inertial Jacobians follow by the chain rule through
    :math:`\mathbf{s}` and the attitude:

    .. math::

        \frac{\partial \mathbf{a}}{\partial r_k} &=
            \frac{\partial \mathbf{C}}{\partial r_k} \mathbf{T}
            + \mathbf{C} \frac{\partial \mathbf{T}}{\partial \mathbf{s}}
            \left(-\mathbf{C}^T \mathbf{e}_k
            - \frac{\partial \mathbf{C}^T}{\partial r_k} \mathbf{r}_{\odot \to sc}\right), \\
        \frac{\partial \mathbf{a}}{\partial v_k} &=
            \frac{\partial \mathbf{C}}{\partial v_k} \mathbf{T}
            - \mathbf{C} \frac{\partial \mathbf{T}}{\partial \mathbf{s}}
            \frac{\partial \mathbf{C}^T}{\partial v_k} \mathbf{r}_{\odot \to sc}.

    For a fixed attitude the position Jacobian reduces to
    :math:`-\mathbf{C}\, (\partial \mathbf{T}/\partial \mathbf{s})\, \mathbf{C}^T`
    and the velocity Jacobian vanishes.

    The kernel is generic over ``f64`` and dual numbers, so
    ``compute_partials_ad`` differentiates the exact function that
    ``compute_acceleration`` evaluates.

    Parameters
    ----------
    p_srp : float
        Solar radiation pressure constant, :math:`P_{SRP}`: the pressure at
        1 AU times 1 AU squared, in kg km s^-2.
    areas : np.ndarray
        (N,) plate areas in km^2.
    diff_ref : np.ndarray
        (N,) diffuse reflectivity coefficients.
    spec_ref : np.ndarray
        (N,) specular reflectivity coefficients.
    double_sided : list[bool], optional
        (N,) flags; a double-sided plate lit from behind acts as a plate
        with the opposite normal. Defaults to all single-sided.

    References
    ----------
    .. [1] P. W. Kenneally, "High geometric fidelity solar radiation pressure
        modeling via graphics processing unit," Master's thesis, University
        of Colorado Boulder, 2016.
    """

    def __init__(
        self,
        p_srp: float,
        areas: np.ndarray,
        diff_ref: np.ndarray,
        spec_ref: np.ndarray,
        double_sided: Optional[List[bool]] = None,
    ) -> None: ...
    def compute_acceleration(
        self,
        sun2body: list,
        dcm_b2i: np.ndarray,
        eta_srp: float,
        mass: float,
        normals: np.ndarray,
    ) -> List[float]:
        """Inertial SRP acceleration, in km/s^2.

        ``sun2body`` is the inertial Sun->spacecraft vector (km), ``dcm_b2i``
        the (3, 3) SC-fixed -> inertial DCM, ``mass`` in kg and ``normals``
        the (N, 3) SC-fixed plate outward normals.
        """
        ...

    def compute_jacobian(
        self,
        sun2body: list,
        dcm_b2i: np.ndarray,
        eta_srp: float,
        mass: float,
        normals: np.ndarray,
        dcm_dr: Optional[np.ndarray] = None,
    ) -> List[float]:
        """Analytical inertial position Jacobian, in 1/s^2, flattened row-major.

        ``-DCM @ dT/ds @ DCM^T``, with ``s`` the SC-fixed body->Sun vector and
        ``T`` the SC-fixed acceleration. When the attitude follows the
        position (nadir pointing), ``dcm_dr`` is the (3, 3, 3) derivative
        ``dcm_dr[j, i, k] = dDCM[i, j]/dr_k`` and the attitude terms are
        added. Normals and mass are held fixed.
        """
        ...

    def compute_jacobian_by_velocity(
        self,
        sun2body: list,
        dcm_b2i: np.ndarray,
        eta_srp: float,
        mass: float,
        normals: np.ndarray,
        dcm_dv: np.ndarray,
    ) -> List[float]:
        """Analytical inertial velocity Jacobian, in 1/s, flattened row-major.

        Non-zero only through the attitude: ``dcm_dv`` is the (3, 3, 3)
        derivative ``dcm_dv[j, i, k] = dDCM[i, j]/dv_k`` (nadir pointing).
        """
        ...

    def compute_partials_ad(
        self,
        sun2body: list,
        dcm_b2i: np.ndarray,
        eta_srp: float,
        mass: float,
        normals: np.ndarray,
        dcm_dr: Optional[np.ndarray] = None,
        dcm_dv: Optional[np.ndarray] = None,
    ) -> tuple[List[float], List[float], List[float]]:
        """Dual-number reference partials of ``compute_acceleration``.

        Returns ``(da/dr, da/dv, da/d eta_srp)``, the Jacobians flattened
        row-major. The DCM is seeded through ``dcm_dr`` / ``dcm_dv`` when
        given (``da/dv`` is zero without ``dcm_dv``); normals, mass and the
        Sun position are held fixed, so d sun2body / dr = I. Used by
        ``check_force_model_partials``.
        """
        ...

class PolyhedronGravityCore:
    r"""Exterior gravity of a constant-density polyhedron.

    Computational core of :class:`~scarabaeus.PolyhedralGravity`: the
    closed-form surface and line integrals of Werner & Scheeres [1]_,
    one solid-angle term per triangular facet and one line term per edge.
    The facet and edge dyads and the mesh topology are built once at
    construction; only the solid angles and edge factors depend on the
    field point.

    Exact everywhere outside the body — there is no truncation degree — and
    the summed solid angle is ``0`` outside and ``4*pi`` inside, which
    doubles as a containment test.

    With :math:`G\rho` the gravitational constant times the density,
    :math:`\mathbf{r}_e` and :math:`\mathbf{r}_f` vectors from the field
    point to any point of edge :math:`e` and facet :math:`f`, the potential
    and its derivatives are [1]_

    .. math::

        U &= \frac{G\rho}{2} \sum_{e} \mathbf{r}_e^T \mathbf{E}_e \mathbf{r}_e\, L_e
            - \frac{G\rho}{2} \sum_{f} \mathbf{r}_f^T \mathbf{F}_f \mathbf{r}_f\, \omega_f, \\
        \nabla U &= -G\rho \sum_{e} \mathbf{E}_e \mathbf{r}_e\, L_e
            + G\rho \sum_{f} \mathbf{F}_f \mathbf{r}_f\, \omega_f, \\
        \nabla\nabla U &= G\rho \sum_{e} \mathbf{E}_e\, L_e
            - G\rho \sum_{f} \mathbf{F}_f\, \omega_f, \qquad
        \nabla^2 U = -G\rho \sum_{f} \omega_f.

    The dyads are :math:`\mathbf{F}_f = \hat{\mathbf{n}}_f \hat{\mathbf{n}}_f^T`
    and :math:`\mathbf{E}_e = \hat{\mathbf{n}}_A \hat{\mathbf{n}}_{A,12}^T
    + \hat{\mathbf{n}}_B \hat{\mathbf{n}}_{B,21}^T`, where :math:`A` and
    :math:`B` are the two facets sharing the edge and
    :math:`\hat{\mathbf{n}}_{A,12}` is the edge's outward normal in the plane
    of :math:`A`. The edge factor uses the distances :math:`a, b` from the
    field point to the edge's ends and its length :math:`e`, and the solid
    angle of a facet with corners :math:`\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3`
    (relative to the field point, :math:`R_i = |\mathbf{r}_i|`) follows
    Van Oosterom & Strackee [2]_:

    .. math::

        L_e &= \ln\frac{a + b + e}{a + b - e}, \\
        \omega_f &= 2 \arctan \frac{\mathbf{r}_1 \cdot (\mathbf{r}_2 \times \mathbf{r}_3)}
            {R_1 R_2 R_3 + (\mathbf{r}_1 \cdot \mathbf{r}_2) R_3
            + (\mathbf{r}_1 \cdot \mathbf{r}_3) R_2 + (\mathbf{r}_2 \cdot \mathbf{r}_3) R_1},

    with the two-argument arctangent.

    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) vertex coordinates in the body-fixed frame, km.
    faces : np.ndarray
        (M, 3) zero-based vertex indices, one row per triangular facet,
        wound counter-clockwise seen from outside.
    density : float
        Bulk density, kg/km^3.
    gravitational_constant : float
        Gravitational constant in km^3 kg^-1 s^-2. Supplied by the caller
        so the value tracks ``scarabaeus.Constants.G``.

    References
    ----------
    .. [1] R. A. Werner and D. J. Scheeres, "Exterior gravitation of a
        polyhedron derived and compared with harmonic and mascon gravitation
        representations of asteroid 4769 Castalia," *Celestial Mechanics and
        Dynamical Astronomy*, vol. 65, no. 3, pp. 313-344, 1997.
    .. [2] A. Van Oosterom and J. Strackee, "The solid angle of a plane
        triangle," *IEEE Transactions on Biomedical Engineering*, vol.
        BME-30, no. 2, pp. 125-126, 1983.
    """

    n_vertices: int
    n_facets: int
    n_edges: int
    density: float
    gravitational_constant: float

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        density: float,
        gravitational_constant: float,
    ) -> None: ...
    def volume(self) -> float:
        """Enclosed volume in km^3; negative when the facets are wound inwards."""
        ...

    def mass(self) -> float:
        """Total mass in kg, ``density * volume``."""
        ...

    def center_of_mass(self) -> List[float]:
        """Centre of mass in the mesh's own frame, km.

        Volume-weighted centroid of the tetrahedra formed by each facet
        with the origin. A mesh whose centroid is not at the origin carries
        a dipole term.
        """
        ...

    def face_normals(self) -> List[float]:
        """Outward unit normal of each facet, flattened row-major (M x 3)."""
        ...

    def compute(self, position: np.ndarray) -> tuple:
        """``(U, grad_U, grad2_U, laplacian, omega)`` at one body-fixed point.

        Units are km^2/s^2, km/s^2 (3), 1/s^2 (9, row-major), 1/s^2, and
        steradians.
        """
        ...

    def compute_potential(self, position: np.ndarray) -> float:
        """Gravitational potential at one point, km^2/s^2."""
        ...

    def compute_acceleration(self, position: np.ndarray) -> List[float]:
        """Gravitational acceleration at one point, km/s^2."""
        ...

    def compute_jacobian(self, position: np.ndarray) -> List[float]:
        """Gravity-gradient matrix at one point, 1/s^2, flattened row-major."""
        ...

    def solid_angle(self, position: np.ndarray) -> float:
        """Summed signed solid angle: 0 outside the body, 4*pi inside."""
        ...

    def facet_solid_angles(self, position: np.ndarray) -> List[float]:
        """Signed solid angle each facet subtends at the field point."""
        ...

    def contains(self, position: np.ndarray) -> bool:
        """Whether the field point lies inside the body."""
        ...

    def compute_acceleration_batch(self, positions: np.ndarray) -> List[float]:
        """Accelerations for (N, 3) field points, flattened row-major."""
        ...

class PolyhedronHarmonics:
    r"""Spherical harmonic coefficients of a constant-density polyhedron.

    Werner's exact recursion [1]_, summed over the tetrahedra each facet forms
    with the origin: exact for the polyhedron at every degree, so the only
    error is the truncation and the mesh itself. The mesh is validated and
    stored at construction; :meth:`compute` runs the recursion.

    For a constant density the fully normalized coefficients of reference
    radius :math:`R` are volume integrals of the solid harmonics

    .. math::

        \bar{c}_{nm} + i\,\bar{s}_{nm} = \frac{1}{2n+1}
            \left(\frac{r}{R}\right)^{n} \bar{P}_{nm}(\sin\phi)\, e^{i m \lambda},
        \qquad
        \bar{C}_{nm} + i\,\bar{S}_{nm} = \frac{1}{V} \int_V
            \left(\bar{c}_{nm} + i\,\bar{s}_{nm}\right) dV.

    Each :math:`\bar{c}_{nm}, \bar{s}_{nm}` is a homogeneous polynomial of
    degree :math:`n` in :math:`(x, y, z)/R`, generated from
    :math:`\bar{c}_{00} = 1` by

    .. math::

        \bar{c}_{mm} + i\,\bar{s}_{mm} &= d_m\, (x + i y)
            \left(\bar{c}_{m-1,m-1} + i\,\bar{s}_{m-1,m-1}\right), \\
        \bar{c}_{m+1,m} &= e_{m+1}\, z\, \bar{c}_{mm}, \\
        \bar{c}_{nm} &= f_{nm}\, z\, \bar{c}_{n-1,m} - g_{nm}\, r^2\, \bar{c}_{n-2,m},

    and the same for :math:`\bar{s}_{nm}`, with :math:`d_1 = 1/\sqrt{3}` and

    .. math::

        d_m = \frac{2m-1}{\sqrt{2m(2m+1)}}, \quad
        e_n = \frac{2n-1}{\sqrt{2n+1}}, \quad
        f_{nm} = (2n-1) \sqrt{\frac{2n-1}{(2n+1)(n+m)(n-m)}}, \quad
        g_{nm} = \sqrt{\frac{(2n-3)(n+m-1)(n-m-1)}{(2n+1)(n+m)(n-m)}}.

    The body is split into the tetrahedra :math:`(\mathbf{0}, \mathbf{v}_0,
    \mathbf{v}_1, \mathbf{v}_2)` on its facets. Inside one,
    :math:`\mathbf{r} = \alpha\mathbf{v}_0 + \beta\mathbf{v}_1 + \gamma\mathbf{v}_2`
    and :math:`dV = \det[\mathbf{v}_0\ \mathbf{v}_1\ \mathbf{v}_2]\,
    d\alpha\, d\beta\, d\gamma`, so the recursion runs on polynomials in
    :math:`(\alpha, \beta, \gamma)` and each monomial integrates exactly over
    the unit simplex:

    .. math::

        \int_{\alpha+\beta+\gamma \le 1} \alpha^i \beta^j \gamma^k\,
            d\alpha\, d\beta\, d\gamma = \frac{i!\, j!\, k!}{(i+j+k+3)!}.

    :meth:`compute` returns the sum without the :math:`1/V`, so
    :math:`\bar{C}_{00} = V`.

    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) body-fixed coordinates in km, centred on the centre of mass.
    faces : np.ndarray
        (M, 3) zero-based triangular facet indices.

    References
    ----------
    .. [1] R. A. Werner, "Spherical harmonic coefficients for the potential of
        a constant-density polyhedron," *Computers & Geosciences*, vol. 23,
        no. 10, pp. 1071-1077, 1997.
    """

    n_vertices: int
    n_facets: int

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None: ...
    def compute(
        self,
        n_degree: int,
        ref_radius: float,
        normalized: bool = True,
    ) -> tuple:
        """Coefficients up to ``n_degree``.

        The returned coefficients still carry the body's volume — ``C[0][0]``
        is that volume — so divide the set by it for the conventional
        ``C00 = 1``. :class:`~scarabaeus.ShapeHarmonics` does this, along with
        the file I/O.

        Parameters
        ----------
        n_degree : int
            Maximum degree and order.
        ref_radius : float
            Normalising radius, km.
        normalized : bool
            Fully normalised coefficients when True, raw when False.

        Returns
        -------
        tuple
            ``(Cnm, Snm)``, each a flattened ``(n_degree + 1)**2`` row-major list.
        """
        ...
