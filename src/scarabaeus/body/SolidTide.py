# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from dataclasses import dataclass, replace

import spiceypy as spice

from scarabaeus import ArrayWUnits, Units
import scarabaeus.utils.NumpyWrapper as np
from scarabaeus.utils.Constants import Constants as constants

km, sec = Units.get_units(["km", "sec"])


@dataclass(frozen=True)
class SolidTide:
    r"""Solid tide displacement of a surface point.

    A station is not fixed in its body-fixed frame. Nearby masses deform the
    solid body it stands on, moving it by tens of centimetres on a diurnal
    and semidiurnal cycle. That is well above the noise of a modern ranging
    system, so a station geometry that omits it carries a signature no
    estimated parameter is supposed to absorb.

    By default the degree-2 in-phase term of the IERS Conventions (2010),
    Chapter 7, Section 7.1.1, step 1; ``model="IERS2010"`` evaluates the
    complete IERS routine instead. The degree-2 term is

    .. math::

        \delta\mathbf r = \sum_j \frac{GM_j}{GM_\oplus}
            \frac{R^4}{R_j^3} \left\{
            h_2 \hat{\mathbf r}
                \left[\tfrac{3}{2}(\hat{\mathbf R}_j\cdot\hat{\mathbf r})^2
                      - \tfrac{1}{2}\right]
            + 3 l_2 (\hat{\mathbf R}_j\cdot\hat{\mathbf r})
                \left[\hat{\mathbf R}_j
                      - (\hat{\mathbf R}_j\cdot\hat{\mathbf r})\hat{\mathbf r}
                \right]\right\}

    summed over the tide-raising bodies.

    One tide model serves two consumers. Attach it to
    :attr:`GroundStation.surface_displacement
    <scarabaeus.GroundStation.surface_displacement>` to move a station on the
    surface, and to :attr:`SphericalHarmonicsGravity.solid_tide
    <scarabaeus.SphericalHarmonicsGravity.solid_tide>` to perturb the gravity
    field.

    Nothing in the model is specific to Earth. Earth is the one body with a
    nominal response built in, because it is the one whose Love and Shida
    numbers are pinned by a convention; a station on any other body gets the
    same treatment once its numbers are supplied.

    Parameters
    ----------
    tide_bodies : tuple of str, optional
        Tide-raising bodies. Defaults to the Moon and the Sun for Earth, where
        everything else is below a millimetre. Required for any other central
        body.
    central_body : str, optional
        Deformed body, default EARTH. Its GM and equatorial radius set the
        scale, and the tide-raiser positions are taken relative to it. Nothing
        in the formulation is Earth-specific -- a station on Mars gets its
        tide from this same class, given Mars values.
    love_h2, shida_l2 : float, optional
        Degree-2 Love and Shida numbers. Default to the IERS nominal values
        of :attr:`Constants.EARTH_SOLID_TIDE`, 0.6078 and 0.0847, **for
        Earth only**. For any other body they are required:
        the tidal response varies by a factor of several between bodies
        (Mars is near h2 = 0.17), so a default would be a large silent error
        rather than an approximation.
    love_k : dict, optional
        *Potential* Love numbers :math:`k_{nm}` keyed by ``(degree, order)``,
        used only by :meth:`geopotential_coefficients`. Defaults to the IERS
        Table 6.3 values for Earth, degrees 2 and 3, from
        :attr:`Constants.EARTH_SOLID_TIDE`. These are a table rather
        than one number for the same reason the field itself is: they vary
        with order (:math:`k_{20}` and :math:`k_{22}` differ by about 1%) and
        fall by a factor of three from degree 2 to degree 3. Supplying a
        subset restricts the perturbation to those terms.

        These are a different quantity from :attr:`love_h2`: :math:`h`
        says how far the surface moves, :math:`k` how much the external
        gravity field changes. Confusing them is a factor-of-two error.
    remove_permanent_tide : bool, optional
        Subtract the zero-frequency (permanent) deformation, IERS eq. 7.14.
        Default false, which is what conventional tide-free station
        coordinates need. See Notes -- this is a real fork, not a preference.
    rate_step : float, optional
        Half-interval in seconds for the central difference that supplies the
        body-fixed rate, default 60.
    model : {"degree2", "IERS2010"}, optional
        ``"degree2"`` (default) is the degree-2 in-phase term above.
        ``"IERS2010"`` is the complete IERS Conventions (2010) routine
        DEHANTTIDEINEL, Earth only, with its own constants
        (:attr:`Constants.EARTH_SOLID_TIDE`): step 1 in and out
        of phase for degrees 2 and 3 with latitude-dependent Love and Shida
        numbers, and the step-2 frequency corrections. It differs from the
        degree-2 term by up to about 13 mm, mostly the diurnal K1 correction.

    Notes
    -----
    The degree-2 model retains the degree-2 in-phase term, which is the whole
    of the effect at the decimetre level -- up to about 0.4 m radially at
    Earth.

    The permanent-tide constants are Earth values from IERS eq. 7.14, kept in
    :attr:`Constants.EARTH_SOLID_TIDE`, so ``remove_permanent_tide`` is only
    meaningful for Earth.

    The degree-2 model omits, with magnitudes: degree 3 (about 2 mm), the
    out-of-phase diurnal and semidiurnal contributions and the latitude
    dependence of :math:`h` and :math:`l` (a few mm), and the
    frequency-dependent step-2 corrections (up to about 13 mm, the diurnal K1
    term above all). ``model="IERS2010"`` includes all of these. Neither
    includes ocean and atmospheric loading (about 1 cm and a few mm) or the
    pole tide (about 1 cm). :meth:`displacement_partials` is the degree-2
    term's sensitivity to the Love and Shida numbers under either model.

    The **permanent tide** is a genuine convention fork rather than a
    refinement. The formula above yields the total deformation including a
    zero-frequency part -- roughly -12 cm radially at the poles and +6 cm at
    the equator. Conventional tide-free coordinates, which is what the ITRF
    and DSN tables publish, already exclude that permanent deformation, so it
    must be *restored* by this formula and ``remove_permanent_tide`` stays
    false. Set it true only for coordinates that are already mean-tide or
    zero-tide, or the permanent part is applied twice.

    The choice is made on the coordinates' stated convention, not on residuals:
    on the C2 and C3 deliveries the two settings differ by 0.010 m in the mean
    range residual and 0.0002 m in its scatter, so the data does not
    discriminate between them.

    See :ref:`orientation-station-estimation` for how the model enters the
    station geometry and which higher-order terms are left out.

    References
    ----------
    IERS Conventions (2010), IERS Technical Note 36, Chapter 7.
    https://iers-conventions.obspm.fr/content/chapter7/icc7.pdf
    """

    #: Central bodies with a nominal response, each a
    #: :class:`~scarabaeus.utils.Constants.SolidTideInfo` from Constants. Only
    #: bodies whose response is actually measured belong here: guessing one is
    #: a factor-of-several error, not a rounding one, so an unlisted body must
    #: be given explicit values.
    NOMINAL_RESPONSE = {"EARTH": constants.EARTH_SOLID_TIDE}

    tide_bodies: tuple = None
    central_body: str = "EARTH"
    love_h2: float = None
    shida_l2: float = None
    love_k: dict = None
    remove_permanent_tide: bool = False
    rate_step: float = 60.0
    model: str = "degree2"

    #: Displacement models: the degree-2 in-phase term, or the complete IERS
    #: Conventions (2010) routine.
    MODELS = ("degree2", "IERS2010")

    #: Row units of :meth:`state_displacement` and of its Love/Shida partials.
    STATE_UNITS = (km,) * 3 + (km / sec,) * 3

    class UnknownTidalResponseError(ValueError):
        """No tidal response is known for this body and none was supplied.

        Raised by the constructor. The degree-2 Love and Shida numbers differ
        by a factor of several between bodies, so falling back on Earth's
        would be a large silent error. Supply ``love_h2``, ``shida_l2`` and
        ``tide_bodies`` explicitly, or, once its response is actually known,
        give it a :class:`~scarabaeus.utils.Constants.SolidTideInfo` in
        Constants and list it in :attr:`NOMINAL_RESPONSE`.
        """

    class UnknownModelError(ValueError):
        """The displacement model is not one of :attr:`MODELS`, or the IERS
        2010 routine was asked for with a body or Love numbers it does not
        define: it is Earth's, with its own fixed numbers."""

    def __post_init__(self):
        """Validate the model and fill unset numbers from the nominal response.

        Raises
        ------
        UnknownModelError
            If :attr:`model` is not in :attr:`MODELS`, or ``"IERS2010"`` is
            asked for with a body other than Earth or with Love or Shida
            numbers other than Earth's.
        UnknownTidalResponseError
            If a number left unset has no nominal value for
            :attr:`central_body`, or :attr:`remove_permanent_tide` is set for
            a body other than Earth.
        ValueError
            If :attr:`tide_bodies` is empty, :attr:`rate_step` is not positive
            and finite, or a Love or Shida number is not finite.
        """
        if self.model not in self.MODELS:
            raise self.UnknownModelError(
                f"model must be one of {self.MODELS}, got {self.model!r}")
        if self.model == "IERS2010":
            if str(self.central_body).upper() != "EARTH":
                raise self.UnknownModelError(
                    f"the IERS 2010 routine is Earth's, not {self.central_body!r}")
            for name in ("love_h2", "shida_l2"):
                value = getattr(constants.EARTH_SOLID_TIDE, name)
                if getattr(self, name) not in (None, value):
                    raise self.UnknownModelError(
                        f"the IERS 2010 routine fixes {name} = {value}")
        nominal = self.NOMINAL_RESPONSE.get(str(self.central_body).upper())
        for name in ("love_h2", "shida_l2", "love_k", "tide_bodies"):
            if getattr(self, name) is None:
                if getattr(nominal, name, None) is None:
                    raise self.UnknownTidalResponseError(
                        f"No nominal {name} for central body "
                        f"{self.central_body!r}; supply it explicitly. Known "
                        f"bodies: {sorted(self.NOMINAL_RESPONSE)}")
                object.__setattr__(self, name, getattr(nominal, name))
        if not self.tide_bodies:
            raise ValueError("Provide at least one tide-raising body")
        if (self.remove_permanent_tide
                and str(self.central_body).upper() != "EARTH"):
            raise self.UnknownTidalResponseError(
                "remove_permanent_tide uses the Earth constants of IERS "
                f"eq. 7.14; there is no equivalent for {self.central_body!r}")
        if not np.isfinite(self.rate_step) or self.rate_step <= 0:
            raise ValueError("rate_step must be positive and finite")
        for name in ("love_h2", "shida_l2"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")

    def _scale(self) -> tuple:
        """Gravitational parameter and reference radius of the deformed body.

        Returns
        -------
        gm_central : float
            GM of :attr:`central_body`, km^3/s^2.
        equatorial_radius : float
            Its equatorial radius, km, the reference radius of the tide.

        Raises
        ------
        AttributeError
            If Constants has no entry named after :attr:`central_body`.
        """
        body = getattr(constants, str(self.central_body).upper())
        return float(body.GM.values), float(body.equatorial_radius.values)

    def displacement(self, position_body_fixed, epoch: float,
                     frame: str = None) -> ArrayWUnits:
        """Tidal displacement of a body-fixed point.

        Parameters
        ----------
        position_body_fixed : ArrayWUnits, ArrayWFrame or array_like, shape (3,)
            Undisplaced position in ``frame``, relative to the body centre.
            Bare numbers are km.
        epoch : float or EpochArray
            TDB seconds past J2000.
        frame : str or Frame, optional
            Body-fixed axes the position is given in and the displacement is
            returned in. Defaults to ``None``, ``'IAU_'`` plus the central body's
            name.

        Returns
        -------
        ArrayWUnits
            Displacement in km, shape (3,).

        Raises
        ------
        ValueError
            At the body centre, where the local vertical is undefined.
        """
        return ArrayWUnits.from_trusted(self._displacement(
            ArrayWUnits.values_in(position_body_fixed, km), epoch, frame), km)

    def _displacement(self, position_body_fixed, epoch: float,
                      frame: str = None) -> np.ndarray:
        """Numeric core of :meth:`displacement`: position in km, displacement
        in km, shape (3,)."""
        position = np.asarray(position_body_fixed, dtype=float).reshape(3)
        radius = np.linalg.norm(position)
        if radius <= 0:
            raise ValueError("Solid-tide displacement is undefined at the body centre")
        frame = getattr(frame, "name", frame) or f"IAU_{str(self.central_body).upper()}"
        epoch = float(getattr(epoch, "et", epoch))

        if self.model == "IERS2010":
            sun, moon = (np.asarray(spice.spkpos(body, epoch, frame, "NONE", "EARTH")[0],
                                    dtype=float) for body in ("SUN", "MOON"))
            # The routine takes the UTC hour of the day and TT centuries.
            julian_date_utc = float(spice.et2utc(epoch, "J", 14).split()[1])
            hours = ((julian_date_utc - 0.5) % 1.0) * 24.0
            centuries = spice.unitim(epoch, "TDB", "TDT") / (36525.0 * 86400.0)
            delta = self.iers2010_displacement(position * 1e3, hours, centuries,
                                               sun * 1e3, moon * 1e3) / 1e3
            if self.remove_permanent_tide:
                delta -= self._permanent_tide(position, frame)
            return delta

        unit = position / radius
        gm_central, reference_radius = self._scale()

        delta = np.zeros(3)
        for name in self.tide_bodies:
            body = getattr(constants, str(name).upper())
            raiser = np.asarray(spice.spkpos(
                str(name), epoch, frame, "NONE", str(self.central_body))[0], dtype=float)
            distance = np.linalg.norm(raiser)
            direction = raiser / distance
            cosine = float(direction @ unit)
            delta += (float(body.GM.values) / gm_central) * (
                reference_radius**4 / distance**3) * (
                self.love_h2 * unit * (1.5 * cosine**2 - 0.5)
                + 3.0 * self.shida_l2 * cosine * (direction - cosine * unit))

        if self.remove_permanent_tide:
            delta -= self._permanent_tide(position, frame)
        return delta

    def displacement_partials(self, position_body_fixed, epoch: float,
                              frame: str = None) -> ArrayWUnits:
        r"""Sensitivity of the displacement to the Love and Shida numbers.

        Parameters
        ----------
        position_body_fixed : ArrayWUnits, ArrayWFrame or array_like, shape (3,)
            Undisplaced position in ``frame``. Bare numbers are km.
        epoch : float or EpochArray
            TDB seconds past J2000.
        frame : str or Frame, optional
            Body-fixed axes; see :meth:`displacement`. Defaults to ``None``.

        Returns
        -------
        ArrayWUnits
            ``(3, 2)`` Jacobian, columns :math:`\partial\delta r/\partial h_2`
            and :math:`\partial\delta r/\partial l_2`, in km per unit.

        Raises
        ------
        ValueError
            At the body centre, where the local vertical is undefined.

        Notes
        -----
        The displacement is linear in :math:`h_2` and :math:`l_2`, so these
        columns are the two terms evaluated with a unit coefficient and are
        exact rather than a linearization. The permanent-tide term does not
        depend on the Love numbers -- it is a tabulated IERS correction -- so
        it contributes nothing here whichever way that option is set.
        """
        return ArrayWUnits.from_trusted(self._displacement_partials(
            ArrayWUnits.values_in(position_body_fixed, km), epoch, frame), km)

    def _displacement_partials(self, position_body_fixed, epoch: float,
                               frame: str = None) -> np.ndarray:
        """Numeric core of :meth:`displacement_partials`: position in km,
        ``(3, 2)`` Jacobian in km."""
        position = np.asarray(position_body_fixed, dtype=float).reshape(3)
        radius = np.linalg.norm(position)
        if radius <= 0:
            raise ValueError("Solid-tide displacement is undefined at the body centre")
        frame = getattr(frame, "name", frame) or f"IAU_{str(self.central_body).upper()}"
        epoch = float(getattr(epoch, "et", epoch))
        unit = position / radius
        gm_central, reference_radius = self._scale()

        love = np.zeros(3)
        shida = np.zeros(3)
        for name in self.tide_bodies:
            body = getattr(constants, str(name).upper())
            raiser = np.asarray(spice.spkpos(
                str(name), epoch, frame, "NONE", str(self.central_body))[0], dtype=float)
            distance = np.linalg.norm(raiser)
            direction = raiser / distance
            cosine = float(direction @ unit)
            scale = (float(body.GM.values) / gm_central) * (
                reference_radius**4 / distance**3)
            love += scale * unit * (1.5 * cosine**2 - 0.5)
            shida += scale * 3.0 * cosine * (direction - cosine * unit)
        return np.column_stack([love, shida])

    @staticmethod
    def _normalized_legendre(max_degree: int, sine: float) -> dict:
        r"""Normalized associated Legendre functions :math:`\bar P_{nm}(\sin\phi)`.

        Parameters
        ----------
        max_degree : int
            Highest degree required.
        sine : float
            Sine of the latitude.

        Returns
        -------
        dict
            ``(degree, order)`` mapped to the value, for every degree up to
            ``max_degree``.

        Notes
        -----
        Geodesy (4-pi) normalization,
        :math:`\bar P_{nm} = \sqrt{(2-\delta_{0m})(2n+1)\frac{(n-m)!}{(n+m)!}}
        \, P_{nm}`, matching the convention of the :math:`\bar C_{nm}` and
        :math:`\bar S_{nm}` these multiply. The Condon-Shortley phase is
        excluded, as it is throughout geodesy; including it would flip the
        sign of every odd order.
        """
        from math import factorial
        cosine = np.sqrt(max(1.0 - sine**2, 0.0))
        # Unnormalized, no Condon-Shortley phase.
        raw = {(0, 0): 1.0}
        for degree in range(1, max_degree + 1):
            raw[(degree, degree)] = (2 * degree - 1) * cosine * raw[(degree - 1, degree - 1)]
            raw[(degree, degree - 1)] = (2 * degree - 1) * sine * raw[(degree - 1, degree - 1)]
            for order in range(degree - 2, -1, -1):
                raw[(degree, order)] = (
                    (2 * degree - 1) * sine * raw[(degree - 1, order)]
                    - (degree + order - 1) * raw[(degree - 2, order)]) / (degree - order)
        out = {}
        for (degree, order), value in raw.items():
            factor = np.sqrt((2 - (order == 0)) * (2 * degree + 1)
                             * factorial(degree - order) / factorial(degree + order))
            out[(degree, order)] = factor * value
        return out

    def geopotential_coefficients(self, epoch: float, frame: str = None) -> dict:
        r"""Tidal perturbation of the body's spherical-harmonic coefficients.

        Parameters
        ----------
        epoch : float
            TDB seconds past J2000.
        frame : str, optional
            Body-fixed axes the tide-raiser positions are taken in. Defaults
            to ``None``, ``'IAU_'`` plus the central body's name.

        Returns
        -------
        dict
            ``(degree, order)`` mapped to ``(dC, dS)``, normalized, for every
            ``(degree, order)`` present in :attr:`love_k`. Add these to the
            nominal field.

        Notes
        -----
        IERS Conventions (2010), Chapter 6, eq. 6.6, step 1:

        .. math::

            \Delta\bar C_{nm} - i\,\Delta\bar S_{nm} =
                \frac{k_{nm}}{2n+1} \sum_j \frac{GM_j}{GM}
                \left(\frac{R}{r_j}\right)^{n+1}
                \bar P_{nm}(\sin\Phi_j)\, e^{-im\lambda_j}

        This is the *other* half of the same tide that :meth:`displacement`
        moves the surface with: the deformed mass changes the external field.
        It is driven by :attr:`love_k`, not by :math:`h_2` and :math:`l_2`.

        Every ``(degree, order)`` in :attr:`love_k` is summed, so the
        resolution follows that table rather than being fixed at degree 2. At
        Earth this reaches about 5e-9 in :math:`\Delta\bar C_{20}`, which is
        1e-5 of the nominal :math:`\bar C_{20}`; the degree-3 terms are three
        orders of magnitude smaller again.

        Each :math:`\Delta\bar C_{nm}` depends on its own :math:`k_{nm}` and
        no other, so the sensitivity is diagonal -- see
        :meth:`geopotential_partials`.

        Frequency-dependent step-2 corrections and the anelastic
        :math:`k^{+}` term are not included; both are well below the step-1
        terms retained here.

        Beware the base field's own tide convention. A geopotential published
        as *zero-tide* already contains the permanent deformation, so adding
        this on top counts it twice; a *tide-free* field does not. The
        difference is about 1e-8 in :math:`\bar C_{20}`, comparable to the
        whole tidal signal, and this method does not know which convention
        the loaded coefficients use.
        """
        frame = getattr(frame, "name", frame) or f"IAU_{str(self.central_body).upper()}"
        epoch = float(epoch)
        gm_central, reference_radius = self._scale()
        max_degree = max(degree for degree, _ in self.love_k)
        total = {key: 0j for key in self.love_k}
        for name in self.tide_bodies:
            body = getattr(constants, str(name).upper())
            raiser = np.asarray(spice.spkpos(
                str(name), epoch, frame, "NONE", str(self.central_body))[0], dtype=float)
            distance = np.linalg.norm(raiser)
            longitude = np.arctan2(raiser[1], raiser[0])
            legendre = self._normalized_legendre(max_degree, raiser[2] / distance)
            ratio = float(body.GM.values) / gm_central
            for (degree, order), love in self.love_k.items():
                total[(degree, order)] += (love / (2 * degree + 1)) * ratio * (
                    reference_radius / distance)**(degree + 1) * legendre[
                        (degree, order)] * np.exp(-1j * order * longitude)
        return {key: (float(value.real), float(-value.imag))
                for key, value in total.items()}

    def geopotential_partials(self, epoch: float, frame: str = None) -> dict:
        r"""Sensitivity of the coefficient perturbation to each :math:`k_{nm}`.

        Parameters
        ----------
        epoch : float
            TDB seconds past J2000.
        frame : str, optional
            Body-fixed axes; see :meth:`geopotential_coefficients`. Defaults
            to ``None``.

        Returns
        -------
        dict
            ``(degree, order)`` mapped to
            ``(dC/dk, dS/dk)`` for that same ``(degree, order)``.

        Notes
        -----
        :math:`\Delta\bar C_{nm}` depends on :math:`k_{nm}` alone and linearly,
        so the Jacobian is diagonal and each entry is exact rather than a
        linearization -- this is the same table evaluated at unit
        :math:`k_{nm}`.
        """
        unit = replace(self, love_k={key: 1.0 for key in self.love_k})
        return unit.geopotential_coefficients(epoch, frame)

    def state_displacement_partials(self, position_body_fixed, epoch: float,
                                    frame: str = None) -> ArrayWUnits:
        """Love/Shida sensitivity of the displacement and of its body-fixed rate.

        Parameters
        ----------
        position_body_fixed : ArrayWUnits, ArrayWFrame or array_like, shape (3,)
            Undisplaced position in ``frame``. Bare numbers are km.
        epoch : float or EpochArray
            TDB seconds past J2000.
        frame : str or Frame, optional
            Body-fixed axes; see :meth:`displacement`. Defaults to ``None``.

        Returns
        -------
        ArrayWUnits
            ``(6, 2)`` Jacobian in km and km/s per unit Love or Shida number,
            columns :math:`h_2` and :math:`l_2` as in
            :meth:`displacement_partials`.

        Raises
        ------
        ValueError
            At the body centre, where the local vertical is undefined.

        Notes
        -----
        The rate rows use the same central difference as
        :meth:`state_displacement`, so the partial is the derivative of what
        the forward model actually applies.
        """
        return ArrayWUnits.from_row_units(self._state_displacement_partials(
            ArrayWUnits.values_in(position_body_fixed, km), epoch, frame),
            self.STATE_UNITS)

    def _state_displacement_partials(self, position_body_fixed, epoch: float,
                                     frame: str = None) -> np.ndarray:
        """Numeric core of :meth:`state_displacement_partials`: position in
        km, ``(6, 2)`` Jacobian in km and km/s."""
        epoch = float(getattr(epoch, "et", epoch))
        here = self._displacement_partials(position_body_fixed, epoch, frame)
        ahead = self._displacement_partials(position_body_fixed, epoch + self.rate_step, frame)
        behind = self._displacement_partials(position_body_fixed, epoch - self.rate_step, frame)
        return np.vstack([here, (ahead - behind) / (2.0 * self.rate_step)])

    def _permanent_tide(self, position, frame: str) -> np.ndarray:
        """Zero-frequency deformation, km, from IERS eq. 7.14a and 7.14b.

        Radial and north components in metres, converted on return. ``sin(phi)``
        is the geocentric latitude sine, consistent with the degree-2 formula
        above, which is also written in geocentric terms.

        Parameters
        ----------
        position : array_like, shape (3,)
            Undisplaced position, km, in body-fixed axes whose z is the
            rotation axis.
        frame : str
            Name of those axes. Not used: the latitude is taken from the
            position itself.

        Returns
        -------
        numpy.ndarray
            Permanent-tide displacement, km, shape (3,), in the axes of the
            position. Only the radial and north components are non-zero; on
            the rotation axis the north one is dropped.
        """
        position = np.asarray(position, dtype=float).reshape(3)
        radius = np.linalg.norm(position)
        unit = position / radius
        sin_phi = unit[2]
        legendre = 1.5 * sin_phi**2 - 0.5
        radial_a, radial_b = constants.EARTH_SOLID_TIDE.permanent_tide_radial
        north_a, north_b = constants.EARTH_SOLID_TIDE.permanent_tide_north
        radial = (radial_a + radial_b * legendre) * legendre
        north_amplitude = (north_a + north_b * legendre) * 2.0 * sin_phi * np.sqrt(
            max(1.0 - sin_phi**2, 0.0))
        horizontal = np.hypot(position[0], position[1])
        if horizontal <= 0:
            north = np.zeros(3)
        else:
            east = np.array([-position[1], position[0], 0.0]) / horizontal
            north = np.cross(unit, east)
        return (radial * unit + north_amplitude * north) / 1e3

    def state_displacement(self, position_body_fixed, epoch: float,
                           frame: str = None) -> ArrayWUnits:
        """Tidal displacement and its body-fixed rate.

        Parameters
        ----------
        position_body_fixed : ArrayWUnits, ArrayWFrame or array_like, shape (3,)
            Undisplaced position in ``frame``. Bare numbers are km.
        epoch : float or EpochArray
            TDB seconds past J2000.
        frame : str or Frame, optional
            Body-fixed axes; see :meth:`displacement`. Defaults to ``None``.

        Returns
        -------
        ArrayWUnits
            Six components in km and km/s, shape (6,).

        Raises
        ------
        ValueError
            At the body centre, where the local vertical is undefined.

        Notes
        -----
        The rate is a central difference over :attr:`rate_step`, which is ample
        for a signal whose shortest period is twelve hours. It reaches roughly
        0.03 mm/s at Earth, comparable to the frame-rotation term the state
        transformation adds on top, and both matter against Doppler noise
        quoted near 1 mm/s.
        """
        return ArrayWUnits.from_row_units(self._state_displacement(
            ArrayWUnits.values_in(position_body_fixed, km), epoch, frame),
            self.STATE_UNITS)

    def _state_displacement(self, position_body_fixed, epoch: float,
                            frame: str = None) -> np.ndarray:
        """Numeric core of :meth:`state_displacement`: position in km,
        displacement and rate in km and km/s, shape (6,)."""
        epoch = float(getattr(epoch, "et", epoch))
        position = np.asarray(position_body_fixed, dtype=float).reshape(3)
        delta = self._displacement(position, epoch, frame)
        ahead = self._displacement(position, epoch + self.rate_step, frame)
        behind = self._displacement(position, epoch - self.rate_step, frame)
        return np.concatenate([delta, (ahead - behind) / (2.0 * self.rate_step)])

    # ------------------------------------------------------------------ #
    # IERS Conventions (2010) DEHANTTIDEINEL, ported line by line from   #
    # the IERS software (iers-conventions.obspm.fr, chapter 7):          #
    # DEHANTTIDEINEL, ST1IDIU, ST1ISEM, ST1L1, STEP2DIU and STEP2LON.    #
    # Metres throughout, as there. Every number is                       #
    # Constants.EARTH_SOLID_TIDE; the port reproduces the routine's      #
    # published test cases exactly.                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def iers2010_displacement(cls, xsta, fhr, t, xsun, xmon) -> np.ndarray:
        """IERS Conventions (2010) solid-tide station displacement, DEHANTTIDEINEL.

        What :meth:`displacement` evaluates for ``model="IERS2010"``, with the
        routine's own interface and units.

        Parameters
        ----------
        xsta : array_like, shape (3,)
            Station position, m, Earth-fixed (ITRF).
        fhr : float
            UTC hours of the day.
        t : float
            TT Julian centuries from J2000.
        xsun, xmon : array_like, shape (3,)
            Geocentric Sun and Moon positions, m, in the same Earth-fixed axes.

        Returns
        -------
        numpy.ndarray
            Displacement in m, Earth-fixed axes: step 1 in and out of phase,
            degrees 2 and 3, with latitude-dependent Love and Shida numbers,
            and the step-2 diurnal and long-period frequency corrections. The
            permanent tide is included, as for conventional tide-free
            coordinates.

        Notes
        -----
        The numbers are :attr:`Constants.EARTH_SOLID_TIDE`; the reference
        radius is ``Constants.EARTH.equatorial_radius``.
        """
        tide = constants.EARTH_SOLID_TIDE
        xsta, xsun, xmon = (np.asarray(v, dtype=float).reshape(3) for v in (xsta, xsun, xmon))
        rsta, rsun, rmon = np.linalg.norm(xsta), np.linalg.norm(xsun), np.linalg.norm(xmon)
        scsun = (xsta @ xsun) / rsta / rsun
        scmon = (xsta @ xmon) / rsta / rmon
        cosphi = np.hypot(xsta[0], xsta[1]) / rsta
        h2 = tide.love_h2 + tide.love_h2_latitude * (1 - 3 / 2 * cosphi**2)
        l2 = tide.shida_l2 + tide.shida_l2_latitude * (1 - 3 / 2 * cosphi**2)
        h3, l3 = tide.love_h3, tide.shida_l3
        p2sun = 3 * (h2 / 2 - l2) * scsun**2 - h2 / 2
        p2mon = 3 * (h2 / 2 - l2) * scmon**2 - h2 / 2
        p3sun = 5 / 2 * (h3 - 3 * l3) * scsun**3 + 3 / 2 * (l3 - h3) * scsun
        p3mon = 5 / 2 * (h3 - 3 * l3) * scmon**3 + 3 / 2 * (l3 - h3) * scmon
        x2sun, x2mon = 3 * l2 * scsun, 3 * l2 * scmon
        x3sun = 3 * l3 / 2 * (5 * scsun**2 - 1)
        x3mon = 3 * l3 / 2 * (5 * scmon**2 - 1)
        re = float(constants.EARTH.equatorial_radius.values) * 1e3
        fac2sun = tide.mass_ratio_sun * re * (re / rsun)**3
        fac2mon = tide.mass_ratio_moon * re * (re / rmon)**3
        fac3sun = fac2sun * (re / rsun)
        fac3mon = fac2mon * (re / rmon)
        dx = (fac2sun * (x2sun * xsun / rsun + p2sun * xsta / rsta)
              + fac2mon * (x2mon * xmon / rmon + p2mon * xsta / rsta)
              + fac3sun * (x3sun * xsun / rsun + p3sun * xsta / rsta)
              + fac3mon * (x3mon * xmon / rmon + p3mon * xsta / rsta))
        factors = (xsta, xsun, xmon, fac2sun, fac2mon)
        dx = dx + cls._iers2010_out_of_phase_diurnal(*factors)
        dx = dx + cls._iers2010_out_of_phase_semidiurnal(*factors)
        dx = dx + cls._iers2010_latitude_l1(*factors)
        dx = dx + cls._iers2010_step2_diurnal(xsta, fhr, t)
        return dx + cls._iers2010_step2_long_period(xsta, t)

    @staticmethod
    def _iers2010_geometry(xsta) -> tuple:
        """Radius and geocentric latitude/longitude sines and cosines.

        Parameters
        ----------
        xsta : numpy.ndarray, shape (3,)
            Station position, m, Earth-fixed.

        Returns
        -------
        rsta : float
            Geocentric distance, m.
        sinphi, cosphi : float
            Sine and cosine of the geocentric latitude.
        sinla, cosla : float
            Sine and cosine of the longitude; not finite on the rotation
            axis.
        """
        rsta = np.linalg.norm(xsta)
        sinphi = xsta[2] / rsta
        cosphi = np.hypot(xsta[0], xsta[1]) / rsta
        return rsta, sinphi, cosphi, xsta[1] / cosphi / rsta, xsta[0] / cosphi / rsta

    @staticmethod
    def _iers2010_to_xyz(dr, de, dn, sinphi, cosphi, sinla, cosla) -> np.ndarray:
        """Radial, east and north components to the Earth-fixed axes.

        Parameters
        ----------
        dr, de, dn : float
            Radial, east and north components, in any length unit.
        sinphi, cosphi : float
            Sine and cosine of the geocentric latitude.
        sinla, cosla : float
            Sine and cosine of the longitude.

        Returns
        -------
        numpy.ndarray
            The same vector in Earth-fixed axes, shape (3,), in the unit of
            the components.
        """
        return np.array([dr * cosla * cosphi - de * sinla - dn * sinphi * cosla,
                         dr * sinla * cosphi + de * cosla - dn * sinphi * sinla,
                         dr * sinphi + dn * cosphi])

    @classmethod
    def _iers2010_out_of_phase_diurnal(cls, xsta, xsun, xmon, fac2sun,
                                       fac2mon) -> np.ndarray:
        """ST1IDIU: out-of-phase diurnal corrections, mantle anelasticity.

        Parameters
        ----------
        xsta : numpy.ndarray, shape (3,)
            Station position, m, Earth-fixed.
        xsun, xmon : numpy.ndarray, shape (3,)
            Geocentric Sun and Moon positions, m, in the same axes.
        fac2sun, fac2mon : float
            Degree-2 scale of each body,
            :math:`(GM_j/GM_E) R_E (R_E/r_j)^3`, m.

        Returns
        -------
        numpy.ndarray
            Correction, m, Earth-fixed, shape (3,).
        """
        dhi, dli = constants.EARTH_SOLID_TIDE.out_of_phase_diurnal
        _, sinphi, cosphi, sinla, cosla = cls._iers2010_geometry(xsta)
        cos2phi = cosphi * cosphi - sinphi * sinphi
        dr = dn = de = 0.0
        for x, fac in ((xsun, fac2sun), (xmon, fac2mon)):
            r = np.linalg.norm(x)
            dr += -3 * dhi * sinphi * cosphi * fac * x[2] * (x[0] * sinla - x[1] * cosla) / r**2
            dn += -3 * dli * cos2phi * fac * x[2] * (x[0] * sinla - x[1] * cosla) / r**2
            de += -3 * dli * sinphi * fac * x[2] * (x[0] * cosla + x[1] * sinla) / r**2
        return cls._iers2010_to_xyz(dr, de, dn, sinphi, cosphi, sinla, cosla)

    @classmethod
    def _iers2010_out_of_phase_semidiurnal(cls, xsta, xsun, xmon, fac2sun,
                                           fac2mon) -> np.ndarray:
        """ST1ISEM: out-of-phase semidiurnal corrections.

        Parameters
        ----------
        xsta : numpy.ndarray, shape (3,)
            Station position, m, Earth-fixed.
        xsun, xmon : numpy.ndarray, shape (3,)
            Geocentric Sun and Moon positions, m, in the same axes.
        fac2sun, fac2mon : float
            Degree-2 scale of each body,
            :math:`(GM_j/GM_E) R_E (R_E/r_j)^3`, m.

        Returns
        -------
        numpy.ndarray
            Correction, m, Earth-fixed, shape (3,).
        """
        dhi, dli = constants.EARTH_SOLID_TIDE.out_of_phase_semidiurnal
        _, sinphi, cosphi, sinla, cosla = cls._iers2010_geometry(xsta)
        costwola = cosla * cosla - sinla * sinla
        sintwola = 2 * cosla * sinla
        dr = dn = de = 0.0
        for x, fac in ((xsun, fac2sun), (xmon, fac2mon)):
            r = np.linalg.norm(x)
            a = (x[0]**2 - x[1]**2) * sintwola - 2 * x[0] * x[1] * costwola
            b = (x[0]**2 - x[1]**2) * costwola + 2 * x[0] * x[1] * sintwola
            dr += -3 / 4 * dhi * cosphi**2 * fac * a / r**2
            dn += 3 / 2 * dli * sinphi * cosphi * fac * a / r**2
            de += -3 / 2 * dli * cosphi * fac * b / r**2
        return cls._iers2010_to_xyz(dr, de, dn, sinphi, cosphi, sinla, cosla)

    @classmethod
    def _iers2010_latitude_l1(cls, xsta, xsun, xmon, fac2sun,
                              fac2mon) -> np.ndarray:
        """ST1L1: transverse corrections from the latitude dependence of l^(1).

        Parameters
        ----------
        xsta : numpy.ndarray, shape (3,)
            Station position, m, Earth-fixed.
        xsun, xmon : numpy.ndarray, shape (3,)
            Geocentric Sun and Moon positions, m, in the same axes.
        fac2sun, fac2mon : float
            Degree-2 scale of each body,
            :math:`(GM_j/GM_E) R_E (R_E/r_j)^3`, m.

        Returns
        -------
        numpy.ndarray
            Diurnal plus semidiurnal correction, m, Earth-fixed, shape (3,).
        """
        l1d, l1sd = constants.EARTH_SOLID_TIDE.shida_l1
        _, sinphi, cosphi, sinla, cosla = cls._iers2010_geometry(xsta)
        dn = de = 0.0
        for x, fac in ((xsun, fac2sun), (xmon, fac2mon)):
            r = np.linalg.norm(x)
            dn += -l1d * sinphi**2 * fac * x[2] * (x[0] * cosla + x[1] * sinla) / r**2
            de += (l1d * sinphi * (cosphi**2 - sinphi**2) * fac * x[2]
                   * (x[0] * sinla - x[1] * cosla) / r**2)
        de, dn = 3 * de, 3 * dn
        diurnal = np.array([-de * sinla - dn * sinphi * cosla,
                            de * cosla - dn * sinphi * sinla, dn * cosphi])
        costwola = cosla**2 - sinla**2
        sintwola = 2.0 * cosla * sinla
        dn = de = 0.0
        for x, fac in ((xsun, fac2sun), (xmon, fac2mon)):
            r = np.linalg.norm(x)
            dn += (-l1sd / 2 * sinphi * cosphi * fac
                   * ((x[0]**2 - x[1]**2) * costwola + 2 * x[0] * x[1] * sintwola) / r**2)
            de += (-l1sd / 2 * sinphi**2 * cosphi * fac
                   * ((x[0]**2 - x[1]**2) * sintwola - 2 * x[0] * x[1] * costwola) / r**2)
        de, dn = 3 * de, 3 * dn
        return diurnal + np.array([-de * sinla - dn * sinphi * cosla,
                                   de * cosla - dn * sinphi * sinla, dn * cosphi])

    @staticmethod
    def _iers2010_arguments(t) -> dict:
        """The step-2 argument polynomials evaluated at ``t``.

        Nested as the routine writes them, so the results are the routine's
        to the bit.

        Parameters
        ----------
        t : float
            TT Julian centuries from J2000.

        Returns
        -------
        dict
            Argument name mapped to its value in degrees, not reduced to
            [0, 360), keyed as in
            ``Constants.EARTH_SOLID_TIDE.step2_arguments``.
        """
        arguments = {}
        for name, coefficients in constants.EARTH_SOLID_TIDE.step2_arguments.items():
            value = 0.0
            for coefficient in reversed(coefficients):
                value = value * t + coefficient
            arguments[name] = value
        return arguments

    @classmethod
    def _iers2010_step2_diurnal(cls, xsta, fhr, t) -> np.ndarray:
        """STEP2DIU: frequency-dependent corrections, diurnal band.

        Parameters
        ----------
        xsta : numpy.ndarray, shape (3,)
            Station position, m, Earth-fixed.
        fhr : float
            UTC hours of the day.
        t : float
            TT Julian centuries from J2000.

        Returns
        -------
        numpy.ndarray
            Correction, m, Earth-fixed, shape (3,).
        """
        arguments = cls._iers2010_arguments(t)
        tau = fhr * 15 + arguments["GMST"] + (-arguments["s"])
        s, tau, h, p, zns, ps = (np.fmod(value, 360.0) for value in (
            arguments["s"] + arguments["pr"], tau, arguments["h"], arguments["p"],
            arguments["N'"], arguments["ps"]))
        _, sinphi, cosphi, sinla, cosla = cls._iers2010_geometry(xsta)
        zla = np.arctan2(xsta[1], xsta[0])
        out = np.zeros(3)
        for row in constants.EARTH_SOLID_TIDE.step2_diurnal:
            theta = np.deg2rad(tau + row[0] * s + row[1] * h + row[2] * p
                               + row[3] * zns + row[4] * ps) + zla
            dr = (row[5] * 2 * sinphi * cosphi * np.sin(theta)
                  + row[6] * 2 * sinphi * cosphi * np.cos(theta))
            dn = (row[7] * (cosphi**2 - sinphi**2) * np.sin(theta)
                  + row[8] * (cosphi**2 - sinphi**2) * np.cos(theta))
            de = row[7] * sinphi * np.cos(theta) - row[8] * sinphi * np.sin(theta)
            out = out + cls._iers2010_to_xyz(dr, de, dn, sinphi, cosphi, sinla, cosla)
        return out / 1000.0

    @classmethod
    def _iers2010_step2_long_period(cls, xsta, t) -> np.ndarray:
        """STEP2LON: frequency-dependent corrections, long-period band.

        Parameters
        ----------
        xsta : numpy.ndarray, shape (3,)
            Station position, m, Earth-fixed.
        t : float
            TT Julian centuries from J2000.

        Returns
        -------
        numpy.ndarray
            Correction, m, Earth-fixed, shape (3,).
        """
        arguments = cls._iers2010_arguments(t)
        s, h, p, zns, ps = (np.fmod(value, 360.0) for value in (
            arguments["s"] + arguments["pr"], arguments["h"], arguments["p"],
            arguments["N'"], arguments["ps"]))
        _, sinphi, cosphi, sinla, cosla = cls._iers2010_geometry(xsta)
        out = np.zeros(3)
        for row in constants.EARTH_SOLID_TIDE.step2_long_period:
            theta = np.deg2rad(row[0] * s + row[1] * h + row[2] * p + row[3] * zns + row[4] * ps)
            dr = (row[5] * (3 * sinphi**2 - 1) / 2 * np.cos(theta)
                  + row[6] * (3 * sinphi**2 - 1) / 2 * np.sin(theta))
            dn = (row[7] * (cosphi * sinphi * 2) * np.cos(theta)
                  + row[8] * (cosphi * sinphi * 2) * np.sin(theta))
            out = out + cls._iers2010_to_xyz(dr, 0.0, dn, sinphi, cosphi, sinla, cosla)
        return out / 1000.0
