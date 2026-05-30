# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Units, ArrayWUnits, EpochArray

# import numpy as np
import scarabaeus.utils.NumpyWrapper as np

# ------------------#
#  Generate Units  #
# ------------------#
kg, km, sec, N, kN = Units.get_units(["kg", "km", "sec", "N", "kN"])


# --------------------#
#  Class Definition  #
# --------------------#
class Maneuver:
    """ Defines a finite-burn maneuver as a collection of time-polynomial parameters.

    Each scalar parameter — thrust magnitude :math:`F(t)`, mass-flow rate
    :math:`\\dot{m}(t)`, and thrust-direction components
    :math:`u_x(t),\\,u_y(t),\\,u_z(t)` — is represented as a 6th-order
    polynomial in ephemeris time:

    .. math::

        p(t) = \\sum_{k=0}^{6} c_k\\, t^k

    and stored as a 7-element coefficient vector
    :math:`[c_0, c_1, \\ldots, c_6]`.  Constant profiles set only
    :math:`c_0`; the remaining coefficients default to zero.

    The thrust-induced acceleration at time :math:`t` is:

    .. math::

        \\mathbf{a}(t) = \\frac{F(t)}{m(t)}\\,\\hat{u}(t)

    where :math:`m(t)` is the spacecraft mass and
    :math:`\\hat{u}(t) = [u_x,\\,u_y,\\,u_z]^T` is the unit thrust-direction
    vector.  Thrust is stored internally in **kN** so that dividing by mass
    in kg directly yields acceleration in km/s².

    Parameters
    ----------
    thrust : ArrayWUnits, optional
        Thrust magnitude polynomial (any force unit; stored internally in kN).
    mass_flow : ArrayWUnits, optional
        Propellant mass-flow rate polynomial (kg/s).
    start_time : ArrayWUnits, optional
        Burn start epoch (TDB seconds). Must be strictly less than ``end_time``.
    end_time : ArrayWUnits, optional
        Burn end epoch (TDB seconds). Must be strictly greater than ``start_time``.
    ux : ArrayWUnits, optional
        X-component of the unit thrust-direction vector (unitless polynomial).
    uy : ArrayWUnits, optional
        Y-component of the unit thrust-direction vector (unitless polynomial).
    uz : ArrayWUnits, optional
        Z-component of the unit thrust-direction vector (unitless polynomial).

    Notes
    -----
    All polynomial coefficient arrays are zero-padded to length 7 (6th-order)
    and stored in ascending degree order internally (index 0 = constant term).
    Thrust is always converted to and stored in **kN** so that dividing by
    spacecraft mass in kg yields acceleration directly in km/s².

    See Also
    --------
    scarabaeus.FiniteBurn : Force model that evaluates Maneuver polynomials during propagation.
    scarabaeus.ManeuverParser : Utility for loading Maneuver objects from a file.
    """

    def __init__(
        self,
        thrust=None,
        mass_flow=None,
        start_time=None,
        end_time=None,
        ux=None,
        uy=None,
        uz=None,
    ):
        self.thrust = thrust
        self.mass_flow = mass_flow
        self.start_time = start_time
        self.end_time = end_time
        self.ux = ux
        self.uy = uy
        self.uz = uz

    @property
    def thrust(self) -> ArrayWUnits | None:
        """
        Thrust-magnitude polynomial stored in kN.

        The coefficient array is in **descending** degree order: index 0 holds
        the highest-order (t⁶) coefficient and index 6 holds the constant term.

        Returns
        -------
        ArrayWUnits or None
            7-element polynomial with units ``[kN/s⁶, kN/s⁵, …, kN]``
            (highest to lowest degree), or ``None`` if not set.
        """
        return self._thrust

    @thrust.setter
    def thrust(self, value):
        """
        Set the thrust-magnitude polynomial, converting to kN.

        The input is unit-converted to **kN** so that dividing by spacecraft
        mass (kg) directly yields acceleration in km/s².  The coefficient
        array is zero-padded on the left to a length of 7 (6th-order
        polynomial) and reversed so that index 0 holds the highest-order term.

        Parameters
        ----------
        value : ArrayWUnits or None
            Thrust polynomial in any force unit (N, kN, …).  ``None`` clears
            the attribute.
        """
        if value is None:
            self._thrust = value
            return
        else:
            # Convert to kN so that thrust / mass_sc gives km/s² directly (kN/kg = km/s²).
            # For a polynomial input (multiple units), scale all coefficients by the force
            # unit → kN factor derived from the first (or only) unit in the array.
            raw_units = value.units
            base_unit = (
                raw_units[0] if isinstance(raw_units, (list, np.ndarray)) else raw_units
            )
            n_to_kn = float(
                ArrayWUnits(np.array([1.0]), base_unit).convert_to(kN).values
            )
            values = np.atleast_1d(np.array(value.values, dtype=float)) * n_to_kn
            pad_length = 7 - len(values)
            if pad_length > 0:
                padded_values = np.pad(
                    values, (pad_length, 0), mode="constant", constant_values=0
                )
                reversed_values = padded_values[::-1]
                degree = len(reversed_values) - 1
                reversed_units = [
                    kN / (sec ** (degree - i)) for i in reversed(range(degree + 1))
                ]
                self._thrust = ArrayWUnits(reversed_values, reversed_units)
            else:
                reversed_values = values[::-1]
                degree = len(reversed_values) - 1
                reversed_units = [
                    kN / (sec ** (degree - i)) for i in reversed(range(degree + 1))
                ]
                self._thrust = ArrayWUnits(reversed_values, reversed_units)

    @property
    def mass_flow(self) -> ArrayWUnits | None:
        """
        Propellant mass-flow rate polynomial (kg/s).

        The coefficient array is in **descending** degree order: index 0 holds
        the highest-order (t⁶) coefficient and index 6 holds the constant term.

        Returns
        -------
        ArrayWUnits or None
            7-element polynomial with units ``[kg/s⁷, kg/s⁶, …, kg/s]``
            (highest to lowest degree), or ``None`` if not set.
        """
        return self._mass_flow

    @mass_flow.setter
    def mass_flow(self, value):
        """
        Set the propellant mass-flow rate polynomial (kg/s).

        Zero-pads the coefficient array to length 7, then reverses it so that
        index 0 is the highest-order (s⁻⁷) coefficient.  Units are fixed to
        ``kg / sⁿ`` for each polynomial degree ``n``.

        Parameters
        ----------
        value : ArrayWUnits or None
            Mass-flow polynomial in ``kg/s`` (or compatible unit).  ``None``
            clears the attribute.
        """
        if value is None:
            self._mass_flow = value
            return

        values = np.atleast_1d(np.array(value.values, dtype=float))
        original_units = (
            list(value.units)
            if isinstance(value.units, (list, np.ndarray))
            else [value.units]
        )
        pad_length = 7 - len(values)

        if pad_length > 0:
            padded_values = np.pad(
                values, (pad_length, 0), mode="constant", constant_values=0
            )
            reversed_values = padded_values[::-1]
            degree = len(reversed_values) - 1
            reversed_units = [
                kg / (sec ** (i + 1)) for i in reversed(range(degree + 1))
            ]
            self._mass_flow = ArrayWUnits(reversed_values, reversed_units)
        else:
            reversed_values = values[::-1]
            degree = len(reversed_values) - 1
            reversed_units = [
                kg / (sec ** (i + 1)) for i in reversed(range(degree + 1))
            ]
            self._mass_flow = ArrayWUnits(reversed_values, reversed_units)

    @property
    def start_time(self) -> EpochArray | None:
        """
        Burn start epoch in seconds (TDB).

        Returns
        -------
        EpochArray or None
        """
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        """
        Set the burn start epoch.

        Parameters
        ----------
        value : EpochArray or None
            Burn start epoch in TDB seconds.  Must be non-negative and strictly
            less than the current ``end_time`` (if one has already been set).
            ``None`` clears the attribute.

        Raises
        ------
        TypeError
            If *value* is not an ``EpochArray``.
        ValueError
            If *value* is negative, or greater than or equal to ``end_time``.
        """
        if value is not None and not isinstance(value, EpochArray):
            raise TypeError("Start time must be EpochArray")
        if value is not None and value.times.values < 0:
            raise ValueError("Start time cannot be negative.")
        end = getattr(self, "_end_time", None)
        if (
            value is not None
            and end is not None
            and value.times.values >= end.times.values
        ):
            raise ValueError("Start time cannot be greater than or equal to end time.")
        self._start_time = value

    @property
    def end_time(self) -> EpochArray | None:
        """
        Burn end epoch in seconds (TDB).

        Returns
        -------
        EpochArray or None
        """
        return self._end_time

    @end_time.setter
    def end_time(self, value):
        """
        Set the burn end epoch.

        Parameters
        ----------
        value : EpochArray or None
            Burn end epoch in TDB seconds.  Must be non-negative and strictly
            greater than the current ``start_time`` (if one has already been set).
            ``None`` clears the attribute.

        Raises
        ------
        TypeError
            If *value* is not an ``EpochArray``.
        ValueError
            If *value* is negative, or less than or equal to ``start_time``.
        """
        if value is not None and not isinstance(value, EpochArray):
            raise TypeError("End time must be EpochArray")
        if value is not None and value.times.values < 0:
            raise ValueError("End time cannot be negative.")
        start = getattr(self, "_start_time", None)
        if (
            value is not None
            and start is not None
            and value.times.values <= start.times.values
        ):
            raise ValueError("End time cannot be less than or equal to start time.")
        self._end_time = value

    @property
    def ux(self) -> ArrayWUnits | None:
        """
        X-component of the unit thrust-direction vector (unitless polynomial).

        Returns
        -------
        ArrayWUnits or None
        """
        return self._ux

    @ux.setter
    def ux(self, value):
        """
        Set the X thrust-direction polynomial (unitless).

        Zero-pads to length 7 and reverses coefficient order (index 0 =
        highest degree).  Units from *value* are preserved; padded leading
        coefficients receive ``None`` as their unit.

        Parameters
        ----------
        value : ArrayWUnits or None
            X-component direction polynomial.  ``None`` clears the attribute.
        """
        if value is None:
            self._ux = value
            return

        values = np.atleast_1d(np.array(value.values, dtype=float))
        original_units = (
            list(value.units)
            if isinstance(value.units, (list, np.ndarray))
            else [value.units]
        )
        pad_length = 7 - len(values)

        if pad_length > 0:
            padded_values = np.pad(
                values, (pad_length, 0), mode="constant", constant_values=0
            )
            reversed_values = padded_values[::-1]
            padded_units = [None] * pad_length + original_units
            reversed_units = padded_units[::-1]

            self._ux = ArrayWUnits(reversed_values, reversed_units)
        else:
            reversed_values = values[::-1]
            reversed_units = original_units[::-1]
            self._ux = ArrayWUnits(reversed_values, reversed_units)

    @property
    def uy(self) -> ArrayWUnits | None:
        """
        Y-component of the unit thrust-direction vector (unitless polynomial).

        Returns
        -------
        ArrayWUnits or None
        """
        return self._uy

    @uy.setter
    def uy(self, value):
        """
        Set the Y thrust-direction polynomial (unitless).

        Zero-pads to length 7 and reverses coefficient order (index 0 =
        highest degree).  Units from *value* are preserved; padded leading
        coefficients receive ``None`` as their unit.

        Parameters
        ----------
        value : ArrayWUnits or None
            Y-component direction polynomial.  ``None`` clears the attribute.
        """
        if value is None:
            self._uy = value
            return

        values = np.atleast_1d(np.array(value.values, dtype=float))
        original_units = (
            list(value.units)
            if isinstance(value.units, (list, np.ndarray))
            else [value.units]
        )
        pad_length = 7 - len(values)

        if pad_length > 0:
            padded_values = np.pad(
                values, (pad_length, 0), mode="constant", constant_values=0
            )
            reversed_values = padded_values[::-1]
            padded_units = [None] * pad_length + original_units
            reversed_units = padded_units[::-1]

            self._uy = ArrayWUnits(reversed_values, reversed_units)
        else:
            reversed_values = values[::-1]
            reversed_units = original_units[::-1]
            self._uy = ArrayWUnits(reversed_values, reversed_units)

    @property
    def uz(self) -> ArrayWUnits | None:
        """
        Z-component of the unit thrust-direction vector (unitless polynomial).

        Returns
        -------
        ArrayWUnits or None
        """
        return self._uz

    @uz.setter
    def uz(self, value):
        """
        Set the Z thrust-direction polynomial (unitless).

        Zero-pads to length 7 and reverses coefficient order (index 0 =
        highest degree).  Units from *value* are preserved; padded leading
        coefficients receive ``None`` as their unit.

        Parameters
        ----------
        value : ArrayWUnits or None
            Z-component direction polynomial.  ``None`` clears the attribute.
        """
        if value is None:
            self._uz = value
            return

        values = np.atleast_1d(np.array(value.values, dtype=float))
        original_units = (
            list(value.units)
            if isinstance(value.units, (list, np.ndarray))
            else [value.units]
        )
        pad_length = 7 - len(values)

        if pad_length > 0:
            padded_values = np.pad(
                values, (pad_length, 0), mode="constant", constant_values=0
            )
            reversed_values = padded_values[::-1]
            padded_units = [None] * pad_length + original_units
            reversed_units = padded_units[::-1]

            self._uz = ArrayWUnits(reversed_values, reversed_units)
        else:
            reversed_values = values[::-1]
            reversed_units = original_units[::-1]
            self._uz = ArrayWUnits(reversed_values, reversed_units)
