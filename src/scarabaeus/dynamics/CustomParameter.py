# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    StateDefinition,
    Units,
)

import numpy as np

km, sec = Units.get_units(["km", "sec"])

# Units of an acceleration contribution, built once: every CustomParameter
# derives its partial-derivative units from it.
_U_KM_S2 = km * sec**-2


class CustomParameter:
    """ Declaration of a user-defined solve-for parameter of a custom force model.

    A ``CustomParameter`` is the script-side counterpart of the built-in
    estimable parameters (``eta_srp``, ``gm``, ``a_fogm``, ...): it names a new
    component that may be added to a :class:`~scarabaeus.StateArray`, estimated
    by the orbit-determination filters, and read back inside the acceleration
    function of a :class:`~scarabaeus.CustomDynamicModel` — all from a user
    script, with no change to the library source.

    Constructing the object registers *name* with
    :meth:`~scarabaeus.StateDefinition.register_custom_parameter`, so the
    parameter passes state-vector validation from that point on.

    Two flavours are supported:

    * **static** (``dynamics=None``, the default) — a constant to be estimated,
      e.g. a drag coefficient.  The filter needs only
      :math:`\\partial \\mathbf{a} / \\partial p`.
    * **dynamic** (``dynamics`` supplied) — a parameter with its own equation of
      motion :math:`\\dot p = f(\\mathbf r, \\mathbf v, p, t)`, integrated along
      with the state.  The filter additionally needs
      :math:`\\partial \\dot p / \\partial p`,
      :math:`\\partial \\dot p / \\partial \\mathbf r` and
      :math:`\\partial \\dot p / \\partial \\mathbf v`.

    Every Jacobian defaults to a central finite difference of the corresponding
    user function, so a working setup needs no hand-derived partials; supply the
    analytic callables when accuracy or speed matters.

    Parameters
    ----------
    name : str
        State-component name, e.g. ``"drag_cd"``. Must be a plain identifier and
        must not collide with a built-in component name.
    dimension : int, optional
        Number of scalar elements. Defaults to 1.
    units : Units, optional
        Units of the parameter itself. Used for the state-vector physical-type
        check and to label the partial derivatives. ``None`` (default) means
        dimensionless / unchecked.
    default : float or array_like, optional
        Fallback value used when the parameter is absent from the propagated
        state (i.e. when the model runs without estimating it). Defaults to
        zeros.
    physical_type : str, optional
        Unit-dimension name enforced on the state entry (e.g. ``"Acceleration"``).
        Defaults to ``None``, which skips the check.
    partial : callable, optional
        :math:`\\partial \\mathbf{a} / \\partial p`, returning a ``(3, dimension)``
        array. Defaults to a finite difference of the model's acceleration.
    dynamics : callable, optional
        :math:`\\dot p`, returning a ``(dimension,)`` array. Supplying it makes
        the parameter dynamic; it must then be added to the state with
        ``dynamics="dynamic"``.
    partial_dynamics_by_param : callable, optional
        :math:`\\partial \\dot p / \\partial p`, returning ``(dimension, dimension)``.
        Defaults to a finite difference of ``dynamics``.
    partial_dynamics_by_position : callable, optional
        :math:`\\partial \\dot p / \\partial \\mathbf r`, returning ``(dimension, 3)``.
        Defaults to a finite difference of ``dynamics``.
    partial_dynamics_by_velocity : callable, optional
        :math:`\\partial \\dot p / \\partial \\mathbf v`, returning ``(dimension, 3)``.
        Defaults to a finite difference of ``dynamics``.
    step : float or array_like, optional
        Absolute finite-difference step used for this parameter. Defaults to
        ``1e-6 * max(|p|, 1)`` evaluated at each call.
    register : bool, optional
        Register the name with :class:`~scarabaeus.StateDefinition` on
        construction. Defaults to ``True``.

    See Also
    --------
    scarabaeus.CustomDynamicModel : Force model that consumes these parameters.
    scarabaeus.StateDefinition : State builder; see ``custom_param``.

    Notes
    -----
    All user callables are called with keyword arguments drawn from the
    evaluation context; declare only the ones you need. See
    :class:`~scarabaeus.CustomDynamicModel` for the available names.

    Examples
    --------
    A drag coefficient estimated as a constant:

    .. code-block:: python

        import scarabaeus as scb

        cd = scb.CustomParameter("drag_cd", default=2.2)

    An exponentially decaying empirical scale factor:

    .. code-block:: python

        tau = 3600.0
        k = scb.CustomParameter(
            "emp_scale",
            default=1.0,
            dynamics=lambda params: -params["emp_scale"] / tau,
        )
    """

    class DimensionError(ValueError):
        """Raised when a supplied value does not match the declared dimension."""

    # -------------#
    # Constructor  #
    # -------------#
    def __init__(
        self,
        name: str,
        dimension: int = 1,
        units: Units | None = None,
        default=None,
        physical_type: str | None = None,
        partial=None,
        dynamics=None,
        partial_dynamics_by_param=None,
        partial_dynamics_by_position=None,
        partial_dynamics_by_velocity=None,
        step=None,
        register: bool = True,
    ):
        self._name = str(name)
        self._dimension = int(dimension)
        self._units = units
        self._physical_type = physical_type

        if default is None:
            default_values = np.zeros(self._dimension)
        else:
            default_values = np.atleast_1d(np.asarray(default, dtype=float)).ravel()
            if default_values.size != self._dimension:
                raise self.DimensionError(
                    f"Custom parameter '{self._name}' declares dimension "
                    f"{self._dimension} but its default value has "
                    f"{default_values.size} element(s)."
                )
        self._default = default_values

        self._partial = partial
        self._dynamics = dynamics
        self._partial_dynamics_by_param = partial_dynamics_by_param
        self._partial_dynamics_by_position = partial_dynamics_by_position
        self._partial_dynamics_by_velocity = partial_dynamics_by_velocity

        if step is None:
            self._step = None
        else:
            step_values = np.atleast_1d(np.asarray(step, dtype=float)).ravel()
            if step_values.size == 1:
                step_values = np.full(self._dimension, float(step_values[0]))
            if step_values.size != self._dimension:
                raise self.DimensionError(
                    f"Custom parameter '{self._name}' declares dimension "
                    f"{self._dimension} but its finite-difference step has "
                    f"{step_values.size} element(s)."
                )
            self._step = step_values

        # Units of d(acceleration)/d(parameter), precomputed once: the STM block
        # is rebuilt on every RHS evaluation during propagation.
        self._acceleration_partial_units = (
            _U_KM_S2 if units is None else _U_KM_S2 / units
        )

        # Units of the parameter's own equation of motion and of its Jacobians.
        self._rate_units = sec**-1 if units is None else units / sec
        self._rate_partial_units = {
            "position": self._rate_units / km,
            "velocity": self._rate_units / (km / sec),
        }

        if register:
            StateDefinition.register_custom_parameter(
                self._name,
                dimension=self._dimension,
                physical_type=self._physical_type,
            )

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def name(self) -> str:
        """State-component name of this parameter."""
        return self._name

    @property
    def dimension(self) -> int:
        """Number of scalar elements of this parameter."""
        return self._dimension

    @property
    def units(self) -> Units | None:
        """Units of the parameter value, or ``None`` when dimensionless."""
        return self._units

    @property
    def default(self) -> np.ndarray:
        """Fallback value used when the parameter is absent from the state."""
        return self._default

    @property
    def is_dynamic(self) -> bool:
        """``True`` when the parameter carries its own equation of motion."""
        return self._dynamics is not None

    @property
    def dynamics(self):
        """User callable returning :math:`\\dot p`, or ``None`` when static."""
        return self._dynamics

    @property
    def partial(self):
        """User callable returning :math:`\\partial \\mathbf a / \\partial p`, or ``None``."""
        return self._partial

    @property
    def partial_dynamics_by_param(self):
        """User callable returning :math:`\\partial \\dot p / \\partial p`, or ``None``."""
        return self._partial_dynamics_by_param

    @property
    def partial_dynamics_by_position(self):
        """User callable returning :math:`\\partial \\dot p / \\partial \\mathbf r`, or ``None``."""
        return self._partial_dynamics_by_position

    @property
    def partial_dynamics_by_velocity(self):
        """User callable returning :math:`\\partial \\dot p / \\partial \\mathbf v`, or ``None``."""
        return self._partial_dynamics_by_velocity

    @property
    def acceleration_partial_units(self) -> Units:
        """Units of :math:`\\partial \\mathbf a / \\partial p`."""
        return self._acceleration_partial_units

    @property
    def rate_units(self) -> Units:
        """Units of :math:`\\dot p`."""
        return self._rate_units

    def rate_partial_units(self, with_respect_to: str) -> Units:
        """
        Units of a Jacobian of :math:`\\dot p` w.r.t. position or velocity.

        Parameters
        ----------
        with_respect_to : str
            ``"position"`` or ``"velocity"``.

        Returns
        -------
        Units
            Units of the requested Jacobian block.
        """
        return self._rate_partial_units[with_respect_to]

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    def finite_difference_step(self, values: np.ndarray) -> np.ndarray:
        """
        Return the per-element finite-difference step for *values*.

        Parameters
        ----------
        values : numpy.ndarray
            Current parameter value.

        Returns
        -------
        numpy.ndarray
            Absolute step size for each element. Uses the explicit ``step``
            given at construction, otherwise a relative step of
            ``1e-6 * max(|p|, 1)``.
        """
        if self._step is not None:
            return self._step

        return 1e-6 * np.maximum(np.abs(values), 1.0)

    def resolve(self, current_state: dict, spice_id) -> np.ndarray:
        """
        Read this parameter's current value out of the propagated state.

        Parameters
        ----------
        current_state : dict
            State mapping ``(name, spice_id) -> values`` handed to the force
            models on every right-hand-side evaluation.
        spice_id : int or None
            SPICE ID of the body owning the parameter.

        Returns
        -------
        numpy.ndarray
            The estimated value when present in the state, otherwise
            :attr:`default`.
        """
        value = current_state.get((self._name, spice_id))
        if value is None:
            return self._default

        values = getattr(value, "values", value)

        return np.atleast_1d(np.asarray(values, dtype=float)).ravel()

    def __repr__(self) -> str:
        kind = "dynamic" if self.is_dynamic else "static"

        return (
            f"CustomParameter(name={self._name!r}, dimension={self._dimension}, "
            f"{kind})"
        )

    # endregion Methods #
    # -------------------#
