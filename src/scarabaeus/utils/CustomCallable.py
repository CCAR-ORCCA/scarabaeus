# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import inspect

import numpy as np


class CustomCallable:
    """ A user-supplied function bound to a fixed keyword vocabulary.

    Both :class:`~scarabaeus.CustomDynamicModel` and
    :class:`~scarabaeus.CustomMeasurement` let a script hand Scarabaeus a plain
    Python function. To keep those functions readable, they are invoked with
    keyword arguments and declare only the context entries they actually need.
    This class performs the binding once, at construction, so a misspelt
    argument is reported immediately rather than on the first evaluation deep
    inside an integrator or a partials loop.

    It also carries the central finite-difference helpers used to fill in any
    Jacobian the user did not supply analytically, so the two model families
    cannot drift apart in how they differentiate.

    Parameters
    ----------
    function : callable
        User-supplied function.
    vocabulary : tuple of str
        Context names the consumer is able to provide.
    role : str
        Human-readable description of what the function is for, used in error
        messages (e.g. ``"acceleration"``).
    owner : str
        Name of the model that owns the function, used in error messages.

    Raises
    ------
    CustomCallable.SignatureError
        If *function* is not callable, uses ``*args``, or declares an argument
        outside *vocabulary*.

    See Also
    --------
    scarabaeus.CustomDynamicModel : User-defined force model.
    scarabaeus.CustomMeasurement : User-defined measurement model.
    """

    class SignatureError(ValueError):
        """Raised when a user callable declares an argument that is not supplied."""

    # -------------#
    # Constructor  #
    # -------------#
    def __init__(self, function, vocabulary, role: str, owner: str):
        if not callable(function):
            raise self.SignatureError(
                f"The {role} of '{owner}' must be callable, got "
                f"{type(function).__name__}."
            )

        self._function = function
        self._role = role
        self._owner = owner

        signature = inspect.signature(function)
        names = []
        takes_all = False

        for parameter in signature.parameters.values():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                takes_all = True
                continue
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                raise self.SignatureError(
                    f"The {role} of '{owner}' declares '*{parameter.name}'. "
                    "Custom callables are invoked with keyword arguments only; "
                    "declare the context names you need. Available: "
                    f"{', '.join(vocabulary)}."
                )
            if parameter.name not in vocabulary:
                raise self.SignatureError(
                    f"The {role} of '{owner}' declares unknown argument "
                    f"'{parameter.name}'. Available: {', '.join(vocabulary)}."
                )
            names.append(parameter.name)

        self._names = tuple(vocabulary) if takes_all else tuple(names)

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def names(self) -> tuple:
        """Context names this callable receives."""
        return self._names

    @property
    def function(self):
        """The wrapped user function."""
        return self._function

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    def __call__(self, context: dict):
        """
        Invoke the user function with the subset of *context* it declared.

        Parameters
        ----------
        context : dict
            Full evaluation context.

        Returns
        -------
        object
            Whatever the user function returns.
        """
        return self._function(**{key: context[key] for key in self._names})

    def reads(self, name: str) -> bool:
        """
        Report whether this callable can see *name*.

        A callable that takes the whole state (or ``**kwargs``) could read the
        quantity from there, so it counts. Used to decide, without evaluating
        anything, whether a Jacobian block is structurally present.

        Parameters
        ----------
        name : str
            Context name, e.g. ``"velocity"``.

        Returns
        -------
        bool
            ``True`` when the callable declares *name* or a container holding it.
        """
        return name in self._names or "state" in self._names

    @staticmethod
    def default_step(values: np.ndarray, relative: float) -> np.ndarray:
        """
        Return a per-component central-difference step.

        Parameters
        ----------
        values : numpy.ndarray
            Point the derivative is taken at.
        relative : float
            Relative step size.

        Returns
        -------
        numpy.ndarray
            ``relative * max(|values|, 1)``, elementwise.
        """
        return relative * np.maximum(np.abs(values), 1.0)

    @staticmethod
    def finite_difference(evaluate, values: np.ndarray, steps, n_rows: int):
        """
        Central-difference Jacobian of *evaluate* with respect to *values*.

        Parameters
        ----------
        evaluate : callable
            Takes a perturbed copy of *values* and returns an ``(n_rows,)`` array.
        values : numpy.ndarray
            Point the Jacobian is evaluated at, shape ``(n_cols,)``.
        steps : numpy.ndarray
            Per-component absolute step sizes, shape ``(n_cols,)``.
        n_rows : int
            Number of rows of the resulting Jacobian.

        Returns
        -------
        numpy.ndarray
            Jacobian of shape ``(n_rows, n_cols)``.
        """
        jacobian = np.zeros((n_rows, values.size))

        for index in range(values.size):
            step = steps[index]

            perturbed = values.copy()
            perturbed[index] = values[index] + step
            plus = evaluate(perturbed)

            perturbed = values.copy()
            perturbed[index] = values[index] - step
            minus = evaluate(perturbed)

            jacobian[:, index] = (plus - minus) / (2.0 * step)

        return jacobian

    def __repr__(self) -> str:
        return (
            f"CustomCallable(owner={self._owner!r}, role={self._role!r}, "
            f"names={list(self._names)})"
        )

    # endregion Methods #
    # -------------------#
