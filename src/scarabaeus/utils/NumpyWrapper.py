# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""NumPy / autograd backend switch for Scarabaeus.

Exports a unified NumPy-compatible namespace so the rest of the codebase can
write ``import scarabaeus.utils.NumpyWrapper as np`` and remain agnostic of
the active backend.

The active backend is controlled by the ``UseAutoGrad`` environment variable:

* ``UseAutoGrad=TRUE`` — imports ``autograd.numpy``, enabling automatic
  differentiation.  The module-level ``autograd_types`` tuple is set to
  ``(ArrayBox,)`` for runtime isinstance checks.
* Otherwise — imports standard ``numpy``.  ``autograd_types`` is an empty
  tuple.

Both branches expose the backend-safe :func:`set` helper for array assignment.

Notes
-----
Import this module **instead of** ``numpy`` or ``autograd.numpy`` directly.
Never import both in the same file; doing so breaks the backend abstraction.
"""
import os

# -----------------------------------------------------------------------------
# Backend switch:
#   - Else select autograd if: "UseAutoGrad" in env AND env["UseAutoGrad"] == "TRUE"
#   - Else: NumPy
# -----------------------------------------------------------------------------

if "UseAutoGrad" in os.environ.keys() and os.environ["UseAutoGrad"] == "TRUE":
    # Maintain identical public API/usage: export numpy-like symbols via star import.
    from autograd.numpy import *  # noqa: F401,F403
    from autograd.numpy.numpy_boxes import ArrayBox

    # Preserve name and meaning exactly.
    autograd_types = (ArrayBox,)

    def set(array, idx, val):
        """Backend-safe item assignment for the autograd backend.

        When *val* is an :class:`~autograd.tracer.ArrayBox`, unwraps it to a
        raw NumPy value, copies *array*, and returns the modified copy —
        required because autograd traces are immutable.  Otherwise performs an
        in-place assignment and returns the same array.

        Parameters
        ----------
        array : numpy.ndarray
            Target array to modify.
        idx : int, slice, or tuple
            Index expression accepted by ``array[idx]``.
        val : float or numpy.ndarray or ArrayBox
            Value to assign at *idx*.

        Returns
        -------
        array : numpy.ndarray
            Modified array (a copy when *val* is an ``ArrayBox``, otherwise
            the same object).
        """
        if isinstance(val, ArrayBox):
            val = val._value  # unwrap to NumPy
            array = array.copy()
            array[idx] = val
        else:
            array[idx] = val
        return array

else:
    # Maintain identical public API/usage: export numpy-like symbols via star import.
    from numpy import *  # noqa: F401,F403

    # Preserve name and meaning exactly.
    autograd_types = ()

    def set(array, idx, val):
        """Backend-safe item assignment for the standard NumPy backend.

        Performs an in-place assignment ``array[idx] = val`` and returns the
        same array object.  Signature is identical to the autograd variant so
        call sites remain backend-agnostic.

        Parameters
        ----------
        array : numpy.ndarray
            Target array to modify.
        idx : int, slice, or tuple
            Index expression accepted by ``array[idx]``.
        val : float or numpy.ndarray
            Value to assign at *idx*.

        Returns
        -------
        array : numpy.ndarray
            The same *array* object after in-place modification.
        """
        array[idx] = val
        return array
