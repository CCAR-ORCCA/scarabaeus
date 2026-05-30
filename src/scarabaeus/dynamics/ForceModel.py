# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from abc import ABC, abstractmethod

from scarabaeus import (
    Units,
    EpochArray,
    StateArray,
    Frame,
)

import scarabaeus.utils.NumpyWrapper as np

kg, km, sec, N = Units.get_units(["kg", "km", "sec", "N"])
J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


class ForceModel(ABC):
    """ Abstract base class for force models.

    A ``ForceModel`` aggregates one or more :class:`~scarabaeus.DynamicModel` sub-models
    into a single configurable interface.  The total translational acceleration
    delivered to the integrator is the sum of all active sub-model contributions:

        .. math::

            \\mathbf{a}(t) =
                \\sum_{i} \\mathbf{a}_i(\\mathbf{r},\\,\\dot{\\mathbf{r}},\\,\\mathbf{p},\\,t)

    where each :math:`\\mathbf{a}_i` is computed by the corresponding
    :class:`~scarabaeus.DynamicModel` instance and :math:`\\mathbf{p}` is the vector of
    solve-for or consider parameters.

    In addition to the total acceleration, the ``ForceModel`` collects the
    partial derivatives :math:`\\partial \\mathbf{a} / \\partial \\mathbf{p}`
    from each active sub-model.  These partials populate the
    :math:`\\partial \\mathbf{f} / \\partial \\mathbf{p}` term required to
    integrate the variational equations

    .. math::

        \\dot{\\mathbf{S}} = F\\,\\mathbf{S} +
            \\frac{\\partial \\mathbf{f}}{\\partial \\mathbf{p}}

    whose solution :math:`\\mathbf{S}` is the sensitivity matrix used by the
    orbit-determination filter.

    Parameters
    ----------
    None

    Notes
    -----
    Concrete sub-classes must override :meth:`_initialize_force_models` and
    :meth:`_compute_total_acceleration`.  Call :meth:`initialize_force_models`
    before invoking any compute methods to bind the model to a concrete
    :class:`~scarabaeus.StateArray`, reference frame, and epoch.

    See Also
    --------
    scarabaeus.ForceModelTranslation : Translational force model aggregator.
    scarabaeus.DynamicModel : Base class for individual force model components.
    """

    def __init__(self) -> None:
        # Set by initialize_force_models
        self._state_vector: StateArray | None = None
        self._base_frame: Frame = J2000
        self._t0: EpochArray | None = None
        self._force_models: dict = {}

    # Init wiring
    def initialize_force_models(
        self,
        state_vector: StateArray,
        base_frame: Frame | None = None,
    ) -> None:
        """
        Bind this ForceModel to a concrete state and frame, then build internal models.

        Must be called before any compute methods are invoked.

        Parameters
        ----------
        state_vector : StateArray
            Initial state vector that internal force models will be constructed
            from.
        base_frame : Frame, optional
            Reference frame for all dynamics computations. Defaults to J2000
            if ``None``.
        """
        self._state_vector = state_vector
        self._base_frame = base_frame if base_frame is not None else J2000

        self._force_models = self._initialize_force_models()

    @abstractmethod
    def _initialize_force_models(self) -> dict:
        """
        Construct and return the internal force model dictionary.

        Implemented by subclasses; relies on `_state_vector`, `_base_frame`,
        and `_t0` being already set.
        """
        raise NotImplementedError

    # Guards & common props
    def _assert_initialized(self) -> None:
        """
        Raise RuntimeError if the model has not been initialized.

        Raises
        ------
        RuntimeError
            If ``initialize_force_models`` has not been called or returned an
            empty force model dictionary.
        """
        if self._state_vector is None or not self._force_models:
            raise RuntimeError(
                "ForceModel is not initialized. Call "
                "initialize_force_models(state_vector, base_frame, t0) first."
            )

    @property
    def state_vector(self) -> StateArray:
        """
        State vector bound to this force model.

        Raises
        ------
        RuntimeError
            If ``initialize_force_models`` has not been called.
        """
        if self._state_vector is None:
            raise RuntimeError("state_vector not set. Call initialize_force_models().")
        return self._state_vector

    @property
    def base_frame(self) -> Frame:
        """Reference frame used for all dynamics computations. Defaults to J2000."""
        return self._base_frame

    @property
    def t0(self) -> EpochArray:
        """
        Initial epoch for this force model.

        Raises
        ------
        RuntimeError
            If ``initialize_force_models`` has not been called.
        """
        if self._t0 is None:
            raise RuntimeError("t0 not set. Call initialize_force_models().")
        return self._t0

    @property
    def force_models(self) -> dict:
        """
        Dictionary of instantiated sub-model objects keyed by model name string.

        Raises
        ------
        RuntimeError
            If the model has not been initialized.
        """
        self._assert_initialized()
        return self._force_models
