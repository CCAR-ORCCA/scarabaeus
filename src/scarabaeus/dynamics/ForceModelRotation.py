# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    ForceModel,
    ArrayWUnits,
    CelestialBody,
)
import numpy as np


class ForceModelRotation(ForceModel):
    """Rotational-dynamics force model — not yet implemented.

    This class is a placeholder for the rotational (attitude) channel of
    Scarabaeus. **Instantiating it always raises** :exc:`NotImplementedError`.

    Planned sub-models include (non-exhaustive):

    * Gravity-gradient torque
    * Solar-radiation-pressure torque
    * Thruster torque (from finite-burn offset)
    * Attitude-state kinematics (quaternion / Euler-angle ODEs)

    Raises
    ------
    NotImplementedError
        Always — rotational dynamics are not yet implemented or verified.

    See Also
    --------
    scarabaeus.ForceModel : Abstract parent class defining the shared interface.
    scarabaeus.ForceModelTranslation : Translational dynamics aggregator.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "ForceModelRotation is not yet implemented or verified. "
            "Rotational dynamics (gravity-gradient torque, SRP torque, "
            "attitude kinematics, …) are planned but not available yet."
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_force_models(self) -> dict:
        """
        Instantiate rotational sub-models — not yet implemented.

        Intended to mirror :meth:`~scarabaeus.ForceModelTranslation._initialize_force_models`
        for the rotational channel, returning a dict of active torque sub-models
        (gravity-gradient, SRP torque, magnetic torque, …) keyed by name.

        Raises
        ------
        NotImplementedError
            Always — rotational sub-models are not yet implemented.
        """
        raise NotImplementedError(
            "ForceModelRotation._initialize_force_models is not yet implemented."
        )

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def _compute_total_angular_acceleration(
        self,
        attitude: ArrayWUnits,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Compute the total angular acceleration from all active torque models.

        The rotational analog of
        :meth:`~scarabaeus.ForceModelTranslation._compute_total_acceleration`.
        Where the translational channel sums forces and divides by mass
        (:math:`\\mathbf{a} = \\mathbf{F}/m`), this method will sum all
        external torques and apply Euler's rotational equation:

        .. math::

            \\boldsymbol{\\alpha}
            = \\mathbf{I}^{-1}
              \\bigl(\\boldsymbol{\\tau} - \\boldsymbol{\\omega}
              \\times \\mathbf{I}\\,\\boldsymbol{\\omega}\\bigr)

        where :math:`\\mathbf{I}` is the inertia tensor,
        :math:`\\boldsymbol{\\tau}` the total torque (N·m), and
        :math:`\\boldsymbol{\\omega}` the angular velocity (rad/s).

        Parameters
        ----------
        attitude : ArrayWUnits
            Current attitude state (quaternion or Euler angles). Reserved for
            future use.
        current_state : dict
            Current state parameters. Reserved for future use.
        epoch : float
            Current epoch in TDB seconds. Reserved for future use.
        body : CelestialBody
            Target body. Reserved for future use.

        Returns
        -------
        np.ndarray
            3-element angular acceleration vector (rad/s²).

        Raises
        ------
        NotImplementedError
            Always — rotational dynamics are not yet implemented.
        """
        raise NotImplementedError(
            "ForceModelRotation._compute_total_angular_acceleration is not yet implemented."
        )

    # ------------------------------------------------------------------
    # Partials (upcoming)
    # ------------------------------------------------------------------

    def _collect_partials(
        self,
        name: str,
        position: ArrayWUnits,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ):
        """
        Partial-derivative lookup — not yet implemented.

        Intended to mirror
        :meth:`~scarabaeus.ForceModelTranslation._collect_partials`
        for the rotational channel, returning Jacobians of angular acceleration
        with respect to attitude, angular velocity, and other estimated
        parameters.

        Parameters
        ----------
        name : str
            Partial-derivative key (e.g. ``"alpha@-10_by_omega@-10"``).
        position : ArrayWUnits
            Spacecraft position in km. Reserved for future use.
        current_state : dict
            Current state parameters. Reserved for future use.
        epoch : float
            Current epoch in TDB seconds. Reserved for future use.
        body : CelestialBody
            Target body. Reserved for future use.

        Raises
        ------
        NotImplementedError
            Always — rotational partial derivatives are not yet implemented.
        """
        raise NotImplementedError(
            f"ForceModelRotation._collect_partials ('{name}') is not yet implemented."
        )

    def _collect_parameter_dynamics(
        self,
        name: str,
        position: ArrayWUnits,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ):
        """
        Parameter ODE lookup — not yet implemented.

        Intended to mirror
        :meth:`~scarabaeus.ForceModelTranslation._collect_parameter_dynamics`
        for the rotational channel, returning the governing ODE for attitude-
        related estimated parameters (e.g. inertia tensor components, attitude
        biases).

        Parameters
        ----------
        name : str
            Parameter name whose ODE is requested.
        position : ArrayWUnits
            Spacecraft position in km. Reserved for future use.
        current_state : dict
            Current state parameters. Reserved for future use.
        epoch : float
            Current epoch in TDB seconds. Reserved for future use.
        body : CelestialBody
            Target body. Reserved for future use.

        Raises
        ------
        NotImplementedError
            Always — rotational parameter dynamics are not yet implemented.
        """
        raise NotImplementedError(
            f"ForceModelRotation._collect_parameter_dynamics ('{name}') is not yet implemented."
        )
