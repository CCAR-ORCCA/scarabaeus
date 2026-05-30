# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC

class DynamicModel:
    """ Abstract base class for all force models in the dynamic simulation.

    Every concrete force model inherits from this class and implements
    :meth:`compute_acceleration`, which returns the non-gravitational or
    gravitational acceleration contribution for a given spacecraft state and
    epoch.  The equation of motion integrated by the propagator is:

    .. math::

        \\ddot{\\mathbf{r}} =
            \\sum_i \\mathbf{a}_i(\\mathbf{r},\\,\\dot{\\mathbf{r}},\\,\\mathbf{p},\\,t)

    where each :math:`\\mathbf{a}_i` is the acceleration contribution from
    force model :math:`i`, :math:`\\mathbf{p}` is the vector of solve-for
    or consider parameters, and :math:`t` is the current epoch.

    Sub-classes also expose ``_compute_partial_by_*`` methods that return
    :math:`\\partial \\mathbf{a}_i / \\partial \\mathbf{p}`, the Jacobian of
    the acceleration with respect to each solve-for parameter.  These partials
    drive the variational equations

    .. math::

        \\dot{\\mathbf{S}} = F\\,\\mathbf{S} +
            \\frac{\\partial \\mathbf{f}}{\\partial \\mathbf{p}}

    whose solution :math:`\\mathbf{S}` maps parameter uncertainty into state
    uncertainty and is required by the orbit-determination filter.

    Parameters
    ----------
    None

    Notes
    -----
    Concrete sub-classes must implement :meth:`compute_acceleration` and
    optionally :meth:`_compute_partial_by_position` and the relevant
    ``_compute_partial_by_*`` methods for any solve-for parameters they
    support.

    See Also
    --------
    scarabaeus.CannonballSRP : Solar radiation pressure (cannonball model).
    scarabaeus.FiniteBurn : Continuous finite-duration thrust model.
    scarabaeus.ImpulsiveBurn : Instantaneous delta-v model.
    scarabaeus.nPlateSRP : Multi-plate solar radiation pressure model.
    scarabaeus.PointMassGravity : Single-body Newtonian gravity.
    scarabaeus.SphericalHarmonicsGravity : Gravity field expansion model.
    scarabaeus.ThreeBodyGravity : Third-body perturbation model.
    """
    #--------------------#
    # region Constructor #
    #--------------------#
    def __init__(self):
        pass