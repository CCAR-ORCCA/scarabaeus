# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""
    Scarabaeus Rust
    ---------------
    Rust implementation of computationally expensive tasks within Scarabaeus.
"""
from typing import Callable
import numpy as np

class IAS15:
    """ Implicit integrator with Adaptive timeStepping, 15th order (IAS15).

        This is a Rust implementation of the "IAS15" algorithm [[1]_] that interfaces 
        with Python.

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
            Allowable error to consider the predictor-corrector converged. Default value is ``1e-10``.
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
    """
    def __init__(cls, dt: float, min_dt: float, 
                 dynamics: Callable[[float, list[float]], list[float]]) -> None: 
        ...
    
    # attribute types
    SAFETY_FACTOR: float
    EPSILON      : float
    
    def integrate(self, interval: np.ndarray | tuple[float, float], 
                  y0: list) -> tuple[list[float], list[float]]:
        """ Perform integration with the propagator.

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
        """ Add an event to search for during integration.

            Parameters
            ----------
            event : IntegrationEvent
                Event to search for during integration.
        """
        ...

    def set_event_tol(self, tolerance: float) -> None:
        """ Set the event detection tolerance.

            Parameters
            ----------
            tolerance : float
                The new event tolerance.
        """
        ...

class IntegrationEvent:
    """ Defines an event based off time, state, or both.

        Parameters
        ----------
        event_finder : Callable[[float, list], bool]
            Function providing logic to determine if an event has occurred.
        find_increasing : bool
            Flag to detect the event when it crosses from negative to positive.
        find_decreasing : bool
            Flag to detect the even when it crosses from positive to negative.
        is_terminal : bool, optional
            Flag to decide if integration should stop when the event is 
            detected. Defaults to ``False``.
    """
    def __init__(self, event_finder: Callable[[float, list], bool], 
                 find_increasing: bool, find_decreasing: bool, 
                 is_terminal: bool) -> None:
        ...