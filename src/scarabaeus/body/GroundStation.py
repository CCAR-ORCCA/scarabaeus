# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Body, Spacecraft, EpochArray, SpiceManager
import scarabaeus.utils.NumpyWrapper as np


# class definition
class GroundStation(Body):
    """Represents a ground station for spacecraft communication modeling.

    A ground station is a :class:`~scarabaeus.Body` with no mass or SRP
    properties.  Its primary role is to supply the observer position for
    radiometric observables (range, range-rate, Doppler).

    Parameters
    ----------
    name : str
        The name of the ground station.

    spice_id : int or float, optional
        The SPICE ID of the ground station, defaults to ``None``.

    See Also
    --------
    scarabaeus.CelestialBody : Planetary body on whose surface a GroundStation is placed.
    """

    # region Constructor
    def __init__(self, name: str, spice_id: int | str = None):
        # Parent class initializations
        super().__init__(name, spice_id)

    # region Methods
    def station_visibility(
        self,
        target: Spacecraft,
        epoch: EpochArray,
        elevation_mask: float = 10.0,
        return_angles: bool = False,
        frame: str = "J2000",
        origin: str = "EARTH",
    ):
        """
        Determines if a spacecraft is visible to the ground station given an elevation
        mask for a single epoch or a sequence of epochs.

        Positions are retrieved via SPICE using the SPICE names/IDs of this
        ground station and the target spacecraft, then the elevation (and
        optionally azimuth) are computed in the requested reference frame.

        The station elevation angle :math:`\\varepsilon` above the local horizon
        is computed from

        .. math::

            \\varepsilon = 90^\\circ - \\theta_{nadir},
            \\qquad
            \\cos\\theta_{nadir}
            = \\frac{(\\mathbf{r}_{sc} - \\mathbf{r}_{gs})\\cdot\\mathbf{r}_{gs}}
                {\\|\\mathbf{r}_{sc} - \\mathbf{r}_{gs}\\|\\,\\|\\mathbf{r}_{gs}\\|}

        The station is visible when :math:`\\varepsilon \\geq \\varepsilon_{mask}`
        (default 10°).

        Parameters
        ----------
        target : Spacecraft
            The target spacecraft. Its SPICE ID is used to query its state
            via SPICE.

        epoch : EpochArray
            The epoch or sequence of epochs at which to evaluate visibility.

        elevation_mask : float, optional
            Minimum elevation angle in degrees above the horizon for the
            spacecraft to be considered visible.  Defaults to ``10.0``.

        return_angles : bool, optional
            If ``True``, also returns the elevation and azimuth angles.
            Defaults to ``False``.

        frame : str, optional
            Reference frame for position queries.  Defaults to ``'J2000'``.

        origin : str, optional
            Observer body for SPICE state queries.  Defaults to ``'SUN'``.

        Returns
        -------
        visible : bool or np.ndarray of bool
            ``True`` if the spacecraft is above the elevation mask as seen from
            this ground station.  Returns a scalar for a single epoch or an
            array for multiple epochs.

        elevation : float or np.ndarray of float
            Elevation angle(s) in degrees above the horizon.  Only returned if
            ``return_angles = True``.

        azimuth : float or np.ndarray of float
            Azimuth angle(s) in degrees measured clockwise from North.  Only
            returned if ``return_angles = True``.
        """
        import spiceypy as spice

        epoch_vals = np.atleast_1d(epoch.times.values)
        scalar_input = epoch_vals.size == 1
        et_arr = epoch_vals.astype(float)

        # Vectorised SPICE queries — one list-comprehension per body avoids
        # Python-loop overhead from SpiceManager wrappers.
        obs_pos = np.array(
            [spice.spkezr(self._name, t, frame, "None", origin)[0][:3] for t in et_arr]
        )
        tgt_pos = np.array(
            [
                spice.spkezr(str(target.spice_id), t, frame, "None", origin)[0][:3]
                for t in et_arr
            ]
        )

        delta = tgt_pos - obs_pos  # (N, 3)
        obs_norm = np.linalg.norm(obs_pos, axis=1)  # (N,)
        delta_norm = np.linalg.norm(delta, axis=1)  # (N,)

        cos_theta = np.sum(delta * obs_pos, axis=1) / (obs_norm * delta_norm)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        elevation = np.degrees(np.pi / 2.0 - np.arccos(cos_theta))  # (N,)

        visible_out = elevation >= elevation_mask  # bool array (N,)

        if scalar_input:
            visible_out = bool(visible_out[0])

        if not return_angles:
            return visible_out

        # Azimuth (only when requested)
        obs_unit = obs_pos / obs_norm[:, None]  # (N, 3)
        delta_unit = delta / delta_norm[:, None]  # (N, 3)
        delta_horiz = (
            delta_unit - np.sum(delta_unit * obs_unit, axis=1)[:, None] * obs_unit
        )

        z_axis = np.array([0.0, 0.0, 1.0])
        north = z_axis - np.sum(z_axis * obs_unit, axis=1)[:, None] * obs_unit
        north_norm = np.linalg.norm(north, axis=1)
        degenerate = north_norm < 1e-10
        north[~degenerate] /= north_norm[~degenerate, None]
        north[degenerate] = np.array([1.0, 0.0, 0.0])

        east = np.cross(obs_unit, north)  # (N, 3)
        azimuth = (
            np.degrees(
                np.arctan2(
                    np.sum(delta_horiz * east, axis=1),
                    np.sum(delta_horiz * north, axis=1),
                )
            )
            % 360.0
        )

        if scalar_input:
            return bool(visible_out), float(elevation[0]), float(azimuth[0])
        return visible_out, elevation, azimuth

    def elevation(
        self,
        target: Spacecraft,
        epoch: EpochArray,
        frame: str = "J2000",
        origin: str = "EARTH",
    ) -> float | np.ndarray:
        """
        Computes the elevation angle of a spacecraft as seen from the ground station.

        Parameters
        ----------
        target : Spacecraft
            The target spacecraft object.  Its SPICE ID is used to query its state
            via SPICE.

        epoch : EpochArray
            The epoch or sequence of epochs at which to evaluate elevation.

        frame : str, optional
            Reference frame for position queries.  Defaults to ``'J2000'``.

        origin : str, optional
            Observer body for SPICE state queries.  Defaults to ``'SUN'``.

        Returns
        -------
        elevation : float or numpy.ndarray
            Elevation angle(s) in degrees above the horizon.  Returns a scalar
            for a single epoch or an array for multiple epochs.
        """
        # condition inputs
        epoch_vals = np.atleast_1d(epoch.times.values)
        scalar_input = epoch_vals.size == 1

        ## compute elevation for each epoch
        elevation_list = []
        for t in epoch_vals:
            # get states of observer and target
            observer_state = SpiceManager.get_state(
                trgt_bdy=self._name,
                epoch_time=t,
                reference_frame=frame,
                obsvr_bdy=origin,
            )
            target_state = SpiceManager.get_state(
                trgt_bdy=str(target.spice_id),
                epoch_time=t,
                reference_frame=frame,
                obsvr_bdy=origin,
            )

            observer_pos = np.asarray(observer_state.values[0:3])
            target_pos = np.asarray(target_state.values[0:3])

            # compute elevation angle at this epoch
            delta = target_pos - observer_pos

            cos_theta = np.dot(delta, observer_pos) / (
                np.linalg.norm(observer_pos) * np.linalg.norm(delta)
            )
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)

            elevation_list.append(np.degrees(np.pi / 2.0 - theta))

        # return epoch or epochs
        return elevation_list[0] if scalar_input else np.array(elevation_list)
