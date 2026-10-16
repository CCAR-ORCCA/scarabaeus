# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    Measurement,
    MissionSequence,
    Units,
    Frame,
    ArrayWUnits,
    ArrayWFrame,
    EpochArray,
    Body,
    Spacecraft,
    Camera,
    Noise,
    SpiceManager,
    StateDefinition,
)
from scarabaeus.measurements.OpNavManager import OpNavManager

import scarabaeus.utils.NumpyWrapper as np

# Units & frames
km, sec = Units.get_units(["km", "sec"])
J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


# --------------------#
#  Class Definition  #
# --------------------#
class CentroidingIdeal(Measurement):
    """ Models the optical ideal centroiding measurement model.

    Simulates centroiding measurements via a line-of-sight unit vector
    towards the target using a pinhole camera model.

    The image-plane coordinates :math:`(u, v)` of a target at inertial
    position :math:`\\mathbf{r}` are

    .. math::

        \\begin{pmatrix}u\\\\v\\end{pmatrix}
        = \\frac{f}{(\\mathbf{R}\\,\\hat{\\boldsymbol{\\ell}})_z}
          \\begin{pmatrix}
            (\\mathbf{R}\\,\\hat{\\boldsymbol{\\ell}})_x \\\\
            (\\mathbf{R}\\,\\hat{\\boldsymbol{\\ell}})_y
          \\end{pmatrix}

    where :math:`f` is the focal length, :math:`\\mathbf{R}` is the camera
    rotation matrix, and :math:`\\hat{\\boldsymbol{\\ell}}` is the unit
    line-of-sight vector.

    Parameters
    ----------
    name : str
        Name of the measurement model.
    camera : Camera
        Camera object from scarabaeus.
    sigma : ArrayWUnits, optional
        Measurement standard deviation for both xy directions in image plane.
        Defaults to ``None``.
    state_definition : list, optional
        StateVector definition list. Defaults to ``None``.
    sequence_definition : MissionSequence, optional
        Mission sequence whose per-leg state definitions replace
        *state_definition*. Defaults to ``None``.

    Raises
    ------
    RuntimeError
        If the units extracted from the measurements are not consistent.
    ValueError
        If neither an EpochArray nor start/end Epochs are provided.

    See Also
    --------
    scarabaeus.Measurement : Abstract base class for all measurement models.

    References
    ----------
    .. [1] "A Tutorial on Horizon-Based Optical Navigation and Attitude Determination with Space Imaging Systems", John A. Christian, IEEE-Access, Vol. 9, 19819-19853, 2021.

    """

    # ------------------#
    # region   Errors  #
    # ------------------#
    # Custom exceptions: named on the model and raised on the
    # line that detects the failure, so tracebacks point at the real cause
    # rather than a generic Exception/RuntimeError.
    class MissingImageIndexError(Exception):
        """A stacked per-image centroid bias has no epoch-to-image mapping.

        Raised by the forward model and the partials when the state carries
        a ``centroid_bias`` of dimension ``2 * n_images`` but
        :meth:`set_image_epochs` was never called, so there is no way to know
        which image slot an epoch belongs to.
        """

    # endregion Errors #
    # ------------------#

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        name: str,
        camera: Camera,
        sigma: ArrayWUnits = None,
        state_definition: list = None,
        sequence_definition: MissionSequence = None,
    ):
        super().__init__(name, camera)

        # Initialize attributes
        self._name = name
        self._camera = camera
        self._sigma = sigma
        self._state_definition = state_definition
        self._sequence_definition = sequence_definition
        self._measurement_type = "CentroidingIdeal"
        self._component_labels = ("sample", "line")
        self._epoch_to_image_index: dict | None = None

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def name(self) -> str:
        """
        The name of the Centroiding model.
        """
        return self._name

    @property
    def camera(self) -> Camera:
        """
        The camera object taking measurements.
        """
        return self._camera

    @property
    def state_definition(self) -> list:
        """
        The state vector definition.
        """
        return self._state_definition

    @property
    def sequence_definition(self) -> MissionSequence:
        """
        The sequence definition
        """
        return self._sequence_definition

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#

    # image-index helpers ---

    def set_image_epochs(self, image_epochs: list | None):
        """
        Enable per-image centroid biases.

        Parameters
        ----------
        image_epochs : list[float] | None
            Ordered list of observation epoch values, one per image.
            Image index equals the position in the list.
            Pass ``None`` to revert to global-bias mode.
        """
        if image_epochs is None:
            self._epoch_to_image_index = None
        else:
            self._epoch_to_image_index = {
                float(t): i for i, t in enumerate(image_epochs)
            }

    def set_image_index(self, mapping: dict | None):
        """Set the epoch-to-image mapping directly.

        Parameters
        ----------
        mapping : dict or None
            ``{epoch_value: image_index}`` mapping used to pick the per-image slot
            of a stacked ``centroid_bias``, or ``None`` to clear it. Use
            :meth:`set_image_epochs` to build it from an ordered list of epochs.
        """
        self._epoch_to_image_index = mapping

    def _get_rel_cam_pos(self, target: Body, epoch: EpochArray) -> ArrayWUnits:
        """
        Return the target position relative to the camera, in the camera frame.

        Parameters
        ----------
        target : Body
            Body being imaged.
        epoch : EpochArray
            Single observation epoch.

        Returns
        -------
        ArrayWUnits
            Position of the target relative to the camera, shape (3,).

        Notes
        -----
        The SPICE camera frame is tried first. When SPICE cannot resolve it, the
        inertial position relative to the camera's host body is rotated with
        :meth:`Camera.get_rotation_to_camera`.
        """
        try:
            return SpiceManager.get_state(
                trgt_bdy=target.name,
                epoch_time=epoch.times.values,
                reference_frame=self.camera.camera_frame,
                obsvr_bdy=self.camera.name,
            )[0:3]
        except Exception:
            rel_inertial = SpiceManager.get_state(
                trgt_bdy=target.name,
                epoch_time=epoch.times.values,
                reference_frame="J2000",
                # spkezr takes the body as a string; a bare int SPICE ID is
                # passed through as blank and resolves to nothing.
                obsvr_bdy=str(self.camera.associated_body_spice_id),
            )[0:3]
            R_ci = self.camera.get_rotation_to_camera(epoch.times.values)
            r_cam = R_ci @ rel_inertial.values
            return ArrayWUnits(r_cam, rel_inertial.units)

    def observed_measurements(
        self,
        file_name,
        meas_name: str = "meas_sim",
        units=None,
        frame: Frame = J2000,
        image_epochs: list | None = None,
    ):
        """
        Load centroiding measurements from a .json file.

        Parameters
        ----------
        file_name : str
            Name of the .json file holding the measurements.
        meas_name : str, optional
            Key of the measurement block to read. Defaults to ``'meas_sim'``.
        units : Units, optional
            Units to interpret the stored values in. Defaults to ``None``.
        frame : Frame, optional
            Reference frame the loaded measurements are expressed in. Defaults
            to ``J2000``.
        image_epochs : list of float, optional
            Ordered list of observation epoch values, one per image. Must be
            provided when the state vector contains a stacked
            ``centroid_bias_<suffix>`` entry (dimension > 2). Defaults to ``None``.

        Returns
        -------
        ObsData
            The observation tuple returned by
            :meth:`Measurement.observed_measurements`.
        """
        result = super().observed_measurements(file_name, meas_name, units, frame)
        if image_epochs is not None:
            self.set_image_epochs(image_epochs)
        return result

    def computed_measurements(
        self,
        target: Body,
        epoch_array: EpochArray = None,
        epoch_start: EpochArray = None,
        epoch_end: EpochArray = None,
        tstep: float = 1.0,
        frame: Frame = J2000,
        noisy: bool = False,
        image_epochs: list | None = None,
    ) -> ArrayWFrame:
        """
        Compute the pixel centroid of a target at each epoch.

        Parameters
        ----------
        target : Body
            The target body to image.
        epoch_array : EpochArray, optional
            Array of epochs at which to compute the measurement. Defaults to
            ``None``.
        epoch_start : EpochArray, optional
            Start epoch when building the array internally. Defaults to ``None``.
        epoch_end : EpochArray, optional
            End epoch when building the array internally. Defaults to ``None``.
        tstep : float, optional
            Time step [s] when building the array internally. Defaults to ``1``.
        frame : Frame, optional
            Reference frame for the computation. Defaults to ``J2000``.
        noisy : bool, optional
            When ``True``, adds Gaussian noise scaled by ``sigma``. Defaults to
            ``False``.
        image_epochs : list of float, optional
            Ordered list of observation epoch values, one per image, passed to
            :meth:`set_image_epochs` first. Needed when the state holds a stacked
            ``centroid_bias``. Defaults to ``None``.

        Returns
        -------
        t1, t2, t3, meas : tuple
            Four values, per the contract in
            :meth:`~scarabaeus.Measurement.computed_measurements`.

            This observable is **single-epoch** -- unlike a two-way radiometric
            measurement it is not spread across a light-time path with distinct
            transmit, reflect and receive times -- so ``t1``, ``t2`` and ``t3``
            are all the observation epoch. They are returned, rather than a bare
            measurement, because the estimator reads ``t1`` when detrending
            residuals and matches measurements on ``t2``; collapsing them here
            is a correct statement about this observable, not padding.

            ``meas`` is the computed observable as an ``ArrayWFrame``.

        Raises
        ------
        MissingImageIndexError
            If the state holds one ``(bu, bv)`` bias per image (more than
            two elements) and no epoch-to-image mapping is set.
        """
        if image_epochs is not None:
            self.set_image_epochs(image_epochs)

        ## Input check
        epoch_array = Measurement._resolve_epoch_array(
            epoch_array, epoch_start, epoch_end, tstep
        )

        # Extract full bias vector once (dim=2 global, dim=2*n_images stacked per-image).
        _bias_full = self._extract_bias("centroid_bias", None)
        if (
            _bias_full is not None
            and _bias_full.values.size > 2
            and self._epoch_to_image_index is None
        ):
            raise CentroidingIdeal.MissingImageIndexError(
                f"The centroid bias has {_bias_full.values.size} elements, so it "
                "holds one (bu, bv) bias per image, but no epoch-to-image "
                "mapping is set and the bias of each epoch is undefined. Call "
                "set_image_epochs() (or pass image_epochs=) first."
            )

        meas_list = []

        for epoch in epoch_array:
            # A single (bu, bv) pair is global: it applies to every image.
            if (
                self._epoch_to_image_index is not None
                and _bias_full is not None
                and _bias_full.values.size > 2
            ):
                _iidx = self._epoch_to_image_index.get(float(epoch.times.values))
                _bvals = _bias_full.values.ravel()
                if _iidx is not None and 2 * _iidx + 1 < len(_bvals):
                    bias_awu = ArrayWUnits(
                        _bvals[2 * _iidx : 2 * (_iidx + 1)], _bias_full.units
                    )
                else:
                    bias_awu = None
            else:
                bias_awu = _bias_full

            rel_cam_pos = self._get_rel_cam_pos(target, epoch)

            r_cam = rel_cam_pos.values  # (3,)
            if r_cam[2] == 0:
                u = 0.0
                v = 0.0
            else:
                u, v = OpNavManager.project(self.camera, r_cam)[0]

            # Apply pixel bias [bu, bv]
            if bias_awu is not None:
                b = bias_awu.values.ravel()
                u = u + b[0]
                v = v + b[1]

            u_meas = ArrayWUnits(u, None)
            v_meas = ArrayWUnits(v, None)

            # Noise (independent on u, v)
            if noisy and self._sigma is not None:
                if self._sigma.size == 2:
                    sig_u, sig_v = self._sigma[0], self._sigma[1]
                else:
                    sig_u = sig_v = self._sigma

                u_meas += Noise().generate_AWGN_with_units(
                    mu=0.0, sigma=sig_u.values, units=sig_u.units
                )
                v_meas += Noise().generate_AWGN_with_units(
                    mu=0.0, sigma=sig_v.values, units=sig_v.units
                )

            meas_list.append((u_meas, v_meas))

        # Pack into ArrayWFrame
        values_u = np.array([m[0].values for m in meas_list])
        values_v = np.array([m[1].values for m in meas_list])
        units_u = meas_list[0][0].units
        units_v = meas_list[0][1].units

        # Ensure consistent units (ideal model usually unitless / pixel)
        if units_u != units_v:
            raise RuntimeError("multiple units are extracted from the measurements.")

        meas_AWF = ArrayWFrame(
            np.column_stack((values_u, values_v)),
            units_u,
            self.camera.camera_frame_object,
        )

        # Single-epoch observable: there is no light-time path with distinct
        # transmit / reflect / receive times to distinguish, so all three
        # collapse to the observation epoch. Returned because the estimator
        # detrends on t1 and matches measurements on t2 -- see
        # Measurement.computed_measurements for the contract.
        t1 = epoch_array
        t2 = epoch_array
        t3 = epoch_array

        return t1, t2, t3, meas_AWF

    # -----------------------------#
    #   Analytic Partial Derivs    #
    # -----------------------------#
    def _compute_H_cam(self, rel_cam_pos: ArrayWUnits) -> np.ndarray:
        """
        Partials of the pixel coordinates with respect to the camera-frame
        position, :math:`\\partial (u, v) / \\partial \\mathbf{r}_{cam}`.

        A thin wrapper over :meth:`~scarabaeus.OpNavManager.project_jacobian`, so
        the projection and its derivative stay in step.

        Parameters
        ----------
        rel_cam_pos : ArrayWUnits
            Target position relative to the camera, in the camera frame, shape
            (3,).

        Returns
        -------
        np.ndarray
            Shape (2, 3).
        """
        return OpNavManager.project_jacobian(self.camera, rel_cam_pos.values)

    def _compute_H_pos_inertial(
        self,
        rel_cam_pos: ArrayWUnits,
        R_ci: np.ndarray,
    ) -> np.ndarray:
        """
        Partials of the pixel coordinates with respect to the inertial target
        position.

        By the chain rule,

        .. math::

            \\frac{\\partial (u, v)}{\\partial \\mathbf{r}_{inertial}} =
                \\frac{\\partial (u, v)}{\\partial \\mathbf{r}_{cam}}\\, R_{ci}

        where :math:`R_{ci}` rotates vectors from the inertial frame (J2000) to the
        camera frame.

        Parameters
        ----------
        rel_cam_pos : ArrayWUnits
            Target position relative to the camera, in the camera frame, shape
            (3,).
        R_ci : np.ndarray
            Rotation from the inertial frame to the camera frame, shape (3, 3).

        Returns
        -------
        np.ndarray
            Shape (2, 3).
        """
        return self._compute_H_cam(rel_cam_pos) @ R_ci

    def _compute_H_vel_target(self) -> np.ndarray:
        """
        Partials of the pixel coordinates with respect to the target velocity.

        Returns
        -------
        np.ndarray
            Zeros of shape (2, 3); geometric centroiding is instantaneous.
        """
        return np.zeros((2, 3), dtype=float)

    # -----------------------------#
    #   Public Partials Interface  #
    # -----------------------------#
    def _partials(
        self,
        target: Spacecraft,
        epoch: EpochArray,
        frame: Frame = J2000,
    ) -> list:
        """
        Assemble the ``u`` and ``v`` rows of the global H-tilde.

        Parameters
        ----------
        target : Spacecraft
            The body being imaged.
        epoch : EpochArray
            Single epoch at which the partials are evaluated.
        frame : Frame, optional
            Reference frame of the state. Defaults to ``J2000``.

        Returns
        -------
        h_tilde : list
            ``[row_u, row_v]``, each a list of floats as long as the state.
            Without a ``state_definition`` each row holds the target position and
            velocity partials only.

        Raises
        ------
        MissingImageIndexError
            If the state holds a stacked per-image ``centroid_bias`` (dimension
            > 2) and no epoch-to-image mapping is set.

        Notes
        -----
        The imaged body's position enters with :math:`+H_{pos}` and the camera's
        host body with :math:`-H_{pos}`, since the camera-frame position is
        :math:`R_{ci} (\\mathbf{r}_{target} - \\mathbf{r}_{camera})`. Velocity
        columns are zero. Components of the camera body get the identity for a
        global ``centroid_bias`` (or ones in the current image's two columns when
        it is stacked), :math:`x_c / z_c` and :math:`y_c / z_c` for the focal
        lengths :math:`f_x, f_y`, and ones for the principal-point offsets
        :math:`c_x, c_y`. Every other component gets zeros.
        """
        # Target position in camera frame (needed for Jacobian)
        rel_cam_pos = self._get_rel_cam_pos(target, epoch)

        # Rotation from inertial (frame) to camera frame — needed for H_pos_inertial
        R_ci = self.camera.get_rotation_to_camera(epoch.times.values)

        # 2×3 Jacobians in inertial frame
        H_pos_inertial = self._compute_H_pos_inertial(rel_cam_pos, R_ci)  # 2×3
        H_vel = self._compute_H_vel_target()  # 2×3

        # Camera-frame coordinates for intrinsic _partials
        x_c, y_c, z_c = rel_cam_pos.values

        # Allowed state patterns
        state_allowed = StateDefinition._generate_allowed_state_dictionary()
        target_sid = getattr(target, "spice_id", None)
        cam_sid = getattr(self._camera, "spice_id", None)

        # If no state_definition: assume only target pos+vel (legacy behavior)
        if self._state_definition is None:
            row_u = list(H_pos_inertial[0]) + list(H_vel[0])
            row_v = list(H_pos_inertial[1]) + list(H_vel[1])
            return [row_u, row_v]

        # Column layout of the estimated state, so each component writes into
        # its own slice instead of the row being assembled by appending.
        n_state, components = self._state_layout()
        H = np.zeros((2, n_state))

        for comp_name, comp_dim, comp_body, start in components:
            comp_body_sid = getattr(comp_body, "spice_id", None)

            # Position (inertial frame) ---
            # r_cam = R_ci @ (r_target − r_camera_body), and H_pos_inertial is
            # d(u, v)/d(r_target − r_camera_body), so
            # imaged body (target):        dmeas/dr_target = +H
            # spacecraft  (camera body):   dmeas/dr_sc     = −H
            # Both may be estimated at once, so every position block is visited
            # rather than resolving a single one.
            if state_allowed["position"][2].match(comp_name):
                if comp_body_sid == target_sid:
                    H[:, start : start + 3] = H_pos_inertial
                elif comp_body_sid == cam_sid:
                    H[:, start : start + 3] = -H_pos_inertial

            # Velocity is left at zero: the instantaneous model has no velocity
            # dependence for either body.
            elif state_allowed["velocity"][2].match(comp_name):
                continue

            elif comp_body_sid != cam_sid:
                # The remaining components are all camera intrinsics/biases.
                continue

            # Centroid pixel bias ---
            # Global (dim=2): du/d[bu,bv] = [1,0], dv/d[bu,bv] = [0,1].
            # Stacked per-image (dim=2*N): indicator at slot [2*img, 2*img+1].
            if state_allowed["centroid_bias"][2].match(comp_name):
                if self._epoch_to_image_index is not None and comp_dim > 2:
                    _cur = self._epoch_to_image_index.get(float(epoch.times.values))
                    if _cur is not None and 2 * _cur + 1 < comp_dim:
                        H[0, start + 2 * _cur] = 1.0
                        H[1, start + 2 * _cur + 1] = 1.0
                elif comp_dim > 2:
                    raise CentroidingIdeal.MissingImageIndexError(
                        f"State component '{comp_name}' has dimension {comp_dim}, so it "
                        "holds one (bu, bv) bias per image, but no epoch-to-image "
                        "mapping is set and the partial for this epoch is undefined. "
                        "Call set_image_epochs() (or pass image_epochs= to "
                        "observed_measurements/computed_measurements) before computing "
                        "partials."
                    )
                else:
                    H[0, start] = 1.0  # global mode (dim=2)
                    H[1, start + 1] = 1.0

            # Camera focal length [fx, fy] (K[0,0] and K[1,1])
            # u = fx*x_c/z_c + cx,  v = fy*y_c/z_c + cy
            elif state_allowed["cam_focal_length"][2].match(comp_name):
                if z_c != 0:
                    H[0, start] = x_c / z_c  # du/dfx
                    H[1, start + 1] = y_c / z_c  # dv/dfy

            # Camera principal point offset [cx, cy] (K[0,2] and K[1,2])
            # u = fx*x_c/z_c + cx → du/dcx = 1, du/dcy = 0
            elif state_allowed["cam_pp_offset"][2].match(comp_name):
                H[0, start] = 1.0
                H[1, start + 1] = 1.0

            # Any other component leaves its columns at zero.

        return [H[0].tolist(), H[1].tolist()]
