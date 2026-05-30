# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    Measurement,
    Units,
    Frame,
    ArrayWUnits,
    ArrayWFrame,
    EpochArray,
    Body,
    Spacecraft,
    StateArray,
    Camera,
    Noise,
    SpiceManager,
    StateDefinition,
)

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
    sequence_definition : list, optional
        Sequence definition list. Defaults to ``None``.

    Raises
    ------
    RuntimeError
        If the units extracted from the measurements are not consistent.
    RuntimeError
        If neither an EpochArray nor start/end Epochs are provided.

    See Also
    --------
    scarabaeus.Measurement : Abstract base class for all measurement models.

    References
    ----------
    "A Tutorial on Horizon-Based Optical Navigation and Attitude Determination with Space Imaging Systems", John A. Christian, IEEE-Access, Vol. 9, 19819-19853, 2021.

    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        name: str,
        camera: Camera,
        sigma: ArrayWUnits = None,
        state_definition: list = None,
        sequence_definition: list = None,
    ):
        super().__init__(name, camera)

        # Initialize attributes
        self._name = name
        self._camera = camera
        self._sigma = sigma
        self._state_definition = state_definition
        self._sequence_definition = sequence_definition
        self._measurement_type = "CentroidingIdeal"
        self._reference_state_vector: StateArray | None = None
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
    def sigma(self) -> float:
        """
        The standard deviation of the measurements.
        """
        return self._sigma

    @property
    def state_definition(self) -> dict:
        """
        The state vector definition.
        """
        return self._state_definition

    @property
    def sequence_definition(self) -> dict:
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
        """Directly assign a pre-built ``{epoch_value: image_index}`` mapping (or None to disable)."""
        self._epoch_to_image_index = mapping

    def _extract_centroid_bias(self) -> ArrayWUnits | None:
        """
        Return centroid pixel bias [bu, bv] for this camera (global mode — first match).
        """
        if self._reference_state_vector is None:
            return None

        state_allowed = StateDefinition._generate_allowed_state_dictionary()
        if "centroid_bias" not in state_allowed:
            return None

        cam_sid = getattr(self._camera, "spice_id", None)
        for state_entry in self._reference_state_vector.state:
            component = state_entry[0]
            body = state_entry[4]
            value = state_entry[5]  # ArrayWFrame

            if state_allowed["centroid_bias"][2].match(component) is None:
                continue
            if getattr(body, "spice_id", None) == cam_sid:
                return value.quantity  # ArrayWUnits, shape (2,) → [bu, bv]

        return None

    def _get_rel_cam_pos(self, target: Body, epoch: EpochArray) -> ArrayWUnits:
        """
        Return target position relative to camera, expressed in camera frame.

        Tries the SPICE camera frame first; falls back to rotating the inertial
        relative position by ``camera.get_rotation_to_camera()``.
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
                obsvr_bdy=self.camera.associated_body_spice_id,
            )[0:3]
            R_ci = self.camera.get_rotation_to_camera(epoch.times.values)
            r_cam = R_ci @ rel_inertial.values
            return ArrayWUnits(r_cam, rel_inertial.units)

    def _determine_in_fov(self, target, epoch, sample: float, line: float) -> bool:
        """
        Checks whether the generated measurement is in the camera
        field of view. We check the direction of the z-component of the delta
        vector between observer and target in the camera frame. We then check
        the line and sample measurements against the max possible line and
        sample determined by the FOV.

        Parameters
        ----------
        sample : float
            The x position of the target in camera frame under gnomonic projection.

        line : float
            The x position of target in camera frame under gnomonic projection.

        z_camera : float
            The z-component of the delta vector between observer and target in the camera frame.

        Returns
        -------
        in_fov_flag : bool
            Flag indicating whether the target is in the camera field of view or not. ``True`` when
            the target is in the field of view, ``False`` when it isn't.
        """
        rel_cam_pos = SpiceManager.get_state(
            trgt_bdy=target.name,
            epoch_time=epoch.times.values,
            reference_frame=self.camera.camera_frame,
            obsvr_bdy=self.camera.name,
        )[0:3]

        z_cam = rel_cam_pos.values[2]
        if z_cam < 0:
            return False

        in_fov = (
            abs(sample.values) < self.camera.max_sample_line[0].values
            and abs(line.values) < self.camera.max_sample_line[1].values
        )
        return in_fov

    def observed_measurements(
        self,
        file_name,
        meas_name: str = "meas_ideal",
        units=None,
        frame: Frame = J2000,
        image_epochs: list | None = None,
    ):
        """
        Load centroiding measurements from a .json file.

        image_epochs : list[float], optional
            Ordered list of observation epoch values, one per image.
            Must be provided when the state vector contains a stacked
            ``centroid_bias_<suffix>`` entry (dim > 2).
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
        Computes the optical navigation measurement for a given target at a specific epoch.

        Parameters
        ----------
        target : Body
            The target body to image.

        epoch_array : EpochArray, optional
            Array of epochs at which to compute the measurement. Defaults to ``None``.

        epoch_start : EpochArray, optional
            Start epoch when building the array internally. Defaults to ``None``.

        epoch_end : EpochArray, optional
            End epoch when building the array internally. Defaults to ``None``.

        tstep : float, optional
            Time step [s] when building the array internally. Defaults to ``1``.

        frame : Frame, optional
            Reference frame for the computation. Defaults to J2000.

        noisy : bool, optional
            When ``True``, adds Gaussian noise scaled by *sigma*. Defaults to ``False``.


        Returns
        -------
        comp_meas : ArrayWUnits
            The computed measurement as an array with units of inverse radians.
        """
        if image_epochs is not None:
            self.set_image_epochs(image_epochs)

        ## Input check
        if epoch_array is None:
            if epoch_start is None or epoch_end is None:
                raise RuntimeError(
                    "Please provide an EpochArray or provide start and end Epochs."
                )
            _units = epoch_start.times.units
            epoch_array = [
                EpochArray(
                    ArrayWUnits(t, _units),
                    timeFrame=epoch_start.system,
                )
                for t in np.arange(
                    epoch_start.times.values,
                    epoch_end.times.values + tstep,
                    tstep,
                )
            ]

        # Extract full bias vector once (dim=2 global, dim=2*n_images stacked per-image).
        _bias_full = self._extract_centroid_bias()

        meas_list = []
        K = np.asarray(self.camera.calibration_matrix)

        for epoch in epoch_array:
            if self._epoch_to_image_index is not None and _bias_full is not None:
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
            z = r_cam[2]
            if z == 0:
                u = 0.0
                v = 0.0
            else:
                pix = (K @ r_cam) / z  # pinhole projection
                u, v = pix[0], pix[1]

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
            np.column_stack((values_u, values_v)), units_u, self.camera.camera_frame
        )

        # For compatibility with other models
        t1 = epoch_array
        t2 = epoch_array
        t3 = epoch_array

        return t1, t2, t3, meas_AWF

    # -----------------------------#
    #   Analytic Partial Derivs    #
    # -----------------------------#
    def _compute_H_cam(self, rel_cam_pos: ArrayWUnits) -> np.ndarray:
        """
        d(u,v)/dr_cam  — Jacobian in camera frame (2×3).

        Uses the pinhole projection: u = K[0,0]*x/z + K[0,2],  v = K[1,1]*y/z + K[1,2].
        """
        K = np.asarray(self.camera.calibration_matrix)
        x, y, z = rel_cam_pos.values

        if z == 0:
            return np.zeros((2, 3))

        H_cam = np.array(
            [
                [K[0, 0] / z, 0.0, -K[0, 0] * x / z**2],
                [0.0, K[1, 1] / z, -K[1, 1] * y / z**2],
            ],
            dtype=float,
        )
        return H_cam

    def _compute_H_pos_inertial(
        self,
        rel_cam_pos: ArrayWUnits,
        R_ci: np.ndarray,
    ) -> np.ndarray:
        """
        d(u,v)/dr_target_inertial  (2×3) via the chain rule:
            d(u,v)/dr_inertial = [d(u,v)/dr_cam] @ R_ci
        where R_ci (3×3) rotates vectors from the inertial frame (J2000) to the
        camera frame.
        """
        return self._compute_H_cam(rel_cam_pos) @ R_ci

    def _compute_H_vel_target(self) -> np.ndarray:
        """
        d(u,v)/dv_target — zero for instantaneous geometric centroiding.
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
        Groups together the different components of measurement
        _partials in the global H-tilde. It returns the H-tilde array for the modelled measurement.

        Parameters
        ----------
        target : Spacecraft
            The target spacecraft.

        epoch : EpochArray
            The epochs.

        frame : Frame
            The reference frame.

        Returns
        -------
        h_tilde : list
            The H-tilde array (2×n) as a list of two rows [row_u, row_v].
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

        # With explicit state_definition: map by component and body
        row_u = []
        row_v = []

        for comp_def in self._state_definition:
            comp_name = comp_def[0]
            comp_dim = comp_def[1]
            comp_body = comp_def[4] if len(comp_def) > 4 else None
            comp_body_sid = getattr(comp_body, "spice_id", None)

            # Position (inertial frame) ---
            # r_cam = R_ci @ (r_spacecraft − r_asteroid)
            # spacecraft (camera body): dmeas/dr_sc   = +H
            # asteroid   (imaged body): dmeas/dr_ast  = −H
            if state_allowed["position"][2].match(comp_name):
                if comp_body_sid == cam_sid:
                    row_u.extend(H_pos_inertial[0].tolist())
                    row_v.extend(H_pos_inertial[1].tolist())
                elif comp_body_sid == target_sid:
                    row_u.extend((-H_pos_inertial[0]).tolist())
                    row_v.extend((-H_pos_inertial[1]).tolist())
                else:
                    row_u.extend([0.0] * comp_dim)
                    row_v.extend([0.0] * comp_dim)

            # Velocity ---
            # Instantaneous model has no velocity dependence for either body.
            elif state_allowed["velocity"][2].match(comp_name):
                row_u.extend([0.0] * comp_dim)
                row_v.extend([0.0] * comp_dim)

            # Centroid pixel bias ---
            # Global (dim=2): du/d[bu,bv] = [1,0], dv/d[bu,bv] = [0,1].
            # Stacked per-image (dim=2*N): indicator at slot [2*img, 2*img+1].
            elif state_allowed["centroid_bias"][2].match(comp_name):
                if comp_body_sid == cam_sid:
                    if self._epoch_to_image_index is not None:
                        _cur = self._epoch_to_image_index.get(float(epoch.times.values))
                        ind_u = [0.0] * comp_dim
                        ind_v = [0.0] * comp_dim
                        if _cur is not None and 2 * _cur + 1 < comp_dim:
                            ind_u[2 * _cur] = 1.0
                            ind_v[2 * _cur + 1] = 1.0
                        row_u.extend(ind_u)
                        row_v.extend(ind_v)
                    else:
                        row_u.extend([1.0, 0.0])  # global mode (dim=2)
                        row_v.extend([0.0, 1.0])
                else:
                    row_u.extend([0.0] * comp_dim)
                    row_v.extend([0.0] * comp_dim)

            # Camera focal length [fx, fy] (K[0,0] and K[1,1])
            # u = fx*x_c/z_c + cx,  v = fy*y_c/z_c + cy
            elif state_allowed["cam_focal_length"][2].match(comp_name):
                if comp_body_sid == cam_sid and z_c != 0:
                    row_u.extend([x_c / z_c, 0.0])  # du/d[fx,fy]
                    row_v.extend([0.0, y_c / z_c])  # dv/d[fx,fy]
                else:
                    row_u.extend([0.0] * comp_dim)
                    row_v.extend([0.0] * comp_dim)

            # Camera principal point offset [cx, cy] (K[0,2] and K[1,2])
            # u = fx*x_c/z_c + cx → du/dcx = 1, du/dcy = 0
            elif state_allowed["cam_pp_offset"][2].match(comp_name):
                if comp_body_sid == cam_sid:
                    row_u.extend([1.0, 0.0])  # du/d[cx,cy]
                    row_v.extend([0.0, 1.0])  # dv/d[cx,cy]
                else:
                    row_u.extend([0.0] * comp_dim)
                    row_v.extend([0.0] * comp_dim)

            # Unknown / unhandled components: zeros
            else:
                row_u.extend([0.0] * comp_dim)
                row_v.extend([0.0] * comp_dim)

        return [row_u, row_v]
