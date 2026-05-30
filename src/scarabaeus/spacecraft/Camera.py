# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Units, ArrayWUnits, Instrument

import scarabaeus.utils.NumpyWrapper as np

# ------------------#
#  Generate Units  #
# ------------------#
km = Units.get_units("km")


# --------------------#
#  Class Definition  #
# --------------------#
class Camera(Instrument):
    """ Represents a camera instrument on a spacecraft.

    The pinhole (perspective) camera model maps a 3-D scene point
    :math:`\\mathbf{P} = (X, Y, Z)^T` (expressed in the camera frame) to
    image-plane pixel coordinates :math:`(u, v)`:

    .. math::

        \\begin{pmatrix}u \\\\ v\\end{pmatrix}
        = \\mathbf{K}\\,\\begin{pmatrix}X/Z \\\\ Y/Z\\end{pmatrix},
        \\qquad
        \\mathbf{K} = \\begin{pmatrix}f/\\delta_u & 0 & c_u \\\\
                                     0 & f/\\delta_v & c_v\\end{pmatrix}

    where :math:`f` is the focal length, :math:`(\\delta_u, \\delta_v)` are
    the instantaneous field-of-view (IFOV) dimensions per pixel, and
    :math:`(c_u, c_v)` is the principal-point offset.  This matrix is
    stored in :attr:`calibration_matrix`.

    Parameters
    ----------
    name : str
        Camera name.

    associated_body_spice_id : int | str
        SPICE ID of the associated body.

    camera_frame : str
        Reference frame of the camera.

    measurement_type : str
        Type of measurement (e.g., image, lidar, etc.).

    fov_angular : ArrayWUnits
        Angular field of view of the camera.

    fov_pixels : tuple[int, int]
        Number of pixels along horizontal and vertical axes.

    focal_length : ArrayWUnits
        Focal length of the camera.

    pp_offset : ArrayWUnits, optional
        Principal point offset in the image plane. If ``pp_offset = None``, defaults to
        ``[0, 0] [unitless ... unitless]``.

    pos_craft_cntrd : ArrayWFrame, optional
        Defaults to ``None``.

    See Also
    --------
    scarabaeus.Instrument : Parent class providing SPICE ID and measurement-type handling.
    scarabaeus.CentroidingIdeal : Optical measurement model that uses Camera internally.
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        name: str,
        associated_body_spice_id: int | str,
        camera_frame: str,
        fov_angular: ArrayWUnits,
        fov_pixels: tuple,
        focal_length: ArrayWUnits,
        ifov_angle: ArrayWUnits,
        pp_offset: ArrayWUnits = None,
        measurement_type: str = "opnav",
    ):
        # Initialize the base Instrument class
        super().__init__(name, associated_body_spice_id, measurement_type)

        # Camera-specific attributes
        self._associated_body_spice_id = associated_body_spice_id
        self._camera_frame = camera_frame
        self._focal_length = focal_length  # Focal length in millimeters
        self._fov_angular = fov_angular
        self._fov_pixels = fov_pixels
        self._ifov_angle = ifov_angle
        self.pp_offset = pp_offset

        # Compute the Instantaneous Field of View (IFOV) for each axis
        self._ifov = 2 * focal_length * np.tan(ifov_angle / 2)  # millimeters

        # Compute focal length in pixels
        self._focal_length_pixels = focal_length / self._ifov  # pixels

        # Compute the camera calibration matrix
        a = self.focal_length_pixels[0].values / self._ifov[0].values
        e = self.focal_length_pixels[1].values / self._ifov[1].values
        c, f = self.pp_offset.values[0], self.pp_offset.values[1]
        self._calibration_matrix = np.array([[a, 0, c], [0, e, f]])

        # Compute the maximum sample and line dimensions
        self._max_sample_line = self._compute_max_sample_line()  # pixels

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def associated_body_spice_id(self) -> int | str:
        """
        SPICE ID of craft or body camera is attached to.
        """
        return self._associated_body_spice_id

    @property
    def camera_frame(self) -> int | str:
        """
        SPICE-defined frame.
        """
        return self._camera_frame

    @property
    def fov_angular(self) -> ArrayWUnits:
        """
        Angular field-of-view of the camera (horizontal, vertical) in radians.
        """
        return self._fov_angular

    @property
    def fov_pixels(self) -> tuple[float, float]:
        """
        Sensor resolution (horizontal, vertical).
        """
        return self._fov_pixels

    @property
    def ifov(self) -> ArrayWUnits:
        """
        Instantaneous field-of-view of the sensor.
        """
        return self._ifov

    @ifov.setter
    def ifov(self, input_val):
        if input_val is None:
            self._ifov = self._fov_angular[0] / self._fov_pixels[0]
        else:
            self._ifov = input_val

    @property
    def focal_length(self) -> float:
        """
        The focal length of the camera. Expressed in units of kilometers.
        """
        return self._focal_length

    @property
    def focal_length_pixels(self) -> tuple[int, int]:
        """
        The focal length of the camera. Expressed in pixels
        """
        return self._focal_length_pixels

    @property
    def pp_offset(self) -> ArrayWUnits:
        """
        Offset of the principal point. Expressed in units of kilometers.
        """
        return self._pp_offset

    @pp_offset.setter
    def pp_offset(self, input_val):
        # -----------------------------------#
        # input validation and conditioning #
        # -----------------------------------#
        if isinstance(input_val, ArrayWUnits):
            # given AWU -> validate
            if input_val.homogeneous_units:
                if input_val.units != km:
                    # expected units in kilometers
                    bad_units_err = (
                        "Principal point offset must be expressed in "
                        f"units of kilometers. Received {input_val.units}."
                    )
                    raise ValueError(bad_units_err)

            else:
                # expected the same units
                non_hom_err = (
                    "Principal point offset must be given in matching units. "
                    f"Received {input_val.units}."
                )
                raise ValueError(non_hom_err)

        elif input_val is None:
            # defaulted to None -> create AWU
            input_val = ArrayWUnits(np.array([0, 0]), km)

        else:
            # given an invalid input type
            bad_type_err = (
                "Principal point offset must be given as an ArrayWUnits. "
                f"Received {type(input_val)}."
            )
            raise TypeError(bad_type_err)

        # -----------------#
        # assign property #
        # -----------------#
        self._pp_offset = input_val

    @property
    def calibration_matrix(self) -> np.ndarray:
        """
        Camera calibration matrix.
        """
        return self._calibration_matrix

    @property
    def max_sample_line(self) -> ArrayWUnits:
        """
        Maximum sampling line. Unitless.
        """
        return self._max_sample_line

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    def get_rotation_to_camera(self, epoch_time: float) -> np.ndarray:
        """
        Return the 3×3 rotation matrix that maps vectors from J2000 to camera frame.

        Tries SPICE first (requires a SPICE camera-frame kernel). Falls back to a
        manually set attitude matrix (``camera.attitude_matrix = R``), or identity
        if neither is available.
        """
        from scarabaeus import SpiceManager

        try:
            return np.asarray(
                SpiceManager.get_xfrm(
                    frame_from="J2000",
                    frame_to=self._camera_frame,
                    epoch=epoch_time,
                ),
                dtype=float,
            )
        except Exception:
            return getattr(self, "_attitude_matrix", np.eye(3))

    @property
    def attitude_matrix(self) -> np.ndarray:
        """
        Fallback 3×3 rotation matrix (J2000 → camera frame) used when SPICE
        does not have the camera frame defined. Identity by default.
        """
        return getattr(self, "_attitude_matrix", np.eye(3))

    @attitude_matrix.setter
    def attitude_matrix(self, R: np.ndarray):
        self._attitude_matrix = np.asarray(R, dtype=float)

    def target_in_fov(
        self,
        target,
        epoch: "EpochArray",
    ) -> "bool | np.ndarray":
        """
        Determines whether a target body is within the camera FOV for a single
        epoch or a sequence of epochs.

        Uses pinhole projection in the camera frame: the target is considered
        visible when its z-component (along the boresight) is positive and the
        projected pixel coordinates fall within the sensor bounds defined by
        ``max_sample_line``.

        Parameters
        ----------
        target : Body or Spacecraft
            The target body.  Its SPICE name is used to query its state.

        epoch : EpochArray
            The epoch or sequence of epochs at which to evaluate visibility.

        Returns
        -------
        in_fov : bool or np.ndarray of bool
            ``True`` if the target projects into the camera sensor bounds.
            Returns a scalar for a single epoch or an array for multiple epochs.
        """
        from scarabaeus import SpiceManager

        epoch_vals = np.atleast_1d(epoch.times.values)
        scalar_input = epoch_vals.size == 1
        K = np.asarray(self._calibration_matrix)

        in_fov_list = []
        for t in epoch_vals:
            try:
                rel_state = SpiceManager.get_state(
                    trgt_bdy=target.name,
                    epoch_time=t,
                    reference_frame=self._camera_frame,
                    obsvr_bdy=self.name,
                )
                r_cam = np.asarray(rel_state.values[0:3])
            except Exception:
                rel_inertial = SpiceManager.get_state(
                    trgt_bdy=target.name,
                    epoch_time=t,
                    reference_frame="J2000",
                    obsvr_bdy=str(self._associated_body_spice_id),
                )
                R_ci = self.get_rotation_to_camera(t)
                r_cam = R_ci @ np.asarray(rel_inertial.values[0:3])

            z = r_cam[2]
            if z <= 0:
                in_fov_list.append(False)
                continue

            pix = (K @ r_cam) / z
            in_fov_list.append(
                abs(pix[0]) < self._max_sample_line.values[0]
                and abs(pix[1]) < self._max_sample_line.values[1]
            )

        return in_fov_list[0] if scalar_input else np.array(in_fov_list)

    def _compute_max_sample_line(self):
        """
        Compute the maximum sample and line half-extents relative to the focal centre.

        Uses the FOV pixel count and the principal-point offset to determine the
        pixel-coordinate bounds of the image plane.

        Returns
        -------
        max_sample_line : ArrayWUnits
            2-element array ``[max_sample, max_line]`` in pixel coordinates.
        """

        # Compute the maximum sample and line dimensions relative to the focal center
        max_sample = self.fov_pixels[0] / 2 + self.pp_offset[0].values
        max_line = self.fov_pixels[1] / 2 + self.pp_offset[1].values

        # Store the max sample and line in pixel coordinates
        # NOTE: Computed in pixels!
        self._max_sample_line = ArrayWUnits(np.array([max_sample, max_line]), None)

        return self._max_sample_line
