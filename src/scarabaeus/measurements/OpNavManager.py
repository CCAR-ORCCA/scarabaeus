# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import numpy as np


# --------------------#
#  Class Definition  #
# --------------------#
class OpNavManager:
    """Pinhole camera projection shared by the optical measurement models.

    A stateless collection of the calibration matrix, the pinhole projection
    and its Jacobian, so the projection and its derivative have exactly one
    implementation and stay in step.

    All methods are class or static methods; the class is never instantiated.

    See Also
    --------
    scarabaeus.CentroidingIdeal : Centroid model built on this projection.
    """

    # -----------------------------#
    # region  Camera projection    #
    # -----------------------------#
    @staticmethod
    def calibration_matrix(camera) -> np.ndarray:
        """
        Return the pinhole calibration matrix K of a camera.

        ``Camera.calibration_matrix`` is stored as the ``(2, 3)`` projection block
        ``[[fx, 0, cx], [0, fy, cy]]``; the homogeneous row is appended here.

        Parameters
        ----------
        camera : Camera
            The camera whose intrinsics are wanted.

        Returns
        -------
        numpy.ndarray
            ``(3, 3)`` calibration matrix.
        """
        K2 = np.asarray(camera.calibration_matrix, dtype=float)
        return np.vstack((K2, np.array([0.0, 0.0, 1.0])))

    @classmethod
    def project(cls, camera, points_camera: np.ndarray) -> np.ndarray:
        """
        Pinhole-project ``(N, 3)`` camera-frame vectors to ``(N, 2)`` pixels.

        Parameters
        ----------
        camera : Camera
            The camera supplying the calibration matrix.
        points_camera : numpy.ndarray
            ``(N, 3)`` vectors from the camera to the points, in the camera
            frame.

        Returns
        -------
        numpy.ndarray
            ``(N, 2)`` pixel coordinates.
        """
        K = cls.calibration_matrix(camera)
        points_camera = np.asarray(points_camera, dtype=float).reshape(-1, 3)
        return (points_camera @ K[0:2].T) / points_camera[:, 2][:, None]

    @classmethod
    def project_jacobian(cls, camera, point_camera: np.ndarray) -> np.ndarray:
        """
        Return ``d(u, v) / d r_cam`` — the pinhole projection Jacobian.

        Differentiates :meth:`project` for a single point:
        :math:`u = f_x x/z + c_x` and :math:`v = f_y y/z + c_y`, so

        .. math::

            \\frac{\\partial(u, v)}{\\partial\\mathbf{r}_{\\text{cam}}} =
            \\begin{pmatrix}
              f_x/z & 0 & -f_x x/z^2 \\\\
              0 & f_y/z & -f_y y/z^2
            \\end{pmatrix}

        Parameters
        ----------
        camera : Camera
            The camera supplying the calibration matrix.
        point_camera : numpy.ndarray
            ``(3,)`` vector from the camera to the point, in the camera frame.

        Returns
        -------
        numpy.ndarray
            ``(2, 3)`` Jacobian, or zeros when the point lies in the focal
            plane (``z == 0``) and the projection is undefined.
        """
        K = cls.calibration_matrix(camera)
        x, y, z = np.asarray(point_camera, dtype=float).ravel()

        if z == 0:
            return np.zeros((2, 3))

        return np.array(
            [
                [K[0, 0] / z, 0.0, -K[0, 0] * x / z**2],
                [0.0, K[1, 1] / z, -K[1, 1] * y / z**2],
            ],
            dtype=float,
        )

    # endregion Camera projection #
    # -----------------------------#
