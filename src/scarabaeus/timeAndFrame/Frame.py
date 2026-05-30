# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Units, ArrayWUnits, EpochArray
import scarabaeus as scb

from typing import Tuple

import scarabaeus.utils.NumpyWrapper as np
import spiceypy as spice
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# define units
unitless = Units.get_units("unitless")

# class definition
class Frame:
    """ SPICE defined reference frame.

        Parameters
        ----------
        name : str
            SPICE recognised frame name (e.g. ``'J2000'``, ``'ITRF93'``,
            ``'ECLIPJ2000'``).

        See Also
        --------
        scarabaeus.ArrayWFrame : 

        Notes
        -----
        If SPICE has no record of the given name, the frame-id and class attributes are 
        set to ``None`` rather than raising an error.

        References
        ----------
        .. [1] NAIF Frames Kernel Tutorial,
          https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/21_fk.pdf

        Examples
        --------
        Create the J2000 frame and examine its properties:

        >>> import scarabaeus as scb
        >>> J2000 = scb.Frame('J2000')
        >>> J2000.disp_properties()
        ========================================
        Frame Properties
        ========================================
        Frame Name:     J2000
        Frame ID:       1
        Origin Name:    SOLAR SYSTEM BARYCENTER
        Origin ID:      0
        Frame Class:    1
        Class ID:       1
        ========================================
    """

    _common_frames_cache = None

    # region Constructor
    def __init__(self, name: str):
        self._name = name

        # Generate additional properties of the frame
        try:
            self._frame_id = scb.SpiceManager.frame_name_to_frame_id(self.name)
            origin, frame_class, class_id = spice.frinfo(self.frame_id)
        except Exception as e:
            origin = None
            frame_class = None
            class_id = None
            print(f"Error retrieving frame information: {e}")
            raise ValueError(
                f"'{name}' is not in the list of regonized frames. Please create a new frame"
            )

        self._origin = origin
        self._origin_name = scb.SpiceManager.get_name_from_id(self._origin)
        self._frame_class = frame_class
        self._class_id = class_id

    # region Properties
    @property
    def name(self) -> str:
        """ The name of the frame. """
        return self._name

    @property
    def frame_id(self) -> int:
        """ The frame ID (SPICE compatible). """
        return self._frame_id

    @property
    def origin(self) -> int:
        """ The origin of the frame (SPICE compatible). """
        return self._origin

    @property
    def origin_name(self) -> str:
        """ The name of the origin of the frame (SPICE compatible). """
        return self._origin_name

    @property
    def frame_class(self) -> int:
        """ The class of the SPICE frame (SPICE compatible). """
        return self._frame_class

    @property
    def class_id(self) -> int:
        """ The class ID the SPICE frame (SPICE compatible). """
        return self._class_id

    # endregion Properties

    # region Methods
    def __repr__(self):
        return f"{self._name} ({self._origin} - {self._origin_name})"

    def disp_properties(self) -> None:
        """ Displays the frame object properties. """
        print("=" * 40)
        print(f"Frame Properties")
        print("=" * 40)
        print(f"Frame Name:     {self._name}")
        print(f"Frame ID:       {self._frame_id}")
        print(f"Origin Name:    {self._origin_name}")
        print(f"Origin ID:      {self._origin}")
        print(f"Frame Class:    {self._frame_class}")
        print(f"Class ID:       {self._class_id}")
        print("=" * 40)

    @staticmethod
    def get_DCM(
        source_frame: Self, target_frame: Self, epoch: EpochArray
    ) -> np.ndarray:
        """ Gets the Direction Cosine Matrix (DCM) given the source 
            and target frames at a given epoch.

            Parameters
            ----------
            source_frame : Frame
                The source frame which the DCM performs a transformation from.

            target_frame : Frame
                The target frame which the DCM performs a transformation to.

            epoch : EpochArray
                The ephemeris time at which the transformation holds.

            Returns
            -------
            DCM : numpy.ndarray
                The requested DCM.
        """
        ## input validation
        # epoch must be TDB
        if epoch.system != "TDB":
            epoch = epoch.to(sys = "TDB")

        # convert given epoch to ephemeris times for SPICE
        epoch_et = epoch.times.values

        # get DCM from spices
        dcm: np.array = scb.SpiceManager.get_xfrm(
            source_frame.name, target_frame.name, epoch_et
        )

        # ensure DCM is valid
        if not dcm.shape[0] == dcm.shape[1]:
            # must be square
            raise ValueError("DCM Rotation Matrix should be a square Matrix")

        if not (
            round(np.linalg.det(dcm)) == 1
            and np.allclose(np.dot(dcm.T, dcm), np.identity(3))
        ):
            # and othonormal
            raise ValueError("DCM Rotation Matrix should be an Orthonormal Matrix.")

        # valid DCM
        return dcm

    @staticmethod
    def get_relative_pos(
        source_frame: Self, target_frame: Self, epoch: EpochArray
    ) -> ArrayWUnits:
        """ Get the translation vector between origin of reference frames
            given the source and target frames at a given epoch.

            Parameters
            ----------
            source_frame : Frame
                The source frame which the translation vector needs to be computed from.

            target_frame : Frame
                The target frame which the translation vector needs to be computed to.

            epoch : EpochArray
                The ephemeris time at which the transformation holds.

            Returns
            -------
            vec : ArrayWUnits
                The requested translation vector between origins.
        """
        if epoch.system != "TDB":
            raise ValueError("Epoch array must be in TDB (ephemeris time)")
        source_origin = source_frame.origin_name
        target_origin = target_frame.origin_name
        target_orientation = target_frame.name
        epoch_et = epoch.times.values
        t_AWU = scb.SpiceManager.get_pos(
            trgt_bdy=target_origin,
            epoch_time=epoch_et,
            reference_frame=target_orientation,
            obsvr_bdy=source_origin,
            ab_correct="None",
        )
        t_AWF = scb.ArrayWFrame(t_AWU, target_frame)
        return t_AWF

    @staticmethod
    def get_relative_vel(
        source_frame: Self, target_frame: Self, epoch: EpochArray
    ) -> ArrayWUnits:
        """ Get the translation vector between origin of reference frames
            given the source and target frames at a given epoch.

            Parameters
            ----------
            source_frame : Frame
                The source frame which the translation vector needs to be computed from.

            target_frame : Frame
                The target frame which the translation vector needs to be computed to.

            epoch : EpochArray
                The ephemeris time at which the transformation holds.

            Returns
            -------
            vec : ArrayWUnits
                The requested translation vector between origins.
        """
        if epoch.system != "TDB":
            raise ValueError("Epoch array must be in TDB (ephemeris time)")
        source_origin = source_frame.origin_name
        target_origin = target_frame.origin_name
        target_orientation = target_frame.name
        epoch_et = epoch.times.values
        t_AWU = scb.SpiceManager.get_vel(
            trgt_bdy=target_origin,
            epoch_time=epoch_et,
            reference_frame=target_orientation,
            obsvr_bdy=source_origin,
            ab_correct="None",
        )
        t_AWF = scb.ArrayWFrame(t_AWU, target_frame)
        return t_AWF

    @staticmethod
    def get_transformation(
        source_frame: Self, target_frame: Self, epoch: EpochArray
    ) -> ArrayWUnits:
        """ Get 4x4 transformation matrix that combine rotation and translation between
            source and target frames at a given epoch.
            Parameters
            ----------
            source_frame : Frame
                The source frame which the transformation matrix needs to be computed from.

            target_frame : Frame
                The target frame which the transformation matrix needs to be computed to.

            epoch : EpochArray
                The ephemeris time at which the transformation holds.

            Returns
            -------
            T : ArrayWUnits
                The requested (4x4) transformation matrix T.
        """
        if epoch.system != "TDB":
            raise ValueError("Epoch array must be in TDB (ephemeris time)")

        n_elements = epoch.size
        # Get rotation matrix and translation vector
        T = []
        for ii in range(0, n_elements):
            if epoch.size > 1:
                R = Frame.get_DCM(source_frame, target_frame, epoch[ii])
                t = Frame.get_relative_pos(source_frame, target_frame, epoch[ii])
            else:
                R = Frame.get_DCM(source_frame, target_frame, epoch)
                t = Frame.get_relative_pos(source_frame, target_frame, epoch)

            # Construct the 4x4 transformation matrix
            # T = [[R|t],[0|1]] of size 4x4 = [[3x3|3x1],[1x3|1x1]]
            # Generate the values
            T_values = np.zeros((4, 4))  # Initialize a 4x4 matrix with zeros
            T_values[:3, :3] = R  # Top-left 3x3 is the rotation matrix
            T_values[:3, 3] = t.quantity.values  # Top-right 3x1 is the translation vector
            T_values[3, 3] = 1  # Bottom-right is 1 for homogeneous coordinates

            # create AWU
            t_units = t.quantity.units
            T_units = np.array(
                [
                    [unitless, unitless, unitless, t_units],
                    [unitless, unitless, unitless, t_units],
                    [unitless, unitless, unitless, t_units],
                    [unitless, unitless, unitless, unitless],
                ]
            )

            T.append(ArrayWUnits(T_values, T_units))

        return T

    @staticmethod
    def generate_common_frames() -> Tuple[Self]:
        """ Gets a set of common frames. Generates four widely used frames:

            * J2000
            * ITRF93
            * ECLIPJ2000
            * IAUEARTH

            Returns
            -------
            commmon_frames : tuple[Frame]
                A tuple of common frames.

            Examples
            --------
            >>> import scarabaeus as scb
            >>> J2000, ITRF93, ECLIPJ2000, IAUEARTH = scb.Frame.generate_common_frames()
        """
        if Frame._common_frames_cache is None:
            J2000 = Frame("J2000")
            ITRF93 = Frame("ITRF93")
            ECLIPJ2000 = Frame("ECLIPJ2000")
            IAUEARTH = Frame("IAU_EARTH")
            Frame._common_frames_cache = (J2000, ITRF93, ECLIPJ2000, IAUEARTH)

        return Frame._common_frames_cache

    @staticmethod
    def write_pck_frame(
        file_name: str, frame_name: str, frame_class: int, frame_id: int
    ) -> None:
        """ Generate a PCK SPICE frame [[1]_].
        
            These frames are used to describe the orientation of natural celestial 
            bodies (e.g., planets, moons, asteroids) in space.

            Parameters
            ----------
            file_name : str
                The name of the output file (e.g., 'EROS_EXAMPLE_FRAME.pck').

            frame_name : str
                The name of the frame (e.g., 'EROS_FIXED').

            frame_class : int
                The frame class (e.g., 2).

            frame_id : int
                The frame ID (e.g., 2000433).

            Returns
            -------
            None

            References
            ----------
            .. [1] Spice Reference Frames Required Reading,
              https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html

            Examples
            --------
            Create a PCK frame for Eros:

            >>> import scarabaeus as scb
            >>> from pathlib import Path
            >>> mk_path = Path('path/to/example_mk.tm')
            >>> scb.SpiceManager.load_kernel_from_mkfile(str(mk_path))  # furnish metakernel
            >>> scb.Frame.write_pck_frame(
            ...     file_name   = 'EROS_EXAMPLE_FRAME.pck',
            ...     frame_name  = 'EROS_FIXED',
            ...     frame_class = 2,
            ...     frame_id    = 2000433,
            ... )
        """
        scb.SpiceManager._write_pck_frame(file_name, frame_name, frame_class, frame_id)

    @staticmethod
    def write_ck_frame(
        file_name: str,
        frame_name: str,
        frame_id: int,
        center_id: int,
        sclk_id: int,
        spk_id: int,
    ) -> None:
        """ Generate a CK SPICE frame [[1]_]. 
        
            Time-dependent frames that represent the orientation of a spacecraft or a 
            part of it (e.g., an instrument, a solar panel) as a function of time.

            Parameters
            ----------
            file_name : str
                The output file name (e.g., 'MGS_EXAMPLE_FRAME.tf').

            frame_name : str
                The name of the frame (e.g., 'MGS_SPACECRAFT').

            frame_id : int
                The unique frame ID (e.g., -94000).

            center_id : int
                The body ID at the center of the frame (e.g., -94).

            sclk_id : int
                The spacecraft clock ID associated with the frame (e.g., -94).

            spk_id : int
                The SPK ID associated with the frame (e.g., -94).

            Returns
            -------
            None

            References
            ----------
            .. [1] Spice Reference Frames Required Reading,
              https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html

            Examples
            --------
            Create a CK for the mars global surveyor:

            >>> import scarabaeus as scb
            >>> from pathlib import Path
            >>> mk_path = Path('path/to/example_mk.tm')
            >>> scb.SpiceManager.load_kernel_from_mkfile(str(mk_path))  # furnish metakernel
            >>> scb.Frame.write_ck_frame(
            ...     file_name  = 'MGS_EXAMPLE_FRAME.tf',
            ...     frame_name = 'MGS_SPACECRAFT',
            ...     frame_id   = -94000,
            ...     center_id  = -94,
            ...     sclk_id    = -94,
            ...     spk_id     = -94,
            ... )
        """
        scb.SpiceManager._write_ck_frame(
            file_name, frame_name, frame_id, center_id, sclk_id, spk_id
        )

    @staticmethod
    def write_tk_frame(
        file_name: str,
        frame_name: str,
        frame_id: int,
        center_id: int,
        relative_frame: int,
        matrix: list = None,
    ) -> None:
        """ Generate a TK SPICE frame [[1]_]. 
        
            TK frames are user-defined or mission-specific frames that are statically 
            defined. They are often used to define spacecraft-specific or 
            instrument-specific frames.

            Parameters
            ----------
            file_name : str
                The output file name (e.g., 'MARS_EXAMPLE_FRAME.tf').

            frame_name : str
                The name of the frame (e.g., 'MARS_FIXED').

            frame_id : int
                The unique frame ID (e.g., 1400499).

            center_id : int
                The body ID at the center of the frame (e.g., 499).

            relative_frame : int
                The relative reference frame (e.g., 'IAU_MARS').

            matrix : list, optional
                3x3 transformation matrix. Defaults to the identity matrix. 
                Defaults to ``None``.

            Returns
            -------
            None

            References
            ----------
            .. [1] Spice Reference Frames Required Reading,
              https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html

            Examples
            --------
            Create a CK for the mars global surveyor:

            >>> import scarabaeus as scb
            >>> from pathlib import Path
            >>> mk_path = Path('path/to/example_mk.tm')
            >>> scb.SpiceManager.load_kernel_from_mkfile(str(mk_path))  # furnish metakernel
            >>> scb.Frame.write_tk_frame(
            ...     file_name      = 'MARS_EXAMPLE_FRAME.tf',
            ...     frame_name     = 'MARS_FIXED',
            ...     frame_id       = 1400499,
            ...     center_id      = 499,
            ...     relative_frame = 'IAU_MARS,
            ...     matrix         = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            ... )
        """
        scb.SpiceManager._write_tk_frame(
            file_name, frame_name, frame_id, center_id, relative_frame, matrix
        )
