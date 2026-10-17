# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import scarabaeus.utils.NumpyWrapper as np
import os
import pickle
import json
from scarabaeus import EpochArray

# --------------------#
#  Class Definition  #
# --------------------#
class Utils:
    """Collection of general-purpose utility methods used throughout Scarabaeus.

    Provides static helpers for array manipulation and file I/O.

    See Also
    --------
    scarabaeus.EpochArray : Time-tag array type accepted by several helpers.
    """
    @staticmethod
    def cross3D(a, b):
        """ 3D vector cross product operation.

            Removes the overhead of numpy.cross for purely 3D vectors.

            Parameters
            ----------
            a : array_like
                First 3D vector, indexable at positions 0, 1 and 2.
            b : array_like
                Second 3D vector, indexable at positions 0, 1 and 2.

            Returns
            -------
            cross_product : numpy.ndarray
                The 3D vector cross product ``a x b``.
        """
        return np.array([a[1]*b[2] - a[2]*b[1],
                         a[2]*b[0] - a[0]*b[2],
                         a[0]*b[1] - a[1]*b[0]])

    @staticmethod
    def skew(vector) -> np.ndarray:
        """ Skew-symmetric cross-product matrix of a 3D vector.

            Parameters
            ----------
            vector : array_like
                3D vector ``a``, holding exactly three elements in any shape.

            Returns
            -------
            skew_matrix : numpy.ndarray
                The (3, 3) matrix ``[a x]``, such that ``skew(a) @ b`` equals
                the cross product ``a x b`` for any 3D vector ``b``.

            Raises
            ------
            ValueError
                If ``vector`` does not hold exactly three elements.
        """
        x, y, z = np.asarray(vector, dtype=float).reshape(3)
        return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])

    @staticmethod
    def norm3D(a):
        return (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]) ** 0.5

    @staticmethod
    def equal_along_rows(arr : np.ndarray) -> bool:
        """
            Check equality along rows of the given array.

            Parameters
            ----------
            arr : numpy.ndarray
                The array to examine.

            Returns
            -------
            are_equal : bool
                Returns ``True`` if all values are equal along rows. Returns 
                ``False`` if they are not.
        """
        return np.all(np.equal(arr[0, :], arr))

    @staticmethod
    def check_slices_equal(arr : np.ndarray):
        """
            Checks if all slices are equal across an array.

            Parameters
            ----------
            arr : numpy.ndarray
                The array to examine.

            Returns
            -------
            are_equal : bool
                Indicates if all slices are equal or not.
        """
        if arr.ndim == 1:
            return True
        # Get the first slice along the last axis
        arr_flat = arr.reshape(-1, arr.shape[-1])
        return Utils.equal_along_rows(arr_flat)

    @staticmethod
    def max_of_last_slice(arr : np.ndarray):
        """
            Finds the maximum value of the last slice of an array.

            Parameters
            ----------
            arr : numpy.ndarray
                The array to examine.

            Returns
            -------
            max_vals : Any
                The max value of the last slice.
        """
        # Compute the maximum along the last axis while keeping the last dimension
        max_vals = np.amax(arr, axis=tuple(range(arr.ndim - 1)))
        return max_vals

    @staticmethod
    def tile_array(arr, desired_shape):
        """
            Construct an array by repeating A the number of times given by reps.

            Parameters
            ----------
            arr : numpy.ndarray
                The array to examine.
            
            desired_shape : tuple of int
                Target shape of the output array.  The last dimension must
                equal ``len(arr)``; all preceding dimensions specify how many
                times to repeat the array along each axis.

            Returns
            -------
            tiled : numpy.ndarray
                The new tiled array.
        """
        if not isinstance(arr, np.ndarray) or len(arr.shape) != 1:
            raise ValueError("Input must be a 1D numpy array")

        if desired_shape[-1] != arr.shape[0]:
            raise ValueError(
                "The last dimension of the desired shape must be equal to the length of the input array"
            )

        # Calculate the tiling factors for each dimension
        tiling_factors = [dim_size for dim_size in desired_shape[:-1]] + [1]

        return np.tile(arr, tiling_factors)

    @staticmethod
    def make_dir(dir: str) -> None:
        """
            Generate a folder in the given directory.

            Parameters
            ----------
            dir : str
                The directory path to a generate a folder in.

            Returns
            -------
            None
        """
        try:
            os.makedirs(dir)
            print(f"Directory '{dir}' created successfully")
        except FileExistsError:
            print(f"Directory '{dir}' already exists")
        except Exception as e:
            print(f"Error: {e}")

    @staticmethod
    def open_pickle(dir : str):
        """
            Return a pickle object handle from dir.

            Parameters
            ----------
            dir : str
                Directory + name of the pickle object.

            Returns
            -------
            pickle_handle : object
                The Python object deserialised from the pickle file.
        """
        with open(dir, "rb") as file:
            pickle_handle = pickle.load(file)
        return pickle_handle
    
    @staticmethod
    def find_date_indices_in_epochs(epochsTDB, target_dates, rtol = 1e-6):
        """
            Find the index or indices of target date(s) in a numpy array of epochs, 
            accounting for floating-point precision.

            Parameters
            ----------
            epochsTDB : EpochArray
                Reference epoch array to search within.

            target_dates : EpochArray or list of EpochArray
                Target epoch(s) to locate inside *epochsTDB*.

            rtol : float, optional
                Relative tolerance for comparing the dates. Defaults to ``1e-6``.

            Returns
            -------
            ind : int or list of ints
                The index or indices of the target date(s), or ``None`` if not found.
        """
        # Ensure target_dates is a list, even if it's a single date
        if not isinstance(target_dates, list):
            target_dates = [target_dates]

        indices = []

        # Loop through each target date and find its index
        for target_date in target_dates:
            # Find the index using np.where and np.isclose for floating-point precision
            index = np.where(
                np.isclose(epochsTDB.times.values, target_date.times.values, rtol=rtol)
            )[0]

            # Check if any index was found and store it
            if index.size > 0:
                indices.append(int(index[0]))
            else:
                indices.append(None)  # Append None if no match is found for that date

        # Return a list of indices, or a single index if only one date was provided
        if len(indices) == 1:
            return indices[0]  # Return a single index if only one date was provided
        else:
            return indices  # Return the list of indices for multiple dates

    @staticmethod
    def load_json(dir: str, print_flag: bool = False) -> dict:
        """
        Load and parse a JSON file from disk.

        Parameters
        ----------
        dir : str
            Path to the ``.json`` file to load.
        print_flag : bool, optional
            When ``True``, prints a confirmation message after a successful
            read. Defaults to ``False``.

        Returns
        -------
        data : dict or list
            Parsed JSON content.

        Raises
        ------
        OSError
            The file could not be read: missing, unreadable, or, for
            cloud-backed storage, not materialised locally.
        json.JSONDecodeError
            The file was read but does not contain valid JSON.
        """
        # Read and parse errors propagate. Swallowing them returns None, which
        # none of the callers check for, so the failure resurfaces later as an
        # unrelated TypeError instead of naming the file that could not be read.
        with open(dir, 'r') as f:
            data = json.load(f)
        if print_flag:
            print(f"JSON file '{dir}' read successfully")

        return data
    
    @staticmethod
    def VandV_plot_errors(
        errorPosition: np.ndarray,
        errorVelocity: np.ndarray,
        times: EpochArray,
        caseId: str,
        integratorName: str,
    ):
        """Plot position and velocity errors in a standardised V&V layout.

        Produces a two-panel figure with position error (top) and velocity
        error (bottom) as functions of elapsed time in days.

        Parameters
        ----------
        errorPosition : numpy.ndarray
            Position error array of shape ``(N, 3)`` in km, computed as
            ``positions_scb - positions_truth``.
        errorVelocity : numpy.ndarray
            Velocity error array of shape ``(N, 3)`` in km/s, computed as
            ``velocities_scb - velocities_truth``.
        times : EpochArray
            Epoch array of length ``N`` corresponding to each error sample.
        caseId : str
            Scenario identifier used as the plot title prefix.
        integratorName : str
            Integration scheme label appended to each subplot title.

        Returns
        -------
        figHandle : matplotlib.figure.Figure
            Handle to the generated figure.
        """


        if isinstance(times, EpochArray):
            times = times.times

        FigHandle = plt.figure()
        plt.subplot(2, 1, 1)
        #plt.plot((times - times[0]) / 86400, abs(errorPosition))
        plt.plot((times - times[0]) / 86400, errorPosition)
        # Set the y-axis to log scale
        #plt.yscale('log')
        plt.title(caseId + " Position Error (" + integratorName + ")")
        plt.xlabel("Days")
        plt.ylabel("Error [km]")
        plt.legend(["X", "Y", "Z"])

        plt.subplot(2, 1, 2)
        #plt.plot((times - times[0]) / 86400, abs(errorVelocity))
        plt.plot((times - times[0]) / 86400, errorVelocity)
        # Set the y-axis to log scale
        #plt.yscale('log')
        plt.title(caseId + " Velocity Error (" + integratorName + ")")
        plt.xlabel("Days")
        plt.ylabel("Error [km/s]")
        plt.legend(["X", "Y", "Z"])

        plt.tight_layout()

        return FigHandle

    @staticmethod
    def VandV_compare(
        positions: np.ndarray,
        velocities: np.ndarray,
        times: EpochArray,
        caseId: str,
        integratorName: str,
        flag_SUN: bool = False,
        leg1_period=None,
        leg2_period=None,
    ):
        """Compare a Scarabaeus-generated trajectory against a SPICE ground truth.

        Loads the SPICE BSP kernel for *caseId*, queries the reference state
        at each epoch in *times*, computes the component-wise and norm errors,
        prints a summary to stdout, and delegates plotting to
        :meth:`VandV_plot_errors`.

        Parameters
        ----------
        positions : numpy.ndarray
            Spacecraft positions from Scarabaeus, shape ``(N, 3)`` in km.
        velocities : numpy.ndarray
            Spacecraft velocities from Scarabaeus, shape ``(N, 3)`` in km/s.
        times : EpochArray
            Epoch array of length ``N`` corresponding to each state sample.
        caseId : str
            Scenario identifier; used to locate the BSP kernel at
            ``<cwd>/data/kernels/validation/<caseId>.bsp``.
        integratorName : str
            Integration scheme label, printed in the summary and plot titles.
        flag_SUN : bool, optional
            When ``True`` uses the Sun as the inertial origin; otherwise uses
            Earth.  Defaults to ``False``.
        leg1_period : ArrayWUnits or None, optional
            Duration of the first trajectory leg in seconds.  Required for
            multi-leg cases (``MONTE_case17a/b/c``); ``None`` otherwise.
        leg2_period : ArrayWUnits or None, optional
            Duration of the second trajectory leg in seconds.  Required for
            multi-leg cases; ``None`` otherwise.

        Returns
        -------
        figHandle : matplotlib.figure.Figure
            Handle to the V&V error plot.
        max_state_error_norm_pos : float
            Maximum position error norm over the trajectory in km.
        max_state_error_norm_vel : float
            Maximum velocity error norm over the trajectory in km/s.
        """
        bspFilePath = [os.getcwd() + "/data/kernels/validation/" + caseId + ".bsp"]
        SpiceManager.load_kernel(bspFilePath)
        bspPositions = np.zeros(positions.shape)
        bspVelocities = np.zeros(velocities.shape)
        
        if flag_SUN is False:
            origin = "EARTH"
        else:
            origin = "SUN"

        if caseId == 'MONTE_case17a' or caseId == 'MONTE_case17c':
            for i in range(1,times.size):
                if times[i].times.values < times[0].times.values + leg1_period.values:
                    epochState = SpiceManager.get_state("-60000", times.times.values[i], "J2000", origin).values
                elif times[i].times.values >= times[0].times.values + leg1_period.values and times[i].times.values < (times[0].times.values + leg1_period.values + leg2_period.values):
                    epochState = SpiceManager.get_state("-60000", times.times.values[i], "J2000", "MARS").values
                else:
                    epochState = SpiceManager.get_state("-60000", times.times.values[i], "J2000", origin).values
                
                bspPositions[i] = epochState[:3]
                bspVelocities[i] = epochState[3:]

    
        elif caseId == 'MONTE_case17b':
            for i in range(1,times.size):
                if times[i].times.values < times[0].times.values + leg1_period.values:
                    epochState = SpiceManager.get_state("-60000", times.times.values[i], "J2000", origin).values
                elif times[i].times.values >= times[0].times.values + leg1_period.values and times[i].times.values < (times[0].times.values + leg1_period.values + leg2_period.values):
                    epochState = SpiceManager.get_state("-60000", times.times.values[i], "J2000", "MARS").values
                else:
                    epochState = SpiceManager.get_state("-60000", times.times.values[i], "J2000", origin).values
                
                bspPositions[i] = epochState[:3]
                bspVelocities[i] = epochState[3:]

        # elif caseId == 'MONTE_case5c':
        #     # sc id is -60 for this one not -600000
        #     for i in range(times.size):
        #         epochState = SpiceManager.get_state(
        #             "-60", times.times.values[i], "J2000", origin
        #         ).values

        #         bspPositions[i] = epochState[:3]
        #         bspVelocities[i] = epochState[3:]

        else:
            for i in range(times.size):
                epochState = SpiceManager.get_state(
                    "-60000", times.times.values[i], "J2000", origin
                ).values

                bspPositions[i] = epochState[:3]
                bspVelocities[i] = epochState[3:]

        print(
            bspPositions[0][0],
            bspPositions[0][1],
            bspPositions[0][2],
            bspVelocities[0][0],
            bspVelocities[0][1],
            bspVelocities[0][2],
        )
        SpiceManager.unload_kernel_from_pool(bspFilePath)
        errorPosition = positions - bspPositions
        errorVelocity = velocities - bspVelocities

        final_state_error_pos = errorPosition[-1]
        final_state_error_vel = errorVelocity[-1]
        final_state_error_norm_pos = np.linalg.norm(errorPosition[-1])
        final_state_error_norm_vel = np.linalg.norm(errorVelocity[-1])

        if caseId == 'MONTE_case17a' or caseId == 'MONTE_case17b' or caseId == 'MONTE_case17c':        
            max_state_error_norm_pos = np.max(np.linalg.norm(errorPosition[1:,:], axis = 1))
            max_state_error_norm_vel = np.max(np.linalg.norm(errorVelocity[1:,:], axis = 1))
        else:
            max_state_error_norm_pos = np.max(np.linalg.norm(errorPosition, axis = 1))
            max_state_error_norm_vel = np.max(np.linalg.norm(errorVelocity, axis = 1))
        
        print('-----------------------------------')
        print(integratorName + " case Final State Error:\nPosition:", final_state_error_pos, "km \nVelocity:", final_state_error_vel, "km/s")
        print('-----------------------------------')
        print('final position error:', final_state_error_norm_pos, 'km')
        print('final velocity error:', final_state_error_norm_vel, 'km/s')
        print('-----------------------------------')
        print('max position error:', max_state_error_norm_pos, 'km')
        print('max velocity error:', max_state_error_norm_vel, 'km/s')
        print('-----------------------------------')

        if caseId == 'MONTE_case17a' or caseId == 'MONTE_case17b' or caseId == 'MONTE_case17c':
            FigHandle = Utils.VandV_plot_errors(errorPosition[1:], errorVelocity[1:], times[1:], caseId, integratorName)
        else:
            FigHandle = Utils.VandV_plot_errors(errorPosition, errorVelocity, times, caseId, integratorName)

        return FigHandle, max_state_error_norm_pos, max_state_error_norm_vel

    # ------------------------------------------------------------------
    # State-space change operators (sequence / mission-node transitions)
    # ------------------------------------------------------------------
    # A node between two legs may contract (drop), reorder and/or expand the
    # estimated state. The change is expressed as one selection/reorder operator
    # Q (Q[i, j] = 1 when old scalar j becomes new scalar i):
    #     X' = Q X,   P' = Q P Q^T,   Phi' = Q Phi Q^T
    # New states (zero rows of Q) keep their a-priori mean (0), covariance
    # (sigma^2) and STM diagonal (identity), added back explicitly below. Blocks
    # are matched on (name, spice_id, occurrence) so multi-body / repeated-name
    # states are handled correctly.

    @staticmethod
    def _state_layout(state_list):
        """
        Lay out a state-definition list into keyed scalar blocks.

        Parameters
        ----------
        state_list : list
            State-definition entries; 5- or 6-tuples where only name (index 0),
            size (index 1) and owner (index 4) are read.

        Returns
        -------
        layout : list of (tuple, int, int)
            One ``(key, start, size)`` per block, where ``key`` is
            ``(name, spice_id, occurrence)`` and ``start`` is the block's scalar
            offset in the stacked state vector.
        total_size : int
            Total number of scalars across all blocks.
        """
        layout = []
        cursor = 0
        counts = {}
        for entry in state_list:
            name = entry[0]
            size = int(entry[1])
            owner = entry[4] if len(entry) > 4 else None
            sid = getattr(owner, "spice_id", None)
            base = (name, sid)
            occ = counts.get(base, 0)
            counts[base] = occ + 1
            layout.append(((name, sid, occ), cursor, size))
            cursor += size
        return layout, cursor

    @staticmethod
    def enumerate_state_blocks(state_list):
        """
        Enumerate the scalar blocks of a state definition with a rich key.

        The key ``(name, size, spice_id, frame_name, occurrence)`` fully
        disambiguates repeated parameters across bodies and occurrences, so it
        can project values between a global state and per-leg definitions with
        one consistent keying convention.

        Parameters
        ----------
        state_list : list
            State-definition entries (6-tuples
            ``(name, size, est, dyn, body, awf)``).

        Yields
        ------
        key : tuple
            ``(name, size, spice_id, frame_name, occurrence)``.
        start : int
            Scalar offset of the block in the stacked state vector.
        size : int
            Number of scalars in the block.
        entry : tuple
            The original state-definition entry.
        """
        cursor = 0
        occ_counter = {}
        for entry in state_list:
            name = entry[0]
            size = int(entry[1])
            body = entry[4] if len(entry) > 4 else None
            awf = entry[5] if len(entry) > 5 else None
            base = (
                name,
                size,
                getattr(body, "spice_id", None),
                getattr(getattr(awf, "frame", None), "name", None),
            )
            occ = occ_counter.get(base, 0)
            occ_counter[base] = occ + 1
            yield (*base, occ), cursor, size, entry
            cursor += size

    @staticmethod
    def build_reorder_operator(old_list, new_list):
        """
        Build the ``Q`` selection/reorder operator mapping the old state to the new.

        Blocks are matched on ``(name, spice_id, occurrence)``. A new block with
        no old counterpart yields an all-zero row in ``Q`` (an expansion target).

        Parameters
        ----------
        old_list : list
            State definition before the node transition.
        new_list : list
            State definition after the node transition.

        Returns
        -------
        Q : numpy.ndarray
            ``(new_n, old_n)`` operator with ``Q[i, j] = 1`` when old scalar ``j``
            maps to new scalar ``i``.
        new_n : int
            Dimension of the new state.
        new_blocks : list of (str, int, int)
            ``(name, start, size)`` of blocks present in ``new_list`` but absent
            from ``old_list`` (expansion targets — zero rows of ``Q``).
        """
        old_layout, old_n = Utils._state_layout(old_list)
        new_layout, new_n = Utils._state_layout(new_list)
        old_idx = {k: (start, size) for k, start, size in old_layout}

        Q = np.zeros((new_n, old_n))
        new_blocks = []
        for key, new_start, new_size in new_layout:
            if key in old_idx:
                old_start, old_size = old_idx[key]
                d = min(new_size, old_size)
                Q[new_start : new_start + d, old_start : old_start + d] = np.eye(d)
            else:
                new_blocks.append((key[0], new_start, new_size))
        return Q, new_n, new_blocks

    @staticmethod
    def reorder_vector(Q, x):
        """
        Apply the reorder operator to a state / deviation vector: ``X' = Q X``.

        Parameters
        ----------
        Q : numpy.ndarray
            Reorder operator from :meth:`build_reorder_operator`.
        x : array_like
            Old state or deviation vector (``old_n`` elements, any shape).

        Returns
        -------
        numpy.ndarray
            The reordered vector as an ``(new_n, 1)`` column. Newly augmented
            states map to zero.
        """
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        return Q @ x

    @staticmethod
    def reorder_covariance(Q, P, new_block_variances=None):
        """
        Reorder a covariance: ``P' = Q P Q^T``, with a-priori variances for new
        states.

        Parameters
        ----------
        Q : numpy.ndarray
            Reorder operator from :meth:`build_reorder_operator`.
        P : array_like
            Old ``(old_n, old_n)`` covariance.
        new_block_variances : list of (int, array_like), optional
            ``(start, variances)`` for each expansion block; ``variances`` is the
            1-D array of a-priori variances (:math:`\\sigma^2`) placed on that
            block's diagonal. Off-diagonal (cross) terms stay zero
            (``P_yz = 0``).

        Returns
        -------
        numpy.ndarray
            The ``(new_n, new_n)`` reordered covariance.
        """
        P2 = Q @ np.asarray(P, dtype=float) @ Q.T
        if new_block_variances:
            for start, variances in new_block_variances:
                variances = np.atleast_1d(np.asarray(variances, dtype=float)).ravel()
                for i, var in enumerate(variances):
                    P2[start + i, start + i] = var
        return P2

    @staticmethod
    def reorder_stm(Q, STM, new_blocks=None):
        """
        Reorder a State Transition Matrix: ``Phi' = Q Phi Q^T``.

        Newly augmented states (zero rows of ``Q``) receive a decoupled identity
        block, matching the convention that a fresh parameter starts uncorrelated
        with unit STM.

        Parameters
        ----------
        Q : numpy.ndarray
            Reorder operator from :meth:`build_reorder_operator`.
        STM : array_like
            Old ``(old_n, old_n)`` State Transition Matrix.
        new_blocks : list of (str, int, int), optional
            ``(name, start, size)`` expansion blocks (as returned by
            :meth:`build_reorder_operator`) to set to identity.

        Returns
        -------
        numpy.ndarray
            The ``(new_n, new_n)`` reordered STM.
        """
        S = Q @ np.asarray(STM, dtype=float) @ Q.T
        if new_blocks:
            for _name, start, size in new_blocks:
                S[start : start + size, start : start + size] = np.eye(size)
        return S
