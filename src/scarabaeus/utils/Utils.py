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

    # ----------------#
    # region Methods  #
    # ----------------#
    
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
        """
        data = None
        try:
            # Open and load the JSON file
            with open(dir, 'r') as f:
                data = json.load(f)
            if print_flag:
                print(f"JSON file '{dir}' read successfully")
        except Exception as e:
            print(f"Error: {e}")

        return data
    
