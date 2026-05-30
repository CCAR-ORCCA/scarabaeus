# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import numpy as np
import os
from scarabaeus import MeasurementEditing


# -----------------------#
#    Class Definition   #
# -----------------------#
class FilterDataManager:
    """ Manages, edits, and organizes measurement datasets for orbit determination filters.

    The class provides utilities to combine multiple
    :class:`~scarabaeus.MeasurementDataSet` objects into a unified time-indexed structure
    keyed by spacecraft time ``t2``. This enables efficient retrieval
    and reconstruction of synchronized measurement blocks across heterogeneous
    datasets, instruments, and measurement types.

    In addition, the class supports optional pre-processing and measurement
    editing operations such as chi-squared filtering, lasso-based selection,
    and date-range editing before the datasets are indexed and passed to the
    estimation framework.

    The resulting combined structure is designed to support linearized and
    statistical orbit determination workflows involving multi-instrument,
    multi-spacecraft, or mixed observable estimation problems.

    Parameters
    ----------
    datasets : list of MeasurementDataSet, optional
        Input datasets. Nested lists are flattened one level.
    editing_method : str or None
        Editing mode to apply before building indices. One of ``'lasso'``,
        ``'chi2'``, ``'date_range'``, ``'interactive_date_range'``, or ``None``.
    editing_datasets : None or str or list of str
        Which datasets to edit. ``None`` = all, a string = one by name,
        a list of strings = a specific subset.
    editing_kwargs : dict or None
        Extra keyword arguments forwarded to the editing method.
    save_path : str or None
        Directory path to save combined data ``.npz`` files.
    save_name : str or None
        Optional name prefix included in saved file names.

    Notes
    -----
    Two-way radiometric time-tag convention used throughout this class:

    * ``t1`` — ground station **transmit** epoch (uplink signal sent).
    * ``t2`` — **spacecraft time** (signal reflection / turnaround epoch).
    * ``t3`` — ground station **receive** epoch (downlink signal received).

    Datasets are indexed by spacecraft epoch ``t2``; all datasets must use the
    same time-tag convention.  Nested list inputs are flattened one level
    automatically.

    See Also
    --------
    scarabaeus.MeasurementEditing : Provides the lasso/chi2/date-range editing methods.
    scarabaeus.FilterOD : Base filter that consumes the combined indexed structure.
    """

    # -------------#
    # Constructor  #
    # -------------#
    def __init__(
        self,
        datasets=None,
        editing_method=None,
        editing_datasets=None,
        editing_kwargs=None,
        save_path=None,
        save_name=None,
    ):
        # Flatten once (only one nesting level needed)
        if datasets is None:
            datasets = []
        else:
            datasets = [
                d
                for item in datasets
                for d in (item if isinstance(item, (list, tuple)) else [item])
            ]

        self._datasets = datasets
        self._indices_by_t2 = None
        self._datasets_names = []

        if self._datasets:
            self._indices_by_t2 = self._combine_indices_by_t2(self._datasets)
            self._datasets_names = [d.set_name for d in self._datasets]

        # Optional: apply measurement editing before building indices
        if editing_method and self._datasets:
            editor = MeasurementEditing(self._datasets)
            kw = editing_kwargs or {}

            # Resolve which dataset names to edit
            if editing_datasets is None:
                target_names = [d.set_name for d in self._datasets]
            elif isinstance(editing_datasets, str):
                target_names = [editing_datasets]
            else:
                target_names = list(editing_datasets)

            for name in target_names:
                if editing_method == "lasso":
                    editor.lasso_editor(name, **kw)
                elif editing_method == "chi2":
                    editor.chi_squared_filter(dataset_name=name, **kw)
                elif editing_method == "date_range":
                    editor.date_range_filter(dataset_name=name, **kw)
                elif editing_method == "interactive_date_range":
                    editor.interactive_date_range(dataset_name=name, **kw)
                else:
                    raise ValueError(
                        f"Unknown editing_method '{editing_method}'. "
                        "Choose: 'lasso', 'chi2', 'date_range', 'interactive_date_range'."
                    )

            self._datasets = editor.get_datasets()
            self._indices_by_t2 = self._combine_indices_by_t2(self._datasets)
            self._datasets_names = [d.set_name for d in self._datasets]

            # Optional: save combined data structure to JSON for inspection
            if save_path is not None and self._datasets:
                self._save_datasets(save_path, save_name)

    # -----------------------------------
    #             Properties
    # -----------------------------------
    @property
    def datasets(self) -> list:
        """Return the list of managed datasets."""
        return self._datasets

    @datasets.setter
    def datasets(self, value) -> None:
        """Set the list of datasets and rebuild indices."""
        self._datasets = value
        if self._datasets:
            self._indices_by_t2 = self._combine_indices_by_t2(self._datasets)
        else:
            self._indices_by_t2 = None

    @property
    def indices_by_t2(self) -> dict:
        """Return the indices grouped by unique t2 values."""
        return self._indices_by_t2

    @property
    def dataset_names(self) -> list:
        """Return the names of the datasets."""
        return self._datasets_names

    # -----------------------------------
    #             Methods
    # -----------------------------------

    @classmethod
    def _combine_indices_by_t2(cls, datasets: list) -> dict:
        """
        Combine multiple MeasurementDataSets into an index dictionary keyed by t2.

        Groups all measurement indices from every dataset by their unique spacecraft
        time ``t2``.  The result is the primary lookup structure used by the
        filter to retrieve measurements epoch by epoch.

        Parameters
        ----------
        datasets : list of MeasurementDataSet
            Input datasets to index.

        Returns
        -------
        dict
            Keys are unique ``t2`` float values sorted in ascending order.
            Each value is a list of dicts with keys:

            * ``'dataset'`` — reference to the originating :class:`~scarabaeus.MeasurementDataSet`.
            * ``'indices'`` — ``np.ndarray`` of integer row indices within that dataset
              that share this ``t2`` value.
        """
        # Group indices by unique t2 across all datasets
        combined_by_t2 = {}
        for dset in datasets:
            t2 = np.array(dset.data["t2"])
            unique_t2, inverse_idx = np.unique(t2, return_inverse=True)
            for i, t in enumerate(unique_t2):
                idxs = np.nonzero(inverse_idx == i)[0]
                key = float(t) if np.isscalar(t) else t
                if key not in combined_by_t2:
                    combined_by_t2[key] = []
                combined_by_t2[key].append({"dataset": dset, "indices": idxs})

        # Return a dict ordered by ascending t2
        return {k: combined_by_t2[k] for k in sorted(combined_by_t2)}

    @staticmethod
    def _create_combined_from_indices(indices_by_t2):
        """
        Construct the combined measurement structure (numeric arrays, stacked partials, list fields)
        from indices grouped by t2.

        Supports single t2 (tuple) or multiple t2 (dict) input.
        Returns a dict whose keys are sorted ascending by t2.
        """

        # Normalize single (t2, blocks) tuple into dict
        if isinstance(indices_by_t2, tuple) and len(indices_by_t2) == 2:
            indices_by_t2 = {float(indices_by_t2[0]): indices_by_t2[1]}

        combined_by_t2 = {}
        t2_keys = sorted(indices_by_t2)

        # Process each t2 group
        for t2_val in t2_keys:
            blocks = indices_by_t2[t2_val]
            combined_block = {
                "t1": [],
                "t2": [],
                "t3": [],
                "observed": [],
                "computed": [],
                "residuals": [],
                "sigma": [],
                "outlier_flag": [],
                "partials": [],
                "instruments": [],
                "measurement_types": [],
                "dataset_names": [],
                "dataset_refs": [],
            }

            for block in blocks:
                dset = block["dataset"]
                idxs = block["indices"]
                if isinstance(idxs, (np.integer, int)):
                    idxs = [int(idxs)]

                # Numeric arrays
                combined_block["t1"].append(np.array(dset.data["t1"])[idxs])
                combined_block["t2"].append(np.array(dset.data["t2"])[idxs])
                combined_block["t3"].append(np.array(dset.data["t3"])[idxs])
                combined_block["observed"].append(np.array(dset.data["observed"])[idxs])
                combined_block["computed"].append(np.array(dset.data["computed"])[idxs])

                # Ensure residuals are 2D column vectors
                combined_block["residuals"].append(
                    np.array(dset.data["residuals"])[idxs].reshape(-1, 1)
                )

                combined_block["sigma"].append(np.array(dset.data["sigma"])[idxs])
                combined_block["outlier_flag"].append(
                    np.array(dset.data["outlier_flag"])[idxs]
                )

                # Partials always 2D
                partials = dset.data["partials"]
                if isinstance(partials, np.ndarray) and partials.dtype != object:
                    part_block = np.atleast_2d(partials)
                else:
                    rows = [np.asarray(p).ravel() for p in partials]
                    widths = np.array([r.size for r in rows], dtype=int)

                    w_min = int(widths.min())
                    w_max = int(widths.max())

                    if w_min == w_max:
                        part_block = np.vstack(rows)
                    else:
                        # treat shorter rows as missing trailing columns -> fill with zeros
                        # This is safe ONLY if the column ordering is consistent and the missing params are at the end.
                        part_block = np.zeros((len(rows), w_max), dtype=float)
                        for i, r in enumerate(rows):
                            if r.size == 0:
                                continue
                            part_block[i, : r.size] = r  # place left-aligned
                combined_block["partials"].append(part_block[idxs, :])

                # List fields
                combined_block["instruments"].extend([dset.instrument] * len(idxs))
                combined_block["measurement_types"].extend(
                    [dset.measurement_type] * len(idxs)
                )
                combined_block["dataset_names"].extend([dset.set_name] * len(idxs))
                combined_block["dataset_refs"].extend([dset] * len(idxs))

            # Concatenate numeric arrays along rows
            for key in [
                "t1",
                "t2",
                "t3",
                "observed",
                "computed",
                "residuals",
                "sigma",
                "outlier_flag",
            ]:
                combined_block[key] = np.concatenate(combined_block[key], axis=0)

            # Stack partials vertically
            combined_block["partials"] = np.vstack(combined_block["partials"])

            # Keep only non-rejected measurements
            keep = ~combined_block["outlier_flag"].astype(bool)
            if keep.any():
                for key in [
                    "t1",
                    "t2",
                    "t3",
                    "observed",
                    "computed",
                    "residuals",
                    "sigma",
                    "outlier_flag",
                ]:
                    combined_block[key] = combined_block[key][keep]
                combined_block["partials"] = combined_block["partials"][keep, :]
                for key in [
                    "instruments",
                    "measurement_types",
                    "dataset_names",
                    "dataset_refs",
                ]:
                    combined_block[key] = [
                        v for v, k in zip(combined_block[key], keep) if k
                    ]
                combined_by_t2[t2_val] = combined_block
            # if not keep.any() → t2 dropped, filter never sees it

        return combined_by_t2

    def _build_indices(self) -> dict:
        """
        Rebuild the ``indices_by_t2`` lookup from the current dataset list.

        Should be called after any in-place modification of dataset contents
        (e.g., after measurement editing) to keep the internal index consistent.

        Returns
        -------
        dict
            Updated ``indices_by_t2`` mapping (same structure as described in
            :meth:`combine_indices_by_t2`).

        Raises
        ------
        ValueError
            If no datasets have been provided.
        """
        if not self._datasets:
            raise ValueError("No datasets provided.")
        self._indices_by_t2 = self._combine_indices_by_t2(self._datasets)
        return self._indices_by_t2

    def get_combined(self) -> dict:
        """
        Build and return the full combined measurement structure for all epochs.

        Iterates over every unique ``t2`` value and assembles numeric arrays,
        stacked partial-derivative matrices, and list-type metadata fields.
        Measurements flagged as outliers are dropped before returning.

        Returns
        -------
        dict
            Keys are unique ``t2`` float values sorted ascending.  Each value
            is a dict with the fields documented in
            :meth:`create_combined_from_indices`:
            ``t1``, ``t2``, ``t3``, ``observed``, ``computed``, ``residuals``,
            ``sigma``, ``outlier_flag``, ``partials``, ``instruments``,
            ``measurement_types``, ``dataset_names``, ``dataset_refs``.

        Raises
        ------
        RuntimeError
            If :meth:`build_indices` has not been called yet.
        """
        if self._indices_by_t2 is None:
            raise RuntimeError("Indices not built yet. Call build_indices() first.")
        return self._create_combined_from_indices(self._indices_by_t2)

    def get_combined_for_t2(self, t2_val: float) -> dict:
        """
        Build and return the combined measurement block for a single epoch.

        Parameters
        ----------
        t2_val : float
            Spacecraft receive time for which measurements are requested.

        Returns
        -------
        dict
            Single-key dict ``{t2_val: block}``, where ``block`` has the same
            field structure as described in :meth:`get_combined`.

        Raises
        ------
        RuntimeError
            If indices have not been built yet.
        KeyError
            If ``t2_val`` is not present in the index.
        """
        if self._indices_by_t2 is None:
            raise RuntimeError("Indices not built yet. Call build_indices() first.")
        if t2_val not in self._indices_by_t2:
            raise KeyError(f"t2={t2_val} not found in the datasets.")
        return self._create_combined_from_indices((t2_val, self._indices_by_t2[t2_val]))

    def get_spacecraft_times(self) -> "np.ndarray":
        """
        Return sorted unique spacecraft receive times with at least one valid measurement.

        A ``t2`` value is included only when at least one of its measurements has
        ``outlier_flag == False``.  Fully rejected epochs are silently omitted.

        Returns
        -------
        np.ndarray
            1-D array of ``float64`` spacecraft receive times sorted ascending.

        Raises
        ------
        RuntimeError
            If indices have not been built yet.
        """
        if self._indices_by_t2 is None:
            raise RuntimeError("Indices not built yet. Call build_indices() first.")

        valid_times = []
        for t2_val, blocks in self._indices_by_t2.items():
            for block in blocks:
                dset = block["dataset"]
                idxs = block["indices"]
                if isinstance(idxs, (np.integer, int)):
                    idxs = [int(idxs)]
                flags = np.array(dset.data["outlier_flag"])[idxs].astype(bool)
                if not flags.all():
                    valid_times.append(t2_val)
                    break  # one non-rejected measurement is enough

        return np.array(sorted(valid_times), dtype=np.float64)

    # Combine and create time mask
    @staticmethod
    def _combine_time_vectors(measurement_times_list: list) -> tuple:
        """
        Merge multiple per-type time arrays into a single sorted grid with an availability mask.

        Parameters
        ----------
        measurement_times_list : list of np.ndarray
            Each element is a 1-D array of epoch values for one measurement type.

        Returns
        -------
        combined_times : np.ndarray
            Sorted 1-D array of unique epochs across all input arrays.
        measurement_mask : np.ndarray
            Boolean/integer array of shape ``(len(combined_times), len(measurement_times_list))``.
            Entry ``[i, j]`` is 1 if epoch ``combined_times[i]`` is present in
            ``measurement_times_list[j]``, 0 otherwise.
        """
        # Combine all time arrays and remove duplicates
        combined_times = np.unique(np.concatenate(measurement_times_list))

        # Create a mask for each measurement
        masks = [
            np.isin(combined_times, times).astype(int)
            for times in measurement_times_list
        ]

        # Stack the masks to create a 2D array (time x measurement)
        measurement_mask = np.vstack(masks).T

        return combined_times, measurement_mask

    def _save_datasets(self, save_path=None, save_name=None):
        """
        Save each dataset's measurement data (including outlier_flag) to a .npz file.
        If a file with the same name already exists, appends an iteration suffix
        (_it1, _it2, ...) to avoid overwriting.

        Parameters
        ----------
        save_path : str or None
            Directory where files are saved. Defaults to './filter_data/'.
        save_name : str or None
            Base name for the output files. If None, uses the dataset set_name.
            If provided, files are named '{save_name}_{set_name}.npz'.
        """
        save_dir = save_path or os.path.join(".", "filter_data")
        os.makedirs(save_dir, exist_ok=True)

        for dset in self._datasets:
            d = dset.data
            clean_set_name = dset.set_name.replace(" ", "_")
            base = f"{save_name}_{clean_set_name}" if save_name else clean_set_name

            # Find a non-colliding filename by appending _it1, _it2, ...
            file_name = os.path.join(save_dir, f"{base}.npz")
            if os.path.isfile(file_name):
                it = 1
                while os.path.isfile(os.path.join(save_dir, f"{base}_it{it}.npz")):
                    it += 1
                file_name = os.path.join(save_dir, f"{base}_it{it}.npz")

            np.savez(
                file_name,
                t1=np.array(d["t1"]),
                t2=np.array(d["t2"]),
                t3=np.array(d["t3"]),
                computed=np.array(d["computed"]),
                observed=np.array(d["observed"]),
                residuals=np.array(d["residuals"]),
                sigma=np.array(d["sigma"]),
                outlier_flag=np.array(d["outlier_flag"]),
                measurement_type=np.array([dset.measurement_type]),
                set_name=np.array([dset.set_name]),
            )
            n_rejected = int(np.array(d["outlier_flag"]).sum())
            n_total = len(np.array(d["outlier_flag"]))
            print(
                f"[FilterDataManager] Saved '{dset.set_name}' → {file_name}  "
                f"({n_rejected}/{n_total} rejected)"
            )
