# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import numpy as np
import os
from typing import Literal
from scarabaeus import MeasurementEditing
from scarabaeus.utils.Serialization import dump, load, format_suffix


class _LoadedFilterDataSet:
    """Read-only stand-in for a :class:`MeasurementDataSet`, reconstructed from a
    file written by :meth:`FilterDataManager._save_datasets`.

    Exposes only the attributes read back downstream — ``set_name``,
    ``measurement_type``, ``frame`` (name), ``units`` (dict), and ``data`` (the
    residual / measurement arrays). It deliberately cannot regenerate predicted
    measurements: the saved payload is a results snapshot and does not carry the
    instrument / target a live ``MeasurementDataSet`` needs, so a manager built
    from these is for inspection and plotting only, not for re-running the filter.

    Parameters
    ----------
    payload : dict
        One dataset as returned by :meth:`FilterDataManager.load_dataset`.
    """

    _DATA_KEYS = (
        "t1", "t2", "t3", "computed", "observed", "residuals", "sigma", "outlier_flag",
    )

    def __init__(self, payload: dict):
        self.set_name = payload.get("set_name")
        self.measurement_type = payload.get("measurement_type")
        self.frame = payload.get("frame")  # frame name as saved (str)
        self.units = payload.get("units")  # units dict as saved
        self.data = {k: payload[k] for k in self._DATA_KEYS if k in payload}

    def __repr__(self) -> str:
        n = len(self.data.get("t2", []))
        return f"_LoadedFilterDataSet(set_name={self.set_name!r}, n={n})"


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
        ``'lasso_all'``, ``'chi2'``, ``'chi2_all'``, ``'kurtosis'``,
        ``'kurtosis_all'``, ``'date_range'``, ``'interactive_date_range'``,
        or ``None``.

        ``'lasso'``, ``'chi2'`` and ``'kurtosis'`` open one figure per dataset
        in sequence, each blocking until it is closed. ``'lasso_all'``,
        ``'chi2_all'`` and ``'kurtosis_all'`` open every dataset's figure at
        once, via :meth:`~scarabaeus.MeasurementEditing.lasso_editor_all`,
        :meth:`~scarabaeus.MeasurementEditing.chi_squared_filter_all` and
        :meth:`~scarabaeus.MeasurementEditing.kurtosis_filter_all`. Those
        are the usable options on remote/browser backends,
        where closing a plot tab does not advance a sequential loop.
    editing_datasets : None or str or list of str
        Which datasets to edit. ``None`` = all, a string = one by name,
        a list of strings = a specific subset.
    editing_kwargs : dict or None
        Extra keyword arguments forwarded to the editing method.
    save_path : str or None
        Directory path to save combined per-dataset JSON files.
    save_name : str or None
        Optional name prefix included in saved file names.

    See Also
    --------
    scarabaeus.MeasurementEditing : Provides the lasso/chi2/kurtosis/date-range editing methods.
    scarabaeus.FilterOD : Base filter that consumes the combined indexed structure.

    Notes
    -----
    Two-way radiometric time-tag convention used throughout this class:

    * ``t1`` — ground station **transmit** epoch (uplink signal sent).
    * ``t2`` — **spacecraft time** (signal reflection / turnaround epoch).
    * ``t3`` — ground station **receive** epoch (downlink signal received).

    Datasets are indexed by spacecraft epoch ``t2``; all datasets must use the
    same time-tag convention.  Nested list inputs are flattened one level
    automatically.
    """

    # Measurements whose spacecraft times t2 differ by no more than this many
    # seconds are processed at one filter epoch. Separate models solve light
    # time separately, so measurements received together agree on t2 only to
    # rounding (about 1e-7 s at current ET); 1 ms is far above that and far
    # below the spacing of any tracking data.
    T2_MERGE_TOL = 1e-3

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
        save_format: Literal["json", "pickle"] = "json",
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

            # The '_all' methods take the whole name list in one call so every
            # figure opens together, so they sit outside the per-dataset loop.
            if editing_method == "lasso_all":
                editor.lasso_editor_all(target_names, **kw)
            elif editing_method == "chi2_all":
                editor.chi_squared_filter_all(target_names, **kw)
            elif editing_method == "kurtosis_all":
                editor.kurtosis_filter_all(target_names, **kw)
            else:
                for name in target_names:
                    if editing_method == "lasso":
                        editor.lasso_editor(name, **kw)
                    elif editing_method == "chi2":
                        editor.chi_squared_filter(dataset_name=name, **kw)
                    elif editing_method == "kurtosis":
                        editor.kurtosis_filter(dataset_name=name, **kw)
                    elif editing_method == "date_range":
                        editor.date_range_filter(dataset_name=name, **kw)
                    elif editing_method == "interactive_date_range":
                        editor.interactive_date_range(dataset_name=name, **kw)
                    else:
                        raise ValueError(
                            f"Unknown editing_method '{editing_method}'. "
                            "Choose: 'lasso', 'lasso_all', 'chi2', 'chi2_all', "
                            "'kurtosis', 'kurtosis_all', 'date_range', "
                            "'interactive_date_range'."
                        )

            self._datasets = editor.get_datasets()
            self._indices_by_t2 = self._combine_indices_by_t2(self._datasets)
            self._datasets_names = [d.set_name for d in self._datasets]

            # Optional: save combined data structure to disk for inspection
            if save_path is not None and self._datasets:
                self._save_datasets(save_path, save_name, fmt=save_format)

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

        Groups all measurement indices from every dataset by spacecraft time
        ``t2``.  The result is the primary lookup structure used by the filter to
        retrieve measurements epoch by epoch.  Measurements whose ``t2`` lie
        within :attr:`T2_MERGE_TOL` of the earliest ``t2`` of a group share one
        filter epoch, keyed by that earliest ``t2``; each keeps its own ``t2`` in
        the combined block.

        Parameters
        ----------
        datasets : list of MeasurementDataSet
            Input datasets to index.

        Returns
        -------
        dict
            Keys are the ``t2`` float value of each group, sorted in ascending
            order.  Each value is a list of dicts with keys:

            * ``'dataset'`` — reference to the originating :class:`~scarabaeus.MeasurementDataSet`.
            * ``'indices'`` — ``np.ndarray`` of integer row indices within that dataset
              that fall in this group.
        """
        # Measurements flagged as outliers are excluded here, which is the
        # single point where measurements enter the filter. Gating here (as
        # opposed to when the blocks are combined) means a t2 whose
        # measurements are all flagged never becomes a key, so it does not
        # appear in get_spacecraft_times() and is never requested from
        # get_combined_for_t2().
        t_parts, d_parts, i_parts = [], [], []
        for d, dset in enumerate(datasets):
            t2 = np.asarray(dset.data["t2"], dtype=float).ravel()
            flags = np.asarray(dset.data.get("outlier_flag", np.zeros(len(t2)))).ravel()
            keep = np.nonzero(flags == 0)[0]
            t_parts.append(t2[keep])
            d_parts.append(np.full(keep.size, d, dtype=int))
            i_parts.append(keep)
        if not t_parts:
            return {}
        t_all = np.concatenate(t_parts)
        d_all = np.concatenate(d_parts)
        i_all = np.concatenate(i_parts)
        if t_all.size == 0:
            return {}

        # Group by t2 within T2_MERGE_TOL rather than by exact float equality.
        # Each model solves its own light time for t2, so two measurements
        # received at the same t3 (a range point and a Doppler point, say) agree
        # only to rounding, and whether they round to the same float can change
        # from one iteration to the next as the reference trajectory moves. With
        # exact keys the number of filter epochs then changes between iterations.
        # A group spans at most T2_MERGE_TOL from its first (earliest) t2, which
        # is the key, so it cannot chain across neighbouring epochs.
        order = np.argsort(t_all, kind="stable")
        t_sorted = t_all[order]
        starts = [0]
        for k in range(1, t_sorted.size):
            if t_sorted[k] - t_sorted[starts[-1]] > cls.T2_MERGE_TOL:
                starts.append(k)
        bounds = starts + [t_sorted.size]

        combined_by_t2 = {}
        for a, b in zip(bounds[:-1], bounds[1:]):
            members = order[a:b]
            blocks = []
            # Datasets in their input order, rows ascending, as before
            for d in np.unique(d_all[members]):
                idxs = np.sort(i_all[members][d_all[members] == d])
                blocks.append({"dataset": datasets[d], "indices": idxs})
            combined_by_t2[float(t_sorted[a])] = blocks

        # Already in ascending t2 order
        return combined_by_t2

    @staticmethod
    def _scalar_column(dset, key, idxs) -> "np.ndarray":
        """
        Select rows ``idxs`` from a per-measurement field and return it flat (1-D).

        Datasets are expected to hold exactly one value per measurement in every
        numeric field. Some store a scalar field as an ``(N, 1)`` column vector and
        others as a flat ``(N,)`` array; both are normalised here so
        :func:`numpy.concatenate` downstream does not fail on mixed dimensionality.

        A genuine multi-component field ``(N, c)`` with ``c > 1`` -- e.g. an
        unsplit 2-component optical (RA/Dec) or multi-column radiometric track --
        cannot be row-aligned with the one-per-measurement ``outlier_flag`` and,
        after the old blanket ``reshape(-1)``, produced an opaque
        ``IndexError`` at the outlier ``keep`` step. Such tracks must be split into
        scalar datasets upstream (see
        :meth:`Measurement.generate_measurement_dataset`, which splits
        ``n_components == 2`` measurements into ``Opnav_u`` / ``Opnav_v``); reaching
        this point unsplit almost always means a multi-column observation file was
        loaded into a scalar measurement model.

        Raises
        ------
        ValueError
            If ``key`` carries more than one value per measurement.
        """
        arr = np.asarray(dset.data[key])[idxs]
        rows = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(-1, 1)
        if rows.shape[1] != 1:
            raise ValueError(
                f"Dataset '{getattr(dset, 'set_name', '?')}' field '{key}' carries "
                f"{rows.shape[1]} values per measurement (stored shape "
                f"{np.asarray(dset.data[key]).shape}), but 'outlier_flag' has one "
                f"value per measurement, so they cannot be row-aligned in the "
                f"combined block. Multi-component measurements must be split into "
                f"scalar datasets before filtering -- this usually means a "
                f"multi-column observation file was loaded into a scalar "
                f"measurement model."
            )
        return rows.reshape(-1)

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

                # Numeric arrays. Each of these is one scalar per measurement, so
                # flatten to 1-D: some datasets store a field as a column vector
                # (N, 1) while others store it flat (N,), which would otherwise make
                # np.concatenate below fail with mismatched dimensions.
                # _scalar_column also guards against multi-component fields (N, c>1)
                # sneaking in unsplit, which would silently misalign with the
                # one-per-measurement outlier_flag and blow up at the keep step.
                combined_block["t1"].append(FilterDataManager._scalar_column(dset, "t1", idxs))
                combined_block["t2"].append(FilterDataManager._scalar_column(dset, "t2", idxs))
                combined_block["t3"].append(FilterDataManager._scalar_column(dset, "t3", idxs))
                combined_block["observed"].append(
                    FilterDataManager._scalar_column(dset, "observed", idxs)
                )
                combined_block["computed"].append(
                    FilterDataManager._scalar_column(dset, "computed", idxs)
                )

                # Ensure residuals are 2D column vectors
                combined_block["residuals"].append(
                    FilterDataManager._scalar_column(dset, "residuals", idxs).reshape(-1, 1)
                )

                combined_block["sigma"].append(
                    FilterDataManager._scalar_column(dset, "sigma", idxs)
                )
                combined_block["outlier_flag"].append(
                    FilterDataManager._scalar_column(dset, "outlier_flag", idxs)
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

            # No outlier filtering here: the indices come from
            # _combine_indices_by_t2, which already left flagged rows out.
            combined_by_t2[t2_val] = combined_block

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
        ``outlier_flag == False``.  Fully rejected epochs are silently omitted;
        that gating happens when the indices are built, in
        :meth:`_combine_indices_by_t2`, so this just returns their keys.

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

        return np.array(sorted(self._indices_by_t2), dtype=np.float64)

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

    def _save_datasets(
        self, save_path=None, save_name=None, fmt: Literal["json", "pickle"] = "json"
    ):
        """
        Save each dataset's measurement data (including outlier_flag) to a JSON or
        pickle file. If a file with the same name already exists, appends an
        iteration suffix (_it1, _it2, ...) to avoid overwriting.

        Parameters
        ----------
        save_path : str or None
            Directory where files are saved. Defaults to './filter_data/'.
        save_name : str or None
            Base name for the output files. If None, uses the dataset set_name.
            If provided, files are named '{save_name}_{set_name}{ext}'.
        fmt : {"json", "pickle"}, optional
            On-disk format; selects the ``.json`` or ``.pkl`` extension. Defaults
            to ``"json"``.
        """
        save_dir = save_path or os.path.join(".", "filter_data")
        os.makedirs(save_dir, exist_ok=True)
        ext = format_suffix(fmt)

        for dset in self._datasets:
            d = dset.data
            clean_set_name = dset.set_name.replace(" ", "_")
            base = f"{save_name}_{clean_set_name}" if save_name else clean_set_name

            # Find a non-colliding filename by appending _it1, _it2, ...
            file_name = os.path.join(save_dir, f"{base}{ext}")
            if os.path.isfile(file_name):
                it = 1
                while os.path.isfile(os.path.join(save_dir, f"{base}_it{it}{ext}")):
                    it += 1
                file_name = os.path.join(save_dir, f"{base}_it{it}{ext}")

            # Record frame/units explicitly so the dataset is self-describing on
            # disk. Both are captured once when the dataset is generated (the
            # filter itself computes on raw numbers), mirroring how frame/units
            # are stored for trajectories and OD solutions.
            dset_frame = getattr(dset, "frame", None)
            dset_units = getattr(dset, "units", None)
            payload = {
                "set_name": dset.set_name,
                "measurement_type": dset.measurement_type,
                "frame": dset_frame.name if dset_frame is not None else None,
                "units": (
                    dset_units.to_dict()
                    if dset_units is not None and hasattr(dset_units, "to_dict")
                    else None
                ),
                "t1": np.asarray(d["t1"]),
                "t2": np.asarray(d["t2"]),
                "t3": np.asarray(d["t3"]),
                "computed": np.asarray(d["computed"]),
                "observed": np.asarray(d["observed"]),
                "residuals": np.asarray(d["residuals"]),
                "sigma": np.asarray(d["sigma"]),
                "outlier_flag": np.asarray(d["outlier_flag"]),
            }
            dump(payload, file_name, fmt=fmt)

            n_rejected = int(np.array(d["outlier_flag"]).sum())
            n_total = len(np.array(d["outlier_flag"]))
            print(
                f"[FilterDataManager] Saved '{dset.set_name}' → {file_name}  "
                f"({n_rejected}/{n_total} rejected)"
            )

    @classmethod
    def load_dataset(cls, file_path):
        """
        Load a dataset previously written by :meth:`_save_datasets`.

        Returns a plain dict with the same fields (arrays restored as numpy
        arrays). The format (JSON or pickle) is auto-detected from the file
        extension. Symmetric counterpart to the save path.
        """
        raw = load(file_path)
        out = dict(raw)
        for key in ("t1", "t2", "t3", "computed", "observed", "residuals", "sigma", "outlier_flag"):
            if key in out and out[key] is not None:
                out[key] = np.asarray(out[key])
        return out

    @classmethod
    def load(cls, file_paths):
        """Reconstruct a :class:`FilterDataManager` from saved dataset file(s).

        Loads one or more files written by :meth:`_save_datasets` (via
        :meth:`load_dataset`) and returns a manager populated with them, so
        callers get an object directly instead of rebuilding one from the raw
        dicts.

        Parameters
        ----------
        file_paths : str, path-like, or iterable of those
            A single saved dataset file, or several. JSON and pickle are both
            auto-detected per file.

        Returns
        -------
        FilterDataManager
            A manager holding the loaded datasets. It is **inspection-only**: the
            saved files are result snapshots (no instrument / target), so the
            datasets cannot regenerate predicted measurements or re-run the
            filter.

        See Also
        --------
        load_dataset : Low-level reader returning one dataset as a plain dict.
        """
        if isinstance(file_paths, (str, os.PathLike)):
            file_paths = [file_paths]
        datasets = [_LoadedFilterDataSet(cls.load_dataset(p)) for p in file_paths]
        return cls(datasets=datasets)
