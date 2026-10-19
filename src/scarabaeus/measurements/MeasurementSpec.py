# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, Optional, Union

import inspect
import warnings
import numpy as np

from scarabaeus.timeAndFrame.SpiceManager import SpiceManager


# --------------------#
#  Class Definition  #
# --------------------#
@dataclass
class MeasurementSpec:
    """ Specification of one or more measurement datasets for filtering.

    This class can represent either a *single* measurement specification
    (leaf mode) or a *container* of multiple ``MeasurementSpec`` objects
    (container mode).  In both cases, the object is iterable and yields
    dictionaries compatible with the Scarabaeus filtering API.

    Attributes
    ----------
    model : Any
        Measurement model instance (e.g., RangeIdeal, RangeRateIdeal, AngularIdeal).
    observed_meas : Any
        Observed measurement quantities as returned by the measurement model.
    epochs : Any
        Epochs corresponding to the observed measurements (used in covariance
        analysis where actual observations are not required).
    dataset_name : str
        Human-readable name of the dataset (used for labeling and plotting).
    file_label : str, optional
        Path or label of the file the observations were read from.
    _items : tuple of MeasurementSpec, optional
        Internal container holding multiple ``MeasurementSpec`` objects.
        If set, this instance acts as a container and yields each contained
        spec when iterated.

    Notes
    -----
    A ``MeasurementSpec`` operates in two modes:

    *Leaf mode* — represents one dataset; iteration yields a single
    dictionary ``{"model": ..., "observed_meas": ..., "dataset_name": ...}``.

    *Container mode* — wraps multiple leaf specs; iteration yields one
    dictionary per contained spec.

    Both modes are accepted wherever the filtering pipeline expects a list
    of measurement dictionaries, providing backward compatibility with
    the legacy ``list[dict]`` interface.

    Examples
    --------
    Single measurement dataset::

        spec = MeasurementSpec(
            model=RangeIdeal(...),
            observed_meas=y_range,
            dataset_name="DSN Range",
        )

    Multiple datasets (container mode)::

        spec = MeasurementSpec.many(
            MeasurementSpec(model=range_model, ...),
            MeasurementSpec(model=rangerate_model, ...),
        )

    Legacy dict interface::

        spec = MeasurementSpec.from_dict(meas_dict)
        spec = MeasurementSpec.from_dicts(list_of_dicts)
    """

    model: Any = None
    observed_meas: Any = None
    epochs: Any = None
    dataset_name: Optional[str] = None
    file_label: Optional[str] = None
    dataset: Any = None             # MeasurementDataSet, set by compute()
    provenance: Optional[dict] = None  # what compute() actually resolved against
    _items: Optional[tuple] = None  # container mode

    # ----------------------------
    # Exceptions
    # ----------------------------
    class RefitSigmaUnitsError(ValueError):
        """
        A refitted sigma cannot be expressed in the units asked for.

        The refit is in the units of the dataset's residuals, and the units
        asked for (``sigma_units``, else those of the model's current sigma)
        are of another dimension.
        """

    # ----------------------------
    # Constructors
    # ----------------------------
    @classmethod
    def many(cls, *specs: "MeasurementSpec") -> "MeasurementSpec":
        """
        Create a container spec holding multiple MeasurementSpec objects.

        Parameters
        ----------
        *specs : MeasurementSpec
            One or more individual ``MeasurementSpec`` instances to group.

        Returns
        -------
        MeasurementSpec
            A container instance whose iteration yields one dict per spec.
        """
        # Flatten any containers that are passed in. Without this, nesting a
        # container silently produced a spec whose leaves the filter could not
        # reach, and callers worked around it by splatting the private
        # ``._items`` themselves.
        flat = []
        for spec in specs:
            if getattr(spec, "_items", None) is not None:
                flat.extend(spec._items)
            else:
                flat.append(spec)
        return cls(_items=tuple(flat))

    @classmethod
    def from_models(cls, models, observed, names=None) -> "MeasurementSpec":
        """
        Build a container MeasurementSpec from measurement models and their
        observed quantities.

        Parameters
        ----------
        models : Sequence
            Measurement model instances (e.g. SequentialRangingReal, DopplerReal).
            Each model already carries its instrument, sigma, bias, media
            corrections and measurement type.
        observed : Sequence
            Observed quantities for each model, as returned by
            ``model.observed_measurements(...)`` (an ObsData).
        names : Sequence[str], optional
            Dataset name for each model. Defaults to each model's ``name``.

        Returns
        -------
        MeasurementSpec
            Container spec, ready to hand to a filter.
        """
        if names is None:
            names = [m.name for m in models]

        specs = [
            cls(model=m, observed_meas=o, dataset_name=n)
            for m, o, n in zip(models, observed, names, strict=True)
        ]
        return cls.many(*specs)

    @classmethod
    def build(cls, streams, config=None, target=None) -> "MeasurementSpec":
        """
        Build a filter-ready spec from a list of measurement streams.

        Each stream is normally a ``(file, meas_type, station)`` tuple. The
        measurement type supplies the model class, the data block to read and
        the units; the config supplies what is shared across streams. Anything
        needing more control can be given as a MeasurementStream instead.

        No trajectory is needed here: the observations are read, the models
        built and the media corrections attached. Computing measurements and
        residuals is a separate step, so the spec is usable straight away for
        a run with no outlier rejection.

        Parameters
        ----------
        streams : sequence
            (file, meas_type, station) tuples, or MeasurementStream objects.
        config : MeasurementConfig, optional
            Settings shared across streams.
        target : Body, optional
            Not used, and deliberately kept. ``build`` reads observations and
            assembles models, none of which needs a target; computing the
            measurements does, and that is :meth:`compute`. Accepting it here
            lets a caller hand the same arguments to both steps rather than
            remembering which one takes what.

        Returns
        -------
        MeasurementSpec
            Container spec holding one entry per stream.

        Examples
        --------
        >>> spec = MeasurementSpec.build(
        ...     [(file1, "SeqRange", DSS_25),
        ...      (file2, "Doppler",  DSS_25)],
        ...     config=cfg,
        ... )
        """
        from scarabaeus.measurements.MeasurementConfig import MeasurementConfig
        from scarabaeus.measurements.MeasurementRegistry import MeasurementRegistry
        from scarabaeus.measurements.MeasurementStream import MeasurementStream
        from scarabaeus.measurements.MediaCorrections import MediaCorrections

        config = config or MeasurementConfig()
        items = [MeasurementStream.coerce(item) for item in streams]

        for item in items:
            MeasurementRegistry.get(item.meas_type)          # fail early, clearly
        names = MeasurementRegistry.assign_dataset_names(
            items, MeasurementRegistry.all()
        )

        specs = []
        extras_used = set()
        for item, name in zip(items, names, strict=True):
            type_spec = MeasurementRegistry.get(item.meas_type)

            sigma = item.sigma
            if sigma is None:
                sigma = config.get_sigma(item.meas_type, name)
            bias = item.bias
            if bias is None:
                bias = config.get_bias(item.meas_type, name)

            # Filter the optional keyword arguments by what this model actually
            # accepts. Only SequentialRangingReal and DopplerReal take both
            # meas_bias= in the constructor and sc_transponder_delay_RU= in the
            # read: the Ideal models override observed_measurements without
            # that keyword, and CentroidingIdeal has no meas_bias. The same
            # signature filtering is used by MeasurementScheduler.
            ctor_kwargs = {"sigma": sigma, "meas_bias": bias, **config.extra}
            ctor_kwargs = MeasurementSpec._accepted_kwargs(
                type_spec.model_class.__init__, ctor_kwargs
            )
            # Which of config.extra this model could take, so that a key no
            # model takes can be reported once at the end rather than doing
            # nothing in silence.
            extras_used.update(k for k in config.extra if k in ctor_kwargs)
            model = type_spec.model_class(name, item.station, **ctor_kwargs)

            read_candidates = {
                "meas_name": type_spec.meas_name,
                "units": type_spec.units,
                "sc_transponder_delay_RU": config.sc_transponder_delay_RU,
            }
            # config.frame was accepted and then never looked at, so a scenario
            # set up in a non-J2000 frame was read in J2000 anyway. Only passed
            # when set, so None keeps each model on its own default.
            if config.frame is not None:
                read_candidates["frame"] = config.frame
            read_kwargs = MeasurementSpec._accepted_kwargs(
                model.observed_measurements, read_candidates
            )
            observed = model.observed_measurements(file_name=item.file, **read_kwargs)

            # No aux copy here: observed_measurements installs the aux data on
            # the model as it reads it, so every way of assembling a spec gets
            # it, not just this one. This line was the only thing connecting
            # ObsData.aux_dict to the computed_measurements_dict the models
            # actually read.

            media = item.media
            if media is None and config.media_folder is not None:
                media = MediaCorrections.for_pass(
                    instrument=item.station,
                    start_epoch_et=observed.meas_time_et.times.values[0],
                    media_folder=config.media_folder,
                    tropo_seasonal_file_path=config.tropo_seasonal_file_path,
                )
            if media is not None:
                model.media_corrections = media

            specs.append(cls(model=model, observed_meas=observed,
                             dataset_name=name, file_label=item.file))

        # Signature filtering is what lets one build() serve every registered
        # type, but it also means an extra no model accepts is dropped without
        # a word -- a typo, or a key meant for a different scenario, silently
        # doing nothing. Say so.
        unused = sorted(set(config.extra) - extras_used)
        if unused:
            warnings.warn(
                "MeasurementConfig.extra " + ", ".join(f"'{k}'" for k in unused)
                + (" was" if len(unused) == 1 else " were")
                + " not accepted by any model in this spec, so "
                + ("it had" if len(unused) == 1 else "they had")
                + " no effect. Check the spelling against the constructor of "
                "the model the key was meant for.",
                RuntimeWarning,
                stacklevel=2,
            )

        return cls.many(*specs)

    def epoch_span(self, pad=0.0, n: int = None):
        """
        Epoch coverage of every stream, optionally padded and resampled.

        Replaces the min/max loop each analysis script wrote by hand to work
        out what span to propagate over.

        Parameters
        ----------
        pad : ArrayWUnits or float, optional
            Time to extend before the first and after the last observation.
            An ``ArrayWUnits`` must have dimensions of time; a bare number is
            taken as seconds.
        n : int, optional
            If given, return this many evenly spaced epochs across the span.
            Otherwise return the sorted unique observation epochs, or just the
            padded end points when ``pad`` is non-zero.

        Returns
        -------
        EpochArray
            Expressed in the time system the observations share, or in TDB
            when the streams use different systems.
        """
        from scarabaeus.timeAndFrame.EpochArray import EpochArray
        from scarabaeus.units.ArrayWUnits import ArrayWUnits
        from scarabaeus.units.Units import Units

        epochs = [leaf.observed_meas[0] for leaf in self.leaves]

        # Work on .et, which is always TDB seconds past J2000, not .times.values,
        # which is in each epoch's own time system -- off by up to about a minute
        # for epochs built in TT, TAI or GPS, as an Ideal model scheduled in
        # those can be.
        et = np.sort(np.unique(np.concatenate(
            [np.asarray(e.et, dtype=float).ravel() for e in epochs]
        )))

        if isinstance(pad, ArrayWUnits):
            if pad.units.dimensions.name != "Time":
                raise ValueError(
                    f"epoch_span pad must have dimensions of time, got {pad.units}."
                )
            pad_sec = float(np.asarray(pad.convert_to(Units.get_units("sec")).values))
        else:
            pad_sec = float(pad)

        systems = {e.system for e in epochs}
        sys = systems.pop() if len(systems) == 1 else "TDB"

        if n is None and pad_sec == 0.0:
            span = EpochArray(et, sys="TDB")
        elif n is None:
            span = EpochArray(np.array([et[0] - pad_sec, et[-1] + pad_sec]), sys="TDB")
        else:
            span = EpochArray.interval(
                et[0] - pad_sec, et[-1] + pad_sec, n_epochs=int(n), sys="TDB"
            )
        # Same instants, expressed in the observations' own system.
        return span.to(sys=sys)

    @classmethod
    def from_dict(cls, d):
        """
        Create a MeasurementSpec from a dictionary or list of dictionaries.

        Parameters
        ----------
        d : dict or list or tuple
            A dictionary with measurement specification data, or a list/tuple
            of such dictionaries to create a container spec.

        Returns
        -------
        MeasurementSpec
            A single spec if *d* is a dict; a container spec if *d* is a
            list or tuple.

        Raises
        ------
        KeyError
            If required keys ``"model"`` or ``"dataset_name"`` are missing.

        Examples
        --------
        >>> spec = MeasurementSpec.from_dict({
        ...     "model": range_model,
        ...     "observed_meas": y_range,
        ...     "dataset_name": "DSN Range",
        ...     "file_label": "dsn_range.json",
        ... })
        """
        if isinstance(d, (list, tuple)):
            return cls.from_dicts(d)

        return cls(
            model=d["model"],
            observed_meas=d.get("observed_meas"),
            epochs=d.get("epochs"),
            dataset_name=d["dataset_name"],
            file_label=d.get("file_label"),
        )

    @classmethod
    def from_dicts(cls, dicts: Iterable[Dict[str, Any]]) -> "MeasurementSpec":
        """
        Construct a container spec from an iterable of measurement dictionaries.

        Parameters
        ----------
        dicts : iterable of dict
            Each dict must satisfy the same requirements as the argument to
            :meth:`from_dict`.

        Returns
        -------
        MeasurementSpec
            A container spec equivalent to ``MeasurementSpec.many(*specs)``.
        """
        return cls.many(*(cls.from_dict(d) for d in dicts))

    # ----------------------------
    # Computation and inspection
    # ----------------------------
    @property
    def leaves(self) -> tuple:
        """The individual specs, whether this is a leaf or a container."""
        return self._items if self._items is not None else (self,)

    def compute(self, target, frame=None, reference_state=None, **kwargs):
        """
        Compute measurements and residuals for every stream, and keep the
        resulting MeasurementDataSet on the spec.

        This is for inspecting residuals (and rejecting outliers) before the
        spec is handed to a filter. It does NOT pre-empt the filter's own
        computation: a filter computes against its own reference trajectory,
        which is generally not the one used here and which changes on every
        iteration. See the note in FilterOD._initialize.

        Parameters
        ----------
        target : Body
            The target whose trajectory the measurements are computed against.
        frame : Frame, optional
            Frame for the computation. Defaults to the model's own default.
        reference_state : StateArray, optional
            Reference state to push into each model before computing.
        **kwargs
            Passed through to Measurement.generate_measurement_dataset.

        Returns
        -------
        MeasurementSpec
            self, so this can be chained.
        """
        for leaf in self.leaves:
            if leaf.model is None:
                raise ValueError("Cannot compute a MeasurementSpec with no model.")
            if reference_state is not None:
                leaf.model.update_reference_state(reference_state)

            if frame is not None:
                kwargs = {**kwargs, "frame": frame}

            leaf.dataset = leaf.model.generate_measurement_dataset(
                leaf.dataset_name,
                target=target,
                observed_meas=leaf.observed_meas,
                epochs=leaf.epochs,
                **kwargs,
            )
            leaf.provenance = self._capture_provenance(
                target, self._leaf_epochs(leaf)
            )
        return self

    @staticmethod
    def _leaf_epochs(leaf):
        """Return the epochs a leaf was computed at.

        Parameters
        ----------
        leaf : MeasurementSpec
            A single-stream spec.

        Returns
        -------
        numpy.ndarray or None
            The epochs of its observations, else of its ``epochs``, in TDB seconds
            past J2000; ``None`` if neither can be read.
        """
        try:
            if leaf.observed_meas is not None:
                return np.asarray(leaf.observed_meas[0].times.values, dtype=float).ravel()
            if leaf.epochs is not None:
                return np.asarray(getattr(leaf.epochs, "values", leaf.epochs), dtype=float).ravel()
        except (AttributeError, TypeError, ValueError):
            pass
        return None

    @staticmethod
    def _capture_provenance(target, epochs=None) -> dict:
        """
        Record what a computation was resolved against.

        A body's ephemeris comes from the global SPICE kernel pool, not from
        anything at the call site, so the same ``compute(target=...)`` call can
        resolve differently depending on what has been furnished since. Without
        this record, which trajectory produced a given residual is unrecoverable.

        Parameters
        ----------
        target : Body or Spacecraft
            The target of the computation.
        epochs : array_like of float, optional
            Epochs of the computation, in TDB seconds past J2000. With them only
            the SPKs that resolve the body at those epochs are recorded; the
            last-loaded SPK is not enough, because it resolves only where its
            coverage reaches and an earlier one still resolves the rest. Defaults
            to ``None``.

        Returns
        -------
        dict
            ``target_name``, ``target_spice_id``, ``spk_files`` (the SPKs used),
            ``trajectory`` (their base names joined by ``" + "``, or ``None``) and
            ``computed_at`` (the UTC time of the call, in ISO format).
        """
        spice_id = getattr(target, "spice_id", None)
        kernels = []
        if spice_id is not None:
            try:
                kernels = SpiceManager.spk_files_for_body(spice_id, epochs=epochs)
            except Exception:
                kernels = []

        return {
            "target_name": getattr(target, "name", None),
            "target_spice_id": spice_id,
            "spk_files": kernels,
            # More than one file only when the epochs straddle coverages
            "trajectory": " + ".join(os.path.basename(k) for k in kernels) or None,
            "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _require_computed(self):
        """
        Check that every stream has been computed.

        Raises
        ------
        RuntimeError
            If :meth:`compute` has not been called for one or more streams.
        """
        missing = [l.dataset_name for l in self.leaves if l.dataset is None]
        if missing:
            raise RuntimeError(
                f"compute() has not been called for: {missing}. "
                "Call spec.compute(target=...) before inspecting residuals."
            )

    def dataset_names(self) -> list:
        """Return the dataset names.

        Returns
        -------
        list of str
            One name per stream, in stream order.
        """
        return [leaf.dataset_name for leaf in self.leaves]

    @staticmethod
    def _leaf_datasets(leaf) -> list:
        """Return the datasets :meth:`compute` built for a leaf.

        A two-component model (AngularIdeal, CentroidingIdeal, ...) builds one
        scalar dataset per component, named ``"<dataset_name>_<label>"``; every
        other model builds one.

        Parameters
        ----------
        leaf : MeasurementSpec
            A single-stream spec.

        Returns
        -------
        list of MeasurementDataSet
            Empty before :meth:`compute`.
        """
        if leaf.dataset is None:
            return []
        if isinstance(leaf.dataset, (list, tuple)):
            return list(leaf.dataset)
        return [leaf.dataset]

    @staticmethod
    def _named_datasets(leaf) -> list:
        """Return the datasets of a leaf with the names the spec reports them by.

        Parameters
        ----------
        leaf : MeasurementSpec
            A single-stream spec.

        Returns
        -------
        list of tuple
            ``(name, dataset)`` pairs: the leaf's ``dataset_name`` for a model
            that builds one dataset, each component's ``set_name`` for a
            two-component one.
        """
        datasets = MeasurementSpec._leaf_datasets(leaf)
        if len(datasets) == 1:
            return [(leaf.dataset_name, datasets[0])]
        return [(ds.set_name, ds) for ds in datasets]

    def _editor_names(self, dataset_name) -> list:
        """Translate a dataset selector into the names MeasurementEditing knows.

        The editor holds the computed datasets, so a two-component stream is
        two datasets to it. A stream's name selects all of its datasets; a
        component's name selects that one.

        Parameters
        ----------
        dataset_name : str or sequence of str
            Names of streams or of their component datasets.

        Returns
        -------
        list of str
            Dataset names, without repeats. A name that matches no stream is
            passed through, for the editor to report.
        """
        selected = [dataset_name] if isinstance(dataset_name, str) else list(dataset_name)
        names = []
        for name in selected:
            expanded = [
                ds.set_name
                for leaf in self.leaves if leaf.dataset_name == name
                for ds in self._leaf_datasets(leaf)
            ] or [name]
            names.extend(n for n in expanded if n not in names)
        return names

    def residuals(self) -> Dict[str, Any]:
        """Return the residuals of each dataset, as computed by :meth:`compute`.

        Returns
        -------
        dict
            ``{dataset_name: numpy.ndarray}`` of the residuals. A two-component
            stream contributes one entry per component, keyed by the
            component's dataset name (``"<dataset_name>_<label>"``).

        Raises
        ------
        RuntimeError
            If :meth:`compute` has not been called for one or more streams.
        """
        self._require_computed()
        return {name: np.asarray(ds.data["residuals"]).ravel()
                for l in self.leaves for name, ds in self._named_datasets(l)}

    def outlier_flags(self) -> Dict[str, Any]:
        """Return the current outlier flags of each dataset.

        Returns
        -------
        dict
            ``{dataset_name: numpy.ndarray}`` of the flags of the source
            observations, including those read from the file.
        """
        return {l.dataset_name: np.asarray(l.observed_meas[3]).ravel()
                for l in self.leaves}

    def sigmas(self) -> Dict[str, Any]:
        """Return the current sigma of each dataset.

        Returns
        -------
        dict
            ``{dataset_name: sigma}`` with the sigma of each model.
        """
        return {l.dataset_name: l.model.sigma for l in self.leaves}

    def summary(self) -> str:
        """Return a one-line summary per stream.

        Returns
        -------
        str
            Per stream, the number of observations and of flagged ones; for a
            computed stream also the RMS of the kept residuals and the target and
            trajectory it was computed against. A computed two-component stream
            has one line per component.
        """
        lines = []
        for leaf in self.leaves:
            n_flag = int(np.sum(np.asarray(leaf.observed_meas[3])))
            n = len(np.asarray(leaf.observed_meas[3]).ravel())
            if leaf.dataset is None:
                lines.append(f"{leaf.dataset_name}: {n} obs, {n_flag} flagged "
                             f"(not computed)")
                continue
            prov = leaf.provenance or {}
            against = (f" vs {prov.get('target_name')}"
                       f"[{prov.get('target_spice_id')}]"
                       f" via {prov.get('trajectory')}") if prov else ""
            for name, ds in self._named_datasets(leaf):
                res = np.asarray(ds.data["residuals"]).ravel()
                keep = np.asarray(ds.data["outlier_flag"]).ravel() == 0
                # The computed dataset also flags measurements the model could
                # not compute, which the observed flags do not carry yet
                n_flag = int(np.sum(~keep))
                rms = float(np.sqrt(np.mean(res[keep] ** 2))) if np.any(keep) else float("nan")
                lines.append(f"{name}: {n} obs, {n_flag} flagged, "
                             f"residual RMS (kept) = {rms:.6e}{against}")
        return "\n".join(lines)

    def edit(self, method: str, sigma_units=None, dataset_name=None, **kwargs) -> Dict[str, Any]:
        """
        Run a MeasurementEditing method over this spec's computed datasets and
        keep the resulting flags.

        MeasurementEditing owns the editing algorithms but works on datasets,
        which a filter rebuilds on every iteration. Routing through here writes
        the flags onto the source observations as well, so they survive that
        rebuild, and applies any refitted sigma to the measurement models,
        which the editor does not own.

        Parameters
        ----------
        method : str
            Name of the MeasurementEditing method, e.g. ``"huber_mad_filter"``,
            ``"chi_squared_filter"``, ``"lasso_editor"``.
        dataset_name : str or sequence of str, optional
            Restrict the edit, and the returned flags, to these datasets. When
            omitted the edit covers every dataset in the spec. Passed through to
            the editor method as well, so the algorithm and the reported result
            agree on scope. The name of a two-component stream selects both of
            its component datasets, ``"<dataset_name>_<label>"``, which can also
            be named one at a time.
        sigma_units : Units or dict, optional
            Units for any refitted sigma, shared or keyed by dataset name. The
            refit is in the units of the residuals and is converted to these.
            Defaults to the units of each model's current sigma, else to the
            residuals' own.
        **kwargs
            Passed straight to the MeasurementEditing method.

        Returns
        -------
        dict
            Newly flagged indices per dataset name, for the datasets the edit
            covered. A two-component stream reports once, under its own name,
            since its components share their observations.

        Raises
        ------
        RefitSigmaUnitsError
            If a refitted sigma cannot be converted to the units asked for.
            Nothing is flagged then.

        Examples
        --------
        >>> spec.compute(target=orbiter, frame=frame)
        >>> spec.edit("huber_mad_filter", dataset_name="GS1 Real Range", n_sigma=5.0)
        """
        from scarabaeus.orbitDetermination.MeasurementEditing import MeasurementEditing

        self._require_computed()

        editor = MeasurementEditing([leaf.dataset for leaf in self.leaves])
        names = None
        if dataset_name is not None:
            names = self._editor_names(dataset_name)
            kwargs["dataset_name"] = names[0] if len(names) == 1 else names
        getattr(editor, method)(**kwargs)

        # The refits are checked before anything is written, so a refused call
        # leaves the flags and the sigmas as they were.
        sigmas = self._refit_sigmas(getattr(editor, "refit_sigmas", {}), sigma_units)
        editor.apply_flags()

        # Restrict the result to the datasets this call targeted. The editor is
        # built over every leaf -- the algorithms need the whole set -- and
        # editor.outlier_flags reports on all of them, so an unscoped return
        # named datasets this call did not touch. Since flag() reports only
        # *newly* flagged indices, a second scoped call then returned an empty
        # array for the first call's dataset, and a caller accumulating results
        # with dict.update() overwrote the real count with zero, as a loop
        # calling this once per dataset does.
        found = {
            name: np.nonzero(np.asarray(flags).ravel())[0]
            for name, flags in editor.outlier_flags.items()
            if names is None or name in names
        }
        return self.flag(found, sigma=sigmas or None)

    def flag_outliers(self, mean_outlier=None, sigma_outlier=None, sigma=None):
        """
        Flag outliers on each stream from its residuals, and write the flags
        back to the source observations.

        The write-back matters: a filter rebuilds its datasets on every
        iteration from the observations, so a flag left only on the dataset
        would be discarded on the next pass. Writing to ObsData.meas_outliers
        means each regenerated dataset inherits it.

        Parameters
        ----------
        mean_outlier, sigma_outlier : float, optional
            Passed to MeasurementEditing.residual_filter. Defaults to 5 sigma
            about the mean of the unflagged residuals.

            Plain numbers in residual units, i.e. the units of each stream's
            measurements (km for range, Hz for Doppler, RU for sequential
            ranging, ...), which is what the residuals column holds. The same
            numbers are applied to every stream, so explicit bounds only make
            sense when the streams share a measurement type; the defaults are
            taken from each stream's own residuals.
        sigma : dict or ArrayWUnits, optional
            Revised measurement sigma, either shared or keyed by dataset name.

        Returns
        -------
        dict
            Newly flagged indices per dataset name.
        """
        from scarabaeus.orbitDetermination.MeasurementEditing import MeasurementEditing

        self._require_computed()
        # The editor works on its own copy of the flags; flag() is what writes
        # them to the observations and mirrors them onto the datasets.
        editor = MeasurementEditing([leaf.dataset for leaf in self.leaves])
        found = editor.residual_filter(
            mean_outlier=mean_outlier, sigma_outlier=sigma_outlier
        )
        return self.flag(found, sigma=sigma)

    def flag(self, indices_by_name: Dict[str, Any], sigma=None) -> Dict[str, Any]:
        """
        Mark the given measurement indices as outliers.

        This is the write path every outlier rejection method goes through.
        Flags land on the source observations (ObsData.meas_outliers), so they
        survive the dataset rebuilds a filter performs on each iteration, and
        are mirrored onto the current dataset when one is present.

        Parameters
        ----------
        indices_by_name : dict
            Indices to flag, keyed by dataset name. Indices address the full
            measurement arrays, which never shrink, so they stay valid across
            repeated rejection passes. A two-component stream can be addressed
            by its own name or by its components' dataset names; either flags
            the observation, and so both components.
        sigma : dict or ArrayWUnits, optional
            Revised measurement sigma, shared or keyed by dataset name, in the
            form the model's ``sigma`` takes.

        Returns
        -------
        dict
            Newly flagged indices per dataset name, excluding any that were
            already flagged, for the streams ``indices_by_name`` addresses.
        """

        newly = {}
        for leaf in self.leaves:
            names = [leaf.dataset_name] + [ds.set_name for ds in self._leaf_datasets(leaf)]
            addressed = [n for n in dict.fromkeys(names) if n in indices_by_name]
            idx = np.unique(np.concatenate(
                [np.asarray(indices_by_name[n], dtype=int).ravel() for n in addressed]
                or [np.zeros(0, dtype=int)]
            ))
            flags = np.asarray(leaf.observed_meas[3]).ravel()

            already = flags[idx] != 0 if idx.size else np.zeros(0, dtype=bool)
            fresh = idx[~already] if idx.size else idx

            self._write_flags_to_observations(leaf, fresh)

            # Mirror onto the computed datasets so residuals() and summary()
            # reflect the rejection without needing a recompute. The components
            # of a two-component stream share their observations, so a flag on
            # either lands on both.
            for ds in self._leaf_datasets(leaf):
                if "outlier_flag" not in ds.data:
                    continue
                ds_flags = np.asarray(ds.data["outlier_flag"]).ravel().copy()
                if fresh.size:
                    ds_flags[fresh] = 1
                ds.data["outlier_flag"] = ds_flags

            # Mirror onto the model's aux copy as well. There are three flag
            # arrays, not two: ObsData.meas_outliers (authoritative, read by the
            # filter and the editors), the computed dataset's column (above),
            # and the model's computed_measurements_dict entry. The last is what
            # Measurement._write_aux_data persists, so leaving it stale meant a
            # spec that had been edited and was then written back out with
            # write_observed_measurements silently saved the *pre-edit* flags.
            aux_keys = getattr(leaf.model, "AUX_KEYS", None)
            aux_key = None if aux_keys is None else aux_keys.outlier_flag
            aux = getattr(leaf.model, "computed_measurements_dict", None)
            if aux_key and isinstance(aux, dict) and aux_key in aux and fresh.size:
                aux_flags = np.asarray(aux[aux_key]).ravel().copy()
                if aux_flags.size >= int(fresh.max()) + 1:
                    aux_flags[fresh] = 1
                    aux[aux_key] = aux_flags

            if sigma is not None:
                value = sigma.get(leaf.dataset_name) if isinstance(sigma, dict) else sigma
                if value is not None:
                    leaf.model.sigma = value

            if addressed:
                newly[leaf.dataset_name] = fresh
        return newly

    def _refit_sigmas(self, refits: dict, sigma_units=None) -> dict:
        """
        Turn the editor's refitted sigmas into one sigma per model.

        The editor refits each dataset on its own, from the scatter of its
        residuals, so a refit is a bare number in the units of the residuals,
        which are the dataset's. A model has one sigma, though; a two-component
        model's is a vector over its components, and a component that was not
        refitted keeps the sigma its dataset was built with.

        The sigma is expressed in ``sigma_units``, else in the units of the
        model's current sigma, else in the dataset's, converting the refit when
        those differ from the dataset's. With no units anywhere the bare value
        is kept, which :meth:`~scarabaeus.Measurement._require_sigma` tolerates;
        a dataset without units is taken to be in the units asked for.

        Parameters
        ----------
        refits : dict
            ``{dataset_name: float}``, MeasurementEditing.refit_sigmas, keyed by
            the editor's dataset names.
        sigma_units : Units or dict or None
            Units given by the caller, for every stream or as
            ``{dataset_name: Units}``.

        Returns
        -------
        dict
            ``{dataset_name: sigma}`` for :meth:`flag`, keyed by the spec's
            dataset names.

        Raises
        ------
        RefitSigmaUnitsError
            If a refit cannot be converted to the units asked for.
        """
        from scarabaeus.units.ArrayWUnits import ArrayWUnits

        sigmas = {}
        for leaf in self.leaves:
            datasets = self._leaf_datasets(leaf)
            if not any(ds.set_name in refits for ds in datasets):
                continue

            vector = [
                refits[ds.set_name] if ds.set_name in refits
                else float(np.asarray(ds.data["sigma"], dtype=float).ravel()[0])
                for ds in datasets
            ]
            value = vector[0] if len(vector) == 1 else np.array(vector)

            dataset_units = datasets[0].units
            if getattr(dataset_units, "unitless", False):
                dataset_units = None
            units = sigma_units.get(leaf.dataset_name) if isinstance(sigma_units, dict) else sigma_units
            if units is None:
                units = getattr(getattr(leaf.model, "sigma", None), "units", None)
                if getattr(units, "unitless", False):
                    units = None
            if units is None:
                units = dataset_units

            if units is None:
                sigmas[leaf.dataset_name] = value
            elif dataset_units is None:
                sigmas[leaf.dataset_name] = ArrayWUnits(value, units)
            else:
                try:
                    sigmas[leaf.dataset_name] = ArrayWUnits(value, dataset_units).convert_to(units)
                except ValueError as err:
                    raise self.RefitSigmaUnitsError(
                        f"The sigma refitted for '{leaf.dataset_name}' is in its "
                        f"residuals' units ({dataset_units}), which cannot be "
                        f"converted to {units}: {err}"
                    ) from err
        return sigmas

    @staticmethod
    def _accepted_kwargs(func, candidates: dict) -> dict:
        """
        Keep only the keyword arguments *func* actually accepts.

        Models differ in what they take: the Ideal models override
        ``observed_measurements`` and drop ``sc_transponder_delay_RU``;
        ``CentroidingIdeal`` has no ``meas_bias``. Filtering here lets one
        ``build()`` serve every registered type instead of only those shaped
        like ``SequentialRangingReal``. A function that takes ``**kwargs`` is
        given everything, since it can sort them out itself.

        Parameters
        ----------
        func : callable
            The constructor or method about to be called.
        candidates : dict
            Keyword arguments to offer it.

        Returns
        -------
        dict
            The subset *func* can accept.
        """
        try:
            params = inspect.signature(func).parameters
        except (TypeError, ValueError):
            return dict(candidates)
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(candidates)
        return {k: v for k, v in candidates.items() if k in params}

    @staticmethod
    def _write_flags_to_observations(leaf, indices):
        """Mark observations of a leaf as outliers.

        The flags of the leaf's source observations are set to 1 at
        ``indices``, in place where possible.

        Parameters
        ----------
        leaf : MeasurementSpec
            The single-stream spec.
        indices : array_like of int
            Indices of the observations to flag.
        """
        if len(indices) == 0:
            return
        flags = leaf.observed_meas[3]
        flags = np.asarray(flags)
        flags[np.asarray(indices, dtype=int)] = 1
        # ObsData is a NamedTuple, so the array is mutated in place where
        # possible and only rebuilt when it is not already an ndarray.
        if leaf.observed_meas[3] is not flags:
            leaf.observed_meas = leaf.observed_meas._replace(meas_outliers=flags)

    # ----------------------------
    # Conversion
    # ----------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the MeasurementSpec instance to a dictionary representation.

        Returns a dictionary containing the model, dataset_name, and either observed_meas
        or epochs (but not both).

        Returns
        -------
        dict
            A dictionary with keys ``"model"``, ``"dataset_name"``, either
            ``"observed_meas"`` or ``"epochs"``, and optionally ``"file_label"``.

        Raises
        ------
        TypeError
            If ``_items`` is not ``None`` (container mode cannot be converted to a single dict).
        ValueError
            If both ``observed_meas`` and ``epochs`` are ``None``, or if both are set.
        """
        if self._items is not None:
            raise TypeError(
                f"to_dict() is for a single measurement spec, but this is a "
                f"container of {len(self._items)}. Iterate it "
                "(`for d in spec:`) to get one dict per contained spec, or call "
                "to_dict() on the individual specs."
            )

        if (self.observed_meas is None) == (self.epochs is None):
            raise ValueError(
                "MeasurementSpec requires exactly one of {observed_meas, epochs}."
            )

        d = {
            "model": self.model,
            "dataset_name": self.dataset_name,
        }

        if self.observed_meas is not None:
            d["observed_meas"] = self.observed_meas
        else:
            d["epochs"] = self.epochs

        if self.file_label is not None:
            d["file_label"] = self.file_label
        return d

    def to_list(self) -> list:
        """
        Convert the MeasurementSpec object to a list of measurement dictionaries.

        Returns
        -------
        list
            A list of dictionaries, one per contained spec.
        """
        return list(self)

    # ----------------------------
    # List-like behavior (dict iterator)
    # ----------------------------
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over measurement specifications as dictionaries.

        Yields dictionaries representing measurement specifications. If the instance
        contains a single measurement, yields its dictionary representation. If the
        instance contains multiple measurements, yields the dictionary representation
        of each measurement in sequence.

        Yields
        ------
        dict
            A dictionary representation of a measurement specification.
        """
        if self._items is None:
            yield self.to_dict()
        else:
            for s in self._items:
                yield s.to_dict()

    def __len__(self) -> int:
        """
        Return the length of the measurement specification.

        Returns
        -------
        int
            ``1`` if no items are set (leaf mode), otherwise the number of contained specs.
        """
        return 1 if self._items is None else len(self._items)
