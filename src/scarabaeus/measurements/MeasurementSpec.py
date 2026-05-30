# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional, Union


# --------------------#
#  Class Definition  #
# --------------------#
@dataclass(frozen=True)
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
        Optional label associated with the measurement file on disk.
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
    _items: Optional[tuple] = None  # container mode

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
        return cls(_items=tuple(specs))

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
    # Conversion
    # ----------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the MeasurementSpec instance to a dictionary representation.

        Returns a dictionary containing the model, dataset_name, and either observed_meas
        or epochs (but not both). Optionally includes file_label if it is set.

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
            raise TypeError(...)

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
