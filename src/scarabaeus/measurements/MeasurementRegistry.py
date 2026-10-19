# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Units

from dataclasses import dataclass
from typing import Any

# define units
km, sec, rad = Units.get_units(["km", "sec", "rad"])


@dataclass(frozen=True)
class MeasurementType:
    """
    What a measurement type implies.

    Attributes
    ----------
    model_class : type
        Measurement model to build.
    meas_name : str
        Block within the tracking json holding this measurement.
    units : Units
        Units the observations are read in. Keeping these with the type is
        what stops a sigma being built in one unit and the observations in
        another, which is easy to do when both are written out by hand.
    label : str
        Human-readable suffix used when naming datasets. Must keep the
        substrings Plotting classifies on ("Range Rate" before "Range").
    real : bool
        True for real tracking data, False for simulated.
    """

    model_class: Any
    meas_name: str
    units: Any
    label: str
    real: bool = True


# class definition
class MeasurementRegistry:
    """
    The table of measurement types a scenario can be built from.

    Maps a short type name such as ``"SeqRange"`` onto the
    :class:`~scarabaeus.MeasurementType` describing it, so a stream can be
    written as ``(file, meas_type, station)`` and everything else looked up.

    Every method is a static method: there is one table per process, not one
    per instance, and :meth:`get` and :meth:`register` both work on it.

    See Also
    --------
    scarabaeus.MeasurementConfig : Settings shared across streams.
    scarabaeus.MeasurementStream : One stream, named by type.
    scarabaeus.MeasurementSpec : Consumes the registry to build a filter-ready
        spec.

    Notes
    -----
    The table is built on first use rather than at import, because building it
    means importing every model class and this module is imported during
    package initialisation.

    Examples
    --------
    .. code-block:: python

        import scarabaeus as scb

        scb.MeasurementRegistry.register(
            "MyRange",
            scb.MeasurementType(MyRangeModel, "meas_sim", km, "Range",
                                real=False),
        )
        spec = scb.MeasurementSpec.build([(file, "MyRange", station)])
    """

    # Built on first use by _build(). A class attribute rather than a module
    # global so it is namespaced with what uses it, and so a test that
    # registers a type can put it back (``MeasurementRegistry._cache = None``).
    _cache = None

    # region Methods
    @staticmethod
    def _build() -> dict:
        """
        Construct the default table.

        Returns
        -------
        dict
            ``{type_name: MeasurementType}`` for the built-in types: ``SeqRange``,
            ``Doppler``, ``IdealRange``, ``IdealRangeRate`` and ``Centroid``.

        Notes
        -----
        The model imports are local because this module is imported during
        package initialisation, and importing the models there would make the
        import circular.
        """
        from scarabaeus.measurements.SequentialRangingReal import SequentialRangingReal
        from scarabaeus.measurements.DopplerReal import DopplerReal
        from scarabaeus.measurements.RangeIdeal import RangeIdeal
        from scarabaeus.measurements.RangeRateIdeal import RangeRateIdeal
        from scarabaeus.measurements.CentroidingIdeal import CentroidingIdeal

        return {
            "SeqRange":       MeasurementType(SequentialRangingReal, "2W_sranging",
                                              km / km, "Range", real=True),
            "Doppler":        MeasurementType(DopplerReal, "2W_doppler",
                                              1 / sec, "Range Rate", real=True),
            "IdealRange":     MeasurementType(RangeIdeal, "meas_sim",
                                              km, "Range", real=False),
            "IdealRangeRate": MeasurementType(RangeRateIdeal, "meas_sim",
                                              km / sec, "Range Rate", real=False),
            "Centroid":       MeasurementType(CentroidingIdeal, "meas_sim",
                                              rad, "OpNav", real=False),
        }

    @staticmethod
    def all() -> dict:
        """
        The whole table, keyed by type name.

        Returns
        -------
        dict
            The live table, not a copy: mutating it changes what later lookups
            see. Prefer :meth:`register` for that.
        """
        if MeasurementRegistry._cache is None:
            MeasurementRegistry._cache = MeasurementRegistry._build()
        return MeasurementRegistry._cache

    @staticmethod
    def register(name: str, spec: MeasurementType) -> None:
        """
        Add a measurement type, or replace one of the same name.

        Lets a new measurement type be used by
        :meth:`~scarabaeus.MeasurementSpec.build` without editing this module.

        Parameters
        ----------
        name : str
            Short name streams will refer to the type by.
        spec : MeasurementType
            What the name implies: model class, data block, units, label.
        """
        MeasurementRegistry.all()[name] = spec

    @staticmethod
    def get(name: str) -> MeasurementType:
        """
        Look up one measurement type.

        Parameters
        ----------
        name : str
            A registered type name.

        Returns
        -------
        MeasurementType
            The specification registered under *name*.

        Raises
        ------
        KeyError
            If *name* is not registered. The message lists the names that are,
            since a typo here is otherwise reported much later as a missing
            model.
        """
        registry = MeasurementRegistry.all()
        if name not in registry:
            raise KeyError(
                f"Unknown measurement type '{name}'. "
                f"Available: {sorted(registry)}"
            )
        return registry[name]

    @staticmethod
    def assign_dataset_names(streams, registry: dict = None) -> list:
        """
        Name every stream, keeping the names unique.

        Streams that share a station and a type would otherwise collide, which
        matters because solutions are keyed by dataset name. Repeats get a
        counter appended, added after the label so that the substrings
        :class:`~scarabaeus.Plotting` classifies on stay intact.

        Parameters
        ----------
        streams : sequence of MeasurementStream
            The streams to name. A stream that already carries a ``name`` keeps
            it, and is still checked against the generated names for collisions.
        registry : dict, optional
            Table to resolve each stream's type against. Defaults to
            :meth:`all`.

        Returns
        -------
        list of str
            One name per stream, in the order given.
        """
        registry = registry or MeasurementRegistry.all()
        names, counts = [], {}

        for stream in streams:
            if stream.name is not None:
                names.append(stream.name)
                continue
            base = stream.default_name(registry[stream.meas_type])
            counts[base] = counts.get(base, 0) + 1
            names.append(base if counts[base] == 1 else f"{base} {counts[base]}")

        # A name given explicitly may still collide with a generated one
        seen, unique = {}, []
        for name in names:
            seen[name] = seen.get(name, 0) + 1
            unique.append(name if seen[name] == 1 else f"{name} {seen[name]}")
        return unique

    # endregion Methods
