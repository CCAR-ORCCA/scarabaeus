# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import MeasurementType

from dataclasses import dataclass, fields as dataclasses_fields
from typing import Any, Optional


# class definition
@dataclass
class MeasurementStream:
    """
    One measurement stream.

    Normally written as a plain ``(file, meas_type, station)`` tuple and
    normalised into this; use the class directly when a stream needs something
    the defaults do not cover.

    See Also
    --------
    scarabaeus.MeasurementRegistry : The measurement types ``meas_type`` names.
    scarabaeus.MeasurementConfig : Settings shared across streams.
    """

    file: str
    meas_type: str
    station: Any
    name: Optional[str] = None
    sigma: Any = None
    bias: Any = None
    media: Any = None

    # region Methods
    @classmethod
    def coerce(cls, item) -> "MeasurementStream":
        """
        Return ``item`` as a MeasurementStream.

        Parameters
        ----------
        item : MeasurementStream or tuple or list
            A stream, returned unchanged, or a ``(file, meas_type, station)``
            sequence of exactly three elements.

        Returns
        -------
        MeasurementStream
            The stream.

        Raises
        ------
        ValueError
            If a tuple or list does not have exactly three elements.
        TypeError
            If ``item`` is neither a stream nor a tuple or list.

        Notes
        -----
        Fields after ``station`` must be passed by name, because their order is
        not one anyone would guess: ``(file, type, station, 0.5)`` written for a
        sigma would otherwise name the dataset "0.5" and set no sigma.
        """
        if isinstance(item, cls):
            return item
        if isinstance(item, (tuple, list)):
            if len(item) != 3:
                fields = ", ".join(
                    f.name for f in dataclasses_fields(cls) if f.name not in
                    ("file", "meas_type", "station")
                )
                raise ValueError(
                    "A measurement stream tuple is exactly "
                    f"(file, meas_type, station); got {len(item)} element(s): "
                    f"{item!r}. Pass anything else by name with "
                    f"MeasurementStream(file=..., meas_type=..., station=..., "
                    f"...), whose remaining fields are: {fields}."
                )
            return cls(*item)
        raise TypeError(
            f"Expected a MeasurementStream or a (file, meas_type, station) "
            f"tuple, got {type(item).__name__}"
        )

    def default_name(self, type_spec: MeasurementType) -> str:
        """Return the default dataset name of this stream.

        Parameters
        ----------
        type_spec : MeasurementType
            Type of the stream, whose ``label`` is appended.

        Returns
        -------
        str
            ``"<station name> <label>"``.
        """
        station_name = getattr(self.station, "name", str(self.station))
        return f"{station_name} {type_spec.label}"

    # endregion Methods
