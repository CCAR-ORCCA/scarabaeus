# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""
Shared JSON serialization helpers.

Scarabaeus persists most artifacts as JSON (trajectory sidecars, propagator
settings, STMs, filter datasets, OD solutions). These helpers give every save
path one numpy-aware encoder and a single ``dump_json`` / ``load_json`` entry
point so the on-disk convention stays consistent.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Union

import numpy as np


class UnknownSerializationFormatError(ValueError):
    pass


class SCBJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands the numpy types Scarabaeus produces."""

    def default(self, o: Any):
        """
        Encode the non-native types Scarabaeus emits into JSON values.

        Converts numpy arrays/scalars, ``set``/``tuple``, and ``pathlib.Path`` to
        JSON-native types; defers anything else to :meth:`json.JSONEncoder.default`.

        Parameters
        ----------
        o : Any
            Object the base encoder could not serialize.

        Returns
        -------
        Any
            A JSON-native representation of ``o``.
        """
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (set, tuple)):
            return list(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def to_jsonable(obj: Any) -> Any:
    """
    Return a deep, JSON-serializable copy of ``obj``.

    Round-trips ``obj`` through the numpy-aware :class:`SCBJSONEncoder` so that
    numpy scalars/arrays, sets, tuples and ``Path`` objects are converted to
    native Python types.

    Parameters
    ----------
    obj : Any
        Object to convert. May contain numpy types nested arbitrarily.

    Returns
    -------
    Any
        A new object containing only JSON-native types (``dict``, ``list``,
        ``str``, ``int``, ``float``, ``bool``, ``None``).
    """
    return json.loads(json.dumps(obj, cls=SCBJSONEncoder))


def dump_json(obj: Any, path: Union[str, Path], indent: int = 2) -> Path:
    """
    Write ``obj`` to ``path`` as JSON using the numpy-aware encoder.

    Parent directories are created automatically.

    Parameters
    ----------
    obj : Any
        Object to serialize; numpy types are handled by :class:`SCBJSONEncoder`.
    path : str or pathlib.Path
        Destination file path.
    indent : int, optional
        Indentation passed to :func:`json.dump`. Defaults to ``2``.

    Returns
    -------
    pathlib.Path
        The resolved destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, cls=SCBJSONEncoder)
    return path


def load_json(path: Union[str, Path]) -> Any:
    """
    Read and return the JSON object stored at ``path``.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a JSON file written by :func:`dump_json`.

    Returns
    -------
    Any
        The decoded JSON object.
    """
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Pickle helpers + format-agnostic dispatch
# ----------------------------------------------------------------------------
# Every JSON save path in Scarabaeus is built on a ``to_dict`` / ``from_dict``
# pair. ``dump_pickle`` / ``load_pickle`` mirror ``dump_json`` / ``load_json`` so
# the *same* dict can be persisted as binary pickle instead of text JSON, and
# ``dump`` / ``load`` pick the format from an explicit ``fmt`` or the file
# extension. This gives every artifact a uniform "write/read as JSON or pickle"
# option without each class re-implementing it.

# File extensions recognised as pickle for format auto-detection.
PICKLE_SUFFIXES = (".pkl", ".pickle", ".pck", ".pcl")


def dump_pickle(
    obj: Any, path: Union[str, Path], protocol: int = pickle.HIGHEST_PROTOCOL
) -> Path:
    """
    Write ``obj`` to ``path`` as a pickle. Parent directories are created.

    Parameters
    ----------
    obj : Any
        Object to serialize (typically a ``to_dict`` result, but any picklable
        object is accepted).
    path : str or pathlib.Path
        Destination file path.
    protocol : int, optional
        Pickle protocol. Defaults to ``pickle.HIGHEST_PROTOCOL``.

    Returns
    -------
    pathlib.Path
        The resolved destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)
    return path


def load_pickle(path: Union[str, Path]) -> Any:
    """
    Read and return the object stored at ``path`` by :func:`dump_pickle`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a pickle file.

    Returns
    -------
    Any
        The unpickled object.
    """
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_format(path: Union[str, Path], fmt: str = None) -> str:
    """
    Resolve the serialization format, returning ``"json"`` or ``"pickle"``.

    Uses ``fmt`` when given (case-insensitive; accepts ``"json"``, ``"pickle"``,
    ``"pkl"``); otherwise infers it from the file extension (``.pkl`` /
    ``.pickle`` gives pickle, anything else json).

    Raises
    ------
    UnknownSerializationFormatError
        If ``fmt`` is given and is none of ``"json"``, ``"pickle"`` and
        ``"pkl"``.
    """
    if fmt is not None:
        f = fmt.lower()
        if f in ("pickle", "pkl"):
            return "pickle"
        if f == "json":
            return "json"
        raise UnknownSerializationFormatError(f"Unknown serialization format {fmt!r}; use 'json' or 'pickle'.")
    return "pickle" if Path(path).suffix.lower() in PICKLE_SUFFIXES else "json"


def format_suffix(fmt: str = None) -> str:
    """Return the file extension (``".json"`` or ``".pkl"``) for ``fmt``.

    ``fmt`` of ``None`` defaults to ``".json"``. Useful for save paths that build
    their own file names and need the extension to follow the chosen format.
    """
    return ".pkl" if resolve_format(".json", fmt) == "pickle" else ".json"


def with_format_suffix(path: Union[str, Path], fmt: str = None) -> Path:
    """Return ``path`` with its extension switched to match ``fmt``.

    When ``fmt`` is ``None`` the path is returned unchanged (the caller's
    extension already selects the format); otherwise the suffix becomes
    ``.json`` or ``.pkl`` so the on-disk name matches the bytes written.
    """
    path = Path(path)
    if fmt is None:
        return path
    return path.with_suffix(format_suffix(fmt))


def dump(obj: Any, path: Union[str, Path], fmt: str = None, indent: int = 2) -> Path:
    """
    Serialize ``obj`` to ``path`` as JSON or pickle (one entry point for both).

    Parameters
    ----------
    obj : Any
        Object to serialize (typically a ``to_dict`` result).
    path : str or pathlib.Path
        Destination file path.
    fmt : {"json", "pickle"}, optional
        Output format. When ``None`` (default), inferred from the file extension.
    indent : int, optional
        Indentation for the JSON format. Ignored for pickle. Defaults to ``2``.

    Returns
    -------
    pathlib.Path
        The resolved destination path.
    """
    if resolve_format(path, fmt) == "pickle":
        return dump_pickle(obj, path)
    return dump_json(obj, path, indent=indent)


def load(path: Union[str, Path], fmt: str = None) -> Any:
    """
    Read a JSON or pickle file written by :func:`dump`.

    Parameters
    ----------
    path : str or pathlib.Path
        Source file path.
    fmt : {"json", "pickle"}, optional
        Input format. When ``None`` (default), inferred from the file extension.

    Returns
    -------
    Any
        The decoded object.
    """
    if resolve_format(path, fmt) == "pickle":
        return load_pickle(path)
    return load_json(path)


# ----------------------------------------------------------------------------
# Shared domain primitives
# ----------------------------------------------------------------------------
# These convert the recurring Scarabaeus value types (``ArrayWUnits`` and the
# state-component owners) to/from plain dicts. They live here — rather than in
# any one domain module — so that every per-class ``to_dict`` / ``from_dict``
# (StateArray, Propagator, …) builds on the same on-disk convention.
#
# The ``scarabaeus`` imports are intentionally deferred to call time: this
# module is imported by core domain modules during package initialization, so a
# top-level domain import here would create a circular import.


def awu_to_dict(awu) -> Union[dict, None]:
    """
    Serialize an ``ArrayWUnits`` to a plain dict.

    Parameters
    ----------
    awu : ArrayWUnits or None
        Quantity to serialize.

    Returns
    -------
    dict or None
        ``{"values": <list>, "units": <units dict>}``, or ``None`` when *awu*
        is ``None``.
    """
    if awu is None:
        return None
    return {
        "values": np.asarray(awu.values).tolist(),
        "units": awu.units.to_dict(),
    }


def awu_from_dict(d):
    """
    Reconstruct an ``ArrayWUnits`` from a dict produced by :func:`awu_to_dict`.

    Parameters
    ----------
    d : dict or None
        Serialized quantity (``{"values", "units"}``).

    Returns
    -------
    ArrayWUnits or None
        The reconstructed quantity, or ``None`` when *d* is ``None``.
    """
    if d is None:
        return None
    from scarabaeus import ArrayWUnits
    from scarabaeus.units.Units import Units

    return ArrayWUnits(np.array(d["values"]), Units.from_dict(d["units"]))


def safe_initial_mass(owner):
    """
    Return the spacecraft's initial mass as an ``ArrayWUnits``.

    Robust to the mass having been converted to a time-varying polynomial during
    a prior propagation: prefers the stored ``_mass_profile``, then evaluates the
    mass at the start of the *first* profile span, then falls back to a plain
    ``mass()`` query.

    The first span is the one that matters: profiles accumulate one per
    propagated leg, so the most recent span starts wherever the last leg did,
    by which point a burn may already have spent propellant.

    Parameters
    ----------
    owner : Spacecraft
        Body whose initial mass is requested.

    Returns
    -------
    ArrayWUnits or None
        The initial mass, or ``None`` if it cannot be determined.
    """
    from scarabaeus import ArrayWUnits

    mp = getattr(owner, "_mass_profile", None)
    if isinstance(mp, ArrayWUnits):
        return mp
    tspans = getattr(owner, "_mass_profiles_tspan", None)
    if tspans:
        try:
            t0 = float(np.asarray(tspans[0][0]).ravel()[0])
            return owner.mass(t0)
        except Exception:
            pass
    try:
        return owner.mass()
    except Exception:
        return None


def scalar_to_dict(value) -> Any:
    """
    Serialize a value that may be a bare number or an ``ArrayWUnits``.

    Several spacecraft properties (``ref_coeff`` above all) are documented as
    plain floats but are just as often built as unitful quantities, so the
    on-disk form has to carry both cases. An ``ArrayWUnits`` becomes the tagged
    dict :func:`awu_to_dict` produces, plus a ``"__awu__"`` marker so
    :func:`scalar_from_dict` knows to rebuild the quantity; anything else is
    written through unchanged.

    Parameters
    ----------
    value : ArrayWUnits, float, int or None
        Value to serialize.

    Returns
    -------
    Any
        A JSON-native representation of *value*.
    """
    from scarabaeus import ArrayWUnits

    if isinstance(value, ArrayWUnits):
        return {"__awu__": True, **awu_to_dict(value)}
    return value


def scalar_from_dict(d):
    """
    Reconstruct a value written by :func:`scalar_to_dict`.

    Parameters
    ----------
    d : dict, float, int or None
        Serialized value.

    Returns
    -------
    ArrayWUnits, float, int or None
        The ``ArrayWUnits`` when *d* carries the ``"__awu__"`` marker, else *d*
        unchanged.
    """
    if isinstance(d, dict) and d.get("__awu__"):
        return awu_from_dict(d)
    return d


def owner_to_dict(owner) -> dict:
    """
    Serialize a state-component owner to a plain dict.

    Captures the type, name and SPICE id of any owner; for a ``Spacecraft`` it
    also records the initial mass, area, reflectivity coefficient, n-plate model
    and attitude configuration so the object can be rebuilt by
    :func:`owner_from_dict`.

    Parameters
    ----------
    owner : Spacecraft, CelestialBody, GroundStation or Body
        State-component owner to serialize.

    Returns
    -------
    dict
        Serialized owner (always has ``type``, ``name``, ``spice_id``).
    """
    from scarabaeus import Spacecraft

    info = {
        "type": owner.__class__.__name__,
        "name": owner.name,
        "spice_id": getattr(owner, "spice_id", None),
    }
    if isinstance(owner, Spacecraft):
        info["tot_mass"] = awu_to_dict(safe_initial_mass(owner))
        info["area"] = awu_to_dict(getattr(owner, "area", None))
        info["ref_coeff"] = scalar_to_dict(getattr(owner, "ref_coeff", None))
        # The n-plate model and the attitude settings are what nPlateSRP reads
        # off the spacecraft, so a reloaded body has to carry them or an
        # n-plate run comes back a cannonball.
        npm = getattr(owner, "n_plate_model", None)
        info["n_plate_model"] = npm.to_dict() if npm is not None else None
        info["body_fixed_frame_id"] = getattr(owner, "body_fixed_frame_id", None)
        info["attitudeMode"] = getattr(owner, "attitudeMode", None)
        info["fixedInertialAttitude"] = _attitude_to_dict(
            getattr(owner, "fixedInertialAttitude", None)
        )
        info["nadirPointingAxis"] = _attitude_to_dict(
            getattr(owner, "nadirPointingAxis", None)
        )
    return info


def _attitude_to_dict(value):
    """Serialize an attitude vector/matrix, which may be a numpy array or ``None``."""
    if value is None:
        return None
    return np.asarray(value).tolist()


def owner_from_dict(d):
    """
    Reconstruct a state-component owner from :func:`owner_to_dict`.

    Parameters
    ----------
    d : dict
        Serialized owner (as produced by :func:`owner_to_dict`).

    Returns
    -------
    Spacecraft, GroundStation, CelestialBody or Body
        The reconstructed owner. Unknown ``type`` values fall back to a generic
        ``Body``.
    """
    from scarabaeus import Spacecraft, GroundStation, Body, nPlateModel

    t = d.get("type")
    if t == "Spacecraft":
        npm = d.get("n_plate_model")
        nadir_axis = d.get("nadirPointingAxis")
        fixed_attitude = d.get("fixedInertialAttitude")
        return Spacecraft(
            d["name"],
            d["spice_id"],
            tot_mass=awu_from_dict(d.get("tot_mass")),
            area=awu_from_dict(d.get("area")),
            ref_coeff=scalar_from_dict(d.get("ref_coeff")),
            n_plate_model=nPlateModel.from_dict(npm) if npm is not None else None,
            body_fixed_frame_id=d.get("body_fixed_frame_id"),
            attitudeMode=d.get("attitudeMode"),
            fixedInertialAttitude=(
                np.array(fixed_attitude) if fixed_attitude is not None else None
            ),
            nadirPointingAxis=(
                np.array(nadir_axis) if nadir_axis is not None else None
            ),
        )
    if t == "GroundStation":
        return GroundStation(d["name"], d.get("spice_id"))
    if t == "CelestialBody":
        from scarabaeus.body.CelestialBody import CelestialBody

        return CelestialBody.from_constants(d["name"])
    # Generic Body fallback
    return Body(d["name"], d.get("spice_id"))
