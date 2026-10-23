# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import importlib
from importlib.metadata import version, metadata, PackageNotFoundError
from typing import TYPE_CHECKING
import pkgutil

from .rust import *

try:
    __version__ = version("scarabaeus")
    __license__ = metadata("scarabaeus")["License"]
except PackageNotFoundError:
    __version__ = "development"
    __license__ = "N/A"


_SUBMODULES = (
    "units",
    "timeAndFrame",
    "body",
    "spacecraft",
    "finiteBurn",
    "dynamics",
    "environment",
    "measurements",
    "orbitDetermination",
    "uncertaintyQuantification",
    "guidance",
    "utils",
)

_ALIASES = {
    "constants": ("scarabaeus.utils.Constants", "Constants"),
    "PCInfo": ("scarabaeus.utils.Constants", "PCInfo"),
    # Names that live in a module named for something else, so the
    # module-name lookup in __getattr__ cannot find them on its own.
    "MeasurementType": ("scarabaeus.measurements.MeasurementRegistry", "MeasurementType"),
}


def __getattr__(name):
    global _submodule_cache
    if name in _ALIASES:
        module_path, attr = _ALIASES[name]
    else:
        if _submodule_cache is None:
            _submodule_cache = _build_submodule_cache()
        if name not in _submodule_cache:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        module_path, attr = _submodule_cache[name], name

    value = getattr(importlib.import_module(module_path), attr)
    globals()[name] = value
    return value


def __dir__():
    global _submodule_cache
    if _submodule_cache is None:
        _submodule_cache = _build_submodule_cache()
    return sorted(set(globals()) | set(_submodule_cache) | set(_ALIASES))


_submodule_cache = None


def _build_submodule_cache():
    """Make map to submodules if it doesn't already exist."""
    symbol_map = {}
    for subpkg in _SUBMODULES:
        spec = importlib.util.find_spec(f"scarabaeus.{subpkg}")
        if spec is None or spec.submodule_search_locations is None:
            continue
        for _, modname, _ in pkgutil.iter_modules(spec.submodule_search_locations):
            symbol_map[modname] = f"scarabaeus.{subpkg}.{modname}"
    return symbol_map


## static checks for IDE
if TYPE_CHECKING:
    from scarabaeus.units.Dimensions import Dimensions
    from scarabaeus.units.Units import Units
    from scarabaeus.units.ArrayWUnits import ArrayWUnits
    from scarabaeus.timeAndFrame.EpochArray import EpochArray
    from scarabaeus.timeAndFrame.Frame import Frame
    from scarabaeus.timeAndFrame.ArrayWFrame import ArrayWFrame
    from scarabaeus.body.Body import Body
    from scarabaeus.timeAndFrame.SpiceManager import SpiceManager
    from scarabaeus.timeAndFrame.BodyOrientation import BodyOrientation
    from scarabaeus.utils.Constants import Constants as constants, PCInfo
    from scarabaeus.spacecraft.nPlateModel import nPlateModel
    from scarabaeus.spacecraft.Spacecraft import Spacecraft
    from scarabaeus.spacecraft.Actuator import Actuator
    from scarabaeus.finiteBurn.Maneuver import Maneuver
    from scarabaeus.spacecraft.Instrument import Instrument
    from scarabaeus.body.GroundStation import GroundStation
    from scarabaeus.body.SolidTide import SolidTide
    from scarabaeus.body.PoleTide import PoleTide
    from scarabaeus.body.OceanLoading import OceanLoading
    from scarabaeus.body.CelestialBody import CelestialBody
    from scarabaeus.spacecraft.Camera import Camera
    from scarabaeus.spacecraft.Antenna import Antenna
    from scarabaeus.dynamics.DynamicModel import DynamicModel
    from scarabaeus.dynamics.ImpulsiveBurn import ImpulsiveBurn
    from scarabaeus.environment.StateDefinition import StateDefinition
    from scarabaeus.environment.StateArray import StateArray
    from scarabaeus.dynamics.CannonballSRP import CannonballSRP
    from scarabaeus.dynamics.nPlateSRP import nPlateSRP
    from scarabaeus.dynamics.PointMassGravity import PointMassGravity
    from scarabaeus.dynamics.ThreeBodyGravity import ThreeBodyGravity
    from scarabaeus.dynamics.SphericalHarmonicsGravity import SphericalHarmonicsGravity
    from scarabaeus.dynamics.FiniteBurn import FiniteBurn
    from scarabaeus.dynamics.SmoothedImpulsiveBurn import SmoothedImpulsiveBurn
    from scarabaeus.dynamics.YarkovskyEffect import YarkovskyEffect
    from scarabaeus.dynamics.DynamicalUtils import DynamicalUtils
    from scarabaeus.dynamics.ThermalRadiation import ThermalRadiation
    from scarabaeus.dynamics.AntennaRadiationPressure import AntennaRadiationPressure
    from scarabaeus.dynamics.FirstOrderGaussMarkov import FirstOrderGaussMarkov
    from scarabaeus.dynamics.CustomParameter import CustomParameter
    from scarabaeus.dynamics.CustomDynamicModel import CustomDynamicModel
    from scarabaeus.dynamics.PiecewiseFirstOrderGaussMarkov import (
        PiecewiseFirstOrderGaussMarkov,
    )
    from scarabaeus.dynamics.PiecewisePolynomialAcceleration import (
        PiecewisePolynomialAcceleration,
    )
    from scarabaeus.dynamics.ForceModel import ForceModel
    from scarabaeus.dynamics.ForceModelTranslation import ForceModelTranslation
    from scarabaeus.dynamics.ForceModelRotation import ForceModelRotation
    from scarabaeus.environment.Propagator import Propagator
    from scarabaeus.environment.MissionSequence import MissionSequence
    from scarabaeus.environment.Trajectory import Trajectory
    from scarabaeus.environment.ScenarioSetup import ScenarioSetup
    from scarabaeus.utils.Noise import Noise
    from scarabaeus.measurements.RampTableManager import RampTableManager
    from scarabaeus.measurements.ObsData import ObsData
    from scarabaeus.measurements.MeasurementDataSet import MeasurementDataSet
    from scarabaeus.measurements.Measurement import Measurement
    from scarabaeus.measurements.RangeIdeal import RangeIdeal
    from scarabaeus.measurements.MediaCorrections import MediaCorrections
    from scarabaeus.measurements.SequentialRangingReal import SequentialRangingReal
    from scarabaeus.measurements.RangeRateIdeal import RangeRateIdeal
    from scarabaeus.measurements.CentroidingIdeal import CentroidingIdeal
    from scarabaeus.measurements.OpNavManager import OpNavManager
    from scarabaeus.measurements.DopplerIdeal import DopplerIdeal
    from scarabaeus.measurements.DopplerReal import DopplerReal
    from scarabaeus.measurements.DiffOneWayRangeIdeal import DiffOneWayRangeIdeal
    from scarabaeus.measurements.AngularIdeal import AngularIdeal
    from scarabaeus.measurements.CustomMeasurement import CustomMeasurement
    from scarabaeus.utils.CustomCallable import CustomCallable
    from scarabaeus.orbitDetermination.MeasurementEditing import MeasurementEditing
    from scarabaeus.orbitDetermination.FilterDataManager import FilterDataManager
    from scarabaeus.measurements.MeasurementSpec import MeasurementSpec
    from scarabaeus.measurements.MeasurementConfig import MeasurementConfig
    from scarabaeus.measurements.MeasurementRegistry import (
        MeasurementRegistry,
        MeasurementType,
    )
    from scarabaeus.measurements.MeasurementStream import MeasurementStream
    from scarabaeus.measurements.MeasurementScheduler import MeasurementScheduler
    from scarabaeus.uncertaintyQuantification.UncertaintyQuantification import (
        UncertaintyQuantification,
    )
    from scarabaeus.uncertaintyQuantification.CovarianceMatrix import CovarianceMatrix
    from scarabaeus.uncertaintyQuantification.MonteCarloSamples import MonteCarloSamples
    from scarabaeus.orbitDetermination.ProcessNoise import ProcessNoise
    from scarabaeus.orbitDetermination.StateNoiseCompensation import (
        StateNoiseCompensation,
    )
    from scarabaeus.orbitDetermination.DynamicalModelCompensation import (
        DynamicalModelCompensation,
    )
    from scarabaeus.orbitDetermination.ProcessNoiseSettings import ProcessNoiseSettings
    from scarabaeus.orbitDetermination.OutputSettings import OutputSettings
    from scarabaeus.orbitDetermination.FilterSettings import FilterSettings
    from scarabaeus.orbitDetermination.FilterOD import FilterOD
    from scarabaeus.orbitDetermination.SolutionOD import SolutionOD
    from scarabaeus.orbitDetermination.SRIF import SRIF
    from scarabaeus.orbitDetermination.SRIFB import SRIFB
    from scarabaeus.orbitDetermination.LKF import LKF
    from scarabaeus.orbitDetermination.CustomFilter import CustomFilter
    from scarabaeus.orbitDetermination.LSB import LSB
    from scarabaeus.orbitDetermination.MCMC import MCMC
    from scarabaeus.orbitDetermination.MultiFilterOD import MultiFilterOD
    from scarabaeus.guidance.Bplane import Bplane
    from scarabaeus.utils.Utils import Utils
    from scarabaeus.utils.OrbitalElements import OrbitalElements
    from scarabaeus.utils.Plotting import Plotting
    from scarabaeus.utils.SCBPolynomial import SCBPolynomial
    from scarabaeus.orbitDetermination.UtilsOD import UtilsOD
    from scarabaeus.finiteBurn.ManeuverParser import ManeuverParser
