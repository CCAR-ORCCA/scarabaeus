# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from importlib.metadata import version, metadata
__version__ = version("scarabaeus")
__license__ = metadata("scarabaeus")["License"]

# import modules
from scarabaeus.units.Dimensions import Dimensions
from scarabaeus.units.Units import Units
from scarabaeus.units.ArrayWUnits import ArrayWUnits
from scarabaeus.timeAndFrame.EpochArray import EpochArray
from scarabaeus.timeAndFrame.Frame import Frame
from scarabaeus.timeAndFrame.ArrayWFrame import ArrayWFrame
from scarabaeus.body.Body import Body
from scarabaeus.timeAndFrame.SpiceManager import SpiceManager

from .scarabaeus_rust import *

from scarabaeus.utils.Constants import Constants as constants
from scarabaeus.utils.Constants import PCInfo

from scarabaeus.spacecraft.nPlateModel import nPlateModel
from scarabaeus.spacecraft.Spacecraft import Spacecraft
from scarabaeus.spacecraft.Actuator import Actuator
from scarabaeus.finiteBurn.Maneuver import Maneuver
from scarabaeus.spacecraft.Instrument import Instrument

from scarabaeus.body.GroundStation import GroundStation
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
from scarabaeus.dynamics.YarkovskyEffect import YarkovskyEffect
from scarabaeus.dynamics.FirstOrderGaussMarkov import FirstOrderGaussMarkov
from scarabaeus.dynamics.PiecewiseFirstOrderGaussMarkov import (
    PiecewiseFirstOrderGaussMarkov,
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
from scarabaeus.measurements.MeasurementDataSet import MeasurementDataSet
from scarabaeus.measurements.Measurement import Measurement
from scarabaeus.measurements.RangeIdeal import RangeIdeal
from scarabaeus.measurements.MediaCorrections import MediaCorrections

from scarabaeus.measurements.SequentialRangingReal import SequentialRangingReal
from scarabaeus.measurements.RangeRateIdeal import RangeRateIdeal
from scarabaeus.measurements.CentroidingIdeal import CentroidingIdeal
from scarabaeus.measurements.DopplerIdeal import DopplerIdeal
from scarabaeus.measurements.DopplerReal import DopplerReal
from scarabaeus.measurements.DiffOneWayRangeIdeal import DiffOneWayRangeIdeal
from scarabaeus.measurements.AngularIdeal import AngularIdeal

from scarabaeus.orbitDetermination.MeasurementEditing import MeasurementEditing
from scarabaeus.orbitDetermination.FilterDataManager import FilterDataManager

from scarabaeus.measurements.MeasurementSpec import MeasurementSpec

from scarabaeus.uncertaintyQuantification.UncertaintyQuantification import UncertaintyQuantification
from scarabaeus.uncertaintyQuantification.CovarianceMatrix import CovarianceMatrix

from scarabaeus.orbitDetermination.ProcessNoise import ProcessNoise
from scarabaeus.orbitDetermination.StateNoiseCompensation import StateNoiseCompensation
from scarabaeus.orbitDetermination.DynamicalModelCompensation import DynamicModelCompensation
from scarabaeus.orbitDetermination.ProcessNoiseSettings import ProcessNoiseSettings
from scarabaeus.orbitDetermination.OutputSettings import OutputSettings
from scarabaeus.orbitDetermination.FilterSettings import FilterSettings
from scarabaeus.orbitDetermination.FilterOD import FilterOD
from scarabaeus.orbitDetermination.SolutionOD import SolutionOD
from scarabaeus.orbitDetermination.SRIF import SRIF
from scarabaeus.orbitDetermination.SRIFB import SRIFB
from scarabaeus.orbitDetermination.LKF import LKF
from scarabaeus.orbitDetermination.LSB import LSB
from scarabaeus.orbitDetermination.MultiFilterOD import MultiFilterOD

from scarabaeus.guidance.Bplane import Bplane

from scarabaeus.utils.Utils import Utils

from scarabaeus.utils.OrbitalElements import OrbitalElements
from scarabaeus.utils.SCBPolynomial import SCBPolynomial
from scarabaeus.orbitDetermination.UtilsOD import UtilsOD

from scarabaeus.finiteBurn.ManeuverParser import ManeuverParser