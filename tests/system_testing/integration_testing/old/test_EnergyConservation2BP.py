"""
Unit Tests for the 2BP energy conservation with DOP853 and PyASA propagators.
"""

"""
# Versioning
__version__ = "0.0.0"
__author__ = "Giovanni Fereoli"

# Imports
import sys
import os

import pytest
import numpy as np
import matplotlib.pyplot as plt
import pyasa as asa

sys.path.append("./src/")
import scarabaeus as scb

# ---------------------------#
#           Setup           #
# ---------------------------#

# Generate common units
kg, km, sec, rad, meter, AU, min, hour, day, deg, Newton = (
    scb.UnitsArray.generate_common_units()
)

# Load spice kernels
furnshKernelFilename = os.getcwd() + "/data/Kernels/kernels.tm"
scb.SpiceManager.load_kernel_from_mkfile(furnshKernelFilename)

# Initialize the body object for the orbiter
Orbiter_drymass = scb.ArrayWUnits(1275.0, kg)
Orbiter_fuelmass = scb.ArrayWUnits(135, kg)
Orbiter_totalmass = Orbiter_drymass + Orbiter_fuelmass
Orbiter_area = scb.ArrayWUnits(1e-06, km**2)
Orbiter_cr_srp = scb.ArrayWUnits(1, None)

# Initialize frame, planet constants and origin
frame = "J2000"
sun_constants = scb.Constants.sun
origin = scb.Planet(
    name="SUN",
    mass=sun_constants["mass"],
    selfBaseFrame="J2000",
    frameTree=[],
    meanRadius=sun_constants["radius"],
    gravityConstant=sun_constants["mu"],
    spice_id=10,
)

# Initialize epochs
dt = 100
steps = 10000
time_0 = scb.SpiceManager.jd2et(2461809.72995654)
time_f = time_0 + steps * dt
epoch_array = scb.EpochArray(np.arange(time_0, time_f, dt), timeFrame="TDB")
epoch_0 = epoch_array[0]
epoch_f = epoch_array[-1]

# Initialize position and velocity vectors
pos_0 = scb.SpiceManager.get_pos(
    str(Orbiter.spice_id), epoch_array[0].time, frame, origin.name
)
vel_0 = scb.SpiceManager.get_vel(
    str(Orbiter.spice_id), epoch_array[0].time, frame, origin.name
)

# Set up propagator
state_0 = [
    ("position", 3, "estimated", "dynamic", Orbiter, pos_0),
    ("velocity", 3, "estimated", "dynamic", Orbiter, vel_0),
]
state_vector_0 = scb.StateVector(
    epoch=epoch_array[0], frame=frame, origin=origin, state=state_0
)
prop = scb.Propagator(
    body=Orbiter,
    state_vector=state_vector_0,
    ets=epoch_array,
    propagate_stm=False,
    integrator="DOP853",
)

# Propagate and get the states with DOP853
prop.propagate()
epochs_new = scb.EpochArray(scb.ArrayWUnits(prop.ets.time, sec), timeFrame="TDB")
time = epochs_new.time.values
r = scb.ArrayWUnits(prop.state[:, 0:3], km)
v = scb.ArrayWUnits(prop.state[:, 3:6], km / sec)

# Propagate and get the states with PyASA
ode = lambda t, y: prop.ode(t, np.array(y))
ias15 = asa.IAS15(dt, 0, ode, state_vector_0.size, prop.propagate_stm)
ts, ys = ias15.integrate(
    np.arange(epoch_0.time, epoch_f.time, dt), state_vector_0.extract_values().values
)
time_pyasa = np.array(ts)
r_pyasa = scb.ArrayWUnits(np.array(ys)[:, 0:3], km)
v_pyasa = scb.ArrayWUnits(np.array(prop.state)[:, 3:6], km / sec)

# Compute total energy for DOP853
kinetic_energy = scb.ArrayWUnits(
    np.array(
        [
            0.5 * Orbiter_totalmass.values * np.linalg.norm(v[i].values) ** 2
            for i in range(0, len(time))
        ]
    ),
    kg * km**2 / sec**2,
)
potential_energy = scb.ArrayWUnits(
    np.array(
        [
            float(origin.gravityConstant.values)
            * Orbiter_totalmass.values
            / np.linalg.norm(r[i].values)
            for i in range(0, len(time))
        ]
    ),
    kg * km**2 / sec**2,
)
total_energy = kinetic_energy - potential_energy

# Compute total energy for PyASA
kinetic_energy_pyasa = scb.ArrayWUnits(
    np.array(
        [
            0.5 * Orbiter_totalmass.values * np.linalg.norm(v_pyasa[i].values) ** 2
            for i in range(0, len(time_pyasa))
        ]
    ),
    kg * km**2 / sec**2,
)
potential_energy_pyasa = scb.ArrayWUnits(
    np.array(
        [
            float(origin.gravityConstant.values)
            * Orbiter_totalmass.values
            / np.linalg.norm(r_pyasa[i].values)
            for i in range(0, len(time_pyasa))
        ]
    ),
    kg * km**2 / sec**2,
)
total_energy_pyasa = kinetic_energy_pyasa - potential_energy_pyasa

"""
# ---------------------------#
#           Tests            #
# ---------------------------#


"""
@pytest.mark.parametrize("tolerance", [1e-10])
def test_total_energy_conservation(tolerance):
# """
# Test the conservation of total energy.

# This function checks if the total energy variation between each step
# is within the specified tolerance. If the variation exceeds the tolerance,
# an assertion error is raised.

# ASSUMPTIONS:
# 1) 2-Body problem with the Sun as the central body.
# 2) Reference frame is J2000.

# Parameters:
# - tolerance (float): The maximum allowed variation in total energy.

# Raises:
# - AssertionError: If the total energy variation exceeds the tolerance."

"""
    initial_energy = total_energy[0].values
    for i in range(1, len(time)):
        assert (
            np.abs((total_energy[i].values - initial_energy) / initial_energy)
            < tolerance
        ), f"Total energy variation exceeded tolerance at step {i} with DOP853 propagator."


def test_total_energy_conservation_pyasa(tolerance):
"""
# Test the conservation of total energy with PyASA propagator.

# This function checks if the total energy variation between each step
# is within the specified tolerance. If the variation exceeds the tolerance,
# an assertion error is raised.

# ASSUMPTIONS:
# 1) 2-Body problem with the Sun as the central body.
# 2) Reference frame is J2000.

# Parameters:
# - tolerance (float): The maximum allowed variation in total energy.

# Raises:
# - AssertionError: If the total energy variation exceeds the tolerance.

"""
    initial_energy_pyasa = total_energy_pyasa[0].values
    for i in range(1, len(time_pyasa)):
        assert (
            np.abs(
                (total_energy_pyasa[i].values - initial_energy_pyasa)
                / initial_energy_pyasa
            )
            < tolerance
        ), f"Total energy variation exceeded tolerance at step {i} with PyASA propagator."

"""

# Plots
# plt.figure(figsize=(10, 6))
# plt.scatter(
#     time,
#     np.abs((total_energy.values - total_energy[0].values) / total_energy[0].values),
#     color="red",
#     s=1,
#     label="DOP853",
# )
# plt.scatter(
#     time_pyasa,
#     np.abs(
#         (total_energy_pyasa.values - total_energy_pyasa[0].values)
#         / total_energy_pyasa[0].values
#     ),
#     color="blue",
#     s=1,
#     label="PyASA",
# )
# plt.yscale("log")
# plt.grid(True, which="both", linestyle="--")
# plt.xlabel("Time [TBD]")
# plt.ylabel("E-E0/E0 [-]")
# plt.title("Orbital Energy Conservation - 2BP")
# plt.legend()
# # plt.show()
