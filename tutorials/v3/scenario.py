"""The "Navigating OSIRIS-REx" mission definition.

Chapters 1-3 build each of these objects on the page; later chapters import them from here
instead of rebuilding them. Definitions only: no analysis, no plotting.

Importing this module loads the series metakernel (``orex_approach.tm``), because the epochs and
the a-priori state below need the leap-seconds kernel and the flight ephemeris. Each chapter then
loads its own products (truth SPK, solutions) explicitly in its setup cell.
"""

import scarabaeus as scb

KERNELS = "orex_approach.tm"
scb.SpiceManager.clear_kernels()
scb.SpiceManager.load_kernel_from_mkfile(KERNELS)

kg, km, m, mm, sec, minute, day = scb.Units.get_units(["kg", "km", "m", "mm", "sec", "min", "day"])
J2000 = scb.Frame("J2000")
SUN = scb.CelestialBody.from_constants("SUN")

# --- Time (chapter 1) ---------------------------------------------------------------------------
# Tracking covers June 1-8; the truth runs one day longer so predictions for June 8 can be scored.
data_epochs = scb.EpochArray.interval(
    scb.SpiceManager.utc2et("2018-06-01T00:00:00"),
    scb.SpiceManager.utc2et("2018-06-08T00:00:00"),
    dt=scb.ArrayWUnits(30, minute),
    sys="TDB",
)
truth_epochs = scb.EpochArray.interval(
    scb.SpiceManager.utc2et("2018-06-01T00:00:00"),
    scb.SpiceManager.utc2et("2018-06-09T00:00:00"),
    dt=scb.ArrayWUnits(30, minute),
    sys="TDB",
)
# First Doppler point of the real DSS-35 pass on June 8 (chapter 8): the epoch every chapter predicts.
t_pass = scb.EpochArray([scb.SpiceManager.utc2et("2018-06-08T05:55:03")], sys="TDB")

# --- Spacecraft and dynamics (chapter 1) --------------------------------------------------------
orex_flight = scb.Spacecraft("OSIRIS-REx (flight)", -64)  # flight team's reconstruction
x0_flight = orex_flight.get_state(data_epochs[0], reference_frame="J2000", origin="SUN").quantity

MASS = scb.ArrayWUnits(1500.0, kg)
SRP_AREA = scb.ArrayWUnits(11.0, m**2).convert_to(km**2)
CR = scb.ArrayWUnits(1.2, None)
PLANETS = ["MERCURY", "VENUS", "EARTH", "MARS", "JUPITER", "SATURN"]

orex_truth = scb.Spacecraft("OREx truth", -64000, MASS, SRP_AREA, CR)

# --- Tracking (chapter 2) -----------------------------------------------------------------------
stations = {dss: scb.GroundStation(dss) for dss in ["DSS-14", "DSS-43", "DSS-63"]}
RANGE_SIGMA = scb.ArrayWUnits(2.0, m).convert_to(km)
RANGE_RATE_SIGMA = scb.ArrayWUnits(0.1, mm / sec).convert_to(km / sec)
ELEVATION_MASK = 10.0  # deg

models = []
for dss, station in stations.items():
    models.append(scb.RangeIdeal(f"{dss} range", station, sigma=RANGE_SIGMA))
    models.append(scb.RangeRateIdeal(f"{dss} range-rate", station, sigma=RANGE_RATE_SIGMA))

# --- The filter's a-priori (chapter 3) ----------------------------------------------------------
# The truth starts from the flight state, so the a-priori error is known: 10 km and 1 cm/s per axis.
orex_reference = scb.Spacecraft("OREx reference", -64001, MASS, SRP_AREA, CR)

r_apriori = x0_flight[:3] + scb.ArrayWUnits([10.0] * 3, km)
v_apriori = x0_flight[3:] + scb.ArrayWUnits([0.01] * 3, m / sec).convert_to(km / sec)
apriori_state = (
    scb.StateDefinition()
    .position(orex_reference, scb.ArrayWFrame(r_apriori, J2000))
    .velocity(orex_reference, scb.ArrayWFrame(v_apriori, J2000))
)
apriori_sigmas = [scb.ArrayWUnits(100.0, km)] * 3 + [scb.ArrayWUnits(0.1, m / sec).convert_to(km / sec)] * 3
P0 = scb.CovarianceMatrix(apriori_sigmas, data_epochs[0], from_list=True)


# 🔵 TODO: filters update the propagator they are given, so every fit needs a fresh one built from
# the a-priori. Remove this function once they stop doing so (PLAN.md, API change 3).
def reference_propagator(spacecraft=orex_reference, state=apriori_state):
    return scb.Propagator(
        primary_body=spacecraft,
        state_vector=scb.StateArray(epoch=data_epochs[0], origin=SUN, state=state),
        tspan=data_epochs,
        force_models=scb.ForceModelTranslation(spacecraft, third_bodies=PLANETS, cannonball_SRP=True),
    )
