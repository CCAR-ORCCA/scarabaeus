"""The "Navigating OSIRIS-REx" mission definition.

Chapters 1-5 build each of these objects on the page; from chapter 6 on, the setup cell imports
them from here instead of rebuilding them. Definitions only: no analysis, no plotting.

Importing this module loads the series metakernel (``orex_approach.tm``), because the epochs and
the a-priori state below need the leap-seconds kernel and the flight ephemeris. Each chapter then
loads the products of earlier chapters (truth SPK, tracking files) explicitly in its setup cell.
"""

import scarabaeus as scb

KERNELS = "orex_approach.tm"
scb.SpiceManager.clear_kernels()
scb.SpiceManager.load_kernel_from_mkfile(KERNELS)

kg, km, m, mm, sec, minute, day = scb.Units.get_units(["kg", "km", "m", "mm", "sec", "min", "day"])
J2000 = scb.Frame("J2000")
sun = scb.CelestialBody.from_constants("SUN")

# --- Time (chapter 1) ---------------------------------------------------------------------------
epochs = scb.EpochArray.interval(
    scb.SpiceManager.utc2et("2018-06-01T00:00:00"),
    scb.SpiceManager.utc2et("2018-06-08T00:00:00"),
    dt=scb.ArrayWUnits(30, minute),
    sys="TDB",
)
# The truth runs three days past the tracking so predictions can be scored (chapter 2).
truth_epochs = scb.EpochArray.interval(
    scb.SpiceManager.utc2et("2018-06-01T00:00:00"),
    scb.SpiceManager.utc2et("2018-06-11T00:00:00"),
    dt=scb.ArrayWUnits(30, minute),
    sys="TDB",
)

# --- Spacecraft and dynamics (chapters 1-2) -----------------------------------------------------
orex_flight = scb.Spacecraft("OSIRIS-REx (flight)", -64)  # flight team's reconstruction
x0_flight = orex_flight.get_state(epochs[0], reference_frame="J2000", origin="SUN").quantity

mass = scb.ArrayWUnits(1500.0, kg)
area = scb.ArrayWUnits(11.0, m**2).convert_to(km**2)
cr = scb.ArrayWUnits(1.2, None)
planets = ["MERCURY", "VENUS", "EARTH", "MARS", "JUPITER", "SATURN"]

orex = scb.Spacecraft("OREx truth", -64000, mass, area, cr)  # propagated from x0_flight (ch. 2)

# --- Tracking (chapter 3) -----------------------------------------------------------------------
TRACKING_DIR = "outputs/03_tracking_data"
stations = {dss: scb.GroundStation(dss) for dss in ["DSS-14", "DSS-43", "DSS-63"]}

models = []
for dss, station in stations.items():
    models.append(scb.RangeIdeal(f"{dss} range", station, sigma=scb.ArrayWUnits(2.0, m).convert_to(km)))
    models.append(
        scb.RangeRateIdeal(
            f"{dss} range-rate", station, sigma=scb.ArrayWUnits(0.1, mm / sec).convert_to(km / sec)
        )
    )

# --- The filter's dynamics model (chapter 4) ----------------------------------------------------
# The truth starts from the flight state, so the a-priori error is known: 10 km and 1 cm/s per axis.
reference = scb.Spacecraft("OREx reference", -64001, mass, area, cr)

r_apriori = x0_flight[:3] + scb.ArrayWUnits([10.0] * 3, km)
v_apriori = x0_flight[3:] + scb.ArrayWUnits([0.01] * 3, m / sec).convert_to(km / sec)
apriori_state = (
    scb.StateDefinition()
    .position(reference, scb.ArrayWFrame(r_apriori, J2000))
    .velocity(reference, scb.ArrayWFrame(v_apriori, J2000))
)
apriori_sigmas = [scb.ArrayWUnits(100.0, km)] * 3 + [scb.ArrayWUnits(0.1, m / sec).convert_to(km / sec)] * 3
P0 = scb.CovarianceMatrix(apriori_sigmas, epochs[0], from_list=True)


# 🔵 TODO: filters update the propagator they are given, so every fit needs a fresh one built from
# the a-priori. Remove this function once they stop doing so (v3/PLAN.md, API change 3).
def reference_propagator(spacecraft=reference, state=apriori_state, cannonball_SRP=True):
    return scb.Propagator(
        primary_body=spacecraft,
        state_vector=scb.StateArray(epoch=epochs[0], origin=sun, state=state),
        tspan=epochs,
        force_models=scb.ForceModelTranslation(spacecraft, third_bodies=planets, cannonball_SRP=cannonball_SRP),
        display_progress=False,
    )
