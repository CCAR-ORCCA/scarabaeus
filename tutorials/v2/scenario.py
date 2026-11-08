"""The "Navigating OSIRIS-REx" mission definition.

Chapters 1-5 build each of these objects on the page; from chapter 6 on, the setup cell imports
them from here instead of rebuilding them. Definitions only: no analysis, no plotting.

Importing this module loads the series metakernel (``orex_approach.tm``), because the epochs and
the a-priori state below need the leap-seconds kernel and the flight ephemeris. Each chapter then
loads the products of earlier chapters (truth SPK, tracking files) explicitly in its setup cell.
"""

import json
import shutil
from pathlib import Path

import numpy as np
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
planets = ["MERCURY", "VENUS", "EARTH", "MARS BARYCENTER", "JUPITER BARYCENTER", "SATURN BARYCENTER"]

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


# Filters update the propagator they are given, so every fit needs a fresh one built from the a priori.
def reference_propagator(spacecraft=reference, state=apriori_state, cannonball_SRP=True, tspan=epochs):
    return scb.Propagator(
        primary_body=spacecraft,
        state_vector=scb.StateArray(epoch=tspan[0], origin=sun, state=state),
        tspan=tspan,
        force_models=scb.ForceModelTranslation(spacecraft, third_bodies=planets, cannonball_SRP=cannonball_SRP),
        display_progress=False,
    )


# --- Chapter outputs ----------------------------------------------------------------------------
def output_folder(chapter):
    """Return ``outputs/<chapter>/``, emptied: each run of a chapter starts from a clean folder."""
    folder = Path("outputs") / chapter
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    return folder


# --- Real tracking data (chapters 9-11) ---------------------------------------------------------
# Three DSN passes of June 7-8, 2018, converted from TRK-2-34; the flight reconstruction (OD044,
# ``orex_flight``) is the reference. Chapter 9 builds and edits them on the page; chapters 10 and
# 11 rebuild them with ``real_tracking`` and apply chapter 9's editing with ``load_editing``.
REAL_DATA_KERNELS = "real_data.tm"  # loaded by chapters 9-11, on top of KERNELS
REAL_DATA_DIR = Path("../supplementary/supp_data/measurements")
MEDIA_DIR = REAL_DATA_DIR / "media_correction"
JUNE_PASSES = {  # pass label -> tracking file
    "DSS-65 June 7": "orex_beno_2018_158_184515_2018_158_230501_65.json",
    "DSS-35 June 8": "orex_beno_2018_159_054533_2018_159_104001_35.json",
    "DSS-65 June 8 PM": "orex_beno_2018_159_183015_2018_159_230501_65.json",
}
real_stations = {dss: scb.GroundStation(dss) for dss in ["DSS-65", "DSS-35"]}

# OSIRIS-REx transponder delay, 1649.9 ns (estimated by the navigation team), in range units:
# 1649.9e-9 s x the X-band range unit (0.1475) x its reference frequency (8445.77 MHz)
TRANSPONDER_DELAY_RU = 2055.8


def real_tracking(passes=tuple(JUNE_PASSES), bias=None, media=True, **extra):
    """Doppler and ranging of the given passes, read with ``MeasurementSpec.build`` (chapter 9).

    ``bias`` sets a fixed measurement bias per dataset name; ``media=False`` leaves out the media
    corrections; ``extra`` goes to the model constructors (e.g. ``electronic_delays=False``).
    Datasets are named "<pass> Doppler" and "<pass> ranging".
    """
    config = scb.MeasurementConfig(
        media_folder=str(MEDIA_DIR) if media else None,
        tropo_seasonal_file_path=str(MEDIA_DIR / "1972_001_2048_001_tro_modified.json"),
        sc_transponder_delay_RU=TRANSPONDER_DELAY_RU,
        sigma={"Doppler": scb.ArrayWUnits(1e-3, 1 / sec), "SeqRange": scb.ArrayWUnits(0.5, km / km)},
        bias=bias or {},
        extra=extra,
    )
    streams = []
    for label in passes:
        file, station = str(REAL_DATA_DIR / "radiometric" / JUNE_PASSES[label]), real_stations[label[:6]]
        streams += [
            scb.MeasurementStream(file, "Doppler", station, name=f"{label} Doppler"),
            scb.MeasurementStream(file, "SeqRange", station, name=f"{label} ranging"),
        ]
    return scb.MeasurementSpec.build(streams, config=config)


def range_unit_m(spec_leaf):
    """One range unit of a ranging dataset as a one-way distance, in meters (chapter 9)."""
    aux = spec_leaf.observed_meas.aux_dict
    seconds = 1.0 / (np.unique(aux["RU"])[0] * np.unique(aux["ul_freq"])[-1])
    return seconds * float(scb.constants.c.values) * 1e3 / 2


def sigma_of(spec_leaf):
    """The σ of a dataset's model as a number, in the units of its observable (Hz or RU)."""
    sigma = spec_leaf.model.sigma
    return float(np.ravel(getattr(sigma, "values", sigma))[0])


def save_editing(spec, path):
    """Write the outlier flags and data weights of every dataset to JSON (end of chapter 9)."""
    record = {
        leaf.dataset_name: {
            "outliers": np.asarray(leaf.observed_meas.meas_outliers, dtype=float).ravel().tolist(),
            "sigma": sigma_of(leaf),
        }
        for leaf in spec.leaves
    }
    Path(path).write_text(json.dumps(record, indent=1))


def load_editing(spec, path):
    """Apply chapter 9's flags and weights to a spec from ``real_tracking`` (chapters 10-11)."""
    record = json.loads(Path(path).read_text())
    for leaf in spec.leaves:
        saved = record[leaf.dataset_name]
        leaf.observed_meas = leaf.observed_meas._replace(meas_outliers=saved["outliers"])
        units = km / km if "ranging" in leaf.dataset_name else 1 / sec
        leaf.model.sigma = scb.ArrayWUnits(saved["sigma"], units)
    return spec


# --- The real-data fit (chapter 10) -------------------------------------------------------------
# Chapter 10 builds this fit on the page; chapter 11 repeats it with ``real_data_fit``.
FIT_PASSES = ("DSS-65 June 7", "DSS-35 June 8")
ARC_START_UTC, ARC_END_UTC = "2018-06-07T17:30:00", "2018-06-08T11:45:00"
PREDICTION_DAYS = 3
PFOGM_BATCH_S, PFOGM_TAU_S = 9 * 3600.0, 86400.0
APRIORI_INFLATION = (1.0, 1e-6)  # km, km/s per axis, added to the simulated-week covariance
BIAS_SIGMA_M = {"DSS-65": 100.0, "DSS-35": 5.0}  # one-way meters
reference_real = scb.Spacecraft("OREx reference (real data)", -64002, mass, area, cr)


def real_arc():
    """Chapter 10's arc: a priori epoch (TDB s), 60 s propagation grid, number of PFOGM batches."""
    t0 = simulated_apriori()[0]
    arc = scb.EpochArray(np.arange(t0, scb.SpiceManager.utc2et(ARC_END_UTC) + 1.0, 60.0), sys="TDB")
    n_batches = int(np.ceil((arc[-1].et - t0 + PREDICTION_DAYS * 86400.0) / PFOGM_BATCH_S))
    return t0, arc, n_batches


def pfogm_forces(spacecraft, n_batches, t0):
    """Chapter 10's force model: the scenario's, plus the PFOGM accelerations (``n_batches=0``: none)."""
    forces = dict(primary_body=spacecraft, third_bodies=planets, cannonball_SRP=True)
    if n_batches:
        forces.update(piecewise_first_order_gauss_markov=True, pfogm_batch_length=PFOGM_BATCH_S,
                      pfogm_n_batches=n_batches, pfogm_beta=np.full(3, 1 / PFOGM_TAU_S), t0=t0)
    return scb.ForceModelTranslation(**forces)


def simulated_apriori(path="outputs/06_first_od_solution/solution.json", utc=ARC_START_UTC):
    """Epoch, state (6) and covariance (6x6) of chapter 6's batch solution at ``utc``."""
    final = json.loads(Path(path).read_text())["final"]
    t = np.asarray(final["timeline_epochs"], dtype=float)
    i = int(np.argmin(np.abs(t - scb.SpiceManager.utc2et(utc))))
    return t[i], np.asarray(final["state_estimated"])[i, :6], np.asarray(final["covariance_full"])[i][:6, :6]


def full_covariance(P, units, epoch):
    """A ``CovarianceMatrix`` from a full matrix (numbers) and the unit of each state element."""
    sigma = np.sqrt(np.diag(P))
    corr = P / np.outer(sigma, sigma)
    n = len(sigma)
    return scb.CovarianceMatrix(
        [scb.ArrayWUnits(s, u) for s, u in zip(sigma, units)], epoch, from_list=True,
        corr_factors=[scb.ArrayWUnits(corr[i, j], None) for i in range(n) for j in range(i + 1, n)],
    )


def real_data_fit(tracking, pfogm_sigma=5e-12, bias_sigma_m=BIAS_SIGMA_M, traj_dir="outputs", tag="fit"):
    """Chapter 10's SRIFB fit of ``tracking``; returns the filter and the solution.

    ``pfogm_sigma`` [km/s^2] (None: no stochastic accelerations) and the range-bias a-priori σ per
    station (one-way meters) can be changed; everything else is chapter 10's setup.
    """
    _, x0, P_sim = simulated_apriori()
    t0, arc, n_batches = real_arc()
    ru_m = {leaf.dataset_name[:6]: range_unit_m(leaf) for leaf in tracking.leaves if "ranging" in leaf.dataset_name}

    state = (scb.StateDefinition()
             .position(reference_real, scb.ArrayWFrame(scb.ArrayWUnits(x0[:3], km), J2000))
             .velocity(reference_real, scb.ArrayWFrame(scb.ArrayWUnits(x0[3:], km / sec), J2000)))
    units, extra = [km] * 3 + [km / sec] * 3, []
    if pfogm_sigma:
        state = state.param("a_pfogm", reference_real,
                            scb.ArrayWFrame(scb.ArrayWUnits(np.zeros(3 * n_batches), km / sec**2), J2000),
                            estimation="estimated", dynamics="dynamic")
        units += [km / sec**2] * (3 * n_batches)
        extra += [pfogm_sigma] * (3 * n_batches)
    state = state.param("eta_srp", reference_real, scb.ArrayWFrame(scb.ArrayWUnits([1.0], None), J2000),
                        estimation="estimated", dynamics="static")
    units.append(None)
    extra.append(0.5)
    for dss, station in real_stations.items():
        state = state.param(f"range_bias_RU_{dss.replace('-', '')}", station,
                            scb.ArrayWFrame(scb.ArrayWUnits([0.0], None), J2000),
                            estimation="estimated", dynamics="static")
        units.append(None)
        extra.append(bias_sigma_m[dss] / ru_m[dss])

    P0 = np.zeros((6 + len(extra), 6 + len(extra)))
    P0[:6, :6] = P_sim + np.diag([APRIORI_INFLATION[0] ** 2] * 3 + [APRIORI_INFLATION[1] ** 2] * 3)
    P0[6:, 6:] = np.diag(np.square(extra))

    propagator = scb.Propagator(primary_body=reference_real,
                                state_vector=scb.StateArray(epoch=arc[0], origin=sun, state=state), tspan=arc,
                                force_models=pfogm_forces(reference_real, n_batches if pfogm_sigma else 0, arc[0]),
                                display_progress=False)
    srifb = scb.SRIFB(propagator=propagator, measurements=tracking,
                      settings=scb.FilterSettings(initial_covariance=full_covariance(P0, units, arc[0])),
                      traj_name=f"{tag}.bsp", traj_dir=str(traj_dir))
    solution, _, _ = srifb.fit(traj_name=f"{tag}.bsp", traj_dir=str(traj_dir), convergence_threshold_cost=1e-2)
    return srifb, solution
