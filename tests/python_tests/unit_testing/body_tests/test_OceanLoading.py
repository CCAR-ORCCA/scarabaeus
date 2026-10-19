# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""Ocean-loading station displacement.

The astronomical arguments are checked against the independent formulation of
the IERS routine ARG (Newcomb's mean longitudes and the constituent speeds),
the UT1 read from an Earth-fixed frame against the IERS Earth rotation angle,
the nodal corrections against Schureman's closed forms, and the BLQ sign
conventions against single-constituent cases.
"""
import numpy as np
import pytest
import spiceypy as spice
import scarabaeus as scb
from scarabaeus.utils.Constants import Constants

TIDES = Constants.EARTH_OCEAN_LOADING.tides
# IERS ARG: multipliers of (h0, s0, p0, 2 pi) and angular speeds in rad/s.
ARG = {"M2": ((2, -2, 0, 0), 1.40519e-4), "S2": ((0, 0, 0, 0), 1.45444e-4),
       "N2": ((2, -3, 1, 0), 1.37880e-4), "K2": ((2, 0, 0, 0), 1.45842e-4),
       "K1": ((1, 0, 0, .25), .72921e-4), "O1": ((1, -2, 0, -.25), .67598e-4),
       "P1": ((-1, 0, 0, -.25), .72523e-4), "Q1": ((1, -3, 1, -.25), .64959e-4),
       "Mf": ((0, 2, 0, 0), .053234e-4), "Mm": ((0, 1, -1, 0), .026392e-4),
       "Ssa": ((2, 0, 0, 0), .003982e-4)}

BLQ = """$$ Ocean loading displacement
$$
$$ END HEADER
$$
  DSS 65
$$ test block
  .00100 .00200 .00300 .00400 .00500 .00600 .00700 .00800 .00900 .01000 .01100
  .00010 .00020 .00030 .00040 .00050 .00060 .00070 .00080 .00090 .00100 .00110
  .00001 .00002 .00003 .00004 .00005 .00006 .00007 .00008 .00009 .00010 .00011
  -64.7  -52.0  -96.2  -55.2  -58.8 -151.4  -65.6 -138.1    8.4    5.2    2.1
   85.5  114.5   56.5  113.6   99.4   19.1   94.1  -10.4 -167.4 -170.0 -177.7
  109.5  147.0   92.7  148.8   45.2  -28.3   36.7  -55.9   93.7  107.8  128.7
$$
  OTHER
  0 0 0 0 0 0 0 0 0 0 0
"""


@pytest.fixture(scope="module", autouse=True)
def kernels():
    spice.kclear()
    spice.furnsh("data/kernels/locked/lsk/naif0012.tls")
    yield
    spice.kclear()


@pytest.mark.parametrize("date", ["2000 JAN 01 00:00:00 UTC", "2026 MAR 14 00:00:00 UTC",
                                  "2026 MAR 14 07:30:00 UTC", "2031 SEP 02 18:00:00 UTC"])
def test_arguments_match_the_iers_arg_routine(date):
    epoch = spice.str2et(date)
    julian_date = float(spice.et2utc(epoch, "J", 10).split()[1])
    fraction = (julian_date - 0.5) % 1.0
    t = (julian_date - fraction - 2415020.0) / 36525.0
    h0 = 279.69668 + 36000.768930485 * t + 3.03e-4 * t**2
    s0 = 270.434358 + 481267.88314137 * t - 0.001133 * t**2 + 1.9e-6 * t**3
    p0 = 334.329653 + 4069.0340329577 * t - 0.010325 * t**2 - 1.2e-5 * t**3
    ours = scb.OceanLoading.arguments(epoch)
    for index, name in enumerate(TIDES):
        factors, speed = ARG[name]
        reference = (np.degrees(speed * fraction * 86400.0) + factors[0] * h0
                     + factors[1] * s0 + factors[2] * p0 + factors[3] * 360.0)
        # ARG's speeds are rounded to six digits, worth a few 0.01 deg a day.
        assert abs((ours[index] - reference + 180.0) % 360.0 - 180.0) < 0.1, name


def test_from_blq_reads_the_named_station(tmp_path):
    path = tmp_path / "stations.blq"
    path.write_text(BLQ)
    loading = scb.OceanLoading.from_blq(path, "dss  65")
    assert loading.amplitudes[0][0] == 0.001 and loading.amplitudes[2][10] == 0.00011
    assert loading.phases[1][3] == 113.6 and loading.phases[2][0] == 109.5
    assert loading.rate_step == 60.0
    assert scb.OceanLoading.from_blq(path, "DSS 65", rate_step=5.0).rate_step == 5.0
    with pytest.raises(scb.OceanLoading.StationNotFoundError):
        scb.OceanLoading.from_blq(path, "DSS 14")
    with pytest.raises(scb.OceanLoading.StationNotFoundError):
        scb.OceanLoading.from_blq(path, "OTHER")


@pytest.mark.parametrize("row, axis, sign", [(0, "up", 1.0), (1, "east", -1.0), (2, "north", -1.0)])
def test_blq_components_are_radial_west_south(row, axis, sign):
    """A single M2 term of 1 cm at zero phase lag, in one BLQ row at a time,
    moves the station along +up, -east (west) or -north (south)."""
    amplitudes = np.zeros((3, len(TIDES)))
    amplitudes[row, 0] = 0.01
    loading = scb.OceanLoading(amplitudes=amplitudes, phases=np.zeros((3, len(TIDES))))
    lat, lon = np.radians(35.4), np.radians(-116.9)
    position = 6371.0 * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    up = position / 6371.0
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    axes = {"up": up, "east": east, "north": np.cross(up, east)}
    epoch = 8.0e8
    factor, angle = scb.OceanLoading.nodal_corrections(epoch)
    argument = np.radians(scb.OceanLoading.arguments(epoch)[0] + angle[0])
    expected = sign * 1e-5 * factor[0] * np.cos(argument)
    state = loading.state_displacement(position, epoch).values
    np.testing.assert_allclose(state[:3], expected * axes[axis], rtol=0, atol=1e-15)
    # Rate: M2 turns at its speed, so the analytic rate is -f A w sin(arg + u).
    rate = -sign * 1e-5 * factor[0] * 1.405189e-4 * np.sin(argument)
    np.testing.assert_allclose(state[3:], rate * axes[axis], rtol=0, atol=1e-12)


def test_earth_fixed_frame_supplies_ut1():
    """ITRF93's rotation gives the tabulated UT1 - UTC through the IERS Earth
    rotation angle, and the frame-based arguments carry the IERS 2010 GMST:
    K1 is tau + s + 90 = GMST + 270 deg and K2 is 2 tau + 2 s = 2 GMST."""
    spice.furnsh("data/kernels/locked/spk/earth_200101_990628_predict.bpc")
    epoch = spice.str2et("2020 JAN 01 00:00:00 UTC")
    ut1_minus_utc = -0.1772  # IERS Bulletin B, 2020 January 1, s
    days = float(spice.et2utc(epoch, "J", 10).split()[1]) + ut1_minus_utc / 86400.0 - 2451545.0
    era = 360.0 * ((0.7790572732640 + 1.00273781191135448 * days) % 1.0)
    ours = np.degrees(scb.OceanLoading.earth_rotation_angle(epoch, "ITRF93"))
    # 1 ms of UT1 is 0.0042 deg/s * 1e-3 s of rotation.
    assert abs((ours - era + 180.0) % 360.0 - 180.0) < 4.2e-6

    t = spice.unitim(epoch, "TDB", "TDT") / (36525.0 * 86400.0)
    gmst = era + (0.014506 + 4612.156534 * t + 1.3915817 * t**2 - 0.00000044 * t**3
                  - 0.000029956 * t**4 - 0.0000000368 * t**5) / 3600.0
    framed = scb.OceanLoading.arguments(epoch, "ITRF93")
    utc = scb.OceanLoading.arguments(epoch)
    k1, k2 = TIDES.index("K1"), TIDES.index("K2")
    wrap = lambda angle: (angle + 180.0) % 360.0 - 180.0
    assert abs(wrap(framed[k1] - gmst - 270.0)) < 1e-5
    assert abs(wrap(framed[k2] - 2.0 * gmst)) < 2e-5
    # UTC for UT1 is off by more than a millidegree; the long-period terms
    # have no tau and do not see the frame.
    assert abs(wrap(utc[k1] - gmst - 270.0)) > 1e-3
    for name in ("Mf", "Mm", "Ssa"):
        assert framed[TIDES.index(name)] == utc[TIDES.index(name)]


def test_nodal_corrections_match_schureman():
    """The series in the Moon's node N agree with Schureman's closed forms in
    the inclination I of the lunar orbit to the equator, built here from the
    spherical triangle of the ecliptic, equator and lunar orbit."""
    obliquity, inclination = np.radians(23.452), np.radians(5.145)
    pole = np.array([0.0, -np.sin(obliquity), np.cos(obliquity)])  # ecliptic pole
    equinox = np.array([1.0, 0.0, 0.0])
    in_ecliptic = np.cross(pole, equinox)
    for epoch in np.linspace(0.0, 19.0 * 365.25 * 86400.0, 39):
        t = epoch / (36525.0 * 86400.0)
        node = np.radians(125.04452 - 1934.136261 * t)
        lunar_node = np.cos(node) * equinox + np.sin(node) * in_ecliptic
        lunar_pole = pole * np.cos(inclination) + np.cross(lunar_node, pole) * np.sin(inclination)
        crossing = np.cross([0.0, 0.0, 1.0], lunar_pole)  # lunar orbit's equator crossing
        crossing /= np.linalg.norm(crossing)
        i_eq = np.arccos(lunar_pole[2])
        nu = np.arctan2(crossing[1], crossing[0])
        xi = node - np.arctan2(lunar_pole @ np.cross(crossing, lunar_node), crossing @ lunar_node)
        exact = {"M2": (np.cos(i_eq / 2)**4 / 0.9154, 2 * xi - 2 * nu),
                 "O1": (np.sin(i_eq) * np.cos(i_eq / 2)**2 / 0.3800, 2 * xi - nu),
                 "Mf": (np.sin(i_eq)**2 / 0.1578, -2 * xi),
                 "Mm": ((2 / 3 - np.sin(i_eq)**2) / 0.5021, 0.0)}
        factor, angle = scb.OceanLoading.nodal_corrections(epoch)
        for name, (f, u) in exact.items():
            index = TIDES.index(name)
            assert abs(factor[index] - f) < 1e-3, name
            assert abs((angle[index] - np.degrees(u) + 180.0) % 360.0 - 180.0) < 0.1, name
        for name in ("S2", "P1", "Ssa"):
            assert factor[TIDES.index(name)] == 1.0 and angle[TIDES.index(name)] == 0.0


def test_rejects_malformed_coefficients():
    with pytest.raises(scb.OceanLoading.CoefficientShapeError):
        scb.OceanLoading(amplitudes=np.zeros((3, 10)), phases=np.zeros((3, 11)))
    with pytest.raises(scb.OceanLoading.CoefficientShapeError):
        scb.OceanLoading(amplitudes=np.full((3, 11), np.nan), phases=np.zeros((3, 11)))


def test_ground_station_sums_several_models():
    """A list of models moves the station by their sum; Love partials come
    from the solid and pole tides inside it, ocean loading adds none."""
    spice.furnsh("data/kernels/locked/pck/pck00010.tpc")
    spice.furnsh("data/kernels/locked/spk/de432s.bsp")
    point = np.array([4849.3, -360.4, 4114.8])
    epoch = 1.0e6
    amplitudes = np.full((3, len(TIDES)), 0.002)
    phases = np.zeros((3, len(TIDES)))
    models = [scb.SolidTide(), scb.PoleTide(pole_x=1e-6, pole_y=2e-6),
              scb.OceanLoading(amplitudes=amplitudes, phases=phases)]
    station = scb.GroundStation("LOAD_GS", 399096, body_fixed_frame="IAU_EARTH",
                                body_fixed_position=point)
    station.surface_displacement = models
    assert station.surface_displacement is models
    expected = np.r_[point, np.zeros(3)] + sum(
        model.state_displacement(point, epoch, "IAU_EARTH").values for model in models)
    np.testing.assert_allclose(station.nominal_state(epoch, "IAU_EARTH").quantity.values, expected,
                               rtol=0, atol=1e-12)

    alone = scb.GroundStation("LOAD_GS2", 399095, body_fixed_frame="IAU_EARTH",
                              body_fixed_position=point)
    alone.surface_displacement = models[:2]
    np.testing.assert_allclose(station.tide_partials(epoch, "J2000").values,
                               alone.tide_partials(epoch, "J2000").values, rtol=0, atol=1e-15)
    with pytest.raises(TypeError, match="state_displacement"):
        station.surface_displacement = [models[0], object()]
