# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""Absolute EOP signs, time scales, PCK rotation/velocity, and precedence."""

from dataclasses import replace

import numpy as np
import pytest
import spiceypy as spice

from scarabaeus import BodyOrientation
from scarabaeus import constants


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Override the suite's large scenario setup; only time constants are needed."""
    spice.kclear()
    spice.pdpool("DELTET/DELTA_T_A", [32.184])
    spice.pdpool("DELTET/K", [1.657e-3])
    spice.pdpool("DELTET/EB", [1.671e-2])
    spice.pdpool("DELTET/M", [6.239996, 1.99096871e-7])
    yield
    spice.kclear()


@pytest.fixture(params=["IAU1976/1980", "IAU2006/2000A", "legacy_with_nutation"])
def model(request):
    with_nutation = request.param == "legacy_with_nutation"
    return BodyOrientation.earth_eop(np.deg2rad(6.4125e-5),
                                  np.deg2rad(1.341694444444444e-4),
                                  37.00446633384733,
                                  model="IAU1976/1980" if with_nutation else request.param,
                                  nutation_delta_psi=np.deg2rad(-0.11323/3600) if with_nutation else 0,
                                  nutation_delta_epsilon=np.deg2rad(-0.00772/3600) if with_nutation else 0)


EPOCH = 925128069.1855


def test_ut1_offset_has_the_correct_sign(model):
    """An increased TAI-UT1 means less Earth rotation at the same TDB epoch."""
    no_pole = replace(model, pole_x=0, pole_y=0)
    original = no_pole.rotation_matrix(EPOCH)
    earlier = replace(no_pole, tai_minus_ut1=model.tai_minus_ut1 + 1).rotation_matrix(EPOCH)
    change = earlier @ original.T
    np.testing.assert_allclose(change, spice.rotate(-7.2921158553e-5, 3), atol=3e-11)


@pytest.mark.parametrize("parameter,axis", [("pole_x", 2), ("pole_y", 1)])
def test_absolute_pole_coordinates_rotate_about_the_correct_axes(model, parameter, axis):
    """An xp/yp change is a pole tilt, not a rotation about the spin axis."""
    no_pole = replace(model, pole_x=0, pole_y=0)
    change = replace(no_pole, **{parameter: 1e-4}).rotation_matrix(EPOCH)
    relative = change @ no_pole.rotation_matrix(EPOCH).T
    np.testing.assert_allclose(relative, spice.rotate(-1e-4, axis), atol=5e-16)


def test_time_conversion_uses_tt_for_nutation_and_ut1_for_spin(model, monkeypatch):
    """Check the time-scale wiring independently of PCK serialization."""
    import erfa

    tt_s = 86400.0 * 10000 + 12345.6789
    tai_s = tt_s - 32.184
    calls = []
    def convert(epoch, source, target):
        calls.append((epoch, source, target))
        return {"TDT": tt_s, "TAI": tai_s}[target]
    monkeypatch.setattr(spice, "unitim", convert)
    tt1, tt2 = 2461545.0, 12345.6789 / 86400.0
    ut11, ut12 = 2461545.0, (12345.6789 - 32.184 - model.tai_minus_ut1) / 86400.0
    if model.model == "IAU1976/1980":
        dpsi, deps = erfa.nut80(tt1, tt2)
        eps = erfa.obl80(tt1, tt2)
        pn = erfa.numat(eps, dpsi + model.nutation_delta_psi,
                        deps + model.nutation_delta_epsilon) @ erfa.pmat76(tt1, tt2)
        expected = (spice.rotate(-model.pole_y, 1) @ spice.rotate(-model.pole_x, 2)
                    @ spice.rotate(erfa.gmst82(ut11, ut12)
                                   + (dpsi + model.nutation_delta_psi)*np.cos(eps), 3) @ pn)
    else:
        expected = erfa.c2t06a(tt1, tt2, ut11, ut12, model.pole_x, model.pole_y)
    np.testing.assert_allclose(model.rotation_matrix(EPOCH), expected, atol=1e-11)
    assert calls == [(EPOCH, "TDB", "TDT"), (EPOCH, "TDB", "TAI")]


def test_written_pck_matches_direct_orientation_and_velocity(model, tmp_path):
    """Round-trip through SPICE at new sample times, including record joins."""
    start, stop = EPOCH, EPOCH + 2 * 86400.0
    path = model.write_pck(tmp_path / "earth.bpc", start, stop)
    spice.furnsh(str(path))
    try:
        epochs = np.unique(np.r_[np.linspace(start, stop, 217),
                                 start + np.arange(9) * 21600.0])
        for epoch in epochs:
            rotation = spice.pxform("J2000", "ITRF93", float(epoch))
            np.testing.assert_allclose(rotation, model.rotation_matrix(epoch), atol=5e-11, rtol=0)
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-15)
        # A five-point derivative has a different construction than the PCK's
        # analytic Chebyshev derivative and catches missing station velocity.
        h = 60.0
        for epoch in (start + 1234.5, start + 21600.0, stop - 1000.0):
            matrices = model.rotation_matrix(epoch + h * np.array([-2., -1., 1., 2.]))
            derivative = (matrices[0] - 8*matrices[1] + 8*matrices[2] - matrices[3]) / (12*h)
            transform = spice.sxform("J2000", "ITRF93", float(epoch))
            np.testing.assert_allclose(transform[3:, :3], derivative, atol=8e-13, rtol=0)
            np.testing.assert_allclose(transform[3:, 3:], transform[:3, :3], atol=0)
        with pytest.raises(spice.utils.exceptions.SpiceyError):
            spice.pxform("J2000", "ITRF93", start - 1)
    finally:
        spice.unload(str(path))


def test_absolute_pck_overrides_prediction_without_double_correction(model, tmp_path):
    """The final frame follows absolute inputs regardless of a competing PCK."""
    baseline = replace(model, pole_x=-5e-4, tai_minus_ut1=38.0)
    base_path = baseline.write_pck(tmp_path / "prediction.bpc", EPOCH, EPOCH+1000)
    absolute_path = model.write_pck(tmp_path / "supplied.bpc", EPOCH, EPOCH+1000)
    spice.furnsh(str(base_path))
    spice.furnsh(str(absolute_path))
    try:
        epoch = EPOCH + 222.2
        np.testing.assert_allclose(spice.pxform("J2000", "ITRF93", epoch),
                                   model.rotation_matrix(epoch), atol=5e-11, rtol=0)
        spice.unload(str(absolute_path))
        np.testing.assert_allclose(spice.pxform("J2000", "ITRF93", epoch),
                                   baseline.rotation_matrix(epoch), atol=5e-11, rtol=0)
    finally:
        spice.unload(str(absolute_path))
        spice.unload(str(base_path))


def test_invalid_inputs_and_existing_files_are_rejected(model, tmp_path):
    with pytest.raises(ValueError, match="finite"):
        replace(model, tai_minus_ut1=float("nan"))
    with pytest.raises(ValueError, match="stop > start"):
        model.write_pck(tmp_path / "bad.bpc", 1, 0)
    path = tmp_path / "existing.bpc"
    path.write_bytes(b"preserve this file")
    with pytest.raises(FileExistsError):
        model.write_pck(path, 0, 1)
    assert path.read_bytes() == b"preserve this file"


def test_celestial_corrections_change_q_without_changing_terrestrial_w_or_spin():
    """Regression: CPO dX/dY must not be treated as terrestrial xp/yp."""
    import erfa

    model = BodyOrientation.earth_eop(1e-6, 2e-6, 37, model="IAU2006/2000A")
    epochs = EPOCH + np.array([[0, 3600], [86400, 90000]])
    w, r, q = model.rotation_components(epochs)
    corrected = replace(model, celestial_pole_dx=2e-7, celestial_pole_dy=-3e-7)
    w2, r2, q2 = corrected.rotation_components(epochs)
    np.testing.assert_array_equal(w2, w)
    np.testing.assert_array_equal(r2, r)
    np.testing.assert_allclose(q2[..., 2, :2] - q[..., 2, :2],
                               np.broadcast_to([2e-7, -3e-7], (2, 2, 2)), atol=5e-16)
    # Check the zero-CPO matrix by the equinox path, independently of W@ERA@Q.
    tt = spice.unitim(EPOCH, "TDB", "TDT")
    tt1, tt2 = model._jd_pair(tt)
    ut11, ut12 = model._jd_pair(spice.unitim(EPOCH, "TDB", "TAI") - 37)
    np.testing.assert_allclose(model.rotation_matrix(EPOCH),
        erfa.c2teqx(erfa.pnm06a(tt1, tt2), erfa.gst06a(ut11, ut12, tt1, tt2), w[0, 0]),
        atol=1e-15, rtol=0)
    with pytest.raises(ValueError, match="not legacy"):
        replace(corrected, model="IAU1976/1980")
    with pytest.raises(ValueError, match="model must"):
        replace(model, model="unknown")
    with pytest.raises(ValueError, match="corrected celestial pole"):
        replace(model, celestial_pole_dx=2).rotation_matrix(EPOCH)


def test_modern_orientation_matches_published_sofa_reference():
    """SOFA/ERFA t_c2t06a at TT=UT1=MJD 53736; independent fixed values.

    Source: https://github.com/liberfa/erfa/blob/master/src/t_erfa_c.c
    Its artificial TT=UT1 requires TAI-UT1=-32.184 s for our time-scale API.
    """
    model = BodyOrientation.earth_eop(2.55060238e-7, 1.860359247e-6, -32.184,
                                   model="IAU2006/2000A")
    tt = (2400000.5 + 53736.0 - 2451545.0) * 86400.0
    epoch = spice.unitim(tt, "TDT", "TDB")
    expected = np.array([
        [-0.1810332128305897282, 0.9834769806938592296, 0.6555550962998436505e-4],
        [-0.9834768134136214897, -0.1810332203649130832, 0.5749800844905594110e-3],
        [0.5773474024748545878e-3, 0.3961816829632690581e-4, 0.9999998325501747785],
    ])
    np.testing.assert_allclose(model.rotation_matrix(epoch), expected, atol=1e-12, rtol=0)


def test_legacy_nutation_longitude_also_corrects_sidereal_rotation():
    """Changing dPsi changes both Q and the equation of the equinoxes in R."""
    import erfa

    model = BodyOrientation.earth_eop(1e-6, 2e-6, 37)
    corrected = replace(model, nutation_delta_psi=-5e-7, nutation_delta_epsilon=-4e-8)
    w, r, q = model.rotation_components(EPOCH)
    wc, rc, qc = corrected.rotation_components(EPOCH)
    tt = spice.unitim(EPOCH, "TDB", "TDT")
    eps = erfa.obl80(*model._jd_pair(tt))
    np.testing.assert_array_equal(wc, w)
    np.testing.assert_allclose(rc @ r.T, spice.rotate(-5e-7*np.cos(eps), 3), atol=5e-16, rtol=0)
    assert np.linalg.norm(qc - q) > 5e-7
    with pytest.raises(ValueError, match="not modern"):
        replace(corrected, model="IAU2006/2000A")


@pytest.mark.parametrize("parameter,delta", [("pole_x", 2e-7), ("pole_y", -3e-7),
                                           ("ut1", 0.004)])
def test_delta_api_matches_absolute_station_position_and_velocity(model, parameter, delta, tmp_path):
    """Physical deltas must reproduce the absolute model, including polar signs."""
    inputs = ({"tai_minus_ut1": model.tai_minus_ut1 - delta} if parameter == "ut1"
              else {parameter: getattr(model, parameter) + delta})
    intended = replace(model, **inputs)
    baseline_path = model.write_pck(tmp_path / "base.bpc", EPOCH, EPOCH+1000)
    intended_path = intended.write_pck(tmp_path / "intended.bpc", EPOCH, EPOCH+1000)
    epoch = EPOCH + 456.7
    station = np.array([4000., 2000., 4500., 0., 0., 0.])
    spice.furnsh(str(baseline_path))
    try:
        baseline = spice.sxform("ITRF93", "J2000", epoch) @ station
        correction = BodyOrientation.earth_correction(**{"delta_" + parameter: delta}).state_displacement(
            station[:3], epoch).values
        spice.furnsh(str(intended_path))
        exact = spice.sxform("ITRF93", "J2000", epoch) @ station - baseline
        np.testing.assert_allclose(correction[:3], exact[:3], atol=3e-8, rtol=0)
        np.testing.assert_allclose(correction[3:], exact[3:], atol=2e-11, rtol=0)
    finally:
        spice.unload(str(intended_path))
        spice.unload(str(baseline_path))


def test_generic_planet_pck_and_correction(tmp_path):
    """A user planetary law round-trips through a custom class-2 frame."""
    frame_id, class_id = 1888999, 1888998
    spice.pcpool(f"FRAME_{frame_id}_NAME", ["SCB_TEST_PLANET"])
    spice.pipool("FRAME_SCB_TEST_PLANET", [frame_id])
    for key, value in [("CLASS",2),("CLASS_ID",class_id),("CENTER",499)]:
        spice.pipool(f"FRAME_{frame_id}_{key}",[value])
    def law(epochs):
        return np.array([spice.rotate(.3+1e-4*float(t-EPOCH),3) @ spice.rotate(.4,1)
                         for t in np.asarray(epochs).ravel()]).reshape(np.shape(epochs)+(3,3))
    model=BodyOrientation.from_rotation(law,class_id,max_interval=3600.,description="Synthetic Mars law")
    path=model.write_pck(tmp_path/'planet.bpc',EPOCH,EPOCH+7000)
    spice.furnsh(str(path))
    try:
        for t in EPOCH+np.array([11.4,3100.9,6555.5]):
            np.testing.assert_allclose(spice.pxform('J2000','SCB_TEST_PLANET',t),law(t),atol=5e-11,rtol=0)
        correction=BodyOrientation([1e-7,2e-7,3e-7],'SCB_TEST_PLANET')
        p,v=np.array([3000.,1200.,100.]),np.array([.001,-.002,.003])
        expected=np.r_[np.cross(correction.rotation_vector,p),np.cross(correction.rotation_vector,v)]
        np.testing.assert_allclose(correction.state_displacement(p,EPOCH,frame_to='SCB_TEST_PLANET',velocity_body_fixed=v).values,expected)
    finally:spice.unload(str(path))


def test_orientation_modes_are_explicit(tmp_path):
    with pytest.raises(ValueError,match='exactly one'):
        BodyOrientation([0,0,0],'ITRF93',pole_x=0,pole_y=0,tai_minus_ut1=37)
    with pytest.raises(ValueError,match='absolute'):
        BodyOrientation.earth_correction().write_pck(tmp_path/'invalid.bpc',0,1)
    with pytest.raises(ValueError,match='correction'):
        BodyOrientation.earth_eop(0,0,37).displacement([1,2,3])
    with pytest.raises(ValueError,match='finite'):
        BodyOrientation.earth_correction(delta_ut1=np.nan)


@pytest.mark.parametrize('option,alias,canonical',[
    ('EOP1','ITRF93','IAU1976/1980'), ('EOP2','IERS2010','IAU2006/2000A')])
def test_eop_options_select_the_requested_frame_convention(option,alias,canonical):
    """Public EOP1/EOP2 options select the explicit legacy/modern rotation law."""
    models=[BodyOrientation.earth_eop(1e-6,2e-6,37,model=name)
            for name in (option,alias,canonical)]
    for orientation in models:
        assert orientation.eop_option==option
        assert orientation.model==canonical
        assert alias in orientation.convention
        np.testing.assert_array_equal(orientation.rotation_matrix(EPOCH),models[0].rotation_matrix(EPOCH))


def test_covariance_cases_explicitly_use_eop1():
    # the covariance cases are internal, so the public copy skips this
    c2=pytest.importorskip('internal_dev.VnV_covariance.c2.spec')
    c3=pytest.importorskip('internal_dev.VnV_covariance.c3.spec')
    assert c2.EARTH_ORIENTATION_MODEL==c3.EARTH_ORIENTATION_MODEL=='EOP1'


def test_ut1_time_scale_changes_only_the_spin_and_by_the_tdb_tt_term():
    """"TDB" omits the TDB->TT conversion when forming UT1, and nothing else.

    The two conventions must differ by exactly a rotation about the spin axis
    of omega * (TDB - TT). Anything else leaking in -- a changed celestial
    pole, a changed polar motion -- would mean the option is doing more than
    it claims.
    """
    epoch = 1.0e6
    common = dict(pole_x=1e-6, pole_y=2e-6, tai_minus_ut1=37.0)
    correct = BodyOrientation.earth_eop(**common)
    omitted = BodyOrientation.earth_eop(**common, ut1_time_scale="TDB")

    w_a, r_a, q_a = correct.rotation_components(epoch)
    w_b, r_b, q_b = omitted.rotation_components(epoch)
    # Polar motion and the celestial part are untouched.
    np.testing.assert_allclose(w_a, w_b, atol=0.0)
    np.testing.assert_allclose(q_a, q_b, atol=0.0)

    # The spin differs by omega * (TDB - TT), to the linearisation of a
    # rotation this small.
    tdb_minus_tt = epoch - spice.unitim(epoch, "TDB", "TDT")
    omega = float(constants.earth_rotation_rate.values)
    delta = r_b @ r_a.T
    angle = np.arctan2(delta[0, 1], delta[0, 0])
    assert angle == pytest.approx(omega * tdb_minus_tt, rel=1e-6)
    # Sub-millisecond, but at its annual peak that is half a metre at the
    # equator. This epoch happens to sit near a zero crossing, so bound the
    # amplitude over a year rather than at one arbitrary date.
    peak = max(abs(t - spice.unitim(t, "TDB", "TDT"))
               for t in np.linspace(0.0, 365.25 * 86400.0, 400))
    assert 0.3 < omega * peak * 6378.1366 * 1e3 < 1.2


def test_ut1_time_scale_default_is_the_correct_one_and_validated():
    assert BodyOrientation.earth_eop(1e-6, 2e-6, 37.0).ut1_time_scale == "TAI"
    with pytest.raises(ValueError, match="TAI"):
        BodyOrientation.earth_eop(1e-6, 2e-6, 37.0, ut1_time_scale="TT")


def test_ut1_time_scale_difference_is_annual_not_constant():
    """The offset tracks TDB-TT, so it changes over a year. A constant
    tai_minus_ut1 cannot absorb it, which is why it survives an EOP fit."""
    omega = float(constants.earth_rotation_rate.values)
    angles = []
    for epoch in (0.0, 90 * 86400.0, 180 * 86400.0, 270 * 86400.0):
        tdb_minus_tt = epoch - spice.unitim(epoch, "TDB", "TDT")
        angles.append(omega * tdb_minus_tt)
    assert np.ptp(angles) * 6378.1366 * 1e3 > 0.5     # metres of spread at the equator
