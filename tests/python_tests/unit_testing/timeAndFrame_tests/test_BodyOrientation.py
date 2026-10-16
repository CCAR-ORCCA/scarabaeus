# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ============================================#
#  Unit Tests for the BodyOrientation class  #
# ============================================#
import src.scarabaeus as scb
import numpy as np
import pytest
import spiceypy as spice

BodyOrientation = scb.BodyOrientation

ET = 7.5e8


# --------------#
# region Tests #
# --------------#
class TestIAUAngles:
    @pytest.mark.parametrize("frame", ["IAU_MARS", "IAU_EARTH", "IAU_MOON"])
    def test_spice_path_matches_pxform(self, frame):
        """With a frame and no override the angles are SPICE's full model of
        that frame, nutation terms included (the Moon has them)."""
        angles, _ = BodyOrientation.iau_angles(ET, frame)
        np.testing.assert_allclose(
            BodyOrientation.iau_matrix(angles), spice.pxform(frame, "J2000", ET), atol=1e-13
        )

    @pytest.mark.parametrize("body", ["MARS", "EARTH"])
    def test_secular_path_matches_harmonics_and_spice(self, body):
        """The secular model is the rotation the harmonics field applies, and
        SPICE's frame for bodies with no nutation terms in pck00010."""
        coefficients = BodyOrientation.iau_pole_coefficients(body)
        angles, _ = BodyOrientation.iau_angles(ET, None, coefficients)
        R = BodyOrientation.iau_matrix(angles)

        np.testing.assert_allclose(
            R, scb.SphericalHarmonicsGravity._build_R_b2j_iau(*np.rad2deg(angles)), atol=1e-15
        )
        np.testing.assert_allclose(R, spice.pxform(f"IAU_{body}", "J2000", ET), atol=1e-11)

    @pytest.mark.parametrize(
        "frame, overrides",
        [
            pytest.param("IAU_MOON", {}, id="spice"),
            pytest.param(None, {}, id="secular"),
            pytest.param(None, {"pole_pm": 350.9}, id="pm"),
            pytest.param(None, {"pole_ra": 0.3, "pole_w0": 1.0}, id="ra-w0"),
        ],
    )
    def test_rates_match_finite_differences(self, frame, overrides):
        """The returned rates are the time derivatives of the returned angles."""
        coefficients = BodyOrientation.iau_pole_coefficients(
            "MOON" if frame else "MARS"
        )
        step = 10.0
        plus, _ = BodyOrientation.iau_angles(ET + step, frame, coefficients, **overrides)
        minus, _ = BodyOrientation.iau_angles(ET - step, frame, coefficients, **overrides)
        _, rates = BodyOrientation.iau_angles(ET, frame, coefficients, **overrides)

        np.testing.assert_allclose(rates, (plus - minus) / (2 * step), rtol=1e-6, atol=1e-16)

    def test_overrides_replace_their_term(self):
        """An estimated value replaces its term of the secular model: RA/DEC
        become constant, PM drops the quadratic term and W0 the constant."""
        coefficients = BodyOrientation.iau_pole_coefficients("MARS")
        days = ET / 86400.0
        angles, rates = BodyOrientation.iau_angles(
            ET, "IAU_MARS", coefficients,
            pole_ra=0.3, pole_dec=0.9, pole_pm=350.9, pole_w0=1.0,
        )

        assert angles[0] == 0.3 and angles[1] == 0.9
        assert rates[0] == 0.0 and rates[1] == 0.0
        assert angles[2] == pytest.approx(1.0 + np.deg2rad(350.9 * days), rel=1e-15)
        assert rates[2] == pytest.approx(np.deg2rad(350.9) / 86400.0, rel=1e-15)

        # W0 alone moves W by exactly the change in W0
        base, _ = BodyOrientation.iau_angles(ET, None, coefficients)
        shifted, _ = BodyOrientation.iau_angles(
            ET, None, coefficients, pole_w0=np.deg2rad(coefficients[2][0]) + 0.25
        )
        np.testing.assert_allclose(shifted - base, [0.0, 0.0, 0.25], atol=1e-9)

    def test_coefficients_come_from_the_kernel_pool(self):
        """Rows are POLE_RA, POLE_DEC and PM as the PCK holds them."""
        table = BodyOrientation.iau_pole_coefficients("MARS")
        assert table.shape == (3, 3)
        for row, item in zip(table, ("POLE_RA", "POLE_DEC", "PM")):
            np.testing.assert_array_equal(row, spice.bodvrd("MARS", item, 3)[1])

    @pytest.mark.parametrize(
        "frame, overrides",
        [(None, {}), ("IAU_MARS", {"pole_w0": 1.0})],
        ids=["bare-body", "override"],
    )
    def test_missing_coefficients_reported(self, frame, overrides):
        """The secular path needs the PCK constants."""
        with pytest.raises(BodyOrientation.MissingPoleCoefficientsError):
            BodyOrientation.iau_angles(ET, frame, None, **overrides)


# endregion Tests #
# --------------#
