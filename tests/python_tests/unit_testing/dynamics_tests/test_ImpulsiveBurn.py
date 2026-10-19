# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==========================================#
#  Unit Tests for the ImpulsiveBurn class  #
# ==========================================#
import src.scarabaeus as scb
import pytest
import numpy as np

km, sec = scb.Units.get_units(["km", "sec"])
J2000 = scb.Frame("J2000")


# --------------------#
# region    Fixtures #
# --------------------#
@pytest.fixture
def delta_v_awf():
    """Delta-v as ArrayWFrame."""
    return scb.ArrayWFrame(np.array([0.1, 0.0, 0.0]), km / sec, J2000)


@pytest.fixture
def dv_magnitude():
    """Scalar delta-v magnitude."""
    return scb.ArrayWUnits(np.array([0.1]), km / sec)


@pytest.fixture
def dv_unit_vector():
    """Unit vector for delta-v direction."""
    return scb.ArrayWUnits(np.array([1.0, 0.0, 0.0]), None)


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
class TestInitialization:
    def test_from_delta_v(self, delta_v_awf):
        """
        Verifies that object is constructed correctly from an ArrayWFrame delta-v.
        """
        burn = scb.ImpulsiveBurn(delta_v=delta_v_awf)

        assert burn._t_burn == 0
        assert isinstance(burn._delta_v, scb.ArrayWUnits)

    def test_from_magnitude_and_unit_vector(self, dv_magnitude, dv_unit_vector):
        """
        Verifies that object is constructed correctly from magnitude and unit vector.
        """
        burn = scb.ImpulsiveBurn(dv_magnitude=dv_magnitude, dv_unit_vector=dv_unit_vector)

        assert burn._dv_magnitude is dv_magnitude
        assert burn._dv_unit_vector is dv_unit_vector
        assert burn._t_burn == 0

    def test_raises_without_input(self):
        """
        Verifies that a ValueError is raised when neither construction path is provided.
        """
        with pytest.raises(ValueError):
            scb.ImpulsiveBurn()


def test_t_burn_property(delta_v_awf):
    """
    Verifies that the t_burn property returns the stored value for both
    the default (0) and a custom burn time.
    """
    burn_default = scb.ImpulsiveBurn(delta_v=delta_v_awf)
    assert burn_default.t_burn == 0

    burn_custom = scb.ImpulsiveBurn(delta_v=delta_v_awf, t_burn=3600.0)
    assert burn_custom.t_burn == pytest.approx(3600.0)


@pytest.mark.parametrize(
    "method, expected_shape",
    [
        ("compute_acceleration", (3, 1)),
        ("_compute_partial_by_dv", (3, 3)),
        ("_compute_partial_by_position", (3, 3)),
    ],
    ids=["Acceleration", "Partial by DV", "Partial by Position"],
)
def test_zero_return_methods(delta_v_awf, method, expected_shape):
    """
    Verifies that compute_acceleration, _compute_partial_by_dv, and
    _compute_partial_by_position each return all-zero arrays of the
    correct shape regardless of input.
    """
    burn = scb.ImpulsiveBurn(delta_v=delta_v_awf)
    pos = np.zeros(3)
    result = getattr(burn, method)(pos, {}, 0.0, None)

    assert result.shape == expected_shape
    assert np.allclose(result, np.zeros(expected_shape))


class TestTcmCovariance:
    def test_covariance_gates(self, delta_v_awf):
        """
        Verifies that the TCM covariance gate matrix has the correct shape.
        """
        burn = scb.ImpulsiveBurn(delta_v=delta_v_awf)
        covar = burn.compute_TCM_covariance_gates()

        assert isinstance(covar, scb.ArrayWUnits)
        assert covar.values.shape == (3, 3)

    def test_simplified_covariance(self, dv_magnitude, dv_unit_vector):
        """
        Verifies that the simplified TCM covariance matrix has the correct shape.
        """
        burn = scb.ImpulsiveBurn(dv_magnitude=dv_magnitude, dv_unit_vector=dv_unit_vector)
        covar = burn.compute_TCM_simplified_covariance()

        assert isinstance(covar, scb.ArrayWUnits)
        assert covar.values.shape == (3, 3)


# NOTE: the rest of this class' methods are tested in integration tests