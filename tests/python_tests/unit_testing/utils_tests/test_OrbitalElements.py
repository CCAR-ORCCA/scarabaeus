# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ============================================#
#  Unit Tests for the OrbitalElements class  #
# ============================================#
import src.scarabaeus as scb

import pytest
import numpy as np

# --------------------#
# region    Fixtures #
# --------------------#

# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
@pytest.mark.parametrize(
    "grav_param, r, v, expected",
    [   (   398600.4418,
            np.array([7000, 0, 0]),
            np.array([0, 7.54604911, 0]),
            (6999.99224475, 1.10789209e-06, 0.0, 0.0, np.pi, -np.pi),),
        (   398600.4418,
            np.array([42164.1696, 0, 0]),
            np.array([0, 3.074661212, 0]),
            (42164.20008787, 7.23075324e-07, 0.0, 0.0, 0.0, 0.0),),
        (   398600.4418,
            np.array([42164.1696 / np.sqrt(2), 42164.1696 / np.sqrt(2), 0]),
            np.array([-3.074661212 / np.sqrt(2), 3.074661212 / np.sqrt(2), 0]),
            (42164.20008787, 7.23075324e-7, 0.0, 0.0, 0.78539816, 0.0),),],
            ids=["circular equatorial orbit", "circular geostationary orbit", "circular inclined orbit"],)
def test_rv2coe(grav_param, r, v, expected):
    """Tests the rv2coe method of the OrbitalElements class."""
    # Arrange
    orbital_elements = scb.OrbitalElements()

    # Act
    result = orbital_elements.rv2coe(grav_param, r, v)

    # Assert
    assert np.allclose(
        result, expected, atol=1e-5
    ), f"Expected {expected}, but got {result}"



MU_EARTH = 398600.4418

#: ``(a, ecc, inc, raan, argp, M)`` spanning the regimes the Jacobian must hold
#: in: nearly circular, strongly eccentric, and nearly polar.
ELEMENT_CASES = [
    (7000.0, 0.001, 0.9, 0.4, 1.1, 2.0),
    (24500.0, 0.72, 0.48, 2.2, 0.3, 0.7),
    (8500.0, 0.15, 0.99 * np.pi / 2, 5.0, 4.4, 5.9),
    (12000.0, 0.90, 1.2, 0.1, 2.9, 0.15),
    (7200.0, 1e-5, 0.6, 1.0, 0.5, 3.3),
]
ELEMENT_CASE_IDS = ["LEO low-e", "GTO high-e", "near-polar", "e=0.9", "near-circular"]


def _wrap(angles):
    """Fold angle differences into ``[-pi, pi)`` so a wrap is not a failure."""
    return (np.asarray(angles) + np.pi) % (2 * np.pi) - np.pi


@pytest.mark.parametrize("elements", ELEMENT_CASES, ids=ELEMENT_CASE_IDS)
def test_coe2rv_round_trip(elements):
    """Tests that coe2rv and rv2mean_elements invert each other."""
    # Act
    r, v = scb.OrbitalElements.coe2rv(MU_EARTH, *elements)
    recovered = scb.OrbitalElements.rv2mean_elements(MU_EARTH, r, v)

    # Assert
    difference = recovered - np.asarray(elements)
    difference[2:] = _wrap(difference[2:])
    assert np.allclose(difference, 0.0, atol=1e-9), (
        f"Round trip changed the elements by {difference}"
    )


@pytest.mark.parametrize("ecc", [0.0, 0.01, 0.3, 0.72, 0.95])
@pytest.mark.parametrize("mean_anomaly", [0.0, 0.3, np.pi / 2, 3.0, 6.0])
def test_solve_kepler(ecc, mean_anomaly):
    """Tests that the eccentric anomaly returned satisfies Kepler's equation."""
    # Act
    E = scb.OrbitalElements.solve_kepler(mean_anomaly, ecc)

    # Assert
    residual = _wrap(E - ecc * np.sin(E) - mean_anomaly)
    assert abs(residual) < 1e-11, f"Kepler residual {residual} at e={ecc}"


@pytest.mark.parametrize("ecc", [-0.1, 1.0, 5.0], ids=["negative", "parabolic", "open"])
def test_solve_kepler_rejects_non_elliptic(ecc):
    """Tests that an eccentricity outside [0, 1) is reported."""
    with pytest.raises(scb.OrbitalElements.NonEllipticOrbitError):
        scb.OrbitalElements.solve_kepler(1.0, ecc)


def test_solve_kepler_does_not_converge():
    """Tests that a stalled Newton iteration is reported, not half-solved."""
    with pytest.raises(scb.OrbitalElements.KeplerConvergenceError):
        scb.OrbitalElements.solve_kepler(0.02, 0.999, max_iter=2)


@pytest.mark.parametrize("elements", ELEMENT_CASES, ids=ELEMENT_CASE_IDS)
def test_state_by_elements_jacobian(elements):
    """Tests the closed-form partials against central differences of coe2rv."""
    # Arrange
    analytic = scb.OrbitalElements.state_by_elements_jacobian(MU_EARTH, *elements)
    elements = np.asarray(elements, dtype=float)

    # Act
    numeric = np.zeros((6, 6))
    for column in range(6):
        # Eccentricity is O(1) here while the rest are angles or kilometres, so
        # it needs its own step to stay off the differencing noise floor.
        step = 1e-9 if column == 1 else 1e-6 * max(abs(elements[column]), 1e-3)
        forward, backward = elements.copy(), elements.copy()
        forward[column] += step
        backward[column] -= step
        r_f, v_f = scb.OrbitalElements.coe2rv(MU_EARTH, *forward)
        r_b, v_b = scb.OrbitalElements.coe2rv(MU_EARTH, *backward)
        numeric[:, column] = (
            np.concatenate([r_f, v_f]) - np.concatenate([r_b, v_b])
        ) / (2 * step)

    # Assert
    relative = np.abs(analytic - numeric) / np.maximum(np.abs(numeric), 1e-6)
    row, column = np.unravel_index(relative.argmax(), relative.shape)
    assert relative.max() < 1e-4, (
        f"Worst partial is row {row}, column "
        f"{scb.OrbitalElements.MEAN_ELEMENT_NAMES[column]}: analytic "
        f"{analytic[row, column]} vs numeric {numeric[row, column]}"
    )


@pytest.mark.parametrize("elements", ELEMENT_CASES, ids=ELEMENT_CASE_IDS)
def test_elements_by_state_jacobian(elements):
    """Tests the inverse Jacobian against differences of rv2mean_elements."""
    # Arrange
    r, v = scb.OrbitalElements.coe2rv(MU_EARTH, *elements)
    state = np.concatenate([r, v])
    analytic = scb.OrbitalElements.elements_by_state_jacobian(MU_EARTH, r, v)

    # Act
    numeric = np.zeros((6, 6))
    for column in range(6):
        step = 1e-4 if column < 3 else 1e-7
        forward, backward = state.copy(), state.copy()
        forward[column] += step
        backward[column] -= step
        difference = scb.OrbitalElements.rv2mean_elements(
            MU_EARTH, forward[:3], forward[3:]
        ) - scb.OrbitalElements.rv2mean_elements(MU_EARTH, backward[:3], backward[3:])
        difference[2:] = _wrap(difference[2:])
        numeric[:, column] = difference / (2 * step)

    # Assert
    relative = np.abs(analytic - numeric) / np.maximum(np.abs(numeric), 1e-8)
    assert relative.max() < 1e-4, f"Worst inverse partial off by {relative.max()}"

    forward_jacobian = scb.OrbitalElements.state_by_elements_jacobian(
        MU_EARTH, *elements
    )
    assert np.allclose(analytic @ forward_jacobian, np.eye(6), atol=1e-5), (
        "elements_by_state_jacobian is not the inverse of state_by_elements_jacobian"
    )


def test_covariance_to_elements_matches_monte_carlo():
    """Tests the linearised element covariance against a sampled one."""
    # Arrange
    generator = np.random.default_rng(20260904)
    r, v = scb.OrbitalElements.coe2rv(MU_EARTH, *ELEMENT_CASES[1])
    sigmas = np.array([0.05, 0.05, 0.05, 5e-6, 5e-6, 5e-6])
    correlation = generator.normal(size=(6, 6)) * 0.3 + np.eye(6)
    correlation = correlation @ correlation.T / np.trace(correlation @ correlation.T) * 6
    covariance = np.diag(sigmas) @ correlation @ np.diag(sigmas)
    covariance = 0.5 * (covariance + covariance.T)

    # Act
    elements, mapped, jacobian = scb.OrbitalElements.covariance_to_elements(
        MU_EARTH, r, v, covariance
    )
    samples = np.concatenate([r, v])[:, None] + np.linalg.cholesky(
        covariance
    ) @ generator.normal(size=(6, 50000))
    sampled_elements = np.array(
        [
            scb.OrbitalElements.rv2mean_elements(MU_EARTH, s[:3], s[3:])
            for s in samples.T
        ]
    )
    deviations = sampled_elements - elements
    deviations[:, 2:] = _wrap(deviations[:, 2:])
    sampled = np.cov(deviations.T)

    # Assert
    assert np.allclose(mapped, mapped.T, atol=0.0), "P_oe is not symmetric"
    assert np.linalg.eigvalsh(mapped).min() > 0.0, "P_oe is not positive definite"
    assert jacobian.shape == (6, 6)
    ratios = np.sqrt(np.diag(mapped)) / np.sqrt(np.diag(sampled))
    assert np.allclose(ratios, 1.0, atol=0.02), (
        f"Analytic and sampled standard deviations disagree: {ratios}"
    )


def test_covariance_to_elements_uses_leading_block():
    """Tests that estimated parameters trailing the state do not change the answer."""
    # Arrange
    r, v = scb.OrbitalElements.coe2rv(MU_EARTH, *ELEMENT_CASES[0])
    covariance = np.diag([1e-4, 1e-4, 1e-4, 1e-10, 1e-10, 1e-10])
    augmented = np.diag([1e-4, 1e-4, 1e-4, 1e-10, 1e-10, 1e-10, 7.0, 3.0])

    # Act
    _, mapped, _ = scb.OrbitalElements.covariance_to_elements(
        MU_EARTH, r, v, covariance
    )
    _, mapped_augmented, _ = scb.OrbitalElements.covariance_to_elements(
        MU_EARTH, r, v, augmented
    )

    # Assert
    assert np.allclose(mapped, mapped_augmented, rtol=0.0, atol=0.0), (
        "Trailing parameter rows leaked into the element covariance"
    )


@pytest.mark.parametrize(
    "covariance",
    [np.eye(5), np.zeros((6, 4)), np.ones(6)],
    ids=["too small", "not square", "not a matrix"],
)
def test_covariance_to_elements_rejects_bad_shapes(covariance):
    """Tests that a covariance that cannot be a Cartesian one is reported."""
    r, v = scb.OrbitalElements.coe2rv(MU_EARTH, *ELEMENT_CASES[0])
    with pytest.raises(scb.OrbitalElements.InvalidCovarianceError):
        scb.OrbitalElements.covariance_to_elements(MU_EARTH, r, v, covariance)


@pytest.mark.parametrize(
    "a, ecc", [(-7000.0, 0.5), (7000.0, 1.0), (7000.0, 1.4)],
    ids=["negative a", "parabolic", "hyperbolic"],
)
def test_non_elliptic_orbit_rejected(a, ecc):
    """Tests that an open orbit is reported rather than silently mangled."""
    with pytest.raises(scb.OrbitalElements.NonEllipticOrbitError):
        scb.OrbitalElements.coe2rv(MU_EARTH, a, ecc, 0.5, 0.1, 0.2, 0.3)

