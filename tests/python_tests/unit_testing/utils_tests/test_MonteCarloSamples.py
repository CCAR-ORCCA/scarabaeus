# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==============================================#
#  Unit Tests for the MonteCarloSamples class  #
# ==============================================#
import src.scarabaeus as scb

import numpy as np
import pytest

EPOCH = 1000.0


# --------------------#
# region    Fixtures #
# --------------------#
@pytest.fixture
def draws():
    """Correlated Gaussian draws, shape (n_samples, n_dimensions)."""
    rng = np.random.default_rng(5)
    A = np.array([[2.0, 0.0, 0.0], [0.5, 1.0, 0.0], [-0.3, 0.2, 0.5]])
    return rng.standard_normal((400, 3)) @ A.T + np.array([1.0, -2.0, 3.0])


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(draws):
    samples = scb.MonteCarloSamples(draws, EPOCH, labels=["a", "b", "c"])

    assert samples.n_samples == 400 and samples.n_dimensions == 3
    assert samples.labels == ["a", "b", "c"]
    assert samples.epoch == EPOCH and samples.frame == "J2000"
    np.testing.assert_array_equal(samples.to_array(), draws)
    np.testing.assert_allclose(samples.weights, 1.0 / 400)
    assert scb.MonteCarloSamples(draws, EPOCH).labels == ["x0", "x1", "x2"]


def test_weights_are_normalised(draws):
    weights = np.arange(1.0, 401.0)
    samples = scb.MonteCarloSamples(draws, EPOCH, weights=weights)

    np.testing.assert_allclose(samples.weights, weights / weights.sum())


def test_equal_weight_moments_match_numpy(draws):
    samples = scb.MonteCarloSamples(draws, EPOCH)

    np.testing.assert_allclose(samples.get_mean(), draws.mean(axis=0))
    np.testing.assert_allclose(samples.get_covariance(), np.cov(draws, rowvar=False))
    np.testing.assert_allclose(
        samples.get_standard_deviations(), draws.std(axis=0, ddof=1)
    )
    assert samples.effective_sample_size() == pytest.approx(400.0)


def test_weighted_moments_match_numpy(draws):
    """Reliability weights: numpy's aweights give the same unbiased estimate."""
    weights = np.random.default_rng(6).uniform(0.1, 2.0, 400)
    samples = scb.MonteCarloSamples(draws, EPOCH, weights=weights)

    np.testing.assert_allclose(
        samples.get_mean(), np.average(draws, axis=0, weights=weights)
    )
    np.testing.assert_allclose(
        samples.get_covariance(), np.cov(draws, rowvar=False, aweights=weights)
    )
    w = weights / weights.sum()
    assert samples.effective_sample_size() == pytest.approx(1.0 / np.sum(w**2))
    assert samples.effective_sample_size() < 400.0


def test_integer_weights_are_repeated_samples():
    """A weight of k counts a sample k times; the extremes stay the samples'."""
    draws = np.array([[0.0], [1.0], [4.0]])
    weighted = scb.MonteCarloSamples(draws, EPOCH, weights=[1.0, 2.0, 1.0])
    repeated = np.array([[0.0], [1.0], [1.0], [4.0]])

    np.testing.assert_allclose(weighted.get_mean(), repeated.mean(axis=0))
    assert weighted.get_percentile(0.0) == pytest.approx(0.0)
    assert weighted.get_percentile(100.0) == pytest.approx(4.0)


@pytest.mark.parametrize("percentile", [0.0, 2.5, 50.0, 97.5, 100.0])
def test_equal_weight_percentile_matches_numpy(draws, percentile):
    samples = scb.MonteCarloSamples(draws, EPOCH)
    result = samples.get_percentile(percentile)

    assert result.shape == (3,)
    np.testing.assert_allclose(result, np.percentile(draws, percentile, axis=0))


def test_percentile_array(draws):
    samples = scb.MonteCarloSamples(draws, EPOCH)
    q = [16.0, 50.0, 84.0]

    result = samples.get_percentile(q)
    assert result.shape == (3, 3)
    np.testing.assert_allclose(result, np.percentile(draws, q, axis=0))


def test_to_covariance_matrix(draws):
    samples = scb.MonteCarloSamples(draws, EPOCH, frame="ECLIPJ2000")
    covariance = samples.to_covariance_matrix()

    assert isinstance(covariance, scb.CovarianceMatrix)
    assert covariance.epoch == EPOCH and covariance.frame == "ECLIPJ2000"
    np.testing.assert_allclose(
        covariance.matrix_without_units(), samples.get_covariance()
    )


@pytest.mark.parametrize(
    "samples, kwargs, error",
    [
        (np.zeros(5), {}, scb.MonteCarloSamples.SamplesShapeError),
        (np.zeros((2, 2, 2)), {}, scb.MonteCarloSamples.SamplesShapeError),
        (np.array([[0.0, np.nan]]), {}, scb.MonteCarloSamples.SamplesShapeError),
        (np.zeros((4, 2)), {"labels": ["a"]}, scb.MonteCarloSamples.LabelsSizeError),
        (np.zeros((4, 2)), {"weights": np.ones(3)}, scb.MonteCarloSamples.WeightsError),
        (
            np.zeros((4, 2)),
            {"weights": [1.0, -1.0, 1.0, 1.0]},
            scb.MonteCarloSamples.WeightsError,
        ),
        (np.zeros((4, 2)), {"weights": np.zeros(4)}, scb.MonteCarloSamples.WeightsError),
        (
            np.zeros((4, 2)),
            {"weights": [1.0, np.inf, 1.0, 1.0]},
            scb.MonteCarloSamples.WeightsError,
        ),
    ],
)
def test_errors(samples, kwargs, error):
    with pytest.raises(error):
        scb.MonteCarloSamples(samples, EPOCH, **kwargs)


# endregion Tests #
# --------------#
