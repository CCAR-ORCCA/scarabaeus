# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ================================================#
#  Unit Tests for the SolutionCloseness utility   #
# ================================================#
import matplotlib

matplotlib.use("Agg")

import src.scarabaeus as scb
import pytest
import numpy as np

SolutionCloseness = scb.SolutionCloseness

# --------------------#
# region    Fixtures #
# --------------------#


def random_covariance(rng, n, spread=1.0):
    """A symmetric positive-definite covariance."""
    root = rng.standard_normal((n, n)) * spread
    return root @ root.T / n + np.eye(n) * spread**2 * 0.5


@pytest.fixture
def rng():
    """Deterministic generator so a reported closeness is reproducible."""
    return np.random.default_rng(42)


@pytest.fixture
def pair(rng):
    """Two unrelated 6-D solutions."""
    first = (np.zeros(6), random_covariance(rng, 6))
    second = (rng.standard_normal(6) * 1.5, random_covariance(rng, 6))
    return first, second


def make_solution_od(covariances, states, timestamps=None, timeline=None,
                     smoothed=None):
    """
    Build a real SolutionOD from a stub filter.

    The filter is stubbed the way the repo's own SolutionOD tests do it, so
    what is exercised here is the solution object's attribute contract rather
    than a full filter run.
    """
    covariances = np.asarray(covariances, dtype=float)
    states = None if states is None else np.asarray(states, dtype=float)
    epochs = (
        [float(i) for i in range(len(covariances))]
        if timestamps is None
        else list(timestamps)
    )

    class StubFilter:
        covariance_analysis = False
        has_consider = False

        class measurement_data:
            @staticmethod
            def get_spacecraft_times():
                return epochs

    settings = scb.OutputSettings(
        save_state_deviation_history=False,
        save_covariance_history=False,
        save_prefit_residuals=False,
        save_postfit_residuals=False,
        save_consider_covariance_history=False,
        save_smoothed_solution=False,
    )
    solution = scb.SolutionOD(StubFilter(), settings)

    # populate the way the filter would
    solution.covariance_est = covariances
    solution.state_est = states
    if smoothed is not None:
        solution.covariance_smooth = np.asarray(smoothed, dtype=float)
    if timeline is not None:
        solution.timeline = np.asarray(timeline, dtype=float)
    return solution


# endregion Fixtures  #
# --------------------#


# --------------------#
# region       Tests #
# --------------------#
class TestBetween:
    """Mathematical properties of the closeness measure between two solutions."""

    def test_equal_covariances_give_half_the_mahalanobis_distance(self):
        """The congruent case has a closed form, and it is matched exactly."""
        covariance = np.array([[4.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0]])
        first = np.array([1.0, 2.0, 3.0])
        second = np.array([4.0, -1.0, 2.5])

        result = SolutionCloseness.between((first, covariance), (second, covariance))

        difference = second - first
        expected = 0.5 * np.sqrt(difference @ np.linalg.inv(covariance) @ difference)
        assert result["d"] == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize(
        "sigma_1, sigma_2",
        [(1.0, 1.0), (2.0, 0.5), (3.0, 7.0)],
        ids=["equal", "one-wider", "both-wide"],
    )
    def test_isotropic_covariances_touch_when_the_radii_meet(self, sigma_1, sigma_2):
        """Spheres touch when their radii sum to the separation."""
        separation = np.array([5.0, 0.0, 0.0])

        result = SolutionCloseness.between(
            (np.zeros(3), sigma_1**2 * np.eye(3)),
            (separation, sigma_2**2 * np.eye(3)),
        )

        assert result["d"] == pytest.approx(
            np.linalg.norm(separation) / (sigma_1 + sigma_2), rel=1e-12
        )

    def test_identical_solutions_are_zero_apart(self, pair):
        """A solution compared with itself is zero sigma away."""
        first, _ = pair

        result = SolutionCloseness.between(first, first)

        assert result["d"] == 0.0
        assert np.array_equal(result["x_tang"], first[0])

    def test_tangency_point_is_equidistant(self, pair):
        """The touching point sits on both ellipsoids at once."""
        first, second = pair

        result = SolutionCloseness.between(first, second)

        from_first, from_second = result["mahalanobis"]
        assert from_first == pytest.approx(result["d"], rel=1e-9)
        assert from_second == pytest.approx(result["d"], rel=1e-9)

        # and independently, from the returned point
        point = result["x_tang"]
        for (mean, covariance) in (first, second):
            offset = point - mean
            distance = np.sqrt(offset @ np.linalg.inv(covariance) @ offset)
            assert distance == pytest.approx(result["d"], rel=1e-9)

    def test_is_symmetric(self, pair):
        """Swapping the two solutions cannot change the answer."""
        first, second = pair

        forward = SolutionCloseness.between(first, second)["d"]
        backward = SolutionCloseness.between(second, first)["d"]

        assert forward == pytest.approx(backward, rel=1e-12)

    def test_is_scale_invariant(self, pair):
        """Re-expressing both solutions in different length units changes nothing."""
        (mean_1, cov_1), (mean_2, cov_2) = pair
        factor = 1000.0  # km -> m

        plain = SolutionCloseness.between((mean_1, cov_1), (mean_2, cov_2))["d"]
        rescaled = SolutionCloseness.between(
            (mean_1 * factor, cov_1 * factor**2),
            (mean_2 * factor, cov_2 * factor**2),
        )["d"]

        assert rescaled == pytest.approx(plain, rel=1e-10)

    def test_wider_covariance_brings_solutions_closer(self, pair):
        """Growing one covariance can only reduce the closeness."""
        (mean_1, cov_1), (mean_2, cov_2) = pair

        tight = SolutionCloseness.between((mean_1, cov_1), (mean_2, cov_2))["d"]
        loose = SolutionCloseness.between((mean_1, 4.0 * cov_1), (mean_2, cov_2))["d"]

        assert loose < tight

    def test_states_selects_a_subset(self, pair):
        """Comparing position alone differs from comparing the full state."""
        first, second = pair

        full = SolutionCloseness.between(first, second)["d"]
        position = SolutionCloseness.between(first, second, states=slice(0, 3))["d"]

        assert position != pytest.approx(full)

        # and the subset must equal the same comparison done by hand
        by_hand = SolutionCloseness.between(
            (first[0][:3], first[1][:3, :3]), (second[0][:3], second[1][:3, :3])
        )["d"]
        assert position == pytest.approx(by_hand, rel=1e-12)


class TestSolutionODInput:
    """Reading solutions directly from SolutionOD objects."""

    def test_reads_estimate_and_covariance(self, rng):
        """A SolutionOD is read for its estimate and covariance."""
        covariance = random_covariance(rng, 6)
        states_a = np.tile(np.zeros(6), (4, 1))
        states_b = np.tile(rng.standard_normal(6), (4, 1))

        solution_a = make_solution_od(np.tile(covariance, (4, 1, 1)), states_a)
        solution_b = make_solution_od(np.tile(covariance, (4, 1, 1)), states_b)

        from_solutions = SolutionCloseness.between(solution_a, solution_b)["d"]
        from_arrays = SolutionCloseness.between(
            (states_a[-1], covariance), (states_b[-1], covariance)
        )["d"]

        assert from_solutions == pytest.approx(from_arrays, rel=1e-12)

    def test_index_selects_the_epoch(self, rng):
        """Earlier epochs can be compared as easily as the final one."""
        covariance = random_covariance(rng, 3)
        # the estimate walks away from the origin over four epochs
        states_a = np.zeros((4, 3))
        states_b = np.array([[0.1, 0, 0], [0.5, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])

        covariances = np.tile(covariance, (4, 1, 1))
        solution_a = make_solution_od(covariances, states_a)
        solution_b = make_solution_od(covariances, states_b)

        first_epoch = SolutionCloseness.between(solution_a, solution_b, index=0)["d"]
        last_epoch = SolutionCloseness.between(solution_a, solution_b, index=-1)["d"]

        assert first_epoch < last_epoch

    def test_smoothed_covariance_can_be_requested(self, rng):
        """The smoothed history is used when asked for."""
        filtered = np.tile(random_covariance(rng, 3), (3, 1, 1))
        smoothed = 0.25 * filtered  # a smoother tightens the covariance
        states_a = np.zeros((3, 3))
        states_b = np.tile(np.array([1.0, 0.0, 0.0]), (3, 1))

        solution_a = make_solution_od(filtered, states_a, smoothed=smoothed)
        solution_b = make_solution_od(filtered, states_b, smoothed=smoothed)

        with_filtered = SolutionCloseness.between(solution_a, solution_b)["d"]
        with_smoothed = SolutionCloseness.between(
            solution_a, solution_b, covariance="smooth"
        )["d"]

        # a tighter covariance puts the same separation further out in sigma
        assert with_smoothed == pytest.approx(2.0 * with_filtered, rel=1e-9)

    def test_mismatched_grids_match_by_epoch(self, rng):
        """A sequence run's covariance and state grids differ and align by epoch."""
        covariance = random_covariance(rng, 3)

        # covariance on a five-point timeline, states on three measurement epochs
        covariances = np.tile(covariance, (5, 1, 1))
        timeline = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        timestamps = np.array([0.0, 2.0, 4.0])
        states = np.array([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])

        solution = make_solution_od(
            covariances, states, timestamps=timestamps, timeline=timeline
        )
        reference = (np.zeros(3), covariance)

        # timeline index 1 (epoch 1.0) is nearest measurement epoch 0.0 -> state 0
        at_epoch_one = SolutionCloseness.between(reference, solution, index=1)["d"]
        assert at_epoch_one == 0.0

        # timeline index -1 (epoch 4.0) -> the last state
        at_end = SolutionCloseness.between(reference, solution, index=-1)
        expected = SolutionCloseness.between(reference, (states[-1], covariance))
        assert at_end["d"] == pytest.approx(expected["d"], rel=1e-12)


class TestErrorHandling:
    """Invalid inputs to SolutionCloseness.between are reported clearly."""

    def test_missing_covariance(self, rng):
        """Asking for a history the run did not store says so."""
        solution = make_solution_od(
            np.tile(random_covariance(rng, 3), (2, 1, 1)), np.zeros((2, 3))
        )

        with pytest.raises(
            SolutionCloseness.SolutionClosenessError, match="no 'smooth' covariance"
        ):
            SolutionCloseness.between(solution, solution, covariance="smooth")

    def test_missing_state_estimate(self, rng):
        """A covariance-analysis solution has no estimate to compare."""
        solution = make_solution_od(
            np.tile(random_covariance(rng, 3), (2, 1, 1)), states=None
        )

        with pytest.raises(
            SolutionCloseness.SolutionClosenessError, match="no state estimate"
        ):
            SolutionCloseness.between(solution, solution)

    def test_unknown_covariance_name(self, rng):
        """A typo in the covariance selector is refused."""
        solution = make_solution_od(
            np.tile(random_covariance(rng, 3), (2, 1, 1)), np.zeros((2, 3))
        )

        with pytest.raises(
            SolutionCloseness.SolutionClosenessError, match="unknown covariance"
        ):
            SolutionCloseness.between(solution, solution, covariance="filtered")

    def test_size_mismatch(self, rng):
        """Solutions of different state size cannot be compared."""
        small = (np.zeros(3), random_covariance(rng, 3))
        large = (np.zeros(6), random_covariance(rng, 6))

        with pytest.raises(
            SolutionCloseness.SolutionClosenessError, match="different state sizes"
        ):
            SolutionCloseness.between(small, large)

    def test_singular_covariance(self):
        """A degenerate ellipsoid is refused with a usable message."""
        singular = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

        with pytest.raises(SolutionCloseness.SolutionClosenessError, match="singular"):
            SolutionCloseness.between(
                (np.zeros(3), singular), (np.ones(3), np.eye(3))
            )


class TestMatrix:
    """The pairwise matrix and plot_matrix built from several solutions."""

    def test_is_symmetric_with_zero_diagonal(self, rng):
        """The pairwise matrix has the structure it claims."""
        solutions = [
            (rng.standard_normal(4), random_covariance(rng, 4)) for _ in range(5)
        ]

        distances = SolutionCloseness.matrix(solutions)

        assert distances.shape == (5, 5)
        assert np.allclose(distances, distances.T, rtol=1e-12)
        assert np.allclose(np.diag(distances), 0.0)
        assert np.all(distances[np.triu_indices(5, k=1)] > 0.0)

    def test_separates_two_families(self, rng):
        """Solutions cluster: within-family distances beat across-family ones."""
        def family(center, spread, count):
            return [
                (center + rng.standard_normal(6) * spread, random_covariance(rng, 6, spread))
                for _ in range(count)
            ]

        tight = family(np.zeros(6), 0.2, 3)
        offset = family(np.full(6, 6.0), 0.2, 3)

        distances = SolutionCloseness.matrix(tight + offset)

        within = np.concatenate([distances[:3, :3][np.triu_indices(3, 1)],
                                 distances[3:, 3:][np.triu_indices(3, 1)]])
        across = distances[:3, 3:].ravel()

        assert np.max(within) < np.min(across)

    def test_plot_builds_a_figure(self, rng):
        """The plot renders headless."""
        solutions = [
            (rng.standard_normal(3), random_covariance(rng, 3)) for _ in range(4)
        ]

        distances, figure = SolutionCloseness.plot_matrix(
            solutions, labels=["A", "B", "C", "D"], scheme="sigma"
        )

        assert distances.shape == (4, 4)
        assert len(figure.get_axes()) >= 1


# endregion Tests     #
# --------------------#
