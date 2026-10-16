# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =============================================#
#  Unit Tests for the MeasurementEditing class #
# =============================================#
"""
MeasurementEditing flags outliers in measurement residuals before they reach
the filter. The kurtosis editor trims the heaviest tails until the surviving
residuals are no longer heavy-tailed: a Gaussian population has kurtosis 3, so
anything above that is outliers rather than noise.

Tests
-----
test_kurtosis_filter_trims_to_the_target
    Trimming stops as soon as the survivors reach the target kurtosis.
test_kurtosis_filter_removes_the_extreme_residuals
    The points it flags are the injected outliers, not arbitrary ones.
test_gaussian_residuals_are_left_alone
    An already-Gaussian population loses no measurements.
test_trimming_curve_matches_the_sample_kurtosis
    The curve driving the decision is the kurtosis of the surviving points.
test_zero_sigma_points_are_never_flagged
    Points with no usable sigma are excluded from the statistic, as in the
    chi-squared editor.
test_already_flagged_points_are_excluded
    Existing outliers stay flagged and do not inflate the kurtosis that drives
    further trimming.
test_max_fraction_caps_the_removal
    The cap bounds how much of a heavy-tailed set can be removed.
test_too_few_points_are_left_alone
    Fewer than four active points is not enough for a fourth moment.
test_invalid_settings_are_reported
    Out-of-range settings are refused before anything is flagged.
test_editors_compose_in_any_order
    Flags OR-merge, so chi-squared and kurtosis commute.
test_preview_figure_builds
    The preview renders without a display attached.
"""

import matplotlib

matplotlib.use("Agg")

import src.scarabaeus as scb
from scipy.stats import kurtosis as sample_kurtosis
import pytest
import numpy as np

# --------------------#
# region    Fixtures #
# --------------------#


class StubDataSet:
    """
    Minimal stand-in for a MeasurementDataSet.

    The editor reads only `set_name` and the residual / sigma / outlier_flag /
    t2 columns, so building a full dataset would add fixture weight without
    exercising any more of the editing path.
    """

    def __init__(self, name, residuals, sigma=None, measurement_type=None,
                 observed=None):
        residuals = np.asarray(residuals, dtype=float)
        self.set_name = name
        # Carried so the isolation forest can recognise the two components of
        # one observable and, for an RA/Dec pair, read the declination.
        self.measurement_type = measurement_type or name
        self.data = {
            "residuals": residuals,
            "sigma": np.ones_like(residuals) if sigma is None else np.asarray(
                sigma, dtype=float
            ),
            "observed": np.zeros_like(residuals) if observed is None else np.asarray(
                observed, dtype=float
            ),
            "outlier_flag": np.zeros(residuals.size, dtype=bool),
            "t2": np.arange(residuals.size, dtype=float),
        }


@pytest.fixture
def rng():
    """Deterministic generator so a flagged count is reproducible."""
    return np.random.default_rng(7)


@pytest.fixture
def outlier_indices():
    """Where the heavy tail is injected."""
    return np.array([5, 40, 120, 200, 260])


@pytest.fixture
def heavy_tailed(rng, outlier_indices):
    """Gaussian core with a handful of far-out residuals."""
    residuals = rng.normal(0.0, 1.0, 300)
    residuals[outlier_indices] += 30.0
    return residuals


@pytest.fixture
def editor(heavy_tailed):
    """Editor holding one heavy-tailed dataset."""
    return scb.MeasurementEditing([StubDataSet("Range", heavy_tailed)])


# endregion Fixtures  #
# --------------------#


# --------------------#
# region       Tests #
# --------------------#
def test_kurtosis_filter_trims_to_the_target(editor, heavy_tailed):
    """Trimming stops as soon as the survivors reach the target kurtosis."""
    editor.kurtosis_filter(preview=False)

    kept = heavy_tailed[~editor.outlier_flags["Range"]]
    assert sample_kurtosis(kept, fisher=False) <= 3.0


def test_kurtosis_filter_removes_the_extreme_residuals(editor, outlier_indices):
    """The points it flags are the injected outliers, not arbitrary ones."""
    editor.kurtosis_filter(preview=False)

    flags = editor.outlier_flags["Range"]
    assert flags[outlier_indices].all()

    # Trimming stops at the target, so a point or two of the Gaussian core may
    # go with the tail; it must not start eating into the core.
    assert int(flags.sum()) <= outlier_indices.size + 2


def test_gaussian_residuals_are_left_alone(rng):
    """An already-Gaussian population loses no measurements."""
    editor = scb.MeasurementEditing([StubDataSet("Clean", rng.normal(0.0, 1.0, 2000))])

    editor.kurtosis_filter(preview=False)

    assert not editor.outlier_flags["Clean"].any()


def test_trimming_curve_matches_the_sample_kurtosis(editor):
    """The curve driving the decision is the kurtosis of the surviving points."""
    _, z, curve, n_remove, _ = editor._kurtosis_evaluate("Range", 3.0, 0.25)

    order = np.argsort(-np.abs(z - np.median(z)))
    for k in (0, 1, n_remove, curve.size - 1):
        assert curve[k] == pytest.approx(
            sample_kurtosis(z[order[k:]], fisher=False), abs=1e-9
        )


def test_zero_sigma_points_are_never_flagged(rng):
    """Points with no usable sigma are excluded from the statistic."""
    residuals = rng.normal(0.0, 1.0, 300)
    residuals[[10, 150]] += 40.0
    sigma = np.ones_like(residuals)
    sigma[10] = 0.0  # unusable: cannot be normalised

    editor = scb.MeasurementEditing([StubDataSet("Mixed", residuals, sigma)])
    editor.kurtosis_filter(preview=False)

    flags = editor.outlier_flags["Mixed"]
    assert not flags[10]
    assert flags[150]


def test_already_flagged_points_are_excluded(heavy_tailed, outlier_indices):
    """Existing outliers stay flagged and stay out of the statistic."""
    dataset = StubDataSet("Range", heavy_tailed)
    dataset.data["outlier_flag"][outlier_indices] = True

    editor = scb.MeasurementEditing([dataset])
    curve_pre_flagged = editor._kurtosis_evaluate("Range", 3.0, 0.25)[2]

    # Counting the flagged tail would report a wildly heavy-tailed set and drive
    # the trimming to chew through the core instead.
    assert curve_pre_flagged[0] < 4.0
    assert sample_kurtosis(heavy_tailed, fisher=False) > 20.0

    editor.kurtosis_filter(preview=False)
    flags = editor.outlier_flags["Range"]

    assert flags[outlier_indices].all()
    assert int(flags.sum()) <= outlier_indices.size + 2


def test_max_fraction_caps_the_removal(rng):
    """The cap bounds how much of a heavy-tailed set can be removed."""
    editor = scb.MeasurementEditing([StubDataSet("Heavy", rng.standard_t(2.0, 400))])

    editor.kurtosis_filter(max_fraction=0.01, preview=False)

    assert int(editor.outlier_flags["Heavy"].sum()) <= int(0.01 * 400)


def test_too_few_points_are_left_alone():
    """Fewer than four active points is not enough for a fourth moment."""
    editor = scb.MeasurementEditing([StubDataSet("Tiny", [1.0, 2.0, 30.0])])

    editor.kurtosis_filter(preview=False)

    assert not editor.outlier_flags["Tiny"].any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_kurtosis": 0.5},
        {"max_fraction": 1.0},
        {"max_fraction": -0.1},
    ],
    ids=["target-below-minimum", "fraction-too-large", "fraction-negative"],
)
def test_invalid_settings_are_reported(editor, kwargs):
    """Out-of-range settings are refused before anything is flagged."""
    with pytest.raises(scb.MeasurementEditing.InvalidKurtosisSettingsError):
        editor.kurtosis_filter(preview=False, **kwargs)

    assert not editor.outlier_flags["Range"].any()


def test_editors_compose_in_any_order(heavy_tailed):
    """Flags OR-merge, so chi-squared and kurtosis commute."""
    chi2_first = scb.MeasurementEditing([StubDataSet("Range", heavy_tailed)])
    chi2_first.chi_squared_filter(preview=False)
    chi2_first.kurtosis_filter(preview=False)

    kurtosis_first = scb.MeasurementEditing([StubDataSet("Range", heavy_tailed)])
    kurtosis_first.kurtosis_filter(preview=False)
    kurtosis_first.chi_squared_filter(preview=False)

    assert np.array_equal(
        chi2_first.outlier_flags["Range"], kurtosis_first.outlier_flags["Range"]
    )


def test_preview_figure_builds(editor):
    """The preview renders without a display attached."""
    residuals, z, curve, n_remove, new_flags = editor._kurtosis_evaluate(
        "Range", 3.0, 0.25
    )

    fig = editor._build_kurtosis_figure(
        "Range", residuals, z, curve, n_remove, new_flags, 3.0
    )

    assert len(fig.get_axes()) == 3


# endregion Tests     #
# --------------------#


# ---------------------------------------------------#
# region  Isolation forest                           #
# ---------------------------------------------------#
"""
An isolation forest scores a residual by how few random splits separate it from
the rest, and the rejection threshold is taken from the knee of the ranked
scores rather than from an assumed contamination fraction. A two-component
observable is scored jointly and rejected together.

Tests
-----
test_kneedle_finds_the_knee_of_a_tailed_population
    The threshold lands in the gap, not at either end.
test_kneedle_on_a_flat_population_rejects_nothing
    With no spread there is no tail to cut.
test_kneedle_needs_three_scores
    Two points cannot define an interior knee.
test_components_of_one_observable_are_grouped
    The two scalar datasets of a vector observable are scored as one.
test_a_rejection_flags_both_components
    A bad image is bad in both coordinates.
test_injected_outliers_are_flagged
    The far-out points are among those rejected.
test_too_few_points_are_left_untouched
    A handful of points is not enough to fit a forest.
test_non_finite_residuals_are_rejected
    A NaN residual is flagged without reaching the forest.
test_invalid_n_estimators_is_reported
    A forest needs at least one tree.
"""


@pytest.fixture
def astrometric_pair(rng):
    """RA/Dec residuals with five far-out pairs, as one split observable."""
    ra = rng.normal(0.0, 0.15, 300)
    dec = rng.normal(0.0, 0.12, 300)
    ra[-5:] += np.array([1.5, -1.8, 2.0, -2.2, 2.5])
    dec[-5:] += np.array([-1.4, 1.6, 2.1, -2.0, 2.4])
    return scb.MeasurementEditing([
        StubDataSet("CAM_AngularIdeal_ra", ra, measurement_type="AngularIdeal ra"),
        StubDataSet("CAM_AngularIdeal_dec", dec, measurement_type="AngularIdeal dec"),
    ])


def test_kneedle_finds_the_knee_of_a_tailed_population():
    """The knee separates the dense core from the tail."""
    scores = np.concatenate([np.linspace(0.0, 0.2, 95), np.linspace(2.0, 3.0, 5)])

    threshold, knee = scb.MeasurementEditing._kneedle_threshold(scores)

    # Rejection is "strictly greater", so the threshold sits on the last point
    # of the core: the five tail points are cut and nothing else is.
    assert np.flatnonzero(scores > threshold).tolist() == list(range(95, 100))
    assert 0 < knee < scores.size - 1


def test_kneedle_on_a_flat_population_rejects_nothing():
    """With no spread there is no tail to cut."""
    scores = np.full(50, 0.3)

    threshold, knee = scb.MeasurementEditing._kneedle_threshold(scores)

    assert threshold == pytest.approx(0.3)
    assert np.count_nonzero(scores > threshold) == 0
    assert knee == scores.size - 1


@pytest.mark.parametrize("scores", [[1.0, 2.0], [np.nan, 1.0, 2.0]])
def test_kneedle_needs_three_scores(scores):
    """Two points cannot define an interior knee, nor can a non-finite one."""
    with pytest.raises(
        scb.MeasurementEditing.InvalidIsolationForestSettingsError
    ):
        scb.MeasurementEditing._kneedle_threshold(scores)


def test_components_of_one_observable_are_grouped(astrometric_pair):
    """The two scalar datasets of a vector observable are scored as one."""
    groups = astrometric_pair._isolation_forest_groups(
        astrometric_pair._resolve_names(None)
    )

    assert groups == [["CAM_AngularIdeal_ra", "CAM_AngularIdeal_dec"]]


def test_a_rejection_flags_both_components(astrometric_pair):
    """A bad image is bad in both coordinates."""
    astrometric_pair.isolation_forest_filter(preview=False)

    ra_flags = astrometric_pair.outlier_flags["CAM_AngularIdeal_ra"]
    dec_flags = astrometric_pair.outlier_flags["CAM_AngularIdeal_dec"]

    assert np.array_equal(ra_flags, dec_flags)
    assert ra_flags.any()


def test_injected_outliers_are_flagged(astrometric_pair):
    """The far-out points are among those rejected."""
    astrometric_pair.isolation_forest_filter(preview=False)

    flagged = np.flatnonzero(astrometric_pair.outlier_flags["CAM_AngularIdeal_ra"])

    assert set(range(295, 300)).issubset(set(flagged.tolist()))


def test_too_few_points_are_left_untouched(rng):
    """A handful of points is not enough to fit a forest."""
    editor = scb.MeasurementEditing([StubDataSet("Range", rng.normal(0, 1, 5))])

    editor.isolation_forest_filter(preview=False)

    assert not editor.outlier_flags["Range"].any()


def test_non_finite_residuals_are_rejected(rng):
    """A NaN residual is flagged without reaching the forest."""
    residuals = rng.normal(0.0, 1.0, 40)
    residuals[3] = np.nan
    editor = scb.MeasurementEditing([StubDataSet("Range", residuals)])

    editor.isolation_forest_filter(preview=False)

    assert editor.outlier_flags["Range"][3]


def test_invalid_n_estimators_is_reported(astrometric_pair):
    """A forest needs at least one tree."""
    with pytest.raises(
        scb.MeasurementEditing.InvalidIsolationForestSettingsError,
        match="n_estimators",
    ):
        astrometric_pair.isolation_forest_filter(n_estimators=0, preview=False)

