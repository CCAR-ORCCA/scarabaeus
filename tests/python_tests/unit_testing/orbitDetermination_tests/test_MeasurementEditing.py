# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =============================================#
#  Unit Tests for the MeasurementEditing class #
# =============================================#
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

    The kurtosis, chi-squared and isolation-forest editors read only `set_name`,
    `measurement_type` and the residual / sigma / observed / outlier_flag / t2
    columns, so building a full dataset would add fixture weight without
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


# Columns of a full MeasurementDataSet, for editors that read more than the stub
NAMES = ["t1", "t2", "t3", "computed", "observed",
         "residuals", "partials", "sigma", "outlier_flag"]


def _pass_dataset(name, residuals, times=None, flagged=()):
    """
    Full MeasurementDataSet for one pass with unit sigma.

    The Huber/MAD editor detrends against t1, which the stub does not carry.
    """
    residuals = np.asarray(residuals, dtype=float)
    n = residuals.size
    times = np.arange(n, dtype=float) if times is None else np.asarray(times, dtype=float)
    flags = np.zeros(n)
    flags[list(flagged)] = 1
    values = [times.copy(), times.copy(), times.copy(), np.zeros(n),
              residuals.copy(), residuals.copy(), np.zeros((n, 6)), np.ones(n), flags]
    return scb.MeasurementDataSet(
        set_name=name, measurements=values, names=NAMES, instrument=None,
        target=None, epochs_t3=None, measurement_type="Fake",
    )


def _residual_filter(ds, **kwargs):
    """Run residual_filter on one dataset and write its flags back."""
    editor = scb.MeasurementEditing([ds])
    newly = editor.residual_filter(**kwargs)[ds.set_name]
    editor.apply_flags()
    return newly


def _sloped_pass(outlier_at=31, n=60, size=40.0, noise=0.0, seed=0):
    """A pass with a real trend, plus one point knocked off it."""
    t = np.arange(n, dtype=float)
    res = 0.5 * t
    if noise:
        res = res + np.random.default_rng(seed).normal(0, noise, n)
    res[outlier_at] += size
    return t, res


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


# endregion Fixtures  #
# --------------------#


# --------------------#
# region       Tests #
# --------------------#
class TestKurtosisFilter:
    """Behavior of MeasurementEditing.kurtosis_filter and its helpers."""

    def test_trims_to_the_target(self, editor, heavy_tailed):
        """Trimming stops as soon as the survivors reach the target kurtosis."""
        editor.kurtosis_filter(preview=False)

        kept = heavy_tailed[~editor.outlier_flags["Range"]]
        assert sample_kurtosis(kept, fisher=False) <= 3.0

    def test_removes_the_extreme_residuals(self, editor, outlier_indices):
        """The points it flags are the injected outliers, not arbitrary ones."""
        editor.kurtosis_filter(preview=False)

        flags = editor.outlier_flags["Range"]
        assert flags[outlier_indices].all()

        # Trimming stops at the target, so a point or two of the Gaussian core may
        # go with the tail; it must not start eating into the core.
        assert int(flags.sum()) <= outlier_indices.size + 2

    def test_leaves_gaussian_residuals_alone(self, rng):
        """An already-Gaussian population loses no measurements."""
        editor = scb.MeasurementEditing([StubDataSet("Clean", rng.normal(0.0, 1.0, 2000))])

        editor.kurtosis_filter(preview=False)

        assert not editor.outlier_flags["Clean"].any()

    def test_trimming_curve_matches_the_sample_kurtosis(self, editor):
        """The curve driving the decision is the kurtosis of the surviving points."""
        _, z, curve, n_remove, _ = editor._kurtosis_evaluate("Range", 3.0, 0.25)

        order = np.argsort(-np.abs(z - np.median(z)))
        for k in (0, 1, n_remove, curve.size - 1):
            assert curve[k] == pytest.approx(
                sample_kurtosis(z[order[k:]], fisher=False), abs=1e-9
            )

    def test_zero_sigma_points_never_flagged(self, rng):
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

    def test_already_flagged_points_excluded(self, heavy_tailed, outlier_indices):
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

    def test_max_fraction_caps_the_removal(self, rng):
        """The cap bounds how much of a heavy-tailed set can be removed."""
        editor = scb.MeasurementEditing([StubDataSet("Heavy", rng.standard_t(2.0, 400))])

        editor.kurtosis_filter(max_fraction=0.01, preview=False)

        assert int(editor.outlier_flags["Heavy"].sum()) <= int(0.01 * 400)

    def test_too_few_points_left_alone(self):
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
    def test_invalid_settings_reported(self, editor, kwargs):
        """Out-of-range settings are refused before anything is flagged."""
        with pytest.raises(scb.MeasurementEditing.InvalidKurtosisSettingsError):
            editor.kurtosis_filter(preview=False, **kwargs)

        assert not editor.outlier_flags["Range"].any()

    def test_preview_figure_builds(self, editor):
        """The preview renders without a display attached."""
        residuals, z, curve, n_remove, new_flags = editor._kurtosis_evaluate(
            "Range", 3.0, 0.25
        )

        fig = editor._build_kurtosis_figure(
            "Range", residuals, z, curve, n_remove, new_flags, 3.0
        )

        assert len(fig.get_axes()) == 3


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


# Isolation forest
class TestIsolationForestFilter:
    """
    An isolation forest scores a residual by how few random splits separate it
    from the rest, and the rejection threshold is taken from the knee of the
    ranked scores rather than from an assumed contamination fraction. A
    two-component observable is scored jointly and rejected together.
    """

    def test_kneedle_finds_the_knee_of_a_tailed_population(self):
        """The knee separates the dense core from the tail."""
        scores = np.concatenate([np.linspace(0.0, 0.2, 95), np.linspace(2.0, 3.0, 5)])

        threshold, knee = scb.MeasurementEditing._kneedle_threshold(scores)

        # Rejection is "strictly greater", so the threshold sits on the last point
        # of the core: the five tail points are cut and nothing else is.
        assert np.flatnonzero(scores > threshold).tolist() == list(range(95, 100))
        assert 0 < knee < scores.size - 1

    def test_kneedle_on_a_flat_population_rejects_nothing(self):
        """With no spread there is no tail to cut."""
        scores = np.full(50, 0.3)

        threshold, knee = scb.MeasurementEditing._kneedle_threshold(scores)

        assert threshold == pytest.approx(0.3)
        assert np.count_nonzero(scores > threshold) == 0
        assert knee == scores.size - 1

    @pytest.mark.parametrize("scores", [[1.0, 2.0], [np.nan, 1.0, 2.0]])
    def test_kneedle_needs_three_scores(self, scores):
        """Two points cannot define an interior knee, nor can a non-finite one."""
        with pytest.raises(
            scb.MeasurementEditing.InvalidIsolationForestSettingsError
        ):
            scb.MeasurementEditing._kneedle_threshold(scores)

    def test_components_of_one_observable_are_grouped(self, astrometric_pair):
        """The two scalar datasets of a vector observable are scored as one."""
        groups = astrometric_pair._isolation_forest_groups(
            astrometric_pair._resolve_names(None)
        )

        assert groups == [["CAM_AngularIdeal_ra", "CAM_AngularIdeal_dec"]]

    def test_a_rejection_flags_both_components(self, astrometric_pair):
        """A bad image is bad in both coordinates."""
        astrometric_pair.isolation_forest_filter(preview=False)

        ra_flags = astrometric_pair.outlier_flags["CAM_AngularIdeal_ra"]
        dec_flags = astrometric_pair.outlier_flags["CAM_AngularIdeal_dec"]

        assert np.array_equal(ra_flags, dec_flags)
        assert ra_flags.any()

    def test_injected_outliers_are_flagged(self, astrometric_pair):
        """The far-out points are among those rejected."""
        astrometric_pair.isolation_forest_filter(preview=False)

        flagged = np.flatnonzero(astrometric_pair.outlier_flags["CAM_AngularIdeal_ra"])

        assert set(range(295, 300)).issubset(set(flagged.tolist()))

    def test_too_few_points_are_left_untouched(self, rng):
        """A handful of points is not enough to fit a forest."""
        editor = scb.MeasurementEditing([StubDataSet("Range", rng.normal(0, 1, 5))])

        editor.isolation_forest_filter(preview=False)

        assert not editor.outlier_flags["Range"].any()

    def test_non_finite_residuals_are_rejected(self, rng):
        """A NaN residual is flagged without reaching the forest."""
        residuals = rng.normal(0.0, 1.0, 40)
        residuals[3] = np.nan
        editor = scb.MeasurementEditing([StubDataSet("Range", residuals)])

        editor.isolation_forest_filter(preview=False)

        assert editor.outlier_flags["Range"][3]

    def test_invalid_n_estimators_is_reported(self, astrometric_pair):
        """A forest needs at least one tree."""
        with pytest.raises(
            scb.MeasurementEditing.InvalidIsolationForestSettingsError,
            match="n_estimators",
        ):
            astrometric_pair.isolation_forest_filter(n_estimators=0, preview=False)


# Huber detrend + MAD
class TestHuberMadFilter:
    """
    huber_mad_filter detrends each pass with a robust (Huber) fit and rejects
    points by the MAD of what is left, so it needs no trusted sigma; it can then
    refit sigma from the survivors.
    """

    def test_flags_outlier_on_a_sloped_pass(self):
        """The trend is why detrending comes first: a plain scatter test reads
        the slope itself as spread."""
        t, res = _sloped_pass()
        editor = scb.MeasurementEditing([_pass_dataset("d", res, t)])

        editor.huber_mad_filter(n_sigma=5.0, refit_sigma=False)

        assert editor.outlier_flags["d"][31]
        assert editor.outlier_flags["d"].sum() == 1

    def test_chi_squared_misses_what_huber_mad_finds(self):
        """chi_squared_filter judges against the attached sigma; before a
        trustworthy sigma exists that is exactly what is missing."""
        t, res = _sloped_pass()
        huber = scb.MeasurementEditing([_pass_dataset("d", res, t)])
        chi = scb.MeasurementEditing([_pass_dataset("d", res, t)])

        huber.huber_mad_filter(n_sigma=5.0, refit_sigma=False)
        chi.chi_squared_filter(sigma_scale=5.0, preview=False)

        assert huber.outlier_flags["d"].sum() == 1
        assert chi.outlier_flags["d"].sum() > 1, "sigma=1 makes the trend look bad"

    def test_existing_flags_excluded_from_the_fit(self):
        """A flagged point stays flagged and does not drag the fit."""
        t = np.arange(60, dtype=float)
        res = np.zeros(60)
        res[10] = 1.0e6
        res[31] = 5.0
        editor = scb.MeasurementEditing([_pass_dataset("d", res, t, flagged=[10])])

        editor.huber_mad_filter(n_sigma=5.0, refit_sigma=False)

        assert editor.outlier_flags["d"][10], "existing flag preserved"
        assert editor.outlier_flags["d"][31], "found once the huge point is excluded"

    def test_mad_scale_switches_convention(self):
        """n_sigma multiplies the raw MAD at mad_scale=1.0."""
        t, res = _sloped_pass(size=5.0, noise=0.2, seed=1)
        tight = scb.MeasurementEditing([_pass_dataset("d", res, t)])
        loose = scb.MeasurementEditing([_pass_dataset("d", res, t)])

        tight.huber_mad_filter(n_sigma=5.0, mad_scale=1.0, refit_sigma=False)
        loose.huber_mad_filter(n_sigma=100.0, mad_scale=1.0, refit_sigma=False)

        assert tight.outlier_flags["d"][31]
        assert loose.outlier_flags["d"].sum() == 0

    def test_refits_sigma_from_surviving_points(self):
        """The refit sigma comes from the survivors, not the attached sigma."""
        t, res = _sloped_pass(noise=0.2, seed=0)
        editor = scb.MeasurementEditing([_pass_dataset("d", res, t)])

        out = editor.huber_mad_filter(n_sigma=5.0, refit_sigma=True)

        assert out is editor.refit_sigmas
        assert 0.0 < editor.refit_sigmas["d"] < 1.0

    def test_refit_sigma_off_records_nothing(self):
        """With refit_sigma=False no sigma is recorded."""
        t, res = _sloped_pass(noise=0.2, seed=0)
        editor = scb.MeasurementEditing([_pass_dataset("d", res, t)])

        editor.huber_mad_filter(refit_sigma=False)

        assert editor.refit_sigmas == {}

    def test_too_few_points_is_safe(self):
        """Too few points to fit a trend flags nothing."""
        editor = scb.MeasurementEditing([_pass_dataset("d", np.zeros(2))])

        editor.huber_mad_filter()

        assert editor.outlier_flags["d"].sum() == 0

    def test_constant_residuals_flag_nothing(self):
        """A zero robust scale must not reject the whole pass."""
        editor = scb.MeasurementEditing([_pass_dataset("d", np.zeros(40))])

        editor.huber_mad_filter()

        assert editor.outlier_flags["d"].sum() == 0

    def test_selects_by_dataset_name(self):
        """Only the named dataset is edited."""
        t, res = _sloped_pass()
        editor = scb.MeasurementEditing(
            [_pass_dataset("a", res, t), _pass_dataset("b", res, t)]
        )

        editor.huber_mad_filter(dataset_name="a", refit_sigma=False)

        assert editor.outlier_flags["a"].sum() == 1
        assert editor.outlier_flags["b"].sum() == 0


class TestResidualFilter:
    """
    Note on the default criterion: 5 sigma about the mean is not robust, since
    an outlier inflates the very spread it is tested against. For a lone
    outlier among n-1 tight points the test only fires when n > 26, whatever
    the outlier's magnitude. Sample sizes below use n = 41 so the default
    actually triggers; huber_mad_filter is the robust alternative.
    """

    def test_flags_gross_outlier(self):
        res = np.zeros(41)
        res[17] = 10.0
        ds = _pass_dataset("d", res)

        newly = _residual_filter(ds)

        assert list(newly) == [17]
        assert ds.data["outlier_flag"][17] == 1
        assert ds.data["outlier_flag"].sum() == 1

    def test_lone_outlier_below_breakdown_size_is_missed(self):
        """Documents the non-robustness above: with n = 5 the outlier hides
        inside its own inflated sigma."""
        ds = _pass_dataset("d", [0.1, -0.1, 0.05, 50.0, -0.08])

        assert _residual_filter(ds).size == 0

    def test_existing_flags_excluded_from_statistics(self):
        """A previously flagged gross outlier must not inflate sigma and mask
        the next one. Including it would put the 5 sigma bound near 7.6e5,
        hiding the residual at 10.0 completely."""
        res = np.zeros(42)
        res[17] = 10.0
        res[41] = 1.0e6
        ds = _pass_dataset("d", res, flagged=[41])

        newly = _residual_filter(ds)

        assert list(newly) == [17], "outlier must be found once the flagged point is excluded"
        assert ds.data["outlier_flag"][41] == 1, "existing flag preserved"

    def test_explicit_bounds(self):
        ds = _pass_dataset("d", [0.0, 1.0, 2.0, 3.0])

        _residual_filter(ds, mean_outlier=0.0, sigma_outlier=1.5)

        assert list(ds.data["outlier_flag"]) == [0, 0, 1, 1]

    def test_all_flagged_returns_empty(self):
        ds = _pass_dataset("d", [1.0, 2.0, 3.0], flagged=[0, 1, 2])
        assert _residual_filter(ds).size == 0

    def test_is_idempotent(self):
        res = np.zeros(41)
        res[17] = 10.0
        ds = _pass_dataset("d", res)

        _residual_filter(ds)
        first = list(ds.data["outlier_flag"])
        second_call = _residual_filter(ds)

        assert second_call.size == 0, "re-running must not flag anything new"
        assert list(ds.data["outlier_flag"]) == first

    def test_missing_residuals_raises(self):
        ds = _pass_dataset("d", [0.0])
        del ds.data["residuals"]
        with pytest.raises(KeyError):
            _residual_filter(ds)

    def test_scoped_to_named_dataset(self):
        """Only the named dataset is touched when dataset_name is given."""
        res = np.zeros(41)
        res[17] = 10.0
        editor = scb.MeasurementEditing([_pass_dataset("a", res), _pass_dataset("b", res)])

        newly = editor.residual_filter(dataset_name="a")

        assert set(newly) == {"a"}
        assert not editor.outlier_flags["b"].any()


# endregion Tests     #
# --------------------#


# ------------------------------------------------------------------#
# region    Pass segmentation for the Huber/MAD filter             #
# ------------------------------------------------------------------#
from src.scarabaeus.orbitDetermination.MeasurementEditing import MeasurementEditing


def test_pass_segments_splits_on_gaps():
    """Observations separated by more than the gap threshold are separate passes."""
    day = 86400.0
    times = np.concatenate([
        661300000.0 + np.arange(10) * 60.0,
        661300000.0 + day + np.arange(8) * 60.0,
        661300000.0 + 2 * day + np.arange(12) * 60.0,
    ])
    segments = MeasurementEditing._pass_segments(times)
    assert [seg.size for seg in segments] == [10, 8, 12]
    for seg in segments:
        assert np.ptp(times[seg]) < float(MeasurementEditing.PASS_GAP.values)


def test_pass_segments_keeps_one_pass_whole():
    """
    The common case -- one dataset read from one pass file -- is a single
    segment, so the per-pass detrend reduces exactly to the previous behaviour
    and real-data results are unchanged.
    """
    times = 661300000.0 + np.arange(42) * 60.0
    segments = MeasurementEditing._pass_segments(times)
    assert len(segments) == 1
    assert segments[0].size == 42


def test_pass_segments_handles_unsorted_input():
    """Grouping is done in time order, but the indices refer to the input order."""
    times = np.concatenate([
        661300000.0 + np.arange(10) * 60.0,
        661300000.0 + 86400.0 + np.arange(8) * 60.0,
    ])
    shuffled = np.random.default_rng(0).permutation(times.size)
    segments = MeasurementEditing._pass_segments(times[shuffled])
    assert sorted(seg.size for seg in segments) == [8, 10]
    assert np.array_equal(
        np.sort(np.concatenate(segments)), np.arange(times.size)
    ), "every observation appears in exactly one segment"
    for seg in segments:
        assert np.ptp(times[shuffled][seg]) < float(MeasurementEditing.PASS_GAP.values)


def test_pass_segments_empty():
    assert MeasurementEditing._pass_segments(np.array([])) == []


def test_per_pass_detrend_does_not_reject_a_biased_pass():
    """
    A pass with its own bias is not thrown away wholesale.

    Detrending a multi-pass dataset with one line leaves the pass-to-pass
    offsets in the residuals. Where a pass sits far from that single line, every
    point in it exceeds the rejection threshold and the whole pass is discarded
    as outliers -- 20 good observations here, instead of the one real outlier.
    """
    from sklearn.linear_model import HuberRegressor

    day = 86400.0
    times = np.concatenate([
        661300000.0 + np.arange(20) * 60.0,
        661300000.0 + day + np.arange(20) * 60.0,
        661300000.0 + 2 * day + np.arange(20) * 60.0,
    ])
    rng = np.random.default_rng(1)
    residuals = rng.normal(0.0, 0.5, times.size)
    for i, offset in enumerate([-8.0, +8.0, -3.0]):  # independent, not collinear
        residuals[20 * i:20 * (i + 1)] += offset
    residuals[35] += 6.0  # the one real outlier

    def flags_from(detrended):
        median = np.median(detrended)
        scale = 1.4826 * np.median(np.abs(detrended - median))
        return np.abs(detrended) > 5.0 * scale

    def detrend(idx):
        t = (times[idx] - times[idx].mean()) / (times[idx].std() or 1.0)
        fit = HuberRegressor().fit(t[:, np.newaxis], residuals[idx])
        return residuals[idx] - fit.predict(t[:, np.newaxis])

    single_trend = flags_from(detrend(np.arange(times.size)))

    per_pass = np.empty(times.size)
    for seg in MeasurementEditing._pass_segments(times):
        per_pass[seg] = detrend(seg)
    per_pass_flags = flags_from(per_pass)

    assert single_trend.sum() > 10, "one trend rejects a whole pass"
    assert list(np.flatnonzero(per_pass_flags)) == [35], (
        "per-pass detrending leaves only the real outlier"
    )


def test_refit_sigma_uses_time_not_sample_index():
    """
    The quadratic refit measures curvature per second, not per observation.

    With ``x = np.linspace(0, len(y), len(y))`` the spacing of the data was
    invisible: points a minute apart and points hours apart counted the same, so
    on an unevenly sampled dataset the quadratic could not follow the real
    curvature and the refitted sigma came out far too large.
    """
    times = np.concatenate([
        np.arange(10) * 60.0,
        21600 + np.arange(10) * 60.0,
        25200 + np.arange(20) * 60.0,
    ])
    curvature, noise = 1e-8, 0.05
    rng = np.random.default_rng(3)
    y = curvature * (times - times.mean()) ** 2 + rng.normal(0, noise, times.size)

    index_x = np.linspace(0, len(y), len(y))
    by_index = np.std(y - np.polyval(np.polyfit(index_x, y, 2), index_x), ddof=1)

    time_x = (times - times.mean()) / times.std()
    by_time = np.std(y - np.polyval(np.polyfit(time_x, y, 2), time_x), ddof=1)

    assert by_time == pytest.approx(noise, rel=0.5), "time recovers the noise level"
    assert by_index > 5 * by_time, "the index fit overstates sigma badly"


# endregion Pass segmentation #
# ------------------------------------------------------------------#


def test_sparse_single_pass_is_not_split():
    """
    A sparse pass stays one segment.

    Sequential ranging inside a single real pass has a median gap of 523 s and a
    maximum of 3300 s over a 6.2 h track. A 300 s threshold cuts that one pass
    into 38 segments, which breaks the filter (see the next test).
    """
    rng = np.random.default_rng(0)
    times = 661300000.0 + np.cumsum(rng.integers(30, 1200, 42)).astype(float)
    segments = MeasurementEditing._pass_segments(times)
    assert len(segments) == 1, "one pass, however sparsely sampled, is one segment"
    assert segments[0].size == 42


def test_tiny_segments_are_merged_not_detrended_alone():
    """
    A segment too short to fit is merged into its neighbour.

    Each segment is detrended on its own, so a segment of one centred on its own
    median has a residual of exactly zero and a gross outlier in it becomes
    invisible to the rejection threshold. Over-splitting therefore silently
    un-flags the points the filter exists to catch, which is why the split is
    conservative and short segments are merged back.
    """
    times = np.concatenate([
        661300000.0 + np.arange(20) * 60.0,
        661300000.0 + 86400.0 + np.arange(2) * 60.0,   # 2 < _MIN_PASS_POINTS
    ])
    segments = MeasurementEditing._pass_segments(times)
    assert len(segments) == 1, "the 2-point straggler is merged, not left alone"
    assert segments[0].size == 22

    # With enough points it is kept separate.
    times = np.concatenate([
        661300000.0 + np.arange(20) * 60.0,
        661300000.0 + 86400.0 + np.arange(MeasurementEditing.PASS_MIN_POINTS) * 60.0,
    ])
    assert len(MeasurementEditing._pass_segments(times)) == 2


def test_already_flagged_points_do_not_count_towards_segment_size():
    """
    Segment sizes are measured over the points that will actually be fitted,
    since points flagged by an earlier pass contribute nothing to a trend.
    """
    times = np.concatenate([
        661300000.0 + np.arange(20) * 60.0,
        661300000.0 + 86400.0 + np.arange(6) * 60.0,
    ])
    keep = np.ones(times.size, dtype=bool)
    assert len(MeasurementEditing._pass_segments(times, keep=keep)) == 2

    keep[-3:] = False          # leaves 3 fittable points in the second pass
    assert len(MeasurementEditing._pass_segments(times, keep=keep)) == 1


def test_pass_gap_is_overridable_from_the_caller():
    """
    The gap threshold and the minimum pass size are tuning knobs, not fixed
    facts: a mission whose tracking cadence differs from DSN deep-space passes
    passes its own through huber_mad_filter, which forwards them here.
    """
    times = np.concatenate([
        661300000.0 + np.arange(10) * 60.0,
        661300000.0 + 7200.0 + np.arange(10) * 60.0,   # 2 h apart
    ])
    # The 6 h default sees one pass.
    assert len(MeasurementEditing._pass_segments(times)) == 1
    # A caller that knows its passes are an hour apart sees two.
    assert len(
        MeasurementEditing._pass_segments(times, gap_threshold_sec=3600.0)
    ) == 2
    # And can insist on larger passes, collapsing them again.
    assert len(
        MeasurementEditing._pass_segments(
            times, gap_threshold_sec=3600.0, min_points=20
        )
    ) == 1


def test_huber_mad_filter_forwards_the_pass_parameters(monkeypatch):
    """huber_mad_filter hands the caller's pass settings down to the evaluator."""
    seen = {}

    def spy(self, name, n_sigma, mad_scale, refit_sigma,
            pass_gap_sec=None, min_pass_points=None):
        seen.update(pass_gap_sec=pass_gap_sec, min_pass_points=min_pass_points)
        return np.zeros(1, dtype=bool), None

    monkeypatch.setattr(MeasurementEditing, "_huber_mad_evaluate", spy)
    monkeypatch.setattr(MeasurementEditing, "_huber_mad_apply",
                        lambda self, *a, **k: None)
    monkeypatch.setattr(MeasurementEditing, "_resolve_names",
                        lambda self, name: ["ds"])

    editor = MeasurementEditing.__new__(MeasurementEditing)
    editor._refit_sigmas = {}
    editor.huber_mad_filter(pass_gap_sec=1234.0, min_pass_points=7)
    assert seen == {"pass_gap_sec": 1234.0, "min_pass_points": 7}


# ------------------------------------------------------------------#
# region    Editor settings                                         #
# ------------------------------------------------------------------#
def test_the_settings_start_at_the_class_defaults():
    editor = scb.MeasurementEditing([StubDataSet("Range", np.zeros(3))])
    assert float(editor.pass_gap.values) == float(MeasurementEditing.PASS_GAP.values) == 21600.0
    assert editor.pass_min_points == MeasurementEditing.PASS_MIN_POINTS == 5
    assert editor.isolation_forest_min_points == MeasurementEditing.ISOLATION_FOREST_MIN_POINTS == 8


def test_the_pass_gap_takes_a_time_or_bare_seconds():
    editor = scb.MeasurementEditing([StubDataSet("Range", np.zeros(3))])
    editor.pass_gap = scb.ArrayWUnits(1.0, scb.Units.get_units("hr"))
    assert float(editor.pass_gap.values) == 3600.0
    editor.pass_gap = 1800
    assert float(editor.pass_gap.values) == 1800.0


@pytest.mark.parametrize("name, value, error", [
    ("pass_gap", 0.0, scb.MeasurementEditing.InvalidHuberMadSettingsError),
    ("pass_gap", -1.0, scb.MeasurementEditing.InvalidHuberMadSettingsError),
    ("pass_gap", np.inf, scb.MeasurementEditing.InvalidHuberMadSettingsError),
    ("pass_gap", scb.ArrayWUnits(1.0, scb.Units.get_units("km")),
     scb.MeasurementEditing.InvalidHuberMadSettingsError),
    ("pass_min_points", 0, scb.MeasurementEditing.InvalidHuberMadSettingsError),
    ("pass_min_points", 2.5, scb.MeasurementEditing.InvalidHuberMadSettingsError),
    ("pass_min_points", True, scb.MeasurementEditing.InvalidHuberMadSettingsError),
    ("pass_min_points", scb.ArrayWUnits(5, None),
     scb.MeasurementEditing.InvalidHuberMadSettingsError),
    ("isolation_forest_min_points", 0, scb.MeasurementEditing.InvalidIsolationForestSettingsError),
    ("isolation_forest_min_points", np.array([8, 8]),
     scb.MeasurementEditing.InvalidIsolationForestSettingsError),
])
def test_a_bad_setting_is_refused(name, value, error):
    editor = scb.MeasurementEditing([StubDataSet("Range", np.zeros(3))])
    with pytest.raises(error, match=name):
        setattr(editor, name, value)


def test_the_editor_pass_settings_reach_the_segmentation():
    """
    Three passes 2 h apart are one under the 6 h default and three under a 1 h
    gap, which removes the pass-to-pass offsets one trend cannot.
    """
    times = np.concatenate([661300000.0 + k * 7200.0 + np.arange(20) * 60.0 for k in range(3)])
    residuals = np.random.default_rng(1).normal(0.0, 0.5, times.size)
    for k, offset in enumerate([-8.0, +8.0, -3.0]):
        residuals[20 * k:20 * (k + 1)] += offset
    residuals[35] += 6.0           # the one real outlier

    one_trend = scb.MeasurementEditing([_pass_dataset("d", residuals, times)])
    one_trend.huber_mad_filter(n_sigma=5.0, refit_sigma=False)
    assert one_trend.outlier_flags["d"].sum() > 1, "one trend misreads the offset"

    per_pass = scb.MeasurementEditing([_pass_dataset("d", residuals, times)])
    per_pass.pass_gap = scb.ArrayWUnits(1.0, scb.Units.get_units("hr"))
    per_pass.huber_mad_filter(n_sigma=5.0, refit_sigma=False)
    assert list(np.flatnonzero(per_pass.outlier_flags["d"])) == [35]

    # Passes shorter than pass_min_points merge back into one.
    merged = scb.MeasurementEditing([_pass_dataset("d", residuals, times)])
    merged.pass_gap = 3600.0
    merged.pass_min_points = 21
    merged.huber_mad_filter(n_sigma=5.0, refit_sigma=False)
    assert merged.outlier_flags["d"].sum() > 1


def test_the_isolation_forest_minimum_is_a_setting():
    """Below isolation_forest_min_points the forest leaves the points alone."""
    residuals = np.random.default_rng(0).normal(0.0, 1.0, 40)
    residuals[3] = 50.0
    editor = scb.MeasurementEditing([StubDataSet("Range", residuals)])
    editor.isolation_forest_min_points = 41
    editor.isolation_forest_filter(preview=False)
    assert not editor.outlier_flags["Range"].any()


# endregion Editor settings #
# ------------------------------------------------------------------#
