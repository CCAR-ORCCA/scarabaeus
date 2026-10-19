# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ============================================#
#  Unit Tests for the MeasurementSpec class  #
# ============================================#
import types

import numpy as np
import pytest

import src.scarabaeus as scb
from src.scarabaeus.measurements.MeasurementSpec import MeasurementSpec

# --------------------#
# region    Fixtures #
# --------------------#
NAMES = ["t1", "t2", "t3", "computed", "observed",
         "residuals", "partials", "sigma", "outlier_flag"]


class FakeModel:
    """Minimal stand-in for a measurement model."""

    def __init__(self, name, residuals, times=None, sigma=1.0, units=None):
        self.name = name
        self._residuals = np.asarray(residuals, float)
        self._times = (np.arange(len(self._residuals), dtype=float)
                       if times is None else np.asarray(times, float))
        self.sigma = sigma
        self.units = units
        self.reference_state = None
        self.calls = 0

    def update_reference_state(self, state):
        self.reference_state = state

    def generate_measurement_dataset(self, dataset_name, target=None,
                                     observed_meas=None, epochs=None, **kw):
        self.calls += 1
        n = len(self._residuals)
        # flags are seeded from the observations, as the real model does
        flags = (np.asarray(observed_meas[3], float).copy()
                 if observed_meas is not None else np.zeros(n))
        values = [self._times.copy(), self._times.copy(), self._times.copy(),
                  np.zeros(n), self._residuals.copy(), self._residuals.copy(),
                  np.zeros((n, 6)), np.ones(n), flags]
        return scb.MeasurementDataSet(
            set_name=dataset_name, measurements=values, names=NAMES,
            instrument=None, target=target, epochs_t3=None,
            measurement_type="Fake", units=self.units,
        )


class TwoComponentModel(FakeModel):
    """Stand-in for a two-component model (AngularIdeal, CentroidingIdeal):
    one dataset per component, named "<dataset_name>_<label>"."""

    component_labels = ("ra", "dec")

    def generate_measurement_dataset(self, dataset_name, **kw):
        return [super(TwoComponentModel, self).generate_measurement_dataset(
                    f"{dataset_name}_{label}", **kw)
                for label in self.component_labels]


def _obs(n, flagged=()):
    flags = np.zeros(n)
    for i in flagged:
        flags[i] = 1
    return scb.ObsData(None, np.arange(n, dtype=float), None, flags, {})


def _spec(residuals, times=None, flagged=()):
    model = FakeModel("GS1 Range", residuals, times)
    obs = _obs(len(residuals), flagged)
    return scb.MeasurementSpec.from_models([model], [obs]), model


def _sloped_pass(outlier_at=31, n=60, size=40.0, noise=0.0, seed=0):
    """A pass with a real trend, plus one point knocked off it."""
    t = np.arange(n, dtype=float)
    res = 0.5 * t
    if noise:
        res = res + np.random.default_rng(seed).normal(0, noise, n)
    res[outlier_at] += size
    return t, res

# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
class TestCompute:
    def test_stores_dataset_and_exposes_residuals(self):
        res = np.zeros(41); res[17] = 10.0
        spec, _ = _spec(res)

        spec.compute(target=None)

        assert spec.leaves[0].dataset is not None
        np.testing.assert_allclose(spec.residuals()["GS1 Range"], res)

    def test_returns_self_for_chaining(self):
        spec, _ = _spec(np.zeros(5))
        assert spec.compute(target=None) is spec

    def test_pushes_reference_state(self):
        spec, model = _spec(np.zeros(5))
        spec.compute(target=None, reference_state="STATE")
        assert model.reference_state == "STATE"

    def test_residuals_before_compute_raises(self):
        spec, _ = _spec(np.zeros(5))
        with pytest.raises(RuntimeError, match="compute"):
            spec.residuals()

    def test_container_computes_every_stream(self):
        a = FakeModel("A", np.zeros(5))
        b = FakeModel("B", np.zeros(5))
        spec = scb.MeasurementSpec.from_models([a, b], [_obs(5), _obs(5)])

        spec.compute(target=None)

        assert a.calls == 1 and b.calls == 1
        assert set(spec.residuals()) == {"A", "B"}


class TestFlagOutliers:
    """
    A filter rebuilds its datasets on every iteration from the observations, so
    a flag recorded only on the dataset is lost on the next pass. Flags must
    land on ObsData.meas_outliers to survive.
    """

    def test_writes_flags_back_to_observations(self):
        """The property that makes flags survive filter iteration."""
        res = np.zeros(41); res[17] = 10.0
        spec, _ = _spec(res)
        spec.compute(target=None)

        newly = spec.flag_outliers()

        assert list(newly["GS1 Range"]) == [17]
        obs_flags = np.asarray(spec.leaves[0].observed_meas[3])
        assert obs_flags[17] == 1, "flag must reach ObsData, not just the dataset"

    def test_flags_survive_dataset_regeneration(self):
        """Recomputing rebuilds the dataset; the flag must come back."""
        res = np.zeros(41); res[17] = 10.0
        spec, _ = _spec(res)
        spec.compute(target=None)
        spec.flag_outliers()

        spec.compute(target=None)          # as a filter iteration would

        regenerated = np.asarray(spec.leaves[0].dataset.data["outlier_flag"])
        assert regenerated[17] == 1, "flag lost on regeneration"

    def test_file_flags_are_inherited(self):
        spec, _ = _spec(np.zeros(5), flagged=[2])
        spec.compute(target=None)
        assert np.asarray(spec.leaves[0].dataset.data["outlier_flag"])[2] == 1

    def test_updates_sigma_when_given(self):
        spec, model = _spec(np.zeros(41))
        spec.compute(target=None)

        spec.flag_outliers(sigma={"GS1 Range": 0.25})

        assert model.sigma == 0.25

    def test_sigma_dict_ignores_unknown_names(self):
        spec, model = _spec(np.zeros(41))
        spec.compute(target=None)
        spec.flag_outliers(sigma={"someone else": 0.25})
        assert model.sigma == 1.0

    def test_flag_before_compute_raises(self):
        spec, _ = _spec(np.zeros(5))
        with pytest.raises(RuntimeError, match="compute"):
            spec.flag_outliers()


class TestSummary:
    def test_reports_counts_and_rms(self):
        res = np.zeros(41); res[17] = 10.0
        spec, _ = _spec(res, flagged=[0])
        spec.compute(target=None)

        line = spec.summary()

        assert "GS1 Range" in line and "41 obs" in line and "1 flagged" in line

    def test_works_before_compute(self):
        spec, _ = _spec(np.zeros(5))
        assert "not computed" in spec.summary()


class TestProvenance:
    """compute() resolves a body's ephemeris from global SPICE state, so what
    it actually used has to be recorded or it is unrecoverable afterwards."""

    class FakeTarget:
        name = "Orbiter"
        spice_id = -62

    def test_records_target_identity(self):
        spec, _ = _spec(np.zeros(5))
        spec.compute(target=self.FakeTarget())

        prov = spec.leaves[0].provenance
        assert prov["target_name"] == "Orbiter"
        assert prov["target_spice_id"] == -62
        assert "computed_at" in prov

    def test_records_trajectory_when_resolvable(self):
        spec, _ = _spec(np.zeros(5))
        spec.compute(target=self.FakeTarget())

        prov = spec.leaves[0].provenance
        # spk_files may be empty without kernels loaded; the key must exist
        assert "spk_files" in prov and "trajectory" in prov

    def test_missing_before_compute(self):
        spec, _ = _spec(np.zeros(5))
        assert spec.leaves[0].provenance is None

    def test_refreshed_on_recompute(self):
        spec, _ = _spec(np.zeros(5))
        spec.compute(target=self.FakeTarget())
        first = spec.leaves[0].provenance

        class Other:
            name = "Orbiter_ref_it1"
            spice_id = -1001

        spec.compute(target=Other())
        second = spec.leaves[0].provenance

        assert first["target_spice_id"] == -62
        assert second["target_spice_id"] == -1001, "provenance must follow the target"

    def test_trajectory_is_the_spk_covering_the_epochs(self, tmp_path):
        """A later SPK shadows an earlier one only inside its own coverage, so
        the last-loaded file for the body is not necessarily what resolved."""
        spice = pytest.importorskip("spiceypy")
        body = -62

        def write(name, start, stop):
            path = str(tmp_path / name)
            handle = spice.spkopn(path, name, 0)
            states = np.tile([7000.0, 0, 0, 0, 7.5, 0], (2, 1))
            spice.spkw09(handle, body, 399, "J2000", start, stop, "seg",
                         1, 2, states, [start, stop])
            spice.spkcls(handle)
            return path

        mission = write("mission.bsp", 0.0, 1000.0)
        reference = write("reference.bsp", 2000.0, 3000.0)
        spice.furnsh(mission)
        spice.furnsh(reference)
        try:
            obs = scb.ObsData(scb.EpochArray(np.array([100.0, 200.0]), sys="TDB"),
                              np.zeros(2), None, np.zeros(2), {})
            spec = scb.MeasurementSpec.from_models(
                [FakeModel("GS1 Range", np.zeros(2))], [obs]
            )
            spec.compute(target=self.FakeTarget())
            prov = spec.leaves[0].provenance
            assert prov["spk_files"] == [mission]
            assert prov["trajectory"] == "mission.bsp"

            assert scb.SpiceManager.spk_files_for_body(body, epochs=[100.0, 2500.0]) == [mission, reference]
            assert scb.SpiceManager.spk_files_for_body(body) == [mission, reference]
        finally:
            spice.unload(reference)
            spice.unload(mission)


# Editing bridge
class TestSpecEditBridge:
    """
    MeasurementEditing works on datasets, which a filter rebuilds on every
    iteration. Going through MeasurementSpec.edit also writes the flags onto the
    source observations, which is what makes them survive that rebuild.
    """

    def test_flags_reach_the_observations(self):
        """The property that makes flags survive a filter iteration."""
        t, res = _sloped_pass()
        spec, _ = _spec(res, t)
        spec.compute(target=None)

        spec.edit("huber_mad_filter", n_sigma=5.0, refit_sigma=False)

        assert np.asarray(spec.leaves[0].observed_meas[3])[31] == 1

    def test_flags_survive_dataset_regeneration(self):
        t, res = _sloped_pass()
        spec, _ = _spec(res, t)
        spec.compute(target=None)
        spec.edit("huber_mad_filter", n_sigma=5.0, refit_sigma=False)

        spec.compute(target=None)      # as a filter iteration would

        assert np.asarray(spec.leaves[0].dataset.data["outlier_flag"])[31] == 1

    def test_refit_sigma_reaches_the_model(self):
        """MeasurementEditing owns datasets, not models; the bridge applies it."""
        t, res = _sloped_pass(noise=0.2, seed=0)
        spec, model = _spec(res, t)
        spec.compute(target=None)

        spec.edit("huber_mad_filter", n_sigma=5.0, refit_sigma=True)

        assert model.sigma != 1.0
        assert 0.0 < float(model.sigma) < 1.0

    def test_reports_newly_flagged_only(self):
        t, res = _sloped_pass()
        spec, _ = _spec(res, t, flagged=[5])
        spec.compute(target=None)

        newly = spec.edit("huber_mad_filter", n_sigma=5.0, refit_sigma=False)

        assert 31 in list(newly["GS1 Range"])
        assert 5 not in list(newly["GS1 Range"]), "already flagged before the edit"

    def test_requires_compute_first(self):
        spec, _ = _spec(np.zeros(5))
        with pytest.raises(RuntimeError, match="compute"):
            spec.edit("huber_mad_filter")

    def test_scoped_edit_reports_only_its_stream(self):
        t, res = _sloped_pass()
        spec = scb.MeasurementSpec.from_models(
            [FakeModel("A", res, t), FakeModel("B", res, t)],
            [_obs(len(res)), _obs(len(res))],
        )
        spec.compute(target=None)

        newly = spec.edit("huber_mad_filter", dataset_name="A", n_sigma=5.0,
                          refit_sigma=False)

        assert list(newly) == ["A"]
        assert not np.any(np.asarray(spec.leaves[1].observed_meas[3]))


class TestRefitUnits:
    """
    A refit is the scatter of the residuals, so it is in the dataset's units.
    It used to be labelled with the units of the model's sigma without being
    converted, so a km scatter went onto a model whose sigma is in m as metres.
    """

    km, m, sec = scb.Units.get_units(["km", "m", "sec"])

    def _edit(self, sigma, units, sigma_units=None):
        t, res = _sloped_pass(noise=0.2, seed=0)
        model = FakeModel("GS1 Range", res, t, sigma=sigma, units=units)
        spec = scb.MeasurementSpec.from_models([model], [_obs(len(res))])
        spec.compute(target=None)
        spec.edit("huber_mad_filter", n_sigma=5.0, refit_sigma=True,
                  sigma_units=sigma_units)
        return model, spec

    def test_falls_back_to_the_dataset(self):
        """With no model sigma the dataset's units label the refit."""
        model, _ = self._edit(None, self.km / self.sec)

        assert model.sigma.units == self.km / self.sec
        assert 0.0 < float(model.sigma.values) < 1.0

    def test_converted_to_the_model_sigma_units(self):
        in_km, _ = self._edit(scb.ArrayWUnits(1.0, self.km), self.km)
        in_m, _ = self._edit(scb.ArrayWUnits(1000.0, self.m), self.km)

        assert in_m.sigma.units == self.m
        np.testing.assert_allclose(float(in_m.sigma.values),
                                   1000.0 * float(in_km.sigma.values))

    def test_converted_to_the_units_asked_for(self):
        in_km, _ = self._edit(None, self.km)
        in_m, _ = self._edit(scb.ArrayWUnits(1.0, self.km), self.km,
                             sigma_units={"GS1 Range": self.m})

        assert in_m.sigma.units == self.m
        np.testing.assert_allclose(float(in_m.sigma.values),
                                   1000.0 * float(in_km.sigma.values))

    def test_incompatible_units_are_refused_and_nothing_is_flagged(self):
        t, res = _sloped_pass(noise=0.2, seed=0)
        model = FakeModel("GS1 Range", res, t,
                          sigma=scb.ArrayWUnits(1.0, self.sec), units=self.km)
        spec = scb.MeasurementSpec.from_models([model], [_obs(len(res))])
        spec.compute(target=None)

        with pytest.raises(scb.MeasurementSpec.RefitSigmaUnitsError, match="GS1 Range"):
            spec.edit("huber_mad_filter", n_sigma=5.0, refit_sigma=True)

        assert not np.any(np.asarray(spec.leaves[0].observed_meas[3]))
        assert not np.any(np.asarray(spec.leaves[0].dataset.data["outlier_flag"]))
        assert model.sigma.units == self.sec


class TestTwoComponentStreams:
    """
    A two-component model builds one dataset per component, named
    "<dataset_name>_<label>", from one set of observations. The spec used to
    assume one dataset per stream, so every method that read it failed.
    """

    @staticmethod
    def _spec(sigma=1.0):
        t, res = _sloped_pass(noise=0.2, seed=0)
        model = TwoComponentModel("Cam", res, t, sigma=sigma)
        spec = scb.MeasurementSpec.from_models([model], [_obs(len(res))])
        return spec.compute(target=None), model

    def test_residuals_and_summary_are_per_component(self):
        spec, _ = self._spec()

        assert set(spec.residuals()) == {"Cam_ra", "Cam_dec"}
        lines = spec.summary().splitlines()
        assert [line.split(":")[0] for line in lines] == ["Cam_ra", "Cam_dec"]

    def test_the_stream_name_edits_both_components(self):
        spec, _ = self._spec()

        newly = spec.edit("huber_mad_filter", dataset_name="Cam", n_sigma=5.0,
                          refit_sigma=False)

        assert list(newly) == ["Cam"]
        assert list(newly["Cam"]) == [31]
        assert np.asarray(spec.leaves[0].observed_meas[3])[31] == 1
        for ds in spec.leaves[0].dataset:
            assert np.asarray(ds.data["outlier_flag"])[31] == 1

    def test_a_component_flag_lands_on_both(self):
        """The components share one observation, and so one flag."""
        spec, _ = self._spec()

        newly = spec.edit("huber_mad_filter", dataset_name="Cam_ra", n_sigma=5.0,
                          refit_sigma=False)

        assert list(newly["Cam"]) == [31]
        for ds in spec.leaves[0].dataset:
            assert np.asarray(ds.data["outlier_flag"])[31] == 1

    def test_the_refit_is_a_vector_over_the_components(self):
        spec, model = self._spec()

        spec.edit("huber_mad_filter", n_sigma=5.0, refit_sigma=True)

        assert np.shape(model.sigma) == (2,)
        assert np.all((0.0 < model.sigma) & (model.sigma < 1.0))

    def test_a_component_not_refitted_keeps_its_sigma(self):
        spec, model = self._spec()

        spec.edit("huber_mad_filter", dataset_name="Cam_ra", n_sigma=5.0,
                  refit_sigma=True)

        assert 0.0 < model.sigma[0] < 1.0
        assert model.sigma[1] == 1.0, "the sigma column Cam_dec was built with"

    def test_editor_methods_that_took_one_name(self):
        """The stream name reaches the editor as a list of its components."""
        spec, _ = self._spec()

        newly = spec.edit("chi_squared_filter", dataset_name="Cam", preview=False)

        assert list(newly) == ["Cam"]


class TestEpochSpan:
    """epoch_span works on the TDB instants, whatever system the epochs use."""

    sec, minute, km = scb.Units.get_units(["sec", "min", "km"])

    @staticmethod
    def _spec_at(*epoch_arrays):
        models, obs = [], []
        for i, ep in enumerate(epoch_arrays):
            n = len(np.asarray(ep.et).ravel())
            models.append(FakeModel(f"GS{i} Range", np.zeros(n)))
            obs.append(scb.ObsData(ep, np.zeros(n), None, np.zeros(n), {}))
        return MeasurementSpec.from_models(models, obs)

    def test_unique_sorted_epochs_without_pad(self):
        a = scb.EpochArray(np.array([30.0, 10.0]), sys="TDB")
        b = scb.EpochArray(np.array([20.0, 10.0]), sys="TDB")
        span = self._spec_at(a, b).epoch_span()
        assert np.allclose(np.asarray(span.et).ravel(), [10.0, 20.0, 30.0])

    def test_uses_tdb_instants_not_display_values(self):
        """Epochs in another system must not be relabelled TDB by value: TAI
        is 32.184 s off TDB, so reading display values would shift the span."""
        ep = scb.EpochArray(np.array([1000.0, 2000.0]), sys="TDB").to(sys="TAI")
        span = self._spec_at(ep).epoch_span()
        assert np.allclose(np.asarray(span.et).ravel(), [1000.0, 2000.0])
        assert span.system == "TAI", "returned in the observations' own system"

    def test_mixed_systems_return_tdb(self):
        a = scb.EpochArray(np.array([10.0]), sys="TDB")
        b = scb.EpochArray(np.array([20.0]), sys="TDB").to(sys="TAI")
        assert self._spec_at(a, b).epoch_span().system == "TDB"

    def test_pad_with_units(self):
        ep = scb.EpochArray(np.array([1000.0, 2000.0]), sys="TDB")
        span = self._spec_at(ep).epoch_span(pad=scb.ArrayWUnits(1.0, self.minute))
        assert np.allclose(np.asarray(span.et).ravel(), [940.0, 2060.0])

    def test_bare_pad_is_seconds(self):
        ep = scb.EpochArray(np.array([1000.0, 2000.0]), sys="TDB")
        span = self._spec_at(ep).epoch_span(pad=60.0)
        assert np.allclose(np.asarray(span.et).ravel(), [940.0, 2060.0])

    def test_resampled_span(self):
        ep = scb.EpochArray(np.array([1000.0, 2000.0]), sys="TDB")
        span = self._spec_at(ep).epoch_span(pad=100.0, n=5)
        assert np.allclose(np.asarray(span.et).ravel(), np.linspace(900.0, 2100.0, 5))

    def test_pad_with_wrong_dimensions_is_refused(self):
        ep = scb.EpochArray(np.array([1000.0, 2000.0]), sys="TDB")
        with pytest.raises(ValueError, match="time"):
            self._spec_at(ep).epoch_span(pad=scb.ArrayWUnits(1.0, self.km))

# endregion Tests #
# --------------#
