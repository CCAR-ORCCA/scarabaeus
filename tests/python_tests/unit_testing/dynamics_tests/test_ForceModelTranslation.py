# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==================================================#
#  Unit Tests for the ForceModelTranslation class  #
# ==================================================#
import src.scarabaeus as scb
import json
import pytest
import numpy as np

km, kg, sec, m, deg, AU = scb.Units.get_units(["km", "kg", "sec", "m", "deg", "AU"])
J2000 = scb.Frame("J2000")

# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture
def epoch():
    """Epoch to define state."""
    return scb.EpochArray("2000 JAN 01 12:00:00.000", "UTC")


@pytest.fixture
def origin():
    """Origin body to define state."""
    return scb.CelestialBody.from_constants("Earth")


@pytest.fixture
def sc():
    """Define spacecraft."""
    return scb.Spacecraft(
        "ORCCA", -1234, area=scb.ArrayWUnits(1e-06, km**2), ref_coeff=1.5
    )


@pytest.fixture
def state(epoch, origin, sc):
    """Define StateArray object."""
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000)

    return scb.StateArray(epoch, origin, state=scb.StateDefinition()
                                                .position(sc, pos_0)
                                                .velocity(sc, vel_0),)


@pytest.fixture
def cs_file(earth_config_file):
    """File containing spherical harmonics coefficients."""
    return str(earth_config_file)


def bind(force_model, state, epoch):
    """Initialize ``force_model`` on ``state`` over a day around ``epoch``."""
    et = float(np.asarray(epoch.times.values).ravel()[0])
    force_model.initialize_force_models(state, J2000, et - 86400.0, et + 86400.0)
    return force_model


#: Dict body forms, which pin IAU frames the locked text PCK can orient.
EARTH = {"spice_name": "Earth", "ref_name": "IAU_EARTH", "SPICE_ID": 399}
MOON = {"spice_name": "Moon", "ref_name": "IAU_MOON", "SPICE_ID": 301}


# endregion Fixtures #
# --------------------#

# --------------#
# region Tests #
# --------------#


class TestListForces:
    """
    ``list_forces`` names every force the model applies to its primary body.

    The labels are read back from the live sub-models once the model is
    bound, so they report what the propagation does with the configuration,
    not the configuration itself.
    """

    def test_one_label_per_mode_in_mode_order(self, sc, state, epoch):
        """The listing follows ``dynamics_mode_list`` entry for entry."""
        force_model = bind(
            scb.ForceModelTranslation(
                primary_body=sc, third_bodies=["SUN", "MOON"], cannonball_SRP=True
            ),
            state, epoch,
        )

        labels = force_model.list_forces()

        assert len(labels) == len(force_model.dynamics_mode_list)
        assert [label.split("(")[0] for label in labels] == force_model.dynamics_mode_list
        assert labels == [
            "ThreeBodyGravity(SUN, MOON)",
            "CannonballSRP",
            "PointMassGravity(EARTH)",
        ]

    def test_before_binding_reads_the_configuration(self, sc):
        """Unbound, the origin is unknown and the third bodies are as configured."""
        force_model = scb.ForceModelTranslation(
            primary_body=sc, third_bodies=["SUN", "MOON"], cannonball_SRP=True
        )

        assert force_model.list_forces() == [
            "ThreeBodyGravity(SUN, MOON)",
            "CannonballSRP",
            "PointMassGravity",
        ]

    def test_harmonics_truncation_is_the_clamped_one(self, sc, state, epoch, tmp_path):
        """A field is reported at the truncation its file allows, once bound."""
        cs_file = tmp_path / "earth_2x2.json"
        cs_file.write_text(json.dumps({
            "BodyName": "Earth",
            "ReferenceRadius": 6378.14,
            "GravitationalParameter": 398600.435436,
            "Normalized": False,
            "MaxDegree": 2,
            "Coefficients": [
                {"degree": 0, "order": 0, "Cnm": 1.0, "Snm": 0.0},
                {"degree": 2, "order": 0, "Cnm": -1.08e-3, "Snm": 0.0},
            ],
        }))
        force_model = scb.ForceModelTranslation(
            primary_body=sc,
            sph_harm=True,
            sph_harm_order=8,
            sph_harm_body=EARTH,
            sph_harm_cs_file=str(cs_file),
            sph_harm_norm_flag=False,
        )
        assert force_model.list_forces() == ["SphericalHarmonicsGravity(EARTH, 8x8)"]

        with pytest.warns(UserWarning, match="only goes to degree"):
            bind(force_model, state, epoch)

        assert force_model.list_forces() == ["SphericalHarmonicsGravity(EARTH, 2x2)"]

    def test_every_harmonics_field_is_listed(self, sc, state, epoch, cs_file):
        """A second field is labelled by its own body, not the mode key."""
        force_model = bind(
            scb.ForceModelTranslation(
                primary_body=sc,
                sph_harm=True,
                sph_harm_order=4,
                sph_harm_body=EARTH,
                sph_harm_cs_file=cs_file,
                sph_harm_norm_flag=False,
                sph_harm_bodies=[{"body": MOON, "order": 2, "cs_file": cs_file}],
            ),
            state, epoch,
        )

        assert force_model.list_forces() == [
            "SphericalHarmonicsGravity(EARTH, 4x4)",
            "SphericalHarmonicsGravity(MOON, 2x2)",
        ]

    def test_reports_the_reinstated_point_mass(self, sc, state, epoch, cs_file):
        """A non-central field brings the origin's monopole back at binding."""
        force_model = scb.ForceModelTranslation(
            primary_body=sc,
            sph_harm=True,
            sph_harm_order=2,
            sph_harm_body=MOON,
            sph_harm_cs_file=cs_file,
            sph_harm_norm_flag=False,
        )
        assert "PointMassGravity" not in force_model.list_forces()

        bind(force_model, state, epoch)

        assert force_model.list_forces() == [
            "SphericalHarmonicsGravity(MOON, 2x2)",
            "PointMassGravity(EARTH)",
        ]

    def test_field_body_leaves_the_third_body_term(self, sc, state, epoch, cs_file):
        """A third body with its own field is attracted by the field alone."""
        force_model = bind(
            scb.ForceModelTranslation(
                primary_body=sc,
                third_bodies=["SUN", "MOON"],
                sph_harm=True,
                sph_harm_order=2,
                sph_harm_body=MOON,
                sph_harm_cs_file=cs_file,
                sph_harm_norm_flag=False,
            ),
            state, epoch,
        )

        assert "ThreeBodyGravity(SUN)" in force_model.list_forces()

    def test_integrated_bodies_join_the_third_body_term(self, epoch, origin, sc):
        """A celestial body in the state attracts even when not configured."""
        moon = scb.CelestialBody.from_constants("Moon")
        state = scb.StateArray(
            epoch, origin,
            state=scb.StateDefinition()
            .position(sc, scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000))
            .velocity(sc, scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000))
            .position(moon, scb.ArrayWFrame(np.array([384400.0, 0.0, 0.0]), km, J2000))
            .velocity(moon, scb.ArrayWFrame(np.array([0.0, 1.0, 0.0]), km / sec, J2000)),
        )
        sc_model = bind(
            scb.ForceModelTranslation(primary_body=sc, third_bodies=["SUN"]), state, epoch
        )
        moon_model = bind(
            scb.ForceModelTranslation(primary_body=moon, third_bodies=["SUN"]), state, epoch
        )

        # The spacecraft feels the integrated Moon; the Moon does not feel itself.
        assert "ThreeBodyGravity(SUN, MOON)" in sc_model.list_forces()
        assert "ThreeBodyGravity(SUN)" in moon_model.list_forces()

    def test_parametrised_models_carry_their_settings(self, sc, epoch):
        """Empirical models are told apart by the settings that shape them."""
        force_model = scb.ForceModelTranslation(
            primary_body=sc,
            t0=epoch,
            piecewise_first_order_gauss_markov=True,
            pfogm_beta=1e-3,
            pfogm_batch_length=3600.0,
            pfogm_n_batches=4,
            fourier_acceleration=True,
            fourier_degree=2,
            fourier_frame="RTN",
            fourier_angle="time",
            fourier_period=5400.0,
        )

        labels = force_model.list_forces()

        assert "PiecewiseFirstOrderGaussMarkov(n_batches=4)" in labels
        assert "FourierAcceleration(degree=2, frame=RTN, angle=time)" in labels

    def test_custom_model_is_labelled_by_name(self, sc):
        """A user-defined model appears under its own name."""
        drag = scb.CustomDynamicModel("Drag", lambda velocity: np.zeros(3))
        force_model = scb.ForceModelTranslation(primary_body=sc, custom_models=[drag])

        assert force_model.list_forces() == [
            "CustomDynamicModel(Drag)",
            "PointMassGravity",
        ]



def thermal_model():
    """Two surfaces at constant temperatures, one lit and one in shadow."""
    table = lambda T: {  # noqa: E731
        "angles": scb.ArrayWUnits(np.array([0.0, 180.0]), deg),
        "distances": scb.ArrayWUnits(np.array([1.0]), AU),
        "temperatures": np.array([T, T]),
        "temperature_units": "K",
    }
    return {
        "temperature_tables": {"HOT": table(350.0), "COLD": table(150.0)},
        "surfaces": [
            {"name": "PX", "area": scb.ArrayWUnits(2.0, m**2), "emissivity": 0.8,
             "normal": [1.0, 0.0, 0.0], "temperature": "HOT"},
            {"name": "MX", "area": scb.ArrayWUnits(2.0, m**2), "emissivity": 0.8,
             "normal": [-1.0, 0.0, 0.0], "temperature": "COLD"},
        ],
    }


def antenna_emitters():
    watt = kg * m**2 * sec**-3
    return [
        {"name": "HGA", "power": scb.ArrayWUnits(100.0, watt),
         "hpbw": scb.ArrayWUnits(1.02, deg), "boresight": [0.0, 0.0, 1.0]},
        {"name": "LGA", "power": scb.ArrayWUnits(60.0, watt),
         "hpbw": scb.ArrayWUnits(70.6, deg), "boresight": [0.0, 0.0, -1.0],
         "on_intervals": [[0.0, 3600.0]]},
    ]


class TestEmissionModels:
    """
    Thermal re-radiation and antenna recoil are wired in as ordinary modes:
    configured by keyword, listed, serialized, and differentiated by their
    scale factors ``eta_thermal`` and ``eta_antenna``.
    """

    @pytest.fixture
    def emitter_sc(self):
        """A spacecraft with a mass and an attitude, which both models need."""
        return scb.Spacecraft(
            "ORX", -64, tot_mass=scb.ArrayWUnits(1390.0, kg),
            attitudeMode="inertialFixedAttitude", fixedInertialAttitude=np.eye(3),
        )

    @pytest.fixture
    def emitter_state(self, epoch, origin, emitter_sc):
        pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
        vel_0 = scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000)
        return scb.StateArray(epoch, origin, state=scb.StateDefinition()
                                                    .position(emitter_sc, pos_0)
                                                    .velocity(emitter_sc, vel_0),)

    def emitting(self, sc, **overrides):
        kwargs = dict(
            primary_body=sc,
            thermal_radiation=True, thermal_model=thermal_model(),
            antenna_radiation_pressure=True, antenna_emitters=antenna_emitters(),
        )
        kwargs.update(overrides)
        return scb.ForceModelTranslation(**kwargs)

    @pytest.mark.parametrize("flag", ["thermal_radiation", "antenna_radiation_pressure"])
    def test_enabled_without_a_definition_is_refused(self, emitter_sc, flag):
        with pytest.raises(scb.ForceModelTranslation.MissingEmissionModelError):
            scb.ForceModelTranslation(primary_body=emitter_sc, **{flag: True})
        assert issubclass(scb.ForceModelTranslation.MissingEmissionModelError, ValueError)

    def test_labels(self, emitter_sc, emitter_state, epoch):
        labels = bind(self.emitting(emitter_sc), emitter_state, epoch).list_forces()

        assert "ThermalRadiation(n_surfaces=2)" in labels
        assert "AntennaRadiationPressure(HGA, LGA)" in labels

    def test_json_round_trip(self, emitter_sc, emitter_state, epoch):
        """The temperature tables and emitters survive a trip through JSON."""
        original = self.emitting(emitter_sc)
        loaded = scb.ForceModelTranslation.from_dict(
            json.loads(json.dumps(original.to_dict())), emitter_sc
        )

        assert loaded.dynamics_mode_list == original.dynamics_mode_list
        bind(original, emitter_state, epoch)
        bind(loaded, emitter_state, epoch)
        et = float(np.asarray(epoch.times.values).ravel()[0])
        position = np.array([7000.0, 0.0, 0.0])
        current = {("velocity", -64): np.array([0.0, 7.5, 0.0])}
        for mode in ("ThermalRadiation", "AntennaRadiationPressure"):
            np.testing.assert_allclose(
                loaded.force_models[mode].compute_acceleration(position, current, et, emitter_sc),
                original.force_models[mode].compute_acceleration(position, current, et, emitter_sc),
                rtol=1e-14,
            )

    @pytest.mark.parametrize(
        "parameter, mode",
        [("eta_thermal", "ThermalRadiation"), ("eta_antenna", "AntennaRadiationPressure")],
    )
    def test_scale_factor_partial_is_dispatched(
        self, emitter_sc, emitter_state, epoch, parameter, mode
    ):
        """
        ``acceleration@sc_by_eta@sc`` is the model's own eta partial as a
        column; another spacecraft's scale factor, or a mode that is off, has
        none.
        """
        force_model = bind(self.emitting(emitter_sc), emitter_state, epoch)
        et = float(np.asarray(epoch.times.values).ravel()[0])
        position = scb.ArrayWUnits(np.array([7000.0, 0.0, 0.0]), km)
        current = {("velocity", -64): np.array([0.0, 7.5, 0.0]), (parameter, -64): 1.3}

        partial = force_model._collect_partials(
            f"acceleration@-64_by_{parameter}@-64", position, current, et, emitter_sc
        )
        expected = force_model.force_models[mode]._compute_partial_by_eta(
            position.values, current, et, emitter_sc
        )
        assert partial.values.shape == (3, 1)
        assert np.abs(expected).max() > 0.0
        np.testing.assert_array_equal(partial.values[:, 0], expected)
        assert partial.units == km / sec**2

        assert force_model._collect_partials(
            f"acceleration@-64_by_{parameter}@-99", position, current, et, emitter_sc
        ) is None
        flag = "thermal_radiation" if mode == "ThermalRadiation" else "antenna_radiation_pressure"
        silent = bind(self.emitting(emitter_sc, **{flag: False}), emitter_state, epoch)
        assert silent._collect_partials(
            f"acceleration@-64_by_{parameter}@-64", position, current, et, emitter_sc
        ) is None


# endregion Tests #
# --------------#
