# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ======================================================#
#  Unit Tests for the SphericalHarmonicsGravity class  #
# ======================================================#
import src.scarabaeus as scb
from scarabaeus.rust import SphericalHarmonicsCore
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import pytest
import numpy as np
import json
import math
import spiceypy as sp

km, kg, sec, N = scb.Units.get_units(["km", "kg", "sec", "N"])
J2000 = scb.Frame("J2000")

# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture
def order():
    """Order of spherical harmonics expansion."""
    return 3


@pytest.fixture
def cs_file(earth_config_file):
    """File containing spherical harmonics coefficients."""
    return str(earth_config_file)


@pytest.fixture
def body():
    """Central body for spherical harmonics expansion."""
    return scb.CelestialBody.from_constants("Earth")


@pytest.fixture
def body_dict():
    """Dict-form body that enables the SPICE frame-transform path in SHG methods."""
    return {"spice_name": "Earth", "ref_name": "IAU_EARTH", "SPICE_ID": 399}


@pytest.fixture
def epoch():
    """Epoch to define state."""
    return scb.EpochArray("2000 JAN 01 12:00:00.000", "UTC")


@pytest.fixture
def et():
    """SPICE ephemeris time (float) corresponding to J2000 epoch."""
    return scb.SpiceManager.cal2et('2000 JAN 01 12:00:00.000')


@pytest.fixture
def origin():
    """Origin body to define state."""
    return scb.CelestialBody.from_constants("Earth")


@pytest.fixture
def sc_area():
    return awu(1e-06, km**2)


@pytest.fixture
def sc_ref_coeff():
    return 1.5


# construction
@pytest.fixture
def sc(sc_area, sc_ref_coeff):
    """Define spacecraft."""
    return scb.Spacecraft("ORCCA", -1234, area=sc_area, ref_coeff=sc_ref_coeff)


@pytest.fixture
def state(epoch, origin, sc):
    """Define StateArray object."""
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0, 70.0, 0.0]), km / sec, J2000)

    return scb.StateArray(epoch, origin, state=scb.StateDefinition()
                                                .position(sc, pos_0)
                                                .velocity(sc, vel_0),)


@pytest.fixture
def pos():
    """Position vector in km (J2000 frame) at r = 7000 km."""
    return np.array([7000.0, 0.0, 0.0])


@pytest.fixture
def sgh(order, cs_file, body, state):
    """SHG instance with CelestialBody body; use explicit R_b2rf to bypass SPICE dict access."""
    return scb.SphericalHarmonicsGravity(order, cs_file, body, state, base_frame=J2000)


@pytest.fixture
def sgh_dict(order, cs_file, body_dict, state):
    """SHG instance with dict body; supports full SPICE-based frame transforms."""
    return scb.SphericalHarmonicsGravity(order, cs_file, body_dict, state, base_frame=J2000)


@pytest.fixture
def sgh_earth(body_dict, state):
    """Order-8 model with the real Earth field (needed so pole/coefficient
    partials are non-negligible for the partial checks)."""
    cs = "data/dynamic_setup/sph_coefficients/Earth_100.json"
    return scb.SphericalHarmonicsGravity(8, cs, body_dict, state, base_frame=J2000)


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(origin, order, cs_file, body, state):
    """
    Verifies that object is constructed correctly.
    """
    # intialize
    spher_harm_gravity = scb.SphericalHarmonicsGravity(
        order, cs_file, body, state, base_frame=J2000
    )

    # ensure correct properties
    assert spher_harm_gravity._order == order
    assert spher_harm_gravity._cs_file == cs_file
    assert spher_harm_gravity._body == body
    assert spher_harm_gravity._origin == origin
    assert spher_harm_gravity._origin_name == origin.name
    assert spher_harm_gravity._ref_frame == J2000


@pytest.mark.parametrize("spice_name", ["Earth", "EARTH", "earth"])
def test_origin_match_ignores_case(order, cs_file, state, recwarn, spice_name):
    """A field body spelled in another case is still the propagation origin."""
    body = {"spice_name": spice_name, "ref_name": "IAU_EARTH", "SPICE_ID": 399}

    model = scb.SphericalHarmonicsGravity(order, cs_file, body, state, base_frame=J2000)

    assert model._sph_body_is_origin
    assert not [w for w in recwarn if "propagation origin is" in str(w.message)]


def test_other_body_still_warns(order, cs_file, state):
    """A field body that is not the origin keeps its warning."""
    body = {"spice_name": "MOON", "ref_name": "IAU_MOON", "SPICE_ID": 301}

    with pytest.warns(RuntimeWarning, match="propagation origin is 'Earth'"):
        model = scb.SphericalHarmonicsGravity(order, cs_file, body, state, base_frame=J2000)

    assert not model._sph_body_is_origin


class TestMultipleHarmonicsFields:
    """
    Tests for a force model carrying more than one harmonic field.

    Flying at Jupiter means the central body and several moons all have a
    field worth modelling, and they act on the same spacecraft. Each field is
    registered as its own entry in the dynamics mode list, so the accumulation
    and partial machinery — which is generic over that list — picks all of them
    up, and each body's coefficients stay separately estimable.
    """

    @staticmethod
    def build(sc, cs_file, primary_field, extra_fields):
        """A force model with one singular field plus ``extra_fields``."""
        return scb.ForceModelTranslation(
            primary_body=sc,
            sph_harm=True,
            sph_harm_order=4,
            sph_harm_body=primary_field,
            sph_harm_cs_file=cs_file,
            sph_harm_norm_flag=True,
            sph_harm_bodies=extra_fields,
        )

    @staticmethod
    def state(sc, origin, epoch):
        """A spacecraft state about ``origin``."""
        position = scb.ArrayWFrame(np.array([25000.0, 8000.0, 3000.0]), km, J2000)
        velocity = scb.ArrayWFrame(np.array([0.0, 4.0, 0.5]), km / sec, J2000)
        return scb.StateArray(
            epoch, origin,
            state=scb.StateDefinition().position(sc, position).velocity(sc, velocity),
        )

    #: Dict body forms, which pin IAU frames the locked text PCK can orient.
    #: The CelestialBody form resolves Earth to ITRF93, whose binary PCK the
    #: test kernel set does not carry.
    EARTH = {"spice_name": "Earth", "ref_name": "IAU_EARTH", "SPICE_ID": 399}
    MOON = {"spice_name": "Moon", "ref_name": "IAU_MOON", "SPICE_ID": 301}

    def make(self, sc, cs_file, body, epoch):
        """A two-field model: the origin's own field plus the Moon's."""
        force_model = self.build(
            sc, cs_file, self.EARTH,
            [{"body": self.MOON, "order": 2, "cs_file": cs_file}],
        )
        et = float(np.asarray(epoch.times.values).ravel()[0])
        force_model.initialize_force_models(
            self.state(sc, body, epoch), J2000, et - 86400.0, et + 86400.0
        )
        return force_model, self.MOON, et

    def test_each_field_gets_its_own_mode(self, sc, cs_file, body, epoch):
        """Both fields are registered, under distinct keys."""
        force_model, _, _ = self.make(sc, cs_file, body, epoch)
        assert "SphericalHarmonicsGravity" in force_model.dynamics_mode_list
        assert "SphericalHarmonicsGravity[MOON]" in force_model.dynamics_mode_list
        assert [m._sph_body_name() for m in force_model._sph_models()] == [
            "Earth", "Moon"
        ]

    def test_acceleration_is_the_sum_of_the_fields(self, sc, cs_file, body, epoch):
        """The total picks up every field, not just the first."""
        # Arrange
        force_model, _, et = self.make(sc, cs_file, body, epoch)
        r = np.array([25000.0, 8000.0, 3000.0])
        current = {("position", sc.spice_id): r}

        # Act
        total = force_model._compute_total_acceleration(
            scb.ArrayWUnits(r, km), current, et, sc
        )
        parts = [
            np.asarray(model.compute_acceleration(r, current, et, sc))
            for model in force_model._sph_models()
        ]

        # Assert
        assert np.allclose(total, sum(parts), rtol=1e-12, atol=0.0)

    def test_coefficient_partials_are_keyed_by_body(self, sc, cs_file, body, epoch):
        """Each field's Cnm partial answers for its own body only."""
        # Arrange
        force_model, moon, et = self.make(sc, cs_file, body, epoch)
        r = np.array([25000.0, 8000.0, 3000.0])
        position = scb.ArrayWUnits(r, km)
        current = {("position", sc.spice_id): r}

        # Act
        partials = {
            owner["spice_name"]: force_model._collect_partials(
                f"acceleration@{sc.spice_id}_by_cnm_2_0@{owner['SPICE_ID']}",
                position, current, et, sc,
            )
            for owner in (self.EARTH, moon)
        }

        # Assert
        assert all(value is not None for value in partials.values()), (
            f"a field's Cnm partial did not resolve: {partials}"
        )
        assert not np.allclose(
            np.asarray(partials["Earth"].values).ravel(),
            np.asarray(partials["Moon"].values).ravel(),
        ), "both bodies returned the same partial, so one field shadowed the other"

    def test_a_body_may_hold_only_one_field(self, sc, cs_file, body):
        """Two fields for one body would double-count its gravity."""
        with pytest.raises(scb.ForceModelTranslation.SphericalHarmonicsSpecError):
            self.build(sc, cs_file, body,
                       [{"body": body, "order": 2, "cs_file": cs_file}])

    def test_list_form_works_without_the_singular_keywords(self, sc, cs_file, body):
        """``sph_harm_bodies`` alone describes a model."""
        force_model = scb.ForceModelTranslation(
            primary_body=sc,
            sph_harm_bodies=[{"body": body, "order": 3, "cs_file": cs_file}],
        )
        assert "SphericalHarmonicsGravity" in force_model.dynamics_mode_list

    def test_every_field_may_be_listed_together(self, sc, cs_file):
        """``sph_harm=True`` takes the central body from the list like any other."""
        # Arrange / Act
        force_model = scb.ForceModelTranslation(
            primary_body=sc,
            sph_harm=True,
            sph_harm_bodies=[
                {"body": self.EARTH, "order": 8, "cs_file": cs_file,
                 "norm_flag": True},
                {"body": self.MOON, "order": 2, "cs_file": cs_file,
                 "norm_flag": True},
            ],
        )

        # Assert
        assert force_model.dynamics_mode_list == [
            "SphericalHarmonicsGravity", "SphericalHarmonicsGravity[MOON]"
        ]
        # The first listed field is the central body's, so its monopole still
        # stands in for point-mass gravity.
        assert "PointMassGravity" not in force_model.dynamics_mode_list

    def test_enabled_with_no_field_described_is_reported(self, sc):
        """``sph_harm=True`` alone says what is missing rather than passing."""
        with pytest.raises(
            scb.ForceModelTranslation.SphericalHarmonicsSpecError,
            match="no harmonic field was described",
        ):
            scb.ForceModelTranslation(primary_body=sc, sph_harm=True)

    def test_partial_singular_field_is_still_reported(self, sc, cs_file):
        """Singular settings without a body are an incomplete field, not none."""
        with pytest.raises(
            scb.ForceModelTranslation.SphericalHarmonicsSpecError,
            match="sph_harm_body",
        ):
            scb.ForceModelTranslation(
                primary_body=sc, sph_harm=True, sph_harm_order=4,
                sph_harm_bodies=[
                    {"body": self.MOON, "order": 2, "cs_file": cs_file}
                ],
            )

    @pytest.mark.parametrize(
        "entry, reason",
        [
            ({"body": None, "order": 2, "cs_file": "f"}, "missing body"),
            ({"body": "x", "order": None, "cs_file": "f"}, "missing order"),
            ({"body": "x", "order": 2}, "missing cs_file"),
            ({"body": "x", "order": 2, "cs_file": "f", "typo": 1}, "unknown key"),
        ],
        ids=["no body", "no order", "no cs_file", "unknown key"],
    )
    def test_incomplete_field_is_reported(self, sc, entry, reason):
        """A field missing a setting is named rather than silently dropped."""
        with pytest.raises(scb.ForceModelTranslation.SphericalHarmonicsSpecError):
            scb.ForceModelTranslation(primary_body=sc, sph_harm_bodies=[entry])


class TestPropagatedHarmonicsBody:
    """
    Tests for a harmonics body that is integrated in the state vector.

    A flyby estimates the target's own position alongside the spacecraft's.
    The field is then centred on a body the filter is moving, so its position
    has to come from the state rather than the ephemeris, and the acceleration
    it causes has to show up in the partial with respect to that position.
    Neither held before, and the target could also be handed its own field —
    an acceleration some 5e8 times the Sun's pull on it.
    """

    #: SPICE id of the dict-form Earth the tests below use. The dict form pins
    #: IAU_EARTH, whose orientation the locked text PCK covers; the
    #: CelestialBody form resolves to ITRF93 and would need a binary PCK the
    #: test kernel set does not carry.
    BODY_ID = 399

    @staticmethod
    def build(cs_file, sph_body, origin, propagated, named=None):
        """A harmonics model for ``sph_body`` about ``origin``."""
        return scb.SphericalHarmonicsGravity(
            3, cs_file, sph_body, origin, True, 100, J2000,
            propagated_bodies=[named] if propagated else None,
        )

    def test_position_is_read_from_the_state(self, cs_file, body, body_dict, et):
        """An estimated body's position comes from the state, not SPICE."""
        # Arrange
        sun = scb.CelestialBody.from_constants("Sun")
        r_sc = np.array([-1.11e8, 8.91e7, 3.87e7])
        ephemeris = np.asarray(
            scb.SpiceManager.get_pos_values("EARTH", et, J2000, "SUN")
        ).ravel()[:3]
        estimated = ephemeris + np.array([5.0e5, -3.0e5, 2.0e5])

        # Act
        from_ephemeris = self.build(cs_file, body_dict, sun, propagated=False)
        from_state = self.build(cs_file, body_dict, sun, propagated=True, named=body)
        a_ephemeris = np.asarray(
            from_ephemeris.compute_acceleration(r_sc, {}, et, None)
        )
        a_state = np.asarray(
            from_state.compute_acceleration(
                r_sc, {("position", self.BODY_ID): estimated}, et, None
            )
        )

        # Assert. The accelerations are ~1e-11, far under np.allclose's default
        # absolute tolerance, so the comparison has to be purely relative.
        assert not np.allclose(a_state, a_ephemeris, rtol=1e-6, atol=0.0), (
            "the harmonics body's position was still taken from the ephemeris"
        )

    def test_cross_partial_matches_dual_numbers(self, cs_file, body, body_dict, et):
        """d a / d r_body matches Rust dual-number differentiation.

        For a field centred away from the origin it is the negative of the
        same-body position partial *plus* the Jacobian of the indirect term,
        which depends on the body's position alone. The two were equal and
        opposite back when the point-mass list carried the indirect half and
        the body's attraction was counted twice as a result.
        """
        # Arrange
        sun = scb.CelestialBody.from_constants("Sun")
        r_sc = np.array([-1.11e8, 8.91e7, 3.87e7])
        r_body = np.asarray(
            scb.SpiceManager.get_pos_values("EARTH", et, J2000, "SUN")
        ).ravel()[:3]
        sph = self.build(cs_file, body_dict, sun, propagated=True, named=body)
        state = {("position", self.BODY_ID): r_body}

        # Act
        analytic = sph._compute_partial_by_sph_body_position(r_sc, state, et, None)

        # The field moves with the body, so the direct half is minus the
        # position Jacobian, here differentiated in Rust with dual numbers.
        R_b2rf = sph._get_R_b2rf_effective(state, et)
        r_bf = sph._rf_transform(r_sc, et, R_b2rf, state)
        jac_bf = np.asarray(
            sph._rust_core.compute_jacobian_ad_bf(r_bf, sph.mu, sph.Cnm, sph.Snm)
        ).reshape(3, 3)
        direct = -sph._jacobian_pos_transform_rf(jac_bf, et, R_b2rf)
        # The indirect half is the monopole Jacobian about the body's own position.
        r_norm = np.linalg.norm(r_body)
        indirect = -sph._mu * (
            np.eye(3) / r_norm**3 - 3.0 * np.outer(r_body, r_body) / r_norm**5
        )

        # Assert
        scale = np.maximum(np.abs(direct), np.abs(indirect))
        relative = np.abs(analytic - (direct + indirect)) / scale
        assert relative.max() < 1e-10, f"cross partial off by {relative.max()}"

    def test_ephemeris_body_has_no_cross_partial(self, cs_file, body_dict, et):
        """A body read from the ephemeris is not a solve-for."""
        sph = self.build(
            cs_file, body_dict, scb.CelestialBody.from_constants("Sun"),
            propagated=False,
        )
        r_sc = np.array([-1.11e8, 8.91e7, 3.87e7])
        assert sph._compute_partial_by_sph_body_position(r_sc, {}, et, None) is None

    def test_missing_state_entry_is_reported(self, cs_file, body, body_dict, et):
        """A propagated body absent from the state is not silently ephemeris-read."""
        sph = self.build(
            cs_file, body_dict, scb.CelestialBody.from_constants("Sun"),
            propagated=True, named=body,
        )
        r_sc = np.array([-1.11e8, 8.91e7, 3.87e7])
        with pytest.raises(
            scb.SphericalHarmonicsGravity.PropagatedBodyStateMissingError
        ):
            sph.compute_acceleration(r_sc, {}, et, None)

    def test_body_cannot_feel_its_own_field(self, cs_file, body):
        """Harmonics on the force model of the field body itself is refused."""
        force_model = scb.ForceModelTranslation(
            primary_body=body,
            sph_harm=True,
            sph_harm_order=3,
            sph_harm_body=body,
            sph_harm_cs_file=cs_file,
            sph_harm_norm_flag=True,
        )
        epoch = scb.EpochArray("2021 JAN 01 00:00:00.000", "UTC")
        sun = scb.CelestialBody.from_constants("Sun")
        position = scb.ArrayWFrame(np.array([-1.11e8, 8.91e7, 3.87e7]), km, J2000)
        velocity = scb.ArrayWFrame(np.array([-19.0, -16.0, -6.5]), km / sec, J2000)
        state = scb.StateArray(
            epoch,
            sun,
            state=scb.StateDefinition().position(body, position).velocity(
                body, velocity
            ),
        )
        with pytest.raises(scb.ForceModelTranslation.SelfGravityError):
            force_model.initialize_force_models(state, J2000, epoch, epoch)


class TestCentralBodyIsNotDropped:
    """
    Tests that a harmonic field for one body does not remove another's gravity.

    ``ForceModelTranslation`` suppresses point-mass gravity when a full-field
    model is configured, on the grounds that the field already carries the
    central body's monopole. That only holds when the field is a field of the
    origin. With spherical harmonics for the Earth about a heliocentric origin
    the suppression left the propagation with no central body at all: the Sun's
    acceleration is roughly 1e5 times what the Earth term replaced it with, so
    the spacecraft simply coasted away.
    """

    @staticmethod
    def build(sc, cs_file, sph_body, origin, epoch):
        """A force model with harmonics for ``sph_body`` about ``origin``."""
        force_model = scb.ForceModelTranslation(
            primary_body=sc,
            sph_harm=True,
            sph_harm_order=3,
            sph_harm_body=sph_body,
            sph_harm_cs_file=cs_file,
            sph_harm_norm_flag=True,
        )
        position = scb.ArrayWFrame(np.array([-1.11e8, 8.91e7, 3.87e7]), km, J2000)
        velocity = scb.ArrayWFrame(np.array([-20.7, -16.8, -6.6]), km / sec, J2000)
        state = scb.StateArray(
            epoch,
            origin,
            state=scb.StateDefinition().position(sc, position).velocity(sc, velocity),
        )
        force_model.initialize_force_models(state, J2000, epoch, epoch)
        return force_model

    def test_offset_body_keeps_point_mass_gravity(self, sc, cs_file, body, epoch):
        """Earth harmonics about the Sun must not delete the Sun."""
        force_model = self.build(
            sc, cs_file, body, scb.CelestialBody.from_constants("Sun"), epoch
        )
        assert "PointMassGravity" in force_model.dynamics_mode_list
        assert "SphericalHarmonicsGravity" in force_model.dynamics_mode_list

    def test_origin_body_still_suppresses_point_mass_gravity(
        self, sc, cs_file, body, epoch
    ):
        """Earth harmonics about the Earth already carry the monopole."""
        force_model = self.build(sc, cs_file, body, body, epoch)
        assert "PointMassGravity" not in force_model.dynamics_mode_list
        assert "SphericalHarmonicsGravity" in force_model.dynamics_mode_list

    def test_dict_form_body_is_recognised(self, sc, cs_file, body_dict, epoch):
        """The raw dict body form is compared by name like a CelestialBody."""
        about_earth = self.build(
            sc, cs_file, body_dict, scb.CelestialBody.from_constants("Earth"), epoch
        )
        about_sun = self.build(
            sc, cs_file, body_dict, scb.CelestialBody.from_constants("Sun"), epoch
        )
        assert "PointMassGravity" not in about_earth.dynamics_mode_list
        assert "PointMassGravity" in about_sun.dynamics_mode_list

    def test_no_harmonics_is_unaffected(self, sc):
        """A model without a full field keeps point-mass gravity as before."""
        force_model = scb.ForceModelTranslation(primary_body=sc)
        assert force_model.dynamics_mode_list == ["PointMassGravity"]


class TestHarmonicsBodyIsNotAlsoAPointMass:
    """
    Tests that a body is attracted once when it is both a field and a third body.

    A harmonic field carries its body's monopole (C00 is forced to 1), so
    listing that body in ``third_bodies`` as well counted its attraction twice.
    Ops configurations do exactly that: Earth harmonics alongside a third-body
    list that already names the Earth. The double term measured a ratio of
    1.0000 against Earth's monopole.

    The point-mass list gives the body up, and the harmonics model carries the
    indirect (frame) half of the perturbation that the third-body term used to
    supply, so the total is unchanged apart from the shape terms themselves.
    """

    @staticmethod
    def build(sc, origin, epoch, et, cs_file=None, sph_body=None, third_bodies=None):
        """A force model, with harmonics for ``sph_body`` only when given.

        ``et`` bounds the propagation interval: ThreeBodyGravity spans it to
        build its ephemeris cache and needs raw seconds, not an EpochArray.
        """
        harmonics = (
            {}
            if sph_body is None
            else dict(
                sph_harm=True,
                sph_harm_order=3,
                sph_harm_body=sph_body,
                sph_harm_cs_file=cs_file,
                sph_harm_norm_flag=True,
            )
        )
        force_model = scb.ForceModelTranslation(
            primary_body=sc, third_bodies=third_bodies, **harmonics
        )
        position = scb.ArrayWFrame(np.array([-1.11e8, 8.91e7, 3.87e7]), km, J2000)
        velocity = scb.ArrayWFrame(np.array([-20.7, -16.8, -6.6]), km / sec, J2000)
        state = scb.StateArray(
            epoch,
            origin,
            state=scb.StateDefinition().position(sc, position).velocity(sc, velocity),
        )
        force_model.initialize_force_models(state, J2000, et, et)
        return force_model

    @staticmethod
    def total_acceleration(force_model, position, epoch):
        """Sum every active model's acceleration at ``position``."""
        return sum(
            np.asarray(
                force_model.force_models[mode].compute_acceleration(
                    position, {}, epoch, None
                )
            ).ravel()
            for mode in force_model.dynamics_mode_list
        )

    def test_field_body_leaves_the_point_mass_list(self, sc, cs_file, body, epoch, et):
        """The harmonics body is no longer handed to ThreeBodyGravity."""
        sun = scb.CelestialBody.from_constants("Sun")
        venus = scb.CelestialBody.from_constants("Venus")
        force_model = self.build(
            sc, sun, epoch, et, cs_file, body, third_bodies=[body, venus]
        )
        remaining = [
            pm.name.upper() for pm in force_model._point_masses_less_sph_bodies()
        ]
        assert "EARTH" not in remaining
        assert "VENUS" in remaining

    def test_other_point_masses_are_untouched(self, sc, cs_file, body, epoch, et):
        """Only the field's own body is dropped."""
        sun = scb.CelestialBody.from_constants("Sun")
        venus = scb.CelestialBody.from_constants("Venus")
        no_harmonics = self.build(sc, sun, epoch, et, third_bodies=[venus])
        assert [pm.name.upper() for pm in
                no_harmonics._point_masses_less_sph_bodies()] == ["VENUS"]

    def test_total_acceleration_is_not_double_counted(
        self, sc, cs_file, body, body_dict, epoch, et
    ):
        """
        Adding harmonics for a third body must not add a second monopole.

        Against the same configuration without harmonics, the only change
        allowed is the field's own shape terms, which at 1 AU are ~1e-7 of the
        monopole. Before the fix the difference was the monopole itself.

        The field is the dict-form Earth, which pins IAU_EARTH: the
        CelestialBody form resolves to ITRF93 and would need a binary PCK the
        test kernel set does not carry. It still matches the CelestialBody
        third body, since bodies are compared by name.
        """
        sun = scb.CelestialBody.from_constants("Sun")
        position = np.array([-1.11e8, 8.91e7, 3.87e7])

        plain = self.build(sc, sun, epoch, et, third_bodies=[body])
        with_field = self.build(
            sc, sun, epoch, et, cs_file, body_dict, third_bodies=[body]
        )

        a_plain = self.total_acceleration(plain, position, et)
        a_field = self.total_acceleration(with_field, position, et)

        r_earth = np.asarray(
            scb.SpiceManager.get_pos_values("EARTH", et, J2000, "SUN")
        ).ravel()[:3]
        monopole = float(body.grav_param.values) / np.linalg.norm(
            position - r_earth
        ) ** 2

        assert np.linalg.norm(a_field - a_plain) < 1e-3 * monopole


def test_kronecker_delta():
    """
    Verifies that _kronecker_delta returns 1 when indices are equal and 0 otherwise.
    """
    assert scb.SphericalHarmonicsGravity._kronecker_delta(0, 0) == 1
    assert scb.SphericalHarmonicsGravity._kronecker_delta(3, 3) == 1
    assert scb.SphericalHarmonicsGravity._kronecker_delta(0, 1) == 0
    assert scb.SphericalHarmonicsGravity._kronecker_delta(2, 5) == 0


def test_normalize_coeffs(order, cs_file, body, state):
    """
    The coefficient file is unnormalized (``Normalized: false``); the model
    always converts to normalized form on load for the Rust core, regardless
    of sph_harm_norm_flag. For n=1, m=0 the normalization factor is sqrt(3),
    so the stored value must equal the raw file value divided by sqrt(3).
    """
    with open(cs_file) as f:
        raw = {
            (c["degree"], c["order"]): c["Cnm"]
            for c in json.load(f)["Coefficients"]
        }
    expected = raw[(1, 0)] / np.sqrt(3)

    for flag in (True, False):
        sgh = scb.SphericalHarmonicsGravity(
            order, cs_file, body, state, sph_harm_norm_flag=flag, base_frame=J2000
        )
        assert np.isclose(sgh.Cnm[1, 0], expected, rtol=1e-6)


class TestFrameTransforms:
    def test_compute_bnm_raw(self, sgh, pos, et):
        """
        Verifies B_nm array length and leading term B[0][0] = r_ref / |r| when a
        pre-computed identity rotation bypasses the SPICE frame lookup.

        With identity rotation and Earth as both body and origin, the body-fixed
        position equals the inertial position so B[0][0] = r_ref / r exactly.
        The B_nm recurrence now lives in the rust core.
        """
        R = np.eye(3)
        r_bf = sgh._rf_transform(pos, et, R_b2rf=R)
        bnm_real, _ = sgh._rust_core.compute_bnm(r_bf)

        assert len(bnm_real) == sgh.order + 3

        r = np.linalg.norm(pos)
        assert np.isclose(float(bnm_real[0][0]), sgh.r_ref / r, rtol=1e-6)

    def test_compute_bnm(self, sgh_dict, pos, et):
        """
        Verifies B_nm array length and leading term using the full SPICE frame
        transform.  Rotation is orthogonal so the body-fixed magnitude equals the
        inertial magnitude, and B[0][0] = r_ref / |r| still holds.
        The B_nm recurrence now lives in the rust core.
        """
        r_bf = sgh_dict._rf_transform(pos, et)
        bnm_real, _ = sgh_dict._rust_core.compute_bnm(r_bf)

        assert len(bnm_real) == sgh_dict.order + 3

        r = np.linalg.norm(pos)
        assert np.isclose(float(bnm_real[0][0]), sgh_dict.r_ref / r, rtol=1e-3)

    def test_rf_transform_raw(self, sgh, pos, et):
        """
        Verifies that _rf_transform with an identity rotation and a body that
        coincides with the propagation origin (Earth/Earth) returns the input
        position vector unchanged.
        """
        R = np.eye(3)
        result = sgh._rf_transform(pos, et, R_b2rf=R)

        assert result.shape == (3,)
        assert np.allclose(result, pos, atol=1e-6)

    def test_rf_transform(self, sgh_dict, pos, et):
        """
        Verifies that _rf_transform via SPICE preserves the position vector
        magnitude (rotation is orthogonal).
        """
        result = sgh_dict._rf_transform(pos, et)

        assert result.shape == (3,)
        assert np.isclose(np.linalg.norm(result), np.linalg.norm(pos), rtol=1e-6)

    def test_acceleration_transform_rf_raw(self, sgh, et):
        """
        Verifies that _acceleration_transform_rf with an identity rotation
        returns the input body-fixed acceleration vector unchanged.
        """
        acc_bf = np.array([1e-4, 0.0, -2e-4])
        R = np.eye(3)
        result = sgh._acceleration_transform_rf(acc_bf, et, R_b2rf=R)

        assert result.shape == (3,)
        assert np.allclose(result, acc_bf, atol=1e-12)

    def test_acceleration_transform_rf(self, sgh_dict, et):
        """
        Verifies that _acceleration_transform_rf via SPICE preserves the
        acceleration vector magnitude (orthogonal rotation).
        """
        acc_bf = np.array([1e-4, 0.0, -2e-4])
        result = sgh_dict._acceleration_transform_rf(acc_bf, et)

        assert result.shape == (3,)
        assert np.isclose(np.linalg.norm(result), np.linalg.norm(acc_bf), rtol=1e-6)

    def test_jacobian_pos_transform_rf_raw(self, sgh, et):
        """
        Verifies that _jacobian_pos_transform_rf with an identity rotation is a
        no-op: R @ J @ R^T = J when R = I.
        """
        jac_bf = np.diag([1e-7, 2e-7, 3e-7])
        R = np.eye(3)
        result = sgh._jacobian_pos_transform_rf(jac_bf, et, R_b2rf=R)

        assert result.shape == (3, 3)
        assert np.allclose(result, jac_bf, atol=1e-15)

    def test_jacobian_pos_transform_rf(self, sgh_dict, et):
        """
        Verifies that _jacobian_pos_transform_rf via SPICE preserves the matrix
        trace (orthogonal similarity transforms preserve eigenvalues).
        """
        jac_bf = np.diag([1e-7, 2e-7, 3e-7])
        result = sgh_dict._jacobian_pos_transform_rf(jac_bf, et)

        assert result.shape == (3, 3)
        assert np.isclose(np.trace(result), np.trace(jac_bf), rtol=1e-6)

    def test_jacobian_c_transform_rf(self, sgh, et):
        """
        Verifies that _jacobian_C_transform_rf returns a (3, order+1, order+1)
        array and is a no-op under an identity rotation.
        """
        n = sgh.order + 1
        jac_c_bf = np.zeros((3, n, n))
        jac_c_bf[0, 0, 0] = 1.0
        R = np.eye(3)
        result = sgh._jacobian_C_transform_rf(jac_c_bf, et, R_b2rf=R)

        assert result.shape == (3, n, n)
        assert np.allclose(result, jac_c_bf, atol=1e-15)


class TestPartials:
    @pytest.mark.parametrize("order", [2, 8, 20, 40])
    def test_jacobian_matches_autodiff(self, order):
        """
        Verifies the hand-coded analytical position Jacobian (the ``s1..s6``
        partial formulas in the Rust kernel) against forward-mode automatic
        differentiation of the acceleration, computed in Rust with dual numbers.

        Both come from the rust core via independent code paths
        (``compute_jacobian_bf`` vs ``compute_jacobian_ad_bf``); agreement to
        machine precision confirms the analytical partials are exact. No SPICE
        or Python math is involved.
        """
        rng = np.random.default_rng(order)
        core = SphericalHarmonicsCore(order, 6378.14)
        C = rng.standard_normal((order + 1, order + 1)) * 1e-6
        C[0, 0] = 1.0
        S = rng.standard_normal((order + 1, order + 1)) * 1e-6
        r_bf = np.array([3200.0, -4100.0, 5300.0])
        mu = 398600.4415

        jac_analytical = np.asarray(core.compute_jacobian_bf(r_bf, mu, C, S)[0]).reshape(3, 3)
        jac_autodiff = np.asarray(core.compute_jacobian_ad_bf(r_bf, mu, C, S)).reshape(3, 3)

        rel = np.max(np.abs(jac_analytical - jac_autodiff)) / np.max(np.abs(jac_autodiff))
        assert rel < 1e-12

    #: Epoch for the partial checks, ~3800 days past J2000 so the pole_pm
    #: partial (which scales with the days elapsed) is not negligible.
    CHECK_EPOCH = "2010 JUN 01 12:00:00.000"
    POSITION = np.array([7000.0, 1200.0, -800.0])
    EARTH_FIELD = "data/dynamic_setup/sph_coefficients/Earth_100.json"

    @classmethod
    def _model(cls, sc, origin, base_frame=J2000):
        """Order-8 real Earth field about ``origin``, dict-form body."""
        body_dict = {"spice_name": "Earth", "ref_name": "IAU_EARTH", "SPICE_ID": 399}
        state = scb.StateArray(
            scb.EpochArray(cls.CHECK_EPOCH, "UTC"),
            scb.CelestialBody.from_constants(origin),
            state=scb.StateDefinition()
            .position(sc, scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000))
            .velocity(sc, scb.ArrayWFrame(np.array([0.0, 7.0, 0.0]), km / sec, J2000)),
        )
        return scb.SphericalHarmonicsGravity(
            8, cls.EARTH_FIELD, body_dict, state, base_frame=base_frame
        )

    @staticmethod
    def _check_lines(sh, position, full_state, et, sc):
        """``(independent, analytical, dual-number)`` for every line the
        partials check writes."""
        lines = []
        scb.ForceModelTranslation._check_sph_partials(
            sh, awu(position, km), full_state, et, sc,
            lambda _model, _dep, ind, an, algo: lines.append((ind, an, algo)),
        )
        return lines

    @pytest.mark.parametrize(
        "origin, base_frame, pole",
        [
            pytest.param("Earth", "J2000", False, id="nominal"),
            pytest.param("Earth", "J2000", True, id="estimated-pole"),
            pytest.param("Earth", "ECLIPJ2000", False, id="ecliptic-frame"),
            pytest.param("Moon", "J2000", False, id="non-origin-body"),
        ],
    )
    def test_partials_match_dual_numbers(self, sc, origin, base_frame, pole):
        """
        Verifies every analytical partial the partials check covers (position,
        GM, pole_ra / pole_dec / pole_pm / pole_w0, every Cnm and Snm) against forward-mode
        automatic differentiation of the acceleration, computed in Rust with dual
        numbers. No finite differences.

        Regression guard for fixed bugs, one case each:

        - estimated-pole: the Cnm/Snm partials rotated with the SPICE pole
          instead of the estimated one.
        - ecliptic-frame: the RA/DEC generators acted about the reference-frame
          axes instead of the J2000 ones.
        - non-origin-body: the pole partials used the acceleration including the
          indirect term and the position relative to the origin instead of the
          harmonics body.
        - all: bodvrd returns (dim, values), so the IAU coefficients must be
          indexed [1] (the pole path used to raise IndexError when estimated).
        - all: da/d(pole_ra), da/d(pole_pm) had their sign fitted to the wrong
          IAU rotation (see ``test_iau_pole_rotation_matches_spice``).
        """
        et = scb.SpiceManager.cal2et(self.CHECK_EPOCH)
        sh = self._model(sc, origin, scb.Frame(base_frame))
        position = self.POSITION
        if origin != "Earth":
            position = position + np.asarray(
                scb.SpiceManager.get_pos_values("EARTH", et, J2000, origin.upper())
            ).ravel()[:3]
        sid = sh._sph_body_spice_id
        full_state = (
            {
                ("pole_ra", sid): 0.01,
                ("pole_dec", sid): 1.5,
                ("pole_pm", sid): 361.0,
                ("pole_w0", sid): math.radians(190.0),
            }
            if pole
            else {}
        )

        lines = self._check_lines(sh, position, full_state, et, sc)

        n = sh.order
        expected = {"x", "y", "z", "gm", "pole_ra", "pole_dec", "pole_pm", "pole_w0"}
        expected |= {f"C{i},{j}" for i in range(n + 1) for j in range(i + 1)}
        expected |= {f"S{i},{j}" for i in range(n + 1) for j in range(1, i + 1)}
        assert {ind for ind, _, _ in lines} == expected
        for ind, analytical, algo in lines:
            assert analytical == pytest.approx(algo, rel=1e-10, abs=1e-20), ind

    @pytest.mark.parametrize(
        "origin, base_frame",
        [("Earth", "ECLIPJ2000"), ("Moon", "J2000")],
        ids=["ecliptic-frame", "non-origin-body"],
    )
    def test_dual_number_pole_rotation_is_the_model_rotation(
        self, sc, origin, base_frame
    ):
        """The pole rotation Rust rebuilds for the dual-number pole partials
        reproduces the model's direct acceleration, so the pole check
        differentiates the rotation the model actually applies."""
        et = scb.SpiceManager.cal2et(self.CHECK_EPOCH)
        sh = self._model(sc, origin, scb.Frame(base_frame))
        position = self.POSITION
        if origin != "Earth":
            position = position + np.asarray(
                scb.SpiceManager.get_pos_values("EARTH", et, J2000, origin.upper())
            ).ravel()[:3]
        sid = sh._sph_body_spice_id
        d = et / sp.spd()
        T = d / 36525.0
        ra_c, dec_c, pm_c = sh._iau_ra_coeffs, sh._iau_dec_coeffs, sh._iau_pm_coeffs
        ra = math.radians(ra_c[0] + ra_c[1] * T + ra_c[2] * T**2)
        dec = math.radians(dec_c[0] + dec_c[1] * T + dec_c[2] * T**2)
        pm = float(pm_c[1])
        pole_state = {("pole_ra", sid): ra, ("pole_dec", sid): dec, ("pole_pm", sid): pm}

        a_model = np.asarray(sh.compute_acceleration(position, pole_state, et, sc)).ravel()
        r_rel = position
        if not sh._sph_body_is_origin:
            r_origin = sh._sph_body_position(pole_state, et)
            r_rel = position - r_origin
            a_model = a_model + sh.mu * r_origin / np.linalg.norm(r_origin) ** 3
        M_j2rf = sh._spice_xfrm_cached("J2000", sh._ref_frame.name, et)
        a_rust, _ = sh._rust_core.compute_partial_by_pole_ad_rf(
            r_rel, M_j2rf, ra, dec, float(pm_c[0]), pm, d, sh.mu, sh.Cnm, sh.Snm
        )

        np.testing.assert_allclose(a_rust, a_model, rtol=1e-12, atol=0.0)

    def test_pole_w0_partial_matches_finite_differences(self, sc):
        """da/dW0 is the spin-axis generator acting on the field; it matches a
        central difference of the acceleration in an estimated W0, and W0 at
        its PCK value reproduces the SPICE-oriented field."""
        et = scb.SpiceManager.cal2et(self.CHECK_EPOCH)
        sh = self._model(sc, "Earth")
        sid = sh._sph_body_spice_id
        w0 = math.radians(float(sh._iau_pm_coeffs[0]))

        def acceleration(value):
            state = {("pole_w0", sid): value}
            return np.asarray(sh.compute_acceleration(self.POSITION, state, et, sc)).ravel()

        step = 1e-4
        numeric = (acceleration(w0 + step) - acceleration(w0 - step)) / (2 * step)
        analytical = np.asarray(
            sh._compute_partial_by_pole_W0(self.POSITION, {("pole_w0", sid): w0}, et, sc)
        ).ravel()
        np.testing.assert_allclose(analytical, numeric, rtol=1e-6, atol=0.0)

        nominal = np.asarray(sh.compute_acceleration(self.POSITION, {}, et, sc)).ravel()
        np.testing.assert_allclose(acceleration(w0), nominal, rtol=1e-10, atol=0.0)

    @pytest.mark.parametrize("body", ["EARTH", "MARS"])
    def test_iau_pole_rotation_matches_spice(self, body):
        """The body-fixed to J2000 rotation built from the secular IAU pole
        angles is SPICE's ``IAU_<body>`` frame (pck00010 has no nutation terms
        for either body). Regression guard: the three passive factors used to
        be multiplied untransposed, which is neither this rotation nor its
        inverse, so an estimated pole moved the field by order one."""
        et = scb.SpiceManager.cal2et(self.CHECK_EPOCH)
        d = et / sp.spd()
        T = d / 36525.0
        ra, dec, pm = (sp.bodvrd(body, key, 3)[1] for key in ("POLE_RA", "POLE_DEC", "PM"))
        M = scb.SphericalHarmonicsGravity._build_R_b2j_iau(
            ra[0] + ra[1] * T + ra[2] * T**2,
            dec[0] + dec[1] * T + dec[2] * T**2,
            pm[0] + pm[1] * d + pm[2] * d**2,
        )

        np.testing.assert_allclose(M, sp.pxform(f"IAU_{body}", "J2000", et), atol=1e-11)


# NOTE: the rest of this class' methods are tested in integration tests

# ------------------------------------------------#
# region Body-fixed frame resolution (regression)  #
# ------------------------------------------------#
# The SPH body may be given as a dict ('ref_name' key) or as a CelestialBody
# ('base_frame' attribute). CelestialBody.base_frame is stored exactly as the
# caller supplied it, so it may hold a Frame object or a plain string, while
# SPICE needs a frame name string. Passing a CelestialBody built with a Frame
# object used to raise "'Frame' object has no attribute 'encode'" inside
# spice.pxform, and the Cnm/Snm partials additionally raised TypeError by
# subscripting the CelestialBody as if it were a dict.
class TestBodyFixedFrameResolution:
    IAU_EARTH = "IAU_EARTH"

    def _body(self, base_frame):
        """Earth CelestialBody whose base_frame is supplied verbatim."""
        earth = scb.CelestialBody.from_constants("Earth")

        return scb.CelestialBody(
            name=earth.name,
            mass=earth.mass(),
            mean_radius=earth.mean_radius,
            grav_param=earth.grav_param,
            base_frame=base_frame,
            spice_id=earth.spice_id,
        )

    def _model(self, body, order, cs_file, state):
        return scb.SphericalHarmonicsGravity(
            order, cs_file, body, state, base_frame=J2000
        )

    def test_frame_is_a_string_for_every_body_form(self, order, cs_file, state):
        """The resolved body-fixed frame is always a name SPICE can consume."""
        bodies = (
            {"spice_name": "Earth", "ref_name": self.IAU_EARTH, "SPICE_ID": 399},
            self._body(self.IAU_EARTH),
            self._body(scb.Frame(self.IAU_EARTH)),
        )

        for body in bodies:
            frame = self._model(body, order, cs_file, state)._sph_body_fixed_frame()
            assert isinstance(frame, str)
            assert frame == self.IAU_EARTH

    def test_acceleration_matches_across_body_forms(self, order, cs_file, state, pos, et):
        """A Frame object, a frame string and a dict must give the same acceleration."""
        reference = self._model(
            {"spice_name": "Earth", "ref_name": self.IAU_EARTH, "SPICE_ID": 399},
            order, cs_file, state,
        ).compute_acceleration(pos, {}, et, None)

        for base_frame in (self.IAU_EARTH, scb.Frame(self.IAU_EARTH)):
            model = self._model(self._body(base_frame), order, cs_file, state)
            acceleration = model.compute_acceleration(pos, {}, et, None)
            assert np.allclose(acceleration, reference, rtol=0.0, atol=0.0)

    def test_coefficient_partials_work_on_celestial_body(
        self, order, cs_file, state, pos, et
    ):
        """The Cnm/Snm partials transform without a precomputed DCM."""
        model = self._model(
            self._body(scb.Frame(self.IAU_EARTH)), order, cs_file, state
        )

        expected = (3, order + 1, order + 1)
        assert model._compute_partial_by_C(pos, {}, et).shape == expected
        assert model._compute_partial_by_S(pos, {}, et).shape == expected

    def test_missing_frame_is_reported(self, order, cs_file, state):
        """A body with no body-fixed frame raises a descriptive error."""
        model = self._model(self._body(None), order, cs_file, state)

        with pytest.raises(scb.SphericalHarmonicsGravity.MissingBodyFixedFrameError):
            model._sph_body_fixed_frame()


# endregion Body-fixed frame resolution #
# ------------------------------------------------#


# ------------------------------------------------#
# region Origin input (regression)                 #
# ------------------------------------------------#
# The constructor used to demand a StateArray but read only its origin, which
# forced the model to be built inside a propagation setup. It now takes the
# origin body directly, so the model can be created in a configuration file and
# reused; a StateArray is still accepted as the historical form.
class TestOriginInput:
    def test_body_can_be_supplied_directly(self, order, cs_file, body_dict, body):
        """No state vector is needed to build the model."""
        model = scb.SphericalHarmonicsGravity(
            order, cs_file, body_dict, body, base_frame=J2000
        )

        assert model._origin is body
        assert model._origin_name == body.name

    def test_state_array_is_still_accepted(self, order, cs_file, body_dict, state):
        """The historical StateArray argument keeps working and is unwrapped."""
        model = scb.SphericalHarmonicsGravity(
            order, cs_file, body_dict, state, base_frame=J2000
        )

        assert model._origin is state.origin

    def test_both_forms_agree(self, order, cs_file, body_dict, state, pos, et):
        """Body and StateArray inputs produce the same model."""
        from_body = scb.SphericalHarmonicsGravity(
            order, cs_file, body_dict, state.origin, base_frame=J2000
        )
        from_state = scb.SphericalHarmonicsGravity(
            order, cs_file, body_dict, state, base_frame=J2000
        )

        assert np.array_equal(
            from_body.compute_acceleration(pos, {}, et, None),
            from_state.compute_acceleration(pos, {}, et, None),
        )

    def test_invalid_origin_is_reported(self, order, cs_file, body_dict):
        """Something that is neither a body nor a state vector is refused."""
        with pytest.raises(scb.SphericalHarmonicsGravity.InvalidOriginError):
            scb.SphericalHarmonicsGravity(
                order, cs_file, body_dict, "EARTH", base_frame=J2000
            )


# endregion Origin input #
# ------------------------------------------------#


# ---------------------------------------------------#
# region  Rust core guard                            #
# ---------------------------------------------------#
"""
The body-fixed kernels run in rust and have no Python implementation, so a
missing extension, and a position autograd is tracing, must each report that
plainly. ``compute_acceleration`` used to branch to a
``_compute_acceleration_python`` that was never written, raising
``AttributeError`` instead.

The autograd NumPy backend on its own is not an error: ``UseAutoGrad=TRUE``
propagations evaluate the model on plain arrays, and its partials are checked
by Rust dual numbers. Refusing the backend outright made
every ``UseAutoGrad=TRUE`` run with harmonics fail, so the other models' partials
could not be verified alongside it.

Tests
-----
test_missing_rust_core_is_reported
    A model without the extension raises the documented error.
test_autograd_tracing_is_reported
    A position autograd is tracing raises, naming the dual-number check.
test_autograd_backend_evaluates_the_model
    Under the autograd backend the core is built and plain arrays evaluate
    exactly as without it.
test_partial_check_runs_under_autograd_backend
    check_force_model_partials verifies the harmonics partials under it.
test_partial_check_without_autograd_skips_only_autograd_models
    Without the backend the harmonics partials are still checked and the
    autograd-verified models are skipped with a warning, not a crash.
test_no_python_acceleration_fallback_is_referenced
    Nothing calls a Python kernel implementation that does not exist.
"""

import scarabaeus.dynamics.SphericalHarmonicsGravity as _sph_module


def _bare_model():
    """A model instance without running __init__, to set the core by hand."""
    return _sph_module.SphericalHarmonicsGravity.__new__(
        _sph_module.SphericalHarmonicsGravity
    )


def test_missing_rust_core_is_reported():
    """A model without the extension raises the documented error."""
    model = _bare_model()
    model._rust_core = None

    with pytest.raises(
        _sph_module.SphericalHarmonicsGravity.RustCoreUnavailableError,
        match="rust extension",
    ):
        model._require_rust_core(np.zeros(3))


def test_autograd_tracing_is_reported(sgh_earth, pos, et, sc, monkeypatch):
    """A position autograd is tracing raises, naming the dual-number check,
    instead of failing on the ArrayBox inside a float() cast or the Rust call."""
    from autograd import jacobian
    from autograd.numpy.numpy_boxes import ArrayBox

    monkeypatch.setattr(_sph_module.np, "autograd_types", (ArrayBox,))

    with pytest.raises(
        _sph_module.SphericalHarmonicsGravity.AutogradTracingError,
        match="compute_jacobian_ad_bf",
    ):
        jacobian(lambda p: sgh_earth.compute_acceleration(p, {}, et, sc))(pos)


def test_autograd_backend_evaluates_the_model(
    body_dict, state, pos, et, sc, sgh_earth, monkeypatch
):
    """Under the autograd backend the core is built — at construction and on
    unpickling — and plain arrays evaluate exactly as without it."""
    import pickle
    from autograd.numpy.numpy_boxes import ArrayBox

    expected = np.asarray(sgh_earth.compute_acceleration(pos, {}, et, sc))
    expected_jac = np.asarray(
        sgh_earth._compute_partial_by_position(pos, {}, et, sc)
    )

    monkeypatch.setattr(_sph_module.np, "autograd_types", (ArrayBox,))
    cs = "data/dynamic_setup/sph_coefficients/Earth_100.json"
    built = scb.SphericalHarmonicsGravity(8, cs, body_dict, state, base_frame=J2000)
    restored = pickle.loads(pickle.dumps(built))

    for model in (built, restored):
        assert model._rust_core is not None
        np.testing.assert_array_equal(
            np.asarray(model.compute_acceleration(pos, {}, et, sc)), expected
        )
        np.testing.assert_array_equal(
            np.asarray(model._compute_partial_by_position(pos, {}, et, sc)),
            expected_jac,
        )


def test_partial_check_runs_under_autograd_backend(sgh_earth, pos, et, sc, monkeypatch):
    """check_force_model_partials verifies the harmonics partials with the
    autograd backend active, every reference by Rust dual numbers."""
    from autograd.numpy.numpy_boxes import ArrayBox

    monkeypatch.setattr(_sph_module.np, "autograd_types", (ArrayBox,))
    lines = []

    scb.ForceModelTranslation._check_sph_partials(
        sgh_earth, awu(pos, km), {}, et, sc,
        lambda *args: lines.append(args),
    )

    by_param = {}
    for _, dep, ind, analytical, algo in lines:
        by_param.setdefault(ind, []).append((analytical, algo))

    # exact to roundoff; abs covers partials that vanish at this position
    for ind in ["x", "y", "z", "gm", "pole_ra", "pole_dec", "pole_pm", "pole_w0", "C2,0", "S2,2", "C8,8"]:
        for analytical, algo in by_param[ind]:
            assert analytical == pytest.approx(algo, rel=1e-10, abs=1e-20)


def test_partial_check_without_autograd_skips_only_autograd_models(
    sgh_earth, pos, et, sc, state, tmp_path, monkeypatch
):
    """Without UseAutoGrad the harmonics partials are still checked (Rust dual
    numbers) and the autograd-verified models are skipped with a warning. The
    point-mass comparison used to read autograd results that were never
    computed and raised UnboundLocalError."""
    import types

    monkeypatch.delenv("UseAutoGrad", raising=False)
    monkeypatch.chdir(tmp_path)
    fm = types.SimpleNamespace(
        force_models={
            "SphericalHarmonicsGravity": sgh_earth,
            "PointMassGravity": scb.PointMassGravity(state.origin, J2000),
        },
        _dynamics_mode_list=["SphericalHarmonicsGravity", "PointMassGravity"],
        _custom_models={},
        _check_sph_partials=scb.ForceModelTranslation._check_sph_partials,
    )

    with pytest.warns(UserWarning, match="PointMassGravity partials are not checked"):
        scb.ForceModelTranslation._check_all_partials(fm, awu(pos, km), {}, et, sc)

    models = {
        line.split(",")[1]
        for line in (tmp_path / "partials_FMoutput.txt").read_text().splitlines()
    }
    assert models == {"SphericalHarmonicsGravity"}


def test_no_python_acceleration_fallback_is_referenced():
    """Nothing calls a Python kernel implementation that does not exist."""
    import inspect

    source = inspect.getsource(_sph_module)
    called = "_compute_acceleration_python" in source
    defined = "def _compute_acceleration_python" in source

    assert not called or defined, (
        "compute_acceleration referenced a Python kernel implementation that "
        "the class does not define"
    )
