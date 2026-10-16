# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""The solid tide acting on the gravity field, rather than on a station."""
import numpy as np
import pytest
import spiceypy as spice
import scarabaeus as scb

# Inside the coverage of the binary Earth PCK loaded below (2006-2037).
ET = 7.9e8
POSITION = np.array([8000.0, 3000.0, 5000.0])


@pytest.fixture(scope="module", autouse=True)
def kernels():
    spice.kclear()
    spice.furnsh("data/kernels/locked/lsk/naif0012.tls")
    spice.furnsh("data/kernels/locked/pck/pck00010.tpc")
    spice.furnsh("data/kernels/locked/spk/de432s.bsp")
    # ITRF93 is the Earth body-fixed frame the field rotates into.
    spice.furnsh("data/kernels/locked/spk/earth_060526_370526_predict.bpc")
    yield
    spice.kclear()


@pytest.fixture
def field():
    earth = scb.CelestialBody.from_constants("EARTH")
    def make():
        return scb.SphericalHarmonicsGravity(
            sph_harm_order=4,
            sph_harm_cs_file="data/dynamic_setup/sph_coefficients/earth_ggm03.json",
            sph_harm_body=earth, origin=earth, sph_harm_norm_flag=True,
            base_frame=scb.Frame("J2000"))
    return make


def test_attached_tide_changes_the_acceleration_by_its_coefficient_delta(field):
    """The field with a tide equals the field with the delta added by hand."""
    plain = field()
    nominal = np.asarray(plain.compute_acceleration(POSITION, None, ET, None), dtype=float)

    tide = scb.SolidTide()
    plain.solid_tide = tide
    perturbed = np.asarray(plain.compute_acceleration(POSITION, None, ET, None), dtype=float)

    manual = field()
    Cnm = np.array(manual.Cnm, copy=True)
    Snm = np.array(manual.Snm, copy=True)
    for (degree, order), (dC, dS) in tide.geopotential_coefficients(
            ET, "IAU_EARTH").items():
        Cnm[degree, order] += dC
        Snm[degree, order] += dS
    manual.Cnm, manual.Snm = Cnm, Snm
    by_hand = np.asarray(manual.compute_acceleration(POSITION, None, ET, None), dtype=float)
    # Tolerance well below the tide's own effect (parts in 1e8), so this still
    # discriminates; the residual difference is the coefficient setter's own
    # rounding, at parts in 1e11.
    np.testing.assert_allclose(perturbed, by_hand, rtol=1e-9, atol=0)

    # Real but small: parts in 1e8 of the total acceleration.
    relative = np.linalg.norm(perturbed - nominal) / np.linalg.norm(nominal)
    assert 1e-9 < relative < 1e-6


def test_detaching_the_tide_restores_the_nominal_field_exactly(field):
    model = field()
    nominal = np.asarray(model.compute_acceleration(POSITION, None, ET, None), dtype=float)
    model.solid_tide = scb.SolidTide()
    assert not np.array_equal(
        np.asarray(model.compute_acceleration(POSITION, None, ET, None), dtype=float),
        nominal)
    model.solid_tide = None
    np.testing.assert_allclose(
        np.asarray(model.compute_acceleration(POSITION, None, ET, None), dtype=float),
        nominal, rtol=0, atol=0)


def test_the_tide_is_time_varying(field):
    """A static coefficient offset would be indistinguishable from a bad field."""
    model = field()
    model.solid_tide = scb.SolidTide()
    accelerations = [np.asarray(model.compute_acceleration(
        POSITION, None, ET + hours * 3600.0, None), dtype=float)
        for hours in (0, 6, 12)]
    assert not np.allclose(accelerations[0], accelerations[1], rtol=0, atol=0)
    assert not np.allclose(accelerations[1], accelerations[2], rtol=0, atol=0)


def test_solid_tide_setter_rejects_an_object_without_the_hook(field):
    model = field()
    with pytest.raises(TypeError, match="geopotential_coefficients"):
        model.solid_tide = object()
    model.solid_tide = None
    assert model.solid_tide is None


# --------------------------------------------------------------------------- #
# love_k_n_m as estimated parameters                                          #
# --------------------------------------------------------------------------- #
SID = 399


def test_love_k_is_an_allowed_state_component_per_degree_and_order():
    """One component per (n, m), exactly as the field itself has for cnm/snm."""
    allowed = scb.StateDefinition._generate_allowed_state_dictionary()
    assert "love_k" in allowed
    assert allowed["love_k"][1] == 1
    for name in ("love_k_2_0", "love_k_2_2", "love_k_3_3"):
        assert allowed["love_k"][2].match(name)
    assert not allowed["love_k"][2].match("love_k2")


def test_every_love_k_partial_matches_a_finite_difference(field):
    """All seven nominal k_nm, not just degree 2 order 0."""
    from dataclasses import replace
    tide = scb.SolidTide()
    model = field()
    model.solid_tide = tide
    nominal = np.asarray(model.compute_acceleration(POSITION, None, ET, None),
                         dtype=float)
    for key in sorted(tide.love_k):
        analytic = model._compute_partial_by_love_k(
            key[0], key[1], POSITION, None, ET).ravel()
        step = 1e-3
        outputs = []
        for sign in (+1, -1):
            table = dict(tide.love_k)
            table[key] = table[key] + sign * step
            perturbed = field()
            perturbed.solid_tide = replace(tide, love_k=table)
            outputs.append(np.asarray(perturbed.compute_acceleration(
                POSITION, None, ET, None), dtype=float))
        difference = (outputs[0] - outputs[1]) / (2 * step)
        # The reference here is the finite difference, and its own resolution
        # is set by cancellation: the two accelerations differ in the last few
        # bits of a number ~1e10 times larger than the signal. Compare against
        # that floor rather than against a fixed relative tolerance, or a
        # degree-3 component is being checked against numerical noise.
        floor = np.linalg.norm(nominal) * np.finfo(float).eps / (2 * step)
        np.testing.assert_allclose(analytic, difference, rtol=1e-4, atol=2 * floor)
        assert np.linalg.norm(analytic) > 0


def test_each_love_k_drives_its_own_coefficient_alone(field):
    """The Jacobian is diagonal: k_nm touches (n, m) and nothing else."""
    from dataclasses import replace
    nominal = scb.SolidTide()
    before = nominal.geopotential_coefficients(ET, "IAU_EARTH")
    table = dict(nominal.love_k)
    table[(2, 0)] = 0.9
    after = replace(nominal, love_k=table).geopotential_coefficients(ET, "IAU_EARTH")
    changed = [key for key in before
               if abs(after[key][0] - before[key][0]) > 1e-30
               or abs(after[key][1] - before[key][1]) > 1e-30]
    assert changed == [(2, 0)]


def test_estimated_love_k_reaches_the_forward_acceleration(field):
    """The filter's iterate must drive the model, not the constructor value."""
    model = field()
    model.solid_tide = scb.SolidTide()
    base = np.asarray(model.compute_acceleration(POSITION, None, ET, None), dtype=float)
    step = 1e-3
    raised = np.asarray(model.compute_acceleration(
        POSITION, {("love_k_2_2", SID): model.solid_tide.love_k[(2, 2)] + step},
        ET, None), dtype=float)
    predicted = model._compute_partial_by_love_k(2, 2, POSITION, None, ET).ravel() * step
    np.testing.assert_allclose(raised - base, predicted, rtol=1e-5)


def test_partial_is_zero_without_a_tide_or_outside_the_table(field):
    """Unobservable cases report an honest zero rather than a wrong number."""
    model = field()
    assert model.solid_tide is None
    np.testing.assert_allclose(
        model._compute_partial_by_love_k(2, 0, POSITION, None, ET), 0.0, atol=0.0)
    model.solid_tide = scb.SolidTide()
    # A degree the tide model carries no Love number for.
    np.testing.assert_allclose(
        model._compute_partial_by_love_k(4, 0, POSITION, None, ET), 0.0, atol=0.0)


def test_love_k_partial_is_exactly_linear(field):
    """Linear in k, so the partial cannot depend on the value it is taken at."""
    from dataclasses import replace
    model = field()
    partials = []
    for value in (0.1, 0.29525, 0.9):
        table = dict(scb.SolidTide().love_k)
        table[(2, 0)] = value
        model.solid_tide = replace(scb.SolidTide(), love_k=table)
        partials.append(model._compute_partial_by_love_k(2, 0, POSITION, None, ET))
    np.testing.assert_allclose(partials[0], partials[1], rtol=1e-12)
    np.testing.assert_allclose(partials[1], partials[2], rtol=1e-12)
