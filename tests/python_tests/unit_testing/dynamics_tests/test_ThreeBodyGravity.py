# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =============================================#
#  Unit Tests for the ThreeBodyGravity class  #
# =============================================#
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import pytest
import os
import subprocess
import sys
import textwrap
import numpy as np

km, kg, sec = scb.Units.get_units(["km", "kg", "sec"])
J2000 = scb.Frame("J2000")


# --------------------#
# region    Fixtures #
# --------------------#
# setup
@pytest.fixture
def epoch():
    """Epoch to define state."""
    return scb.EpochArray("2000 JAN 01 12:00:00.000", "UTC")


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
    """Define spacecraft"""
    return scb.Spacecraft("ORCCA", -1234, area=sc_area, ref_coeff=sc_ref_coeff)


@pytest.fixture
def state(epoch, origin, sc):
    """Define stateArray object"""
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0, 70.0, 0.0]), km / sec, J2000)

    return scb.StateArray(epoch, origin, state=scb.StateDefinition()
                                                .position(sc, pos_0)
                                                .velocity(sc, vel_0),)


@pytest.fixture
def third_body():
    """List of third-body masses affecting the system."""
    return list(scb.CelestialBody.from_constants(["Mars", "Venus"]))


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(state, third_body, origin):
    """
    Verifies that object is constructed correctly.
    """
    # intialize
    # t0/tf bound the ephemeris cache and are required by the constructor
    t0 = state.epoch.et if hasattr(state.epoch, "et") else 0.0
    three_body_gravity = scb.ThreeBodyGravity(
        state.origin, third_body, J2000, t0=float(t0), tf=float(t0) + 86400.0
    )

    # ensure correct properties
    assert three_body_gravity._origin_name == origin.name
    assert three_body_gravity._origin_mu == awu.get_value_in_target_units(
        origin.grav_param, km**3 / sec**2
    )
    assert three_body_gravity._ref_frame == J2000
    assert three_body_gravity._pnt_masses == [b.name for b in third_body]


# NOTE: the rest of this class' methods are tested in integration tests


# ------------------------------------------------------------------#
#  Position and GM partials                                         #
# ------------------------------------------------------------------#
class TestPositionPartials:
    """The SPICE bodies' acceleration is numpy over positions from the Rust
    ephemeris cache. Autograd checks its analytical position and GM partials
    through check_force_model_partials; the autograd backend is chosen at
    import, so those checks run in a ``UseAutoGrad=TRUE`` process."""

    @pytest.fixture
    def tb(self, state, third_body):
        t0 = float(state.epoch.et)
        bodies = third_body + list(scb.CelestialBody.from_constants(["Sun", "Moon"]))
        return scb.ThreeBodyGravity(state.origin, bodies, J2000, t0=t0, tf=t0 + 86400.0)

    @pytest.fixture
    def samples(self, state):
        """Positions near Earth at epochs inside the cached interval."""
        rng = np.random.default_rng(3)
        t0 = float(state.epoch.et)
        epochs = t0 + rng.uniform(0.0, 86400.0, 10)
        positions = rng.uniform(-1.0, 1.0, (10, 3)) * rng.uniform(7e3, 4e5, (10, 1))
        return list(zip(epochs, positions))

    def test_cached_positions_match_single_lookups(self, tb, samples, origin):
        """The batched cache lookup gives, row by row, the single-body lookup."""
        assert [sid for _, sid, _ in tb._cached_bodies] == [499, 299, 10, 301]
        for t, _ in samples:
            single = [
                tb._ephemeris.get_relative_position(sid, origin.spice_id, t)
                for _, sid, _ in tb._cached_bodies
            ]
            np.testing.assert_array_equal(tb._cached_bodies_positions(t), np.array(single))

    def test_gm_state_override(self, tb, samples, sc):
        """A ``("gm", spice_id)`` state entry sets a SPICE body's GM in the
        acceleration, by its GM partial, and in the position partial, as it
        already did for propagated bodies; the cache used to ignore it. The GM
        partials of SPICE bodies used to be zero."""
        t, pos = samples[0]
        a_nominal = tb.compute_acceleration(pos, {}, t, sc)
        # the difference of two total accelerations is only good to eps*|a|
        roundoff = 8.0 * np.finfo(float).eps * np.abs(a_nominal).max()
        for name, sid, gm in tb._cached_bodies:
            state = {("gm", sid): awu(1.001 * gm, km**3 / sec**2)}
            da_dgm = tb._compute_partial_by_gm(pos, {}, t, sc, name).ravel()
            assert np.all(np.abs(0.001 * gm * da_dgm) > 1e3 * roundoff)
            np.testing.assert_allclose(
                tb.compute_acceleration(pos, state, t, sc) - a_nominal,
                0.001 * gm * da_dgm, rtol=1e-9, atol=roundoff,
            )

        state = {("gm", 10): awu(1.001 * tb._cached_bodies[2][2], km**3 / sec**2)}
        assert not np.allclose(
            tb._compute_partial_by_position(pos, state, t, sc),
            tb._compute_partial_by_position(pos, {}, t, sc),
            rtol=1e-6, atol=0.0,
        )

    @pytest.mark.parametrize("case, labels", [
        ("cached", {"x", "y", "z", "gm@499", "gm@299", "gm@10", "gm@301"}),
        ("propagated", {"x", "y", "z", "gm@499", "gm@10", "gm@301"}),
    ])
    def test_partial_check_under_autograd_backend(self, case, labels, tmp_path):
        """check_force_model_partials matches the analytical position and GM
        partials with autograd at several epochs and positions: with every
        body from the cache, and with the Moon propagated and GM overrides in
        the state."""
        script = textwrap.dedent('''
            import os, sys, types
            import numpy
            import src.scarabaeus as scb
            from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu

            km, sec = scb.Units.get_units(["km", "sec"])
            J2000 = scb.Frame("J2000")
            for kernel in ("data/kernels/locked/lsk/naif0012.tls",
                           "data/kernels/locked/spk/de430.bsp",
                           "data/kernels/scenario/mar097.bsp"):
                scb.SpiceManager.load_kernel(kernel)

            case, out_dir = sys.argv[1:]
            t0 = scb.SpiceManager.cal2et("2000 JAN 01 12:00:00.000")
            mars, venus, sun, moon = scb.CelestialBody.from_constants(
                ["Mars", "Venus", "Sun", "Moon"]
            )
            if case == "cached":
                bodies, propagated = [mars, venus, sun, moon], []
            else:
                bodies, propagated = [mars, sun, moon], [moon]
            tb = scb.ThreeBodyGravity(
                scb.CelestialBody.from_constants("Earth"), bodies, J2000,
                t0=t0, tf=t0 + 86400.0, propagated_bodies=propagated,
            )
            fm = types.SimpleNamespace(
                force_models={"ThreeBodyGravity": tb},
                _dynamics_mode_list=["ThreeBodyGravity"],
                _custom_models={},
            )

            os.chdir(out_dir)
            rng = numpy.random.default_rng(3)
            epochs = t0 + rng.uniform(0.0, 86400.0, 5)
            positions = rng.uniform(-1.0, 1.0, (5, 3)) * rng.uniform(7e3, 4e5, (5, 1))
            for t, pos in zip(epochs, positions):
                state = {}
                if case == "propagated":
                    state = {
                        ("position", 301): numpy.array(
                            scb.SpiceManager.get_pos_values("MOON", t, J2000, "EARTH")
                        ),
                        ("gm", 10): awu(1.001 * sun.grav_param.values, km**3 / sec**2),
                        ("gm", 301): 0.999 * moon.grav_param.values,
                    }
                scb.ForceModelTranslation._check_all_partials(
                    fm, awu(pos, km), state, t, scb.Spacecraft("x", -1234),
                )
        ''')
        repo = os.path.dirname(os.path.dirname(os.path.dirname(scb.__file__)))
        env = {**os.environ, "UseAutoGrad": "TRUE", "PYTHONPATH": repo}
        run = subprocess.run(
            [sys.executable, "-c", script, case, str(tmp_path)], cwd=repo, env=env,
            capture_output=True, text=True,
        )
        assert run.returncode == 0, run.stderr

        lines = [line.split(",") for line in (tmp_path / "partials_FMoutput.txt").read_text().splitlines()]
        assert {line[1] for line in lines} == {"ThreeBodyGravity"}
        assert {line[2].split("/d(")[1].rstrip(")") for line in lines} == labels
        assert len(lines) == 5 * (9 + 3 * (len(labels) - 3))
        # A distant body's GM partial d/|d|^3 - b/|b|^3 cancels to ~|r|/|b| of
        # each term, so the two evaluations of it agree only to ~1e-10
        for line in lines:
            rel = 1e-9 if "/d(gm@" in line[2] else 1e-12
            assert float(line[3]) == pytest.approx(float(line[4]), rel=rel, abs=0.0)


# ------------------------------------------------------------------#
#  Third-body name resolution (ForceModelTranslation coercion)      #
# ------------------------------------------------------------------#
class TestThirdBodyCoercion:
    def test_accepts_names_and_objects(self):
        """third_bodies must accept plain names as well as CelestialBody objects.

        Third bodies used to be given as names and resolved through the constants
        library; when they became CelestialBody objects every name-based caller
        broke with ``AttributeError: 'str' object has no attribute 'name'`` deep
        inside ThreeBodyGravity. Names are resolved at the ForceModelTranslation
        boundary, so both spellings reach the model identically.
        """
        coerce = scb.ForceModelTranslation._coerce_third_bodies

        from_names = coerce(["MERCURY", "VENUS", "SUN"])
        assert all(isinstance(b, scb.CelestialBody) for b in from_names)
        assert [b.spice_id for b in from_names] == [199, 299, 10]

        # Objects pass through untouched, and a mixed list is fine
        venus = scb.CelestialBody.from_constants("VENUS")
        mixed = coerce(["MERCURY", venus])
        assert mixed[1] is venus
        assert mixed[0].spice_id == 199

        # A name and its object resolve to the same body and gravitational parameter
        assert from_names[1].spice_id == venus.spice_id
        assert from_names[1].grav_param.values == venus.grav_param.values

        # Empty inputs are left alone
        assert coerce(None) is None
        assert coerce([]) == []

    def test_accepts_barycenter_names(self):
        """Barycentre spellings must resolve to the system barycentre.

        The name-based API accepted "<BODY> BARYCENTER" (and the underscore and
        run-together spellings) before third bodies became objects, by stripping the
        suffix. Callers still pass them, so all spellings resolve here.
        """
        coerce = scb.ForceModelTranslation._coerce_third_bodies

        for spelling in ["MARS BARYCENTER", "MARS_BARYCENTER", "MARSBARYCENTER",
                         "mars barycenter", "  Mars_Barycenter  "]:
            body = coerce([spelling])[0]
            assert isinstance(body, scb.CelestialBody)
            assert body.spice_id == 4, f"{spelling!r} -> {body.spice_id}"

        # Other systems resolve to their own barycentre, and the plain name still
        # resolves to the body centre.
        assert coerce(["JUPITER BARYCENTER"])[0].spice_id == 5
        assert coerce(["EARTH BARYCENTER"])[0].spice_id == 3
        assert coerce(["MARS"])[0].spice_id == 499

        # Mixed with objects
        venus = scb.CelestialBody.from_constants("VENUS")
        assert [b.spice_id for b in coerce(["MARS BARYCENTER", venus])] == [4, 299]

    def test_rejects_unknown_entries(self):
        """An unresolvable name or a wrong type must say so plainly."""
        coerce = scb.ForceModelTranslation._coerce_third_bodies

        with pytest.raises(
            scb.ForceModelTranslation.UnknownThirdBodyError, match="constants library"
        ):
            coerce(["NOT_A_PLANET"])

        # ...including when it carries a BARYCENTER suffix
        with pytest.raises(
            scb.ForceModelTranslation.UnknownThirdBodyError, match="constants library"
        ):
            coerce(["NOT_A_PLANET BARYCENTER"])

        with pytest.raises(
            scb.ForceModelTranslation.UnknownThirdBodyError, match="int"
        ):
            coerce([399])
