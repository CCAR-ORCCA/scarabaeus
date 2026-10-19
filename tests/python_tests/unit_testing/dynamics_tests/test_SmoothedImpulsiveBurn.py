# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =================================================#
#  Unit Tests for the SmoothedImpulsiveBurn class  #
# =================================================#
import src.scarabaeus as scb
import pytest
import numpy as np

kg, km, sec, unitless = scb.Units.get_units(["kg", "km", "sec", "unitless"])
J2000 = scb.Frame("J2000")

SIB = scb.SmoothedImpulsiveBurn

# Shared LEO scenario: a half-orbit arc with the burn at its midpoint.
T0, T_BURN, T_END, DT = 0.0, 1800.0, 3600.0, 10.0
R0 = np.array([7000.0, 0.0, 0.0])
V0 = np.array([0.0, 7.5460491, 0.0])
DV = np.array([0.0, 0.010, 0.005])


# --------------------#
# region    Fixtures #
# --------------------#
@pytest.fixture(scope="module")
def earth():
    """Propagation origin."""
    return scb.CelestialBody.from_constants("EARTH")


@pytest.fixture(scope="module")
def sc():
    """Maneuvering spacecraft."""
    return scb.Spacecraft(
        name="SMOOTH_DV_SC",
        spice_id=-60000,
        tot_mass=scb.ArrayWUnits(1000, kg),
        area=scb.ArrayWUnits(1e-8, km**2),
        ref_coeff=scb.ArrayWUnits(1.2, unitless),
    )


@pytest.fixture
def burn():
    """Nominal impulsive maneuver at T_BURN."""
    return scb.ImpulsiveBurn(
        delta_v=scb.ArrayWFrame(DV, km / sec, J2000), t_burn=T_BURN
    )


@pytest.fixture
def model(sc, burn):
    """Smoothed burn with a pulse the LEO arc comfortably resolves."""
    return SIB(
        spacecraft=sc,
        maneuvers=burn,
        base_frame=J2000,
        t0=T0,
        smoothing_factor=0.5,
    )


# --------------------#
# region    Helpers  #
# --------------------#
def _base_state(sc, r0, v0):
    """Position/velocity state components."""
    return [
        ("position", 3, "estimated", "dynamic", sc, scb.ArrayWFrame(r0, km, J2000)),
        (
            "velocity",
            3,
            "estimated",
            "dynamic",
            sc,
            scb.ArrayWFrame(v0, km / sec, J2000),
        ),
    ]


def _propagate(sc, earth, force_models, state, epochs, with_stm=False):
    """Propagate and return the Propagator."""
    sv = scb.StateArray(
        epochs[0], earth, scb.StateDefinition.from_components(state)
    )
    prop = scb.Propagator(
        primary_body=sc,
        state_vector=sv,
        tspan=epochs,
        force_models=force_models,
        integrator="DOP853",
        atol=1e-13,
        rtol=1e-13,
        propagate_STM=with_stm,
    )
    prop.propagate()
    return prop


def _final_rv(prop):
    """Final position/velocity as a flat 6-vector."""
    return np.concatenate(
        (prop.state.quantity.values[-1, 0:3], prop.state.quantity.values[-1, 3:6])
    )


def _smoothed_state(sc, dv_vec, dt_val):
    """State augmented with the two estimable maneuver parameters."""
    return _base_state(sc, R0, V0) + [
        (
            "dv_smooth1",
            3,
            "estimated",
            "static",
            sc,
            scb.ArrayWFrame(np.asarray(dv_vec, dtype=float), km / sec, J2000),
        ),
        (
            "dt_dv1",
            1,
            "estimated",
            "static",
            sc,
            scb.ArrayWFrame(scb.ArrayWUnits(np.array([float(dt_val)]), sec), None),
        ),
    ]


def _propagate_smoothed(sc, earth, burn, dv_vec, dt_val, with_stm):
    """Propagate the augmented state through the smoothed burn."""
    epochs = scb.EpochArray(np.arange(T0, T_END + DT, DT), "TDB")
    fm = scb.ForceModelTranslation(
        primary_body=sc, smoothed_maneuvers=[burn], dv_smoothing_factor=1.0
    )
    return _propagate(
        sc, earth, fm, _smoothed_state(sc, dv_vec, dt_val), epochs, with_stm
    )


@pytest.fixture(scope="module")
def two_leg_final(sc, earth):
    """
    Exact impulsive reference: gravity-only to the burn, delta-v applied by
    hand, gravity-only onward. This is the answer the smoothed single arc
    has to reproduce.
    """
    grav = scb.ForceModelTranslation(primary_body=sc)

    leg1 = scb.EpochArray(np.arange(T0, T_BURN + DT, DT), "TDB")
    p1 = _propagate(sc, earth, grav, _base_state(sc, R0, V0), leg1)
    r_mid = p1.state.quantity.values[-1, 0:3]
    v_mid = p1.state.quantity.values[-1, 3:6]

    leg2 = scb.EpochArray(np.arange(T_BURN, T_END + DT, DT), "TDB")
    p2 = _propagate(sc, earth, grav, _base_state(sc, r_mid, v_mid + DV), leg2)
    return _final_rv(p2)


@pytest.fixture(scope="module")
def stm_reference(sc, earth):
    """Propagated STM and its parameter column indices, computed once."""
    b = scb.ImpulsiveBurn(
        delta_v=scb.ArrayWFrame(DV, km / sec, J2000), t_burn=T_BURN
    )
    prop = _propagate_smoothed(sc, earth, b, DV, 0.0, with_stm=True)

    n = prop.state.quantity.values.shape[1]
    idx = prop.full_state_vector.index_rep
    cols_dv = list(range(*idx[("dv_smooth1", sc.spice_id)].indices(n)))
    col_dt = list(range(*idx[("dt_dv1", sc.spice_id)].indices(n)))[0]
    return prop.STM.quantity.values[-1].reshape(n, n), cols_dv, col_dt


# endregion Fixtures #
# --------------------#


# ------------------------#
# region  Construction   #
# ------------------------#
class TestConstruction:
    def test_rejects_empty_maneuvers(self, sc):
        """An empty maneuver list is a configuration error, not a silent no-op."""
        with pytest.raises(SIB.NoManeuversError, match="at least one ImpulsiveBurn"):
            SIB(spacecraft=sc, maneuvers=[])

    @pytest.mark.parametrize("bad_a", [0.0, -1.0])
    def test_rejects_nonpositive_smoothing_factor(self, sc, burn, bad_a):
        """A non-positive smoothing factor would divide by zero in the pulse."""
        with pytest.raises(SIB.InvalidSmoothingFactorError, match="strictly positive"):
            SIB(spacecraft=sc, maneuvers=burn, smoothing_factor=bad_a)

    def test_estimable_parameter_names(self, sc, burn, model):
        """Unsuffixed aliases are offered only when they are unambiguous."""
        assert set(model.estimable_parameter_names) == {
            "dv_smooth1",
            "dt_dv1",
            "dv_smooth",
            "dt_dv",
        }

        two = SIB(spacecraft=sc, maneuvers=[burn, burn])
        assert set(two.estimable_parameter_names) == {
            "dv_smooth1",
            "dt_dv1",
            "dv_smooth2",
            "dt_dv2",
        }


# ------------------------#
# region  Pulse & partials
# ------------------------#
class TestPulseAndPartials:
    @pytest.mark.parametrize("a", [0.1, 1.0, 5.0])
    def test_pulse_delivers_exact_delta_v(self, sc, burn, a):
        """
        The normalization of Eq. (18) must make the pulse integrate to exactly
        the commanded delta-v, on a grid that resolves it.
        """
        m = SIB(
            spacecraft=sc, maneuvers=burn, base_frame=J2000, t0=T0, smoothing_factor=a
        )
        # Integrate only across the pulse: sech^2 is ~e^-80 by 40/a out, so the
        # rest of the arc contributes nothing and a full-arc grid at this
        # resolution would be millions of points.
        half_width = min(40.0 / a, T_BURN)
        grid = np.linspace(
            T_BURN - half_width, T_BURN + half_width, 20001
        )
        delivered = m.total_delivered_dv(grid)

        assert np.linalg.norm(delivered - DV) / np.linalg.norm(DV) < 1e-8

    @pytest.mark.parametrize("lead", [1.0, 3.0])
    def test_normalization_accounts_for_arc_start_truncation(self, sc, lead):
        """
        A burn only a fraction of a pulse width after the arc start still has to
        deliver the full delta-v.

        Integrating from t0 clips the leading half of the pulse, which is exactly
        what the (1 + tanh(a (t_dv - t0))) factor in Eq. (18) corrects for. With a
        burn deep inside the arc that factor saturates at 1 and the correction is
        invisible, so this is the only case that pins it down.
        """
        a = 0.5
        t_burn_early = T0 + lead  # a * lead = O(1), so tanh is well below 1
        b = scb.ImpulsiveBurn(
            delta_v=scb.ArrayWFrame(DV, km / sec, J2000), t_burn=t_burn_early
        )
        m = SIB(
            spacecraft=sc, maneuvers=b, base_frame=J2000, t0=T0, smoothing_factor=a
        )

        # Integrate over the arc only: everything before t0 is outside the domain.
        grid = np.linspace(T0, T0 + 80.0 / a, 40001)
        delivered = m.total_delivered_dv(grid)

        # Looser than the deep-arc case on purpose. There the integrand decays to
        # zero at both ends, so the trapezoid rule converges exponentially and hits
        # 1e-15; here it is cut off mid-pulse at t0, the Euler-Maclaurin boundary
        # terms survive, and the quadrature falls back to O(h^2) ~ 2e-7. Dropping
        # the truncation correction misses by 27%, so this still bites hard.
        assert np.linalg.norm(delivered - DV) / np.linalg.norm(DV) < 1e-6

    @pytest.mark.parametrize("offset", [-4.0, 0.0, 2.0])
    def test_partial_by_dv_matches_finite_difference(self, sc, model, offset):
        """d(acc)/d(delta-v) is s/N * I, checked against central differences."""
        sid = sc.spice_id
        epoch = T_BURN + offset
        state = {("dv_smooth1", sid): DV, ("dt_dv1", sid): np.array([0.0])}

        analytic = model._compute_partial_by_dv_smooth(None, state, epoch, sc, index=1)

        numeric = np.zeros((3, 3))
        h = 1e-7
        for j in range(3):
            dvp, dvm = DV.copy(), DV.copy()
            dvp[j] += h
            dvm[j] -= h
            ap = model.compute_acceleration(
                None, {("dv_smooth1", sid): dvp, ("dt_dv1", sid): np.array([0.0])}, epoch, sc
            )
            am = model.compute_acceleration(
                None, {("dv_smooth1", sid): dvm, ("dt_dv1", sid): np.array([0.0])}, epoch, sc
            )
            numeric[:, j] = (ap - am) / (2 * h)

        scale = max(np.max(np.abs(numeric)), 1e-30)
        assert np.max(np.abs(analytic - numeric)) / scale < 1e-6

    @pytest.mark.parametrize("offset", [-4.0, 2.0])
    def test_partial_by_dt_matches_finite_difference(self, sc, model, offset):
        """d(acc)/d(epoch offset), the term that makes the epoch estimable."""
        sid = sc.spice_id
        epoch = T_BURN + offset
        state = {("dv_smooth1", sid): DV, ("dt_dv1", sid): np.array([0.0])}

        analytic = model._compute_partial_by_dt_dv(
            None, state, epoch, sc, index=1
        ).reshape(3)

        h = 1e-4
        ap = model.compute_acceleration(
            None, {("dv_smooth1", sid): DV, ("dt_dv1", sid): np.array([h])}, epoch, sc
        )
        am = model.compute_acceleration(
            None, {("dv_smooth1", sid): DV, ("dt_dv1", sid): np.array([-h])}, epoch, sc
        )
        numeric = (ap - am) / (2 * h)

        scale = max(np.max(np.abs(numeric)), 1e-30)
        assert np.max(np.abs(analytic - numeric)) / scale < 1e-6

    def test_partial_by_dt_covers_normalization_derivative(self, sc):
        """
        Same early-burn regime for the epoch partial.

        When the maneuver sits near the arc start, moving it also moves how much
        of the pulse falls inside the arc, so d(normalization)/d(t_dv) is non-zero
        and contributes a second term to the partial. Deep inside the arc that
        term vanishes and is never exercised.
        """
        a, sid = 0.5, sc.spice_id
        t_burn_early = T0 + 2.0
        b = scb.ImpulsiveBurn(
            delta_v=scb.ArrayWFrame(DV, km / sec, J2000), t_burn=t_burn_early
        )
        m = SIB(
            spacecraft=sc, maneuvers=b, base_frame=J2000, t0=T0, smoothing_factor=a
        )

        # The truncation derivative must actually be live here.
        assert m._d_normalization_d_t(t_burn_early) > 1e-3

        epoch = t_burn_early + 1.0
        analytic = m._compute_partial_by_dt_dv(
            None,
            {("dv_smooth1", sid): DV, ("dt_dv1", sid): np.array([0.0])},
            epoch,
            sc,
            index=1,
        ).reshape(3)

        h = 1e-4
        ap = m.compute_acceleration(
            None, {("dv_smooth1", sid): DV, ("dt_dv1", sid): np.array([h])}, epoch, sc
        )
        am = m.compute_acceleration(
            None, {("dv_smooth1", sid): DV, ("dt_dv1", sid): np.array([-h])}, epoch, sc
        )
        numeric = (ap - am) / (2 * h)

        scale = max(np.max(np.abs(numeric)), 1e-30)
        assert np.max(np.abs(analytic - numeric)) / scale < 1e-6

    def test_partial_by_position_is_zero(self, sc, model):
        """The pulse depends on time and parameters only, never on position."""
        partial = model._compute_partial_by_position(R0, {}, T_BURN, sc)
        assert partial.shape == (3, 3)
        assert np.all(partial == 0.0)


# ------------------------------#
# region  Optimal smoothing factor
# ------------------------------#
class TestOptimalSmoothingFactor:
    @pytest.mark.parametrize("dt_dv", [50.0, 100.0, 300.0])
    def test_criterion_residual_vanishes_at_optimal_factor(self, dt_dv):
        """The GSRA root must actually solve g = 0."""
        a = SIB.optimal_smoothing_factor(dt_dv, eta=0.9)
        assert abs(SIB.criterion_residual(a, dt_dv, 0.9)) < 1e-10

    def test_scales_inversely_with_epoch_uncertainty(self):
        """
        In the S2 -> 1 limit the criterion depends only on u = a * dt / 2, so the
        optimal factor is inversely proportional to the epoch uncertainty and
        independent of the maneuver magnitude and the orbit.
        """
        us = [SIB.optimal_smoothing_factor(dt, 0.9) * dt / 2 for dt in (50.0, 100.0, 300.0)]
        assert np.ptp(us) < 1e-9
        assert us[0] == pytest.approx(0.776325, abs=1e-5)

    def test_is_sign_symmetric(self):
        """Advancing or delaying the maneuver gives the same smoothing factor."""
        assert SIB.optimal_smoothing_factor(-100.0) == pytest.approx(
            SIB.optimal_smoothing_factor(100.0), rel=1e-12
        )

    def test_branches_are_distinct(self):
        """h(u) rises then falls, so g = 0 has two roots and both are reachable."""
        lo = SIB.optimal_smoothing_factor(100.0, eta=0.9, branch="low")
        hi = SIB.optimal_smoothing_factor(100.0, eta=0.9, branch="high")

        assert lo < hi
        assert abs(SIB.criterion_residual(lo, 100.0, 0.9)) < 1e-10
        assert abs(SIB.criterion_residual(hi, 100.0, 0.9)) < 1e-10

    def test_maneuver_epoch_barely_affects_the_factor(self):
        """
        t_dv enters only through S2 = tanh(a (t_dv - t0)), which saturates within a
        few pulse widths of the arc start.
        """
        deep = SIB.optimal_smoothing_factor(100.0, t_dv=5000.0, t0=0.0)
        shallow = SIB.optimal_smoothing_factor(100.0, t_dv=100.0, t0=0.0)
        unbounded = SIB.optimal_smoothing_factor(100.0)

        assert deep == pytest.approx(unbounded, rel=1e-6)
        assert shallow == pytest.approx(unbounded, rel=0.02)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"dt_dv": 100.0, "eta": 1.5}, "eta must lie in"),
            ({"dt_dv": 100.0, "eta": 0.0}, "eta must lie in"),
            ({"dt_dv": 0.0}, "must be non-zero"),
            ({"dt_dv": 100.0, "branch": "middle"}, "branch must be"),
            ({"dt_dv": 100.0, "a_bounds": (10.0, 20.0)}, "No root"),
        ],
    )
    def test_rejects_bad_input(self, kwargs, match):
        """Every unreachable request fails loudly rather than returning a guess."""
        with pytest.raises(SIB.NoSmoothingFactorRootError, match=match):
            SIB.optimal_smoothing_factor(**kwargs)

    @pytest.mark.parametrize("dt_dv, a", [(100.0, 0.015526), (300.0, 0.005175)])
    def test_mapped_dv_ratio_matches_numeric_integration(self, sc, burn, dt_dv, a):
        """
        Independent check of the Eq. (28) closed form: integrate Eq. (26),
        d(dv_x)/dt = Dv_x [u_th2 * dt + 1/2 u_th3 * dt^2], directly.
        """
        m = SIB(
            spacecraft=sc, maneuvers=burn, base_frame=J2000, t0=T0, smoothing_factor=a
        )
        h = 1e-3
        grid = np.linspace(T0, T_BURN + dt_dv / 2.0, 40001)

        p_hi = np.array([m._pulse(t, T_BURN + h) for t in grid])
        p_0 = np.array([m._pulse(t, T_BURN) for t in grid])
        p_lo = np.array([m._pulse(t, T_BURN - h) for t in grid])

        u2 = (p_hi - p_lo) / (2 * h)
        u3 = (p_hi - 2 * p_0 + p_lo) / h**2
        numeric = np.trapezoid(u2 * dt_dv + 0.5 * u3 * dt_dv**2, grid)

        closed = SIB.mapped_dv_ratio(a, dt_dv, t_dv=T_BURN, t0=T0)
        assert closed == pytest.approx(numeric, rel=1e-4)

    @pytest.mark.parametrize("dt_dv", [100.0, -100.0])
    def test_mapped_dv_ratio_hits_eta_at_optimal_factor(self, dt_dv):
        """
        The criterion's defining property: at a*, the mapped midpoint velocity
        deviation is -eta * Dv_x for a delayed maneuver and +eta * Dv_x for an
        advanced one.
        """
        eta = 0.9
        a = SIB.optimal_smoothing_factor(dt_dv, eta=eta)
        ratio = SIB.mapped_dv_ratio(a, abs(dt_dv))
        assert ratio == pytest.approx(-eta, abs=1e-9)


# ------------------------#
# region  Propagation    #
# ------------------------#
def _final_error_vs_impulsive(sc, earth, burn, two_leg_final, a):
    """Final position error of the smoothed arc against the impulsive truth."""
    epochs = scb.EpochArray(np.arange(T0, T_END + DT, DT), "TDB")
    fm = scb.ForceModelTranslation(
        primary_body=sc, smoothed_maneuvers=[burn], dv_smoothing_factor=a
    )
    final = _final_rv(_propagate(sc, earth, fm, _base_state(sc, R0, V0), epochs))
    return np.linalg.norm(final[0:3] - two_leg_final[0:3])


class TestPropagation:
    def test_smoothed_matches_two_leg_impulsive(self, sc, earth, burn, two_leg_final):
        """
        The payoff: one continuous arc straight through the smoothed maneuver
        reproduces the exact two-leg impulsive answer, with no leg splitting.
        """
        epochs = scb.EpochArray(np.arange(T0, T_END + DT, DT), "TDB")
        fm = scb.ForceModelTranslation(
            primary_body=sc, smoothed_maneuvers=[burn], dv_smoothing_factor=2.0
        )
        smoothed = _final_rv(
            _propagate(sc, earth, fm, _base_state(sc, R0, V0), epochs)
        )

        assert np.linalg.norm(smoothed[0:3] - two_leg_final[0:3]) < 1e-3  # < 1 m
        assert np.linalg.norm(smoothed[3:6] - two_leg_final[3:6]) < 1e-6  # < 1 mm/s

    def test_accuracy_improves_quadratically_with_smoothing_factor(
        self, sc, earth, burn, two_leg_final
    ):
        """
        While the integrator resolves the pulse, accuracy is O(1/a^2): the pulse is
        symmetric, so its leading error term is quadratic. Quadrupling a should buy
        roughly 16x.
        """
        errs = {
            a: _final_error_vs_impulsive(sc, earth, burn, two_leg_final, a)
            for a in (0.05, 0.2, 1.0)
        }

        assert errs[0.05] / errs[0.2] > 8.0
        assert errs[0.2] / errs[1.0] > 8.0

    def test_unresolved_pulse_degrades_badly(self, sc, earth, burn, two_leg_final):
        """
        Past the resolution limit the trend reverses hard rather than plateauing.

        The adaptive controller steps clean over a pulse this narrow, so its error
        estimate never spikes and it never refines: the maneuver is silently
        under-applied. Scarabaeus exposes no maximum-step control, so this is
        avoided by choosing `a`, not by tightening tolerances. Guards the documented
        upper bound on the smoothing factor.
        """
        resolved = _final_error_vs_impulsive(sc, earth, burn, two_leg_final, 1.0)
        unresolved = _final_error_vs_impulsive(sc, earth, burn, two_leg_final, 5.0)

        assert resolved < 1e-3
        assert unresolved > 1.0
        assert unresolved / resolved > 1e4

    @pytest.mark.parametrize("param", ["dv_x", "dv_z", "dt"])
    def test_stm_column_matches_finite_difference(self, sc, earth, stm_reference, param):
        """
        The variational equations must actually carry both maneuver parameters.
        A partial that is wired up but silently dropped shows as a zero column.
        """
        stm, cols_dv, col_dt = stm_reference
        b = scb.ImpulsiveBurn(
            delta_v=scb.ArrayWFrame(DV, km / sec, J2000), t_burn=T_BURN
        )

        if param == "dt":
            col, h = col_dt, 1e-3
            hi = _final_rv(_propagate_smoothed(sc, earth, b, DV, h, False))
            lo = _final_rv(_propagate_smoothed(sc, earth, b, DV, -h, False))
        else:
            j = 0 if param == "dv_x" else 2
            col, h = cols_dv[j], 1e-6
            dvp, dvm = DV.copy(), DV.copy()
            dvp[j] += h
            dvm[j] -= h
            hi = _final_rv(_propagate_smoothed(sc, earth, b, dvp, 0.0, False))
            lo = _final_rv(_propagate_smoothed(sc, earth, b, dvm, 0.0, False))

        numeric = (hi - lo) / (2 * h)
        analytic = stm[0:6, col]

        assert np.linalg.norm(analytic) > 0.0, "STM column is identically zero"
        scale = max(np.max(np.abs(numeric)), 1e-30)
        assert np.max(np.abs(analytic - numeric)) / scale < 1e-5
