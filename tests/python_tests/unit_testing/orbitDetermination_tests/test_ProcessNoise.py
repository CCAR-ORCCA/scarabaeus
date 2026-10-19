# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =========================================#
#  Unit Tests for the process noise models  #
# =========================================#
import src.scarabaeus as scb
import numpy as np
import pytest
import scipy.linalg

km, kg, sec, N, rad = scb.Units.get_units(["km", "kg", "sec", "N", "rad"])
J2000 = scb.Frame("J2000")


# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture
def sc():
    return scb.Spacecraft("pn_sc", -60000, scb.ArrayWUnits(2300.0, kg),
                          scb.ArrayWUnits(1e-06, km**2), scb.ArrayWUnits(1.5, None))


@pytest.fixture
def ast():
    return scb.Spacecraft("pn_ast", -60001, scb.ArrayWUnits(1e12, kg),
                          scb.ArrayWUnits(1e-06, km**2), scb.ArrayWUnits(1.5, None))


@pytest.fixture
def pos_vel(sc):
    return (scb.StateDefinition()
            .position(sc, scb.ArrayWFrame(
                np.array([-2.65649403e08, 1.88325868e08, 1.01977022e08]), km, J2000))
            .velocity(sc, scb.ArrayWFrame(
                np.array([-9.07473227, -12.48458138, -4.46117482]), km / sec, J2000)))


@pytest.fixture
def q_cont():
    return scb.CovarianceMatrix([scb.ArrayWUnits(s, km * sec**-2) for s in (1e-8, 2e-8, 3e-8)],
                                scb.EpochArray(0.0, sys="TDB"), from_list=True)


@pytest.fixture
def q_cont_rtn():
    """Different radial, transverse and normal sigmas, so a wrong axis shows."""
    return scb.CovarianceMatrix([scb.ArrayWUnits(s, km * sec**-2) for s in (1e-8, 2e-8, 3e-8)],
                                scb.EpochArray(0.0, sys="TDB"), frame="RTN", from_list=True)


def _rtn_axes(position, velocity):
    r = position / np.linalg.norm(position)
    n = np.cross(position, velocity)
    n = n / np.linalg.norm(n)
    return r, np.cross(n, r), n


def _assert_rtn_variances(Q, position, velocity, sigmas):
    """``Q`` (inertial) has the RTN sigmas along the axes of the given state."""
    axes = np.array(_rtn_axes(position, velocity))
    np.testing.assert_allclose(axes @ Q @ axes.T, np.diag(np.square(sigmas)),
                               rtol=1e-12, atol=1e-12 * max(sigmas) ** 2)


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
class TestDiscreteTimeMatrices:
    """The matrices follow the epochs they are computed on and are sized at use."""

    def test_snc_intervals_follow_the_epochs(self, pos_vel, q_cont):
        snc = scb.StateNoiseCompensation(q_cont, state_definition=pos_vel)
        # An event epoch (100) splitting a 60 s measurement interval
        epochs = [0.0, 60.0, 100.0, 120.0]
        snc._compute_discrete_time_covariances(epochs)

        np.testing.assert_array_equal(snc.epochs, epochs)
        np.testing.assert_array_equal(snc.gamma_at(0, 6), np.zeros((6, 3)))
        for k, dt in zip((1, 2, 3), (60.0, 40.0, 20.0)):
            np.testing.assert_array_equal(
                snc.gamma_at(k, 6), snc._construct_gamma_matrix(dt))

    def test_snc_sized_for_the_state_it_is_added_to(self, pos_vel, q_cont):
        snc = scb.StateNoiseCompensation(q_cont, state_definition=pos_vel)
        snc._compute_discrete_time_covariances([0.0, 60.0])
        gamma = snc._construct_gamma_matrix(60.0)
        Q_block = gamma @ q_cont.matrix_without_units() @ gamma.T

        for n in (6, 9):
            Q = snc.covariance_at(1, n)
            assert Q.shape == (n, n)
            np.testing.assert_array_equal(Q[:6, :6], Q_block)
            assert not Q[6:, :].any() and not Q[:, 6:].any()

            gamma_n = snc.gamma_at(1, n)
            assert gamma_n.shape == (n, 3)
            np.testing.assert_array_equal(gamma_n[:6], gamma)
            assert not gamma_n[6:].any()

    def test_dmc_sized_for_the_state_it_is_added_to(self, sc, pos_vel, q_cont):
        definition = pos_vel.param(
            "a_fogm", sc, scb.ArrayWFrame(np.zeros(3), km * sec**-2, J2000))
        dmc = scb.DynamicalModelCompensation(
            q_cont, state_definition=definition, beta=np.array([1e-3] * 3))
        dmc._compute_discrete_time_covariances([0.0, 60.0])

        Q = dmc.covariance_at(1, 11)
        assert Q.shape == (11, 11)
        np.testing.assert_array_equal(Q[:9, :9], dmc.discrete_time_process_covariances[1])
        assert not Q[9:, :].any() and not Q[:, 9:].any()

        gamma = dmc.gamma_at(1, 11)
        assert gamma.shape == (11, 3)
        assert not gamma[9:].any()
        np.testing.assert_array_equal(
            Q[:9, :9], gamma[:9] @ q_cont.matrix_without_units() @ gamma[:9].T)


def _constant_input_response(A, B, dt):
    """State change over ``dt`` per unit of a constant input ``u`` to
    ``x' = A x + B u``, from ``expm([[A, B], [0, 0]] dt)`` (Van Loan)."""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    return scipy.linalg.expm(M * dt)[:n, n:]


class TestGammaMatrices:
    """Gamma maps an input held constant over the step into the state."""

    @pytest.mark.parametrize("dt", [0.5, 60.0, 3600.0])
    def test_snc_gamma_is_the_constant_acceleration_response(self, pos_vel, q_cont, dt):
        snc = scb.StateNoiseCompensation(q_cont, state_definition=pos_vel)
        A = np.block([[np.zeros((3, 3)), np.eye(3)], [np.zeros((3, 6))]])
        B = np.vstack((np.zeros((3, 3)), np.eye(3)))
        gamma = snc._construct_gamma_matrix(dt)
        np.testing.assert_allclose(gamma, _constant_input_response(A, B, dt),
                                   rtol=1e-12, atol=1e-12 * dt**2)

        snc._compute_discrete_time_covariances([0.0, dt])
        Q_c = q_cont.matrix_without_units()
        np.testing.assert_allclose(
            snc.covariance_at(1, 6),
            dt**2 * np.block([[dt**2 / 4 * Q_c, dt / 2 * Q_c], [dt / 2 * Q_c, Q_c]]),
            rtol=1e-12)


    @pytest.mark.parametrize("dt", [0.5, 60.0, 3600.0])
    def test_dmc_gamma_is_the_constant_input_response(self, sc, pos_vel, q_cont, dt):
        # One correlation time per axis, down to months, where the closed
        # form in 1/beta cancels to nothing
        beta = np.array([1e-1, 1 / 3600, 1e-7])
        dmc = scb.DynamicalModelCompensation(
            q_cont, state_definition=_dmc_definition(sc, pos_vel), beta=beta)
        A = np.zeros((9, 9))
        A[:3, 3:6], A[3:6, 6:] = np.eye(3), np.eye(3)
        A[6:, 6:] = -np.diag(beta)
        B = np.vstack((np.zeros((6, 3)), np.eye(3)))
        expected = _constant_input_response(A, B, dt)

        # The exponential itself is good to about 1e-9 over the longest step
        gamma = dmc._construct_gamma_matrix_dmc(dt, beta)
        for block in (slice(0, 3), slice(3, 6), slice(6, 9)):
            np.testing.assert_allclose(np.diag(gamma[block]), np.diag(expected[block]),
                                       rtol=1e-8)
        np.testing.assert_allclose(gamma, expected, atol=1e-8 * np.abs(expected).max())

        dmc._compute_discrete_time_covariances([0.0, dt])
        assert not dmc.gamma_at(0, 9).any()
        np.testing.assert_array_equal(dmc.gamma_at(1, 9), gamma)
        Q_c = q_cont.matrix_without_units()
        np.testing.assert_array_equal(dmc.Q_tilde[1], Q_c)
        np.testing.assert_array_equal(dmc.covariance_at(1, 9), gamma @ Q_c @ gamma.T)

    def test_dmc_gamma_constant_acceleration_limit(self, sc, pos_vel, q_cont):
        """As beta dt -> 0 the input integrates into a ramp in acceleration."""
        dt = 3600.0
        dmc = scb.DynamicalModelCompensation(
            q_cont, state_definition=_dmc_definition(sc, pos_vel), beta=1e-15)
        # Up to the first-order term in beta dt, of 1e-12
        np.testing.assert_allclose(dmc._construct_gamma_matrix_dmc(dt, [1e-15] * 3),
                                   np.kron([[dt**3 / 6], [dt**2 / 2], [dt]], np.eye(3)),
                                   rtol=1e-11)


def _dmc_definition(sc, pos_vel):
    return pos_vel.param("a_fogm", sc, scb.ArrayWFrame(np.zeros(3), km * sec**-2, J2000))


class TestDMCBeta:
    """The decay rates DMC is discretized with."""

    def test_reference_decay_rates_per_interval(self, sc, pos_vel, q_cont):
        """The decay rates are read where each interval's noise acts, and
        replace the ones given to the constructor."""
        requested = []

        def drifting(times):
            requested.append(np.array(times))
            return 1e-3 * (1.0 + np.outer(times, [1e-3, 2e-3, 3e-3]))

        definition = _dmc_definition(sc, pos_vel)
        dmc = scb.DynamicalModelCompensation(q_cont, state_definition=definition, beta=1.0)
        epochs = [0.0, 60.0, 100.0, 120.0]
        dmc._compute_discrete_time_covariances(epochs, reference_beta=drifting)

        (mid_epochs,) = requested
        np.testing.assert_array_equal(mid_epochs, [0.0, 30.0, 80.0, 110.0])
        for k, (dt, beta) in enumerate(zip(np.diff(epochs), drifting(mid_epochs)[1:]), start=1):
            single = scb.DynamicalModelCompensation(q_cont, state_definition=definition,
                                                    beta=beta)
            single._compute_discrete_time_covariances([0.0, dt])
            np.testing.assert_array_equal(dmc.covariance_at(k, 9), single.covariance_at(1, 9))

    def test_beta_required(self, sc, pos_vel, q_cont):
        dmc = scb.DynamicalModelCompensation(q_cont, state_definition=_dmc_definition(sc, pos_vel))
        assert dmc.beta is None
        with pytest.raises(scb.DynamicalModelCompensation.MissingBetaError):
            dmc._compute_discrete_time_covariances([0.0, 60.0])

    @pytest.mark.parametrize("beta, error", [
        (np.array([1e-3, 1e-3]), scb.DynamicalModelCompensation.BetaSizeError),
        (0.0, scb.DynamicalModelCompensation.NonPositiveBetaError),
        (np.array([1e-3, -1e-3, 1e-3]), scb.DynamicalModelCompensation.NonPositiveBetaError),
    ])
    def test_invalid_beta_reported(self, sc, pos_vel, q_cont, beta, error):
        with pytest.raises(error):
            scb.DynamicalModelCompensation(
                q_cont, state_definition=_dmc_definition(sc, pos_vel), beta=beta)


class TestRTNFrame:
    """An RTN covariance is rotated to the inertial frame along the reference."""

    def test_rotation_keeps_each_sigma_on_its_axis(self, pos_vel, q_cont_rtn):
        snc = scb.StateNoiseCompensation(q_cont_rtn, state_definition=pos_vel)
        position = np.array([-2.65649403e08, 1.88325868e08, 1.01977022e08])
        velocity = np.array([-9.07473227, -12.48458138, -4.46117482])

        Q = snc.transform_from_rtn(q_cont_rtn.matrix_without_units(), position, velocity)
        _assert_rtn_variances(Q, position, velocity, (1e-8, 2e-8, 3e-8))

    def test_rotation_follows_the_reference_over_each_interval(self, pos_vel, q_cont_rtn):
        """Each interval is rotated with the reference at its middle, the first
        entry (no interval) at the first epoch."""
        radius, rate = 7000.0, 2.0 * np.pi / 400.0
        requested = []

        def circular(times):
            requested.append(np.array(times))
            angle = rate * np.asarray(times)
            positions = radius * np.column_stack(
                (np.cos(angle), np.sin(angle), np.zeros_like(angle)))
            velocities = radius * rate * np.column_stack(
                (-np.sin(angle), np.cos(angle), np.zeros_like(angle)))
            return positions, velocities

        snc = scb.StateNoiseCompensation(q_cont_rtn, state_definition=pos_vel)
        snc._compute_discrete_time_covariances([0.0, 60.0, 100.0, 120.0], circular)

        (mid_epochs,) = requested
        np.testing.assert_array_equal(mid_epochs, [0.0, 30.0, 80.0, 110.0])
        positions, velocities = circular(mid_epochs)
        for Q, position, velocity in zip(snc.Q_tilde, positions, velocities):
            _assert_rtn_variances(Q, position, velocity, (1e-8, 2e-8, 3e-8))

    def test_reference_required(self, pos_vel, q_cont_rtn):
        snc = scb.StateNoiseCompensation(q_cont_rtn, state_definition=pos_vel)
        with pytest.raises(scb.StateNoiseCompensation.MissingReferenceStateError):
            snc._compute_discrete_time_covariances([0.0, 60.0])


class TestMaxInterval:
    """An interval longer than ``max_interval``, such as a data gap, gets the
    noise of one ``max_interval`` long; shorter ones are unchanged."""

    def test_snc_long_interval_cut(self, pos_vel, q_cont):
        snc = scb.StateNoiseCompensation(q_cont, state_definition=pos_vel,
                                         max_interval=600.0)
        # A one-hour gap after two measurement intervals
        snc._compute_discrete_time_covariances([0.0, 60.0, 100.0, 3700.0])

        for k, dt in zip((1, 2, 3), (60.0, 40.0, 600.0)):
            np.testing.assert_array_equal(
                snc.gamma_at(k, 6), snc._construct_gamma_matrix(dt))

    def test_dmc_long_interval_cut(self, sc, pos_vel, q_cont):
        definition = pos_vel.param(
            "a_fogm", sc, scb.ArrayWFrame(np.zeros(3), km * sec**-2, J2000))
        dmc = scb.DynamicalModelCompensation(
            q_cont, state_definition=definition, beta=np.array([1e-3] * 3),
            max_interval=600.0)
        dmc._compute_discrete_time_covariances([0.0, 60.0, 100.0, 3700.0])

        for k, dt in zip((1, 2, 3), (60.0, 40.0, 600.0)):
            single = scb.DynamicalModelCompensation(
                q_cont, state_definition=definition, beta=np.array([1e-3] * 3))
            single._compute_discrete_time_covariances([0.0, dt])
            np.testing.assert_array_equal(dmc.covariance_at(k, 9), single.covariance_at(1, 9))
            np.testing.assert_array_equal(dmc.gamma_at(k, 9), single.gamma_at(1, 9))

    def test_noise_continuous_and_non_decreasing(self, pos_vel, q_cont):
        """No jump at the cap, unlike dropping the noise over a gap."""
        cap = 600.0
        snc = scb.StateNoiseCompensation(q_cont, state_definition=pos_vel,
                                         max_interval=cap)
        dts = np.linspace(10.0, 3.0 * cap, 60)
        variances = []
        for dt in dts:
            snc._compute_discrete_time_covariances([0.0, dt])
            variances.append(np.diag(snc.covariance_at(1, 6)))
        variances = np.array(variances)

        assert np.all(np.diff(variances, axis=0) >= 0.0)
        snc._compute_discrete_time_covariances([0.0, cap])
        np.testing.assert_array_equal(variances[dts >= cap],
                                      np.tile(np.diag(snc.covariance_at(1, 6)),
                                              ((dts >= cap).sum(), 1)))

    def test_rtn_rotated_where_the_noise_acts(self, pos_vel, q_cont_rtn):
        """A cut interval is rotated at the middle of its last max_interval."""
        requested = []

        def straight(times):
            requested.append(np.array(times))
            times = np.asarray(times)
            positions = np.column_stack((7000.0 + times, np.ones_like(times),
                                         np.zeros_like(times)))
            velocities = np.tile([0.0, 1.0, 0.0], (len(times), 1))
            return positions, velocities

        snc = scb.StateNoiseCompensation(q_cont_rtn, state_definition=pos_vel,
                                         max_interval=600.0)
        snc._compute_discrete_time_covariances([0.0, 60.0, 3660.0], straight)

        (mid_epochs,) = requested
        np.testing.assert_array_equal(mid_epochs, [0.0, 30.0, 3360.0])

    def test_units_converted_to_seconds(self, pos_vel, q_cont):
        hr = scb.Units.get_units("hr")
        snc = scb.StateNoiseCompensation(q_cont, state_definition=pos_vel,
                                         max_interval=scb.ArrayWUnits(2, hr))
        assert snc.max_interval == 7200.0
        settings = scb.ProcessNoiseSettings(type="SNC", Q_cont=q_cont,
                                            max_interval=scb.ArrayWUnits(2, hr))
        assert settings.max_interval == 7200.0
        assert scb.ProcessNoiseSettings(type="SNC", Q_cont=q_cont).max_interval is None

    @pytest.mark.parametrize("max_interval", [0.0, -60.0, np.nan])
    def test_non_positive_reported(self, pos_vel, q_cont, max_interval):
        with pytest.raises(scb.ProcessNoise.NonPositiveIntervalError):
            scb.StateNoiseCompensation(q_cont, state_definition=pos_vel,
                                       max_interval=max_interval)
        with pytest.raises(scb.ProcessNoise.NonPositiveIntervalError):
            scb.ProcessNoiseSettings(type="SNC", Q_cont=q_cont, max_interval=max_interval)


def _run_burn_sequence(filter_cls, process_noise, out_dir, initial_covariance=None):
    """
    Covariance analysis over coast, finite burn (adding k_thrust1, dRA1 and
    dDEC1), coast, with measurements every 60 s offset so that neither event
    epoch is a measurement epoch. ``initial_covariance`` replaces the default
    one, which covers the first leg and seeds the burn states from
    ``extra_apriori_sigmas``.
    """
    t_m1 = 923000647.3506516
    t_m1_end = t_m1 + 600.0
    start_time, end_time = t_m1 - 600.0, t_m1_end + 600.0
    u = 1.0 / np.sqrt(2.0)

    maneuver = scb.Maneuver()
    maneuver.thrust = scb.ArrayWUnits(0.2, N)
    maneuver.mass_flow = scb.ArrayWUnits(0.2 / (2200.0 * 9.80665), kg / sec)
    maneuver.start_time = scb.EpochArray(t_m1, "TDB")
    maneuver.end_time = scb.EpochArray(t_m1_end, "TDB")
    maneuver.ux = scb.ArrayWUnits(u, None)
    maneuver.uy = scb.ArrayWUnits(0.0, None)
    maneuver.uz = scb.ArrayWUnits(u, None)

    sc = scb.Spacecraft("pn_seq_sc", -60000, scb.ArrayWUnits(2300.0, kg),
                        scb.ArrayWUnits(1e-06, km**2), scb.ArrayWUnits(1.5, None))
    origin = scb.CelestialBody.from_constants("SUN")

    def grid(t0, t1):
        return scb.EpochArray(np.arange(t0, t1 + 10.0, 10.0), sys="TDB")

    def prop(definition, t0, t1, burn=None):
        state = scb.StateArray(epoch=scb.EpochArray(t0, sys="TDB"), origin=origin,
                               state=definition)
        force_model = (
            scb.ForceModelTranslation(primary_body=sc, cannonball_SRP=True,
                                      finite_burn=True, maneuver=burn)
            if burn is not None
            else scb.ForceModelTranslation(primary_body=sc, cannonball_SRP=True))
        return scb.Propagator(primary_body=sc, state_vector=state, tspan=grid(t0, t1),
                              force_models=force_model)

    coast = (scb.StateDefinition()
             .position(sc, scb.ArrayWFrame(
                 np.array([-2.65643964e08, 1.88333350e08, 1.01979698e08]), km, J2000))
             .velocity(sc, scb.ArrayWFrame(
                 np.array([-9.07510541, -12.48432651, -4.46105273]), km / sec, J2000)))
    burn = (scb.StateDefinition.empty_pv(sc, frame=J2000)
            .param("dRA1", sc, scb.ArrayWFrame(0.4, rad, J2000),
                   estimation="estimated", dynamics="static")
            .param("dDEC1", sc, scb.ArrayWFrame(0.3, rad, J2000),
                   estimation="estimated", dynamics="static")
            .param("k_thrust1", sc, scb.ArrayWFrame(0.9, None, None),
                   estimation="estimated", dynamics="static"))

    sequence = scb.MissionSequence("pn_sequence")
    sequence.addLeg(name="coast", prop=prop(coast, start_time, t_m1))
    sequence.addFiniteBurn(name="burn", prop=prop(burn, t_m1, t_m1_end, maneuver))
    sequence.addLeg(name="final coast", prop=prop(
        scb.StateDefinition.empty_pv(sc, frame=J2000), t_m1_end, end_time))

    gs = scb.GroundStation("DSS-14")
    meas_epochs = scb.EpochArray(np.arange(start_time + 30.0, end_time - 30.0, 60.0),
                                 sys="TDB")
    measurements = scb.MeasurementSpec.from_dict([
        {"model": scb.RangeIdeal("R", gs, sigma=scb.ArrayWUnits(1e-3, km),
                                 sequence_definition=sequence),
         "dataset_name": "R", "epochs": meas_epochs},
        {"model": scb.RangeRateIdeal("RR", gs, sigma=scb.ArrayWUnits(1e-6, km / sec),
                                     sequence_definition=sequence),
         "dataset_name": "RR", "epochs": meas_epochs},
    ])

    if initial_covariance is None:
        initial_covariance = scb.CovarianceMatrix(
            [scb.ArrayWUnits(100.0, km)] * 3 + [scb.ArrayWUnits(0.01, km / sec)] * 3,
            scb.EpochArray(start_time, sys="TDB"), from_list=True,
            extra_apriori_sigmas={
                "dRA1": scb.ArrayWUnits(np.deg2rad(0.5), rad),
                "dDEC1": scb.ArrayWUnits(np.deg2rad(0.5), rad),
                "k_thrust1": scb.ArrayWUnits(0.05, None),
            })
    settings = scb.FilterSettings(
        initial_covariance=initial_covariance,
        output=scb.OutputSettings(
            solution_output_path=str(out_dir), solution_output_name="solution",
            save_reference_trajectory=False, save_estimated_trajectory=False),
        covariance_analysis=True,
        process_noise=process_noise,
    )
    traj_name = f"pn_sequence_{filter_cls.__name__}.bsp"
    od = filter_cls(propagator=sequence, settings=settings, measurements=measurements,
                    traj_name=traj_name, traj_dir=str(out_dir))
    od.fit(max_iterations=1, verbose=False, traj_name=traj_name)
    return od


class TestBurnSequence:
    """Process noise on a sequence whose legs carry different states."""

    def test_filters_agree_across_legs(self, tmp_path):
        """
        The noise is discretized over the filtering timeline, so an event epoch
        splits the interval it falls in, and it is sized for each leg (6 and 9
        states). LKF and SRIF add it through independent paths (the covariance
        and the gamma matrix), so they agree only if both are aligned.
        """
        snc = scb.ProcessNoiseSettings(
            type="SNC",
            Q_cont=scb.CovarianceMatrix([scb.ArrayWUnits(1e-8, km * sec**-2)] * 3,
                                        scb.EpochArray(0.0, sys="TDB"), from_list=True))
        lkf = _run_burn_sequence(scb.LKF, snc, tmp_path / "lkf")
        srif = _run_burn_sequence(scb.SRIF, snc, tmp_path / "srif")

        measurement_times = lkf.measurement_data.get_spacecraft_times()
        assert not np.isin(lkf.events_epochs, measurement_times).any()
        np.testing.assert_array_equal(lkf.process_noise.epochs, lkf.timeline_times)

        shapes = [np.shape(P) for P in lkf.covariance_history]
        assert {(6, 6), (9, 9)} <= set(shapes)
        assert shapes == [np.shape(P) for P in srif.covariance_history]
        for P_lkf, P_srif in zip(lkf.covariance_history, srif.covariance_history):
            np.testing.assert_allclose(np.sqrt(np.diag(P_lkf)), np.sqrt(np.diag(P_srif)),
                                       rtol=1e-5)

    def test_noise_enters_every_leg(self, tmp_path):
        """With SNC the position/velocity uncertainty grows on every leg."""
        snc = scb.ProcessNoiseSettings(
            type="SNC",
            Q_cont=scb.CovarianceMatrix([scb.ArrayWUnits(1e-8, km * sec**-2)] * 3,
                                        scb.EpochArray(0.0, sys="TDB"), from_list=True))
        noisy = _run_burn_sequence(scb.LKF, snc, tmp_path / "snc")
        quiet = _run_burn_sequence(scb.LKF, None, tmp_path / "none")

        times = noisy.timeline_times
        np.testing.assert_array_equal(times, quiet.timeline_times)
        sigma_noisy = np.array([np.sqrt(np.diag(P)[:6]) for P in noisy.covariance_history])
        sigma_quiet = np.array([np.sqrt(np.diag(P)[:6]) for P in quiet.covariance_history])
        # Last row of each leg: just before each event, and the final row
        ends = [t for t in noisy.events_epochs if times[0] < t] + [np.inf]
        for t_end in ends:
            last = np.flatnonzero(times < t_end)[-1]
            assert np.all(sigma_noisy[last] > sigma_quiet[last])

    def test_max_interval_reaches_the_filter(self, tmp_path):
        """The cap set in the settings cuts every timeline interval."""
        snc = scb.ProcessNoiseSettings(
            type="SNC",
            Q_cont=scb.CovarianceMatrix([scb.ArrayWUnits(1e-8, km * sec**-2)] * 3,
                                        scb.EpochArray(0.0, sys="TDB"), from_list=True),
            max_interval=scb.ArrayWUnits(20.0, sec))
        od = _run_burn_sequence(scb.LKF, snc, tmp_path)

        assert od.process_noise.max_interval == 20.0
        intervals = np.diff(od.timeline_times)
        assert (intervals > 20.0).any()
        for k, dt in enumerate(intervals, start=1):
            np.testing.assert_array_equal(
                od.process_noise.gamma_at(k, 6),
                od.process_noise._construct_gamma_matrix(min(dt, 20.0)))

    def test_rtn_noise_rotated_on_every_leg(self, tmp_path, q_cont_rtn):
        """
        An RTN noise is rotated with each leg's own reference trajectory.
        Rotated, it is not diagonal, so LKF and SRIF agree only if SRIF
        factors it the right way round.
        """
        rtn = scb.ProcessNoiseSettings(type="SNC", Q_cont=q_cont_rtn)
        lkf = _run_burn_sequence(scb.LKF, rtn, tmp_path / "lkf")
        srif = _run_burn_sequence(scb.SRIF, rtn, tmp_path / "srif")

        times = lkf.timeline_times
        mid_epochs = np.concatenate((times[:1], 0.5 * (times[:-1] + times[1:])))
        positions, velocities = lkf._reference_position_velocity(mid_epochs)
        assert len(lkf.process_noise.Q_tilde) == len(times)
        for Q, position, velocity in zip(lkf.process_noise.Q_tilde, positions, velocities):
            _assert_rtn_variances(Q, position, velocity, (1e-8, 2e-8, 3e-8))

        for P_lkf, P_srif in zip(lkf.covariance_history, srif.covariance_history):
            np.testing.assert_allclose(np.sqrt(np.diag(P_lkf)), np.sqrt(np.diag(P_srif)),
                                       rtol=1e-5)

    def test_reference_follows_each_leg(self, tmp_path, ast):
        """The reference read from the SPK reproduces each leg's samples. The
        SPK holds only the primary body, so another body's is reported."""
        od = _run_burn_sequence(scb.LKF, None, tmp_path)
        legs = od.trajectory._legs
        for i, leg in enumerate(legs):
            # SPICE returns one of the two states at a node, so only the
            # samples inside the leg are compared
            inside = slice(0 if i == 0 else 1, None if i == len(legs) - 1 else -1)
            epochs = np.asarray(leg["epoch_leg"].et, dtype=float)[inside]
            positions, velocities = od._reference_position_velocity(epochs)
            np.testing.assert_allclose(positions, leg["pos_awf"].quantity.values[inside],
                                       rtol=1e-12)
            # SPICE differentiates the Hermite polynomial through the
            # heliocentric positions, which costs a few digits of velocity
            np.testing.assert_allclose(velocities, leg["vel_awf"].quantity.values[inside],
                                       rtol=1e-9)
        with pytest.raises(scb.FilterOD.NoiseReferenceUnavailableError, match="pn_ast"):
            od._reference_position_velocity(epochs, body=ast)

    def test_initial_covariance_covers_the_first_leg(self, tmp_path):
        """A P0 sized for the burn states as well is reported: they are
        seeded from extra_apriori_sigmas when their leg starts."""
        full_state = scb.CovarianceMatrix(
            [scb.ArrayWUnits(100.0, km)] * 3 + [scb.ArrayWUnits(0.01, km / sec)] * 3
            + [scb.ArrayWUnits(np.deg2rad(0.5), rad)] * 2 + [scb.ArrayWUnits(0.05, None)],
            scb.EpochArray(0.0, sys="TDB"), from_list=True)
        with pytest.raises(scb.FilterOD.InitialCovarianceSizeError, match="first leg"):
            _run_burn_sequence(scb.LKF, None, tmp_path, initial_covariance=full_state)



def _dmc_settings(sigma, settings_beta=None):
    """DMC driven by an input of standard deviation ``sigma`` (km/s^3), held
    constant over each interval."""
    q_cont = scb.CovarianceMatrix([scb.ArrayWUnits(sigma, km * sec**-3)] * 3,
                                  scb.EpochArray(0.0, sys="TDB"), from_list=True)
    return scb.ProcessNoiseSettings(type="DMC", Q_cont=q_cont, beta=settings_beta)


def _dmc_arc(filter_cls, process_noise, out_dir, fogm_beta, beta_state=None,
             covariance_analysis=True, **fogm_frame):
    """
    A 20 min heliocentric arc estimating position, velocity and a_fogm,
    with range and range rate every 60 s. ``beta_state`` adds a considered
    ``beta_fogm`` state, which the dynamics use instead of ``fogm_beta``;
    ``fogm_frame`` holds the ``fogm_frame`` settings of the force model.
    """
    t0 = 923000647.3506516
    t1 = t0 + 1200.0
    sc = scb.Spacecraft("pn_dmc_sc", -60000, scb.ArrayWUnits(2300.0, kg),
                        scb.ArrayWUnits(1e-06, km**2), scb.ArrayWUnits(1.5, None))
    definition = (scb.StateDefinition()
                  .position(sc, scb.ArrayWFrame(
                      np.array([-2.65649403e08, 1.88325868e08, 1.01977022e08]), km, J2000))
                  .velocity(sc, scb.ArrayWFrame(
                      np.array([-9.07473227, -12.48458138, -4.46117482]), km / sec, J2000))
                  .param("a_fogm", sc, scb.ArrayWFrame(np.zeros(3), km * sec**-2, J2000)))
    # A 100 km a priori leaves the smoothers ill-conditioned, to ~1e-4
    sigmas = ([scb.ArrayWUnits(1.0, km)] * 3 + [scb.ArrayWUnits(1e-5, km / sec)] * 3
              + [scb.ArrayWUnits(1e-8, km * sec**-2)] * 3)
    if beta_state is not None:
        definition = definition.param(
            "beta_fogm", sc, scb.ArrayWFrame(np.asarray(beta_state, float), sec**-1, None),
            estimation="considered", dynamics="static")
        sigmas += [scb.ArrayWUnits(1e-6, sec**-1)] * 3

    state = scb.StateArray(epoch=scb.EpochArray(t0, sys="TDB"),
                           origin=scb.CelestialBody.from_constants("SUN"), state=definition)
    propagator = scb.Propagator(
        primary_body=sc, state_vector=state,
        tspan=scb.EpochArray(np.arange(t0, t1 + 10.0, 10.0), sys="TDB"),
        force_models=scb.ForceModelTranslation(
            primary_body=sc, first_order_gauss_markov=True, fogm_beta=fogm_beta,
            **fogm_frame))

    gs = scb.GroundStation("DSS-14")
    meas_epochs = scb.EpochArray(np.arange(t0, t1 - 30.0, 60.0), sys="TDB")
    measurements = scb.MeasurementSpec.from_dict([
        {"model": scb.RangeIdeal("R", gs, sigma=scb.ArrayWUnits(1e-3, km),
                                 state_definition=definition),
         "dataset_name": "R", "epochs": meas_epochs},
        {"model": scb.RangeRateIdeal("RR", gs, sigma=scb.ArrayWUnits(1e-6, km / sec),
                                     state_definition=definition),
         "dataset_name": "RR", "epochs": meas_epochs},
    ])
    settings = scb.FilterSettings(
        initial_covariance=scb.CovarianceMatrix(sigmas, scb.EpochArray(t0, sys="TDB"),
                                                from_list=True),
        output=scb.OutputSettings(
            solution_output_path=str(out_dir), solution_output_name="solution",
            save_reference_trajectory=False, save_estimated_trajectory=False),
        covariance_analysis=covariance_analysis,
        process_noise=process_noise,
    )
    # One kernel name per run, as SPICE keeps the earlier runs' kernels loaded
    traj_name = f"pn_dmc_{out_dir.name}.bsp"
    return filter_cls(propagator=propagator, settings=settings, measurements=measurements,
                      traj_name=traj_name, traj_dir=str(out_dir))


def _assert_dmc_decay_rates(od, beta):
    """The filter's noise is that of a DMC with these decay rates."""
    process_noise = od.process_noise
    expected = scb.DynamicalModelCompensation(
        process_noise.continuous_time_covariance,
        state_definition=od.propagator.full_state_vector.state, beta=beta)
    expected._compute_discrete_time_covariances(od.timeline_times)
    assert len(process_noise.discrete_time_process_covariances) == len(od.timeline_times)
    for Q, Q_expected in zip(process_noise.discrete_time_process_covariances,
                             expected.discrete_time_process_covariances):
        np.testing.assert_array_equal(Q, Q_expected)


class TestDMCDecayRates:
    """The filters discretize DMC with the decay rates of the dynamics."""

    def test_taken_from_the_force_model(self, tmp_path):
        beta = np.array([1e-3, 2e-3, 3e-3])
        od = _dmc_arc(scb.LKF, _dmc_settings(1e-11), tmp_path, fogm_beta=beta)
        assert od.process_noise.beta is None
        _assert_dmc_decay_rates(od, beta)

    def test_taken_from_the_state(self, tmp_path):
        """A beta_fogm state overrides the force model, as in the dynamics."""
        beta = np.array([4e-4, 5e-4, 6e-4])
        od = _dmc_arc(scb.LKF, _dmc_settings(1e-11), tmp_path,
                      fogm_beta=np.array([1e-3] * 3), beta_state=beta)
        _assert_dmc_decay_rates(od, beta)

    def test_matching_settings_accepted(self, tmp_path):
        beta = np.array([1e-3, 2e-3, 3e-3])
        od = _dmc_arc(scb.LKF, _dmc_settings(1e-11, settings_beta=beta), tmp_path,
                      fogm_beta=beta)
        _assert_dmc_decay_rates(od, beta)

    @pytest.mark.parametrize("beta_state", [None, np.array([4e-4, 5e-4, 6e-4])])
    def test_mismatched_settings_reported(self, tmp_path, beta_state):
        """Also the force model's own rates, when a beta_fogm state replaces
        them in the dynamics."""
        fogm_beta = np.array([1e-3, 2e-3, 3e-3])
        settings_beta = 2.0 * fogm_beta if beta_state is None else fogm_beta
        with pytest.raises(scb.FilterOD.ProcessNoiseBetaMismatchError):
            _dmc_arc(scb.LKF, _dmc_settings(1e-11, settings_beta=settings_beta),
                     tmp_path, fogm_beta=fogm_beta, beta_state=beta_state)


class TestDMCFilters:
    """LKF adds the DMC covariance, SRIF its gamma matrix."""

    def test_covariance_analysis_agrees(self, tmp_path):
        # Correlation time of 1000 s, against a 20 min arc
        dmc = _dmc_settings(1e-10)
        lkf = _dmc_arc(scb.LKF, dmc, tmp_path / "lkf", fogm_beta=np.array([1e-3] * 3))
        srif = _dmc_arc(scb.SRIF, dmc, tmp_path / "srif", fogm_beta=np.array([1e-3] * 3))
        quiet = _dmc_arc(scb.LKF, None, tmp_path / "none", fogm_beta=np.array([1e-3] * 3))
        for od in (lkf, srif, quiet):
            od.fit(max_iterations=1, verbose=False)

        # The noise shows in a_fogm, and raises every other sigma
        sigma_noisy = np.sqrt(np.diag(lkf.covariance_history[-1]))
        sigma_quiet = np.sqrt(np.diag(quiet.covariance_history[-1]))
        assert np.all(sigma_noisy[6:] > 2.0 * sigma_quiet[6:])
        assert np.all(sigma_noisy[:6] > sigma_quiet[:6])
        assert len(lkf.covariance_history) == len(srif.covariance_history) > 0
        for P_lkf, P_srif in zip(lkf.covariance_history, srif.covariance_history):
            np.testing.assert_allclose(np.sqrt(np.diag(P_lkf)), np.sqrt(np.diag(P_srif)),
                                       rtol=1e-8)

    def test_smoothed_covariance_agrees(self, tmp_path):
        """The SRIF smoother takes the noise rows of each step's size."""
        dmc = _dmc_settings(1e-10)
        smoothed = []
        for filter_cls in (scb.LKF, scb.SRIF):
            od = _dmc_arc(filter_cls, dmc, tmp_path / filter_cls.__name__,
                          fogm_beta=np.array([1e-3] * 3), covariance_analysis=False)
            od.fit(max_iterations=1, verbose=False, if_sequential_smooth=True)
            smoothed.append(od._solution_history_smooth[-1].covariance_smooth)

        assert len(smoothed[0]) == len(smoothed[1]) > 0
        for P_lkf, P_srif in zip(*smoothed):
            np.testing.assert_allclose(np.sqrt(np.diag(P_lkf)), np.sqrt(np.diag(P_srif)),
                                       rtol=1e-8)


def _rotation(rotvec):
    """Proper rotation about ``rotvec``, by its norm."""
    x, y, z = rotvec
    return scipy.linalg.expm(np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]))


class TestDMCFrame:
    """A Gauss-Markov acceleration read in the RTN or a body-fixed frame
    drives the inertial state through its rotation T."""

    @pytest.mark.parametrize("dt", [0.5, 60.0, 3600.0])
    def test_rotated_gamma_is_the_constant_input_response(self, sc, pos_vel, q_cont, dt):
        """w' = -B w + T^T u and r'' = T w, with u inertial and T held over the step."""
        beta = np.array([1e-1, 1 / 3600, 1e-7])
        T = _rotation([0.3, -1.1, 0.7])
        dmc = scb.DynamicalModelCompensation(
            q_cont, state_definition=_dmc_definition(sc, pos_vel), beta=beta)
        A = np.zeros((9, 9))
        A[:3, 3:6], A[3:6, 6:] = np.eye(3), T
        A[6:, 6:] = -np.diag(beta)
        B = np.vstack((np.zeros((6, 3)), T.T))
        expected = _constant_input_response(A, B, dt)

        gamma = dmc._rotate_gamma(dmc._construct_gamma_matrix_dmc(dt, beta), T)
        for block in (slice(0, 3), slice(3, 6), slice(6, 9)):
            np.testing.assert_allclose(gamma[block], expected[block],
                                       atol=1e-8 * np.abs(expected[block]).max())

    def test_identity_rotation_changes_nothing(self, sc, pos_vel, q_cont):
        """An inertial Gauss-Markov acceleration keeps the unrotated noise, bit for bit."""
        definition = _dmc_definition(sc, pos_vel)
        epochs = [0.0, 60.0, 100.0, 120.0]
        plain = scb.DynamicalModelCompensation(q_cont, state_definition=definition,
                                               beta=1e-3)
        framed = scb.DynamicalModelCompensation(q_cont, state_definition=definition,
                                                beta=1e-3)
        plain._compute_discrete_time_covariances(epochs)
        framed._compute_discrete_time_covariances(
            epochs, reference_frame=lambda times: np.tile(np.eye(3), (len(times), 1, 1)))

        for k in range(len(epochs)):
            np.testing.assert_array_equal(framed.gamma_at(k, 9), plain.gamma_at(k, 9))
            np.testing.assert_array_equal(framed.covariance_at(k, 9),
                                          plain.covariance_at(k, 9))

    def test_rotation_read_where_each_interval_acts(self, sc, pos_vel, q_cont):
        """T is read at the noise epochs and held over its own interval."""
        requested = []

        def turning(times):
            requested.append(np.array(times))
            return np.array([_rotation([1e-2 * t, 0.4, -0.2]) for t in times])

        definition = _dmc_definition(sc, pos_vel)
        beta = np.array([1e-3, 2e-3, 3e-3])
        dmc = scb.DynamicalModelCompensation(q_cont, state_definition=definition, beta=beta)
        epochs = [0.0, 60.0, 100.0, 120.0]
        dmc._compute_discrete_time_covariances(epochs, reference_frame=turning)

        (mid_epochs,) = requested
        np.testing.assert_array_equal(mid_epochs, [0.0, 30.0, 80.0, 110.0])
        for k, (dt, T) in enumerate(zip(np.diff(epochs), turning(mid_epochs)[1:]), start=1):
            single = scb.DynamicalModelCompensation(q_cont, state_definition=definition,
                                                    beta=beta)
            single._compute_discrete_time_covariances(
                [0.0, dt], reference_frame=lambda times, T=T: np.array([T] * len(times)))
            np.testing.assert_array_equal(dmc.covariance_at(k, 9), single.covariance_at(1, 9))

    def test_isotropic_noise_only_turns_the_correlation(self, sc, pos_vel):
        """With one decay rate and one sigma, position and velocity see no frame,
        and a_fogm is correlated with them through T."""
        q_iso = scb.CovarianceMatrix([scb.ArrayWUnits(2e-8, km * sec**-2)] * 3,
                                     scb.EpochArray(0.0, sys="TDB"), from_list=True)
        definition = _dmc_definition(sc, pos_vel)
        T = _rotation([0.3, -1.1, 0.7])
        plain = scb.DynamicalModelCompensation(q_iso, state_definition=definition,
                                               beta=1 / 1800)
        framed = scb.DynamicalModelCompensation(q_iso, state_definition=definition,
                                                beta=1 / 1800)
        plain._compute_discrete_time_covariances([0.0, 600.0])
        framed._compute_discrete_time_covariances(
            [0.0, 600.0], reference_frame=lambda times: np.array([T] * len(times)))

        P, P_framed = plain.covariance_at(1, 9), framed.covariance_at(1, 9)
        atol = 1e-12 * np.abs(P).max()
        np.testing.assert_allclose(P_framed[:6, :6], P[:6, :6], rtol=1e-12, atol=atol)
        np.testing.assert_allclose(P_framed[6:, 6:], P[6:, 6:], rtol=1e-12, atol=atol)
        np.testing.assert_allclose(P_framed[:6, 6:], P[:6, 6:] @ T, rtol=1e-12, atol=atol)

    @pytest.mark.parametrize("frame", ["RTN", "body_fixed"])
    def test_filter_rotates_along_the_reference(self, tmp_path, frame):
        """The filter rotates the noise by the force model's frame, read at the
        noise epochs along the reference trajectory."""
        beta = np.array([1e-3, 2e-3, 3e-3])
        settings = {"fogm_frame": frame}
        if frame == "body_fixed":
            settings["fogm_body_fixed_frame"] = "IAU_EARTH"
        od = _dmc_arc(scb.LKF, _dmc_settings(1e-11), tmp_path, fogm_beta=beta, **settings)

        process_noise = od.process_noise
        epochs = process_noise._noise_epochs(np.asarray(od.timeline_times))
        if frame == "RTN":
            positions, velocities = od._reference_position_velocity(
                epochs, process_noise.body)
            rotations = np.array([np.column_stack(_rtn_axes(r, v))
                                  for r, v in zip(positions, velocities)])
        else:
            rotations = np.array([scb.SpiceManager.get_xfrm("IAU_EARTH", "J2000", t)
                                  for t in epochs], dtype=float)
        expected = scb.DynamicalModelCompensation(
            process_noise.continuous_time_covariance,
            state_definition=od.propagator.full_state_vector.state, beta=beta)
        expected._compute_discrete_time_covariances(
            od.timeline_times, reference_frame=lambda times: rotations)

        assert len(process_noise.discrete_time_process_covariances) == len(epochs)
        for Q, Q_expected in zip(process_noise.discrete_time_process_covariances,
                                 expected.discrete_time_process_covariances):
            np.testing.assert_allclose(Q, Q_expected, rtol=1e-12,
                                       atol=1e-12 * np.abs(Q_expected).max())
        # The rotation reaches the noise
        inertial = scb.DynamicalModelCompensation(
            process_noise.continuous_time_covariance,
            state_definition=od.propagator.full_state_vector.state, beta=beta)
        inertial._compute_discrete_time_covariances(od.timeline_times)
        Q, Q_inertial = (process_noise.discrete_time_process_covariances[-1],
                         inertial.discrete_time_process_covariances[-1])
        assert np.abs(Q - Q_inertial).max() > 1e-3 * np.abs(Q_inertial).max()

    def test_rtn_filters_agree(self, tmp_path):
        """LKF adds the rotated covariance, SRIF the rotated gamma matrix."""
        dmc = _dmc_settings(1e-10)
        history = []
        for filter_cls in (scb.LKF, scb.SRIF):
            od = _dmc_arc(filter_cls, dmc, tmp_path / filter_cls.__name__,
                          fogm_beta=np.array([1e-3, 2e-3, 3e-3]), fogm_frame="RTN")
            od.fit(max_iterations=1, verbose=False)
            history.append(od.covariance_history)

        assert len(history[0]) == len(history[1]) > 0
        for P_lkf, P_srif in zip(*history):
            np.testing.assert_allclose(np.sqrt(np.diag(P_lkf)), np.sqrt(np.diag(P_srif)),
                                       rtol=1e-8)


def _two_body_definition(sc, ast, asteroid_first=False, sc_fogm=False,
                         asteroid_estimation="estimated"):
    """
    Position and velocity of the spacecraft and of an asteroid 1000 km and
    1 m/s away, in either order. ``sc_fogm`` adds the spacecraft's a_fogm,
    which follows its velocity.
    """
    r = np.array([-2.65649403e08, 1.88325868e08, 1.01977022e08])
    v = np.array([-9.07473227, -12.48458138, -4.46117482])
    bodies = [(sc, r, v, "estimated"), (ast, r + 1000.0, v + 1e-3, asteroid_estimation)]
    if asteroid_first:
        bodies.reverse()
    definition = scb.StateDefinition()
    for body, position, velocity, estimation in bodies:
        definition = (definition
                      .position(body, scb.ArrayWFrame(position, km, J2000),
                                estimation=estimation)
                      .velocity(body, scb.ArrayWFrame(velocity, km / sec, J2000),
                                estimation=estimation))
        if sc_fogm and body is sc:
            definition = definition.param(
                "a_fogm", sc, scb.ArrayWFrame(np.zeros(3), km * sec**-2, J2000))
    return definition


class TestMultiBodyModels:
    """Each model drives the states of its own body, and the filters stack
    the models of several bodies."""

    @pytest.mark.parametrize("asteroid_first", [False, True])
    def test_rows_follow_the_body(self, sc, ast, q_cont, asteroid_first):
        definition = _two_body_definition(sc, ast, asteroid_first=asteroid_first)
        first, second = np.arange(6), np.arange(6, 12)
        sc_rows, ast_rows = (second, first) if asteroid_first else (first, second)
        # No body is the owner of the first position, a SPICE ID its body
        cases = [(None, ast if asteroid_first else sc, first),
                 (sc, sc, sc_rows), (ast, ast, ast_rows), (ast.spice_id, ast, ast_rows)]
        for body, owner, rows in cases:
            snc = scb.StateNoiseCompensation(q_cont, state_definition=definition, body=body)
            snc._compute_discrete_time_covariances([0.0, 60.0])
            assert snc.body is owner
            np.testing.assert_array_equal(snc.noise_rows_at(1), rows)

            gamma = snc.gamma_at(1, 12)
            np.testing.assert_array_equal(gamma[rows], snc._construct_gamma_matrix(60.0))
            assert not np.delete(gamma, rows, axis=0).any()

    def test_combined_matrices(self, sc, ast, q_cont):
        """
        DMC on the spacecraft and SNC on the asteroid: the gamma matrices
        side by side, their inputs block-diagonal and the covariances summed,
        so the stacked gamma maps the stacked input to the summed covariance.
        """
        definition = _two_body_definition(sc, ast, sc_fogm=True)
        dmc = scb.DynamicalModelCompensation(q_cont, state_definition=definition,
                                             beta=np.array([1e-3] * 3))
        snc = scb.StateNoiseCompensation(q_cont, state_definition=definition, body=ast)
        models = [dmc, snc]
        for model in models:
            model._compute_discrete_time_covariances([0.0, 60.0, 120.0])
        np.testing.assert_array_equal(dmc.noise_rows_at(1), np.arange(9))
        np.testing.assert_array_equal(snc.noise_rows_at(1), np.arange(9, 15))

        for k in (1, 2):
            gamma = scb.ProcessNoise.combined_gamma_at(models, k, 15)
            assert gamma.shape == (15, 6)
            np.testing.assert_array_equal(gamma[:, :3], dmc.gamma_at(k, 15))
            np.testing.assert_array_equal(gamma[:, 3:], snc.gamma_at(k, 15))

            Q_tilde = scb.ProcessNoise.combined_input_covariance_at(models, k)
            np.testing.assert_array_equal(
                Q_tilde, scipy.linalg.block_diag(dmc.input_covariance_at(k),
                                                 snc.input_covariance_at(k)))

            Q = scb.ProcessNoise.combined_covariance_at(models, k, 15)
            np.testing.assert_array_equal(
                Q, dmc.covariance_at(k, 15) + snc.covariance_at(k, 15))
            np.testing.assert_allclose(Q, gamma @ Q_tilde @ gamma.T,
                                       rtol=1e-12, atol=1e-12 * np.abs(Q).max())
            # The noise does not correlate the bodies
            assert not Q[:9, 9:].any()

    def test_single_model_unchanged(self, pos_vel, q_cont):
        snc = scb.StateNoiseCompensation(q_cont, state_definition=pos_vel)
        snc._compute_discrete_time_covariances([0.0, 60.0])
        np.testing.assert_array_equal(
            scb.ProcessNoise.combined_gamma_at([snc], 1, 6), snc.gamma_at(1, 6))
        np.testing.assert_array_equal(
            scb.ProcessNoise.combined_covariance_at([snc], 1, 6), snc.covariance_at(1, 6))
        np.testing.assert_array_equal(
            scb.ProcessNoise.combined_input_covariance_at([snc], 1),
            snc.input_covariance_at(1))

    def test_body_not_in_state_reported(self, ast, pos_vel, q_cont):
        with pytest.raises(scb.ProcessNoise.NoiseBodyNotInStateError, match="pn_ast"):
            scb.StateNoiseCompensation(q_cont, state_definition=pos_vel, body=ast)

    def test_missing_state_reported(self, sc, ast, q_cont):
        """DMC on the asteroid, whose a_fogm the state does not carry."""
        definition = _two_body_definition(sc, ast, sc_fogm=True)
        with pytest.raises(scb.ProcessNoise.MissingNoiseStateError, match="a_fogm"):
            scb.DynamicalModelCompensation(q_cont, state_definition=definition,
                                           beta=np.array([1e-3] * 3), body=ast)

    def test_considered_state_reported(self, sc, ast, q_cont):
        definition = _two_body_definition(sc, ast, asteroid_estimation="considered")
        with pytest.raises(scb.ProcessNoise.ConsideredNoiseStateError, match="pn_ast"):
            scb.StateNoiseCompensation(q_cont, state_definition=definition, body=ast)


def _two_body_arc(filter_cls, process_noise, sc, ast, out_dir, fogm_beta=None,
                  covariance_analysis=True):
    """
    A 20 min heliocentric arc estimating the position and velocity of the
    spacecraft and of the asteroid, with range and range rate of the
    spacecraft every 60 s. ``fogm_beta`` adds the spacecraft's a_fogm,
    integrated with these decay rates.
    """
    t0 = 923000647.3506516
    t1 = t0 + 1200.0
    definition = _two_body_definition(sc, ast, sc_fogm=fogm_beta is not None)
    pos_vel_sigmas = [scb.ArrayWUnits(1.0, km)] * 3 + [scb.ArrayWUnits(1e-5, km / sec)] * 3
    sigmas = pos_vel_sigmas * 2
    if fogm_beta is not None:
        sigmas[6:6] = [scb.ArrayWUnits(1e-8, km * sec**-2)] * 3
        sc_forces = scb.ForceModelTranslation(
            primary_body=sc, first_order_gauss_markov=True, fogm_beta=fogm_beta)
    else:
        sc_forces = scb.ForceModelTranslation(primary_body=sc)

    state = scb.StateArray(epoch=scb.EpochArray(t0, sys="TDB"),
                           origin=scb.CelestialBody.from_constants("SUN"), state=definition)
    propagator = scb.Propagator(
        primary_body=sc, state_vector=state,
        tspan=scb.EpochArray(np.arange(t0, t1 + 10.0, 10.0), sys="TDB"),
        force_models=[sc_forces, scb.ForceModelTranslation(primary_body=ast)])

    gs = scb.GroundStation("DSS-14")
    meas_epochs = scb.EpochArray(np.arange(t0, t1 - 30.0, 60.0), sys="TDB")
    measurements = scb.MeasurementSpec.from_dict([
        {"model": scb.RangeIdeal("R", gs, sigma=scb.ArrayWUnits(1e-3, km),
                                 state_definition=definition),
         "dataset_name": "R", "epochs": meas_epochs},
        {"model": scb.RangeRateIdeal("RR", gs, sigma=scb.ArrayWUnits(1e-6, km / sec),
                                     state_definition=definition),
         "dataset_name": "RR", "epochs": meas_epochs},
    ])
    settings = scb.FilterSettings(
        initial_covariance=scb.CovarianceMatrix(sigmas, scb.EpochArray(t0, sys="TDB"),
                                                from_list=True),
        output=scb.OutputSettings(
            solution_output_path=str(out_dir), solution_output_name="solution",
            save_reference_trajectory=False, save_estimated_trajectory=False),
        covariance_analysis=covariance_analysis,
        process_noise=process_noise,
    )
    # One kernel name per run, as SPICE keeps the earlier runs' kernels loaded
    traj_name = f"pn_two_{out_dir.parent.name}_{out_dir.name}.bsp"
    return filter_cls(propagator=propagator, settings=settings, measurements=measurements,
                      traj_name=traj_name, traj_dir=str(out_dir))


class TestMultiBodyFilters:
    """One ProcessNoiseSettings per body, each noise on its own body's rows."""

    def test_each_body_noised(self, sc, ast, q_cont, tmp_path):
        """
        The measurements observe only the spacecraft, and nothing couples the
        bodies, so a body's noise shows only in its own sigmas.
        """
        both = [scb.ProcessNoiseSettings(type="SNC", Q_cont=q_cont),
                scb.ProcessNoiseSettings(type="SNC", Q_cont=q_cont, body=ast)]
        noisy = _two_body_arc(scb.LKF, both, sc, ast, tmp_path / "both")
        # Without a body the noise drives the primary body, and the
        # asteroid is left without one
        with pytest.warns(scb.FilterOD.UnnoisedBodyWarning, match="pn_ast"):
            sc_only = _two_body_arc(scb.LKF, scb.ProcessNoiseSettings(type="SNC", Q_cont=q_cont),
                                    sc, ast, tmp_path / "sc")
        quiet = _two_body_arc(scb.LKF, None, sc, ast, tmp_path / "none")
        # The filter holds its own copies of the bodies
        assert [model.body.spice_id for model in noisy.process_noise] == [-60000, -60001]
        assert sc_only.process_noise.body.spice_id == -60000

        sigmas = []
        for od in (noisy, sc_only, quiet):
            od.fit(max_iterations=1, verbose=False)
            sigmas.append(np.sqrt(np.diag(od.covariance_history[-1])))
        sigma_noisy, sigma_sc_only, sigma_quiet = sigmas
        assert np.all(sigma_noisy > sigma_quiet)
        assert np.all(sigma_sc_only[:6] > sigma_quiet[:6])
        np.testing.assert_allclose(sigma_sc_only[:6], sigma_noisy[:6], rtol=1e-10)
        np.testing.assert_allclose(sigma_sc_only[6:], sigma_quiet[6:], rtol=1e-10)

    def test_duplicate_body_reported(self, sc, ast, q_cont, tmp_path):
        """The models add up, so the primary body named again would be
        noised twice."""
        settings = [scb.ProcessNoiseSettings(type="SNC", Q_cont=q_cont),
                    scb.ProcessNoiseSettings(type="SNC", Q_cont=q_cont, body=sc)]
        with pytest.raises(scb.FilterOD.DuplicateProcessNoiseBodyError, match="pn_sc"):
            _two_body_arc(scb.LKF, settings, sc, ast, tmp_path)

    def test_filters_agree(self, sc, ast, q_cont_rtn, tmp_path):
        """
        DMC on the spacecraft and an RTN SNC on the asteroid, rotated with
        the asteroid's own reference. LKF sums the covariances and SRIF
        stacks the gamma matrices, so they agree only if both are aligned.
        """
        settings = [_dmc_settings(1e-10),
                    scb.ProcessNoiseSettings(type="SNC", Q_cont=q_cont_rtn, body=ast)]
        fogm_beta = np.array([1e-3] * 3)
        lkf = _two_body_arc(scb.LKF, settings, sc, ast, tmp_path / "lkf", fogm_beta)
        srif = _two_body_arc(scb.SRIF, settings, sc, ast, tmp_path / "srif", fogm_beta)

        times = lkf.timeline_times
        mid_epochs = np.concatenate((times[:1], 0.5 * (times[:-1] + times[1:])))
        positions, velocities = lkf._reference_position_velocity(mid_epochs, body=ast)
        sc_positions, _ = lkf._reference_position_velocity(mid_epochs)
        np.testing.assert_allclose(positions[0] - sc_positions[0], [1000.0] * 3, rtol=1e-9)
        rtn = lkf.process_noise[1]
        assert len(rtn.Q_tilde) == len(times)
        for Q, position, velocity in zip(rtn.Q_tilde, positions, velocities):
            _assert_rtn_variances(Q, position, velocity, (1e-8, 2e-8, 3e-8))

        for od in (lkf, srif):
            od.fit(max_iterations=1, verbose=False)
        assert len(lkf.covariance_history) == len(srif.covariance_history) > 0
        for P_lkf, P_srif in zip(lkf.covariance_history, srif.covariance_history):
            assert P_lkf.shape == (15, 15)
            np.testing.assert_allclose(np.sqrt(np.diag(P_lkf)), np.sqrt(np.diag(P_srif)),
                                       rtol=1e-8)

    def test_smoothed_covariance_agrees(self, sc, ast, q_cont, tmp_path):
        """The SRIF smoother stacks the gamma matrices as the filter does."""
        settings = [_dmc_settings(1e-10),
                    scb.ProcessNoiseSettings(type="SNC", Q_cont=q_cont, body=ast)]
        smoothed = []
        for filter_cls in (scb.LKF, scb.SRIF):
            od = _two_body_arc(filter_cls, settings, sc, ast, tmp_path / filter_cls.__name__,
                               fogm_beta=np.array([1e-3] * 3), covariance_analysis=False)
            od.fit(max_iterations=1, verbose=False, if_sequential_smooth=True)
            smoothed.append(od._solution_history_smooth[-1].covariance_smooth)

        assert len(smoothed[0]) == len(smoothed[1]) > 0
        for P_lkf, P_srif in zip(*smoothed):
            np.testing.assert_allclose(np.sqrt(np.diag(P_lkf)), np.sqrt(np.diag(P_srif)),
                                       rtol=1e-8)


# endregion Tests #
# -----------------#
