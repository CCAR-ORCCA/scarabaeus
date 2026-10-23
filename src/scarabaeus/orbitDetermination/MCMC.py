# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import warnings
from typing import Callable, Optional

import emcee

from scarabaeus import FilterOD, SolutionOD, MonteCarloSamples
import scarabaeus.utils.NumpyWrapper as np
from scarabaeus.utils.Utils import Utils


# --------------------#
#  Class Definition  #
# --------------------#
class MCMC(FilterOD):
    """
    Markov chain Monte Carlo sampling of a converged orbit determination.

    A filter (:class:`~scarabaeus.LSB`, :class:`~scarabaeus.SRIFB`,
    :class:`~scarabaeus.LKF`, :class:`~scarabaeus.SRIF`, ...) is first fitted
    to convergence. Its last pass fixes the reference trajectory, the prefit
    residuals :math:`y_k`, the measurement partials :math:`H_k` and the STMs
    :math:`\\Phi(t_k, t_0)`. About that reference the measurements are linear
    in the deviation :math:`x` at the reference epoch,

    .. math::

        y_k = A_k x + \\epsilon_k, \\qquad A_k = H_k \\Phi(t_k, t_0), \\qquad
        \\epsilon_k \\sim \\mathcal{N}(0, R_k),

    and the posterior of :math:`x` is sampled with the affine-invariant
    ensemble sampler of ``emcee``:

    .. math::

        \\log p(x \\mid y) = -\\frac{1}{2} \\sum_k (y_k - A_k x)^T R_k^{-1}
            (y_k - A_k x) + \\log p(x) + \\mathrm{const}.

    No trajectory is propagated and no measurement is recomputed while
    sampling: every step costs two small matrix products. The data enter
    through the normal matrix and vector

    .. math::

        D = \\sum_k A_k^T R_k^{-1} A_k, \\qquad
        d = \\sum_k A_k^T R_k^{-1} y_k.

    **Whitening**

    The walkers move in :math:`\\theta`, with :math:`x = \\hat{x} + L\\theta`,
    where :math:`\\hat{x} = M^{-1} b` and :math:`L L^T = M^{-1}` are the batch
    least-squares solution and covariance (:math:`M = \\bar{P}_0^{-1} + D`,
    :math:`b = \\bar{P}_0^{-1} \\bar{x}_0 + d`). With the filter's Gaussian
    a-priori the log-posterior is then :math:`-\\tfrac{1}{2}\\theta^T\\theta`,
    whatever the conditioning of :math:`M`. A user prior :math:`\\log p` replaces
    the Gaussian a-priori, and the log-posterior becomes

    .. math::

        -\\frac{1}{2} \\theta^T L^T D L\\, \\theta
        + \\theta^T L^T (d - D \\hat{x})
        + \\log p(\\hat{x} + L\\theta - \\bar{x}_0),

    up to a constant. The prior is evaluated on the deviation from the
    a-priori state, so a Gaussian
    :math:`\\log p(\\delta) = -\\tfrac{1}{2}\\delta^T \\bar{P}_0^{-1}\\delta`
    reproduces the default.

    **Sequences**

    For a :class:`~scarabaeus.MissionSequence` every leg is mapped to one
    global vector :math:`z`: the first leg's state at its epoch, followed by
    each block a later leg adds at its node, with its a-priori sigma. Leg
    :math:`j` is :math:`x_j = T_j z`, with
    :math:`T_j = Q_j \\Phi_{j-1}(t_{node}) T_{j-1}` plus an identity on the
    blocks the node adds, so a measurement of leg :math:`j` has
    :math:`A_k = H_k \\Phi_j(t_k) T_j`. This is the batch of
    :class:`~scarabaeus.LSB`, with every leg mapped back through the nodes,
    as long as no considered state is dropped at a node: a dropped one stays
    considered in :math:`z`, whereas the filter folds its spread into the next
    leg's a priori and lets that leg's data reduce it
    (:class:`ConsiderDroppedWarning`).

    **Consider parameters**

    Considered states are not sampled. The walkers sample the estimated states
    with the considered ones at their reference, and the consider spread is
    added linearly as in the Schmidt batch: each particle draws
    :math:`c \\sim \\mathcal{N}(0, P_{cc})` and is shifted by
    :math:`S_{xc} c`, with :math:`S_{xc} = -M_{xx}^{-1} M_{xc}`.

    **Solution**

    :meth:`sample` returns a :class:`~scarabaeus.SolutionOD` whose deviation
    and covariance histories are the chain mean and covariance (plus the
    consider part) mapped to every epoch with the STMs, and whose filter
    carries ``particles`` (the posterior samples mapped the same way), so
    :attr:`SolutionOD.uq_metrics` is filled. The samples at the reference
    epoch are also stored as :attr:`samples`, ready for
    :meth:`~scarabaeus.Plotting.plot_corner`.

    Parameters
    ----------
    filter_obj : FilterOD or SolutionOD
        A filter fitted to convergence with :meth:`FilterOD.fit`, or the
        solution it returned. It is copied, not modified.
    verbose : bool, optional
        Print the initialization banner. Defaults to ``True``.
    walkers_per_dimension : int, optional
        Walkers per estimated state when :meth:`sample` is not given
        ``n_walkers``. The stretch move needs more than twice the dimension;
        four times keeps the ensemble from collapsing onto a subspace.
        Defaults to 4.
    min_walkers : int, optional
        Fewest walkers when :meth:`sample` is not given ``n_walkers``.
        Defaults to 32.
    walker_spread : float, optional
        Spread of the initial walkers, in units of the filter's 1-sigma. The
        walkers are drawn from the converged filter's Gaussian scaled by this
        factor, so with 1 the chain starts close to stationarity on the
        linearized problem and the burn-in is short. Defaults to 1.0.
    max_initial_redraws : int, optional
        Times an initial walker is redrawn when the log-posterior is not finite
        at its position (a ``log_prior`` that rules out part of the Gaussian)
        before :meth:`sample` gives up. Defaults to 100.
    stretch_scale : float, optional
        Scale parameter :math:`a` of the affine-invariant stretch move of
        ``moves="stretch"``, the default of Goodman & Weare (2010) and of
        ``emcee``. Defaults to 2.0.
    de_weights : tuple of float, optional
        Relative weights of the differential-evolution and snooker moves of
        ``moves="de"``; ``emcee`` normalizes them. Defaults to ``(0.7, 0.3)``.
    burn_in_factor : float, optional
        Burn-in, in multiples of the largest integrated autocorrelation time
        of the chain, and at most half the chain. Defaults to 2.0.
    thin_factor : float, optional
        Thinning, in multiples of the smallest integrated autocorrelation time
        of the chain, and at least one step. Defaults to 0.5.
    tau_multiple : float, optional
        Chain length, in largest integrated autocorrelation times, below which
        the autocorrelation estimate (and with it burn-in and thinning) is not
        trusted, and :class:`ShortChainWarning` is issued. The ``tol`` default
        of ``emcee``. Defaults to 50.
    min_acceptance : float, optional
        Mean acceptance fraction below which the ensemble mixes poorly and
        :class:`AcceptanceFractionWarning` is issued. There is no upper bound:
        the stretch move scales its steps with the spread of the ensemble, so
        a high acceptance does not mean short steps. On the whitened,
        near-Gaussian posterior of a few states it is about 0.5 or more.
        Defaults to 0.2.

    Raises
    ------
    FilterTypeError
        If ``filter_obj`` is not a filter (or the solution of one), or is
        itself an :class:`MCMC`.
    CovarianceAnalysisError
        If the filter ran a covariance analysis: there are no residuals to
        sample from.
    UnfittedFilterError
        If the filter was never fitted.
    FilterNotConvergedError
        If the last fit did not converge: the linearization would not be about
        the solution.
    ProcessNoiseError
        If the filter has process noise, which the batch posterior of the
        reference-epoch state cannot represent.

    See Also
    --------
    scarabaeus.LSB : The Gaussian batch solution the chain reproduces.
    scarabaeus.MonteCarloSamples : Container of the reference-epoch samples.
    scarabaeus.Plotting.plot_corner : Corner plot of the samples.

    Notes
    -----
    The sampled posterior is the linearized one: the dynamics and measurement
    models are not re-evaluated, so the chain reproduces the filter's Gaussian
    unless a non-Gaussian prior is given. Its value is in the prior and in a
    sample-based uncertainty description consistent with the particle filters.

    The source filter's smoother, gain and innovation histories describe its
    own run, not the sampled posterior, and are not carried over.

    The sampler settings, ``walkers_per_dimension`` to ``min_acceptance``, are
    kept as attributes of the same name, so they can also be changed between
    calls to :meth:`sample`.

    References
    ----------
    .. [1] Foreman-Mackey, D., Hogg, D. W., Lang, D., & Goodman, J. (2013).
       emcee: The MCMC Hammer. *Publications of the Astronomical Society of
       the Pacific*, 125(925), 306-312.
    .. [2] Goodman, J., & Weare, J. (2010). Ensemble samplers with affine
       invariance. *Communications in Applied Mathematics and Computational
       Science*, 5(1), 65-80.
    .. [3] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
       Orbit Determination*. Elsevier Academic Press.
    """

    # The posterior is the batch one at the reference epoch.
    _supports_process_noise = False

    # Every leg holds the global solution, mapped back through the nodes.
    _maps_back_sequence = True

    # Attributes of the source filter's own run (its smoother, gains,
    # innovations, particles), not of the sampled posterior: dropped when the
    # filter is copied, so they do not end up in the MCMC solution.
    _SOURCE_ONLY_ATTRIBUTES = (
        "smoothed_state_deviation",
        "_smoothed_state_deviation",
        "smoothed_covariance",
        "_smoothed_covariance",
        "postfit_residuals_smooth",
        "postfit_epochs_smooth",
        "kalman_gains",
        "_kalman_gains",
        "innovation_covariances",
        "_innovation_covariances",
        "innovations",
        "_innovations",
        "particles",
        "weights",
    )

    # ---------------#
    # region Errors #
    # ---------------#
    class FilterTypeError(TypeError):
        """The object to sample is not a fitted filter or its solution."""

    class CovarianceAnalysisError(ValueError):
        """The filter ran a covariance analysis, so there are no residuals."""

    class UnfittedFilterError(ValueError):
        """The filter was never fitted."""

    class FilterNotConvergedError(ValueError):
        """The filter's last fit did not converge."""

    class ProcessNoiseError(ValueError):
        """The filter has process noise, which the batch posterior cannot hold."""

    class ConsiderPartitionError(ValueError):
        """A state is estimated on one leg and considered on the next."""

    class AprioriSigmaSizeError(ValueError):
        """A block added at a node has fewer a-priori sigmas than scalars."""

    class NoEstimatedStatesError(ValueError):
        """Every state is considered, so there is nothing to sample."""

    class PosteriorCovarianceError(ValueError):
        """The batch covariance is not positive definite."""

    class WalkerCountError(ValueError):
        """Too few walkers for the ensemble moves."""

    class InitialWalkersError(ValueError):
        """The log-posterior is not finite at any redrawn initial walker."""

    class UnknownMoveError(ValueError):
        """The move name is not one of the known ensemble moves."""

    class SampleSizeError(ValueError):
        """The number of steps or particles is not a positive integer."""

    class FitNotSupportedError(NotImplementedError):
        """MCMC samples a fitted filter and does not fit itself."""

    # endregion Errors #
    # ------------------#

    # -----------------#
    # region Warnings #
    # -----------------#
    class ShortChainWarning(UserWarning):
        """
        The chain is shorter than ``tau_multiple`` times its largest
        autocorrelation time, so the burn-in, thinning and effective
        sample size are rough. Run more steps, or silence it with
        ``warnings.filterwarnings("ignore", category=MCMC.ShortChainWarning)``.
        """

    class AcceptanceFractionWarning(UserWarning):
        """
        The mean acceptance fraction is below ``min_acceptance``: the
        walkers mix poorly. Silence
        it with ``warnings.filterwarnings("ignore",
        category=MCMC.AcceptanceFractionWarning)``.
        """

    class ConsiderDroppedWarning(UserWarning):
        """
        A considered state leaves the state vector at a sequence node. The
        chain keeps it considered over the later legs, whereas the sequential
        filters fold its spread into the next leg's a priori, where that leg's
        data reduce it, so the sampled posterior is not the filter's. Keep the
        considered state in the later legs to match the filter, or silence it
        with ``warnings.filterwarnings("ignore",
        category=MCMC.ConsiderDroppedWarning)``.
        """

    # endregion Warnings #
    # --------------------#

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        filter_obj: FilterOD | SolutionOD,
        verbose: bool = True,
        *,
        walkers_per_dimension: int = 4,
        min_walkers: int = 32,
        walker_spread: float = 1.0,
        max_initial_redraws: int = 100,
        stretch_scale: float = 2.0,
        de_weights: tuple = (0.7, 0.3),
        burn_in_factor: float = 2.0,
        thin_factor: float = 0.5,
        tau_multiple: float = 50,
        min_acceptance: float = 0.2,
    ):
        """
        Copy a converged filter to sample its linearized posterior.

        Parameters
        ----------
        filter_obj : FilterOD or SolutionOD
            Filter fitted to convergence, or its solution.
        verbose : bool, optional
            Print the initialization banner. Defaults to ``True``.
        walkers_per_dimension : int, optional
            Walkers per estimated state when ``n_walkers`` is not given.
            Defaults to 4.
        min_walkers : int, optional
            Fewest walkers when ``n_walkers`` is not given. Defaults to 32.
        walker_spread : float, optional
            Spread of the initial walkers, in filter 1-sigmas. Defaults to 1.0.
        max_initial_redraws : int, optional
            Redraws of an initial walker where the log-posterior is not
            finite. Defaults to 100.
        stretch_scale : float, optional
            Scale of the stretch move. Defaults to 2.0.
        de_weights : tuple of float, optional
            Weights of the differential-evolution and snooker moves. Defaults
            to ``(0.7, 0.3)``.
        burn_in_factor : float, optional
            Burn-in, in largest autocorrelation times. Defaults to 2.0.
        thin_factor : float, optional
            Thinning, in smallest autocorrelation times. Defaults to 0.5.
        tau_multiple : float, optional
            Chain length, in largest autocorrelation times, below which
            :class:`ShortChainWarning` is issued. Defaults to 50.
        min_acceptance : float, optional
            Mean acceptance fraction below which
            :class:`AcceptanceFractionWarning` is issued. Defaults to 0.2.

        Raises
        ------
        FilterTypeError
            If ``filter_obj`` is not a filter or is an :class:`MCMC`.
        CovarianceAnalysisError
            If the filter ran a covariance analysis.
        UnfittedFilterError
            If the filter was never fitted.
        FilterNotConvergedError
            If the last fit did not converge.
        ProcessNoiseError
            If the filter has process noise.
        """
        if isinstance(filter_obj, SolutionOD):
            filter_obj = filter_obj.filter
        if not isinstance(filter_obj, FilterOD) or isinstance(filter_obj, MCMC):
            raise MCMC.FilterTypeError(
                f"MCMC samples a fitted filter (LSB, SRIFB, LKF, SRIF, ...) or its "
                f"SolutionOD; got {type(filter_obj).__name__}."
            )
        if filter_obj.covariance_analysis:
            raise MCMC.CovarianceAnalysisError(
                "The filter ran a covariance analysis: there are no residuals to "
                "sample the posterior from."
            )
        if filter_obj.convergence_info is None:
            raise MCMC.UnfittedFilterError(
                f"The {type(filter_obj).__name__} filter was never fitted: call "
                f"fit() first."
            )
        if not filter_obj.convergence_info.get("converged", False):
            raise MCMC.FilterNotConvergedError(
                f"The {type(filter_obj).__name__} filter did not converge after "
                f"{filter_obj.convergence_info.get('iterations')} iteration(s): "
                f"the linearized posterior would not be about the solution."
            )
        if (
            filter_obj.process_noise is not None
            or filter_obj.SNC_flag
            or filter_obj.DMC_flag
        ):
            raise MCMC.ProcessNoiseError(
                "The filter has process noise (SNC/DMC), which the batch posterior "
                "of the reference-epoch state cannot represent. Estimate the "
                "unmodelled accelerations as stochastic parameters instead."
            )

        # Shallow copy: the reference trajectory and measurement data are read,
        # never modified, and every history is rebuilt as a new object.
        self.__dict__.update(filter_obj.__dict__)
        for name in MCMC._SOURCE_ONLY_ATTRIBUTES:
            self.__dict__.pop(name, None)

        self.walkers_per_dimension = walkers_per_dimension
        self.min_walkers = min_walkers
        self.walker_spread = walker_spread
        self.max_initial_redraws = max_initial_redraws
        self.stretch_scale = stretch_scale
        self.de_weights = de_weights
        self.burn_in_factor = burn_in_factor
        self.thin_factor = thin_factor
        self.tau_multiple = tau_multiple
        self.min_acceptance = min_acceptance

        self.source_filter = type(filter_obj).__name__
        self.samples = None
        self.chain = None
        self.log_prob = None
        self.diagnostics = None
        self.posterior_mean = None
        self.posterior_covariance = None
        self._labels = None

        if verbose:
            print("")
            print("=" * 80)
            print(f"Initializing MCMC sampler on the converged {self.source_filter}...")
            print("=" * 80)

    # endregion Constructor #
    # -----------------------#

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def labels(self) -> Optional[list]:
        """Name of each entry of the global state :attr:`samples` holds."""
        return self._labels

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    def fit(self, *args, **kwargs):
        """
        Not supported: the source filter is fitted, then :meth:`sample` runs.

        Raises
        ------
        FitNotSupportedError
            Always.
        """
        raise MCMC.FitNotSupportedError(
            "MCMC does not fit: fit the source filter to convergence, then call "
            "MCMC(filter).sample()."
        )

    def sample(
        self,
        n_steps: int = 5000,
        n_walkers: Optional[int] = None,
        moves="stretch",
        log_prior: Optional[Callable[[np.ndarray], float]] = None,
        n_particles: int = 1000,
        seed: Optional[int] = None,
        meas_corr: Optional[list] = None,
        progress: bool = False,
        verbose: bool = True,
    ) -> SolutionOD:
        """
        Sample the linearized posterior and pack it as a solution.

        Parameters
        ----------
        n_steps : int, optional
            Steps of each walker. Defaults to 5000.
        n_walkers : int, optional
            Number of walkers, at least twice the number of estimated states.
            Defaults to :attr:`walkers_per_dimension` per estimated state, and
            at least :attr:`min_walkers`.
        moves : str or emcee move(s), optional
            ``"stretch"`` (affine-invariant stretch move, scale
            :attr:`stretch_scale`), ``"de"`` (differential evolution mixed with
            snooker moves, :attr:`de_weights`), or any ``moves`` argument of
            :class:`emcee.EnsembleSampler`. Defaults to ``"stretch"``.
        log_prior : callable, optional
            Log-density of the prior, called on the deviation of the estimated
            states from their a-priori value (a 1-D array, in the order of
            the estimated entries of :attr:`labels`). It replaces the
            filter's Gaussian a-priori; return ``-np.inf`` outside the
            support. Defaults to the Gaussian a-priori.
        n_particles : int, optional
            Posterior samples kept as particles. Drawn from the burnt-in,
            thinned chain, with replacement only if it is shorter. Defaults to
            1000.
        seed : int, optional
            Seed of the initial walkers, the sampler, the particle draw and
            the consider draws. Defaults to ``None`` (not reproducible).
        meas_corr : list, optional
            Measurement correlation factors, as in :meth:`FilterOD.fit`.
            Defaults to ``None``.
        progress : bool, optional
            Show the ``emcee`` progress bar. Defaults to ``False``.
        verbose : bool, optional
            Print the chain diagnostics. Defaults to ``True``.

        Returns
        -------
        SolutionOD
            The chain mean and covariance mapped to every epoch, the residuals
            about the mean, and :attr:`SolutionOD.uq_metrics` of the particles.
            It is not saved; call :meth:`SolutionOD.save`.

        Raises
        ------
        SampleSizeError
            If ``n_steps`` or ``n_particles`` is not positive.
        NoEstimatedStatesError
            If every state is considered.
        WalkerCountError
            If ``n_walkers`` is less than twice the number of estimated states.
        UnknownMoveError
            If ``moves`` is an unknown name.
        PosteriorCovarianceError
            If the batch covariance is not positive definite.
        InitialWalkersError
            If ``log_prior`` is not finite at the redrawn initial walkers.
        ConsiderPartitionError
            If a state changes between estimated and considered at a node.
        AprioriSigmaSizeError
            If a block added at a node has too few a-priori sigmas.

        Warns
        -----
        ShortChainWarning
            If the chain is short compared with its autocorrelation time.
        AcceptanceFractionWarning
            If the mean acceptance fraction is below :attr:`min_acceptance`.
        ConsiderDroppedWarning
            If a considered state is dropped at a sequence node.

        Notes
        -----
        Besides the solution this sets :attr:`samples` (global state at the
        reference epoch, as deviations from the converged reference),
        :attr:`chain` (every step of every walker, estimated states only,
        shape ``(n_steps, n_walkers, n_estimated)``), :attr:`log_prob`,
        :attr:`diagnostics`, :attr:`posterior_mean` and
        :attr:`posterior_covariance`. ``particles`` has shape
        ``(n_particles, n, n_epochs)``, zero-padded on legs with fewer states.
        """
        if int(n_steps) != n_steps or n_steps < 1:
            raise MCMC.SampleSizeError(
                f"n_steps must be a positive integer; got {n_steps}."
            )
        if int(n_particles) != n_particles or n_particles < 1:
            raise MCMC.SampleSizeError(
                f"n_particles must be a positive integer; got {n_particles}."
            )
        n_steps, n_particles = int(n_steps), int(n_particles)
        rng = np.random.default_rng(seed)

        # Start from the first leg: the source left its last leg's partition.
        if self.flag_sequence:
            self._validate_sequence_batch()
            self.n = self.legs_n[0]
            self.flag_node = False
        self._init_consider_partition()

        records = self._epoch_records(meas_corr)
        T, z_bar, P_z, I_z, con, labels = self._global_layout()
        est = ~con
        idx_est, idx_con = np.flatnonzero(est), np.flatnonzero(con)
        n_est = idx_est.size
        if n_est == 0:
            raise MCMC.NoEstimatedStatesError(
                "Every state is considered: there is nothing to sample."
            )
        self._labels = labels

        if n_walkers is None:
            n_walkers = max(self.walkers_per_dimension * n_est, self.min_walkers)
        if n_walkers < 2 * n_est:
            raise MCMC.WalkerCountError(
                f"{n_walkers} walkers for {n_est} estimated states: the ensemble "
                f"moves need at least {2 * n_est}."
            )
        move = self._ensemble_moves(moves)

        # Normal equations of the data, and of the batch with the a-priori
        D, d, D_xc = self._normal_equations(records, T, idx_est, idx_con)
        I_xx = I_z[np.ix_(idx_est, idx_est)]
        M_xx = I_xx + D
        b_x = I_xx @ z_bar[idx_est] + d
        M_xc = I_z[np.ix_(idx_est, idx_con)] + D_xc

        P_xx = self._safe_inv(M_xx)
        P_xx = 0.5 * (P_xx + P_xx.T)
        x_hat = np.linalg.solve(M_xx, b_x)
        try:
            L = np.linalg.cholesky(P_xx)
        except np.linalg.LinAlgError as exc:
            raise MCMC.PosteriorCovarianceError(
                "The batch covariance of the estimated states is not positive "
                "definite: the data and a-priori do not determine every state."
            ) from exc

        log_prob_fn = self._log_posterior(x_hat, L, D, d, z_bar[idx_est], log_prior)
        theta0 = self._initial_walkers(log_prob_fn, n_walkers, n_est, rng)

        sampler = emcee.EnsembleSampler(
            n_walkers, n_est, log_prob_fn, moves=move, vectorize=True
        )
        sampler.random_state = np.random.RandomState(
            int(rng.integers(2**32 - 1))
        ).get_state()
        sampler.run_mcmc(theta0, n_steps, progress=progress)

        theta, burn, thin = self._converged_chain(sampler, n_steps)
        X = x_hat + theta @ L.T
        chain_mean = X.mean(axis=0)
        chain_cov = np.atleast_2d(np.cov(X, rowvar=False))

        # Consider spread, linear as in the Schmidt batch
        P_cc = P_z[np.ix_(idx_con, idx_con)]
        S_xc = -P_xx @ M_xc
        z_mean = np.zeros(P_z.shape[0])
        z_mean[idx_est] = chain_mean
        P_post = np.zeros_like(P_z)
        P_post[np.ix_(idx_est, idx_est)] = chain_cov + S_xc @ P_cc @ S_xc.T
        P_post[np.ix_(idx_est, idx_con)] = S_xc @ P_cc
        P_post[np.ix_(idx_con, idx_est)] = (S_xc @ P_cc).T
        P_post[np.ix_(idx_con, idx_con)] = P_cc

        picks = rng.choice(X.shape[0], n_particles, replace=n_particles > X.shape[0])
        w, V = np.linalg.eigh(P_cc)
        C = rng.standard_normal((n_particles, idx_con.size)) @ (
            V * np.sqrt(np.clip(w, 0.0, None))
        ).T
        Z = np.zeros((n_particles, P_z.shape[0]))
        Z[:, idx_est] = X[picks] + C @ S_xc.T
        Z[:, idx_con] = C

        self.posterior_mean = z_mean
        self.posterior_covariance = P_post
        self.samples = MonteCarloSamples(Z, epoch=self._reference_epoch(), labels=labels)
        self.chain = x_hat + sampler.get_chain() @ L.T
        self.log_prob = sampler.get_log_prob()
        self._store_diagnostics(sampler, n_steps, n_walkers, burn, thin, X.shape[0])
        self._map_solution(records, T, z_mean, P_post, Z)

        if verbose:
            self._print_diagnostics()
        return SolutionOD(self)

    def _epoch_records(self, meas_corr: Optional[list]) -> list:
        """
        Collect, at every timeline epoch, its leg, STM and measurement block.

        Same walk as :class:`~scarabaeus.LSB`: a measurement at a sequence
        event epoch is not accumulated, but is mapped with the new leg.

        Parameters
        ----------
        meas_corr : list or None
            Measurement correlation factors.

        Returns
        -------
        list of dict
            Per epoch: ``epoch``, ``leg``, ``stm``, ``accumulate``, and when
            there are measurements, ``block`` (its partials cut to the leg's
            size) and the weight ``W``.
        """
        measurement_times, _, timeline_times = self._build_timeline()
        records = []
        counter_events = 0
        for k, epoch in enumerate(timeline_times):
            has_measurement = bool(np.any(measurement_times == epoch))
            at_event = False
            if self.flag_sequence and counter_events < len(self.events_epochs):
                if epoch == self.events_epochs[counter_events]:
                    counter_events += 1
                    at_event = True
                    self.n = self.legs_n[counter_events]

            if self.flag_sequence:
                stm = self.trajectory.get_STM_sequence(
                    epoch=float(epoch), idx_leg=counter_events
                )
            else:
                stm = self.trajectory.get_STM(epoch=epoch, idx=k)

            record = {
                "epoch": epoch,
                "leg": counter_events,
                "stm": np.asarray(stm, dtype=float),
                "accumulate": has_measurement and not at_event,
                "block": None,
            }
            if has_measurement:
                block = dict(self._epoch_measurement_block(epoch))
                block["partials"] = np.asarray(block["partials"])[:, : self.n]
                record["block"] = block
                record["W"] = self._safe_inv(
                    self._measurement_covariance(block["sigma"], epoch, meas_corr)
                )
            records.append(record)
        return records

    def _global_layout(self) -> tuple:
        """
        Map every leg to the global state :math:`z` the chain samples.

        Returns
        -------
        T : list of np.ndarray
            Per leg, the ``(n_leg, n_z)`` map :math:`x_j = T_j z`.
        z_bar : np.ndarray
            A-priori deviation of :math:`z`, shape ``(n_z,)``.
        P_z, I_z : np.ndarray
            A-priori covariance and information of :math:`z`.
        con : np.ndarray
            Boolean mask of the considered entries of :math:`z`.
        labels : list of str
            Name of each entry of :math:`z`.

        Raises
        ------
        ConsiderPartitionError
            If a state carried across a node changes partition.
        AprioriSigmaSizeError
            If a new block has fewer sigmas than scalars.
        """
        P0 = np.asarray(self.initial_covariance.matrix_without_units(), dtype=float)
        n0 = P0.shape[0]
        con_legs = [self._consider_mask(0)]
        labels = self._leg_labels(0)
        T = [np.eye(n0)]
        z_bar = [self._apriori_deviation(0).ravel()]
        con = [con_legs[0]]
        sigmas = []
        n_z = n0

        n_legs = len(self.legs_n) if self.flag_sequence else 1
        for k in range(1, n_legs):
            old_model, new_model = self._node_state_models(k, True)
            Q, new_n, new_blocks = Utils.build_reorder_operator(
                old_model.state_definition, new_model.state_definition
            )
            prev_n = self.legs_n[k - 1]
            phi_end = np.asarray(self.trajectory._STMs[k - 1][-1]).reshape(
                prev_n, prev_n
            )
            con_legs.append(self._consider_mask(k))
            rows, cols = np.nonzero(Q)
            if np.any(con_legs[k][rows] != con_legs[k - 1][cols]):
                raise MCMC.ConsiderPartitionError(
                    f"A state carried into leg {k} changes between estimated and "
                    f"considered at the node; keep its partition across legs."
                )
            dropped = np.setdiff1d(np.flatnonzero(con_legs[k - 1]), cols)
            if dropped.size:
                prev_labels = self._leg_labels(k - 1)
                names = [prev_labels[i] for i in dropped if i < len(prev_labels)]
                warnings.warn(
                    f"Considered state(s) {names or dropped.tolist()} of leg {k - 1} "
                    f"are dropped at the node into leg {k}. The chain keeps them "
                    f"considered over every later leg, while the filter folds their "
                    f"spread into leg {k}'s a priori, which that leg's data reduce: "
                    f"the sampled mean and covariance differ from the filter's.",
                    MCMC.ConsiderDroppedWarning,
                )

            n_new = sum(size for _name, _start, size in new_blocks)
            T_k = np.zeros((new_n, n_z + n_new))
            T_k[:, :n_z] = Q @ phi_end @ T[k - 1]
            block_units = self._state_block_units(new_model)
            x_bar_k = self._apriori_deviation(k).ravel()
            leg_labels = self._leg_labels(k)
            for name, start, size in new_blocks:
                sigma = np.asarray(
                    self._apriori_sigma_values(name, size, block_units.get(start)),
                    dtype=float,
                ).ravel()
                if sigma.size != size:
                    raise MCMC.AprioriSigmaSizeError(
                        f"State '{name}' added at leg {k} has {size} entries but "
                        f"{sigma.size} a-priori sigma(s)."
                    )
                T_k[start : start + size, n_z : n_z + size] = np.eye(size)
                sigmas.append(sigma)
                z_bar.append(x_bar_k[start : start + size])
                con.append(con_legs[k][start : start + size])
                labels.extend(leg_labels[start : start + size])
                n_z += size
            T.append(T_k)

        T = [np.hstack([T_j, np.zeros((T_j.shape[0], n_z - T_j.shape[1]))]) for T_j in T]
        new_var = np.concatenate(sigmas) ** 2 if sigmas else np.zeros(0)
        P_z = np.zeros((n_z, n_z))
        I_z = np.zeros((n_z, n_z))
        P_z[:n0, :n0] = P0
        I_z[:n0, :n0] = self._safe_inv(P0)
        P_z[n0:, n0:] = np.diag(new_var)
        I_z[n0:, n0:] = np.diag(1.0 / new_var)
        if len(labels) != n_z:
            labels = [f"z[{i}]" for i in range(n_z)]
        return T, np.concatenate(z_bar), P_z, I_z, np.concatenate(con), labels

    def _consider_mask(self, idx_leg: int) -> np.ndarray:
        """Boolean mask of the considered entries of a leg's state."""
        _idx_est, idx_con, n = self._partition_indices(
            self._leg_state_definition(idx_leg)
        )
        mask = np.zeros(n, dtype=bool)
        mask[idx_con] = True
        return mask

    def _leg_labels(self, idx_leg: int) -> list:
        """Name of each entry of a leg's state, as in the solution files."""
        return SolutionOD._state_labels(
            SolutionOD._components_metadata(self._leg_state_definition(idx_leg))
        )

    @staticmethod
    def _normal_equations(records: list, T: list, idx_est, idx_con) -> tuple:
        """
        Accumulate the data part of the normal equations in :math:`z`.

        Parameters
        ----------
        records : list of dict
            Output of :meth:`_epoch_records`.
        T : list of np.ndarray
            Leg maps from :meth:`_global_layout`.
        idx_est, idx_con : np.ndarray
            Estimated and considered entries of :math:`z`.

        Returns
        -------
        D, d, D_xc : np.ndarray
            :math:`\\sum A_x^T W A_x`, :math:`\\sum A_x^T W y` and
            :math:`\\sum A_x^T W A_c`.
        """
        n_est, n_con = idx_est.size, idx_con.size
        D = np.zeros((n_est, n_est))
        d = np.zeros(n_est)
        D_xc = np.zeros((n_est, n_con))
        for record in records:
            if not record["accumulate"]:
                continue
            block = record["block"]
            A = block["partials"] @ record["stm"] @ T[record["leg"]]
            A_x_W = A[:, idx_est].T @ record["W"]
            D += A_x_W @ A[:, idx_est]
            d += A_x_W @ np.asarray(block["residuals"], dtype=float).ravel()
            D_xc += A_x_W @ A[:, idx_con]
        return D, d, D_xc

    @staticmethod
    def _log_posterior(x_hat, L, D, d, x_bar, log_prior) -> Callable:
        """
        Vectorized log-posterior in the whitened coordinates :math:`\\theta`.

        Parameters
        ----------
        x_hat, L : np.ndarray
            Batch solution and Cholesky factor of its covariance.
        D, d : np.ndarray
            Data normal matrix and vector.
        x_bar : np.ndarray
            A-priori deviation of the estimated states.
        log_prior : callable or None
            User log-prior of the deviation from ``x_bar``; ``None`` for the
            Gaussian a-priori.

        Returns
        -------
        callable
            Maps walkers ``(n_walkers, n_est)`` to log-posteriors
            ``(n_walkers,)``, ``-inf`` where not finite.
        """
        if log_prior is None:
            return lambda theta: -0.5 * np.sum(theta**2, axis=1)

        D_w = L.T @ D @ L
        d_w = L.T @ (d - D @ x_hat)

        def log_prob(theta):
            deviations = x_hat + theta @ L.T - x_bar
            prior = np.array([float(log_prior(dx)) for dx in deviations])
            value = (
                -0.5 * np.einsum("wi,ij,wj->w", theta, D_w, theta)
                + theta @ d_w
                + prior
            )
            return np.where(np.isfinite(value), value, -np.inf)

        return log_prob

    def _initial_walkers(self, log_prob_fn, n_walkers: int, n_est: int, rng) -> np.ndarray:
        """
        Draw the initial walkers about the batch solution.

        Parameters
        ----------
        log_prob_fn : callable
            Vectorized log-posterior.
        n_walkers, n_est : int
            Ensemble size and dimension.
        rng : numpy.random.Generator
            Random generator.

        Returns
        -------
        np.ndarray
            Initial walkers, shape ``(n_walkers, n_est)``.

        Raises
        ------
        InitialWalkersError
            If some walkers stay where the log-posterior is not finite.
        """
        spread = self.walker_spread
        theta = spread * rng.standard_normal((n_walkers, n_est))
        for _ in range(self.max_initial_redraws):
            bad = ~np.isfinite(log_prob_fn(theta))
            if not np.any(bad):
                return theta
            theta[bad] = spread * rng.standard_normal((int(bad.sum()), n_est))
        raise MCMC.InitialWalkersError(
            f"The log-posterior is not finite at {int(bad.sum())} initial walker(s) "
            f"after {self.max_initial_redraws} redraws: the prior rules "
            f"out the filter solution's neighbourhood."
        )

    def _ensemble_moves(self, moves):
        """
        Translate a move name into ``emcee`` moves.

        Parameters
        ----------
        moves : str or emcee move(s)
            ``"stretch"``, ``"de"``, or moves passed through as they are.

        Returns
        -------
        emcee move(s)

        Raises
        ------
        UnknownMoveError
            If ``moves`` is an unknown name.
        """
        if not isinstance(moves, str):
            return moves
        if moves == "stretch":
            return emcee.moves.StretchMove(a=self.stretch_scale)
        if moves == "de":
            w_de, w_snooker = self.de_weights
            return [
                (emcee.moves.DEMove(), w_de),
                (emcee.moves.DESnookerMove(), w_snooker),
            ]
        raise MCMC.UnknownMoveError(
            f"Unknown move '{moves}': use 'stretch', 'de' or emcee moves."
        )

    def _converged_chain(self, sampler, n_steps: int) -> tuple:
        """
        Burn-in and thin the chain from its autocorrelation times.

        Parameters
        ----------
        sampler : emcee.EnsembleSampler
            Sampler after the run.
        n_steps : int
            Steps of each walker.

        Returns
        -------
        theta : np.ndarray
            Flat burnt-in, thinned chain in whitened coordinates.
        burn, thin : int
            Steps discarded and thinning stride.

        Warns
        -----
        ShortChainWarning
            If ``n_steps`` is below :attr:`tau_multiple` times the largest
            autocorrelation time.
        """
        tau = np.asarray(sampler.get_autocorr_time(tol=0), dtype=float)
        tau = np.where(np.isfinite(tau) & (tau > 0.0), tau, float(n_steps))
        self._tau = tau
        if n_steps < self.tau_multiple * tau.max():
            warnings.warn(
                f"The chain has {n_steps} steps, fewer than "
                f"{self.tau_multiple} autocorrelation times "
                f"({tau.max():.1f} steps): burn-in, thinning and the effective "
                f"sample size are rough.",
                MCMC.ShortChainWarning,
            )
        burn = min(int(self.burn_in_factor * tau.max()), n_steps // 2)
        thin = max(1, int(self.thin_factor * tau.min()))
        return sampler.get_chain(discard=burn, thin=thin, flat=True), burn, thin

    def _store_diagnostics(
        self, sampler, n_steps: int, n_walkers: int, burn: int, thin: int, n_kept: int
    ) -> None:
        """
        Store the chain diagnostics in :attr:`diagnostics`.

        Warns
        -----
        AcceptanceFractionWarning
            If the mean acceptance fraction is below :attr:`min_acceptance`.
        """
        acceptance = np.asarray(sampler.acceptance_fraction, dtype=float)
        if acceptance.mean() < self.min_acceptance:
            warnings.warn(
                f"Mean acceptance fraction {acceptance.mean():.3f} is below "
                f"{self.min_acceptance}: the walkers mix poorly.",
                MCMC.AcceptanceFractionWarning,
            )
        self.diagnostics = {
            "n_steps": n_steps,
            "n_walkers": n_walkers,
            "burn_in": burn,
            "thin": thin,
            "n_samples": n_kept,
            "autocorrelation_time": self._tau,
            "effective_sample_size": n_walkers * (n_steps - burn) / self._tau,
            "acceptance_fraction": acceptance,
            "mean_acceptance_fraction": float(acceptance.mean()),
            "source_filter": self.source_filter,
        }

    def _reference_epoch(self) -> float:
        """Epoch of the global state: the first leg's, or the arc's initial one."""
        if self.flag_sequence:
            return float(np.asarray(self.legs_epochs[0]).ravel()[0])
        return float(np.asarray(self.propagator.full_state_vector.epoch.et).ravel()[0])

    def _map_solution(self, records: list, T: list, z_mean, P_post, Z) -> None:
        """
        Map the posterior and its particles to every epoch, as a filter would.

        Fills the deviation, covariance and residual histories, the node
        solutions and ``particles``.

        Parameters
        ----------
        records : list of dict
            Output of :meth:`_epoch_records`.
        T : list of np.ndarray
            Leg maps from :meth:`_global_layout`.
        z_mean : np.ndarray
            Posterior mean of :math:`z`.
        P_post : np.ndarray
            Posterior covariance of :math:`z`.
        Z : np.ndarray
            Particles of :math:`z`, shape ``(n_particles, n_z)``.
        """
        # New lists: the shallow copy shares the source filter's ones.
        self.state_deviation_history = []
        self.covariance_history = []
        self.covariance_consider_history = []
        prefits, postfits = [], []
        n_max = max(self.legs_n) if self.flag_sequence else T[0].shape[0]
        particles = np.zeros((Z.shape[0], n_max, len(records)))

        for j, record in enumerate(records):
            M = record["stm"] @ T[record["leg"]]
            deviation = (M @ z_mean).reshape(-1, 1)
            covariance = M @ P_post @ M.T
            P, P_cons = self._split_consider_covariance(
                record["leg"], 0.5 * (covariance + covariance.T)
            )
            self.state_deviation_history.append(deviation)
            self.covariance_history.append(P)
            self.covariance_consider_history.append(P_cons)
            particles[:, : M.shape[0], j] = Z @ M.T

            block = record["block"]
            if block is not None:
                residuals = block["residuals"]
                postfit = residuals - (block["partials"] @ deviation).reshape(
                    np.shape(residuals)
                )
                meta = [
                    block["sigma"],
                    block["instruments"],
                    block["measurement_types"],
                    block["dataset_names"],
                    block["t2"],
                ]
                prefits.append([residuals, *meta])
                postfits.append([postfit, *meta])

        self._node_deviations = []
        self._node_covariances = []
        for k in range(1, len(T)):
            prev_n = self.legs_n[k - 1]
            phi_end = np.asarray(self.trajectory._STMs[k - 1][-1]).reshape(
                prev_n, prev_n
            )
            M = phi_end @ T[k - 1]
            self._node_deviations.append((M @ z_mean).reshape(-1, 1))
            self._node_covariances.append(M @ P_post @ M.T)

        self.particles = particles
        self.prefit_residuals = self._unwrap_residuals(
            prefits, self.measurement_data.dataset_names, epoch_attr="prefit_epochs"
        )
        self.postfit_residuals = self._unwrap_residuals(
            postfits, self.measurement_data.dataset_names, epoch_attr="postfit_epochs"
        )

    def _print_diagnostics(self) -> None:
        """Print the chain diagnostics."""
        diag = self.diagnostics
        print("")
        print(
            f"MCMC: {diag['n_walkers']} walkers x {diag['n_steps']} steps, "
            f"burn-in {diag['burn_in']}, thin {diag['thin']}, "
            f"{diag['n_samples']} samples kept"
        )
        print(
            f"  mean acceptance fraction = {diag['mean_acceptance_fraction']:.3f}, "
            f"autocorrelation time = {diag['autocorrelation_time'].min():.1f}"
            f"-{diag['autocorrelation_time'].max():.1f} steps, "
            f"min ESS = {diag['effective_sample_size'].min():.0f}"
        )

    # endregion Methods #
    # -------------------#
