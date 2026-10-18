# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, Button, TextBox
from matplotlib.path import Path
from matplotlib.patches import Patch
from scipy.stats import chi2
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import HuberRegressor
from datetime import datetime

from scarabaeus import constants

# Plot style
_RCPARAMS = {
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.4,
    "font.family": "serif",
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.linestyle": "--",
}

_C_ACTIVE = "#1f77b4"
_C_OUTLIER = "#d62728"


# -----------------------#
#    Class Definition   #
# -----------------------#
class MeasurementEditing:
    """ Interactive and automated editing of measurement residuals for orbit determination.

    Provides these editing modes:

    1. **Lasso Selector** — graphically select points to flag as outliers on a plot.
    2. **Chi-Squared** — automatically flag points whose residuals exceed a chi-squared threshold.
    3. **Kurtosis** — automatically trim the heaviest tails until the residual
       kurtosis falls below a target (3, the Gaussian value, by default).
    4. **Date Range** — manually specify time windows to flag or unflag observations.
    5. **Isolation Forest** — flag anomalous residuals by tree-isolation depth,
       with the rejection threshold taken from the knee of the ranked scores
       rather than from an assumed contamination fraction. Works jointly on the
       two components of a vector observable.
    6. **Huber / MAD** — detrend each pass with a Huber regression, then
       threshold on a robust MAD scale.
    7. **Residual Interval** — flag residuals outside a fixed interval, by
       default 5 sigma about the mean.

    Parameters
    ----------
    datasets : list of MeasurementDataSet
        The datasets to edit. Nested lists are flattened one level.

    See Also
    --------
    scarabaeus.FilterDataManager : Combines datasets and applies editing before indexing.
    scarabaeus.FilterOD : Base filter that consumes the edited datasets.

    Notes
    -----
    Edits are non-destructive: original ``outlier_flag`` arrays are copied
    internally and not written back until :meth:`apply_flags` or
    :meth:`get_datasets` is called.

    Examples
    --------
    .. code-block:: python

        editor = MeasurementEditing(datasets)
        editor.lasso_editor(dataset_name="DSN_Range")
        editor.chi_squared_filter(sigma_scale=3.0)
        editor.kurtosis_filter(target_kurtosis=3.0)
        editor.date_range_filter("2024-01-01T00:00:00", "2024-01-05T12:00:00")
        edited_datasets = editor.get_datasets()

    :meth:`lasso_editor_all`, :meth:`chi_squared_filter_all` and
    :meth:`kurtosis_filter_all` are the all-figures-at-once counterparts of
    :meth:`lasso_editor`, :meth:`chi_squared_filter` and
    :meth:`kurtosis_filter`, for editing several datasets without blocking on
    one figure at a time.
    """

    # -------------#
    # Constructor  #
    # -------------#
    def __init__(self, datasets):
        if datasets is None:
            datasets = []

        # Flatten one level of nesting
        self._datasets = [
            d
            for item in datasets
            for d in (item if isinstance(item, (list, tuple)) else [item])
        ]

        # Mirror outlier_flag arrays so edits are non-destructive on originals
        self._outlier_flags = {
            d.set_name: np.array(d.data["outlier_flag"], dtype=bool).copy()
            for d in self._datasets
        }
        # Sigmas re-estimated by huber_mad_filter, keyed by dataset name. The
        # editor only owns datasets, so applying these to the measurement
        # models is left to the caller.
        self._refit_sigmas = {}

    # -----------------------------------
    #             Properties
    # -----------------------------------
    @property
    def datasets(self) -> list:
        """
        The list of managed datasets.

        Returns
        -------
        list of MeasurementDataSet
            Datasets as stored internally (flags not yet written back unless
            :meth:`apply_flags` or :meth:`get_datasets` has been called).
        """
        return self._datasets

    @property
    def outlier_flags(self) -> dict:
        """Dict of {dataset_name: bool array} of current outlier flags."""
        return self._outlier_flags

    @property
    def refit_sigmas(self) -> dict:
        """Dict of {dataset_name: float} sigmas re-estimated by huber_mad_filter."""
        return self._refit_sigmas

    # -----------------------------------
    #     Lasso Selector
    # -----------------------------------
    def lasso_editor(
        self,
        dataset_name=None,
        x_field: str = "t2",
        y_field: str = "residuals",
        title: str = None,
    ):
        """
        Open an interactive matplotlib figure for lasso-based outlier selection.

        Left-click drag draws a lasso; enclosed *visible* points are toggled as
        outliers. Buttons:

        * **[Confirm]** — commit the current flags to the editor, hide the
          rejected points, and rescale the axes to the remaining points.
          The figure stays open so editing can continue.
        * **[Show/Hide Rejected]** — toggle visibility of rejected points
          (including points whose ``outlier_flag`` was already True when the
          editor opened), rescaling the axes each time. Hidden points cannot
          be lassoed.
        * **[Reset]** — clear all flags for the displayed dataset(s).
        * **[Done]** — commit and close the figure.

        Multiple datasets of the same data type can be edited together (e.g.
        all range measurements regardless of which measurement model produced
        them) by passing a list of names or ``None`` for all managed datasets.
        Each dataset's active points get their own color; flags are committed
        back to each dataset individually.

        Parameters
        ----------
        dataset_name : str or list of str or None
            Name(s) of the dataset(s) to edit (must match
            ``MeasurementDataSet.set_name``). ``None`` edits all managed
            datasets at once.
        x_field : str
            Field to use for x-axis (default: 't2').
        y_field : str
            Field to use for y-axis (default: 'residuals').
        title : str, optional
            Plot title override.
        """
        names = self._resolve_names(dataset_name)
        self._build_lasso_figure(names, x_field, y_field, title)
        plt.show()

    def _resolve_names(self, dataset_name):
        """Resolve a dataset selector (str, list of str, or None = all) to a
        list of dataset names."""
        if dataset_name is None:
            return [d.set_name for d in self._datasets]
        if isinstance(dataset_name, str):
            return [dataset_name]
        return list(dataset_name)

    def lasso_editor_all(
        self,
        dataset_name=None,
        x_field: str = "t2",
        y_field: str = "residuals",
        title: str = None,
    ):
        """
        Open a SEPARATE lasso-editor figure for each dataset and display them
        all at once with a single ``plt.show()``.

        Each figure is fully interactive on its own (its own lasso plus the
        Confirm / Show-Hide Rejected / Reset / Done buttons); committing writes
        the flags back to that dataset. Control returns once every figure has
        been closed.

        Use this instead of calling :meth:`lasso_editor` in a per-dataset loop.
        Looping blocks on each dataset's ``plt.show()``, which is clunky on
        remote/browser backends: closing a plot tab there does
        not advance the loop to the next dataset. Here a single ``plt.show()``
        drives every dataset's editor together.

        Parameters
        ----------
        dataset_name : str or list of str or None
            Dataset(s) to edit — one figure each. ``None`` edits all managed
            datasets.
        x_field : str
            Field to use for the x-axis (default: 't2').
        y_field : str
            Field to use for the y-axis (default: 'residuals').
        title : str, optional
            Title override applied to every figure.
        """
        names = self._resolve_names(dataset_name)
        # Hold references so the figures (and their widgets) stay alive for the
        # single blocking plt.show() below.
        self._open_figs = [
            self._build_lasso_figure([name], x_field, y_field, title)
            for name in names
        ]
        plt.show()

    def _build_lasso_figure(
        self,
        names,
        x_field: str = "t2",
        y_field: str = "residuals",
        title: str = None,
    ):
        """
        Build (but do NOT display) an interactive lasso-editor figure for the
        given dataset ``names``.

        Points are concatenated across datasets with per-dataset colors and an
        owner map so flags commit back per dataset. Returns the Matplotlib
        figure; the caller is responsible for calling ``plt.show()``.
        """
        # Concatenate points across datasets, keeping an owner map so flags
        # can be split back per dataset on commit.
        xs, ys, sel_parts, owners = [], [], [], []
        for i, name in enumerate(names):
            dset = self._get_dataset(name)
            x_i = np.array(dset.data[x_field]).ravel()
            y_i = np.array(dset.data[y_field]).ravel()
            xs.append(x_i)
            ys.append(y_i)
            sel_parts.append(self._outlier_flags[name].copy())
            owners.append(np.full(len(x_i), i))
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        sel = np.concatenate(sel_parts)  # working selection (True = rejected)
        owner = np.concatenate(owners)
        xy = np.column_stack([x, y])

        show_rejected = [True]  # mutable for closures
        label = ", ".join(names)

        plt.rcParams.update(_RCPARAMS)
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.18)

        # One active-scatter per dataset (own color); one shared rejected-scatter
        colors = (
            [_C_ACTIVE]
            if len(names) == 1
            else [plt.cm.tab10(i % 10) for i in range(len(names))]
        )
        scatters = [
            ax.scatter([], [], s=14, color=colors[i], alpha=0.8, zorder=3)
            for i in range(len(names))
        ]
        scatter_out = ax.scatter(
            [], [], s=14, c=_C_OUTLIER, alpha=0.6, marker="x", zorder=3
        )

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(x_field)
        ax.set_ylabel(y_field)
        ax.legend(
            handles=[
                *(
                    Patch(color=colors[i], label=names[i])
                    for i in range(len(names))
                ),
                Patch(color=_C_OUTLIER, label="Rejected"),
            ],
            fontsize=9,
        )

        def _visible_mask():
            # Active points are always visible; rejected only when toggled on
            return ~sel | (sel & show_rejected[0])

        def _refresh(rescale=False):
            for i, sc in enumerate(scatters):
                m = ~sel & (owner == i)
                sc.set_offsets(
                    np.column_stack([x[m], y[m]]) if m.any() else np.empty((0, 2))
                )
            m_out = sel & show_rejected[0]
            scatter_out.set_offsets(
                np.column_stack([x[m_out], y[m_out]])
                if m_out.any()
                else np.empty((0, 2))
            )
            ax.set_title(
                title
                or f"Lasso Editor — {label}  |  {int(sel.sum())} outliers flagged"
                f"{'' if show_rejected[0] else '  (rejected hidden)'}"
            )
            if rescale:
                vis = _visible_mask()
                if vis.any():
                    x_v, y_v = x[vis], y[vis]
                    dx = (x_v.max() - x_v.min()) or 1.0
                    dy = (y_v.max() - y_v.min()) or 1.0
                    ax.set_xlim(x_v.min() - 0.05 * dx, x_v.max() + 0.05 * dx)
                    ax.set_ylim(y_v.min() - 0.05 * dy, y_v.max() + 0.05 * dy)
            fig.canvas.draw_idle()

        def _lasso_callback(verts):
            path = Path(verts)
            inside = path.contains_points(xy) & _visible_mask()
            if not inside.any():
                return
            # Toggle: if all visible points inside are already flagged → unflag;
            # else flag them
            if sel[inside].all():
                sel[inside] = False
            else:
                sel[inside] = True
            _refresh()

        lasso = LassoSelector(ax, _lasso_callback, useblit=True)

        # Buttons
        ax_toggle = plt.axes([0.10, 0.04, 0.20, 0.06])
        ax_reset = plt.axes([0.34, 0.04, 0.14, 0.06])
        ax_confirm = plt.axes([0.52, 0.04, 0.14, 0.06])
        ax_done = plt.axes([0.70, 0.04, 0.14, 0.06])
        btn_toggle = Button(
            ax_toggle, "Show/Hide Rejected", color="#fff3e0", hovercolor="#ffe0b2"
        )
        btn_reset = Button(ax_reset, "Reset", color="#e8eaf6", hovercolor="#c5cae9")
        btn_confirm = Button(
            ax_confirm, "Confirm", color="#c8e6c9", hovercolor="#a5d6a7"
        )
        btn_done = Button(ax_done, "Done", color="#bbdefb", hovercolor="#90caf9")

        def _commit():
            for i, name in enumerate(names):
                new_flags = sel[owner == i].copy()
                self._outlier_flags[name] = new_flags
                print(
                    f"[MeasurementEditing] Lasso: {int(new_flags.sum())} outliers "
                    f"flagged in '{name}'."
                )

        def _on_confirm(_):
            # Commit, hide rejected points, rescale, and keep editing
            _commit()
            show_rejected[0] = False
            _refresh(rescale=True)

        def _on_done(_):
            _commit()
            plt.close(fig)

        def _on_reset(_):
            sel[:] = False
            _refresh(rescale=True)

        def _on_toggle(_):
            show_rejected[0] = not show_rejected[0]
            _refresh(rescale=True)

        btn_confirm.on_clicked(_on_confirm)
        btn_done.on_clicked(_on_done)
        btn_reset.on_clicked(_on_reset)
        btn_toggle.on_clicked(_on_toggle)

        # Keep widget references alive for the lifetime of the figure
        fig._lasso_widgets = (lasso, btn_toggle, btn_reset, btn_confirm, btn_done)

        _refresh(rescale=True)
        return fig
    
    # -----------------------------------
    #     Residual Interval
    # -----------------------------------
    def residual_filter(
        self,
        dataset_name=None,
        mean_outlier: float = None,
        sigma_outlier: float = None,
    ):
        """
        Flag measurements whose residuals fall outside a fixed interval.

        A measurement is rejected when its residual lies outside
        ``[mean_outlier - sigma_outlier, mean_outlier + sigma_outlier]``. With
        neither bound given this is a 5-sigma test about the mean. Both
        statistics come from the points not already flagged: including
        known-bad points would inflate the spread and hide the outliers this is
        meant to find, which matters because flags accumulate across passes.

        The default is not robust. An outlier inflates the spread it is tested
        against, so a lone outlier among n - 1 tight points is only caught when
        n > 26, whatever its size. Use :meth:`huber_mad_filter` unless explicit
        bounds are what is wanted.

        Has no preview figure, like ``date_range_filter``.

        Parameters
        ----------
        dataset_name : str, list of str, or None
            Datasets to filter. ``None`` = all.
        mean_outlier : float, optional
            Centre of the accepted interval, in residual units. Defaults to the
            mean of the unflagged residuals.
        sigma_outlier : float, optional
            Half-width of the accepted interval, in residual units. Defaults to
            5 times the standard deviation of the unflagged residuals.

            Both are plain numbers by definition: they are compared directly
            with the ``residuals`` column, which is in the units of the
            dataset's measurements (km for range, Hz for Doppler, RU for
            sequential ranging, ...). The same values are applied to every
            dataset filtered, so give explicit bounds one measurement type at
            a time (``dataset_name``).

        Returns
        -------
        dict
            ``{dataset_name: numpy.ndarray}`` of the indices flagged by this
            call, excluding any that were already flagged.

        Raises
        ------
        KeyError
            If a dataset has no ``residuals`` column.
        """
        newly = {}
        for name in self._resolve_names(dataset_name):
            dset = self._get_dataset(name)
            if "residuals" not in dset.data:
                raise KeyError(
                    f"Dataset '{name}' has no 'residuals' column; compute "
                    "residuals before calling residual_filter()."
                )
            residuals = np.asarray(dset.data["residuals"], dtype=float).ravel()
            keep = ~self._outlier_flags[name]
            if not np.any(keep):
                newly[name] = np.array([], dtype=int)
                continue
            reference = residuals[keep]

            mean = np.mean(reference) if mean_outlier is None else mean_outlier
            half_width = 5.0 * np.std(reference) if sigma_outlier is None else sigma_outlier
            outside = (residuals < mean - half_width) | (residuals > mean + half_width)

            newly[name] = np.nonzero(outside & keep)[0]
            self._outlier_flags[name] = self._outlier_flags[name] | outside
            print(
                f"[MeasurementEditing] Residual interval ({mean:.6g} +/- {half_width:.6g}): "
                f"{newly[name].size} new outliers flagged in '{name}'."
            )
        return newly

    # -----------------------------------
    #     Huber / MAD
    # -----------------------------------
    def huber_mad_filter(
        self,
        dataset_name=None,
        n_sigma: float = 5.0,
        mad_scale: float = 1.4826,
        refit_sigma: bool = True,
        pass_gap_sec: float = None,
        min_pass_points: int = None,
    ):
        """
        Flag outliers by detrending residuals against time with a Huber
        regression, then thresholding on a MAD-based robust scale.

        A tracking pass usually carries a slope that a plain mean-and-sigma
        test reads as spread, so the trend is removed first; the median
        absolute deviation then keeps the scale estimate from being dragged
        by the outliers it is meant to find. Unlike ``chi_squared_filter``,
        this judges the residuals by their own scatter rather than by the
        sigma already attached to the measurements, which is what makes it
        usable before a trustworthy sigma exists.

        Has no preview figure, like ``date_range_filter``.

        Parameters
        ----------
        dataset_name : str, list of str, or None
            Datasets to filter. ``None`` = all.
        n_sigma : float
            Acceptance width, in units of the robust scale below.
        mad_scale : float
            Factor turning the median absolute deviation into that scale.
            Defaults to 1.4826, which makes the MAD a consistent estimator of
            the standard deviation for normal data, so ``n_sigma`` reads as a
            number of sigmas. Pass 1.0 to threshold on the raw MAD instead.
        refit_sigma : bool
            Re-estimate a measurement sigma from the surviving points and
            record it in :attr:`refit_sigmas`. The editor does not own the
            measurement models, so applying it is left to the caller.
        pass_gap_sec : float, optional
            Gap above which consecutive observations belong to separate passes,
            each detrended on its own. Defaults to
            :attr:`Constants.PASS_GAP_DEFAULT` (6 h); raise or lower it for a mission
            whose tracking cadence differs from DSN deep-space passes.
        min_pass_points : int, optional
            Fewest points a pass may have before it is merged into its
            neighbour instead of being detrended alone. Defaults to
            :attr:`Constants.PASS_MIN_POINTS_DEFAULT`.

        Returns
        -------
        dict
            ``{dataset_name: float}`` of re-estimated sigmas, empty when
            ``refit_sigma`` is False.
        """
        for name in self._resolve_names(dataset_name):
            new_flags, sigma_new = self._huber_mad_evaluate(
                name, n_sigma, mad_scale, refit_sigma, pass_gap_sec, min_pass_points
            )
            self._huber_mad_apply(name, new_flags, n_sigma, sigma_new)
        return self._refit_sigmas

    def _huber_mad_evaluate(
        self,
        name,
        n_sigma,
        mad_scale,
        refit_sigma,
        pass_gap_sec=None,
        min_pass_points=None,
    ):
        """
        Decide which observations in one dataset to reject, and optionally
        re-estimate its measurement sigma.

        Does not modify anything: it reads the dataset's residuals and epochs and
        returns its verdict. :meth:`_huber_mad_filter` loops over the datasets
        and :meth:`_huber_mad_apply` is what records the result.

        The work is in four steps.

        1. **Split into passes.** Observations separated by more than
           *pass_gap_sec* belong to different tracking passes, which have
           different geometry, station and bias, so each is treated on its own.
           See :meth:`_pass_segments`.
        2. **Detrend each pass.** A pass carries a slope that a plain
           mean-and-sigma test would read as spread, so a Huber regression of
           residual against epoch is fitted and subtracted. The Huber loss is
           used rather than least squares so the outliers being hunted do not
           drag the trend towards themselves. Already-flagged points are excluded
           from the fit but still detrended, so an earlier edit does not skew the
           trend yet its points still get a verdict.
        3. **Set a robust scale.** The median absolute deviation of the
           detrended residuals, pooled over the passes, times *mad_scale*. The
           MAD is used instead of a standard deviation because it is not inflated
           by the very points being looked for.
        4. **Threshold.** Anything beyond *n_sigma* times that scale is rejected.

        Parameters
        ----------
        name : str
            Name of the dataset to evaluate. Its ``residuals`` and ``t1`` columns
            are read; nothing is written.
        n_sigma : float
            Acceptance width, in units of the robust scale from step 3. A point
            is rejected when ``|detrended residual| > n_sigma * scale``.
        mad_scale : float
            Factor converting the median absolute deviation into that scale.
            1.4826 makes the MAD a consistent estimator of the standard
            deviation for normal data, so *n_sigma* reads as a number of sigmas;
            1.0 thresholds on the raw MAD instead.
        refit_sigma : bool
            Whether to also re-estimate the measurement sigma from the surviving
            points. When False, ``sigma_new`` comes back None and the quadratic
            fit is skipped entirely.
        pass_gap_sec : float, optional
            Gap above which observations belong to separate passes. Defaults to
            :attr:`Constants.PASS_GAP_DEFAULT`.
        min_pass_points : int, optional
            Fewest points a pass may have before being merged into its
            neighbour. Defaults to :attr:`Constants.PASS_MIN_POINTS_DEFAULT`.

        Returns
        -------
        new_flags : numpy.ndarray of bool
            One entry per observation in the dataset. True means reject. Points
            already flagged are left False here, since this reports only what
            *this* call adds; :meth:`_huber_mad_apply` ORs it into the working
            set. All False when the dataset has fewer than three unflagged
            points, or when the robust scale comes out zero.
        sigma_new : float or None
            Re-estimated measurement sigma: the scatter of the surviving
            residuals about a per-pass quadratic in time, pooled across passes.
            None when *refit_sigma* is False, when no pass has more than three
            surviving points, or when the dataset was too small to evaluate.

        Notes
        -----
        The sigma is returned rather than applied because the editor does not own
        the measurement models. :meth:`MeasurementSpec.edit` is what puts it back
        on them, resolving the units it should carry.
        """
        dset = self._get_dataset(name)
        residuals = np.array(dset.data["residuals"]).ravel()
        times = np.array(dset.data["t1"]).ravel()

        # Fit on the points not already flagged, so earlier edits do not skew
        # the trend or inflate the scale.
        keep = ~self._outlier_flags[name]
        none_flagged = np.zeros(residuals.shape, dtype=bool)
        if int(keep.sum()) < 3:
            return none_flagged, None

        # Detrend one pass at a time. A dataset that spans several passes has no
        # single trend: the geometry, the station and the bias change at every
        # boundary, so a line fitted across all of them tracks none of them and
        # leaves pass-to-pass offsets in the residuals. Where a pass then sits
        # far from that single line, every point in it clears the rejection
        # threshold and the whole pass is discarded.
        #
        # Splitting has to be conservative: see _pass_segments for why a
        # too-eager split is worse than not splitting at all.
        # Computed once and reused by the refit below, so the detrend and the
        # sigma estimate cannot disagree about where the pass boundaries are.
        segments = MeasurementEditing._pass_segments(
            times,
            gap_threshold_sec=pass_gap_sec,
            keep=keep,
            min_points=min_pass_points,
        )

        global_median = float(np.median(residuals[keep]))
        detrended = np.array(residuals, dtype=float)
        for segment in segments:
            fit_idx = segment[keep[segment]]
            if fit_idx.size < 3:
                # Too few points for a trend. Centre on the median of the whole
                # dataset, never on the segment's own median: for a segment of
                # one that subtracts the point from itself, so the residual is
                # exactly zero and a gross outlier becomes invisible to the
                # threshold below -- silently un-flagging the very points this
                # filter exists to catch.
                detrended[segment] = residuals[segment] - global_median
                continue

            # Centre and normalise the epoch before fitting. The times are
            # absolute ephemeris seconds (~6.6e8) spanning a pass of a few
            # thousand, so the raw feature is badly scaled for the solver. This
            # is a pure reparameterisation -- the fitted line, and therefore
            # ``detrended``, is unchanged -- applied identically to the fit and
            # the prediction.
            t0 = float(np.mean(times[fit_idx]))
            t_scale = float(np.std(times[fit_idx])) or 1.0
            t_fit = (times[segment] - t0) / t_scale
            t_fit_keep = (times[fit_idx] - t0) / t_scale

            # The residuals are centred and scaled too: good points and
            # acquisition rows can span 1e5, which leaves the l-BFGS solver
            # ill-conditioned even on a normalised epoch.
            r_fit = residuals[fit_idx]
            r_mid = float(np.median(r_fit))
            r_scale = float(np.median(np.abs(r_fit - r_mid))) or (float(np.std(r_fit)) or 1.0)

            huber = HuberRegressor()
            try:
                huber.fit(t_fit_keep[:, np.newaxis], (r_fit - r_mid) / r_scale)
                detrended[segment] = residuals[segment] - (
                    huber.predict(t_fit[:, np.newaxis]) * r_scale + r_mid
                )
            except Exception:
                # scikit-learn's Huber solver aborts on residuals with an
                # extreme dynamic range -- a single gross outlier among
                # otherwise smooth values, which for sequential ranging is what
                # a modulo wrap looks like (one point at -1.1e6 RU against a
                # trend of ~2.2e3). That is exactly the case this filter exists
                # to catch, so failing the whole run here would be perverse.
                #
                # Fall back to a constant detrend about the median. The linear
                # trend is not removed, which makes the test slightly less
                # sensitive, but the MAD scale and the rejection below are
                # unaffected and the gross outlier is still caught on the next
                # pass.
                detrended[segment] = residuals[segment] - np.median(
                    residuals[fit_idx]
                )

        median = np.median(detrended[keep])
        robust_std = mad_scale * np.median(np.abs(detrended[keep] - median))
        if robust_std == 0:
            return none_flagged, None

        new_flags = (np.abs(detrended - median) > n_sigma * robust_std) & keep

        sigma_new = None
        if refit_sigma:
            # Fit the residual curvature against time, per pass.
            #
            # This used ``x = np.linspace(0, len(y), len(y))`` -- the sample
            # index -- so it measured curvature per observation rather than per
            # second. Any gap in the data was then invisible: two points a
            # second apart and two points three hours apart were treated as
            # equally spaced, and across a pass boundary the quadratic was
            # fitted through a discontinuity it could not see. The epoch is
            # centred and normalised per pass as above, which also matters more
            # here than for the line, since a quadratic in raw ephemeris seconds
            # squares ~6.6e8 into ~4.4e17.
            surviving = keep & ~new_flags
            residual_parts = []
            for segment in segments:
                idx = segment[surviving[segment]]
                if idx.size <= 3:
                    continue
                t = times[idx]
                t = (t - float(np.mean(t))) / (float(np.std(t)) or 1.0)
                y = detrended[idx]
                residual_parts.append(y - np.polyval(np.polyfit(t, y, 2), t))

            # One scale for the dataset, pooled over the passes: the sigma is a
            # per-dataset quantity, and each pass on its own is usually too
            # short for a stable estimate.
            if residual_parts:
                pooled = np.concatenate(residual_parts)
                if pooled.size > 1:
                    sigma_new = float(np.std(pooled, ddof=1))

        return new_flags, sigma_new

    @staticmethod
    def _pass_segments(
        times,
        gap_threshold_sec: float = None,
        keep=None,
        min_points: int = None,
    ):
        """
        Group observation indices into passes, split on gaps in time.

        Splitting is deliberately conservative, because an over-eager split is
        worse than no split at all. Each segment is detrended on its own, so a
        segment too short to support a fit gets centred instead, and a segment of
        one centred on itself has a residual of exactly zero -- which hides a
        gross outlier from the rejection threshold completely. A 300 s threshold
        cut a single real ranging pass into 38 near-empty segments and stopped
        the filter catching the modulo-wrap outliers it had caught before. So: a
        large gap threshold, and any segment still too small is merged back.

        Parameters
        ----------
        times : array_like
            Observation epochs [s]. Need not be sorted; the grouping is done in
            time order and the indices returned refer to the original order.
        gap_threshold_sec : float, optional
            Gap above which a new pass begins. Defaults to
            :attr:`Constants.PASS_GAP_DEFAULT`.
        keep : array_like of bool, optional
            Mask of points that will actually be fitted. Segment sizes are
            measured over these, since already-flagged points do not help a fit.
        min_points : int, optional
            Fewest points a segment may have. Defaults to
            :attr:`Constants.PASS_MIN_POINTS_DEFAULT`.

        Returns
        -------
        list of numpy.ndarray
            One index array per pass, together covering every observation
            exactly once. A single pass gives one array covering everything,
            which is the common case for real tracking data, where each dataset
            is read from one pass file -- so the per-pass detrend then reduces
            exactly to fitting the dataset as a whole.
        """
        times = np.asarray(times, dtype=float).ravel()
        if times.size == 0:
            return []
        if gap_threshold_sec is None:
            gap_threshold_sec = float(constants.PASS_GAP_DEFAULT.values)
        if min_points is None:
            min_points = constants.PASS_MIN_POINTS_DEFAULT
        if keep is None:
            keep = np.ones(times.size, dtype=bool)
        else:
            keep = np.asarray(keep, dtype=bool).ravel()

        order = np.argsort(times, kind="stable")
        breaks = np.flatnonzero(np.diff(times[order]) > float(gap_threshold_sec)) + 1
        segments = [seg for seg in np.split(order, breaks) if seg.size]

        # Merge any segment with too few fittable points into its neighbour, so
        # no segment is detrended on a sample too small to mean anything.
        merged = []
        for segment in segments:
            if merged and int(keep[segment].sum()) < min_points:
                merged[-1] = np.concatenate([merged[-1], segment])
            else:
                merged.append(segment)
        # The first segment can only merge forwards.
        while len(merged) > 1 and int(keep[merged[0]].sum()) < min_points:
            merged[0] = np.concatenate([merged[0], merged[1]])
            del merged[1]

        return merged

    def _huber_mad_apply(self, name, new_flags, n_sigma, sigma_new):
        """
        Merge Huber/MAD flags into the working set (OR) and report the count.

        Parameters
        ----------
        name : str
            Dataset name.
        new_flags : numpy.ndarray of bool
            Newly computed rejection mask.
        n_sigma : float
            Threshold width, for the printed message only.
        sigma_new : float or None
            Re-estimated sigma to record, if any.
        """
        self._outlier_flags[name] = self._outlier_flags[name] | new_flags
        if sigma_new is not None:
            self._refit_sigmas[name] = sigma_new
        n = int(new_flags.sum())
        print(
            f"[MeasurementEditing] Huber/MAD ({n_sigma}x): "
            f"{n} new outliers flagged in '{name}'."
        )
        
    # -----------------------------------
    #     Chi-Squared Auto Filter
    # -----------------------------------

    def chi_squared_filter(
        self,
        dataset_name: str = None,
        sigma_scale: float = 3.0,
        dof: int = 1,
        preview: bool = True,
    ):
        """
        Automatically flag measurements whose normalised residuals exceed a
        chi-squared threshold.

        Criterion: (residual / sigma)^2 > chi2.ppf(1 - alpha, dof)
        where alpha = 1 - CDF(sigma_scale^2, dof=1), equivalent to N-sigma.

        Parameters
        ----------
        dataset_name : str, list of str, or None
            Name(s) of the dataset(s) to filter. If None, applies to all datasets.
        sigma_scale : float
            Equivalent Gaussian sigma multiplier (e.g. 3.0 for 3-sigma rejection).
        dof : int
            Degrees of freedom for chi-squared distribution (default 1).
        preview : bool
            If True, show a summary plot before applying flags.
        """
        names = self._resolve_names(dataset_name) if dataset_name else list(self._outlier_flags.keys())
        threshold = chi2.ppf(chi2.cdf(sigma_scale**2, df=1), df=dof)

        for name in names:
            residuals, sigma, chi2_vals, new_flags = self._chi2_evaluate(name, threshold)

            if preview:
                self._chi2_preview(
                    name, residuals, sigma, chi2_vals, new_flags, threshold, sigma_scale
                )

            self._chi2_apply(name, new_flags, sigma_scale)

    def chi_squared_filter_all(
        self,
        dataset_name=None,
        sigma_scale: float = 3.0,
        dof: int = 1,
        preview: bool = True,
    ):
        """
        Chi-squared filtering with every preview figure shown AT ONCE.

        Same rejection criterion and same flag merging as
        :meth:`chi_squared_filter` — the only difference is presentation: all
        preview figures are built first and displayed together by a single
        ``plt.show()``, instead of one blocking figure per dataset.

        Use this instead of calling :meth:`chi_squared_filter` in a per-dataset
        loop. Looping blocks on each dataset's ``plt.show()``, which is clunky
        on remote/browser backends: closing a plot tab there
        does not advance the loop to the next dataset.

        Flags are merged only after the previews are closed, so the previews
        still show the state before any flags are applied. Note the previews
        are informational, not a confirmation prompt — the flags are applied
        either way.

        Parameters
        ----------
        dataset_name : str or list of str or None
            Dataset(s) to filter. ``None`` filters all managed datasets.
        sigma_scale : float
            Equivalent Gaussian sigma multiplier (e.g. 3.0 for 3-sigma rejection).
        dof : int
            Degrees of freedom for chi-squared distribution (default 1).
        preview : bool
            If True, show the summary plots before applying flags.
        """
        names = self._resolve_names(dataset_name)
        threshold = chi2.ppf(chi2.cdf(sigma_scale**2, df=1), df=dof)

        results = [(name, *self._chi2_evaluate(name, threshold)) for name in names]

        if preview:
            # Hold references so the figures stay alive for the single blocking
            # plt.show() below.
            self._open_chi2_figs = [
                self._build_chi2_figure(
                    name, residuals, sigma, chi2_vals, new_flags, threshold, sigma_scale
                )
                for name, residuals, sigma, chi2_vals, new_flags in results
            ]
            plt.show()

        for name, _, _, _, new_flags in results:
            self._chi2_apply(name, new_flags, sigma_scale)

    def _chi2_evaluate(self, name, threshold):
        """
        Compute the chi-squared values and resulting flags for one dataset.

        Parameters
        ----------
        name : str
            Dataset name.
        threshold : float
            Chi-squared rejection threshold.

        Returns
        -------
        residuals, sigma, chi2_vals, new_flags : tuple of numpy.ndarray
            The residuals, sigmas, ``(residual / sigma)^2`` values, and the
            boolean rejection mask (True = reject).
        """
        dset = self._get_dataset(name)
        residuals = np.array(dset.data["residuals"]).ravel()
        sigma = np.array(dset.data["sigma"]).ravel()

        # Guard against zero sigma
        safe_sigma = np.where(sigma == 0, np.nan, sigma)
        chi2_vals = (residuals / safe_sigma) ** 2
        return residuals, sigma, chi2_vals, chi2_vals > threshold

    def _chi2_apply(self, name, new_flags, sigma_scale):
        """
        Merge chi-squared flags into the working set (OR) and report the count.

        Parameters
        ----------
        name : str
            Dataset name.
        new_flags : numpy.ndarray of bool
            Newly computed rejection mask.
        sigma_scale : float
            Sigma multiplier, for the printed message only.
        """
        self._outlier_flags[name] = self._outlier_flags[name] | new_flags
        n = int(new_flags.sum())
        print(
            f"[MeasurementEditing] Chi-squared ({sigma_scale}σ): "
            f"{n} new outliers flagged in '{name}'."
        )

    def _chi2_preview(
        self, name, residuals, sigma, chi2_vals, new_flags, threshold, sigma_scale
    ):
        """
        Display a two-panel summary plot before applying chi-squared flags.

        The left panel shows the residual time series with newly flagged points
        highlighted in red.  The right panel shows a histogram of all chi-squared
        values with a vertical line marking the rejection threshold.

        Parameters
        ----------
        name : str
            Dataset name for the plot title.
        residuals : np.ndarray
            1-D array of measurement residuals.
        sigma : np.ndarray
            1-D array of measurement standard deviations.
        chi2_vals : np.ndarray
            Pre-computed ``(residuals / sigma)^2`` values.
        new_flags : np.ndarray of bool
            Boolean mask of measurements that would be flagged (True = reject).
        threshold : float
            Chi-squared rejection threshold corresponding to ``sigma_scale``.
        sigma_scale : float
            Equivalent Gaussian sigma multiplier used to derive ``threshold``.
        """
        self._build_chi2_figure(
            name, residuals, sigma, chi2_vals, new_flags, threshold, sigma_scale
        )
        plt.show()

    def _build_chi2_figure(
        self, name, residuals, sigma, chi2_vals, new_flags, threshold, sigma_scale
    ):
        """
        Build (but do NOT display) the two-panel chi-squared summary figure.

        The left panel shows the residual time series with newly flagged points
        highlighted in red.  The right panel shows a histogram of all chi-squared
        values with a vertical line marking the rejection threshold.

        Returns the Matplotlib figure; the caller is responsible for calling
        ``plt.show()``. See :meth:`_chi2_preview` for the parameters.
        """
        plt.rcParams.update(_RCPARAMS)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"Chi-Squared Filter Preview — {name}  ({sigma_scale}σ, threshold={threshold:.2f})"
        )

        t = np.arange(len(residuals))

        # Left: residuals with outliers highlighted
        ax0 = axes[0]
        ax0.scatter(
            t[~new_flags],
            residuals[~new_flags],
            s=8,
            c=_C_ACTIVE,
            alpha=0.7,
            label="OK",
        )
        ax0.scatter(
            t[new_flags],
            residuals[new_flags],
            s=12,
            c=_C_OUTLIER,
            alpha=0.9,
            marker="x",
            label="Flagged",
        )
        ax0.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax0.set_xlabel("Index")
        ax0.set_ylabel("Residual")
        ax0.set_title("Residuals")
        ax0.legend(fontsize=9)

        # Right: chi2 histogram
        ax1 = axes[1]
        finite = chi2_vals[np.isfinite(chi2_vals)]
        ax1.hist(finite, bins=40, color=_C_ACTIVE, alpha=0.7, edgecolor="white")
        ax1.axvline(
            threshold,
            color=_C_OUTLIER,
            linewidth=1.5,
            linestyle="--",
            label=f"Threshold ({sigma_scale}σ)",
        )
        ax1.set_xlabel("χ² value")
        ax1.set_ylabel("Count")
        ax1.set_title("Chi-Squared Distribution")
        ax1.legend(fontsize=9)

        fig.tight_layout()
        return fig

    # -----------------------------------
    #     Kurtosis Auto Filter
    # -----------------------------------
    class InvalidKurtosisSettingsError(Exception):
        """Raised when the kurtosis filter receives an unusable setting."""

    def kurtosis_filter(
        self,
        dataset_name: str = None,
        target_kurtosis: float = 3.0,
        max_fraction: float = 0.25,
        preview: bool = True,
    ):
        """
        Trim the most extreme normalised residuals until the sample kurtosis of
        the remaining points drops below ``target_kurtosis``.

        A Gaussian residual population has kurtosis 3 (Pearson definition,
        i.e. zero excess kurtosis); anything above that is heavy-tailed, which
        for tracking residuals means outliers rather than genuine noise. Unlike
        :meth:`chi_squared_filter`, no absolute sigma threshold is assumed —
        the data decides how much of the tail has to go.

        Procedure: normalised residuals ``z = residual / sigma`` are ordered by
        ``|z - median(z)|`` (most extreme first) and removed one at a time; the
        kurtosis of the surviving points is evaluated after every removal, and
        the smallest number of removals that brings it to ``target_kurtosis``
        is the one applied. Only points that are still active are considered —
        already-flagged points would inflate the kurtosis and drive
        over-trimming — and points with zero or non-finite sigma are ignored
        (never flagged), as in :meth:`chi_squared_filter`.

        Parameters
        ----------
        dataset_name : str, list of str, or None
            Name(s) of the dataset(s) to filter. If None, applies to all datasets.
        target_kurtosis : float
            Kurtosis to trim down to (default 3.0, the Gaussian value). Must be
            at least 1.0, the smallest kurtosis any distribution can have.
        max_fraction : float
            Safety cap on the fraction of active points that may be removed
            (default 0.25). If the target is not reached within the cap, the
            capped number of points is flagged and a warning is printed.
        preview : bool
            If True, show a summary plot before applying flags.

        Raises
        ------
        InvalidKurtosisSettingsError
            If ``target_kurtosis < 1`` or ``max_fraction`` is outside [0, 1).

        Notes
        -----
        Datasets with fewer than four active points are left untouched: the
        fourth moment is not meaningful there.
        """
        self._check_kurtosis_settings(target_kurtosis, max_fraction)
        names = self._resolve_names(dataset_name) if dataset_name else list(self._outlier_flags.keys())

        for name in names:
            residuals, z, curve, n_remove, new_flags = self._kurtosis_evaluate(
                name, target_kurtosis, max_fraction
            )

            if preview:
                self._kurtosis_preview(
                    name, residuals, z, curve, n_remove, new_flags, target_kurtosis
                )

            self._kurtosis_apply(name, new_flags, curve, n_remove, target_kurtosis)

    def kurtosis_filter_all(
        self,
        dataset_name=None,
        target_kurtosis: float = 3.0,
        max_fraction: float = 0.25,
        preview: bool = True,
    ):
        """
        Kurtosis filtering with every preview figure shown AT ONCE.

        Same trimming criterion and same flag merging as
        :meth:`kurtosis_filter` — the only difference is presentation: all
        preview figures are built first and displayed together by a single
        ``plt.show()``, instead of one blocking figure per dataset.

        Use this instead of calling :meth:`kurtosis_filter` in a per-dataset
        loop. Looping blocks on each dataset's ``plt.show()``, which is clunky
        on remote/browser backends: closing a plot tab there
        does not advance the loop to the next dataset.

        Flags are merged only after the previews are closed, so the previews
        still show the state before any flags are applied. Note the previews
        are informational, not a confirmation prompt — the flags are applied
        either way.

        Parameters
        ----------
        dataset_name : str or list of str or None
            Dataset(s) to filter. ``None`` filters all managed datasets.
        target_kurtosis : float
            Kurtosis to trim down to (default 3.0, the Gaussian value).
        max_fraction : float
            Safety cap on the fraction of active points that may be removed.
        preview : bool
            If True, show the summary plots before applying flags.
        """
        self._check_kurtosis_settings(target_kurtosis, max_fraction)
        names = self._resolve_names(dataset_name)

        results = [
            (name, *self._kurtosis_evaluate(name, target_kurtosis, max_fraction))
            for name in names
        ]

        if preview:
            # Hold references so the figures stay alive for the single blocking
            # plt.show() below.
            self._open_kurtosis_figs = [
                self._build_kurtosis_figure(
                    name, residuals, z, curve, n_remove, new_flags, target_kurtosis
                )
                for name, residuals, z, curve, n_remove, new_flags in results
            ]
            plt.show()

        for name, _, _, curve, n_remove, new_flags in results:
            self._kurtosis_apply(name, new_flags, curve, n_remove, target_kurtosis)

    def _check_kurtosis_settings(self, target_kurtosis, max_fraction):
        """
        Validate the kurtosis-filter settings shared by both entry points.

        Parameters
        ----------
        target_kurtosis : float
            Requested kurtosis target (>= 1.0).
        max_fraction : float
            Requested removal cap, in [0, 1).

        Raises
        ------
        InvalidKurtosisSettingsError
            If either setting is outside its admissible range.
        """
        if target_kurtosis < 1.0:
            raise self.InvalidKurtosisSettingsError(
                f"target_kurtosis must be >= 1.0 (no distribution has a smaller "
                f"kurtosis), got {target_kurtosis}."
            )
        if not 0.0 <= max_fraction < 1.0:
            raise self.InvalidKurtosisSettingsError(
                f"max_fraction must be in [0, 1), got {max_fraction}."
            )

    def _kurtosis_evaluate(self, name, target_kurtosis, max_fraction):
        """
        Compute the kurtosis trimming curve and resulting flags for one dataset.

        Parameters
        ----------
        name : str
            Dataset name.
        target_kurtosis : float
            Kurtosis to trim down to.
        max_fraction : float
            Cap on the fraction of active points that may be removed.

        Returns
        -------
        residuals : numpy.ndarray
            The dataset residuals.
        z : numpy.ndarray
            Normalised residuals ``residual / sigma`` (NaN where sigma is 0).
        curve : numpy.ndarray
            Kurtosis of the surviving points after removing ``k`` of them, for
            ``k = 0 ... k_max``; empty if the dataset is too small to trim.
        n_remove : int
            Number of removals selected (index into ``curve``).
        new_flags : numpy.ndarray of bool
            Boolean rejection mask (True = reject).
        """
        dset = self._get_dataset(name)
        residuals = np.array(dset.data["residuals"]).ravel()
        sigma = np.array(dset.data["sigma"]).ravel()

        # Guard against zero sigma (same convention as the chi-squared filter:
        # such points are excluded from the statistic and never flagged).
        safe_sigma = np.where(sigma == 0, np.nan, sigma)
        z = residuals / safe_sigma

        new_flags = np.zeros(len(residuals), dtype=bool)
        candidates = np.flatnonzero(~self._outlier_flags[name] & np.isfinite(z))
        n = candidates.size
        if n < 4:
            # The fourth moment is meaningless on fewer than four points.
            return residuals, z, np.empty(0), 0, new_flags

        # Removal order: most extreme deviation from the median first. The
        # median (not the mean) keeps a couple of wild points from dragging the
        # centre out and mis-ordering the rest of the tail.
        zc = z[candidates] - np.median(z[candidates])
        order = np.argsort(-np.abs(zc))
        zs = zc[order]

        k_max = max(min(int(max_fraction * n), n - 4), 0)

        # Moments of every surviving set at once: total sums minus the running
        # sums of the points removed so far.
        head = np.arange(k_max + 1)
        m = (n - head).astype(float)
        removed = [np.concatenate(([0.0], np.cumsum(zs**p))) for p in (1, 2, 3, 4)]
        s1, s2, s3, s4 = (r[-1] - r[head] for r in removed)

        mean = s1 / m
        mu2 = s2 / m - mean**2
        mu4 = s4 / m - 4 * mean * s3 / m + 6 * mean**2 * s2 / m - 3 * mean**4
        with np.errstate(divide="ignore", invalid="ignore"):
            curve = np.where(mu2 > 0, mu4 / mu2**2, np.inf)

        # Fewest removals that bring the survivors to the target; the cap if
        # the target is never reached.
        below = np.flatnonzero(curve <= target_kurtosis)
        n_remove = int(below[0]) if below.size else k_max

        new_flags[candidates[order[:n_remove]]] = True
        return residuals, z, curve, n_remove, new_flags

    def _kurtosis_apply(self, name, new_flags, curve, n_remove, target_kurtosis):
        """
        Merge kurtosis flags into the working set (OR) and report the result.

        Parameters
        ----------
        name : str
            Dataset name.
        new_flags : numpy.ndarray of bool
            Newly computed rejection mask.
        curve : numpy.ndarray
            Kurtosis-vs-removals curve from :meth:`_kurtosis_evaluate`.
        n_remove : int
            Number of removals selected.
        target_kurtosis : float
            Kurtosis target, for the printed message and the warning check.
        """
        self._outlier_flags[name] = self._outlier_flags[name] | new_flags
        n = int(new_flags.sum())

        if curve.size == 0:
            print(
                f"[MeasurementEditing] Kurtosis (target {target_kurtosis}): "
                f"'{name}' has fewer than 4 active points, nothing flagged."
            )
            return

        k_start, k_end = curve[0], curve[n_remove]
        print(
            f"[MeasurementEditing] Kurtosis (target {target_kurtosis}): "
            f"{n} new outliers flagged in '{name}' "
            f"(kurtosis {k_start:.2f} → {k_end:.2f})."
        )
        if k_end > target_kurtosis:
            print(
                f"[MeasurementEditing] WARNING: '{name}' is still at kurtosis "
                f"{k_end:.2f} after removing the {n}-point cap — raise "
                f"max_fraction to keep trimming."
            )

    def _kurtosis_preview(
        self, name, residuals, z, curve, n_remove, new_flags, target_kurtosis
    ):
        """
        Display a three-panel summary plot before applying kurtosis flags.

        Panels: the residual time series with newly flagged points highlighted,
        the kurtosis-vs-removals curve with the target and the selected number
        of removals marked, and the normalised-residual histogram before and
        after trimming.

        Parameters
        ----------
        name : str
            Dataset name for the plot title.
        residuals : np.ndarray
            1-D array of measurement residuals.
        z : np.ndarray
            Normalised residuals ``residual / sigma``.
        curve : np.ndarray
            Kurtosis of the surviving points after ``k`` removals.
        n_remove : int
            Number of removals selected.
        new_flags : np.ndarray of bool
            Boolean mask of measurements that would be flagged (True = reject).
        target_kurtosis : float
            Kurtosis target used to pick ``n_remove``.
        """
        self._build_kurtosis_figure(
            name, residuals, z, curve, n_remove, new_flags, target_kurtosis
        )
        plt.show()

    def _build_kurtosis_figure(
        self, name, residuals, z, curve, n_remove, new_flags, target_kurtosis
    ):
        """
        Build (but do NOT display) the three-panel kurtosis summary figure.

        Panels: the residual time series with newly flagged points highlighted,
        the kurtosis-vs-removals curve with the target and the selected number
        of removals marked, and the normalised-residual histogram before and
        after trimming.

        Returns the Matplotlib figure; the caller is responsible for calling
        ``plt.show()``. See :meth:`_kurtosis_preview` for the parameters.
        """
        k_start = curve[0] if curve.size else np.nan
        k_end = curve[n_remove] if curve.size else np.nan

        plt.rcParams.update(_RCPARAMS)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Kurtosis Filter Preview — {name}  "
            f"(target={target_kurtosis:.2f}, {k_start:.2f} → {k_end:.2f})"
        )

        t = np.arange(len(residuals))

        # Left: residuals with outliers highlighted
        ax0 = axes[0]
        ax0.scatter(
            t[~new_flags],
            residuals[~new_flags],
            s=8,
            c=_C_ACTIVE,
            alpha=0.7,
            label="OK",
        )
        ax0.scatter(
            t[new_flags],
            residuals[new_flags],
            s=12,
            c=_C_OUTLIER,
            alpha=0.9,
            marker="x",
            label="Flagged",
        )
        ax0.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax0.set_xlabel("Index")
        ax0.set_ylabel("Residual")
        ax0.set_title("Residuals")
        ax0.legend(fontsize=9)

        # Middle: kurtosis as points are trimmed away
        ax1 = axes[1]
        ax1.plot(np.arange(curve.size), curve, color=_C_ACTIVE, linewidth=1.5)
        ax1.axhline(
            target_kurtosis,
            color=_C_OUTLIER,
            linewidth=1.5,
            linestyle="--",
            label=f"Target ({target_kurtosis:.2f})",
        )
        ax1.axvline(
            n_remove,
            color="gray",
            linewidth=1.2,
            linestyle=":",
            label=f"Removed ({n_remove})",
        )
        ax1.set_xlabel("Points removed")
        ax1.set_ylabel("Kurtosis")
        ax1.set_title("Kurtosis vs. Trimming")
        ax1.legend(fontsize=9)

        # Right: normalised residuals before vs after trimming
        ax2 = axes[2]
        finite = z[np.isfinite(z)]
        kept = z[np.isfinite(z) & ~new_flags]
        if finite.size:
            bins = np.histogram_bin_edges(finite, bins=40)
            ax2.hist(
                finite, bins=bins, color=_C_OUTLIER, alpha=0.5, label="Before"
            )
            ax2.hist(
                kept, bins=bins, color=_C_ACTIVE, alpha=0.8, label="After"
            )
        ax2.set_xlabel("Residual / σ")
        ax2.set_ylabel("Count")
        ax2.set_title("Normalised Residuals")
        ax2.legend(fontsize=9)

        fig.tight_layout()
        return fig

    # -----------------------------------
    #     Date Range Filter
    # -----------------------------------
    def date_range_filter(
        self,
        t_start,
        t_end,
        dataset_name: str = None,
        action: str = "flag",
        time_field: str = "t2",
    ):
        """
        Flag or unflag all measurements whose time falls within [t_start, t_end].

        Parameters
        ----------
        t_start : float or str
            Start of the window. Either a numeric epoch (seconds) or ISO 8601 string
            (e.g. "2024-01-01T00:00:00").
        t_end : float or str
            End of the window (same format as t_start).
        dataset_name : str, list of str, or None
            Dataset(s) to apply to. ``None`` applies to all datasets.
        action : str
            ``"flag"`` marks measurements in the window as outliers;
            ``"unflag"`` removes the outlier flag from them.
        time_field : str
            Time field to use for comparison (default: 't2').
        """
        t_start = self._to_epoch(t_start)
        t_end = self._to_epoch(t_end)
        names = self._resolve_names(dataset_name) if dataset_name else list(self._outlier_flags.keys())

        for name in names:
            dset = self._get_dataset(name)
            times = np.array(dset.data[time_field]).ravel()
            in_win = (times >= t_start) & (times <= t_end)
            n = int(in_win.sum())

            if action == "flag":
                self._outlier_flags[name][in_win] = True
                print(
                    f"[MeasurementEditing] Date range: {n} measurements flagged "
                    f"in '{name}' between {t_start} and {t_end}."
                )
            elif action == "unflag":
                self._outlier_flags[name][in_win] = False
                print(
                    f"[MeasurementEditing] Date range: {n} measurements unflagged "
                    f"in '{name}' between {t_start} and {t_end}."
                )
            else:
                raise ValueError(f"action must be 'flag' or 'unflag', got '{action}'.")

    def interactive_date_range(self, dataset_name: str = None):
        """
        Open an interactive GUI for manual date-range flagging and unflagging.

        Displays a residuals scatter plot for the first targeted dataset and
        provides two text boxes (``t_start``, ``t_end``) and three buttons:

        * **Flag**   — apply :meth:`date_range_filter` with ``action='flag'`` to
          all target datasets for the entered epoch range.
        * **Unflag** — apply :meth:`date_range_filter` with ``action='unflag'``.
        * **Done**   — close the figure and return.

        Both text boxes accept numeric epoch strings (seconds, e.g.
        ``"705427200.0"``) or ISO 8601 date strings (e.g.
        ``"2024-01-01T00:00:00"``).

        The plot updates immediately after each Flag/Unflag action to reflect
        the current outlier state of the first dataset.

        Parameters
        ----------
        dataset_name : str, list of str, or None
            Dataset(s) to target.  If None, all managed datasets are targeted
            (the plot preview always shows the first one).
        """
        names = self._resolve_names(dataset_name)

        plt.rcParams.update(_RCPARAMS)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.subplots_adjust(bottom=0.42)

        # Preview of first dataset
        target_name = names[0]
        dset = self._get_dataset(target_name)
        t = np.array(dset.data["t2"]).ravel()
        res = np.array(dset.data["residuals"]).ravel()

        sc_ok = ax.scatter(t, res, s=10, c=_C_ACTIVE, alpha=0.7, label="Active")
        sc_out = ax.scatter(
            [], [], s=10, c=_C_OUTLIER, alpha=0.8, marker="x", label="Rejected"
        )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("t2 (epoch)")
        ax.set_ylabel("Residual")
        ax.set_title(f"Date Range Editor — {target_name}")
        ax.legend(fontsize=9)

        def _refresh():
            flags = self._outlier_flags[target_name]
            sc_ok.set_offsets(np.column_stack([t[~flags], res[~flags]]))
            sc_out.set_offsets(
                np.column_stack([t[flags], res[flags]])
                if flags.any()
                else np.empty((0, 2))
            )
            ax.set_title(
                f"Date Range Editor — {target_name}  |  {flags.sum()} outliers"
            )
            fig.canvas.draw_idle()

        # TextBoxes for start/end
        ax_t0 = plt.axes([0.12, 0.25, 0.35, 0.06])
        ax_t1 = plt.axes([0.55, 0.25, 0.35, 0.06])
        tb_t0 = TextBox(ax_t0, "t_start", color="#f5f5f5", hovercolor="#eeeeee")
        tb_t1 = TextBox(ax_t1, "t_end", color="#f5f5f5", hovercolor="#eeeeee")

        # Buttons
        ax_flag = plt.axes([0.12, 0.10, 0.18, 0.07])
        ax_unflag = plt.axes([0.34, 0.10, 0.18, 0.07])
        ax_close = plt.axes([0.72, 0.10, 0.18, 0.07])
        btn_flag = Button(ax_flag, "Flag", color="#ffcdd2", hovercolor="#ef9a9a")
        btn_unflag = Button(ax_unflag, "Unflag", color="#c8e6c9", hovercolor="#a5d6a7")
        btn_close = Button(ax_close, "Done", color="#bbdefb", hovercolor="#90caf9")

        def _apply(action):
            try:
                t0 = self._to_epoch(tb_t0.text.strip())
                t1 = self._to_epoch(tb_t1.text.strip())
            except Exception as e:
                print(f"[MeasurementEditing] Bad date input: {e}")
                return
            for nm in names:
                self.date_range_filter(t0, t1, dataset_name=nm, action=action)
            _refresh()

        btn_flag.on_clicked(lambda _: _apply("flag"))
        btn_unflag.on_clicked(lambda _: _apply("unflag"))
        btn_close.on_clicked(lambda _: plt.close(fig))

        _refresh()
        plt.show()

    # -----------------------------------
    #     Isolation Forest
    # -----------------------------------
    class InvalidIsolationForestSettingsError(ValueError):
        """Raised when the isolation-forest settings cannot produce a fit."""

    @staticmethod
    def _robust_standardize(points):
        """
        Centre each coordinate on its median and scale it by its MAD.

        Residual components rarely share a dispersion — right ascension and
        declination need not, and a detector's two axes need not — so without
        this the forest would isolate on whichever coordinate happens to be
        noisier rather than on genuine outliers.

        Parameters
        ----------
        points : numpy.ndarray
            ``(n, d)`` residual coordinates.

        Returns
        -------
        numpy.ndarray
            The same array, robustly standardised column by column.
        """
        centre = np.median(points, axis=0)
        # 1.4826 makes the MAD comparable to sigma for Gaussian data. A
        # constant coordinate is left unscaled rather than dividing by zero.
        scale = 1.4826 * np.median(np.abs(points - centre), axis=0)
        scale = np.where(scale <= np.finfo(float).eps, 1.0, scale)
        return (points - centre) / scale

    @staticmethod
    def _kneedle_threshold(scores):
        """
        Locate the knee of monotonically ranked anomaly scores.

        Scores are sorted least- to most-anomalous and normalised onto the unit
        square; for the resulting increasing curve the knee is the point of
        maximum vertical drop below the endpoint chord, ``max(x - y)``. This is
        what lets the data set the rejection threshold instead of the caller
        having to assume a contamination fraction.

        Parameters
        ----------
        scores : array_like
            Anomaly scores, larger meaning more anomalous.

        Returns
        -------
        threshold : float
            Score above which a point is an outlier (strictly greater).
        knee_rank : int
            Zero-based rank of the knee in sorted order.

        Raises
        ------
        InvalidIsolationForestSettingsError
            If fewer than three finite scores are supplied.
        """
        values = np.asarray(scores, dtype=float).ravel()
        if values.size < 3 or not np.all(np.isfinite(values)):
            raise MeasurementEditing.InvalidIsolationForestSettingsError(
                "locating a knee needs at least three finite anomaly scores, "
                f"got {values.size} ({np.count_nonzero(np.isfinite(values))} "
                "finite)."
            )

        ranked = np.sort(values)
        span = ranked[-1] - ranked[0]
        if span <= np.finfo(float).eps:
            # Every point is equally anomalous: there is no tail to cut.
            return float(ranked[-1]), values.size - 1

        x = np.linspace(0.0, 1.0, values.size)
        y = (ranked - ranked[0]) / span
        # Endpoints cannot be interior knees.
        knee = int(np.argmax((x - y)[1:-1]) + 1)
        return float(ranked[knee]), knee

    @staticmethod
    def _component_stem(dset):
        """
        Split a dataset name into the observable it belongs to and its component.

        A two-component observable is stored as two scalar datasets named
        ``<stem>_<label>``, with the label repeated at the end of
        ``measurement_type`` (``"AngularIdeal ra"``). Recovering the stem is
        what lets the forest see one image as a single 2-D point instead of two
        unrelated scalars.

        Returns
        -------
        tuple
            ``(stem, label)``; ``label`` is ``None`` for a scalar observable.
        """
        tokens = str(getattr(dset, "measurement_type", "") or "").split()
        name = dset.set_name
        if len(tokens) >= 2 and name.endswith(f"_{tokens[-1]}"):
            return name[: -(len(tokens[-1]) + 1)], tokens[-1]
        return name, None

    def _isolation_forest_groups(self, names):
        """
        Group dataset names by the observable they are components of.

        Parameters
        ----------
        names : list of str
            Dataset names to consider.

        Returns
        -------
        list of list of str
            One entry per observable, in the order the datasets were given.
        """
        groups, order = {}, []
        for name in names:
            stem, _ = self._component_stem(self._get_dataset(name))
            if stem not in groups:
                groups[stem] = []
                order.append(stem)
            groups[stem].append(name)
        return [groups[stem] for stem in order]

    def _isolation_forest_evaluate(
        self, names, standardize, scale_by_cos_dec, n_estimators, random_state
    ):
        """
        Score one observable's residuals and pick the rejection threshold.

        Only points that are still active are scored: an already-flagged
        outlier would stretch the score range and move the knee. Points whose
        residuals are not finite are rejected outright without reaching the
        forest.

        Returns
        -------
        points : numpy.ndarray
            ``(n, d)`` residual coordinates, before standardisation.
        scores : numpy.ndarray
            Anomaly score per point, ``+inf`` where not finite, NaN where the
            point was not scored (already flagged).
        threshold : float or None
            Knee threshold, or None when the observable was left untouched.
        new_flags : numpy.ndarray of bool
            Points this filter would newly flag.
        """
        dsets = [self._get_dataset(name) for name in names]
        columns = [np.asarray(d.data["residuals"], dtype=float).ravel() for d in dsets]
        points = np.column_stack(columns)

        # A vector observable is one measurement: flag its components together.
        active = np.ones(points.shape[0], dtype=bool)
        for name in names:
            active &= ~self._outlier_flags[name]

        if scale_by_cos_dec and len(names) == 2:
            labels = [self._component_stem(d)[1] for d in dsets]
            if labels == ["ra", "dec"] or labels == ["dec", "ra"]:
                ra_col = labels.index("ra")
                dec_col = 1 - ra_col
                dec_obs = np.asarray(
                    dsets[dec_col].data["observed"], dtype=float
                ).ravel()
                points[:, ra_col] = points[:, ra_col] * np.cos(dec_obs)

        finite = np.all(np.isfinite(points), axis=1)
        scores = np.full(points.shape[0], np.nan, dtype=float)
        scores[active & ~finite] = np.inf
        new_flags = active & ~finite

        candidates = np.flatnonzero(active & finite)
        if candidates.size < constants.ISOLATION_FOREST_MIN_POINTS:
            return points, scores, None, new_flags

        fitted = points[candidates]
        if standardize:
            fitted = self._robust_standardize(fitted)

        forest = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        )
        forest.fit(fitted)
        # score_samples is larger for ordinary points, so negate it: larger
        # then consistently means "more anomalous".
        candidate_scores = -forest.score_samples(fitted)
        scores[candidates] = candidate_scores

        threshold, _ = self._kneedle_threshold(candidate_scores)
        new_flags = new_flags | np.isin(
            np.arange(points.shape[0]), candidates[candidate_scores > threshold]
        )
        return points, scores, threshold, new_flags

    def _isolation_forest_apply(self, names, new_flags, threshold):
        """Merge the new flags into every component of the observable."""
        n_new = int(np.count_nonzero(new_flags))
        for name in names:
            self._outlier_flags[name] = self._outlier_flags[name] | new_flags

        label = " + ".join(names)
        if threshold is None:
            print(
                f"[MeasurementEditing] {label}: fewer than "
                f"{constants.ISOLATION_FOREST_MIN_POINTS} active points, left "
                "untouched."
            )
        else:
            print(
                f"[MeasurementEditing] {label}: flagged {n_new} point(s) "
                f"above the knee threshold {threshold:.4f}."
            )

    def _build_isolation_forest_figure(
        self, names, points, scores, threshold, new_flags
    ):
        """
        Preview figure: the residuals themselves, and the ranked scores.

        The left panel shows what is being cut in the space the operator thinks
        in; the right shows *why* — where the knee fell on the ranked scores.
        """
        with plt.rc_context(_RCPARAMS):
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            label = " + ".join(names)

            active = ~np.isnan(scores) | np.isinf(scores)
            keep = active & ~new_flags
            cut = new_flags

            if points.shape[1] >= 2:
                axes[0].scatter(points[keep, 0], points[keep, 1], s=14,
                                color=_C_ACTIVE, label="kept")
                axes[0].scatter(points[cut, 0], points[cut, 1], s=36,
                                color=_C_OUTLIER, marker="x", label="flagged")
                axes[0].set_xlabel(f"{names[0]} residual")
                axes[0].set_ylabel(f"{names[1]} residual")
            else:
                index = np.arange(points.shape[0])
                axes[0].scatter(index[keep], points[keep, 0], s=14,
                                color=_C_ACTIVE, label="kept")
                axes[0].scatter(index[cut], points[cut, 0], s=36,
                                color=_C_OUTLIER, marker="x", label="flagged")
                axes[0].set_xlabel("observation index")
                axes[0].set_ylabel(f"{names[0]} residual")
            axes[0].set_title(f"{label} — residuals")
            axes[0].legend(loc="best", fontsize=10)

            scored = scores[np.isfinite(scores)]
            if scored.size:
                ranked = np.sort(scored)
                axes[1].plot(np.arange(ranked.size), ranked, color=_C_ACTIVE,
                             lw=1.4, label="ranked anomaly score")
                if threshold is not None:
                    axes[1].axhline(threshold, color=_C_OUTLIER, ls="--", lw=1.5,
                                    label=f"knee = {threshold:.4f}")
            axes[1].set_xlabel("rank")
            axes[1].set_ylabel("anomaly score")
            axes[1].set_title(f"{label} — knee of the ranked scores")
            axes[1].legend(loc="best", fontsize=10)

            fig.tight_layout()
        return fig

    def isolation_forest_filter(
        self,
        dataset_name=None,
        standardize: bool = True,
        scale_by_cos_dec: bool = False,
        n_estimators: int = 500,
        random_state: int = 0,
        preview: bool = True,
    ):
        """
        Flag anomalous residuals with an isolation forest, cutting at the knee
        of the ranked anomaly scores.

        An isolation forest scores a point by how few random splits it takes to
        separate it from the rest: outliers isolate quickly. Unlike
        :meth:`chi_squared_filter` no sigma threshold is assumed, and unlike
        :meth:`kurtosis_filter` no distribution shape is assumed — but the
        forest still has to be told *how much* to reject, and rather than fixing
        a contamination fraction the threshold is taken from the **knee** of the
        ranked scores (maximum drop below the endpoint chord), so the data
        decides where the tail begins.

        A two-component observable is scored **jointly**: the two scalar
        datasets an optical track is split into are recombined into one 2-D
        point per observation, and a rejection flags both components — a bad
        image is bad in both coordinates, not in half of one.

        Parameters
        ----------
        dataset_name : str or list of str or None
            Dataset(s) to filter. Components of the same observable are grouped
            automatically. If None, every dataset is considered.
        standardize : bool
            Robustly rescale each coordinate by its median and MAD before
            fitting (default True). Without it the forest isolates on whichever
            component happens to be noisier.
        scale_by_cos_dec : bool
            For a right-ascension/declination pair, multiply the RA residual by
            ``cos(dec)`` so the pair is compared on the tangent plane. Ignored
            for any other observable. Defaults to False.
        n_estimators : int
            Number of trees (default 500).
        random_state : int
            Seed, so a rejection is reproducible (default 0).
        preview : bool
            If True, show the residuals and the knee before applying flags.

        Raises
        ------
        InvalidIsolationForestSettingsError
            If ``n_estimators`` is below 1.

        Notes
        -----
        An observable with fewer than eight active points is left untouched —
        a forest cannot say anything useful about a handful of points.

        A statistical label is not proof that a measurement is wrong. Refit and
        recompute residuals after rejecting, and check that the solution is not
        unduly sensitive to what was cut.

        References
        ----------
        .. [1] Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest.
           *Eighth IEEE International Conference on Data Mining*, 413-422.
        .. [2] Satopää, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011). Finding a
           "Kneedle" in a Haystack. *31st International Conference on Distributed
           Computing Systems Workshops*, 166-171.
        """
        if int(n_estimators) < 1:
            raise self.InvalidIsolationForestSettingsError(
                f"n_estimators must be at least 1, got {n_estimators!r}."
            )

        for names in self._isolation_forest_groups(
            self._resolve_names(dataset_name)
        ):
            points, scores, threshold, new_flags = self._isolation_forest_evaluate(
                names, standardize, scale_by_cos_dec, n_estimators, random_state
            )

            if preview:
                self._build_isolation_forest_figure(
                    names, points, scores, threshold, new_flags
                )
                plt.show()

            self._isolation_forest_apply(names, new_flags, threshold)

    def isolation_forest_filter_all(
        self,
        dataset_name=None,
        standardize: bool = True,
        scale_by_cos_dec: bool = False,
        n_estimators: int = 500,
        random_state: int = 0,
        preview: bool = True,
    ):
        """
        Isolation-forest filtering with every preview figure shown AT ONCE.

        Same rejection as :meth:`isolation_forest_filter`; the figures are all
        built first and displayed by a single blocking ``plt.show()`` instead of
        one figure per observable. Flags are merged only after the previews are
        closed, so each preview still shows the state before anything was
        applied.

        Parameters
        ----------
        dataset_name : str or list of str or None
            Dataset(s) to filter. See :meth:`isolation_forest_filter`.
        standardize, scale_by_cos_dec, n_estimators, random_state : optional
            As in :meth:`isolation_forest_filter`.
        preview : bool
            If True, show all preview figures together.

        Raises
        ------
        MeasurementEditing.InvalidIsolationForestSettingsError
            If ``n_estimators`` is less than 1.
        """
        if int(n_estimators) < 1:
            raise self.InvalidIsolationForestSettingsError(
                f"n_estimators must be at least 1, got {n_estimators!r}."
            )

        results = [
            (names, *self._isolation_forest_evaluate(
                names, standardize, scale_by_cos_dec, n_estimators, random_state
            ))
            for names in self._isolation_forest_groups(
                self._resolve_names(dataset_name)
            )
        ]

        if preview:
            # Hold references so the figures stay alive for the single show().
            self._open_isolation_forest_figs = [
                self._build_isolation_forest_figure(
                    names, points, scores, threshold, new_flags
                )
                for names, points, scores, threshold, new_flags in results
            ]
            plt.show()

        for names, _, _, threshold, new_flags in results:
            self._isolation_forest_apply(names, new_flags, threshold)

    # -----------------------------------
    #     Apply & Export
    # -----------------------------------
    def apply_flags(self):
        """
        Write the edited outlier_flag arrays back into each dataset's data dict.
        Call this when satisfied with edits, before passing datasets to the filter.
        """
        for dset in self._datasets:
            dset.data["outlier_flag"] = self._outlier_flags[dset.set_name].copy()
        print("[MeasurementEditing] Outlier flags written back to all datasets.")

    def get_datasets(self):
        """
        Apply flags and return the list of edited datasets.

        Returns
        -------
        list of MeasurementDataSet
        """
        self.apply_flags()
        return self._datasets

    def summary(self):
        """Print a per-dataset summary of flagged vs. total measurements."""
        print("\n── MeasurementEditing Summary ──")
        for name, flags in self._outlier_flags.items():
            total = len(flags)
            flagged = int(flags.sum())
            print(
                f"  {name:<30s}  {flagged:>5d} / {total:>5d} flagged  ({100*flagged/max(total,1):.1f}%)"
            )
        print()

    def reset(self, dataset_name: str = None):
        """
        Clear all outlier flags.

        Parameters
        ----------
        dataset_name : str or None
            If None, resets all datasets.
        """
        names = [dataset_name] if dataset_name else list(self._outlier_flags.keys())
        for name in names:
            self._outlier_flags[name][:] = False
        print(f"[MeasurementEditing] Flags reset for: {names}")

    # -----------------------------------
    #     Internals
    # -----------------------------------
    def _get_dataset(self, name: str):
        """
        Look up a dataset by its ``set_name`` attribute.

        Parameters
        ----------
        name : str
            The ``set_name`` of the desired dataset.

        Returns
        -------
        MeasurementDataSet
            The matching dataset object.

        Raises
        ------
        KeyError
            If no dataset with the given name is found, including the list of
            available names in the message.
        """
        for d in self._datasets:
            if d.set_name == name:
                return d
        raise KeyError(
            f"Dataset '{name}' not found. "
            f"Available: {[d.set_name for d in self._datasets]}"
        )

    @staticmethod
    def _to_epoch(value):
        """
        Convert value to a float epoch (seconds).
        Accepts floats/ints directly, or ISO 8601 strings via datetime.fromisoformat.
        """
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        if isinstance(value, str):
            try:
                # handles '886950000', '8.8695e8', '8.8695E+08' but not '8.8695*1e8'
                return float(value.replace("*", "e").replace("x", "e"))
            except ValueError:
                dt = datetime.fromisoformat(value)
                return dt.timestamp()
        raise TypeError(f"Cannot convert {type(value)} to epoch.")
