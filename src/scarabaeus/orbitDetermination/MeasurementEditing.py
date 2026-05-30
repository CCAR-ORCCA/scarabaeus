# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, Button, TextBox
from matplotlib.path import Path
from matplotlib.patches import Patch
from scipy.stats import chi2
from datetime import datetime

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

    Provides three editing modes:

    1. **Lasso Selector** — graphically select points to flag as outliers on a plot.
    2. **Chi-Squared** — automatically flag points whose residuals exceed a chi-squared threshold.
    3. **Date Range** — manually specify time windows to flag or unflag observations.

    Parameters
    ----------
    datasets : list of MeasurementDataSet
        The datasets to edit. Nested lists are flattened one level.

    Notes
    -----
    Edits are non-destructive: original ``outlier_flag`` arrays are copied
    internally and not written back until :meth:`apply_flags` or
    :meth:`get_datasets` is called.

    See Also
    --------
    scarabaeus.FilterDataManager : Combines datasets and applies editing before indexing.
    scarabaeus.FilterOD : Base filter that consumes the edited datasets.

    Examples
    --------
    .. code-block:: python

        editor = MeasurementEditing(datasets)
        editor.lasso_editor(dataset_name="DSN_Range")
        editor.chi_squared_filter(sigma_scale=3.0)
        editor.date_range_filter("2024-01-01T00:00:00", "2024-01-05T12:00:00")
        edited_datasets = editor.get_datasets()
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

    # -----------------------------------
    #     1. Lasso Selector
    # -----------------------------------
    def lasso_editor(
        self,
        dataset_name: str,
        x_field: str = "t2",
        y_field: str = "residuals",
        title: str = None,
    ):
        """
        Open an interactive matplotlib figure for lasso-based outlier selection.

        Left-click drag  → draw lasso; enclosed points are toggled as outliers.
        Button [Confirm] → apply and close.
        Button [Reset]   → clear all flags for this dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to edit (must match MeasurementDataSet.set_name).
        x_field : str
            Field to use for x-axis (default: 't2').
        y_field : str
            Field to use for y-axis (default: 'residuals').
        title : str, optional
            Plot title override.
        """
        dset = self._get_dataset(dataset_name)
        x = np.array(dset.data[x_field]).ravel()
        y = np.array(dset.data[y_field]).ravel()
        flags = self._outlier_flags[dataset_name]

        plt.rcParams.update(_RCPARAMS)
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.18)

        # Plot
        xy = np.column_stack([x, y])
        scatter = ax.scatter(
            x[~flags], y[~flags], s=14, c=_C_ACTIVE, alpha=0.8, label="Active", zorder=3
        )
        scatter_out = ax.scatter(
            x[flags],
            y[flags],
            s=14,
            c=_C_OUTLIER,
            alpha=0.6,
            marker="x",
            label="Rejected",
            zorder=3,
        )

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(x_field)
        ax.set_ylabel(y_field)
        ax.set_title(title or f"Lasso Editor — {dataset_name}")
        ax.legend(
            handles=[
                Patch(color=_C_ACTIVE, label="Active"),
                Patch(color=_C_OUTLIER, label="Rejected"),
            ],
            fontsize=9,
        )

        selected_indices = set(np.where(flags)[0])

        def _refresh():
            current_flags = np.zeros(len(x), dtype=bool)
            current_flags[list(selected_indices)] = True
            scatter.set_offsets(np.column_stack([x[~current_flags], y[~current_flags]]))
            scatter_out.set_offsets(
                np.column_stack([x[current_flags], y[current_flags]])
                if current_flags.any()
                else np.empty((0, 2))
            )
            ax.set_title(
                f"Lasso Editor — {dataset_name}  |  {current_flags.sum()} outliers flagged"
            )
            fig.canvas.draw_idle()

        def _lasso_callback(verts):
            path = Path(verts)
            inside = path.contains_points(xy)
            toggled = set(np.where(inside)[0])
            # Toggle: if all inside are already flagged → unflag; else flag them
            if toggled.issubset(selected_indices):
                selected_indices.difference_update(toggled)
            else:
                selected_indices.update(toggled)
            _refresh()

        lasso = LassoSelector(ax, _lasso_callback, useblit=True)

        # Buttons
        ax_confirm = plt.axes([0.72, 0.04, 0.12, 0.06])
        ax_reset = plt.axes([0.56, 0.04, 0.12, 0.06])
        btn_confirm = Button(
            ax_confirm, "Confirm", color="#c8e6c9", hovercolor="#a5d6a7"
        )
        btn_reset = Button(ax_reset, "Reset", color="#e8eaf6", hovercolor="#c5cae9")

        def _on_confirm(_):
            new_flags = np.zeros(len(x), dtype=bool)
            new_flags[list(selected_indices)] = True
            self._outlier_flags[dataset_name] = new_flags
            print(
                f"[MeasurementEditing] Lasso: {new_flags.sum()} outliers "
                f"flagged in '{dataset_name}'."
            )
            plt.close(fig)

        def _on_reset(_):
            selected_indices.clear()
            _refresh()

        btn_confirm.on_clicked(_on_confirm)
        btn_reset.on_clicked(_on_reset)

        plt.show()

    # -----------------------------------
    #     2. Chi-Squared Auto Filter
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
        where alpha = 1 - CDF(sigma_scale^2, dof=1)  →  equivalent to N-sigma.

        Parameters
        ----------
        dataset_name : str or None
            Name of the dataset to filter. If None, applies to all datasets.
        sigma_scale : float
            Equivalent Gaussian sigma multiplier (e.g. 3.0 → 3-sigma rejection).
        dof : int
            Degrees of freedom for chi-squared distribution (default 1).
        preview : bool
            If True, show a summary plot before applying flags.
        """
        names = [dataset_name] if dataset_name else list(self._outlier_flags.keys())
        threshold = chi2.ppf(chi2.cdf(sigma_scale**2, df=1), df=dof)

        for name in names:
            dset = self._get_dataset(name)
            residuals = np.array(dset.data["residuals"]).ravel()
            sigma = np.array(dset.data["sigma"]).ravel()

            # Guard against zero sigma
            safe_sigma = np.where(sigma == 0, np.nan, sigma)
            chi2_vals = (residuals / safe_sigma) ** 2
            new_flags = chi2_vals > threshold

            if preview:
                self._chi2_preview(
                    name, residuals, sigma, chi2_vals, new_flags, threshold, sigma_scale
                )

            # Merge with existing flags (OR)
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

        plt.tight_layout()
        plt.show()

    # -----------------------------------
    #     3. Date Range Filter
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
        dataset_name : str or None
            Dataset to apply to. None → all datasets.
        action : str
            "flag"   → mark measurements in window as outliers.
            "unflag" → remove outlier flag from measurements in window.
        time_field : str
            Time field to use for comparison (default: 't2').
        """
        t_start = self._to_epoch(t_start)
        t_end = self._to_epoch(t_end)
        names = [dataset_name] if dataset_name else list(self._outlier_flags.keys())

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
        dataset_name : str or None
            Dataset to target.  If None, all managed datasets are targeted
            (the plot preview always shows the first one).
        """
        names = [dataset_name] if dataset_name else [d.set_name for d in self._datasets]

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
