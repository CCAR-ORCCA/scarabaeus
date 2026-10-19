# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import os
import json

from scarabaeus import (
    constants,
    EpochArray,
    MonteCarloSamples,
    SolutionOD,
    Trajectory,
    Bplane,
    FilterOD,
    Utils,
    ScenarioSetup,
    MediaCorrections,
)

import scarabaeus.utils.NumpyWrapper as np
import pickle
from cycler import cycler

from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import ticker as mtick
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as mticker
from matplotlib.ticker import LogLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from scipy.stats import norm
import corner


from scarabaeus import SolutionOD
from scarabaeus import Trajectory
from scarabaeus import EpochArray
from scarabaeus import Bplane
from scarabaeus import FilterOD
from scarabaeus import Utils
from scarabaeus import ScenarioSetup


# ----------------------#
#   Class Definition   #
# ----------------------#
class Plotting:
    """
    Collection of standardised plotting utilities for Scarabaeus output data.

    Provides methods to visualise OD solutions, trajectories, residuals,
    covariance ellipses, B-plane maps, and more.  All methods apply a
    consistent style defined in :meth:`set_custom_styling`.

    Notes
    -----
    Scarabaeus' plotting library contains multiple color palettes. See below
    for the colors contained within each:

    .. plot:: api_ref/utils/plotting/show_colors.py
    """

    # ---------------------------#
    # region    Class Constants #
    # ---------------------------#
    color_palettes = {
        "cyanR_palette": [
            "#17BECF",  # Cyan
            "#D95F02",  # Purple
        ],
        "cu_palette": [
            "#E4B542",  # CU Gold
            "#000000",  # CU Black
        ],
        "scb_palette": [
            "#4067BE",  # SCB Logo Blue
            "#15788c",  # Teal
            "#E4B542",  # CU Gold
            "#000000",  # CU Black
        ],
        "colorblind_friendly_palette": [
            "#E69F00",  # Orange
            "#56B4E9",  # Sky Blue
            "#009E73",  # Bluish Green
            "#F0E442",  # Yellow
            "#0072B2",  # Blue
            "#D55E00",  # Vermillion
            "#CC79A7",  # Reddish Purple
        ],
        "matplot_default_palette": [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
            "#8c564b",  # Brown
            "#9467bd",  # Purple
            "#d62728",  # Red
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Yellow-Green
            "#17becf",  # Cyan
        ],
    }

    # endregion Class Constants #
    # ---------------------------#

    # ---------------#
    # region Errors #
    # ---------------#
    class CornerSamplesError(ValueError):
        """The corner-plot samples or the dimensions picked are malformed."""

    class CornerOverlayError(ValueError):
        """A label, truth, mean or covariance does not match the samples."""

    # endregion Errors #
    # ------------------#

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(self) -> None:
        """
        Initialize a Plotting instance and apply the default style sheet.
        """
        # Apply custom style
        self.set_custom_styling()

    # ----------------#
    # region Methods #
    # ----------------#
    def save_plot(
        fig_handle, save_dir=None, save_name: str = "default", save_format: str = "png"
    ):
        """
        Save a Matplotlib figure to disk.

        Parameters
        ----------
        fig_handle : matplotlib.figure.Figure
            The figure object to save.
        save_dir : str or path-like, optional
            Directory in which to write the file. Created automatically if it
            does not exist. Defaults to ``None`` (current working directory).
        save_name : str, optional
            Base name for the output file (without extension for ``"png"``).
            Defaults to ``"default"``.
        save_format : {"png", "pickle"}, optional
            Output format.  ``"png"`` saves a raster image; ``"pickle"``
            serialises the figure object. Defaults to ``"png"``.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If *save_format* is not ``"png"`` or ``"pickle"``.
        """
        Utils.make_dir(save_dir)
        if save_format == "png":
            fig_handle.savefig(os.path.join(save_dir, save_name))
        elif save_format == "pickle":
            pickle.dump(
                fig_handle, open(os.path.join(save_dir, save_name + ".fig.pkl"), "wb")
            )
        else:
            raise RuntimeError(
                "The format chosen is not compatible. Only <png> and <pickle> can be used"
            )

    @staticmethod
    def set_custom_styling():
        """
        Apply the Scarabaeus default ``rcParams`` style to all subsequent plots.

        Sets font family, sizes, grid appearance, and the default color cycle
        to the ``matplot_default_palette``.  Called automatically by
        :meth:`__init__`; can also be called standalone to reset styling after
        external ``rcParams`` changes.
        """

        plt.rcParams.update(
            {
                # Color Styling
                "axes.prop_cycle": cycler(
                    color=Plotting.color_palettes["matplot_default_palette"]
                ),
                "grid.color": "#E0E0E0",
                "grid.linewidth": 0.4,
                # Font Styling
                "font.family": "serif",
                "font.size": 16,
                "axes.titlesize": 20,
                "axes.titleweight": "bold",
                "axes.labelsize": 18,
                "axes.labelweight": "bold",
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                # Grid Styling
                "axes.grid": True,
                "grid.color": "gray",
                "grid.linestyle": "--",
                "grid.linewidth": 0.4,
                # Tight Layout Adjustments
                "figure.autolayout": True,
            }
        )

        plt.rc("axes", grid=True)

    def apply_scale_to_axes(ax, scale="linxy"):
        """
        Apply logarithmic or linear scaling to a Matplotlib axes object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to apply the scaling.
        scale : str, optional
            Scaling option.  One of:

            * ``"logx"``  — log x-axis, linear y-axis.
            * ``"logy"``  — linear x-axis, log y-axis.
            * ``"logxy"`` — log on both axes.
            * ``"linx"``  — linear x-axis only.
            * ``"liny"``  — linear y-axis only.
            * ``"linxy"`` — linear on both axes (default).

        Returns
        -------
        ax : matplotlib.axes.Axes
            The same axes object with the requested scale applied.

        Raises
        ------
        ValueError
            If *scale* is not one of the six recognised options.
        """
        # Validating inputs
        valid_scales = ["logx", "logy", "logxy", "linx", "liny", "linxy"]
        if scale not in valid_scales:
            raise ValueError(
                f"Invalid scale option '{scale}'. Choose from {valid_scales}."
            )

        # Scaling based on the input
        if scale == "logx":
            ax.set_xscale("log")
            ax.set_yscale("linear")
        elif scale == "logy":
            ax.set_xscale("linear")
            ax.set_yscale("log")
        elif scale == "logxy":
            ax.set_xscale("log")
            ax.set_yscale("log")
        elif scale == "linx" or scale == "linxy":
            ax.set_xscale("linear")
        elif scale == "liny":
            ax.set_yscale("linear")

        return ax

    # X-axis Offset
    def plot_with_x_offset(values: list, x_offset_flag: bool = False):
        """
        Subtract the first value from all elements to create a zero-based x-axis.

        Parameters
        ----------
        values : list
            List of x-axis values to shift.
        x_offset_flag : bool, optional
            When ``True``, subtracts ``values[0]`` from every element.
            When ``False``, returns *values* unchanged. Defaults to ``False``.

        Returns
        -------
        values : list
            The (possibly offset) list of x-axis values.
        """
        # Apply offset if the flag is True
        if x_offset_flag:
            offset = values[0]  # Subtract the first value as the offset
            values = [val - offset for val in values]
        return values

    # ----------------------#
    # region SolutionOD    #
    # ----------------------#

    def plot_state_errors(
        solution_it: SolutionOD,
        true_trajectory: Trajectory,
        epochs: EpochArray,
        n_sigma: int,
        title: str = "State Errors",
        t_0: bool = True,
        scale: str = "linxy",
        x_offset_flag: bool = False,
        use_smoothed: bool = True,
    ):
        """
        Plots the state errors and sigma bounds for each component of position and velocity.

        :param solution_it: Solution OD Object.
        :type solution_it: SolutionOD

        :param true_trajectory: The true trajectory of the object.
        :type true_trajectory: Trajectory

        :param epochs: The array of epochs.
        :type epochs: EpochArray

        :param n_sigma: The number of standard deviations for the sigma bounds.
        :type n_sigma: int

        :param title: Title for the plot.
        :type title: str

        :param t_0: If True, convert the x-axis to time in days relative to the initial epoch.
        :type t_0: bool

        :param scale: Scale type for axes ('linxy', 'logxy', etc.).
        :type scale: str

        :param x_offset_flag: If True, apply x-axis offset.
        :type x_offset_flag: bool

        :param use_smoothed: If True and smoothed solution is available, uses smoothed covariance.
        :type use_smoothed: bool

        :returns: Figure object.
        :rtype: fig
        """
        # Setting custom color for sigma fill and error line
        colors = Plotting.color_palettes["cu_palette"]
        sigma_color = colors[0]
        error_color = colors[1]

        # Retrieve errors using the smart estimated_trajectory method
        estimated_position, estimated_velocity, _ = solution_it.estimated_trajectory(
            epochs, use_smoothed=use_smoothed
        )

        covariance_source = solution_it.propagate_covariance(
            epochs, use_smoothed=use_smoothed
        )

        t = epochs.times.values
        n = len(t)

        # Preallocate arrays
        err_position = np.zeros(shape=(n, 3))
        err_velocity = np.zeros(shape=(n, 3))
        sigmas = np.zeros(shape=(n, 6))

        # Vectorized computation where possible
        for k in range(n):
            if solution_it.filter.flag_sequence is True:
                idx_leg = next(
                    (
                        (
                            i + 1
                            if epochs[k].times.values == leg[-1]
                            and i != len(solution_it.filter.legs_epochs) - 1
                            else i
                        )
                        for i, leg in enumerate(solution_it.filter.legs_epochs)
                        if (
                            (epochs[k].times.values in leg)
                            if isinstance(leg, np.ndarray)
                            else (epochs[k].times.values == leg)
                        )
                    ),
                    None,
                )
            else:
                idx_leg = None

            # Retrieve true state
            state_nominal = true_trajectory.get_state(
                epoch_input=epochs[k], idx_leg_input=idx_leg
            )

            # Compute errors
            err_position[k] = estimated_position[k] - state_nominal["position"].values
            err_velocity[k] = estimated_velocity[k] - state_nominal["velocity"].values

            # Extract sigmas from covariance (already stored as matrix)
            marginal_variances = np.diag(covariance_source[k])
            sigmas[k] = np.sqrt(marginal_variances[:6])

        # Optionally convert time to days relative to the initial epoch
        if t_0:
            t = (t - t[0]) / (24 * 3600)  # Convert to time in days

        # Plot the state errors and sigma bounds
        fig, axs = plt.subplots(2, 3, figsize=(19, 9))

        plt.subplots_adjust(wspace=0.6, hspace=0.4)
        # Add title
        fig.suptitle(title, fontweight="bold")

        # Apply scaling
        for row in axs:
            for ax in row:
                ax = Plotting.apply_scale_to_axes(ax, scale)

        # Apply x_axis offset
        if x_offset_flag:
            t = Plotting.plot_with_x_offset(t, x_offset_flag)

        # Iterate over each subplot and enable minor ticks
        for ax in axs.flat:
            # Set minor locators explicitly for both axes
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            # Enable minor ticks
            ax.minorticks_on()

        # Set the x-axis label based on whether t_0 is used
        x_label_time = f"Time - $t_0$ [days]" if t_0 else f"Time [{epochs.time.units}]"

        sigma_levels = [i for i in range(1, n_sigma + 1)]  # sigma levels
        alpha_values = {1: 0.6, 2: 0.4, 3: 0.3}  # sigma - alpha mapping

        # Precompute time limits
        t_min, t_max = min(t), max(t)

        # Position x-axis
        axs[0, 0].plot(t, err_position[:, 0], error_color, label="Error", linestyle = "none", marker = "o", markersize = 3)
        for level in sigma_levels:
            axs[0, 0].fill_between(
                t,
                -level * sigmas[:, 0],
                level * sigmas[:, 0],
                color=sigma_color,
                alpha=alpha_values[level],
                label=f"{level}σ",
            )
        axs[0, 0].set(xlabel=x_label_time, ylabel=r"$\varepsilon_x$ [km]")
        axs[0, 0].grid(True, which="minor", linestyle="--")
        axs[0, 0].legend(loc="upper left", framealpha=0.8)
        axs[0, 0].set_xlim(t_min, t_max)
        axs[0, 0].autoscale(False)

        # Position y-axis
        axs[0, 1].plot(t, err_position[:, 1], error_color, label="Error", linestyle = "none", marker = "o", markersize = 3)
        for level in sigma_levels:
            axs[0, 1].fill_between(
                t,
                -level * sigmas[:, 1],
                level * sigmas[:, 1],
                color=sigma_color,
                alpha=alpha_values[level],
                label=f"{level}σ",
            )
        axs[0, 1].set(xlabel=x_label_time, ylabel=r"$\varepsilon_y$ [km]")
        axs[0, 1].grid(True, which="minor", linestyle="--")
        axs[0, 1].legend(loc="upper left", framealpha=0.8)
        axs[0, 1].set_xlim(t_min, t_max)

        # Position z-axis
        axs[0, 2].plot(t, err_position[:, 2], error_color, label="Error", linestyle = "none", marker = "o", markersize = 3)
        for level in sigma_levels:
            axs[0, 2].fill_between(
                t,
                -level * sigmas[:, 2],
                level * sigmas[:, 2],
                color=sigma_color,
                alpha=alpha_values[level],
                label=f"{level}σ",
            )
        axs[0, 2].set(xlabel=x_label_time, ylabel=r"$\varepsilon_z$ [km]")
        axs[0, 2].grid(True, which="minor", linestyle="--")
        axs[0, 2].legend(loc="upper left", framealpha=0.8)
        axs[0, 2].set_xlim(t_min, t_max)

        # Velocity x-axis
        axs[1, 0].plot(t, err_velocity[:, 0], error_color, label="Error", linestyle = "none", marker = "o", markersize = 3)
        for level in sigma_levels:
            axs[1, 0].fill_between(
                t,
                -level * sigmas[:, 3],
                level * sigmas[:, 3],
                color=sigma_color,
                alpha=alpha_values[level],
                label=f"{level}σ",
            )
        axs[1, 0].set(xlabel=x_label_time, ylabel=r"$\varepsilon_{\dot{x}}$ [km/s]")
        axs[1, 0].grid(True, which="minor", linestyle="--")
        axs[1, 0].legend(loc="upper left", framealpha=0.8)
        axs[1, 0].set_xlim(t_min, t_max)

        # Velocity y-axis
        axs[1, 1].plot(t, err_velocity[:, 1], error_color, label="Error", linestyle = "none", marker = "o", markersize = 3)
        for level in sigma_levels:
            axs[1, 1].fill_between(
                t,
                -level * sigmas[:, 4],
                level * sigmas[:, 4],
                color=sigma_color,
                alpha=alpha_values[level],
                label=f"{level}σ",
            )
        axs[1, 1].set(xlabel=x_label_time, ylabel=r"$\varepsilon_{\dot{y}}$ [km/s]")
        axs[1, 1].grid(True, which="minor", linestyle="--")
        axs[1, 1].legend(loc="upper left", framealpha=0.8)
        axs[1, 1].set_xlim(t_min, t_max)

        # Velocity z-axis
        axs[1, 2].plot(t, err_velocity[:, 2], error_color, label="Error", linestyle = "none", marker = "o", markersize = 3)
        for level in sigma_levels:
            axs[1, 2].fill_between(
                t,
                -level * sigmas[:, 5],
                level * sigmas[:, 5],
                color=sigma_color,
                alpha=alpha_values[level],
                label=f"{level}σ",
            )
        axs[1, 2].set(xlabel=x_label_time, ylabel=r"$\varepsilon_{\dot{z}}$ [km/s]")
        axs[1, 2].grid(True, which="minor", linestyle="--")
        axs[1, 2].legend(loc="upper left", framealpha=0.8)
        axs[1, 2].set_xlim(t_min, t_max)

        # Plot
        plt.tight_layout()

        return fig

    def plot_state_errors_norm(
        solution_it: SolutionOD,
        true_trajectory: Trajectory,
        epochs: EpochArray,
        n_sigma: int,
        title: str = "State Errors Norm",
        t_0: bool = True,
        scale: str = "linxy",
        x_offset_flag: bool = False,
        use_smoothed: bool = True,
    ):
        """
        Plots the state errors and sigma values over time.

        :param solution_it: Solution OD Object.
        :type solution_it: SolutionOD

        :param true_trajectory: The true trajectory.
        :type true_trajectory: Trajectory

        :param epochs: The array of epochs.
        :type epochs: EpochArray

        :param n_sigma: The number of standard deviations for sigma.
        :type n_sigma: int

        :param title: Title for the plot.
        :type title: str

        :param t_0: If True, convert the x-axis to time in days relative to the initial epoch.
        :type t_0: bool

        :param scale: Scaling option ['logx', 'logy', 'logxy', 'linx', 'liny', 'linxy']
        :type scale: str

        :param x_offset_flag: Offset value for X axis
        :type x_offset_flag: bool

        :param use_smoothed: If True and smoothed solution is available, uses smoothed covariance.
        :type use_smoothed: bool

        :returns: Figure object.
        :rtype: fig
        """

        # Setting custom color for sigma fill and error line
        colors = Plotting.color_palettes["cu_palette"]
        sigma_color = colors[0]
        error_color = colors[1]

        # Retrieve errors using the smart estimated_trajectory method
        estimated_position, estimated_velocity, _ = solution_it.estimated_trajectory(
            epochs, use_smoothed=use_smoothed
        )

        covariance_source = solution_it.propagate_covariance(
            epochs, use_smoothed=use_smoothed
        )

        t = epochs.times.values
        n = len(t)

        # Preallocate arrays
        err_position = np.zeros(shape=(n, 3))
        err_velocity = np.zeros(shape=(n, 3))
        sigmas = np.zeros(shape=(n, 6))

        for k in range(n):
            if solution_it.filter.flag_sequence is True:
                idx_leg = next(
                    (
                        (
                            i + 1
                            if epochs[k].times.values == leg[-1]
                            and i != len(solution_it.filter.legs_epochs) - 1
                            else i
                        )
                        for i, leg in enumerate(solution_it.filter.legs_epochs)
                        if (
                            (epochs[k].times.values in leg)
                            if isinstance(leg, np.ndarray)
                            else (epochs[k].times.values == leg)
                        )
                    ),
                    None,
                )
            else:
                idx_leg = None

            # Retrieve true state
            state_nominal = true_trajectory.get_state(
                epoch_input=epochs[k], idx_leg_input=idx_leg
            )

            # Compute errors
            err_position[k] = estimated_position[k] - state_nominal["position"].values
            err_velocity[k] = estimated_velocity[k] - state_nominal["velocity"].values

            # Extract sigmas from covariance (already stored as matrix)
            marginal_variances = np.diag(covariance_source[k])
            sigmas[k] = np.sqrt(marginal_variances[:6])

        # Optionally convert time to days relative to the initial epoch
        if t_0:
            t = (t - t[0]) / (24 * 3600)  # Convert to time in days

        # Plot the state errors and sigma values
        fig, axs = plt.subplots(
            2,
            3,
            figsize=(19, 9),
            sharex=False,
        )

        plt.subplots_adjust(wspace=0.6, hspace=0.3)
        # Add a title
        fig.suptitle(title, fontweight="bold")

        # Apply scaling
        for row in axs:
            for ax in row:
                ax = Plotting.apply_scale_to_axes(ax, scale)

        # Apply x_axis offset
        if x_offset_flag:
            t = Plotting.plot_with_x_offset(t, x_offset_flag)

        # Iterate over each subplot and enable minor ticks
        for ax in axs.flat:
            # Use AutoMinorLocator for the x-axis (linear scale)
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

            # Use LogLocator for the y-axis (log scale)
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto"))

            # Enable minor ticks
            ax.minorticks_on()

        # Set the x-axis label based on whether t_0 is used
        x_label_time = f"Time - $t_0$ [days]" if t_0 else f"Time [{epochs.times.units}]"

        sigma_levels = [i for i in range(1, n_sigma + 1)]  # sigma level
        alpha_values = {1: 0.6, 2: 0.4, 3: 0.3}  # alpha - sigma mapping

        # Position x-axis
        axs[0, 0].semilogy(t, np.abs(err_position[:, 0]), error_color, marker = 'o', linestyle = "none", label="Error", markersize = 3)
        for level in sigma_levels:
            axs[0, 0].fill_between(
                t,
                level * sigmas[:, 0],
                color=sigma_color,
                alpha=alpha_values[level],
                label=f"{level}σ",
            )
        axs[0, 0].set(xlabel=x_label_time, ylabel=r"$\varepsilon_x$ [km]")
        axs[0, 0].grid(True, which="minor", linestyle="--")
        axs[0, 0].legend(loc="upper left")

        # Position y-axis
        axs[0, 1].semilogy(t, np.abs(err_position[:, 1]), error_color, marker = "o", linestyle = "none", label="Error", markersize = 3)
        for level in sigma_levels:
            axs[0, 1].fill_between(
                t,
                level * sigmas[:, 1],
                color=sigma_color,
                alpha=alpha_values[level],
                label=f"{level}σ",
            )
        axs[0, 1].set(xlabel=x_label_time, ylabel=r"$\varepsilon_y$ [km]")
        axs[0, 1].grid(True, which="minor", linestyle="--")
        axs[0, 1].legend(loc="upper left")

        # Position z-axis
        axs[0, 2].semilogy(t, np.abs(err_position[:, 2]), error_color, marker = "o", linestyle = "none", label="Error", markersize = 3)
        for level in sigma_levels:
            axs[0, 2].fill_between(
                t,
                level * sigmas[:, 2],
                color=sigma_color,
                alpha=alpha_values[level],
                label=f"{level}σ",
            )
        axs[0, 2].set(xlabel=x_label_time, ylabel=r"$\varepsilon_z$ [km]")
        axs[0, 2].grid(True, which="minor", linestyle="--")
        axs[0, 2].legend(loc="upper left")

        # Velocity x-axis
        axs[1, 0].semilogy(t, np.abs(err_velocity[:, 0]), error_color, marker = "o", linestyle = "none", label="Error", markersize = 3)
        for level in sigma_levels:
            axs[1, 0].fill_between(
                t,
                level * sigmas[:, 3],
                color=sigma_color,
                alpha=alpha_values[level],
                label=f"{level}σ",
            )
        axs[1, 0].set(xlabel=x_label_time, ylabel=r"$\varepsilon_{\dot{x}}$ [km/s]")
        axs[1, 0].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
        axs[1, 0].grid(True, which="minor", linestyle="--")
        axs[1, 0].legend(loc="upper left")

        # Velocity y-axis
        axs[1, 1].semilogy(t, np.abs(err_velocity[:, 1]), error_color, marker = "o", linestyle = "none", label="Error", markersize = 3)
        for level in sigma_levels:
            axs[1, 1].fill_between(
                t,
                level * sigmas[:, 4],
                color=sigma_color,
                alpha=alpha_values[level],
                label=f"{level}σ",
            )
        axs[1, 1].set(xlabel=x_label_time, ylabel=r"$\varepsilon_{\dot{y}}$ [km/s]")
        axs[1, 1].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
        axs[1, 1].grid(True, which="minor", linestyle="--")
        axs[1, 1].legend(loc="upper left")

        # Velocity z-axis
        axs[1, 2].semilogy(t, np.abs(err_velocity[:, 2]), error_color, marker = "o", linestyle = "none", label="Error", markersize  = 3)
        for level in sigma_levels:
            axs[1, 2].fill_between(
                t,
                level * sigmas[:, 5],
                color=sigma_color,
                alpha=alpha_values[level],
                label=f"{level}σ",
            )
        axs[1, 2].set(xlabel=x_label_time, ylabel=r"$\varepsilon_{\dot{z}}$ [km/s]")
        axs[1, 2].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
        axs[1, 2].grid(True, which="minor", linestyle="--")
        axs[1, 2].legend(loc="upper left")

        # Plot
        plt.tight_layout()

        return fig

    def plot_compare_state_error_norm(
        solutions,
        true_trajectory,
        epochs_list,
        n_sigma,
        title="State Error Comparison",
        labels=None,
        t_0=True,
        scale="linxy",
        x_offset_flag=False,
        use_smoothed=True,
    ):
        """
        Plots state errors norm for multiple iterations on the same figure for comparison.

        :param solutions: List of SolutionOD objects.
        :type solutions: list

        :param true_trajectory: The true trajectory.
        :type true_trajectory: Trajectory

        :param epochs_list: List of EpochArray objects, one for each solution.
        :type epochs_list: list

        :param n_sigma: The number of standard deviations for sigma.
        :type n_sigma: int

        :param title: Title of the plot.
        :type title: str

        :param labels: Labels for the solutions (e.g., ['IT1', 'IT2', 'IT3']).
        :type labels: list, optional

        :param t_0: If True, convert the x-axis to time in days relative to the initial epoch.
        :type t_0: bool, optional

        :param scale: Scale type for plots.
        :type scale: str, optional

        :param x_offset_flag: If True, apply x-axis offset.
        :type x_offset_flag: bool, optional

        :param use_smoothed: If True and smoothed solution is available, uses smoothed covariance.
        :type use_smoothed: bool

        :returns: Figure object.
        :rtype: matplotlib.figure.Figure
        """

        # Setting the sigma fill color
        sigma_color = Plotting.color_palettes["scb_palette"][2]
        # Setting custom colors for each iteration line
        it_colors = Plotting.color_palettes["colorblind_friendly_palette"]
        it_1_color = it_colors[6]
        it_2_color = it_colors[1]
        it_3_color = it_colors[2]
        colors = [it_1_color, it_2_color, it_3_color]

        # Use provided labels or generate default ones
        if labels is None:
            labels = [f"IT{i+1}" for i in range(len(solutions))]

        fig, axs = plt.subplots(2, 3, figsize=(19, 9), sharex=False)

        # Iterate over each subplot and enable minor ticks
        for ax in axs.flat:
            # Set minor locators explicitly for both axes
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            # Enable minor ticks
            ax.minorticks_on()

        plt.subplots_adjust(wspace=0.6, hspace=0.3)

        # Add a title
        fig.suptitle(title, fontweight="bold")

        sigma_levels = [i for i in range(1, n_sigma + 1)]
        alpha_values = {1: 0.6, 2: 0.4, 3: 0.3}

        for i, (solution, epochs) in enumerate(zip(solutions, epochs_list)):
            t = epochs.times.values
            if t_0:
                t = (t - t[0]) / (24 * 3600)  # Convert to time in days

            # Retrieve errors using the smart estimated_trajectory method
            estimated_position, estimated_velocity, _ = solution.estimated_trajectory(
                epochs, use_smoothed=use_smoothed
            )

            covariance_source = solution.propagate_covariance(
                epochs, use_smoothed=use_smoothed
            )

            # Preallocate arrays
            err_position = np.zeros((len(t), 3))
            err_velocity = np.zeros((len(t), 3))
            sigmas = np.zeros((len(t), 6))

            for k in range(len(t)):
                state_nominal = true_trajectory.get_state(epoch_input=epochs[k])
                err_position[k] = (
                    estimated_position[k] - state_nominal["position"].values
                )
                err_velocity[k] = (
                    estimated_velocity[k] - state_nominal["velocity"].values
                )

                # Extract sigmas from covariance (already stored as matrix)
                marginal_variances = np.diag(covariance_source[k])
                sigmas[k] = np.sqrt(marginal_variances[:6])

            # Plot each iteration
            for j in range(3):
                axs[0, j].semilogy(
                    t, np.abs(err_position[:, j]), color=colors[i], label=labels[i]
                )
                axs[1, j].semilogy(
                    t, np.abs(err_velocity[:, j]), color=colors[i], label=labels[i]
                )
                if i == 0:  # Only plot sigma bounds once (from first solution)
                    for level in sigma_levels:
                        axs[0, j].fill_between(
                            t,
                            level * sigmas[:, j],
                            color=sigma_color,
                            alpha=alpha_values[level],
                            label=f"{level}σ" if j == 0 else None,  # Label only once
                        )
                        axs[1, j].fill_between(
                            t,
                            level * sigmas[:, j + 3],
                            color=sigma_color,
                            alpha=alpha_values[level],
                            label=f"{level}σ" if j == 0 else None,  # Label only once
                        )

        # Label axes
        state_labels = [
            r"$\varepsilon_x$ [km]",
            r"$\varepsilon_y$ [km]",
            r"$\varepsilon_z$ [km]",
        ]
        vel_labels = [
            r"$\varepsilon_{\dot{x}}$ [km/s]",
            r"$\varepsilon_{\dot{y}}$ [km/s]",
            r"$\varepsilon_{\dot{z}}$ [km/s]",
        ]

        x_label_time = f"Time - $t_0$ [days]" if t_0 else "Time [days]"

        for j in range(3):
            axs[0, j].set(xlabel=x_label_time, ylabel=state_labels[j])
            axs[1, j].set(xlabel=x_label_time, ylabel=vel_labels[j])
            axs[0, j].grid(True, which="minor", linestyle="--")
            axs[1, j].grid(True, which="minor", linestyle="--")
            axs[1, j].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))

        axs[0, 0].legend(loc="upper left")
        plt.tight_layout()
        return fig

    def plot_parameters_errors(
        solution_it: SolutionOD,
        true_trajectory: Trajectory,
        epochs: EpochArray,
        n_sigma: int,
        title: str = "Parameters Errors",
        t_f: bool = True,
        use_smoothed: bool = True,
    ):
        """
        Plots the state errors and sigma bounds for each component of parameters.

        Returns
        -------
        list
            List of figures.
        """

        # --- STYLE: match plot_state_errors
        colors = Plotting.color_palettes["cu_palette"]
        sigma_color = colors[0]
        error_color = colors[1]

        sigma_levels = [i for i in range(1, n_sigma + 1)]
        alpha_values = {1: 0.6, 2: 0.4, 3: 0.3}  # same mapping as plot_state_errors

        # Retrieve estimated parameters
        _, _, estimated_parameters = solution_it.estimated_trajectory(
            epochs, use_smoothed=use_smoothed
        )

        covariance_source = solution_it.propagate_covariance(
            epochs, use_smoothed=use_smoothed
        )

        # Initialize
        t = epochs.times.values
        n = len(t)

        # Convert time to days relative to the final epoch if t_f is True
        if t_f:
            t = (t - t[-1]) / (24 * 3600)

        if not solution_it.filter.flag_sequence:
            shape = (
                (n, 1)
                if len(solution_it.filter.trajectory.parameters.shape) == 1
                else (n, solution_it.filter.trajectory.parameters.shape[1])
            )
            err_parameters = np.zeros(shape=shape)
            sigmas = np.zeros(shape=(n, solution_it.filter.n))
        else:
            # use number of legs, not len(trajectory.parameters)
            nlegs = len(solution_it.filter.legs_epochs)

            # remove the duplicated node epoch slot for legs > 0
            # Convention: node epoch belongs to previous leg, so each leg after the first
            # is treated as half-open on the left.
            leg_lengths_eff = [
                len(solution_it.filter.legs_epochs[i]) - (1 if i > 0 else 0)
                for i in range(nlegs)
            ]

            err_parameters = [
                [
                    np.zeros(shape=(solution_it.filter.legs_n[i] - 6))
                    for _ in range(leg_lengths_eff[i])
                ]
                for i in range(nlegs)
            ]
            sigmas = [
                [
                    np.zeros(shape=(solution_it.filter.legs_n[i]))
                    for _ in range(leg_lengths_eff[i])
                ]
                for i in range(nlegs)
            ]

        # Calculate errors and sigmas
        for k in range(n):
            if solution_it.filter.flag_sequence is True:
                # robust leg membership (no early switch)
                idx_leg = next(
                    (
                        i
                        for i, leg in enumerate(solution_it.filter.legs_epochs)
                        if (
                            np.any(
                                np.isclose(
                                    epochs[k].times.values, leg, rtol=0.0, atol=1e-9
                                )
                            )
                            if isinstance(leg, np.ndarray)
                            else np.isclose(
                                epochs[k].times.values, leg, rtol=0.0, atol=1e-9
                            )
                        )
                    ),
                    None,
                )
            else:
                idx_leg = None

            # Retrieve true state
            state_nominal = true_trajectory.get_state(
                epoch_input=epochs[k], idx_leg_input=idx_leg
            )

            # Compute k_offset (your existing approach)
            k_offset = None
            if solution_it.filter.flag_sequence:
                k_offset = (
                    sum(
                        len(stm)
                        for stm in solution_it.filter.trajectory._STMs[:idx_leg]
                    )
                    - idx_leg
                )

            # shift local index by -1 for legs > 0
            # because we removed the duplicated node epoch slot at the beginning.
            if solution_it.filter.flag_sequence:
                kk = k - k_offset
                if idx_leg > 0:
                    kk -= 1
                # If we are exactly on the duplicated node epoch, kk becomes -1: skip it.
                if kk < 0:
                    continue

            # Compute parameter errors
            if solution_it.filter.flag_sequence is False:
                err_parameters[k] = (
                    estimated_parameters[k] - state_nominal["parameters"].values
                )
            else:
                if (
                    hasattr(state_nominal["parameters"], "values")
                    and state_nominal["parameters"].values is not None
                ):
                    err_parameters[idx_leg][kk] = (
                        estimated_parameters[idx_leg][kk]
                        - state_nominal["parameters"].values
                    )

            # Extract sigmas from covariance (global k is correct per your statement)
            marginal_variances = np.diag(covariance_source[k])
            if solution_it.filter.flag_sequence is False:
                sigmas[k] = np.sqrt(marginal_variances)
            else:
                sigmas[idx_leg][kk] = np.sqrt(marginal_variances)

        # Common x-label
        x_label_time = "Time - $t_f$ [days]" if t_f else "Time [TBD]"

        # PLOTTING
        if solution_it.filter.flag_sequence is True:
            figures = []
            for idx_leg, (model, epochs_leg) in enumerate(
                zip(solution_it.filter.legs_model, solution_it.filter.legs_epochs)
            ):
                state_definition = model.state
                param_def = [
                    state_definition[j + 2][0] for j in range(len(state_definition) - 2)
                ]
                param_size = [
                    state_definition[j + 2][1] for j in range(len(state_definition) - 2)
                ]
                param_units = [
                    state_definition[j + 2][5].quantity.units
                    for j in range(len(state_definition) - 2)
                ]

                # plot epochs without the duplicated first epoch for legs > 0
                epochs_leg_plot = epochs_leg[1:] if idx_leg > 0 else epochs_leg

                if param_def:
                    k0 = 0
                    for param, size in zip(param_def, param_size):
                        fig, axs = plt.subplots(size, 1, figsize=(5, 3 * size))
                        plt.subplots_adjust(wspace=0.6, hspace=0.3)
                        fig.suptitle(f"{title} - {param}", fontweight="bold")

                        if size == 1:
                            axs = [axs]

                        for j in range(size):
                            # Error line
                            axs[j].plot(
                                epochs_leg_plot,
                                [elem[k0 + j] for elem in err_parameters[idx_leg][:]],
                                color=error_color,
                                label="Error",
                            )

                            # Multi-level sigma shades
                            for level in sigma_levels:
                                sigma_plus = [
                                    level * elem[6 + k0 + j] for elem in sigmas[idx_leg]
                                ]
                                sigma_minus = [
                                    -level * elem[6 + k0 + j]
                                    for elem in sigmas[idx_leg]
                                ]
                                axs[j].fill_between(
                                    epochs_leg_plot,
                                    sigma_minus,
                                    sigma_plus,
                                    color=sigma_color,
                                    alpha=alpha_values.get(level, 0.3),
                                    label=f"{level}σ",
                                )

                            axs[j].set(xlabel="Time [TBD]")
                            axs[j].grid(True, which="minor", linestyle="--")
                            axs[j].minorticks_on()
                            axs[j].xaxis.set_minor_locator(AutoMinorLocator())
                            axs[j].yaxis.set_minor_locator(AutoMinorLocator())
                            axs[j].legend(loc="upper left", framealpha=0.8)

                        k0 += size
                        plt.tight_layout()
                        figures.append(fig)

        else:
            state_definition = solution_it.filter.trajectory._state_definition
            param_def = [
                state_definition[i + 2][0] for i in range(len(state_definition) - 2)
            ]
            param_size = [
                state_definition[i + 2][1] for i in range(len(state_definition) - 2)
            ]
            param_units = [
                state_definition[i + 2][5].quantity.units
                for i in range(len(state_definition) - 2)
            ]

            t_min, t_max = min(t), max(t)

            k0 = 0
            figures = []
            for param, size in zip(param_def, param_size):
                fig, axs = plt.subplots(size, 1, figsize=(5, 3 * size))
                plt.subplots_adjust(wspace=0.6, hspace=0.3)
                fig.suptitle(f"{title} - {param}", fontweight="bold")

                if size == 1:
                    axs = [axs]

                for j in range(size):
                    axs[j].plot(
                        t, err_parameters[:, k0 + j], color=error_color, label="Error"
                    )

                    for level in sigma_levels:
                        axs[j].fill_between(
                            t,
                            -level * sigmas[:, 6 + k0 + j],
                            level * sigmas[:, 6 + k0 + j],
                            color=sigma_color,
                            alpha=alpha_values.get(level, 0.3),
                            label=f"{level}σ",
                        )

                    axs[j].set(xlabel=x_label_time)
                    axs[j].grid(True, which="minor", linestyle="--")
                    axs[j].minorticks_on()
                    axs[j].xaxis.set_minor_locator(AutoMinorLocator())
                    axs[j].yaxis.set_minor_locator(AutoMinorLocator())
                    axs[j].legend(loc="upper left", framealpha=0.8)
                    axs[j].set_xlim(t_min, t_max)

                k0 += size
                plt.tight_layout()
                figures.append(fig)

        return figures

    def plot_parameters_estimate(
        solution_it: SolutionOD,
        epochs: EpochArray,
        n_sigma: int,
        title="Parameters Errors",
        sample_points: int = 40,
        t_0: bool = True,
        use_smoothed: bool = True,
    ):
        """
        Plots the estimated parameters over time with error bars and shaded uncertainty bounds (:math:`\\pm n\\sigma`).
        Only a sample of `sample_points` is plotted to reduce clutter.

        :param solution_it: The SolutionOD object.
        :type solution_it: SolutionOD

        :param epochs: The array of epochs.
        :type epochs: EpochArray

        :param n_sigma: The number of standard deviations for uncertainty bounds.
        :type n_sigma: int

        :param title: Title for the plot.
        :type title: str

        :param sample_points: Number of data points to sample for plotting.
        :type sample_points: int

        :param t_0: If True, convert the x-axis to time in days relative to the initial epoch.
        :type t_0: bool

        :param use_smoothed: If True and smoothed solution is available, uses smoothed covariance.
        :type use_smoothed: bool

        :returns: List of figure objects.
        :rtype: list of matplotlib.figure.Figure
        """

        # Setting custom color for data point and sigma fill
        colors = Plotting.color_palettes["scb_palette"]
        error_point_color = colors[0]
        sigma_color = colors[2]

        # Retrieve estimated parameters using the smart estimated_trajectory method
        _, _, estimated_parameters = solution_it.estimated_trajectory(
            epochs, use_smoothed=use_smoothed
        )

        covariance_source = solution_it.propagate_covariance(
            epochs, use_smoothed=use_smoothed
        )

        # Initialize
        t = epochs.times.values
        n = len(t)

        # Convert time to days relative to the initial epoch if t_0 is True
        if t_0:
            t = (t - t[0]) / (24 * 3600)  # Convert to time in days

        # Preallocate sigmas array
        sigmas = np.zeros(shape=(n, solution_it.filter.n))

        # Extract sigmas from covariance (already stored as matrix)
        for k in range(n):
            marginal_variances = np.diag(covariance_source[k])
            sigmas[k] = np.sqrt(marginal_variances)

        # Sample indices for evenly spaced data points
        sampled_indices = np.linspace(0, n - 1, min(sample_points, n), dtype=int)
        t_sampled = t[sampled_indices]

        # Get parameter definitions
        state_definition = solution_it.filter.trajectory._state_definition
        param_def = [
            state_definition[i + 2][0] for i in range(len(state_definition) - 2)
        ]
        param_size = [
            state_definition[i + 2][1] for i in range(len(state_definition) - 2)
        ]

        # Plot estimated parameters with error bars
        figures = []
        k = 0
        for i, (param, size) in enumerate(zip(param_def, param_size)):
            fig, axs = plt.subplots(size, 1, figsize=(18, 11 * size))

            # Ensure axs is always iterable
            if not isinstance(axs, np.ndarray):
                axs = np.array([axs])

            # Iterate over each subplot and enable minor ticks
            for ax in axs.flat:
                # Set minor locators explicitly for both axes
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                # Enable minor ticks
                ax.minorticks_on()

            plt.subplots_adjust(hspace=0.5)
            fig.suptitle(title, fontweight="bold")

            if size == 1:
                axs = [axs[0]]  # Extract single element from array

            for j in range(size):
                mean_values_sampled = estimated_parameters[sampled_indices, k + j]
                sigma_values = sigmas[
                    sampled_indices, 6 + k + j
                ]  # Parameters start at index 6

                # Plot estimated parameters with error bars
                axs[j].errorbar(
                    t_sampled,
                    mean_values_sampled,
                    yerr=n_sigma * sigma_values,
                    fmt="o",
                    ecolor=error_point_color,
                    capsize=5,
                    label=f"Estimated {param} ± {n_sigma}σ",
                )

                # Plot uncertainty bounds
                alpha_value_list = [0.6, 0.45, 0.3]
                for sigma_level in range(n_sigma, 0, -1):
                    alpha_value = alpha_value_list[sigma_level - 1]
                    axs[j].fill_between(
                        t_sampled,
                        mean_values_sampled - sigma_level * sigma_values,
                        mean_values_sampled + sigma_level * sigma_values,
                        color=sigma_color,
                        alpha=alpha_value,
                        label=f"{sigma_level}σ",
                    )

                axs[j].set(
                    xlabel="Time - $t_0$ [days]" if t_0 else "Time [TBD]",
                    ylabel=f"{param}",
                )
                axs[j].grid(True, which="minor", linestyle="--")
                axs[j].legend(loc="upper left")

            k += size
            figures.append(fig)

        plt.tight_layout()
        return figures

    def plot_fogm_parameters_estimate(
        solution_it: SolutionOD,
        epochs: EpochArray,
        n_sigma: int,
        title: str = "FOGM Parameter Estimates",
        sample_points: int = 40,
        t_0: bool = True,
        use_smoothed: bool = True,
    ):
        """
        Plot only stochastic Gauss-Markov parameters over time.

        Parameters
        ----------
        solution_it : SolutionOD
            The orbit determination solution object.

        epochs : EpochArray
            Epoch array for plotting.

        n_sigma : int
            Number of sigma levels to show.

        title : str, optional
            Figure title prefix.

        sample_points : int, optional
            Number of sampled points to plot.

        t_0 : bool, optional
            If True, plot time in days relative to initial epoch.

        use_smoothed : bool, optional
            If True and available, use smoothed estimates/covariances.

        Returns
        -------
        list
            List of matplotlib figure objects.

        Notes
        -----
        Supported parameters: a_fogm, beta_fogm, a_pfogm, beta_pfogm.

        For piecewise parameters (a_pfogm, beta_pfogm), the stored state may have
        size 3*n_batches, but only the active batch is plotted at each epoch.
        Therefore the output always has 3 rows (x, y, z) for each stochastic parameter.
        """

        # -----------------------------
        # Styling
        # -----------------------------
        colors = Plotting.color_palettes["scb_palette"]
        error_point_color = colors[0]
        sigma_color = colors[2]

        stochastic_params = {"a_fogm", "beta_fogm", "a_pfogm", "beta_pfogm"}

        # -----------------------------
        # Retrieve estimates
        # -----------------------------
        _, _, estimated_parameters = solution_it.estimated_trajectory(
            epochs, use_smoothed=use_smoothed
        )

        # Select covariance source
        if use_smoothed and solution_it.covariance_smooth is not None:
            covariance_source = solution_it.covariance_smooth
        else:
            covariance_source = solution_it.covariance_est

        # -----------------------------
        # Time axis
        # -----------------------------
        t_raw = np.asarray(epochs.times.values, dtype=float)
        n = len(t_raw)

        if t_0:
            t = (t_raw - t_raw[0]) / (24 * 3600)
            x_label = "Time - $t_0$ [days]"
        else:
            t = t_raw
            x_label = "Time [TDB]"

        # -----------------------------
        # Extract sigmas
        # -----------------------------
        sigmas = np.zeros((n, solution_it.filter.n))
        for k in range(n):
            marginal_variances = np.diag(covariance_source[k])
            sigmas[k] = np.sqrt(marginal_variances)

        # -----------------------------
        # Sample points
        # -----------------------------
        sampled_indices = np.linspace(0, n - 1, min(sample_points, n), dtype=int)
        t_sampled = t[sampled_indices]
        t_raw_sampled = t_raw[sampled_indices]

        # -----------------------------
        # State definition
        # -----------------------------
        state_definition = solution_it.filter.trajectory._state_definition
        param_def = [
            state_definition[i + 2][0] for i in range(len(state_definition) - 2)
        ]
        param_size = [
            state_definition[i + 2][1] for i in range(len(state_definition) - 2)
        ]

        # -----------------------------
        # Piecewise model helper
        # -----------------------------
        def _get_piecewise_model():
            prop = getattr(solution_it.filter, "propagator", None)
            if prop is None:
                return None

            fms = getattr(prop, "_force_models", None)
            if fms is None:
                fms = getattr(prop, "force_models", None)
            if fms is None:
                return None

            if isinstance(fms, dict):
                iterable = fms.values()
            else:
                iterable = fms

            # Prefer the built sub-model: it owns the batch grid in raw ET and
            # the authoritative index lookup. The force model wrapping it also
            # answers to `t0`, but that one is rebound to each propagation leg.
            for fm in iterable:
                if hasattr(fm, "_get_active_batch_index"):
                    return fm
                inner = getattr(fm, "force_models", None)
                if isinstance(inner, dict):
                    sub = inner.get("PiecewiseFirstOrderGaussMarkov")
                    if sub is not None:
                        return sub

            for fm in iterable:
                if hasattr(fm, "t0") and (
                    hasattr(fm, "n_batches") or hasattr(fm, "pfogm_n_batches")
                ):
                    return fm
            return None

        pfogm_model = _get_piecewise_model()

        def _active_batch_index(epoch_value, full_size):
            """
            Resolve active batch index for a piecewise stochastic state.
            """
            if pfogm_model is None:
                return 0

            if hasattr(pfogm_model, "_get_active_batch_index"):
                try:
                    return pfogm_model._get_active_batch_index(epoch_value)
                except Exception:
                    pass

            if hasattr(pfogm_model, "batch_length") and hasattr(
                pfogm_model, "n_batches"
            ):
                batch_length = pfogm_model.batch_length
                n_batches = pfogm_model.n_batches
                t0 = pfogm_model.t0
            elif hasattr(pfogm_model, "pfogm_batch_length") and hasattr(
                pfogm_model, "pfogm_n_batches"
            ):
                batch_length = pfogm_model.pfogm_batch_length
                n_batches = pfogm_model.pfogm_n_batches
                t0 = pfogm_model.t0
            else:
                return 0

            # t0 arrives either as an EpochArray (as configured) or as raw
            # ephemeris seconds (as bound to a leg); .et is the raw value.
            t0_et = float(np.asarray(getattr(t0, "et", t0)).ravel()[0])
            k = int(np.floor((float(epoch_value) - t0_et) / float(batch_length)))
            n_batches_from_size = full_size // 3
            n_batches = min(n_batches, n_batches_from_size)
            return max(0, min(k, n_batches - 1))

        def _compress_piecewise(mean_block, sigma_block, epochs_block):
            """
            Convert (Ns, 3*n_batches) to (Ns, 3) by selecting the active batch at each epoch.
            """
            ns, full_size = mean_block.shape
            out_mean = np.zeros((ns, 3))
            out_sigma = np.zeros((ns, 3))

            for ii in range(ns):
                kb = _active_batch_index(epochs_block[ii], full_size)
                i0 = 3 * kb
                i1 = i0 + 3
                out_mean[ii, :] = mean_block[ii, i0:i1]
                out_sigma[ii, :] = sigma_block[ii, i0:i1]

            return out_mean, out_sigma

        # -----------------------------
        # Plot only stochastic parameters
        # -----------------------------
        figures = []
        k0 = 0

        for param, size in zip(param_def, param_size):
            if param not in stochastic_params:
                k0 += size
                continue

            mean_values_sampled = estimated_parameters[sampled_indices, k0 : k0 + size]
            sigma_values_sampled = sigmas[sampled_indices, 6 + k0 : 6 + k0 + size]

            # Standard FOGM: size should be 3
            if param in {"a_fogm", "beta_fogm"}:
                plot_mean = mean_values_sampled
                plot_sigma = sigma_values_sampled
                nrows = size

            # Piecewise FOGM: compress active batch to 3 rows
            elif param in {"a_pfogm", "beta_pfogm"}:
                if size % 3 != 0:
                    raise ValueError(
                        f"{param} should have size multiple of 3, got {size}."
                    )
                plot_mean, plot_sigma = _compress_piecewise(
                    mean_values_sampled, sigma_values_sampled, t_raw_sampled
                )
                nrows = 3

            else:
                k0 += size
                continue

            fig, axs = plt.subplots(nrows, 1, figsize=(10, 3.5 * nrows), sharex=True)

            if not isinstance(axs, np.ndarray):
                axs = np.array([axs])

            component_labels = (
                ["x", "y", "z"] if nrows == 3 else [str(i) for i in range(nrows)]
            )

            for ax in axs.flat:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.minorticks_on()

            plt.subplots_adjust(hspace=0.35)
            fig.suptitle(f"{title} - {param}", fontweight="bold")

            alpha_value_list = [0.6, 0.45, 0.3]

            for j in range(nrows):
                mean_j = plot_mean[:, j]
                sigma_j = plot_sigma[:, j]

                axs[j].errorbar(
                    t_sampled,
                    mean_j,
                    yerr=n_sigma * sigma_j,
                    fmt="o",
                    ecolor=error_point_color,
                    color=error_point_color,
                    capsize=4,
                    label=f"Estimate ± {n_sigma}σ",
                )

                for sigma_level in range(min(n_sigma, 3), 0, -1):
                    alpha_value = alpha_value_list[sigma_level - 1]
                    axs[j].fill_between(
                        t_sampled,
                        mean_j - sigma_level * sigma_j,
                        mean_j + sigma_level * sigma_j,
                        color=sigma_color,
                        alpha=alpha_value,
                        label=f"{sigma_level}σ",
                    )

                ylabel = f"{param}[{component_labels[j]}]" if nrows == 3 else param
                axs[j].set(
                    xlabel=x_label,
                    ylabel=ylabel,
                )
                axs[j].grid(True, which="minor", linestyle="--")
                axs[j].legend(loc="upper left")

            figures.append(fig)
            k0 += size

        plt.tight_layout()
        return figures

    def plot_parameters_errors_norm(
        solution_it: SolutionOD,
        true_trajectory: Trajectory,
        epochs: EpochArray,
        n_sigma: int,
        title: str = "Parameters errors norm",
        t_0: bool = True,
        scale: str = "linxy",
        x_offset_flag: bool = False,
        use_smoothed: bool = True,
    ):
        """
        Plots the state errors and sigma bounds for each component of parameters.

        :param solution_it: The SolutionOD object.
        :type solution_it: SolutionOD

        :param true_trajectory: The true trajectory of the object.
        :type true_trajectory: Trajectory

        :param epochs: The array of epochs.
        :type epochs: EpochArray

        :param n_sigma: The number of standard deviations for the sigma bounds.
        :type n_sigma: int

        :param title: Title for the plot.
        :type title: str

        :param t_0: If True, convert the x-axis to time in days relative to the initial epoch.
        :type t_0: bool

        :param scale: Scale type for plots.
        :type scale: str

        :param x_offset_flag: If True, apply x-axis offset.
        :type x_offset_flag: bool

        :param use_smoothed: If True and smoothed solution is available, uses smoothed covariance.
        :type use_smoothed: bool

        :returns: List of figure objects.
        :rtype: list of matplotlib.figure.Figure
        """
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        error_color = colors[0]

        # Retrieve errors using the smart estimated_trajectory method
        _, _, estimated_parameters = solution_it.estimated_trajectory(
            epochs, use_smoothed=use_smoothed
        )

        # Select covariance source (smoothed if available and requested, otherwise filtered)
        if use_smoothed and solution_it.covariance_smooth is not None:
            covariance_source = solution_it.covariance_smooth
        else:
            covariance_source = solution_it.covariance_est

        # Initialize
        t = epochs.times.values
        n = len(t)

        # Convert time to days relative to the initial epoch if t_0 is True
        if t_0:
            t = (t - t[0]) / (24 * 3600)  # Convert to time in days

        if not solution_it.filter.flag_sequence:
            shape = (
                (n, 1)
                if len(solution_it.filter.trajectory.parameters.shape) == 1
                else (n, solution_it.filter.trajectory.parameters.shape[1])
            )
            err_parameters = np.zeros(shape=shape)
            sigmas = np.zeros(shape=(n, solution_it.filter.n))
        else:
            err_parameters = [
                [
                    np.zeros(shape=(solution_it.filter.legs_n[i] - 6))
                    for _ in range(len(solution_it.filter.legs_epochs[i]))
                ]
                for i in range(len(solution_it.filter.trajectory.parameters))
            ]
            sigmas = [
                [
                    np.zeros(shape=(solution_it.filter.legs_n[i]))
                    for _ in range(len(solution_it.filter.legs_epochs[i]))
                ]
                for i in range(len(solution_it.filter.trajectory.parameters))
            ]

        # Calculate errors and sigmas
        for k in range(n):
            if solution_it.filter.flag_sequence is True:
                idx_leg = next(
                    (
                        (
                            i + 1
                            if epochs[k].times.values == leg[-1]
                            and i != len(solution_it.filter.legs_epochs) - 1
                            else i
                        )
                        for i, leg in enumerate(solution_it.filter.legs_epochs)
                        if (
                            (epochs[k].times.values in leg)
                            if isinstance(leg, np.ndarray)
                            else (epochs[k].times.values == leg)
                        )
                    ),
                    None,
                )
            else:
                idx_leg = None

            # Retrieve true state
            state_nominal = true_trajectory.get_state(
                epoch_input=epochs[k], idx_leg_input=idx_leg
            )

            # Calculate k_offset once before the loop or conditional checks
            k_offset = None
            if solution_it.filter.flag_sequence:
                k_offset = (
                    sum(
                        len(stm)
                        for stm in solution_it.filter.trajectory._STMs[:idx_leg]
                    )
                    - idx_leg
                )

            # Compute parameter errors
            if solution_it.filter.flag_sequence is False:
                err_parameters[k] = (
                    estimated_parameters[k] - state_nominal["parameters"]
                )
            else:
                if state_nominal["parameters"] is not None:
                    err_parameters[idx_leg][k - k_offset] = (
                        estimated_parameters[idx_leg][k - k_offset]
                        - state_nominal["parameters"]
                    )

            # Extract sigmas from covariance (already stored as matrix)
            marginal_variances = np.diag(covariance_source[k])

            if solution_it.filter.flag_sequence is False:
                sigmas[k] = np.sqrt(marginal_variances)
            else:
                sigmas[idx_leg][k - k_offset] = np.sqrt(marginal_variances)

        # Sequence case
        if solution_it.filter.flag_sequence is True:
            figures = []
            for idx_leg, (model, epochs_leg) in enumerate(
                zip(solution_it.filter.legs_model, solution_it.filter.legs_epochs)
            ):
                # Get parameters definition
                state_definition = model.state
                param_def = [
                    state_definition[j + 2][0] for j in range(len(state_definition) - 2)
                ]
                param_size = [
                    state_definition[j + 2][1] for j in range(len(state_definition) - 2)
                ]
                param_units = [
                    state_definition[j + 2][5].quantity.units
                    for j in range(len(state_definition) - 2)
                ]

                # Plot the state errors and sigma bounds for each parameter component
                if param_def:
                    k = 0
                    for i, (param, size) in enumerate(zip(param_def, param_size)):
                        fig, axs = plt.subplots(size, 1, figsize=(5, 3 * size))

                        # Iterate over each subplot and enable minor ticks
                        if not isinstance(axs, np.ndarray):
                            axs = np.array([axs])
                        for ax in axs.flat:
                            # Set minor locators explicitly for both axes
                            ax.xaxis.set_minor_locator(AutoMinorLocator())
                            ax.yaxis.set_minor_locator(AutoMinorLocator())
                            # Enable minor ticks
                            ax.minorticks_on()

                        plt.subplots_adjust(wspace=0.6, hspace=0.3)

                        # Apply scaling
                        for ax in axs.flat:
                            ax = Plotting.apply_scale_to_axes(ax, scale)

                        # Apply x_axis offset
                        t_plot = epochs_leg
                        if x_offset_flag:
                            t_plot = Plotting.plot_with_x_offset(t_plot, x_offset_flag)

                        if size == 1:
                            axs = [axs[0]]

                        for j in range(size):
                            # Plot error using semilogy
                            axs[j].semilogy(
                                t_plot,
                                [
                                    np.abs(elem[k + j])
                                    for elem in err_parameters[idx_leg][:]
                                ],
                                error_color,
                                label="Error",
                            )

                            axs[j].fill_between(
                                t_plot,
                                [
                                    n_sigma * elem[6 + k + j]
                                    for elem in sigmas[idx_leg][:]
                                ],
                                color="red",
                                alpha=0.2,
                                label=f"{n_sigma}σ-bound",
                            )
                            axs[j].set(
                                xlabel=f"Time [{epochs.times.units}]",
                            )
                            axs[j].grid(True, which="minor", linestyle="--")
                            axs[j].legend(loc="upper left")
                        k += size
                        figures.append(fig)
        # Nominal case
        else:
            # Get parameters definition
            state_definition = solution_it.filter.trajectory._state_definition
            param_def = [
                state_definition[i + 2][0] for i in range(len(state_definition) - 2)
            ]
            param_size = [
                state_definition[i + 2][1] for i in range(len(state_definition) - 2)
            ]
            param_units = [
                state_definition[i + 2][5]._units
                for i in range(len(state_definition) - 2)
            ]

            # Plot the state errors and sigma bounds for each parameter component
            k = 0
            figures = []
            for i, (param, size) in enumerate(zip(param_def, param_size)):
                fig, axs = plt.subplots(size, 1, figsize=(5, 3 * size))
                plt.subplots_adjust(wspace=0.6, hspace=0.3)
                if size == 1:
                    axs = [
                        axs
                    ]  # Make sure axs is iterable even if there's only one subplot

                for j in range(size):
                    axs[j].semilogy(
                        t, np.abs(err_parameters[:, k + j]), error_color, label="Error"
                    )

                    axs[j].fill_between(
                        t,
                        np.abs(n_sigma * sigmas[:, 6 + k + j]),
                        color="red",
                        alpha=0.2,
                        label=f"{n_sigma}σ-bound",
                    )
                    axs[j].set(
                        xlabel=(
                            f"Time - $t_0$ [days]"
                            if t_0
                            else f"Time [{epochs.times.units}]"
                        ),
                    )
                    axs[j].grid(True, which="minor", linestyle="--")
                    axs[j].legend(loc="upper left")
                k += size
                figures.append(fig)

        # Plot
        plt.tight_layout()

        return figures

    def plot_postfits_scatter(
        solution_it: SolutionOD,
        epochs: EpochArray,
        dataset_names,
        n_sigma=3,
        scale: str = "linxy",
        x_offset_flag: bool = False,
        use_smoothed: bool = True,
    ):
        """
        Plots the postfit residuals in a scatter plot for various measurements (e.g., range, range rate, OpNav u, OpNav v) dynamically
        based on dataset names. It handles residuals from any ground station (e.g., GS1, GS2) and groups them by measurement type.

        :param solution_it: The SolutionOD object.
        :type solution_it: SolutionOD

        :param epochs: Array of epochs.
        :type epochs: EpochArray

        :param dataset_names: List of dataset names.
        :type dataset_names: list

        :param n_sigma: Number of standard deviations for the sigma lines.
        :type n_sigma: float

        :param scale: Scale type for plots.
        :type scale: str, optional

        :param x_offset_flag: If True, apply x-axis offset.
        :type x_offset_flag: bool, optional

        :param use_smoothed: If True and smoothed solution is available, uses smoothed postfits.
        :type use_smoothed: bool

        :returns: Tuple of (cleaned_dict, list of figures).
        :rtype: tuple(dict, list of matplotlib.figure.Figure)
        """

        # Setting custom color for data point and sigma fill
        colors = Plotting.color_palettes["scb_palette"]
        scatter_point_color = colors[0]
        sigma_color = colors[2]

        # Select postfit source (smoothed if available and requested, otherwise filtered)
        if use_smoothed and solution_it.postfits_smoother is not None:
            postfits_source = solution_it.postfits_smoother
        else:
            postfits_source = solution_it.postfits

        # Initialize
        t = epochs.times.values
        n = len(t)

        # Initialize containers for range, range rate, and opnav u, v residuals and sigmas
        residuals_dict = {"Range": [], "Range Rate": [], "OpNav u": [], "OpNav v": []}
        sigmas_dict = {"Range": [], "Range Rate": [], "OpNav u": [], "OpNav v": []}

        # Classify dataset names by measurement type (ignoring ground station names like GS1, GS2)
        for dataset in dataset_names:
            # Check for measurement types (Range, Range Rate, OpNav u, OpNav v) regardless of ground station
            if "Range Rate" in dataset:
                measurement_type = "Range Rate"
            elif "Range" in dataset:
                measurement_type = "Range"
            elif "OpNav Sample" in dataset:
                measurement_type = "OpNav u"
            elif "OpNav Line" in dataset:
                measurement_type = "OpNav v"
            else:
                raise ValueError(f"Unknown measurement type in dataset name: {dataset}")

            # Retrieve residuals and sigmas for the current dataset
            residuals = [postfits_source[dataset][k][0] for k in range(n)]
            sigmas = [postfits_source[dataset][k][1] for k in range(n)]

            # Append to the appropriate measurement type
            residuals_dict[measurement_type].append((dataset, residuals))
            sigmas_dict[measurement_type].append(sigmas)

        # Determine which measurement types are available
        available_measurements = [
            m for m in residuals_dict if len(residuals_dict[m]) > 0
        ]

        figs = []
        removed_indices_dict = {}

        # Plot residuals for each available measurement type
        for measurement_type in available_measurements:
            fig, ax = plt.subplots(figsize=(15, 8))
            # Apply scaling
            ax = Plotting.apply_scale_to_axes(ax, scale)

            # Make a copy of t for plotting (to avoid modifying original)
            t_plot = t.copy()
            # Apply x_axis offset
            if x_offset_flag:
                t_plot = Plotting.plot_with_x_offset(t_plot, x_offset_flag)

            # Levels for sigma bounds
            sigma_levels = [i for i in range(1, n_sigma + 1)]
            # Sigma - alpha mapping
            alpha_values = {1: 0.6, 2: 0.45, 3: 0.3}

            # Plot each dataset within the current measurement type (for all ground stations)
            for j, (label, residuals) in enumerate(residuals_dict[measurement_type]):
                # Get sigma values for the current dataset
                sigmas = sigmas_dict[measurement_type][j]
                # Fill-between for multiple sigma bounds with varying transparency
                for level in sigma_levels:
                    ax.fill_between(
                        t_plot,
                        -level * np.array(sigmas),
                        level * np.array(sigmas),
                        color=sigma_color,
                        alpha=alpha_values[level],
                        label=f"{level}σ" if j == 0 else "",
                    )

            # Set appropriate y-label and title based on the measurement type
            if measurement_type == "Range":
                ax.set_ylabel(r"$\delta\rho$ [km]")
                title_suffix = (
                    " (Smoothed)"
                    if use_smoothed and solution_it.postfits_smoother is not None
                    else ""
                )
                ax.set_title(f"Range Residuals{title_suffix}", fontweight="bold")
            elif measurement_type == "Range Rate":
                ax.set_ylabel(r"$\delta\dot{\rho}$ [km/s]")
                title_suffix = (
                    " (Smoothed)"
                    if use_smoothed and solution_it.postfits_smoother is not None
                    else ""
                )
                ax.set_title(f"Range Rate Residuals{title_suffix}", fontweight="bold")
            elif measurement_type == "OpNav u":
                ax.set_ylabel(r"$\delta u$ [pixels]")
                title_suffix = (
                    " (Smoothed)"
                    if use_smoothed and solution_it.postfits_smoother is not None
                    else ""
                )
                ax.set_title(f"OpNav u Residuals{title_suffix}", fontweight="bold")
            elif measurement_type == "OpNav v":
                ax.set_ylabel(r"$\delta v$ [pixels]")
                title_suffix = (
                    " (Smoothed)"
                    if use_smoothed and solution_it.postfits_smoother is not None
                    else ""
                )
                ax.set_title(f"OpNav v Residuals{title_suffix}", fontweight="bold")

            ax.set_xlim(
                min(t_plot), max(t_plot)
            )  # Ensure x-axis starts at first data point
            ax.set_xlabel(f"Time [{epochs.times.units}]")
            ax.grid(True, which="minor", linestyle="--")
            ax.legend(loc="upper left")

            plt.tight_layout()

            # Scatter plot for residuals
            ax.scatter(
                t_plot,
                residuals,
                marker=".",
                label=label,
                color=scatter_point_color,
            )
            figs.append(fig)

        cleaned_dict = {
            key: [int(x) for x in value] for key, value in removed_indices_dict.items()
        }
        return cleaned_dict, figs

    def plot_postfits_histogram(
        solution_it: SolutionOD,
        epochs: EpochArray,
        dataset_names,
        n_sigma: int = 3,
        log_scale: bool = True,
        x_offset_flag: bool = False,
        use_smoothed: bool = True,
    ):
        """
        Plots the postfit residuals in a histogram for various measurements (e.g., range, range rate, OpNav u, OpNav v) dynamically
        based on dataset names. It handles residuals from any ground station (e.g., GS1, GS2) and groups them by measurement type.

        :param solution_it: The SolutionOD object.
        :type solution_it: SolutionOD

        :param epochs: Array of epochs.
        :type epochs: EpochArray

        :param dataset_names: List of dataset names.
        :type dataset_names: list

        :param n_sigma: Number of standard deviations for the sigma lines.
        :type n_sigma: int


        :param log_scale: If True, use absolute values for residuals.
        :type log_scale: bool, optional

        :param x_offset_flag: If True, apply x-axis offset.
        :type x_offset_flag: bool, optional

        :param use_smoothed: If True and smoothed solution is available, uses smoothed postfits.
        :type use_smoothed: bool

        :returns: Tuple of (cleaned_dict, list of figures).
        :rtype: tuple(dict, list of matplotlib.figure.Figure)
        """

        # Setting custom color for data point and sigma fill
        colors = Plotting.color_palettes["scb_palette"]
        histogram_color = colors[0]
        sigma_color = colors[2]

        # Select postfit source (smoothed if available and requested, otherwise filtered)
        if use_smoothed and solution_it.postfits_smoother is not None:
            postfits_source = solution_it.postfits_smoother
        else:
            postfits_source = solution_it.postfits

        # Initialize
        t = epochs.times.values
        n = len(t)

        # Initialize containers for range, range rate, and opnav u, v residuals and sigmas
        residuals_dict = {"Range": [], "Range Rate": [], "OpNav u": [], "OpNav v": []}
        sigmas_dict = {"Range": [], "Range Rate": [], "OpNav u": [], "OpNav v": []}

        # Classify dataset names by measurement type (ignoring ground station names like GS1, GS2)
        for dataset in dataset_names:
            # Check for measurement types (Range, Range Rate, OpNav u, OpNav v) regardless of ground station
            if "Range Rate" in dataset:
                measurement_type = "Range Rate"
            elif "Range" in dataset:
                measurement_type = "Range"
            elif "OpNav Sample" in dataset:
                measurement_type = "OpNav u"
            elif "OpNav Line" in dataset:
                measurement_type = "OpNav v"
            else:
                raise ValueError(f"Unknown measurement type in dataset name: {dataset}")

            # Retrieve residuals and sigmas for the current dataset
            residuals = [postfits_source[dataset][k][0] for k in range(n)]
            sigmas = [postfits_source[dataset][k][1] for k in range(n)]

            # Append to the appropriate measurement type
            residuals_dict[measurement_type].append((dataset, residuals))
            sigmas_dict[measurement_type].append(sigmas)

        # Determine which measurement types are available
        available_measurements = [
            m for m in residuals_dict if len(residuals_dict[m]) > 0
        ]

        figs = []
        removed_indices_dict = {}

        # Plot residuals for each available measurement type
        for measurement_type in available_measurements:
            # Plot each dataset within the current measurement type (for all ground stations)
            for j, (label, residuals) in enumerate(residuals_dict[measurement_type]):

                fig, ax = plt.subplots(figsize=(15, 8))

                # If log scale, take absolute values
                if log_scale:
                    residuals = np.abs(residuals)
                data = np.array(residuals)

                ax.set_ylabel("Frequency")

                # Set appropriate x-label and title based on the measurement type
                title_suffix = (
                    " (Smoothed)"
                    if use_smoothed and solution_it.postfits_smoother is not None
                    else ""
                )

                if measurement_type == "Range":
                    ax.set_xlabel(r"$\delta\rho$ [km]")
                    ax.set_title(f"Range Residuals{title_suffix}", fontweight="bold")
                elif measurement_type == "Range Rate":
                    ax.set_xlabel(r"$\delta\dot{\rho}$ [km/s]")
                    ax.set_title(
                        f"Range Rate Residuals{title_suffix}", fontweight="bold"
                    )
                elif measurement_type == "OpNav u":
                    ax.set_xlabel(r"$\delta u$ [pixels]")
                    ax.set_title(f"OpNav u Residuals{title_suffix}", fontweight="bold")
                elif measurement_type == "OpNav v":
                    ax.set_xlabel(r"$\delta v$ [pixels]")
                    ax.set_title(f"OpNav v Residuals{title_suffix}", fontweight="bold")

                ax.grid(True, which="minor", linestyle="--")
                ax.minorticks_on()
                ax.set_axisbelow(True)
                ax.legend(loc="upper left")
                plt.tight_layout()

                ax.hist(
                    data,
                    bins=20,
                    color=histogram_color,
                    label=label,
                    edgecolor="black",
                )

                figs.append(fig)
                plt.show()

        cleaned_dict = {
            key: [int(x) for x in value] for key, value in removed_indices_dict.items()
        }
        return cleaned_dict, figs

    def plot_postfits_asynchronous(
        solution_it: SolutionOD,
        dataset_names,
        n_sigma,
        title="Postfit Residuals Asynchronous",
        time_vectors=None,
        scale: str = "linxy",
        x_offset_flag: bool = False,
        use_smoothed: bool = True,
    ):
        """
        Plots the postfit residuals for various measurements (e.g., range, range rate, OpNav u, OpNav v) dynamically
        based on dataset names. It handles residuals from any ground station (e.g., GS1, GS2) and groups them by measurement type.
        Datasets can have different lengths, and the x-axis will represent either the time vector or the number of measurements
        if no time vector is provided.

        :param solution_it: The SolutionOD object.
        :type solution_it: SolutionOD

        :param dataset_names: List of dataset names.
        :type dataset_names: list

        :param n_sigma: Number of standard deviations for the sigma lines.
        :type n_sigma: float

        :param title: Title of the plot.
        :type title: str

        :param time_vectors: List of time vectors for each measurement type. If only one time vector is provided, it will be used for all measurement types.
                            If not provided, the x-axis will represent the number of measurements.
        :type time_vectors: list, optional

        :param scale: Scale type for plots.
        :type scale: str, optional

        :param x_offset_flag: If True, apply x-axis offset.
        :type x_offset_flag: bool, optional

        :param use_smoothed: If True and smoothed solution is available, uses smoothed postfits.
        :type use_smoothed: bool
        """
        # Predefined list of colors, one per plot
        colors = Plotting.color_palettes["matplot_default_palette"]
        error_color = colors[0]

        # Select postfit source (smoothed if available and requested, otherwise filtered)
        if use_smoothed and solution_it.postfits_smoother is not None:
            postfits_source = solution_it.postfits_smoother
            title_suffix = " (Smoothed)"
        else:
            postfits_source = solution_it.postfits
            title_suffix = ""

        # Initialize containers for range, range rate, and opnav u, v residuals and sigmas
        residuals_dict = {"Range": [], "Range Rate": [], "OpNav u": [], "OpNav v": []}
        sigmas_dict = {"Range": [], "Range Rate": [], "OpNav u": [], "OpNav v": []}

        # Classify dataset names by measurement type (ignoring ground station names like GS1, GS2)
        for dataset in dataset_names:
            # Check for measurement types (Range, Range Rate, OpNav u, OpNav v) regardless of ground station
            if "Range Rate" in dataset:
                measurement_type = "Range Rate"
            elif "Range" in dataset:
                measurement_type = "Range"
            elif "OpNav Sample" in dataset:
                measurement_type = "OpNav u"
            elif "OpNav Line" in dataset:
                measurement_type = "OpNav v"
            else:
                raise ValueError(f"Unknown measurement type in dataset name: {dataset}")

            # Retrieve residuals and sigmas for the current dataset
            residuals = [
                postfits_source[dataset][k][0]
                for k in range(len(postfits_source[dataset]))
            ]
            sigmas = [
                postfits_source[dataset][k][1]
                for k in range(len(postfits_source[dataset]))
            ]

            # Append to the appropriate measurement type
            residuals_dict[measurement_type].append((dataset, residuals))
            sigmas_dict[measurement_type].append(sigmas)

        # Determine which measurement types are available
        available_measurements = [
            m for m in residuals_dict if len(residuals_dict[m]) > 0
        ]

        # Check if only one time vector is provided
        single_time_vector = isinstance(time_vectors, list) and len(time_vectors) == 1

        # Create subplots dynamically based on available measurements
        num_plots = len(available_measurements)
        fig, axs = plt.subplots(num_plots, 1, figsize=(8, 2 * num_plots), sharex=False)

        # Iterate over each subplot and enable minor ticks
        if num_plots == 1:
            axs = [axs]  # Ensure axs is always a list for consistent iteration

        for ax in axs:
            # Set minor locators explicitly for both axes
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            # Enable minor ticks
            ax.minorticks_on()

        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        fig.suptitle(f"{title}{title_suffix}", fontweight="bold")

        # Apply scaling
        for ax in axs:
            Plotting.apply_scale_to_axes(ax, scale)

        # Note: x_offset_flag logic moved to plotting section where x_values are defined

        # Plot residuals for each available measurement type
        for i, measurement_type in enumerate(available_measurements):
            ax = axs[i]
            color = colors[
                i % len(colors)
            ]  # Assign a unique color for each measurement type plot

            # Retrieve time vector for this measurement type
            if single_time_vector:
                x_values = time_vectors[0]
            elif time_vectors and len(time_vectors) > i:
                x_values = time_vectors[i]
            else:
                x_values = None  # Will use measurement number as x-axis if no time vector is available

            # Apply x_axis offset if time vector is available
            if x_values is not None and x_offset_flag:
                x_values = Plotting.plot_with_x_offset(x_values, x_offset_flag)

            # Plot each dataset within the current measurement type (for all ground stations)
            for j, (label, residuals) in enumerate(residuals_dict[measurement_type]):
                # Use time vector or default to the number of measurements
                if x_values is not None:
                    ax.scatter(
                        x_values[: len(residuals)],
                        residuals,
                        marker=".",
                        label=label,
                        color=color,
                    )
                    sigmas = sigmas_dict[measurement_type][j]
                    ax.plot(
                        x_values[: len(sigmas)],
                        n_sigma * np.array(sigmas),
                        "r--",
                        label=str(n_sigma) + "σ" if j == 0 else "",
                    )
                    ax.plot(x_values[: len(sigmas)], -n_sigma * np.array(sigmas), "r--")
                else:
                    x_vals_default = np.arange(len(residuals))
                    ax.scatter(
                        x_vals_default, residuals, marker=".", label=label, color=color
                    )
                    sigmas = sigmas_dict[measurement_type][j]
                    ax.plot(
                        x_vals_default,
                        n_sigma * np.array(sigmas),
                        "r--",
                        label=str(n_sigma) + "σ" if j == 0 else "",
                    )
                    ax.plot(x_vals_default, -n_sigma * np.array(sigmas), "r--")

            # Set appropriate y-label and title based on the measurement type
            if measurement_type == "Range":
                ax.set_ylabel(r"$\delta\rho$ [km]")
                ax.set_title("Range Residuals")
            elif measurement_type == "Range Rate":
                ax.set_ylabel(r"$\delta\dot{\rho}$ [km/s]")
                ax.set_title("Range Rate Residuals")
            elif measurement_type == "OpNav u":
                ax.set_ylabel(r"$\delta u$ [pixels]")
                ax.set_title("OpNav Sample Residuals")
            elif measurement_type == "OpNav v":
                ax.set_ylabel(r"$\delta v$ [pixels]")
                ax.set_title("OpNav Line Residuals")

            ax.grid()
            ax.legend(loc="upper left")

        # Set the x-label only on the bottom plot
        if time_vectors:
            axs[-1].set_xlabel("Time [TDB]")
        else:
            axs[-1].set_xlabel("Measurement Number [-]")

        plt.tight_layout()
        plt.show()

    def plot_postfits_asynchronous_with_hist(
        solution_it,
        dataset_names,
        n_sigma=3,
        title="",
        time_vectors=None,
        scale="linxy",
        x_offset_flag=False,
        plot_sigmas=False,
        iter_num=None,
        use_smoothed=True,
    ):
        """
        Plots postfit residuals with histograms for various measurement types.

        :param solution_it: The SolutionOD object.
        :type solution_it: SolutionOD

        :param dataset_names: List of dataset names.
        :type dataset_names: list

        :param n_sigma: Number of standard deviations for the sigma lines.
        :type n_sigma: int

        :param title: Title of the plot.
        :type title: str

        :param time_vectors: List of time vectors for each measurement type.
        :type time_vectors: list, optional

        :param scale: Scale type for plots.
        :type scale: str

        :param x_offset_flag: If True, apply x-axis offset.
        :type x_offset_flag: bool

        :param plot_sigmas: If True, plot sigma bounds.
        :type plot_sigmas: bool

        :param iter_num: Iteration number for file naming.
        :type iter_num: int, optional

        :param use_smoothed: If True and smoothed solution is available, uses smoothed postfits.
        :type use_smoothed: bool
        """
        # Use SCB palette
        colors = Plotting.color_palettes[
            "scb_palette"
        ]  # ['#4067BE', '#15788c', '#E4B542', '#000000']

        # Select postfit source (smoothed if available and requested, otherwise filtered)
        if use_smoothed and solution_it.postfits_smoother is not None:
            postfits_source = solution_it.postfits_smoother
            title_suffix = " (Smoothed)"
        else:
            postfits_source = solution_it.postfits
            title_suffix = ""

        # Initialize dictionaries for residuals and sigmas
        residuals_dict = {"Range": [], "Doppler": [], "OpNav u": [], "OpNav v": []}
        sigmas_dict = {"Range": [], "Doppler": [], "OpNav u": [], "OpNav v": []}
        label_dict = {
            "Range": r"$\delta\rho$ [RU]",
            "Doppler": r"$\delta\dot{\rho}$ [Hz]",
            "OpNav u": r"$\delta u$ [pixels]",
            "OpNav v": r"$\delta v$ [pixels]",
        }

        # Extract residuals and sigmas
        for dataset in dataset_names:
            if "Range Rate" in dataset:
                measurement_type = "Doppler"
            elif "Range" in dataset:
                measurement_type = "Range"
            elif "OpNav Sample" in dataset:
                measurement_type = "OpNav u"
            elif "OpNav Line" in dataset:
                measurement_type = "OpNav v"
            else:
                raise ValueError(f"Unknown measurement type in dataset name: {dataset}")

            residuals = [
                postfits_source[dataset][k][0]
                for k in range(len(postfits_source[dataset]))
            ]
            sigmas = [
                postfits_source[dataset][k][1]
                for k in range(len(postfits_source[dataset]))
            ]
            residuals_dict[measurement_type].append((dataset, residuals))
            sigmas_dict[measurement_type].append(sigmas)

        # Filter available measurement types
        available_measurements = [
            m for m in residuals_dict if len(residuals_dict[m]) > 0
        ]
        single_time_vector = isinstance(time_vectors, list) and len(time_vectors) == 1
        num_plots = len(available_measurements)

        # Set up figure and grid
        fig = plt.figure(figsize=(12, 2.5 * num_plots))
        outer_grid = gridspec.GridSpec(
            num_plots, 2, width_ratios=[3, 1], wspace=0.35, hspace=0.4
        )

        for i, measurement_type in enumerate(available_measurements):
            ax_scatter = plt.Subplot(fig, outer_grid[i, 0])
            ax_hist = plt.Subplot(fig, outer_grid[i, 1], sharey=ax_scatter)
            fig.add_subplot(ax_scatter)
            fig.add_subplot(ax_hist)

            # Plot scatter for each dataset
            for j, (label, residuals) in enumerate(residuals_dict[measurement_type]):
                x_values = (
                    time_vectors[0]
                    if single_time_vector
                    else (
                        time_vectors[i]
                        if time_vectors and len(time_vectors) > i
                        else np.arange(len(residuals))
                    )
                )
                x_values = Plotting.plot_with_x_offset(
                    x_values, x_offset_flag
                )  # Apply x-offset if flagged
                ax_scatter.scatter(
                    x_values[: len(residuals)],
                    residuals,
                    marker=".",
                    label=label,
                    color=colors[0],
                    s=40,
                    alpha=0.7,
                )

                if plot_sigmas:
                    sigmas = sigmas_dict[measurement_type][j]
                    ax_scatter.plot(
                        x_values[: len(sigmas)],
                        n_sigma * np.array(sigmas),
                        linestyle="--",
                        color=colors[-1],
                        label=f"+{n_sigma}σ" if j == 0 else "",
                    )
                    ax_scatter.plot(
                        x_values[: len(sigmas)],
                        -n_sigma * np.array(sigmas),
                        linestyle="--",
                        color=colors[-1],
                    )

            # Apply scaling and styling
            ax_scatter = Plotting.apply_scale_to_axes(ax_scatter, scale)
            ax_scatter.grid(True)
            ax_scatter.legend(loc="upper left", fontsize=10)
            ax_scatter.set_ylabel(
                label_dict[measurement_type], fontsize=plt.rcParams["axes.labelsize"]
            )

            # Histogram
            all_resids = np.concatenate(
                [r for _, r in residuals_dict[measurement_type]]
            )
            mu, std = np.mean(all_resids), np.std(all_resids)
            ax_hist.hist(
                all_resids,
                bins=40,
                orientation="horizontal",
                color=colors[2],
                alpha=0.6,
                density=True,
                edgecolor=colors[3],
                linewidth=0.5,
            )
            x_vals = np.linspace(min(all_resids), max(all_resids), 200)
            ax_hist.plot(
                norm.pdf(x_vals, mu, std),
                x_vals,
                color=colors[1],
                linestyle="--",
                linewidth=1.2,
            )
            ax_hist.text(
                0.95,
                0.95,
                f"$\\mu$ = {mu:.3f}\n$\\sigma$ = {std:.3f}",
                transform=ax_hist.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
            )
            ax_hist.grid(True)

            # Set y-limits for sigma plots
            if plot_sigmas:
                max_sigma = max([np.max(sig) for sig in sigmas_dict[measurement_type]])
                ylim = 3 * n_sigma * max_sigma
                ax_scatter.set_ylim(-ylim, ylim)

            # Set x-label for the last plot
            if i == num_plots - 1:
                ax_scatter.set_xlabel(
                    "Time [TDB]" if time_vectors else "Measurement Number [-]",
                    fontsize=plt.rcParams["axes.labelsize"],
                )

        if title:
            fig.suptitle(
                f"{title}{title_suffix}",
                fontsize=plt.rcParams["axes.titlesize"],
                fontweight=plt.rcParams["axes.titleweight"],
            )

        # Save to HTML (commented out, but ready to use)
        # import mpld3
        # html_str = mpld3.fig_to_html(fig)
        # if iter_num == None:
        #    with open("postfits.html", "w") as f:
        #        f.write(html_str)
        #        fig.savefig("postfits.pdf", bbox_inches="tight")
        # else:
        #    with open(f"postfits_it{iter_num}.html", "w") as f:
        #        f.write(html_str)
        #        fig.savefig(f"postfits_it{iter_num}.pdf", bbox_inches="tight")

        return fig

    def plot_covariance_est(
        solution_it,
        n_sigma,
        title: str = "Covariance History",
        scale: str = "linxy",
        x_offset_flag: bool = False,
        use_smoothed: bool = True,
    ):
        """
        Plots the covariance history.

        :param solution_it: The SolutionOD object.
        :type solution_it: SolutionOD

        :param n_sigma: The number of standard deviations to plot.
        :type n_sigma: int

        :param title: Title of the plot.
        :type title: str

        :param scale: Scale type for plots.
        :type scale: str, optional

        :param x_offset_flag: If True, apply x-axis offset.
        :type x_offset_flag: bool, optional

        :param use_smoothed: If True and smoothed solution is available, uses smoothed covariance.
        :type use_smoothed: bool

        :returns: Matplotlib figure object.
        :rtype: matplotlib.figure.Figure
        """
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        error_color = colors[0]

        # Select covariance source (smoothed if available and requested, otherwise filtered)
        if use_smoothed and solution_it.covariance_smooth is not None:
            covariance_source = solution_it.covariance_smooth
            title_suffix = " (Smoothed)"
        else:
            covariance_source = solution_it.covariance_est
            title_suffix = ""

        # Initialize
        n = len(covariance_source)
        sigmas = np.zeros(shape=(n, solution_it.filter.n))

        # Retrieve sigmas (covariances already stored as matrices)
        for k in range(n):
            marginal_variances = np.diag(covariance_source[k])
            sigmas[k] = np.sqrt(marginal_variances)

        # Plot the covariance history
        t = [x[0] for x in solution_it.filter.measurement_data]
        fig, axs = plt.subplots(
            2,
            3,
            figsize=(8, 8),
            sharex=False,
        )

        # Iterate over each subplot and enable minor ticks
        for ax in axs.flat:
            # Set minor locators explicitly for both axes
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            # Enable minor ticks
            ax.minorticks_on()

        plt.subplots_adjust(wspace=0.5, hspace=0.2)
        # Add title
        fig.suptitle(f"{title}{title_suffix}", fontweight="bold")

        # Apply scaling
        for row in axs:
            for ax in row:
                ax = Plotting.apply_scale_to_axes(ax, scale)

        # Apply x_axis offset
        if x_offset_flag:
            t = Plotting.plot_with_x_offset(t, x_offset_flag)

        # Position x
        axs[0, 0].semilogy(t, n_sigma * sigmas[:, 0], "r--", label=f"{n_sigma}σ-bound")
        axs[0, 0].fill_between(
            t, -n_sigma * sigmas[:, 0], n_sigma * sigmas[:, 0], alpha=0.25, color="red"
        )
        axs[0, 0].set(xlabel="Time [TBD]", ylabel=r"$\varepsilon_x$ [km]")
        axs[0, 0].grid(True, which="minor", linestyle="--")
        axs[0, 0].legend(loc="upper left")

        # Position y
        axs[0, 1].semilogy(t, n_sigma * sigmas[:, 1], "r--", label=f"{n_sigma}σ-bound")
        axs[0, 1].fill_between(
            t, -n_sigma * sigmas[:, 1], n_sigma * sigmas[:, 1], alpha=0.25, color="red"
        )
        axs[0, 1].set(xlabel="Time [TBD]", ylabel=r"$\varepsilon_y$ [km]")
        axs[0, 1].grid(True, which="minor", linestyle="--")
        axs[0, 1].legend(loc="upper left")

        # Position z
        axs[0, 2].semilogy(t, n_sigma * sigmas[:, 2], "r--", label=f"{n_sigma}σ-bound")
        axs[0, 2].fill_between(
            t, -n_sigma * sigmas[:, 2], n_sigma * sigmas[:, 2], alpha=0.25, color="red"
        )
        axs[0, 2].set(xlabel="Time [TBD]", ylabel=r"$\varepsilon_z$ [km]")
        axs[0, 2].grid(True, which="minor", linestyle="--")
        axs[0, 2].legend(loc="upper left")

        # Velocity x
        axs[1, 0].semilogy(t, n_sigma * sigmas[:, 3], "r--", label=f"{n_sigma}σ-bound")
        axs[1, 0].fill_between(
            t, -n_sigma * sigmas[:, 3], n_sigma * sigmas[:, 3], alpha=0.25, color="red"
        )
        axs[1, 0].set(xlabel="Time [TBD]", ylabel=r"$\varepsilon_{\dot{x}}$ [km/s]")
        axs[1, 0].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
        axs[1, 0].grid(True, which="minor", linestyle="--")
        axs[1, 0].legend(loc="upper left")

        # Velocity y
        axs[1, 1].semilogy(t, n_sigma * sigmas[:, 4], "r--", label=f"{n_sigma}σ-bound")
        axs[1, 1].fill_between(
            t, -n_sigma * sigmas[:, 4], n_sigma * sigmas[:, 4], alpha=0.25, color="red"
        )
        axs[1, 1].set(xlabel="Time [TBD]", ylabel=r"$\varepsilon_{\dot{y}}$ [km/s]")
        axs[1, 1].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
        axs[1, 1].grid(True, which="minor", linestyle="--")
        axs[1, 1].legend(loc="upper left")

        # Velocity z
        axs[1, 2].semilogy(t, n_sigma * sigmas[:, 5], "r--", label=f"{n_sigma}σ-bound")
        axs[1, 2].fill_between(
            t, -n_sigma * sigmas[:, 5], n_sigma * sigmas[:, 5], alpha=0.25, color="red"
        )
        axs[1, 2].set(xlabel="Time [TBD]", ylabel=r"$\varepsilon_{\dot{z}}$ [km/s]")
        axs[1, 2].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
        axs[1, 2].grid(True, which="minor", linestyle="--")
        axs[1, 2].legend(loc="upper left")

        return fig

    @staticmethod
    def plot_corner(
        samples,
        labels=None,
        mean=None,
        covariance=None,
        truths=None,
        indices=None,
        n_sigma=(1, 2, 3),
        title=None,
        **corner_kwargs,
    ):
        """
        Corner plot of posterior samples, optionally against a Gaussian.

        Every pair of dimensions gets a 2-D histogram with contours, and every
        dimension a marginal histogram, through :func:`corner.corner`. The
        contours enclose the probability a 2-D Gaussian holds within
        :math:`n` standard deviations, :math:`1 - e^{-n^2/2}`, so with
        ``covariance`` the overlaid :math:`n`-sigma ellipses coincide with them
        when the samples are Gaussian.

        Parameters
        ----------
        samples : MonteCarloSamples or array_like
            Samples, e.g. :attr:`MCMC.samples`, or an array of shape
            ``(n_samples, n_dimensions)``. The weights and labels of a
            :class:`~scarabaeus.MonteCarloSamples` are used.
        labels : sequence of str, optional
            Name of each dimension of ``samples``. Defaults to the labels of a
            :class:`~scarabaeus.MonteCarloSamples`, otherwise none.
        mean : array_like, optional
            Centre of the Gaussian overlay, shape ``(n_dimensions,)``.
            Defaults to the weighted sample mean when ``covariance`` is given.
        covariance : array_like, optional
            Covariance of the Gaussian overlay (e.g. the filter's),
            shape ``(n_dimensions, n_dimensions)``. Its ``n_sigma`` ellipses
            are drawn on the pairs and its density on the marginals. Defaults
            to no overlay.
        truths : array_like, optional
            Reference value of each dimension, marked by lines. Defaults to
            none.
        indices : sequence of int, optional
            Dimensions to plot. Defaults to all.
        n_sigma : sequence of float, optional
            Contour and ellipse levels, in standard deviations. Defaults to
            ``(1, 2, 3)``.
        title : str, optional
            Figure title. Defaults to none.
        **corner_kwargs
            Passed to :func:`corner.corner` (e.g. ``bins``, ``smooth``,
            ``show_titles``, ``quantiles``, ``fig``).

        Returns
        -------
        matplotlib.figure.Figure
            The corner plot.

        Raises
        ------
        CornerSamplesError
            If ``samples`` is not a 2-D array with at least two samples, or
            ``indices`` picks dimensions it does not have.
        CornerOverlayError
            If ``labels``, ``truths``, ``mean`` or ``covariance`` does not
            match the dimensions of ``samples``.

        Examples
        --------
        >>> solution = scb.MCMC(lsb).sample(seed=0)
        >>> fig = scb.Plotting.plot_corner(
        ...     solution.filter.samples,
        ...     covariance=lsb_solution.covariance_est[0],
        ... )
        """
        weights = corner_kwargs.pop("weights", None)
        if isinstance(samples, MonteCarloSamples):
            labels = samples.labels if labels is None else labels
            weights = samples.weights if weights is None else weights
            samples = samples.samples
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[0] < 2:
            raise Plotting.CornerSamplesError(
                f"Samples must be a 2-D array with at least two samples; got shape "
                f"{samples.shape}."
            )
        n_dim = samples.shape[1]
        indices = np.arange(n_dim) if indices is None else np.asarray(indices, int)
        if indices.ndim != 1 or np.any((indices < 0) | (indices >= n_dim)):
            raise Plotting.CornerSamplesError(
                f"Indices must pick among the {n_dim} dimensions; got {indices}."
            )

        def _pick(values, name, shape):
            if values is None:
                return None
            values = np.asarray(values)
            if values.shape != shape:
                raise Plotting.CornerOverlayError(
                    f"{name} must have shape {shape}; got {values.shape}."
                )
            return values[np.ix_(indices, indices)] if values.ndim == 2 else values[
                indices
            ]

        labels = _pick(labels, "Labels", (n_dim,))
        truths = _pick(truths, "Truths", (n_dim,))
        covariance = _pick(covariance, "Covariance", (n_dim, n_dim))
        if covariance is not None and mean is None:
            w = np.full(samples.shape[0], 1.0) if weights is None else weights
            mean = np.asarray(w, dtype=float) @ samples / np.sum(w)
        mean = _pick(mean, "Mean", (n_dim,))
        samples = samples[:, indices]

        n_sigma = np.atleast_1d(np.asarray(n_sigma, dtype=float))
        fig = corner.corner(
            samples,
            weights=weights,
            labels=None if labels is None else list(labels),
            truths=None if truths is None else np.asarray(truths, dtype=float),
            levels=1.0 - np.exp(-0.5 * n_sigma**2),
            plot_datapoints=corner_kwargs.pop("plot_datapoints", False),
            **corner_kwargs,
        )

        if covariance is not None:
            covariance = np.asarray(covariance, dtype=float)
            mean = np.asarray(mean, dtype=float)
            n_plot = indices.size
            axes = np.asarray(fig.axes).reshape(n_plot, n_plot)
            overlay_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
            total = samples.shape[0] if weights is None else float(np.sum(weights))
            bins = corner_kwargs.get("bins", 20)
            factor = corner_kwargs.get("hist_bin_factor", 1)
            for i in range(n_plot):
                ax = axes[i, i]
                lo, hi = ax.get_xlim()
                b = bins[i] if np.ndim(bins) else bins
                f = factor[i] if np.ndim(factor) else factor
                n_bins = max(1, int(np.round(f * b)))
                x = np.linspace(lo, hi, 200)
                sigma = np.sqrt(covariance[i, i])
                ax.plot(
                    x,
                    total * (hi - lo) / n_bins * norm.pdf(x, mean[i], sigma),
                    color=overlay_color,
                )
                for j in range(i):
                    sub = covariance[np.ix_([j, i], [j, i])]
                    eigvals, eigvecs = np.linalg.eigh(sub)
                    eigvals = np.clip(eigvals, 0.0, None)
                    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
                    for n in n_sigma:
                        axes[i, j].add_patch(
                            Ellipse(
                                (mean[j], mean[i]),
                                width=2.0 * n * np.sqrt(eigvals[1]),
                                height=2.0 * n * np.sqrt(eigvals[0]),
                                angle=angle,
                                fill=False,
                                color=overlay_color,
                                linestyle="--",
                            )
                        )

        if title is not None:
            fig.suptitle(title, fontweight="bold")
        return fig

    # endregion SolutionOD #
    # ----------------------#

    # ------------------#
    # region BPlane    #
    # ------------------#
    def plot_covariance(
        bplane: Bplane, P_bplane_parameters, title: str = "B-plane Covariance"
    ):
        """
        Given a B-plane covariance, plot a graphical representation. It assumes no correlation between
        position in the B-plane and linearized time of flight.

        :param bplane: The Bplane Object
        :type bplane: Bplane

        :param P_bplane_parameters: The B-plane covariance matrix.
        :type P_bplane_parameters: numpy.ndarray


        """

        # Setting custom color for sigma fill and error line
        colors = Plotting.color_palettes["cu_palette"]
        ellipse_color = colors[0]
        target_color = colors[1]

        # Plot the B-plane
        plt.figure(figsize=(8, 8))
        # plt.plot(
        #    [0, self._bt_ref.values],
        #    [0, self._br_ref.values],
        #    "k-",
        #    label=r"$\mathbf{B}$ vector",
        # )
        # plt.plot(
        #    0,
        #    0,
        #    marker="D",
        #    color="blue",
        #    linestyle="",
        #    label=r"Fly-by Body",
        # )

        # Plot the covariance ellipse
        eigenvalues, eigenvectors = np.linalg.eig(P_bplane_parameters[:2, :2].values)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0].imag, eigenvectors[0, 0].real))
        width, height = 2 * np.sqrt(eigenvalues.real)
        ellipse = Ellipse(
            xy=(bplane._bt_ref.values, bplane._br_ref.values),
            width=width,
            height=height,
            angle=angle,
            color=ellipse_color,
            alpha=0.6,
            label=r"1$\sigma$ Ellipse",
        )
        plt.gca().add_patch(ellipse)
        plt.scatter(
            bplane._bt_ref.values,
            bplane._br_ref.values,
            marker="D",
            color=target_color,
            label=r"Target",
            edgecolors="black",
        )
        # Add a title to the plot
        plt.title(title, fontweight="bold")

        # Set plot limits and labels
        plt.xlabel(r"$\mathbf{\hat{T}}$ [km]")
        plt.gca().invert_yaxis()
        plt.ylabel(r"$\mathbf{\hat{R}}$ [km]")
        plt.legend(loc="upper right")

        # Show plot
        plt.grid(True)
        # plt.gca().set_aspect("equal", adjustable="box")
        plt.show()

    # Method to plot multiple covariances
    def plot_multiple_covariance(
        bplane: Bplane,
        sc_states,
        sc_covariances,
        labels=None,
        zoom_flag=True,
        ast_covariances=None,
        scale: str = "linxy",
        x_offset_flag: bool = False,
    ):
        """
        Plot multiple covariances corresponding to different spacecraft positions and their covariances in the B-plane.

        :param bplane: The Bplane Object
        :type bplane: Bplane

        :param sc_states: List of relative spacecraft states (position and velocity in J2000) at different epochs.
        :type sc_states: list of numpy.ndarray

        :param sc_covariances: List of covariance matrices for the spacecraft states in J2000.
        :type sc_covariances: list of numpy.ndarray

        :param labels: List of labels for each ellipse (optional).
        :type labels: list of str

        :param zoom_flag: Flag to indicate whether to apply zoom on the last ellipse and hide axes when zoomed.
        :type zoom_flag: bool

        :param scale: Scale type for plots.
        :type scale: str, optional

        :param x_offset_flag: If True, apply x-axis offset.
        :type x_offset_flag: bool, optional

        :returns: None
        :rtype: None
        """

        # Predefined list of colors, one per plot
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        # Check if labels are provided
        if labels is None:
            labels = [f"Ellipse {i+1}" for i in range(len(sc_states))]

        # First plot: One subplot for each covariance ellipse
        num_ellipses = len(sc_states)
        fig1, axs = plt.subplots(1, num_ellipses, figsize=(15, 8))

        # Ensure axs is always iterable
        if num_ellipses == 1:
            axs = [axs]

        # Apply scaling
        axs = [Plotting.apply_scale_to_axes(ax, scale) for ax in axs]

        for i, (state, cov) in enumerate(zip(sc_states, sc_covariances)):
            # Compute B-plane parameters for this state
            bpar_vec = bplane.compute_parameters(sc_rel_state_j2000=state)

            # Compute B-plane covariance
            if ast_covariances is not None:
                P_bplane_parameters = bplane.target_spacecraft_covariance(
                    np.block(
                        [
                            [cov, np.zeros((6, 6))],
                            [np.zeros((6, 6)), ast_covariances[i]],
                        ]
                    ),
                    state,
                )
            else:
                P_bplane_parameters = bplane.target_covariance(cov, state)

            # Extract the covariance matrix for the B-plane position (bt, br)
            P_bplane_2D = P_bplane_parameters[:2, :2].values

            # Eigenvalue decomposition of covariance matrix
            eigenvalues, eigenvectors = np.linalg.eig(P_bplane_2D)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 6 * np.sqrt(eigenvalues)  # 3-sigma ellipse dimensions

            # Apply x_axis offset
            if x_offset_flag:
                bpar_vec[0].values = Plotting.plot_with_x_offset(
                    bpar_vec[0].values, x_offset_flag
                )
            # Plot each ellipse in its own subplot
            ellipse = Ellipse(
                xy=(bpar_vec[0].values, bpar_vec[1].values),
                width=width,
                height=height,
                angle=angle,
                edgecolor=colors[i % len(colors)],
                facecolor="none",
                label="3σ Ellipse",
            )
            axs[i].add_patch(ellipse)
            axs[i].plot(
                bpar_vec[0].values,
                bpar_vec[1].values,
                "o",
                color=colors[i],
                label=r"Target",
            )
            axs[i].set_xlabel(r"$\mathbf{\hat{T}}$ [km]")
            axs[i].invert_yaxis()
            axs[i].set_ylabel(r"$\mathbf{\hat{R}}$ [km]")
            axs[i].grid(True)
            axs[i].set_title(labels[i])
            axs[i].ticklabel_format(useOffset=False, style="plain")
            axs[i].legend(loc="upper right")
            axs[i].set_aspect("equal", "datalim")
            axs[i].tick_params(axis="x", rotation=45)  # Rotate x-axis ticks
        plt.subplots_adjust(wspace=0.3)  # Increase horizontal space between subplots
        plt.tight_layout()

        # Second plot: All ellipses together around (self._bt_ref.values, self._br_ref.values)
        plt.figure(figsize=(10, 10))
        ax = plt.gca()  # Main axis
        plt.grid(True)

        # Plot all ellipses on the main plot
        for i, (state, cov) in enumerate(zip(sc_states, sc_covariances)):
            # Compute B-plane parameters for this state
            bpar_vec = bplane.compute_parameters(sc_rel_state_j2000=state)

            # Compute B-plane covariance

            # Compute B-plane covariance
            if ast_covariances is not None:
                P_bplane_parameters = bplane.target_spacecraft_covariance(
                    np.block(
                        [
                            [cov, np.zeros((6, 6))],
                            [np.zeros((6, 6)), ast_covariances[i]],
                        ]
                    ),
                    state,
                )
            else:
                P_bplane_parameters = bplane.target_covariance(cov, state)

            # Extract the covariance matrix for the B-plane position (bt, br)
            P_bplane_2D = P_bplane_parameters[:2, :2].values

            # Eigenvalue decomposition of covariance matrix
            eigenvalues, eigenvectors = np.linalg.eig(P_bplane_2D)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 6 * np.sqrt(eigenvalues)  # 3-sigma ellipse dimensions

            # Plot all ellipses together around reference point
            ellipse = Ellipse(
                xy=(bplane._bt_ref.values, bplane._br_ref.values),
                width=width,
                height=height,
                angle=angle,
                edgecolor=colors[i % len(colors)],
                facecolor="none",
                label=f"{labels[i]} 3σ Ellipse",
            )
            ax.add_patch(ellipse)

        # Apply x_axis offset
        if x_offset_flag:
            bplane._bt_ref.values = Plotting.plot_with_x_offset(
                bplane._bt_ref.values, x_offset_flag
            )

        # Plot the reference point
        plt.scatter(
            bplane._bt_ref.values,
            bplane._br_ref.values,
            marker="o",
            color=colors[i + 1],
            edgecolors="black",
            label="CA Target",
        )

        if zoom_flag:
            # Zoomed inset plot for the last ellipse
            axins = zoomed_inset_axes(ax, zoom=10, loc="lower right")
            bpar_vec = bplane.compute_parameters(sc_rel_state_j2000=sc_states[-1])
            if ast_covariances is not None:
                P_bplane_parameters = bplane.target_spacecraft_covariance(
                    np.block(
                        [
                            [sc_covariances[-1], np.zeros((6, 6))],
                            [np.zeros((6, 6)), ast_covariances[-1]],
                        ]
                    ),
                    sc_states[-1],
                )
            else:
                P_bplane_parameters = bplane.target_covariance(
                    sc_covariances[-1], sc_states[-1]
                )

            P_bplane_2D = P_bplane_parameters[:2, :2].values
            eigenvalues, eigenvectors = np.linalg.eig(P_bplane_2D)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 6 * np.sqrt(eigenvalues)

            # Plot the last ellipse in the inset
            last_ellipse = Ellipse(
                xy=(bplane._bt_ref.values, bplane._br_ref.values),
                width=width,
                height=height,
                angle=angle,
                edgecolor=colors[i % len(colors)],
                facecolor="none",
            )
            axins.add_patch(last_ellipse)

            # Plot the reference point in the inset as well
            axins.plot(
                bplane._bt_ref.values, bplane._br_ref.values, "o", color=colors[i + 1]
            )
            axins.plot(
                [
                    bplane._bt_ref.values - width / 2,
                    bplane._bt_ref.values + width / 2,
                    bplane._bt_ref.values + width / 2,
                    bplane._bt_ref.values - width / 2,
                    bplane._bt_ref.values - width / 2,
                ],
                [
                    bplane._br_ref.values - height / 2,
                    bplane._br_ref.values - height / 2,
                    bplane._br_ref.values + height / 2,
                    bplane._br_ref.values + height / 2,
                    bplane._br_ref.values - height / 2,
                ],
                linestyle="--",
                color="black",
            )

            # Set the limits for the inset to zoom in around the last ellipse
            xlim = [
                bplane._bt_ref.values - width / 2,
                bplane._bt_ref.values + width / 2,
            ]
            ylim = [
                bplane._br_ref.values - height / 2,
                bplane._br_ref.values + height / 2,
            ]
            axins.set_xlim(xlim)
            axins.set_ylim(ylim)

            axins.axis("off")
            axins.grid(True)
            axins.ticklabel_format(useOffset=False, style="plain")
            axins.invert_yaxis()  # Invert the y-axis as in the main plot

            # Mark the region of the main plot being zoomed in
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        # Set plot limits and labels for the combined plot
        plt.xlabel(r"$\mathbf{\hat{T}}$ [km]")
        plt.ylabel(r"$\mathbf{\hat{R}}$ [km]")
        ax.invert_yaxis()
        ax.set_aspect("equal", "datalim")
        ax.legend(loc="upper right")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(axis="x", rotation=45)  # Rotate x-axis ticks
        plt.show()

    # endregion BPlane #
    # ------------------#

    # --------------------#
    # region FilterOD    #
    # --------------------#
    def plot_prefit_residuals(
        filterOD: FilterOD,
        n_sigma,
        title: str = "Prefit Residuals",
        scale: str = "linxy",
        x_offset_flag: bool = False,
    ):
        """
        Plots the prefit residuals for range and range rate measurements.

        :param filterOD: The FilterOD object.
        :type filterOD: FilterOD

        :param n_sigma: The number of standard deviations to use for the sigma bounds.
        :type n_sigma: float

        :param scale: Scale type for plots.
        :type scale: str, optional

        :param x_offset_flag: If True, apply x-axis offset.
        :type x_offset_flag: bool, optional

        :returns: None
        :rtype: None

        """
        # Initialize variables
        tlength = len(FilterOD.range_residuals.data["GS1 residuals"])
        t = range(tlength)

        # Plot prefit residuals
        fig, axs = plt.subplots(2, 1)
        # Add title
        fig.suptitle(title, fontweight="bold")

        # Apply scaling
        for row in axs:
            for ax in row:
                ax = Plotting.apply_scale_to_axes(ax, scale)
        # Apply x_axis offset
        if x_offset_flag:
            t = Plotting.plot_with_x_offset(t, x_offset_flag)

        # enable minor ticks for minor grid
        ax.minorticks_on()

        axs[0].scatter(t, filterOD.range_residuals.data["GS1 residuals"], label="GS1")
        axs[0].scatter(t, filterOD.range_residuals.data["GS2 residuals"], label="GS2")
        axs[0].plot(
            t,
            n_sigma * 1e-3 * np.ones((tlength, 1)),
            "r--",
            label=str(n_sigma) + " sigma bounds",
        )
        axs[0].plot(t, -n_sigma * 1e-3 * np.ones((tlength, 1)), "r--")
        axs[0].set(xlabel="time index", ylabel="range residual [km]")
        axs[0].set_ylim(-3 * n_sigma * 1e-3, 3 * n_sigma * 1e-3)
        axs[0].grid(True, which="both")
        axs[0].legend(loc="lower right")
        axs[1].scatter(
            t, filterOD.range_rate_residuals.data["GS1 residuals"], label="GS1"
        )
        axs[1].scatter(
            t, filterOD.range_rate_residuals.data["GS2 residuals"], label="GS2"
        )
        axs[1].plot(
            t,
            n_sigma * 1e-7 * np.ones((tlength, 1)),
            "r--",
            label=str(n_sigma) + " sigma bounds",
        )
        axs[1].plot(t, -n_sigma * 1e-7 * np.ones((tlength, 1)), "r--")
        axs[1].set(xlabel="time index", ylabel="range rate residual [km/s]")
        axs[1].set_ylim(-3 * n_sigma * 1e-7, 3 * n_sigma * 1e-7)
        axs[1].grid(True, which="minor")
        axs[1].legend(loc="lower right")
        plt.show()

    # endregion FilterOD #
    # --------------------#

    # -----------------#
    # region Desat    #
    # -----------------#
    def plot_scenario_param(
        scenario: ScenarioSetup,
        scale: str = "linxy",
        x_offset_flag: bool = False,
    ):
        """
        Plots the scenario paramters

        :param scenario: The ScenarioSetup object.
        :type scenario: ScenarioSetup

        :param title: Title of the plot.
        :type title: str

        :param scale: Scale type for plots.
        :type scale: str, optional

        :param x_offset_flag: If True, apply x-axis offset.
        :type x_offset_flag: bool, optional

        :returns: None
        :rtype: None

        """
        # Initialize
        total_positions = scenario.total_position.quantity.values
        total_velocities = scenario.total_velocity.quantity.values
        total_ets = scenario.total_epochsTDB.times.values

        # Apply x_axis offset
        if x_offset_flag:
            total_ets = Plotting.plot_with_x_offset(total_ets, x_offset_flag)
        # Plotting position and velocitu
        _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8))
        # Apply Scaling
        ax1 = Plotting.apply_scale_to_axes(ax1, scale)
        ax1.plot(total_ets, total_positions)
        ax1.set_xlabel(f"Time [{scenario.total_epochsTDB.times.units}]")
        ax1.set_ylabel("Position [km]")
        # Apply Scaling
        ax2 = Plotting.apply_scale_to_axes(ax2, scale)
        ax2.plot(total_ets, total_velocities)
        ax2.set_xlabel(f"Time [{scenario.total_epochsTDB.times.units}]")
        ax2.set_ylabel("Velocity [km/sec]")
        # Apply Scaling
        ax3 = Plotting.apply_scale_to_axes(ax3, scale)
        ax3.plot(total_ets, total_ets)
        ax3.set_xlabel(f"Time [{scenario.total_epochsTDB.times.units}]")
        ax3.set_ylabel(f"ETS [{scenario.total_epochsTDB.times.units}]")
        plt.tight_layout()
        plt.show()
        pass

    # endregion Desat #
    # -----------------#

    # ------------------------#
    # region Standardized    #
    # ------------------------#
    # standardized time-series scatter (2D)
    @staticmethod
    def time_series_basic_scatter(
        time,
        data,
        title: str = "Scatter Plot",
        x_axis_label: str = "Time",
        y_axis_label: str = "Value",
        show: bool = False,
        save_dir: str = None,
        save_name: str = None,
        save_format: str = None,
        scale: str = "linxy",
        x_offset_flag: bool = False,
    ):
        """
        Method for plotting time series data on a scatter plot.

        :param time: timestamps values
        :type time: np.array()

        :param data: data values
        :type data: np.array()

        :param title: Title of plot. Defaults to "Scatter Plot".
        :type title: str

        :param x_axis_label: X-axis label of plot. Defaults to "Time".
        :type x_axis_label: str

        :param y_axis_label: Y-axis label of plot. Defaults to "Value".
        :type y_axis_label: str

        :param show: Option to show plot. Defaults to False.
        :type show: bool

        :param save_dir: Save directory. Defaults to None.
        :type save_dir: str

        :param save_name: Figure saving name. Defaults to None.
        :type save_name: str

        :param save_format: Figure saving format. Can be "pickle" or "png". Defaults to None.
        :type save_format: str

        :param scale: Scaling option ['logx', 'logy', 'logxy', 'linx', 'liny', 'linxy']
        :type scale: str

        :param x_offset_flag: Offset value for X axis
        :type x_offset_flag: bool


        """

        fig, ax = plt.subplots()
        # Apply x_axis offset
        if x_offset_flag:
            time = Plotting.plot_with_x_offset(time, x_offset_flag)
        # Apply Scaling
        ax = Plotting.apply_scale_to_axes(ax, scale)
        ax.scatter(time, data)
        ax.set(title=title, xlabel=x_axis_label, ylabel=y_axis_label)
        ax.grid(True, which="both")

        if save_name is not None:
            Plotting.save_plot(
                fig, save_dir=save_dir, save_name=save_name, save_format=save_format
            )

        if show:
            plt.show()

    # standardized time-series line (2D)
    @staticmethod
    def time_series_basic_line(
        time,
        data,
        title: str = "Line Plot",
        x_axis_label: str = "Time",
        y_axis_label: str = "Value",
        show: bool = False,
        save_dir: str = None,
        save_name: str = None,
        save_format: str = None,
        scale: str = "linxy",
        x_offset_flag: bool = False,
    ):
        """
        Method for plotting time series data on a line plot.

        :param time: timestamps values
        :type time: np.array()

        :param data: data values
        :type data: np.array()

        :param title: Title of plot. Defaults to "Line Plot".
        :type title: str

        :param x_axis_label: X-axis label of plot. Defaults to "Time".
        :type x_axis_label: str

        :param y_axis_label: Y-axis label of plot. Defaults to "Value".
        :type y_axis_label: str

        :param show: Option to show plot. Defaults to False.
        :type show: bool

        :param save_dir: Save directory. Defaults to None.
        :type save_dir: str

        :param save_name: Figure saving name. Defaults to None.
        :type save_name: str

        :param save_format: Figure saving format. Can be "pickle" or "png". Defaults to None.
        :type save_format: str

        :param scale: Scaling option ['logx', 'logy', 'logxy', 'linx', 'liny', 'linxy']
        :type scale: str

        :param x_offset_flag: Offset value for X axis
        :type x_offset_flag: bool

        """

        fig, ax = plt.subplots()
        # Apply x_axis offset
        if x_offset_flag:
            time = Plotting.plot_with_x_offset(time, x_offset_flag)
        # Apply Scaling
        ax = Plotting.apply_scale_to_axes(ax, scale)
        ax.plot(time, data)
        ax.set(title=title, xlabel=x_axis_label, ylabel=y_axis_label)
        ax.grid(True, which="both")

        if save_name is not None:
            Plotting.save_plot(
                fig, save_dir=save_dir, save_name=save_name, save_format=save_format
            )

        if show:
            plt.show()

    # standardized single trajectory plot (3D)
    @staticmethod
    def plot_orbit_3d(
        positions,
        title: str = "3D Trajectory",
        origin=None,
        units: str = "km",
        show: bool = False,
        save_dir: str = None,
        save_name: str = None,
        save_format: str = None,
    ):
        """
        Method for plotting a single 3D trajectory

        :param positions: object position in X,Y,Z coordinates (no frame assuemd).
        :type positions: np.array(N,3)

        :param origin: origin of the plot in X,Y,Z coordinates (no frame assumed). Defaults to None.
        :type origin: np.array(,3)

        :param title: Title of plot. Defaults to "3D Trajectory".
        :type title: str

        :param units: Units to use in the X,Y,Z labels. Defaults to "km".
        :type units: str

        :param show: Option to show plot. Defaults to False.
        :type show: bool

        :param save_dir: Save directory. Defaults to None.
        :type save_dir: str

        :param save_name: Figure saving name. Defaults to None.
        :type save_name: str

        :param save_format: Figure saving format. Can be "pickle" or "png". Defaults to None.
        :type save_format: str

        """

        # Create 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # Plot spacecraft trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
        if origin is not None:
            ax.plot(0, 0, 0, marker="x", label="Origin: {0}".format(origin))
        # Set labels and title
        ax.set_xlabel("X [{0}]".format(units))
        ax.set_ylabel("Y [{0}]".format(units))
        ax.set_zlabel("Z [{0}]".format(units))
        plt.title(title, fontweight="bold")
        plt.legend()
        plt.tight_layout()
        plt.axis("equal")

        if save_name is not None:
            Plotting.save_plot(
                fig, save_dir=save_dir, save_name=save_name, save_format=save_format
            )

        if show:
            plt.show()

        return fig

    # Standardized multiple trajectories plot (3D)
    @staticmethod
    def plot_multiple_orbits_3d(
        positions_list,
        labels=None,
        origin=None,
        title: str = "@Sun, J2000",
        units: str = "km",
        show: bool = False,
        save_dir: str = None,
        save_name: str = None,
        save_format: str = None,
        line_styles=None,
        colors=None,
        markers=None,
        view_xy_plane: bool = True,  # New parameter to control view of the XY plane
    ):
        """
        Method for plotting multiple 3D trajectories, with default view as XY-plane.

        :param positions_list: list of several objects positions in X,Y,Z coordinates.
        :type positions_list: list

        :param labels: list of several objects labels. Defaults to None.
        :type labels: list

        :param origin: origin of the plot in X,Y,Z coordinates. Defaults to None.
        :type origin: np.array(,3)

        :param title: Title of plot. Defaults to "Reference Frame: J2000".
        :type title: str

        :param units: Units to use in the X,Y,Z labels. Defaults to "km".
        :type units: str

        :param show: Option to show plot. Defaults to False.
        :type show: bool

        :param save_dir: Save directory. Defaults to None.
        :type save_dir: str

        :param save_name: Figure saving name. Defaults to None.
        :type save_name: str

        :param save_format: Figure saving format. Defaults to None.
        :type save_format: str

        :param line_styles: List of line styles for each orbit. Defaults to solid lines.
        :type line_styles: list

        :param colors: List of colors for each orbit. Defaults to default matplotlib colors.
        :type colors: list

        :param markers: List of markers for each orbit. Defaults to None (no markers).
        :type markers: list

        :param view_xy_plane: Boolean to set the default view to XY-plane. Defaults to True.
        :type view_xy_plane: bool

        :return: Matplotlib figure object.
        """
        # Create 3D plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        # Iterate over each trajectory and plot it
        for i, positions in enumerate(positions_list):
            label = labels[i] if labels is not None else f"Object {i+1}"
            line_style = line_styles[i] if line_styles is not None else "-"
            color = colors[i] if colors is not None else None
            marker = markers[i] if markers is not None else None

            # Plot the trajectory
            ax.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                label=label,
                linestyle=line_style,
                color=color,
                marker=marker,
            )

        # Optionally mark the origin
        if origin is not None:
            ax.scatter(0, 0, 0, color="red", marker=".", s=100, label=origin)

        # Set axis labels with units
        ax.set_xlabel(f"X [{units}]")
        ax.set_ylabel(f"Y [{units}]")
        ax.set_zlabel(f"Z [{units}]")
        ax.set_title(title)

        # Set the default view to XY-plane (top-down view)
        if view_xy_plane:
            ax.view_init(elev=90, azim=-90)
            # Hide the z-axis when viewing from above
            ax.zaxis.line.set_lw(0.0)
            ax.set_zticks([])  # Remove z-axis ticks
            ax.zaxis.label.set_visible(False)  # Hide z-axis label

        # Add a grid and legend
        ax.grid(True)
        ax.legend(loc="center right")

        # Adjust layout
        # plt.tight_layout()

        # Optionally save the figure
        if save_name is not None and save_dir is not None and save_format is not None:
            file_path = f"{save_dir}/{save_name}.{save_format}"
            fig.savefig(file_path, format=save_format)
            print(f"Plot saved to {file_path}")

        # Show plot if specified
        if show:
            plt.show()

        return fig

    # ground-stations visibility plot
    @staticmethod
    def plot_gs_visibility(
        visibility_windows: dict,
        title: str = "Ground Station Visibility Windows",
        epoch_start: str = None,
        show: bool = False,
        save_dir: str = None,
        save_name: str = None,
        save_format: str = None,
        scale: str = "linxy",
    ):
        """
        Method to plot when the spacecraft can be seen by a set of ground stations

        :param visibility_windows: dictionary of the GS visibilities
        :type visibility_windows: dict

        :param epoch_start: Starting epoch
        :type epoch_start: str

        :param show: Option to show plot. Defaults to False.
        :type show: bool

        :param save_dir: Save directory. Defaults to None.
        :type save_dir: str

        :param save_name: Figure saving name. Defaults to None.
        :type save_name: str

        :param save_format: Figure saving format. Can be "pickle" or "png". Defaults to None.
        :type save_format: str

        :param scale: Scaling option ['logx', 'logy', 'logxy', 'linx', 'liny', 'linxy']
        :type scale: str

        """

        # Method to plot when the spacecraft can be seen by a set of ground stations

        # Create figure and axis
        fig = plt.figure()
        ax = fig.subplots()

        # Setting custom color
        colors = Plotting.color_palettes["scb_palette"]
        line_color = colors[0]

        # Apply Scaling
        ax = Plotting.apply_scale_to_axes(ax, scale)
        y_values = []
        y_labels = []

        counter = 1
        for gs, windows in visibility_windows.items():
            y_values.append(counter)
            y_labels.append(gs.name)

            if not windows:  # Check for empty list
                continue

            for window in windows:
                # Ensure xmin and xmax are always defined
                if epoch_start is not None:
                    xmin = window[0] - epoch_start
                    xmax = window[1] - epoch_start
                else:
                    xmin = window[0]
                    xmax = window[1]

                ax.hlines(y=counter, xmin=xmin, xmax=xmax, colors=[line_color])

            counter += 1

        # Set y labels
        ax.set_yticks(y_values, y_labels)
        plt.gcf().autofmt_xdate()

        # Set labels and title
        ax.set_xlabel("Time")
        ax.set_ylabel("Ground Station")
        ax.set_title(title, fontweight="bold")

        # Save the figure if requested
        if save_name is not None:
            Plotting.save_plot(
                fig, save_dir=save_dir, save_name=save_name, save_format=save_format
            )

        # Show plot if requested
        if show:
            plt.show()

        return fig

    # standarized pos-vel plot
    @staticmethod
    def plot_position_velocity(
        data,
        times=None,
        title: str = "Pos-Vel",
        show: bool = False,
        save_dir: str = None,
        save_name: str = None,
        save_format: str = None,
        scale: str = "linxy",
        x_offset_flag: bool = False,
    ):
        """
        Method to plot the position-velocity as function of time.

        :param data: Array of the state vector made by position and velocity
        :type data: np.array(n,6)

        :param times: Time values
        :type times: np.array()

        :param title: Title of plot. Defaults to "Pos-Vel".
        :type title: str

        :param show: Option to show plot. Defaults to False.
        :type show: bool

        :param save_dir: Save directory. Defaults to None.
        :type save_dir: str

        :param save_name: Figure saving name. Defaults to None.
        :type save_name: str

        :param save_format: Figure saving format. Can be "pickle" or "png". Defaults to None.
        :type save_format: str


        """

        # Check time
        if times is None:
            times = [x for x in range(len(data))]
        # Plots
        fig, axs = plt.subplots(
            3,
            2,
            figsize=(8, 8),
            sharex=False,
        )

        plt.subplots_adjust(wspace=0.5, hspace=0.2)
        # Add title
        fig.suptitle(title, fontweight="bold")

        # Apply scaling
        for row in axs:
            for ax in row:
                ax = Plotting.apply_scale_to_axes(ax, scale)
        # Apply x_axis offset
        if x_offset_flag:
            times = Plotting.plot_with_x_offset(times, x_offset_flag)

        # Iterate over each subplot and enable minor ticks
        for ax in axs.flat:
            # Use AutoMinorLocator for the x-axis (linear scale)
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

            # Use LogLocator for the y-axis (log scale)
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto"))

            # Enable minor ticks
            ax.minorticks_on()

        # Plot x component of position
        axs[0, 0].plot(times, data[:, 0])
        axs[0, 0].set_xlabel("Time [s]")
        axs[0, 0].set_ylabel("x [km]")
        axs[0, 0].grid(True, which="minor", linestyle="--")

        # Plot y component of position
        axs[1, 0].plot(times, data[:, 1])
        axs[1, 0].set_xlabel("Time [s]")
        axs[1, 0].set_ylabel("y [km]")
        axs[1, 0].grid(True, which="minor", linestyle="--")

        # Plot z component of position
        axs[2, 0].plot(times, data[:, 2])
        axs[2, 0].set_xlabel("Time [s]")
        axs[2, 0].set_ylabel("z [km]")
        axs[2, 0].grid(True, which="minor", linestyle="--")

        # Plot x component of velocity
        axs[0, 1].plot(times, data[:, 3])
        axs[0, 1].set_xlabel("Time [s]")
        axs[0, 1].set_ylabel(r"$\dot{x}$ [km/s]")
        axs[0, 1].grid(True, which="minor", linestyle="--")

        # Plot y component of velocity
        axs[1, 1].plot(times, data[:, 4])
        axs[1, 1].set_xlabel("Time [s]")
        axs[1, 1].set_ylabel(r"$\dot{y}$ [km/s]")
        axs[1, 1].grid(True, which="minor", linestyle="--")

        # Plot z component of velocity
        axs[2, 1].plot(times, data[:, 5])
        axs[2, 1].set_xlabel("Time [s]")
        axs[2, 1].set_ylabel(r"$\dot{z}$ [km/s]")
        axs[2, 1].grid(True, which="minor", linestyle="--")

        if save_name is not None:
            Plotting.save_plot(
                fig, save_dir=save_dir, save_name=save_name, save_format=save_format
            )

        if show:
            plt.show()

        return fig

    # standarized measurement plot (opnav)
    @staticmethod
    def plot_opnav_measurements(
        opnav_measurements,
        title: str = "Opnav Measurements",
        show: bool = False,
        save_dir: str = None,
        save_name: str = None,
        save_format: str = None,
        scale: str = "linxy",
        x_offset_flag: bool = False,
    ):
        """
        Method to plot opnav measurements as function of time.

        :param opnav_measurements: Dictionary of the measurement, generated from the MeasurementDataSet
        :type opnav_measurements: list[EpochArray,np.array,ArrayWFrame]


        :param title: Title of plot. Defaults to None.
        :type title: str

        :param show: Option to show plot. Defaults to False.
        :type show: bool

        :param save_dir: Save directory. Defaults to None.
        :type save_dir: str

        :param save_name: Figure saving name. Defaults to None.
        :type save_name: str

        :param save_format: Figure saving format. Can be "pickle" or "png". Defaults to None.
        :type save_format: str

        :param scale: Scaling option ['logx', 'logy', 'logxy', 'linx', 'liny', 'linxy']
        :type scale: str

        :param x_offset_flag: Offset value for X axis
        :type x_offset_flag: bool

        """
        Plotting.set_custom_styling()

        # Extract measurement name, time, and data lists
        meas_time = opnav_measurements[0].times
        meas_data = opnav_measurements[2].quantity
        # Get units and time frame from first index (assumed consistent for all epochs)
        data_unit = str(opnav_measurements[2].quantity.units)
        time_unit = str(opnav_measurements[0].times.units)
        sys = str(opnav_measurements[0].system)

        # Get data-time values as lists
        data = []
        time = []
        for ii in range(0, len(meas_data.values)):
            data.append(meas_data[ii].values)
            time.append(meas_time[ii].values)

        xlabel_str = "u coordinate [px]"
        ylabel_str = "v coordinate [px]"

        fig = plt.figure()
        ax = fig.subplots()

        # enable minor ticks for minor grid
        ax.minorticks_on()

        # Apply x_axis offset
        if x_offset_flag:
            opnav_measurements[:, 0] = Plotting.plot_with_x_offset(
                opnav_measurements[:, 0], x_offset_flag
            )

        # Apply Scaling
        ax = Plotting.apply_scale_to_axes(ax, scale)

        ax.scatter(x=meas_data[:, 0].values, y=meas_data[:, 1].values)
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title("Measurement plot - OpNav")
        ax.set_xlabel(xlabel_str)
        ax.set_ylabel(ylabel_str)
        ax.grid("on", "both")

        if save_name is not None:
            Plotting.save_plot(
                fig, save_dir=save_dir, save_name=save_name, save_format=save_format
            )

        if show:
            plt.show()

        return fig

    # standarized measurement plot (radiometric)
    @staticmethod
    def plot_radiometric_measurements(
        radiometric_measurement,
        title: str = None,
        show: bool = False,
        save_dir: str = None,
        save_name: str = None,
        save_format: str = None,
        meas_name: str = "meas_name",
        scale: str = "linxy",
        x_offset_flag: bool = False,
    ):
        """
        Method to plot the radiometric measurements as function of time.

        :param radiometric_measurement: Dictionary of the measurement, generated from the MeasurementDataSet
        :type radiometric_measurement: list[EpochArray,np.array,ArrayWFrame]

        :param title: Title of plot. Defaults to None.
        :type title: str

        :param show: Option to show plot. Defaults to False.
        :type show: bool

        :param save_dir: Save directory. Defaults to None.
        :type save_dir: str

        :param save_name: Figure saving name. Defaults to None.
        :type save_name: str

        :param save_format: Figure saving format. Can be "pickle" or "png". Defaults to None.
        :type save_format: str

        :param scale: Scaling option ['logx', 'logy', 'logxy', 'linx', 'liny', 'linxy']
        :type scale: str

        :param x_offset_flag: Offset value for X axis
        :type x_offset_flag: bool

        """
        # Extract measurement name, time, and data lists
        meas_time = radiometric_measurement[0].times
        meas_data = radiometric_measurement[2].quantity
        # Get units and time frame from first index (assumed consistent for all epochs)
        data_unit = str(radiometric_measurement[2].quantity.units)
        time_unit = str(radiometric_measurement[0].times.units)
        sys = str(radiometric_measurement[0].system)
        # Get data-time values as lists
        data = []
        time = []
        for ii in range(0, len(meas_data.values)):
            data.append(meas_data[ii].values)
            time.append(meas_time[ii].values)

        xlabel_str = meas_name + " [$" + data_unit + "$] "
        ylabel_str = "Time [$" + time_unit + "$] (" + sys + ")"
        # Generate 2D plot of the measurement with adaptive labeling
        fig, ax = plt.subplots()

        # Apply x_axis offset
        if x_offset_flag:
            time = Plotting.plot_with_x_offset(time, x_offset_flag)

        # Apply Scaling
        ax = Plotting.apply_scale_to_axes(ax, scale)
        ax.plot(time, data, ".")
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title("Measurement plot - " + meas_name)
        ax.set_xlabel(xlabel_str)
        ax.set_ylabel(ylabel_str)
        ax.grid("on", "both")

        if save_name is not None:
            Plotting.save_plot(
                fig, save_dir=save_dir, save_name=save_name, save_format=save_format
            )
        if show:
            plt.show()

        return fig

    # standarized measurement plot (radiometric)
    @staticmethod
    def plot_measurements_dataset(
        measurement_data,
        title: str = None,
        show: bool = False,
        save_dir: str = None,
        save_name: str = None,
        save_format: str = None,
    ):
        """
        Method to plot the radiometric measurements as function of time.
        """
        names = [x[8] for x in measurement_data]
        times = [x[0] for x in measurement_data]
        observed = [x[1] for x in measurement_data]
        computed = [x[2] for x in measurement_data]
        residuals = [x[3] for x in measurement_data]
        variances = [x[5] for x in measurement_data]
        instruments = [x[6] for x in measurement_data]
        types = [x[7] for x in measurement_data]

        unique_names = sorted(set(item for sublist in names for item in sublist))
        n_measurement_data = len(unique_names)

        list_names = []
        list_times = []
        list_observed = []
        list_computed = []
        list_residuals = []
        list_variances = []
        for ii in range(0, len(measurement_data)):
            indices = [unique_names.index(name) for name in names[ii]]
            for jj in range(0, len(indices)):
                list_names.append(unique_names[indices[jj]])
                list_times.append(times[ii])
                list_observed.append(observed[ii][jj])
                list_computed.append(computed[ii][jj])
                list_residuals.append(residuals[ii][jj])
                list_variances.append(variances[ii][jj])

        for jj in range(0, len(indices)):
            indices = [i for i, x in enumerate(list_names) if x == unique_names[jj]]
            plot_times = [list_times[i] for i in indices]
            plot_observed = [list_observed[i] for i in indices]
            plot_computed = [list_computed[i] for i in indices]
            plot_residuals = [list_residuals[i] for i in indices]
            plot_variances = [list_variances[i] for i in indices]

            fig1_jj = plt.figure()
            plt.scatter(plot_times, plot_observed, marker="o", edgecolors="black")
            plt.scatter(plot_times, plot_computed, marker="o", edgecolors="black")

            xlabel_str = "Time [et]"
            if title is not None:
                fig1_jj.suptitle(title)
            else:
                fig1_jj.suptitle("Observed and Computed - " + unique_names[jj])
            plt.xlabel(xlabel_str)
            plt.grid("on", "both")

            fig2_jj = plt.figure()
            plt.scatter(plot_times, plot_residuals, marker="o", edgecolors="black")

            xlabel_str = "Time [et]"
            if title is not None:
                fig2_jj.suptitle(title)
            else:
                fig2_jj.suptitle("Residuals - " + unique_names[jj])
            plt.xlabel(xlabel_str)
            plt.grid("on", "both")

            fig3_jj = plt.figure()

            plt.scatter(plot_times, plot_variances, marker="o", edgecolors="black")

            xlabel_str = "Time [et]"
            if title is not None:
                fig3_jj.suptitle(title)
            else:
                fig3_jj.suptitle("Variances - " + unique_names[jj])
            plt.xlabel(xlabel_str)
            plt.grid("on", "both")

        if show:
            plt.show()

    @staticmethod
    def plot_asteroid_3d(
        obj_file=None, scale_factor=1.0, ax=None, radius=None, name="Asteroid"
    ):
        """
        Plot a 3D representation of an asteroid, either from an object file or a simple spherical mesh.

        Parameters:
        - obj_file (str, optional): Path to the asteroid 3D model file (e.g., .obj). If not provided, a simple spherical mesh is plotted.
        - scale_factor (float): Scaling factor to adjust the size of the asteroid. Default is 1.0.
        - ax (matplotlib.axes._subplots.Axes3DSubplot, optional): 3D axes to plot the asteroid on. If not provided, a new figure is created.
        - radius (float, optional): Radius of the asteroid in kilometers if a simple spherical mesh is used. Default is None.
        - name (str, optional): Name of the asteroid to display in the legend. Default is "Asteroid".

        Returns:
        - ax: The 3D axes containing the plot.

        Raises:
        - ValueError: If both obj_file and radius are None.
        """

        # Validate input
        if obj_file is None and radius is None:
            raise ValueError("Either 'obj_file' or 'radius' must be provided.")

        # Initialize the axes if not provided
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        if obj_file:
            # Load the 3D mesh from the object file
            mesh = trimesh.load_mesh(obj_file)
            vertices = np.array(mesh.vertices) * scale_factor
            faces = mesh.faces
        else:
            # Generate a simple spherical mesh
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            x = radius * np.cos(u) * np.sin(v)
            y = radius * np.sin(u) * np.sin(v)
            z = radius * np.cos(v)
            vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
            faces = trimesh.creation.icosphere(subdivisions=3).faces

        if obj_file:
            # Plot the 3D mesh from the object file
            asteroid_mesh = Poly3DCollection(
                [vertices[face] for face in faces],
                facecolor="gray",
                edgecolor="k",
                linewidths=1,
                alpha=1,
                label=name,
            )
            ax.add_collection3d(asteroid_mesh)
        else:
            # Plot the spherical mesh
            ax.plot_surface(
                x,
                y,
                z,
                color="gray",
                edgecolor="k",
                linewidth=0.5,
                alpha=0.8,
                label=name,
            )

        # Return the axes object with the plot
        return ax

    def compare_state_errors_norm(
        solution1,
        solution2,
        true_trajectory,
        epochs,
        n_sigma: int,
        labels: list[str] = ["w SNC", "w/o SNC"],
        t_0: bool = True,
        use_smoothed: bool = True,
    ):
        """
        Compares the state errors and sigma values of two solutions by plotting them together.

        Parameters
        ----------
        solution1 :
            The first SolutionOD object to compare.
        solution2 :
            The second SolutionOD object to compare.
        true_trajectory : Trajectory
            The true trajectory.
        epochs : EpochArray
            The array of epochs.
        n_sigma : int
            The number of standard deviations for sigma.
        labels : list of str
            Labels for the solutions to be compared.
        t_0 : bool
            If True, convert the x-axis to time in days relative to the initial epoch.
        use_smoothed : bool
            If True and smoothed solution is available, uses smoothed covariance.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """

        # Retrieve the errors and sigmas for both solutions
        def get_errors_and_sigmas(solution, true_trajectory, epochs, use_smoothed):
            # Use smart estimated_trajectory method
            estimated_position, estimated_velocity, _ = solution.estimated_trajectory(
                epochs, use_smoothed=use_smoothed
            )

            # Select covariance source (smoothed if available and requested, otherwise filtered)
            if use_smoothed and solution.covariance_smooth is not None:
                covariance_source = solution.covariance_smooth
            else:
                covariance_source = solution.covariance_est

            t = epochs.times.values
            n = len(t)
            err_position = np.zeros(shape=(n, 3))
            err_velocity = np.zeros(shape=(n, 3))
            sigmas = np.zeros(shape=(n, 6))

            for k in range(n):
                if solution.filter.flag_sequence:
                    idx_leg = next(
                        (
                            (
                                i + 1
                                if epochs[k].times.values == leg[-1]
                                and i != len(solution.filter.legs_epochs) - 1
                                else i
                            )
                            for i, leg in enumerate(solution.filter.legs_epochs)
                            if (
                                (epochs[k].times.values in leg)
                                if isinstance(leg, np.ndarray)
                                else (epochs[k].times.values == leg)
                            )
                        ),
                        None,
                    )
                else:
                    idx_leg = None

                # Retrieve true state
                state_nominal = true_trajectory.get_state(
                    epoch_input=epochs[k], idx_leg_input=idx_leg
                )
                err_position[k] = (
                    estimated_position[k] - state_nominal["position"].values
                )
                err_velocity[k] = (
                    estimated_velocity[k] - state_nominal["velocity"].values
                )

                # Extract sigmas from covariance (already stored as matrix)
                marginal_variances = np.diag(covariance_source[k])
                sigmas[k] = np.sqrt(marginal_variances[:6])

            return t, err_position, err_velocity, sigmas

        # Retrieve errors and sigmas for both solutions
        t1, err_pos1, err_vel1, sigmas1 = get_errors_and_sigmas(
            solution1, true_trajectory, epochs, use_smoothed
        )
        t2, err_pos2, err_vel2, sigmas2 = get_errors_and_sigmas(
            solution2, true_trajectory, epochs, use_smoothed
        )

        # Optionally convert time to days relative to the initial epoch
        if t_0:
            t1 = (t1 - t1[0]) / (86400)  # Convert to time in days
            t2 = (t2 - t2[0]) / (86400)  # Convert to time in days

        # Plot the state errors and sigma values for both solutions
        fig, axs = plt.subplots(2, 3, figsize=(15, 5))
        plt.subplots_adjust(wspace=0.6, hspace=0.3)

        x_label_time = r"Time - $t_0$ [days]" if t_0 else f"Time [{epochs.times.units}]"

        # Helper function to plot comparisons
        def plot_comparison(
            ax, t1, t2, err1, err2, sigma1, sigma2, idx, idx_sig, label, unit
        ):
            # Retrieve color cycle from Matplotlib (which follows set_custom_styling)
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key()["color"]
            error_color = colors[0]
            # enable minor ticks for minor grid
            ax.minorticks_on()
            # ax.semilogy(t1, np.abs(err1[:, idx]), label=f"{labels[0]}", color="k")
            # ax.semilogy(t2, np.abs(err2[:, idx]), label=f"{labels[1]}", color="g")
            ax.semilogy(
                t1,
                np.abs(n_sigma * sigma1[:, idx_sig]),
                "r--",
                label=f"{labels[0]} {n_sigma}σ-bound",
            )
            ax.fill_between(
                t1, np.abs(n_sigma * sigma1[:, idx_sig]), color="red", alpha=0.2
            )

            ax.semilogy(
                t2,
                np.abs(n_sigma * sigma2[:, idx_sig]),
                error_color,
                label=f"{labels[1]} {n_sigma}σ-bound",
            )
            ax.fill_between(
                t2, np.abs(n_sigma * sigma2[:, idx_sig]), color=error_color, alpha=0.2
            )
            ax.set(xlabel=x_label_time, ylabel=f"{label} [{unit}]")
            ax.grid(True, which="minor", linestyle="--")
            ax.legend(loc="upper right")

        # Position x-axis
        plot_comparison(
            axs[0, 0],
            t1,
            t2,
            err_pos1,
            err_pos2,
            sigmas1,
            sigmas2,
            0,
            0,
            r"$\varepsilon_x$",
            "km",
        )

        # Position y-axis
        plot_comparison(
            axs[0, 1],
            t1,
            t2,
            err_pos1,
            err_pos2,
            sigmas1,
            sigmas2,
            1,
            1,
            r"$\varepsilon_y$",
            "km",
        )

        # Position z-axis
        plot_comparison(
            axs[0, 2],
            t1,
            t2,
            err_pos1,
            err_pos2,
            sigmas1,
            sigmas2,
            2,
            2,
            r"$\varepsilon_z$",
            "km",
        )

        # Velocity vx-axis
        plot_comparison(
            axs[1, 0],
            t1,
            t2,
            err_vel1,
            err_vel2,
            sigmas1,
            sigmas2,
            0,
            3,
            r"$\varepsilon_{\dot{x}}$",
            "km/s",
        )

        # Velocity vy-axis
        plot_comparison(
            axs[1, 1],
            t1,
            t2,
            err_vel1,
            err_vel2,
            sigmas1,
            sigmas2,
            1,
            4,
            r"$\varepsilon_{\dot{y}}$",
            "km/s",
        )

        # Velocity vz-axis
        plot_comparison(
            axs[1, 2],
            t1,
            t2,
            err_vel1,
            err_vel2,
            sigmas1,
            sigmas2,
            2,
            5,
            r"$\varepsilon_{\dot{z}}$",
            "km/s",
        )

        # Adjust plot layout
        plt.tight_layout()

        return fig

    # ------------------------------------------------------------------ #
    #  JSON-driven boilerplate plots                                      #
    #                                                                     #
    #  These read a ``SolutionOD`` JSON file directly (no live filter),   #
    #  so a solution can be replotted from disk. They rely on the         #
    #  self-describing fields written by ``SolutionOD`` (per-residual     #
    #  epoch tags, ``timeline_epochs``, ``leg_index``, ``legs``).         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_solution_json(json_path, iteration: str = "final"):
        """Load one solution dict (and metadata) from a SolutionOD JSON file.

        Handles both ``SolutionOD.create_JSON`` output (a single solution at the
        top level) and ``SolutionOD.save`` output (multiple iterations stored
        under ``iter_1`` … ``iter_N`` plus ``final``).

        Parameters
        ----------
        json_path : str or path-like
            Path to the solution JSON file.
        iteration : str, optional
            Which solution to load for the multi-iteration ``save`` format:
            ``"final"`` (default), ``"iter_3"``, etc. Ignored for single-solution
            files.

        Returns
        -------
        (solution, metadata) : tuple of dict
        """
        # Auto-detect JSON vs pickle from the file extension.
        from scarabaeus.utils.Serialization import load as _load_serialized

        data = _load_serialized(json_path)
        metadata = data.get("metadata", {})

        has_iters = "final" in data or any(
            str(k).startswith("iter_") for k in data
        )
        if has_iters:
            if iteration in data:
                solution = data[iteration]
            elif "final" in data:
                solution = data["final"]
            else:
                iter_keys = sorted(k for k in data if str(k).startswith("iter_"))
                solution = data[iter_keys[-1]]
        else:
            solution = data
        return solution, metadata

    @staticmethod
    def _covariance_diagonals(solution: dict, name: str = "covariance"):
        """Covariance diagonals of a solution block, whichever way they were saved.

        Parameters
        ----------
        solution : dict
            One solution block of a ``SolutionOD`` JSON file.
        name : str, optional
            ``"covariance"``, ``"covariance_smooth"`` or
            ``"covariance_consider"``. Defaults to ``"covariance"``.

        Returns
        -------
        list or None
            ``<name>_diagonals`` when present, otherwise the diagonals of the
            ``<name>_full`` matrices (None rows kept), otherwise None.
        """
        diags = solution.get(f"{name}_diagonals")
        if diags:
            return diags
        full = solution.get(f"{name}_full")
        if not full:
            return None
        return [None if P is None else np.diag(np.asarray(P, dtype=float)).tolist()
                for P in full]

    @staticmethod
    def _time_axis(epochs, time_unit: str = "hr", t0: float = None):
        """Convert TDB-second epochs to an elapsed-time axis. Returns (x, t0)."""
        scale = {"s": 1.0, "min": 60.0, "hr": 3600.0, "day": 86400.0}.get(
            time_unit, 3600.0
        )
        epochs = np.asarray(epochs, dtype=float)
        if t0 is None:
            t0 = float(np.nanmin(epochs)) if epochs.size else 0.0
        return (epochs - t0) / scale, t0

    # ------------------------------------------------------------------ #
    #  Tutorial-notebook plot styling (calendar axes, tab10, clean grid) #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _et2dt(epochs):
        """Convert TDB (ET) seconds past J2000 to a list of ``datetime`` objects.

        Matches the convention used in the OD tutorial notebooks (J2000 epoch +
        timedelta); the small TDB/UTC offset is irrelevant for plotting.
        """
        from datetime import datetime, timedelta

        j2000 = datetime(2000, 1, 1, 12, 0, 0)
        return [j2000 + timedelta(seconds=float(t)) for t in np.atleast_1d(epochs)]

    @staticmethod
    def _apply_tutorial_style():
        """Apply the compact tutorial-notebook ``rcParams`` for JSON plots."""
        plt.rcParams.update(
            {
                "axes.prop_cycle": cycler(color=[cm.tab10(i / 10) for i in range(10)]),
                "font.family": "sans-serif",
                "font.size": 10,
                "axes.titlesize": 11,
                "axes.titleweight": "bold",
                "axes.labelsize": 10,
                "axes.labelweight": "normal",
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 8,
                "figure.dpi": 110,
                "axes.grid": True,
                "grid.alpha": 0.35,
                "grid.linestyle": "--",
                "grid.color": "gray",
                "figure.autolayout": False,
            }
        )

    # Curated palette used by the showcase-styled JSON replots (mirrors the
    # per-component colors of the flagship plotters such as plot_state_errors).
    _SHOWCASE_PALETTE = (
        color_palettes["colorblind_friendly_palette"]
        + color_palettes["scb_palette"][:2]
    )

    @staticmethod
    def _apply_showcase_style():
        """Apply the rich "showcase" ``rcParams`` used by the library's flagship plots.

        Mirrors :meth:`set_custom_styling` (serif family, bold titles/labels,
        dashed grey grid, curated color cycle) so figures replotted from a
        solution JSON match the look of the in-notebook plots rather than the
        compact :meth:`_apply_tutorial_style` look. ``figure.autolayout`` is kept
        ``False`` so the JSON plotters' explicit :func:`~matplotlib.pyplot.tight_layout`
        and secondary hours-axis placement behave as before.
        """
        plt.rcParams.update(
            {
                "axes.prop_cycle": cycler(color=Plotting._SHOWCASE_PALETTE),
                "font.family": "serif",
                "font.size": 13,
                "axes.titlesize": 17,
                "axes.titleweight": "bold",
                "axes.labelsize": 15,
                "axes.labelweight": "bold",
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 10,
                "legend.framealpha": 0.85,
                "figure.dpi": 120,
                "lines.linewidth": 1.8,
                "axes.axisbelow": True,
                "axes.grid": True,
                "grid.color": "gray",
                "grid.linestyle": "--",
                "grid.linewidth": 0.4,
                "grid.alpha": 0.5,
                "figure.autolayout": False,
            }
        )

    @staticmethod
    def _showcase_grid(ax, date_axis: bool = True, log_y: bool = False):
        """Enable the showcase major + minor dashed grid on ``ax``.

        Adds y-axis minor ticks (skipped on a log y-axis, where matplotlib's own
        decade minor ticks are used) and x-axis minor ticks only when the x-axis
        is numeric — calendar/date axes keep their date locator untouched.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to style.
        date_axis : bool, optional
            ``True`` when the x-axis carries ``datetime`` ticks (default), so no
            numeric minor locator is forced onto it.
        log_y : bool, optional
            ``True`` when the y-axis is logarithmic, so the linear minor locator
            is not applied. Defaults to ``False``.
        """
        if not log_y:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        if not date_axis:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.55)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.30)

    @staticmethod
    def _fmt_calendar(ax):
        """Apply calendar-date tick formatting (``%b %d\\n%H:%M``) to ``ax``."""
        import matplotlib.dates as mdates

        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        ax.set_xlabel("Calendar date [UTC]")

    @staticmethod
    def _add_hours_axis(ax, t0_et, time_unit: str = "hr"):
        """Add a secondary top x-axis showing elapsed time from ``t0_et``."""
        import matplotlib.ticker as mticker
        from matplotlib.dates import date2num

        scale = {"s": 1.0, "min": 60.0, "hr": 3600.0, "day": 86400.0}.get(
            time_unit, 3600.0
        )
        unit_lbl = {"s": "s", "min": "min", "hr": "h", "day": "d"}.get(time_unit, "h")
        t0_num = date2num(Plotting._et2dt([t0_et])[0])
        sec_per_day = 86400.0
        ax2 = ax.secondary_xaxis(
            "top",
            functions=(
                lambda x: (x - t0_num) * sec_per_day / scale,
                lambda u: u * scale / sec_per_day + t0_num,
            ),
        )
        ax2.set_xlabel(f"Time from t0 [{unit_lbl}]", fontsize=8)
        ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
        return ax2

    @staticmethod
    def _component_labels(solution: dict, ncol: int):
        """Human label for each state-covariance/deviation column.

        Columns 0–5 default to ``x, y, z, vx, vy, vz``; columns covered by an
        estimated parameter in the ``legs`` metadata are named accordingly
        (e.g. ``dv_man[0]``). Anything else falls back to ``state[i]``.
        """
        base = ["x", "y", "z", "vx", "vy", "vz"]
        labels = [base[i] if i < 6 else f"state[{i}]" for i in range(ncol)]
        for leg in (solution.get("legs") or []):
            for comp in leg.get("components", []):
                if str(comp.get("name")).lower() in ("position", "velocity"):
                    continue
                dim = int(comp.get("dim", 1))
                for off in range(dim):
                    col = int(comp.get("index_start", 0)) + off
                    if col < ncol:
                        labels[col] = (
                            f"{comp['name']}[{off}]" if dim > 1 else str(comp["name"])
                        )
        return labels

    @staticmethod
    def _draw_leg_boundaries(ax, solution: dict):
        """Draw faint vertical lines at sequence event / leg-boundary epochs.

        Boundaries are placed on a ``datetime`` x-axis (matching the JSON
        plotters' calendar axis).
        """
        events = solution.get("events_epochs")
        if not events:
            return
        for i, e in enumerate(events):
            ax.axvline(
                Plotting._et2dt([e])[0],
                color="0.4",
                ls=":",
                lw=0.9,
                alpha=0.7,
                label="leg boundary" if i == 0 else None,
            )

    @staticmethod
    def plot_residuals_json(
        json_path,
        iteration: str = "final",
        kind: str = "both",
        weighted: bool = False,
        n_sigma: float = 3.0,
        time_unit: str = "hr",
        save_dir: str = None,
        save_name: str = None,
        show: bool = False,
    ):
        """Plot prefit/postfit residuals directly from a solution JSON file.

        One stacked subplot per dataset (measurement type). Each residual is
        plotted against its reception epoch from the per-entry time tag
        (``[residual, sigma, epoch]``) written by ``SolutionOD``; older files
        without the tag fall back to the measurement index. Per-group RMS and
        count are shown in the legend, ``n_sigma`` reference lines are drawn,
        and sequence leg boundaries are marked.

        Parameters
        ----------
        json_path : str or path-like
            Path to a ``SolutionOD`` JSON file.
        iteration : str, optional
            Which iteration to plot for multi-iteration files (``"iter_1"`` is
            the raw first pass, ``"final"`` the converged solution). Defaults to
            ``"final"``.
        kind : {"both", "prefit", "postfit"}, optional
            Which residual set(s) to draw. Defaults to ``"both"``.
        weighted : bool, optional
            If True, plot ``residual / sigma`` (dimensionless); the ``n_sigma``
            lines become flat at plus and minus ``n_sigma``. Otherwise plot raw residuals with
            an envelope of plus and minus ``n_sigma * sigma``. Defaults to False.
        n_sigma : float, optional
            Multiplier for the sigma reference lines. Defaults to ``3``.
        time_unit : {"s", "min", "hr", "day"}, optional
            Units for the elapsed-time x-axis. Defaults to ``"hr"``.
        save_dir, save_name : str, optional
            If both given, the figure is written via :meth:`save_plot`.
        show : bool, optional
            Display the figure. Defaults to False.

        Returns
        -------
        matplotlib.figure.Figure
        """
        Plotting._apply_showcase_style()
        solution, _ = Plotting._load_solution_json(json_path, iteration)

        groups = {
            "prefit": ["prefits"],
            "postfit": ["postfits"],
            "both": ["prefits", "postfits"],
        }.get(kind)
        if groups is None:
            raise ValueError("kind must be one of 'both', 'prefit', 'postfit'.")

        datasets = sorted(
            {ds for g in groups if solution.get(g) for ds in solution[g]}
        )
        if not datasets:
            raise ValueError(
                f"No {' / '.join(groups)} residuals found in {json_path}."
            )

        # Shared epoch origin across every plotted point.
        all_eps = [
            e[2]
            for g in groups
            if solution.get(g)
            for ds in solution[g]
            for e in solution[g][ds]
            if len(e) > 2 and e[2] is not None
        ]
        t0 = float(np.min(all_eps)) if all_eps else None
        have_dates = t0 is not None

        colors = Plotting._SHOWCASE_PALETTE
        n = len(datasets)
        fig, axes = plt.subplots(
            n, 1, figsize=(13, 3.2 * n + 1.2), squeeze=False, sharex=True
        )
        axes = axes[:, 0]

        for ax, ds in zip(axes, datasets):
            sig_drawn = False
            for gi, g in enumerate(groups):
                entries = (solution.get(g) or {}).get(ds)
                if not entries:
                    continue
                res = np.array([e[0] for e in entries], dtype=float)
                sig = np.array([e[1] for e in entries], dtype=float)
                use_dates = have_dates and all(
                    len(e) > 2 and e[2] is not None for e in entries
                )
                if use_dates:
                    x = Plotting._et2dt([e[2] for e in entries])
                    order = np.argsort([e[2] for e in entries])
                else:
                    x = np.arange(len(entries), dtype=float)
                    order = np.argsort(x)

                y = res / sig if weighted else res
                rms = float(np.sqrt(np.mean(y**2))) if y.size else float("nan")
                ax.plot(
                    x,
                    y,
                    ".",
                    ms=7,
                    alpha=0.75,
                    color=colors[(gi * 2) % len(colors)],
                    label=f"{g[:-1]} (n={y.size}, RMS={rms:.2e})",
                )
                # Raw mode: per-point ±n_sigma*sigma envelope (drawn once).
                if not weighted and not sig_drawn:
                    xo = [x[i] for i in order] if use_dates else x[order]
                    ax.plot(xo, (n_sigma * sig)[order], color="r", lw=0.9, ls="--",
                            alpha=0.7, label=f"±{n_sigma:g}σ")
                    ax.plot(xo, (-n_sigma * sig)[order], color="r", lw=0.9, ls="--",
                            alpha=0.7)
                    sig_drawn = True

            # Weighted mode: flat ±n_sigma reference lines.
            if weighted:
                ax.axhline(n_sigma, color="r", lw=0.9, ls="--", alpha=0.7,
                           label=f"±{n_sigma:g}σ")
                ax.axhline(-n_sigma, color="r", lw=0.9, ls="--", alpha=0.7)

            ax.axhline(0.0, color="k", lw=0.6, ls="--")
            if have_dates:
                Plotting._draw_leg_boundaries(ax, solution)
            ax.set_ylabel(f"{ds}\n(res/σ)" if weighted else ds)
            Plotting._showcase_grid(ax, date_axis=have_dates)
            ax.legend(loc="upper right", framealpha=0.85)

        if have_dates:
            Plotting._fmt_calendar(axes[-1])
            Plotting._add_hours_axis(axes[0], t0, time_unit)
        else:
            axes[-1].set_xlabel("Measurement index")
        fig.suptitle(
            f"{'Weighted ' if weighted else ''}Residuals — {iteration}",
            fontweight="bold",
        )
        fig.tight_layout()
        return Plotting.get_fig_handle(
            fig,
            show=show,
            save_dir=save_dir,
            save_name=save_name,
            save_format="png" if save_name else None,
        )

    @staticmethod
    def plot_parameter_uncertainty_json(
        json_path,
        iteration: str = "final",
        use_smoothed: bool = True,
        time_unit: str = "hr",
        save_dir: str = None,
        save_name: str = None,
        show: bool = False,
    ):
        """Plot the 1-sigma uncertainty time history of estimated parameters from JSON.

        "Parameters" are the state components beyond position/velocity. For a
        sequence each parameter is drawn only over the leg(s) where it is
        estimated, using the ``legs`` layout and per-row ``leg_index`` written by
        ``SolutionOD``; the covariance diagonals are per timeline row, so 1-sigma is
        ``sqrt(diag)``.

        Parameters
        ----------
        json_path : str or path-like
            Path to a ``SolutionOD`` JSON file.
        iteration : str, optional
            Which iteration to plot for multi-iteration files. Defaults to
            ``"final"``.
        use_smoothed : bool, optional
            Prefer the smoother covariance (``covariance_smooth_diagonals``, or
            the diagonals of ``covariance_smooth_full``) when present and
            non-empty, else the filter covariance. Defaults to True.
        time_unit : {"s", "min", "hr", "day"}, optional
            Units for the elapsed-time x-axis. Defaults to ``"hr"``.
        save_dir, save_name : str, optional
            If both given, the figure is written via :meth:`save_plot`.
        show : bool, optional
            Display the figure. Defaults to False.

        Returns
        -------
        matplotlib.figure.Figure
        """
        Plotting._apply_showcase_style()
        solution, _ = Plotting._load_solution_json(json_path, iteration)

        timeline = solution.get("timeline_epochs") or solution.get("timestamps")
        if timeline is None:
            raise ValueError(
                "JSON has no 'timeline_epochs' or 'timestamps' to plot against."
            )
        timeline = np.asarray(timeline, dtype=float)

        cov = (Plotting._covariance_diagonals(solution, "covariance_smooth")
               if use_smoothed else None)
        if not cov:
            cov = Plotting._covariance_diagonals(solution)
        if not cov:
            raise ValueError("JSON has no covariance diagonals to plot.")

        t0 = float(np.nanmin(timeline))
        dts = Plotting._et2dt(timeline)

        legs = solution.get("legs")
        leg_index = solution.get("leg_index")

        # Build (label, column, row-mask) for every estimated parameter scalar.
        series = []
        if legs and leg_index is not None:
            leg_index = np.asarray(leg_index)
            for leg in legs:
                mask = leg_index == leg["index"]
                for comp in leg["components"]:
                    if str(comp["name"]).lower() in ("position", "velocity"):
                        continue
                    for off in range(comp["dim"]):
                        col = comp["index_start"] + off
                        label = (
                            f"{comp['name']}[{off}] (leg {leg['index']})"
                            if comp["dim"] > 1
                            else f"{comp['name']} (leg {leg['index']})"
                        )
                        series.append((label, col, mask))
        else:
            # Single-arc fallback: parameters are columns beyond pos+vel (>=6).
            ncol = max(len(r) for r in cov)
            all_rows = np.ones(len(timeline), dtype=bool)
            for col in range(6, ncol):
                series.append((f"param[{col - 6}]", col, all_rows))

        if not series:
            raise ValueError(
                "No estimated parameters found (state is position/velocity only)."
            )

        fig, ax = plt.subplots(figsize=(13, 7))
        colors = Plotting._SHOWCASE_PALETTE
        plotted = 0
        for i, (label, col, mask) in enumerate(series):
            xs, ys = [], []
            for row in np.where(mask)[0]:
                diag = cov[row]
                if col < len(diag) and diag[col] is not None and diag[col] >= 0.0:
                    xs.append(dts[row])
                    ys.append(np.sqrt(diag[col]))
            if xs:
                ax.plot(
                    xs,
                    ys,
                    marker=".",
                    ms=5,
                    lw=1.6,
                    color=colors[i % len(colors)],
                    label=label,
                )
                plotted += 1

        if plotted == 0:
            raise ValueError(
                "Estimated parameters exist but no covariance entries cover them."
            )

        Plotting._draw_leg_boundaries(ax, solution)
        Plotting._showcase_grid(ax, date_axis=True)
        Plotting._fmt_calendar(ax)
        Plotting._add_hours_axis(ax, t0, time_unit)
        ax.set_ylabel("Parameter 1σ uncertainty")
        ax.set_title(
            f"Estimated-parameter uncertainty — {iteration}", fontweight="bold"
        )
        ax.legend(loc="best", framealpha=0.85)
        fig.tight_layout()
        return Plotting.get_fig_handle(
            fig,
            show=show,
            save_dir=save_dir,
            save_name=save_name,
            save_format="png" if save_name else None,
        )

    @staticmethod
    def plot_covariance_json(
        json_path,
        iteration: str = "final",
        use_smoothed: bool = True,
        time_unit: str = "hr",
        log_scale: bool = True,
        save_dir: str = None,
        save_name: str = None,
        show: bool = False,
    ):
        """Plot the 1-sigma (the square root of the covariance diagonal) of every state component over time, from JSON.

        One line per state-covariance column, drawn against ``timeline_epochs``
        (the deviation/covariance grid), labelled from the leg metadata where
        available. Sequence leg boundaries are marked. Note the components carry
        mixed units (km, km/s, parameter units), so a log y-axis is used by
        default to show their relative evolution on one panel.

        Parameters
        ----------
        json_path : str or path-like
            Path to a ``SolutionOD`` JSON file.
        iteration : str, optional
            Which iteration to plot for multi-iteration files. Defaults to
            ``"final"``.
        use_smoothed : bool, optional
            Prefer the smoother covariance when present and non-empty, else
            the filter covariance; each is read from ``*_diagonals`` or from the
            diagonals of ``*_full``. Defaults to True.
        time_unit : {"s", "min", "hr", "day"}, optional
            Units for the elapsed-time x-axis. Defaults to ``"hr"``.
        log_scale : bool, optional
            Use a logarithmic y-axis (recommended for mixed-unit sigmas).
            Defaults to True.
        save_dir, save_name : str, optional
            If both given, the figure is written via :meth:`save_plot`.
        show : bool, optional
            Display the figure. Defaults to False.

        Returns
        -------
        matplotlib.figure.Figure
        """
        Plotting._apply_showcase_style()
        solution, _ = Plotting._load_solution_json(json_path, iteration)

        timeline = solution.get("timeline_epochs") or solution.get("timestamps")
        if timeline is None:
            raise ValueError("JSON has no 'timeline_epochs'/'timestamps' axis.")
        timeline = np.asarray(timeline, dtype=float)

        cov = (Plotting._covariance_diagonals(solution, "covariance_smooth")
               if use_smoothed else None)
        if not cov:
            cov = Plotting._covariance_diagonals(solution)
        if not cov:
            raise ValueError("JSON has no covariance diagonals to plot.")

        ncol = max(len(r) for r in cov)
        labels = Plotting._component_labels(solution, ncol)
        t0 = float(np.nanmin(timeline))
        dts = Plotting._et2dt(timeline)
        colors = Plotting._SHOWCASE_PALETTE

        fig, ax = plt.subplots(figsize=(13, 7))
        for col in range(ncol):
            xs, ys = [], []
            for row in range(len(cov)):
                diag = cov[row]
                if col < len(diag) and diag[col] is not None and diag[col] >= 0.0:
                    xs.append(dts[row])
                    ys.append(np.sqrt(diag[col]))
            if xs:
                ax.plot(xs, ys, lw=1.7, color=colors[col % len(colors)], label=labels[col])

        Plotting._draw_leg_boundaries(ax, solution)
        if log_scale:
            ax.set_yscale("log")
        Plotting._showcase_grid(ax, date_axis=True, log_y=log_scale)
        Plotting._fmt_calendar(ax)
        Plotting._add_hours_axis(ax, t0, time_unit)
        ax.set_ylabel("State 1σ uncertainty  (mixed units)")
        ax.set_title(
            f"Estimation covariance — 1σ per state component — {iteration}",
            fontweight="bold",
        )
        ax.legend(loc="best", framealpha=0.85, ncol=2)
        fig.tight_layout()
        return Plotting.get_fig_handle(
            fig,
            show=show,
            save_dir=save_dir,
            save_name=save_name,
            save_format="png" if save_name else None,
        )

    @staticmethod
    def plot_estimate_json(
        json_path,
        iteration: str = "final",
        use_smoothed: bool = False,
        time_unit: str = "hr",
        save_dir: str = None,
        save_name: str = None,
        show: bool = False,
    ):
        """Plot the estimated state deviation per component over time, from JSON.

        One stacked subplot per state component showing ``deviation_estimated``
        (or the smoothed deviation) against ``timeline_epochs``, with a 1-sigma
        band from the covariance diagonal. Sequence leg boundaries are marked.

        Parameters
        ----------
        json_path : str or path-like
            Path to a ``SolutionOD`` JSON file.
        iteration : str, optional
            Which iteration to plot for multi-iteration files. Defaults to
            ``"final"``.
        use_smoothed : bool, optional
            Plot ``deviation_smoothed`` when present and non-empty instead of the
            filter ``deviation_estimated``. Defaults to False.
        time_unit : {"s", "min", "hr", "day"}, optional
            Units for the elapsed-time x-axis. Defaults to ``"hr"``.
        save_dir, save_name : str, optional
            If both given, the figure is written via :meth:`save_plot`.
        show : bool, optional
            Display the figure. Defaults to False.

        Returns
        -------
        matplotlib.figure.Figure
        """
        Plotting._apply_showcase_style()
        solution, _ = Plotting._load_solution_json(json_path, iteration)

        dev = solution.get("deviation_smoothed") if use_smoothed else None
        if not dev:
            dev = solution.get("deviation_estimated")
        if not dev:
            raise ValueError("JSON has no estimated deviation to plot.")
        dev = np.asarray(dev, dtype=float)
        if dev.ndim == 1:
            dev = dev.reshape(-1, 1)
        nrows, ncol = dev.shape

        timeline = solution.get("timeline_epochs") or solution.get("timestamps")
        if timeline is None:
            raise ValueError("JSON has no 'timeline_epochs'/'timestamps' axis.")
        timeline = np.asarray(timeline, dtype=float)[:nrows]

        cov = Plotting._covariance_diagonals(solution)  # ±1σ band (forward filter)
        labels = Plotting._component_labels(solution, ncol)
        t0 = float(np.nanmin(timeline))
        dts = Plotting._et2dt(timeline)
        colors = Plotting._SHOWCASE_PALETTE

        fig, axes = plt.subplots(
            ncol, 1, figsize=(13, 2.0 * ncol + 1.2), squeeze=False, sharex=True
        )
        axes = axes[:, 0]
        for col in range(ncol):
            ax = axes[col]
            c = colors[col % len(colors)]
            ax.plot(
                dts, dev[:, col], lw=1.7, color=c,
                label="deviation" if col == 0 else None,
            )
            # ±1σ band from the covariance diagonal where available.
            if cov:
                sig = np.array(
                    [
                        np.sqrt(cov[r][col])
                        if r < len(cov) and col < len(cov[r])
                        and cov[r][col] is not None and cov[r][col] >= 0.0
                        else np.nan
                        for r in range(nrows)
                    ]
                )
                ax.fill_between(
                    dts, -sig, sig, color=c, alpha=0.22,
                    label="±1σ" if col == 0 else None,
                )
            ax.axhline(0.0, color="k", lw=0.6, ls="--")
            Plotting._draw_leg_boundaries(ax, solution)
            Plotting._showcase_grid(ax, date_axis=True)
            ax.set_ylabel(labels[col])
        axes[0].legend(loc="upper right", framealpha=0.85)

        Plotting._fmt_calendar(axes[-1])
        Plotting._add_hours_axis(axes[0], t0, time_unit)
        fig.suptitle(
            f"Estimated state deviation (±1σ) — {iteration}", fontweight="bold"
        )
        fig.tight_layout()
        return Plotting.get_fig_handle(
            fig,
            show=show,
            save_dir=save_dir,
            save_name=save_name,
            save_format="png" if save_name else None,
        )

    @staticmethod
    def get_fig_handle(
        fig_handle,
        show: bool = False,
        save_dir: str = None,
        save_name: str = None,
        save_format: str = None,
    ):
        """
        Get the figure handle from another class

        :param fig_handle: Figure handle generated outside the Plotting class
        :type fig_handle: matplotlib plt.figure() object

        :param show: Option to show plot. Defaults to False.
        :type show: bool

        :param save_dir: Save directory. Defaults to None.
        :type save_dir: str

        :param save_name: Figure saving name. Defaults to None.
        :type save_name: str

        :param save_format: Figure saving format. Can be "pickle" or "png". Defaults to None.
        :type save_format: str

        """

        # Ensure it is a list
        if not isinstance(fig_handle, (list)):
            fig_handle = [fig_handle]

        for idx, fig in enumerate(fig_handle):
            if save_name is not None:
                # Add an index to the save name for multiple figures
                save_suffix = f"_{idx}" if len(fig_handle) > 1 else ""
                full_save_name = f"{save_name}{save_suffix}"
                Plotting.save_plot(
                    fig,
                    save_dir=save_dir,
                    save_name=full_save_name,
                    save_format=save_format,
                )

            if show:
                fig.show()

        return fig_handle

    def plot_uv_measurements_over_time(
        measurements, scale: str = "linxy", x_offset_flag: bool = False
    ):
        """
        Plot u vs v measurements, colored by time, and add subplots of u and v pixels over time
        with a semilog y-axis for the time.

        Parameters:
        measurements (MeasurementDataSet): A dataset containing opnav measurements.

        :param scale: Scaling option ['logx', 'logy', 'logxy', 'linx', 'liny', 'linxy']
        :type scale: str

        :param x_offset_flag: Offset value for X axis
        :type x_offset_flag: bool

        """

        # Extract u, v, and time values, and filter out (0, 0) points
        u_pixels = []
        v_pixels = []
        times = []

        for i in range(len(measurements.data["opnav"])):
            u = measurements.data["opnav"][i][0].values
            v = measurements.data["opnav"][i][1].values
            if not (u == 0 and v == 0):  # Skip (0, 0) points
                u_pixels.append(u)
                v_pixels.append(v)
                times.append(measurements.epochs.times.values[i])

        # Convert to NumPy arrays for plotting
        u_pixels = np.array(u_pixels)
        v_pixels = np.array(v_pixels)
        times = np.array(times)

        # Create a figure with two subplots: one for u vs v scatter plot and another for u/v over time
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        # Normalize time values for color mapping
        norm = plt.Normalize(times.min(), times.max())
        cmap = cm.get_cmap(
            "plasma"
        )  # You can use 'viridis', 'plasma', 'coolwarm', etc.

        # Apply x_axis offset
        if x_offset_flag:
            u_pixels = Plotting.plot_with_x_offset(u_pixels, x_offset_flag)

        # Apply Scaling
        ax1 = Plotting.apply_scale_to_axes(ax1, scale)

        # Scatter plot of u vs v, colored by time, with smaller points
        scatter = ax1.scatter(u_pixels, v_pixels, c=times, cmap=cmap, s=50, norm=norm)

        # Add a color bar to show the time scale
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label("Time [TDB]", fontsize=12)

        # Add labels and title to the scatter plot
        ax1.set_xlabel("u [pix]", fontsize=12)
        ax1.set_ylabel("v [pix]", fontsize=12)
        ax1.grid(True, which="minor", linestyle="--", linewidth=0.7, alpha=0.7)

        # Apply x_axis offset
        if x_offset_flag:
            times = Plotting.plot_with_x_offset(times, x_offset_flag)

        # Apply Scaling
        ax2 = Plotting.apply_scale_to_axes(ax2, scale)

        # Plot the binary array showing where both u_pixels and v_pixels are non-zero
        non_zero_condition = np.where((u_pixels != 0) & (v_pixels != 0), 1, 0)
        ax2.plot(
            times,
            non_zero_condition,
            label="Visibility",
            color="red",
            marker="x",
            linestyle="--",
        )

        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["Not Visible", "Visible"])
        ax2.set_xlabel("Time [TDB]", fontsize=12)
        # ax2.set_ylabel("Visible [-]", fontsize=12)
        # ax2.set_yscale("log")  # Set semi-log scale for the y-axis
        ax2.legend()
        ax2.grid(True, which="minor")

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    # Plotting the availability of each measurement over time
    def plot_measurement_availability(
        combined_times,
        measurement_mask,
        measurement_names,
        scale: str = "linxy",
        x_offset_flag: bool = False,
    ):
        """
        Plots the availability of each measurement over time with x-axis labels as 'time - t_0' in integer days,
        where t_0 is the last time in combined_times.

        Parameters
        ----------
        combined_times : np.ndarray
            Array of combined unique epochs.
        measurement_mask : np.ndarray
            Mask array showing the availability of each measurement at each time.
        measurement_names : list of str
            List of names corresponding to each measurement type.
        scale : str
            Scaling option ['logx', 'logy', 'logxy', 'linx', 'liny', 'linxy'].
        x_offset_flag : bool
            Offset value for X axis.
        """
        # Use the last time in combined_times as t_0 (initial time)
        t_0 = combined_times[0]

        # Calculate time difference in days (time - t_0)
        time_diff_days = (combined_times - t_0) / constants.day_to_sec

        # Round time differences to integer days
        time_diff_days_int = np.floor(time_diff_days).astype(int)

        # Get unique integer day ticks
        unique_days = np.unique(time_diff_days_int)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(15, 5))

        # Apply Scaling
        ax = Plotting.apply_scale_to_axes(ax, scale)

        # Adjust y-values for each measurement to plot on separate lines
        for i, (measurement_name, marker) in enumerate(
            zip(measurement_names, ["o", "x", "s", "^", "v"])
        ):
            available_times = combined_times[measurement_mask[:, i] == 1]
            available_times_diff = (
                available_times - t_0
            ) / constants.day_to_sec  # Convert to days

            # Apply x_axis offset
            if x_offset_flag:
                available_times_diff = Plotting.plot_with_x_offset(
                    available_times_diff, x_offset_flag
                )

            ax.scatter(
                available_times_diff,
                [i + 1] * len(available_times_diff),
                marker=marker,
                label=measurement_name,
                s=40,
            )

        # Formatting the plot
        ax.set_title("Measurement Visibility Over Time")
        ax.set_xlabel(r"Time - $t_0$ [days]")

        # Set y-ticks and labels
        ax.set_yticks(range(1, len(measurement_names) + 1))
        ax.set_yticklabels(measurement_names)

        # Set x-ticks and labels to integer days
        ax.set_xticks(unique_days)
        ax.set_xticklabels([f"{int(day)}" for day in unique_days])

        # Display the legend and grid
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.grid(True)

        # Adjust layout
        plt.tight_layout()

        # Show the plot
        plt.show()

    def plot_media_coverage(json_path: str, type: str = "tropo"):
        """
        Plot DSN coverage intervals (in Ephemeris Time) from a media correction JSON file.

        This function is designed to handle both **iono** and **tropo** files, which share
        the same structure but differ in the top-level dictionary key (`aux_iono` or `aux_tropo`).

        Parameters
        ----------
        json_path : str
            Path to the JSON file containing the media corrections (iono or tropo).

        type : str, optional
            Type of file to process, either `'iono'` or `'tropo'`. The function will
            look for the corresponding key in the JSON (`aux_iono` or `aux_tropo`).
            Default is `'tropo'`.

        Notes
        -----
        The expected JSON has the following fields under the block ``aux_type``:
        ``"start_day"``, ``"start_time"``, ``"end_day"``, ``"end_time"`` (all lists of
        strings), and ``"dsn"`` (list of DSN identifiers, e.g. ``"C10"``, ``"C40"``).

        To compute the CSP UTC day-time to ET conversion, the ``_csp_time_to_et``
        method is used internally and requires a leapsecond kernel loaded in the script.

        Examples
        --------
        >>> plot_media_coverage("orex_2018_iono.json", type="iono")
        >>> plot_media_coverage("orex_2018_tropo.json", type="tropo")
        """
        # Read JSON
        json_data = Utils.load_json(json_path)
        data = json_data.get("aux_" + type, json_data)
        x, y = [], []
        dsn_map = {"C10": 1, "C40": 2, "C60": 3}

        for ii in range(len(data["start_time"])):
            start_et = MediaCorrections._csp_time_to_et(
                data["start_day"][ii], data["start_time"][ii]
            )
            end_et = MediaCorrections._csp_time_to_et(
                data["end_day"][ii], data["end_time"][ii]
            )
            dsn = data["dsn"][ii]
            x.append((start_et, end_et - start_et))
            y.append(dsn_map.get(dsn, 0))

        # --- Plot as coverage bars ---
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
        for (start, dur), dsn_row in zip(x, y):
            ax.broken_barh(
                [(start, dur)],
                (dsn_row - 0.4, 0.8),
                facecolors=colors.get(dsn_row, "gray"),
                alpha=0.7,
            )
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["C10", "C40", "C60"])
        ax.set_xlabel("Ephemeris Time [s]")
        ax.set_ylabel("DSN site")
        ax.set_title(type + " File Coverage Intervals")
        ax.grid(True, which="both", axis="x", linestyle="--", alpha=0.6)
        # Adjust layout
        plt.tight_layout()
        # Show the plot
        plt.show()

    # endregion Standardized #
    # ------------------------#
