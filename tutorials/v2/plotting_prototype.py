"""Prototype plots for the tutorials, to be upstreamed into ``scb.Plotting``.

These functions are planned to move into ``scarabaeus.utils.Plotting``; the notebooks will then
call them from there.

Functions
---------
plot_residuals(source, ...)
    Residuals grouped by measurement type (one row each), stations as colors, with a histogram
    of each row on the right, the reference distribution, and the statistics that say whether the
    fit is consistent (mean, RMS, χ²/n against its expected spread).
residuals_from_solution(solution, measurements, kind="postfit")
    The input of ``plot_residuals`` from an in-memory ``SolutionOD`` and its ``MeasurementSpec``
    (in-memory residuals carry no epochs; the saved JSON does).
state_errors(solution, truth, epochs, ...)
    Estimate minus truth and the covariance at any epochs: along the line of sight from Earth and
    across it, in the plane of sky (default), or in the RTN axes of the orbit about a body.
plot_residual_summary(source, ...)
    One row per tracking pass: the mean (bias) and spread of its normalized residuals for each
    measurement type, against what white noise allows. Scales to campaigns with dozens of passes.
plot_parameter_learning(source, ...)
    What the data taught the filter: for every estimated quantity, the posterior σ against the
    a-priori σ, and how far the estimate moved from the a priori, in units of the a-priori σ.
plot_covariance_shape(at_epoch, ...)
    Several solutions at one epoch: plane-of-sky (T-N) error ellipses with the actual errors, and
    the line-of-sight position and velocity intervals, each on its own scale.
plot_correlation(source, ...)
    Correlation matrix of a saved solution at its epoch (position and velocity in RTN): which
    parameters trade off against which state components.
plot_convergence(source, ...)
    The iterations of a fit: the cost of the pre- and post-fit residuals, the two convergence
    tests against their thresholds (which one stopped the fit), and each dataset's pre-fit RMS.
plot_tracking_timeline(stations, target, epochs, ...)
    When each station could see the spacecraft (shaded by elevation) and when it took data,
    with the Sun-Earth-probe angle on top.
plot_editing(editor, ...)
    The preview the Huber/MAD editing filter lacks: residuals with the per-pass robust trend, the
    acceptance band, the points it would flag and those already flagged, and the re-estimated σ.
plot_prior_posterior(source, ...)
    Corner plot of the a priori and the posterior of selected state elements (Gaussian, no
    samples): ellipses per pair, densities on the diagonal, the truth when known.
plot_monte_carlo(errors, covariances, ...)
    Corner plot of the errors of many fits (Monte Carlo over noise and a-priori draws) against the
    filter's covariance, with the NEES test of covariance realism and a bias test per axis.
bplane_mapping(bplane, state, covariance, samples=0)
    B-plane coordinates of a state and their covariance, mapped with the full Jacobian (position
    and velocity), optionally with Monte Carlo samples mapped through the conic.
plot_bplane(mappings, target=None, ...)
    Overview of the flyby (body, impact radius, B-vectors), the aimpoint dispersion ellipses of
    several solutions relative to the target, and the time of flight.
interactive_legend(fig, on_toggle=None)
    Click a legend entry to hide or show what it labels on every panel (interactive backends).
plot_consistency(errors, ...)
    Is the error consistent with the covariance? Position and velocity errors inside ±3σ,
    normalized errors e/σ, and the Mahalanobis χ² against its bounds, with the tracking periods
    and the data cutoff marked.
"""

import contextlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Okabe-Ito: distinguishable with the common forms of color blindness, and in grayscale print.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]

# Applied with plt.rc_context, so it never changes global settings. Every key that
# Plotting.set_custom_styling sets globally is pinned here, so the result does not depend on
# whether a library plot ran first.
STYLE = {
    "figure.autolayout": False,
    "font.family": "sans-serif",
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "grid.linestyle": "-",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#555555",
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.6,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "legend.frameon": False,
    "legend.fontsize": 8,
}

@contextlib.contextmanager
def _styled():
    """Apply STYLE for one figure and restore only those keys afterwards.

    ``plt.rc_context`` restores every rcParam on exit, including ``interactive``, which the inline
    backend turns on when the first figure of a session is created: inside an ``rc_context`` it is
    switched off again on exit, and later figures are no longer displayed.
    """
    previous = {key: plt.rcParams[key] for key in STYLE}
    plt.rcParams.update(STYLE)
    try:
        yield
    finally:
        plt.rcParams.update(previous)


def _tag(artist, *keys):
    """Register an artist under legend keys: it is shown only while none of its keys is toggled off."""
    keys = set(getattr(artist, "_legend_keys", set())) | set(keys)
    artist._legend_keys = keys
    return artist


def interactive_legend(fig, on_toggle=None):
    """Click a legend entry to hide or show what it labels, on every panel, without replotting.

    Works with every interactive backend. In a notebook, ``%matplotlib macosx`` (``qt`` on Linux and
    Windows) opens the figure in its own window, in any editor; ``%matplotlib widget`` (ipympl) draws
    it in the notebook in JupyterLab, but VS Code's Jupyter extension fails to load that widget and
    shows nothing. Scripts use the Qt or macOS windows by default. With the inline backend the
    figure is a static image and clicks do nothing (the figure is otherwise unchanged).

    An entry controls the artists tagged with its label (the prototypes tag theirs); in any other
    figure, it controls the artists whose ``label`` matches, so this also works on figures made
    elsewhere (``interactive_legend(plt.gcf())``). Hidden entries are drawn faded.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    on_toggle : callable, optional
        Called with the set of hidden labels after each click, before the redraw (e.g. to
        recompute statistics or rescale axes from what is still visible).

    Returns
    -------
    matplotlib.figure.Figure
        The same figure.
    """
    legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    entries = {}  # legend artist -> label
    faded = {}  # label -> [(handle, text)]
    for legend in legends:
        handles = getattr(legend, "legend_handles", None) or legend.legendHandles
        for handle, text in zip(handles, legend.get_texts()):
            label = text.get_text()
            for artist in (handle, text):
                artist.set_picker(True)
                if hasattr(artist, "set_pickradius"):
                    artist.set_pickradius(6)
                entries[artist] = label
            faded.setdefault(label, []).append((handle, text))
    if not entries:
        return fig
    legend_artists = set(entries)

    def members(label):
        found = []
        for ax in fig.axes:
            for artist in ax.get_children():
                if artist in legend_artists:
                    continue
                keys = getattr(artist, "_legend_keys", None)
                if keys is not None:
                    if label in keys:
                        found.append(artist)
                elif getattr(artist, "get_label", lambda: None)() == label:
                    found.append(artist)
        return found

    hidden = set()
    fig._legend_hidden = hidden

    def on_pick(event):
        label = entries.get(event.artist)
        if label is None:
            return
        hidden.symmetric_difference_update({label})
        for ax in fig.axes:
            for artist in ax.get_children():
                keys = getattr(artist, "_legend_keys", None)
                if keys is not None:
                    artist.set_visible(not (keys & hidden))
        for artist in members(label):
            if getattr(artist, "_legend_keys", None) is None:
                artist.set_visible(label not in hidden)
        for handle, text in faded[label]:
            handle.set_alpha(0.25 if label in hidden else 1.0)
            text.set_alpha(0.35 if label in hidden else 1.0)
        if on_toggle is not None:
            on_toggle(set(hidden))
        fig.canvas.draw_idle()

    previous = getattr(fig, "_legend_pick_cid", None)
    if previous is not None:
        fig.canvas.mpl_disconnect(previous)
    fig._legend_pick_cid = fig.canvas.mpl_connect("pick_event", on_pick)
    return fig


# Measurement type -> (factor from the filter's units, display unit). Saved solutions do not
# record measurement units, so they are inferred from the type.
DISPLAY_UNITS = {
    "Range": (1e3, "m"),                  # km -> m
    "Range-rate": (1e6, "mm/s"),          # km/s -> mm/s
    "Doppler": (1e3, "mHz"),              # Hz -> mHz
    "Sequential ranging": (1.0, "RU"),
}

_TYPES = [  # checked in order: "range-rate" before "range"
    (r"range[-_ ]?rate|rangerate", "Range-rate"),
    (r"doppler", "Doppler"),
    (r"s?ranging", "Sequential ranging"),
    (r"range", "Range"),
    (r"ddor|dor", "ΔDOR"),
    (r"opnav|centroid|pixel", "Optical"),
]


def classify(dataset_name):
    """(measurement type, station or instrument) guessed from a dataset name."""
    lowered = dataset_name.lower()
    kind = next((label for pattern, label in _TYPES if re.search(pattern, lowered)), dataset_name)
    station = re.search(r"dss[-_ ]?\d+", lowered)
    observer = station.group(0).upper().replace("_", "-").replace(" ", "-") if station else dataset_name
    return kind, observer


def residuals_from_solution(solution, measurements, kind="postfit"):
    """``{dataset: (epochs_tdb_sec, residual, sigma)}`` from a ``SolutionOD`` and its measurements."""
    source = solution.postfits if kind == "postfit" else solution.prefits
    out = {}
    for spec in measurements:
        name = spec["dataset_name"]
        if name not in source:
            continue
        rows = np.asarray(source[name], dtype=float).reshape(len(source[name]), -1)
        t = np.asarray(spec["observed_meas"].meas_time_et.times.values, dtype=float).ravel()
        if rows.shape[1] >= 3:
            t = rows[:, 2]
        elif len(t) != len(rows):  # edited points are not in the residuals; the epochs must match
            raise ValueError(f"{name}: {len(rows)} residuals but {len(t)} observation epochs")
        out[name] = (t, rows[:, 0], rows[:, 1])
    return out


def _load(source, kind, iteration):
    """Normalize the supported inputs to ``{dataset: (t, r, sigma)}`` plus a description."""
    if isinstance(source, (str, Path)):
        with open(source) as f:
            source = json.load(f)
    if isinstance(source, dict) and "metadata" in source:  # a saved SolutionOD
        block = source[iteration][f"{kind}s"]
        data = {}
        for name, rows in block.items():
            rows = np.asarray(rows, dtype=float)
            if rows.ndim != 2 or rows.shape[1] < 3:
                raise ValueError("saved residuals need the epoch column (Scarabaeus 2.0 solutions)")
            data[name] = (rows[:, 2], rows[:, 0], rows[:, 1])
        label = "final iteration" if iteration == "final" else iteration.replace("iter_", "iteration ")
        return data, label
    return {name: tuple(np.asarray(v, dtype=float) for v in arrays) for name, arrays in source.items()}, ""


def _time_axis(t_all, time_unit):
    t0 = float(np.min(t_all))
    span = (np.max(t_all) - t0) / 3600.0
    if time_unit == "auto":
        time_unit = "day" if span > 48 else "hr" if span > 2 else "min"
    scale = {"day": 86400.0, "hr": 3600.0, "min": 60.0}[time_unit]
    try:  # UTC label when the leap-seconds kernel is loaded
        import scarabaeus as scb

        origin = scb.SpiceManager.et2utc(t0)[:16].replace("T", " ") + " UTC"
    except Exception:
        origin = "the first observation"
    names = {"day": "days", "hr": "hours", "min": "minutes"}
    return t0, scale, f"{names[time_unit]} since {origin}"


def _running_median(x, y, window):
    """Running median of y, computed separately in each pass (gaps of more than 5 median spacings)."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x.size < 5:
        return x, np.full_like(y, np.nan)
    spacing = np.median(np.diff(x))
    breaks = np.nonzero(np.diff(x) > 5 * spacing)[0] + 1
    xs, ys = [], []
    for xp, yp in zip(np.split(x, breaks), np.split(y, breaks)):
        half = max(2, min(window, xp.size) // 2)
        med = np.array([np.median(yp[max(0, k - half): k + half + 1]) for k in range(xp.size)])
        xs += [xp, [np.nan]]
        ys += [med, [np.nan]]
    return np.concatenate(xs), np.concatenate(ys)


def _lag1(series):
    """Pooled lag-1 autocorrelation of normalized residuals and its number of pairs.

    ``series`` is a list of (t, z) per dataset. Pairs are taken in time order within each dataset
    and never across a gap of more than 5 median spacings. For white noise the result is about
    0 ± 1/sqrt(pairs).
    """
    num = den = 0.0
    pairs = 0
    for t, z in series:
        order = np.argsort(t)
        t, z = t[order], z[order] - np.mean(z)
        if t.size < 3:
            continue
        linked = np.diff(t) <= 5 * np.median(np.diff(t))
        num += np.sum((z[:-1] * z[1:])[linked])
        den += np.sum(z**2)
        pairs += int(linked.sum())
    return (num / den if den else np.nan), pairs


def _fmt(value):
    return f"{value:.2f}" if abs(value) < 1e3 else f"{value:.2e}"


def plot_residuals(
    source,
    kind="postfit",
    iteration="final",
    normalized=None,
    datasets=None,
    groups=None,
    units=None,
    time_unit="auto",
    histogram=True,
    trend=False,
    title=None,
    row_height=2.2,
    width=11.0,
):
    """Residuals by measurement type, colored by station, with a histogram of each row.

    Each row is one measurement type (range, Doppler, ...); colors are the stations. The header of
    each row reports n, mean, RMS and χ²/n of the normalized residuals; χ²/n is followed by its
    expected spread for white noise, 1 ± sqrt(2/n). Rows whose RMS is far above σ (pre-fit residuals
    against a poor reference) are drawn as a signal, without noise bands or outlier marks.

    Parameters
    ----------
    source : str, Path, dict
        A saved solution (path or loaded JSON), or ``{dataset: (epochs_tdb_sec, residual, sigma)}``,
        e.g. from :func:`residuals_from_solution`.
    kind : {"postfit", "prefit"}
        Which residuals to read from a saved solution.
    iteration : str
        ``"final"`` or ``"iter_<n>"``, for a saved solution.
    normalized : bool, optional
        Plot residual / σ, or residuals in display units (``DISPLAY_UNITS``). Default: normalized
        for post-fit residuals, units for pre-fit residuals.
    datasets : list of str, optional
        Datasets to include, in this order. Default: all.
    groups : dict, optional
        ``{dataset: (row label, legend label)}`` overriding the guess from the dataset name.
    units : dict, optional
        ``{row label: (factor, unit)}`` overriding ``DISPLAY_UNITS``.
    time_unit : {"auto", "day", "hr", "min"}
    histogram : bool
        Add the histogram panel on the right of each row.
    trend : bool
        Draw a running median per dataset and pass, to show systematic signatures.
    title : str, optional
        Figure title. Default: the kind, iteration, number of observations and χ²/n.

    Click a station in the legend to hide or show it on every row; the row statistics, the
    histograms and the default title are recomputed from the stations still shown. The "beyond 3σ"
    and "running median" entries toggle those layers. Needs an interactive backend
    (``%matplotlib macosx`` or ``qt``); see :func:`interactive_legend`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data, iteration_label = _load(source, kind, iteration)
    if normalized is None:
        normalized = kind == "postfit"
    names = [n for n in (datasets or data) if n in data and len(data[n][0])]
    labels = {n: (groups or {}).get(n) or classify(n) for n in names}
    rows = list(dict.fromkeys(labels[n][0] for n in names))
    observers = list(dict.fromkeys(labels[n][1] for n in names))
    color = {obs: PALETTE[i % len(PALETTE)] for i, obs in enumerate(observers)}
    t0, t_scale, t_label = _time_axis(np.concatenate([data[n][0] for n in names]), time_unit)
    unit_of = {**DISPLAY_UNITS, **(units or {})}
    shade = "#4d4d4d"
    any_outliers = False

    with _styled():
        height = 0.9 + row_height * len(rows)
        fig = plt.figure(figsize=(width, height))
        grid = fig.add_gridspec(len(rows), 2 if histogram else 1, width_ratios=[6, 1] if histogram else [1],
                                wspace=0.02, hspace=0.6, top=1.0 - 0.95 / height)
        axes, row_state = [], []
        for i, row in enumerate(rows):
            ax = fig.add_subplot(grid[i, 0], sharex=axes[0][0] if axes else None)
            hx = fig.add_subplot(grid[i, 1], sharey=ax) if histogram else None
            axes.append((ax, hx))
            members = [n for n in names if labels[n][0] == row]
            factor, unit = unit_of.get(row, (1.0, ""))

            z_row = np.concatenate([data[n][1] / data[n][2] for n in members])
            sigma_row = np.concatenate([data[n][2] * factor for n in members])
            rms_z = float(np.sqrt(np.mean(z_row**2)))
            noise_like = rms_z < 5.0  # residuals at the noise level: draw bands and outliers
            values = {n: ((data[n][0] - t0) / t_scale,
                          data[n][1] / data[n][2] if normalized else data[n][1] * factor) for n in members}
            y_all = np.concatenate([v[1] for v in values.values()])
            band = 1.0 if normalized else float(np.median(sigma_row))

            if noise_like:
                for k, alpha in [(3, 0.07), (1, 0.12)]:
                    ax.axhspan(-k * band, k * band, color=shade, alpha=alpha, lw=0, zorder=0)
                limit = max(4.0 * band, 1.15 * np.percentile(np.abs(y_all), 99.5))
                y_range = (-limit, limit)
            else:  # a signal: fit the axis to the data, keeping zero in view
                low, high = min(0.0, y_all.min()), max(0.0, y_all.max())
                pad = 0.08 * (high - low)
                y_range = (low - pad, high + pad)
            ax.axhline(0, color=shade, lw=0.6, zorder=1)

            for n in members:
                x, y = values[n]
                c = color[labels[n][1]]
                inside = (y >= y_range[0]) & (y <= y_range[1])
                who = labels[n][1]
                _tag(ax.plot(x[inside], y[inside], "o", ms=2.6, mew=0, color=c, alpha=0.8 if not trend else 0.45,
                             zorder=3)[0], who)
                if noise_like:
                    outlier = inside & (np.abs(data[n][1] / data[n][2]) > 3)
                    any_outliers |= bool(outlier.any())
                    _tag(ax.plot(x[outlier], y[outlier], "o", ms=6, mfc="none", mec="#D55E00", mew=0.9, zorder=4)[0],
                         who, "beyond 3σ")
                    for edge, marker in [(y_range[1], "^"), (y_range[0], "v")]:
                        clip = (y > edge) if marker == "^" else (y < edge)
                        _tag(ax.plot(x[clip], np.full(clip.sum(), 0.97 * edge), marker, ms=5, color=c, zorder=4)[0],
                             who)
                if trend:
                    window = max(5, x.size // 15)
                    _tag(ax.plot(*_running_median(x, y, window), color=c, lw=1.6, zorder=5)[0], who, "running median")
            ax.set_ylim(*y_range)
            ax.set_ylabel(f"[{'σ' if normalized else unit}]" if (normalized or unit) else "")
            ax.margins(x=0.01)
            if i < len(rows) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)

            ax.set_title(row, loc="left", fontsize=10, color="#222222", pad=15)
            header = ax.text(0, 1.02, "", transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="#555555")
            row_state.append(dict(members=members, ax=ax, hx=hx, header=header, y_range=y_range, band=band,
                                  noise_like=noise_like, unit=unit, values=values, bars=[]))

            if histogram and not noise_like:  # the distribution of a signal says nothing
                hx.set_visible(False)
            elif histogram:
                if noise_like:
                    grid_y = np.linspace(*y_range, 300)
                    pdf = np.exp(-0.5 * (grid_y / band) ** 2) / (band * np.sqrt(2 * np.pi))
                    hx.plot(pdf, grid_y, color="#222222", lw=1.0)
                    for k, alpha in [(3, 0.07), (1, 0.12)]:
                        hx.axhspan(-k * band, k * band, color=shade, alpha=alpha, lw=0)
                hx.axhline(0, color=shade, lw=0.6)
                hx.set_xticks([])
                hx.tick_params(axis="y", labelleft=False)
                hx.grid(False)
                hx.spines["bottom"].set_visible(False)

        def refresh(hidden):
            """Row statistics, histograms and the title from the datasets still shown."""
            shown_z = []
            for r in row_state:
                shown = [n for n in r["members"] if labels[n][1] not in hidden]
                if r["bars"]:
                    for bar in r["bars"]:
                        bar.remove()
                    r["bars"] = []
                if not shown:
                    r["header"].set_text("all datasets hidden")
                    continue
                z_row = np.concatenate([data[n][1] / data[n][2] for n in shown])
                y_all = np.concatenate([r["values"][n][1] for n in shown])
                shown_z.append(z_row)
                rms_z = float(np.sqrt(np.mean(z_row**2)))
                chi2 = float(np.mean(z_row**2))
                stats = [f"n {z_row.size}", f"mean {'+' if np.mean(z_row) >= 0 else ''}{_fmt(float(np.mean(z_row)))}σ",
                         f"RMS {_fmt(rms_z)}σ"]
                if not normalized and r["unit"]:
                    stats.insert(1, f"RMS {np.sqrt(np.mean(y_all**2)):.3g} {r['unit']}")
                stats.append(f"χ²/n {_fmt(chi2)}" + (f" (white noise: 1 ± {np.sqrt(2 / z_row.size):.2f})"
                                                      if r["noise_like"] else ""))
                if r["noise_like"]:  # whiteness: χ²/n can be 1 while the residuals carry a signature
                    rho, pairs = _lag1([(data[n][0], data[n][1] / data[n][2]) for n in shown])
                    if pairs >= 10:
                        stats.append(f"lag-1 corr {rho:+.2f} (white: 0 ± {1 / np.sqrt(pairs):.2f})")
                beyond = int(np.sum(np.abs(z_row) > 3))
                if r["noise_like"] and beyond:
                    stats.append(f"{beyond} beyond 3σ")
                if hidden & {labels[n][1] for n in r["members"]}:
                    stats.append("shown datasets only")
                r["header"].set_text("  ·  ".join(stats))
                if histogram and r["noise_like"]:
                    bins = np.linspace(*r["y_range"], 41)
                    r["bars"] = list(r["hx"].hist(y_all, bins=bins, orientation="horizontal", density=True,
                                                  color="#9a9a9a", alpha=0.6, lw=0)[2])
                    r["hx"].relim()
                    r["hx"].autoscale_view(scalex=True, scaley=False)
            if auto_title is not None:
                z = np.concatenate(shown_z) if shown_z else np.zeros(0)
                fig._suptitle.set_text(auto_title(z))


        axes[-1][0].set_xlabel(t_label)
        if histogram:
            axes[-1][1].set_xlabel("histogram\n(curve: white noise)", fontsize=8, color="#555555")

        handles = [Line2D([], [], ls="", marker="o", ms=5, color=color[o], label=o) for o in observers]
        if any_outliers:
            handles.append(Line2D([], [], ls="", marker="o", ms=6, mfc="none", mec="#D55E00", label="beyond 3σ"))
        if trend:
            handles.append(Line2D([], [], color="#555555", lw=1.6, label="running median"))
        auto_title = None
        if title is None:
            what = {"postfit": "Post-fit residuals", "prefit": "Pre-fit residuals"}[kind]

            def auto_title(z):
                return "  ·  ".join([what] + ([iteration_label] if iteration_label else [])
                                    + [f"{z.size} observations"]
                                    + ([f"χ²/n = {_fmt(float(np.mean(z**2)))}"] if z.size else []))
        top = 1.0 - 0.1 / fig.get_figheight()
        fig.suptitle(title or "", x=0.125, ha="left", y=top, fontsize=11)
        refresh(set())
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.9, top + 0.005), ncol=len(handles),
                   handletextpad=0.2, columnspacing=1.2)
        interactive_legend(fig, on_toggle=refresh)
    return fig


# ------------------------------------------------------------------------------------------------
# State errors and their consistency with the covariance
# ------------------------------------------------------------------------------------------------

# χ² with 3 degrees of freedom: mean 3, 95 % and 99.73 % (3σ-equivalent) quantiles
CHI2_3 = {"mean": 3.0, "95%": 7.815, "99.73%": 14.16}


FRAMES = ("LOS", "RTN", "J2000")


def _frame_axes(frame, center):
    """Center of the rotation, axis labels and a description for a frame name.

    ``"LOS"``: the line of sight from Earth and two axes spanning the plane of sky, built as the
    RTN axes of the spacecraft's motion relative to Earth (PoS 1 along that motion, PoS 2 normal to
    it). What Earth-based tracking measures and what it infers. ``"RTN"``: radial, transverse and
    normal axes of the orbit about ``center`` (the Sun by default), the axes in which a trajectory
    error evolves. ``"J2000"``: no rotation.
    """
    if frame == "LOS":
        return "EARTH", ("LOS", "PoS 1", "PoS 2"), "line of sight (LOS) and plane of sky (PoS) from Earth"
    if frame == "RTN":
        return center, ("R", "T", "N"), f"RTN about the {center.title()}"
    if frame == "J2000":
        return None, ("x", "y", "z"), "J2000 axes"
    raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")


def state_errors(solution, truth, epochs, frame="LOS", center="SUN", origin="SUN", **mapping):
    """Estimate minus truth, and the estimate's covariance, at ``epochs``.

    Parameters
    ----------
    solution : SolutionOD
    truth : Body
        A body whose trajectory is in the kernel pool (a truth SPK, or a reference ephemeris such
        as the flight reconstruction).
    epochs : EpochArray
        Any epochs, inside or after the data arc.
    frame : {"LOS", "RTN", "J2000"}
        ``"LOS"`` (default): the line of sight from Earth and the plane of sky, what Earth-based
        radiometric tracking does and does not measure. ``"RTN"``: the axes of the orbit about
        ``center``, in which a trajectory error evolves (along-track drift, radial offsets).
    center : str
        Center of the orbit for ``frame="RTN"``. Defaults to the Sun.
    origin : str
        Origin of the solution's state (the central body of its propagation).
    **mapping
        Passed to ``SolutionOD.propagate_state_covariance`` (``use_smoothed``, ``lookback_s``,
        ``use_all_data``, ...).

    Returns
    -------
    dict with ``t`` (TDB seconds), ``error`` (n, 6), ``covariance`` (n, 6, 6), ``axes`` (labels)
    and ``measurement_epochs`` (the solution's timestamps, to mark the tracking periods).
    """
    import scarabaeus as scb

    pairs = solution.propagate_state_covariance(epochs, **mapping)
    x = np.array([np.asarray(state.values0.values, dtype=float).ravel()[:6] for state, _ in pairs])
    P = np.array([np.asarray(C, dtype=float)[:6, :6] for _, C in pairs])
    t = np.asarray(epochs.times.values, dtype=float).ravel()
    # one query per epoch: get_state takes one epoch per call (EpochArray support is planned)
    x_true = np.array([truth.get_state(epochs[k], reference_frame="J2000", origin=origin).quantity.values
                       for k in range(t.size)], dtype=float)
    error = x - x_true
    axes = ("x", "y", "z")
    center, axes, frame_text = _frame_axes(frame, center)
    if center is not None:
        relative = np.array([truth.get_state(epochs[k], reference_frame="J2000", origin=center).quantity.values
                             for k in range(t.size)], dtype=float)
        for k in range(t.size):
            A = scb.DynamicalUtils.rtn(relative[k, :3], relative[k, 3:])[0].T  # rows: R, T, N
            R6 = np.kron(np.eye(2), A)  # components along the axes (no transport terms)
            error[k] = R6 @ error[k]
            P[k] = R6 @ P[k] @ R6.T
    return {"t": t, "error": error, "covariance": P, "axes": axes, "frame": frame, "center": center,
            "frame_text": frame_text, "measurement_epochs": np.asarray(solution.timestamps, dtype=float)}


def _length_unit(values_km, quantity):
    """Display factor and unit for errors given in km or km/s."""
    peak = np.nanmax(np.abs(values_km))
    if quantity == "position":
        return (1e3, "m") if peak < 2.0 else (1.0, "km")
    return (1e6, "mm/s") if peak < 2e-3 else (1e3, "m/s")


def _passes(t_meas, gap):
    """(start, end) of each group of measurement epochs separated by more than ``gap`` seconds."""
    t_meas = np.sort(t_meas)
    breaks = np.nonzero(np.diff(t_meas) > gap)[0] + 1
    return [(seg[0], seg[-1]) for seg in np.split(t_meas, breaks)]


def plot_consistency(errors, quantity="both", physical=True, style="line", time_unit="auto", title=None,
                     label=None, width=11.0):
    """Is the estimation error consistent with the covariance?

    Rows, sharing the time axis, for position and velocity (``quantity="both"``, the default) or
    one of them:

    1. (``physical=True``) the error of each axis inside its ±3σ band, in display units, one row
       per quantity;
    2. the normalized errors e/σ of the three axes, with ±1σ and ±3σ bands and their histogram,
       one row per quantity;
    3. the Mahalanobis distance χ² = δᵀ P⁻¹ δ of the 3-D error of each quantity (position solid,
       velocity dashed), against the χ²(3) mean (3) and its 95 % and 99.73 % bounds, with its
       histogram against the χ²(3) density.

    Tracking periods (from the solution's measurement epochs) are shaded and the last measurement
    is marked, so the data arc and the prediction read at a glance. The samples are correlated in
    time, so the histograms are a visual check, not a statistical test.

    Parameters
    ----------
    errors : dict
        The output of :func:`state_errors`.
    quantity : {"both", "position", "velocity"}
    physical : bool
        Include the rows of physical errors.
    style : {"line", "scatter"}
        Lines through the evaluated epochs (default), or markers only. Scatter shows the tracking
        pattern when ``errors`` is evaluated at the measurement epochs, but the statistics are then
        averaged over those epochs, weighting the tracked periods; a uniform grid weights every
        epoch equally.
    time_unit : {"auto", "day", "hr", "min"}
    title, label : str, optional
        Figure title, and a short name of the solution for the default title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    style_name = style
    draw = {"line": dict(lw=1.4),
            "scatter": dict(lw=0, marker="o", ms=2.6, mew=0, ls="")}[style]
    quantities = ["position", "velocity"] if quantity == "both" else [quantity]
    t = errors["t"]
    data = {}
    for q in quantities:
        block = slice(0, 3) if q == "position" else slice(3, 6)
        e = errors["error"][:, block]
        P = errors["covariance"][:, block, block]
        sigma = np.sqrt(np.einsum("nii->ni", P))
        data[q] = dict(e=e, sigma=sigma, z=e / sigma,
                       chi2=np.einsum("ni,ni->n", e, np.linalg.solve(P, e[..., None])[..., 0]))
    t0, t_scale, t_label = _time_axis(t, time_unit)
    x = (t - t0) / t_scale
    passes = _passes(errors["measurement_epochs"], gap=3 * 3600.0)
    cutoff = (np.max(errors["measurement_epochs"]) - t0) / t_scale
    axis_colors = PALETTE[:3]
    chi2_styles = {"position": ("#000000", "-"), "velocity": ("#555555", "--")}
    shade = "#4d4d4d"

    def mark_data(ax):
        for a, b in passes:
            ax.axvspan((a - t0) / t_scale, (b - t0) / t_scale, color="#56B4E9", alpha=0.10, lw=0, zorder=0)
        if cutoff < x.max():
            ax.axvline(cutoff, color="#333333", lw=0.9, ls="--", zorder=1)

    n_q = len(quantities)
    with _styled():
        heights = ([1.0] * n_q if physical else []) + [1.0] * n_q + [1.15]
        height = 0.9 + 2.0 * sum(heights)
        fig = plt.figure(figsize=(width, height))
        outer = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.75, top=1.0 - 0.95 / height,
                                 bottom=0.45 / height)
        lower = [outer[r].subgridspec(1, 2, width_ratios=[6, 1], wspace=0.02) for r in range(len(heights))]
        row = 0
        shared = None

        if physical:
            for q in quantities:
                e, sigma = data[q]["e"], data[q]["sigma"]
                for j, name in enumerate(errors["axes"]):
                    factor, unit = _length_unit(np.concatenate([e[:, j], 3 * sigma[:, j]]), q)
                    ax = fig.add_subplot(outer[row].subgridspec(1, 3, wspace=0.3)[0, j], sharex=shared)
                    shared = shared or ax
                    mark_data(ax)
                    ax.fill_between(x, -3 * sigma[:, j] * factor, 3 * sigma[:, j] * factor, color=shade, alpha=0.10,
                                    lw=0)
                    ax.plot(x, e[:, j] * factor, color=axis_colors[j], **draw)
                    ax.axhline(0, color=shade, lw=0.6)
                    ax.set_title(f"{q} {name}  (±3σ shaded)", loc="left", fontsize=9.5, color="#222222")
                    ax.set_ylabel(f"[{unit}]")
                    plt.setp(ax.get_xticklabels(), visible=False)
                row += 1

        # Normalized errors, one row per quantity
        histograms = []
        for k, q in enumerate(quantities):
            z = data[q]["z"]
            ax_z = fig.add_subplot(lower[row][0, 0], sharex=shared)
            shared = shared or ax_z
            mark_data(ax_z)
            for n_sig, alpha in [(3, 0.07), (1, 0.12)]:
                ax_z.axhspan(-n_sig, n_sig, color=shade, alpha=alpha, lw=0, zorder=0)
            ax_z.axhline(0, color=shade, lw=0.6)
            for j, name in enumerate(errors["axes"]):
                ax_z.plot(x, z[:, j], color=axis_colors[j], label=name, **draw)
            z_limit = max(4.0, 1.1 * np.nanmax(np.abs(z)))
            ax_z.set_ylim(-z_limit, z_limit)
            ax_z.set_ylabel("e / σ")
            inside = np.mean(np.abs(z) <= 3, axis=0)
            ax_z.set_title(f"Normalized {q} error", loc="left", fontsize=10, color="#222222", pad=15)
            ax_z.text(0, 1.02, "  ·  ".join(f"{name}: max |e/σ| {np.nanmax(np.abs(z[:, j])):.1f}, "
                                            f"{inside[j]:.0%} within 3σ" for j, name in enumerate(errors["axes"])),
                      transform=ax_z.transAxes, fontsize=8, color="#555555", va="bottom")
            if k == 0:
                data_handles = [plt.Rectangle((0, 0), 1, 1, color="#56B4E9", alpha=0.25, label="tracking"),
                                Line2D([], [], color="#333333", lw=0.9, ls="--", label="last measurement")]
                ax_z.legend(handles=ax_z.get_legend_handles_labels()[0] + data_handles, loc="lower right", ncol=5,
                            bbox_to_anchor=(1.0, 1.08), handlelength=1.5)
            plt.setp(ax_z.get_xticklabels(), visible=False)
            hz = fig.add_subplot(lower[row][0, 1], sharey=ax_z)
            bins = np.linspace(-z_limit, z_limit, 41)
            hz.hist(z.ravel(), bins=bins, orientation="horizontal", density=True, color="#9a9a9a", alpha=0.6, lw=0)
            grid_z = np.linspace(-z_limit, z_limit, 300)
            hz.plot(np.exp(-0.5 * grid_z**2) / np.sqrt(2 * np.pi), grid_z, color="#222222", lw=1.0)
            for n_sig, alpha in [(3, 0.07), (1, 0.12)]:
                hz.axhspan(-n_sig, n_sig, color=shade, alpha=alpha, lw=0)
            histograms.append(hz)
            row += 1

        # Mahalanobis χ² of each quantity (3 dof each, so they share the bounds)
        all_chi2 = np.concatenate([data[q]["chi2"] for q in quantities])
        ax_c = fig.add_subplot(lower[row][0, 0], sharex=shared)
        mark_data(ax_c)
        low = max(1e-2, 0.5 * np.nanmin(all_chi2[all_chi2 > 0])) if np.any(all_chi2 > 0) else 1e-2
        c_range = (min(low, 0.3), max(30.0, 2.0 * np.nanmax(all_chi2)))
        ax_c.axhspan(CHI2_3["95%"], CHI2_3["99.73%"], color="#E69F00", alpha=0.10, lw=0)
        ax_c.axhspan(CHI2_3["99.73%"], c_range[1], color="#D55E00", alpha=0.10, lw=0)
        ax_c.axhline(CHI2_3["mean"], color=shade, lw=0.8, ls=":")
        stats = []
        for q in quantities:
            chi2 = data[q]["chi2"]
            color, line_style = chi2_styles[q]
            ax_c.plot(x, chi2, color=color, label=q, **{**draw, "ls": "" if style_name == "scatter" else line_style})
            stats.append(f"{q}: mean {np.nanmean(chi2):.3g}, {np.mean(chi2 <= CHI2_3['95%']):.0%} below 95 %, "
                         f"max {np.nanmax(chi2):.3g}")
        ax_c.set_yscale("log")
        ax_c.set_ylim(*c_range)
        ax_c.set_ylabel("χ²  (3 dof)")
        ax_c.set_xlabel(t_label)
        ax_c.set_title("Mahalanobis distance of the error", loc="left", fontsize=10, color="#222222", pad=27)
        ax_c.text(0, 1.02, "  ·  ".join(stats) + "\nconsistent: mean 3, median 2.37, 95 % of epochs below the 95 % "
                  "bound  ·  shaded: beyond 95 % / 99.73 %", transform=ax_c.transAxes, fontsize=8, color="#555555",
                  va="bottom", linespacing=1.4)
        if n_q > 1:
            ax_c.legend(loc="lower right", ncol=2, handlelength=2.2)
        hc = fig.add_subplot(lower[row][0, 1], sharey=ax_c)
        log_bins = np.logspace(np.log10(c_range[0]), np.log10(c_range[1]), 41)
        hc.hist(np.clip(all_chi2, *c_range), bins=log_bins, orientation="horizontal", density=True, color="#9a9a9a",
                alpha=0.6, lw=0)
        grid_c = np.logspace(np.log10(c_range[0]), np.log10(c_range[1]), 300)
        hc.plot(np.sqrt(grid_c / (2 * np.pi)) * np.exp(-grid_c / 2), grid_c, color="#222222", lw=1.0)  # χ²(3) pdf
        hc.axhspan(CHI2_3["95%"], CHI2_3["99.73%"], color="#E69F00", alpha=0.10, lw=0)
        hc.axhspan(CHI2_3["99.73%"], c_range[1], color="#D55E00", alpha=0.10, lw=0)
        for h in histograms + [hc]:
            h.set_xticks([])
            h.tick_params(axis="y", labelleft=False)
            h.grid(False)
            h.spines["bottom"].set_visible(False)
        hc.set_xlabel("histogram\n(curve: consistent)", fontsize=8, color="#555555")

        if title is None:
            frame_text = errors.get("frame_text", "J2000 axes")
            what = "State" if n_q > 1 else quantity.title()
            title = "  ·  ".join(([label] if label else []) + [f"{what} error vs. covariance", frame_text])
        top = 1.0 - 0.1 / height
        fig.suptitle(title, x=0.125, ha="left", y=top, fontsize=11)
    return fig


# ------------------------------------------------------------------------------------------------
# Residuals pass by pass
# ------------------------------------------------------------------------------------------------

def _split_passes(data, labels, gap):
    """``[(observer, start, end, {dataset: index mask})]`` in time order.

    A pass is a run of one observer's epochs, all measurement types together, without a gap of
    more than ``gap`` seconds.
    """
    passes = []
    for observer in dict.fromkeys(obs for _, obs in labels.values()):
        members = [n for n in data if labels[n][1] == observer]
        t_all = np.sort(np.concatenate([data[n][0] for n in members]))
        breaks = np.nonzero(np.diff(t_all) > gap)[0] + 1
        for segment in np.split(t_all, breaks):
            a, b = segment[0], segment[-1]
            passes.append((observer, a, b, {n: (data[n][0] >= a) & (data[n][0] <= b) for n in members}))
    return sorted(passes, key=lambda p: p[1])


def plot_residual_summary(source, kind="postfit", iteration="final", datasets=None, groups=None, gap_hours=2.0,
                          title=None, width=11.0, row_height=0.24):
    """Residual statistics pass by pass.

    One row per pass (an observer's continuous tracking; a gap of more than ``gap_hours`` starts a
    new pass), in time order. For each measurement type, two columns of the normalized residuals:

    - **mean** with its white-noise band ±3/√n: a mean outside it is a bias the fit did not absorb
      (a calibration error, a mismodeled delay, a dynamics error that differs from pass to pass);
    - **spread** (standard deviation) against 1 and its band 1 ± 3/√(2n): too large means the
      data are noisier than their weight, too small that they are over-weighted.

    Rows that fail either test are drawn in vermillion; the label of a pass gives its date, observer
    and number of points.

    Parameters
    ----------
    source, kind, iteration, datasets, groups
        As in :func:`plot_residuals`.
    gap_hours : float
        A gap longer than this starts a new pass.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data, iteration_label = _load(source, kind, iteration)
    names = [n for n in (datasets or data) if n in data and len(data[n][0])]
    data = {n: data[n] for n in names}
    labels = {n: (groups or {}).get(n) or classify(n) for n in names}
    types = list(dict.fromkeys(labels[n][0] for n in names))
    observers = list(dict.fromkeys(labels[n][1] for n in names))
    color = {obs: PALETTE[i % len(PALETTE)] for i, obs in enumerate(observers)}
    passes = _split_passes(data, labels, gap_hours * 3600.0)
    flag_color = "#D55E00"
    shade = "#4d4d4d"

    try:
        import scarabaeus as scb

        def day(et):
            return scb.SpiceManager.et2utc(et)[5:10]  # MM-DD
    except Exception:
        def day(et):
            return f"{(et - passes[0][1]) / 86400:.1f} d"

    # Statistics per (pass, type): mean, std, n of the normalized residuals
    stats = {}
    for p, (observer, a, b, masks) in enumerate(passes):
        for kind_label in types:
            z = np.concatenate([(data[n][1] / data[n][2])[masks[n]] for n in masks if labels[n][0] == kind_label]
                               or [np.empty(0)])
            if z.size:
                stats[p, kind_label] = (float(np.mean(z)), float(np.std(z, ddof=1)) if z.size > 1 else np.nan, z.size)

    with _styled():
        n_rows = len(passes)
        height = 1.6 + row_height * n_rows
        fig = plt.figure(figsize=(width, height))
        outer = fig.add_gridspec(1, len(types), wspace=0.08, top=1.0 - 0.7 / height, bottom=0.9 / height)
        y = np.arange(n_rows)[::-1]
        row_labels = []
        for p, (observer, a, b, masks) in enumerate(passes):
            n_pass = sum(int(m.sum()) for m in masks.values())
            row_labels.append(f"{day(a)}  {observer}  ({n_pass})")
        flagged_rows = set()
        first_axis = None
        for c, kind_label in enumerate(types):
            inner = outer[0, c].subgridspec(1, 2, width_ratios=[1.6, 1], wspace=0.08)
            ax_m = fig.add_subplot(inner[0, 0], sharey=first_axis)
            ax_s = fig.add_subplot(inner[0, 1], sharey=ax_m)
            first_axis = first_axis or ax_m
            means, spreads = [], []
            for p in range(n_rows):
                if (p, kind_label) not in stats:
                    continue
                mean, std, n = stats[p, kind_label]
                mean_band, std_band = 3 / np.sqrt(n), 3 / np.sqrt(2 * max(n - 1, 1))
                bad_mean = abs(mean) > mean_band
                bad_std = np.isfinite(std) and abs(std - 1) > std_band
                if bad_mean or bad_std:
                    flagged_rows.add(p)
                c_obs = color[passes[p][0]]
                # Mean with the band white noise allows for this n
                ax_m.plot([-mean_band, mean_band], [y[p], y[p]], color="#cfcfcf", lw=3.2, solid_capstyle="butt",
                          zorder=1)
                ax_m.plot(mean, y[p], "o", ms=4.5, color=flag_color if bad_mean else c_obs, zorder=3,
                          mec="white", mew=0.5)
                # Spread against 1 ± its band
                if np.isfinite(std):
                    ax_s.plot([1 - std_band, 1 + std_band], [y[p], y[p]], color="#cfcfcf", lw=3.2,
                              solid_capstyle="butt", zorder=1)
                    ax_s.plot(std, y[p], "D", ms=3.8, color=flag_color if bad_std else c_obs, zorder=3,
                              mec="white", mew=0.5)
                means.append(mean)
                spreads.append(std)
            # A thin line through each observer's means shows drifts from pass to pass
            for observer in observers:
                rows_obs = [p for p in range(n_rows) if passes[p][0] == observer and (p, kind_label) in stats]
                if len(rows_obs) > 2:
                    ax_m.plot([stats[p, kind_label][0] for p in rows_obs], [y[p] for p in rows_obs],
                              color=color[observer], lw=0.7, alpha=0.45, zorder=2)
            ax_m.axvline(0, color=shade, lw=0.7)
            ax_s.axvline(1, color=shade, lw=0.7)
            m_limit = max(1.0, 1.15 * np.nanmax(np.abs(means)))
            ax_m.set_xlim(-m_limit, m_limit)
            ax_s.set_xlim(0, max(2.0, 1.1 * np.nanmax(spreads)))
            for ax in (ax_m, ax_s):
                ax.grid(axis="y", visible=False)
                ax.tick_params(axis="y", length=0)
            ax_m.set_xlabel("mean [σ]")
            ax_s.set_xlabel("spread [σ]")
            ax_m.set_title(kind_label, loc="left", fontsize=10, color="#222222")
            plt.setp(ax_s.get_yticklabels(), visible=False)
            if c > 0:
                plt.setp(ax_m.get_yticklabels(), visible=False)
        first_axis.set_yticks(y)
        first_axis.set_yticklabels(row_labels, fontsize=8, family="monospace")
        for p in flagged_rows:
            first_axis.get_yticklabels()[p].set_color(flag_color)
        first_axis.set_ylim(-0.7, n_rows - 0.3)

        if title is None:
            what = {"postfit": "Post-fit residuals by pass", "prefit": "Pre-fit residuals by pass"}[kind]
            title = "  ·  ".join([what] + ([iteration_label] if iteration_label else [])
                                 + [f"{n_rows} pass{'es' if n_rows != 1 else ''}", f"{len(flagged_rows)} flagged"])
        top = 1.0 - 0.1 / height
        fig.suptitle(title, x=0.125, ha="left", y=top, fontsize=11)
        handles = [Line2D([], [], ls="", marker="o", ms=5, color=color[o], label=o) for o in observers]
        handles += [Line2D([], [], color="#cfcfcf", lw=3.2, label="white-noise band (3σ of the statistic)"),
                    Line2D([], [], ls="", marker="o", ms=5, color=flag_color, label="outside the band")]
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.12, 0.05 / height), ncol=len(handles),
                   handletextpad=0.3, columnspacing=1.2)
    return fig


# ------------------------------------------------------------------------------------------------
# What the data taught the filter
# ------------------------------------------------------------------------------------------------

def _solution_at_epoch(record, frame="LOS", center="SUN", origin="SUN", truth=None, truth_values=None):
    """A priori and solution of a saved solution at its epoch, position and velocity in ``frame``.

    Returns a dict with ``prior_x``, ``prior_P``, ``post_x``, ``post_P``, ``true_x``, ``components``
    and ``axes`` (the labels of the position and velocity axes).
    """
    meta, final = record["metadata"], record["final"]["epoch_solution"]
    prior_x = np.asarray(meta["apriori"]["state"], dtype=float)
    prior_P = np.asarray(meta["apriori"]["covariance"], dtype=float)
    post_x = np.asarray(final["state"], dtype=float)
    # With considered parameters the data-only covariance is not the honest one (and in 2.0 it is not
    # even positive semi-definite, see PLAN.md); the consider covariance includes their uncertainty.
    consider = final.get("covariance_consider")
    post_P = np.asarray(consider if consider is not None else final["covariance"], dtype=float)
    epoch = float(final["epoch"])
    true_x = np.full(prior_x.size, np.nan)

    components = meta["state_components"]
    index = {c["name"]: c for c in components}
    has_pv = "position" in index and "velocity" in index
    axes = ("x", "y", "z")
    if has_pv:
        import scarabaeus as scb

        epochs = scb.EpochArray([epoch], sys="TDB")
        pv = np.r_[index["position"]["index_start"]:index["position"]["index_end"],
                   index["velocity"]["index_start"]:index["velocity"]["index_end"]]
        if truth is not None:
            true_x[pv] = truth.get_state(epochs, reference_frame="J2000", origin=origin).quantity.values
        center, frame_axes, _ = _frame_axes(frame, center)
        if center is not None:
            center_state = scb.CelestialBody.from_constants(center).get_state(
                epochs, reference_frame="J2000", origin=origin).quantity.values
            relative = post_x[pv] - center_state
            A = scb.DynamicalUtils.rtn(relative[:3], relative[3:])[0].T
            R = np.eye(prior_x.size)
            R[np.ix_(pv, pv)] = np.kron(np.eye(2), A)
            # Rotate deviations about the a priori, so the a-priori value maps to itself
            unknown = np.isnan(true_x)  # elements without a truth stay without one
            prior_x, post_x, true_x = R @ prior_x, R @ post_x, R @ np.nan_to_num(true_x, nan=0.0)
            true_x[unknown] = np.nan
            prior_P, post_P = R @ prior_P @ R.T, R @ post_P @ R.T
            axes = frame_axes
    for c in components:
        if c["name"] in (truth_values or {}):
            true_x[c["index_start"]:c["index_end"]] = np.atleast_1d(truth_values[c["name"]])

    return dict(prior_x=prior_x, prior_P=prior_P, post_x=post_x, post_P=post_P, true_x=true_x,
                components=components, axes=axes)


def _rows_from_solution(record, frame, center, origin, truth, truth_values):
    """One dict per state element: label, prior/posterior value and σ, truth, display unit."""
    epoch_solution = _solution_at_epoch(record, frame, center, origin, truth, truth_values)
    prior_x, prior_P, post_x, post_P, true_x = (epoch_solution[k] for k in
                                                ("prior_x", "prior_P", "post_x", "post_P", "true_x"))
    components, axes = epoch_solution["components"], epoch_solution["axes"]

    rows = []
    for c in components:
        name, start, dim = c["name"], c["index_start"], c["dim"]
        unit = c["units"]["name"] if c.get("units") else ""
        factor = 1.0
        if name == "position":
            factor, unit = 1.0, "km"
        elif name == "velocity":
            factor, unit = 1e6, "mm/s"
        elif "RU" in name:
            unit = "RU"
        elif unit == "unitless":
            unit = ""
        for k in range(dim):
            i = start + k
            if name in ("position", "velocity"):
                label = f"{name} {axes[k]}"
            else:
                label = name + (f"[{k}]" if dim > 1 else "")
            rows.append(dict(label=label, group=name, owner=c.get("owner", ""), considered=c["estimation"] == "considered",
                             prior=prior_x[i], prior_sigma=np.sqrt(prior_P[i, i]), post=post_x[i],
                             post_sigma=np.sqrt(post_P[i, i]), truth=true_x[i], factor=factor, unit=unit))
    return rows


def _value_text(value, sigma, unit):
    """'12.3 ± 0.4 km' with a precision set by σ."""
    if not np.isfinite(sigma) or sigma == 0:
        return f"{value:.4g} {unit}".strip()
    digits = max(0, -int(np.floor(np.log10(sigma))) + 1)
    if abs(value) >= 1e5 or digits > 6:
        return f"{value:.3e} ± {sigma:.2g} {unit}".strip()
    return f"{value:.{digits}f} ± {sigma:.2g} {unit}".strip()


def plot_parameter_learning(source, truth=None, truth_values=None, frame="LOS", center="SUN", origin="SUN",
                            components=None, title=None, width=11.0, row_height=0.32):
    """What did the data teach the filter?

    One row per estimated (or considered) quantity at the solution epoch, grouped by state
    component. Position and velocity are shown along the line of sight from Earth and in the plane
    of sky by default (``frame="LOS"``), where what tracking determines and what it does not is
    plain.

    - **Left: σ_post / σ_prior** on a log scale. Near 1, the data added nothing and the result is
      the a priori (shaded "prior-dominated"); far below 1, the data determined the quantity.
    - **Middle: the move**, (value − a priori) / σ_prior. The a-priori ±1σ is the gray band, the
      posterior ±1σ the dark bar, the estimate the dot, the truth (when given) the diamond.
    - **Right:** a priori → estimate, in physical units.

    With considered parameters the posterior σ include their uncertainty (the consider covariance).

    Parameters
    ----------
    source : str, Path or dict
        A saved solution (Scarabaeus 2.0 JSON, which stores the a priori and the solution at its
        epoch).
    truth : Body, optional
        Body whose ephemeris gives the true position and velocity (simulations).
    truth_values : dict, optional
        ``{component name: true value(s)}`` for the parameters, e.g. ``{"eta_srp": 11 / 15}``.
    frame : {"LOS", "RTN", "J2000"}
        As in :func:`state_errors`.
    center, origin : str
        Center of the orbit for ``frame="RTN"`` (the Sun by default), and origin of the solution's state.
    components : list of str, optional
        State components to show, in order. Default: all.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if isinstance(source, (str, Path)):
        with open(source) as f:
            source = json.load(f)
    rows = _rows_from_solution(source, frame, center, origin, truth, truth_values)
    if components:
        rows = [r for c in components for r in rows if r["group"] == c]
    n = len(rows)
    y = np.arange(n)[::-1]
    shade, dark, flag = "#4d4d4d", "#0072B2", "#D55E00"

    ratio = np.array([r["post_sigma"] / r["prior_sigma"] for r in rows])
    move = np.array([(r["post"] - r["prior"]) / r["prior_sigma"] for r in rows])
    post_band = np.array([r["post_sigma"] / r["prior_sigma"] for r in rows])
    true_move = np.array([(r["truth"] - r["prior"]) / r["prior_sigma"] for r in rows])

    with _styled():
        height = 1.5 + row_height * n
        fig = plt.figure(figsize=(width, height))
        grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 1.35], wspace=0.05,
                                top=1.0 - 0.8 / height, bottom=0.75 / height, left=0.17, right=0.98)
        ax_r = fig.add_subplot(grid[0, 0])
        ax_m = fig.add_subplot(grid[0, 1], sharey=ax_r)
        ax_t = fig.add_subplot(grid[0, 2], sharey=ax_r)

        # Group separators
        for k in range(1, n):
            if rows[k]["group"] != rows[k - 1]["group"]:
                for ax in (ax_r, ax_m, ax_t):
                    ax.axhline(y[k] + 0.5, color="#bbbbbb", lw=0.6)

        # σ_post / σ_prior
        smallest = np.nanmin(ratio[ratio > 0]) if np.any(ratio > 0) else 1e-2
        low = 10.0 ** np.floor(np.log10(0.5 * smallest))
        ax_r.axvspan(0.7, 1.5, color=shade, alpha=0.10, lw=0)
        for k, r in enumerate(rows):
            c = "#9a9a9a" if r["considered"] else dark
            ax_r.plot([ratio[k], 1.0], [y[k], y[k]], color=c, lw=2.4, alpha=0.35, solid_capstyle="butt")
            ax_r.plot(ratio[k], y[k], "o", ms=5, color=c, mec="white", mew=0.5, zorder=3)
        ax_r.set_xscale("log")
        ax_r.set_xlim(low, 1.5)
        ax_r.axvline(1.0, color=shade, lw=0.7)
        ax_r.set_xlabel("σ posterior / σ a priori")
        ax_r.set_title("Information from the data", loc="left", fontsize=10, color="#222222")

        # Move from the a priori
        limit = max(3.0, 1.15 * np.nanmax(np.abs(np.r_[move + post_band, move - post_band,
                                                         true_move[np.isfinite(true_move)]])))
        ax_m.axvspan(-1, 1, color=shade, alpha=0.10, lw=0)
        ax_m.axvline(0, color=shade, lw=0.7)
        for k, r in enumerate(rows):
            c = "#9a9a9a" if r["considered"] else dark
            ax_m.plot([move[k] - post_band[k], move[k] + post_band[k]], [y[k], y[k]], color=c, lw=3.0,
                      solid_capstyle="butt", alpha=0.55)
            ax_m.plot(move[k], y[k], "o", ms=5, color=c, mec="white", mew=0.5, zorder=3)
            if np.isfinite(true_move[k]):
                off = abs(move[k] - true_move[k]) > 3 * post_band[k]
                ax_m.plot(true_move[k], y[k], "D", ms=4.5, mfc="none", mec=flag if off else "#222222", mew=1.1, zorder=4)
        ax_m.set_xlim(-limit, limit)
        ax_m.set_xlabel("(value − a priori) / σ a priori")
        ax_m.set_title("Estimate against the a priori", loc="left", fontsize=10, color="#222222")

        # Values
        ax_t.set_xlim(0, 1)
        ax_t.axis("off")
        for k, r in enumerate(rows):
            f, u = r["factor"], r["unit"]
            if r["group"] in ("position", "velocity"):  # changes: absolute positions are ~1e8 km
                text = f"{(r['post'] - r['prior']) * f:+8.3g} ± {r['post_sigma'] * f:<7.2g} {u:<4s} (σ a priori {r['prior_sigma'] * f:.3g})"
            else:
                text = (f"{_value_text(r['prior'] * f, r['prior_sigma'] * f, '')}  →  "
                        f"{_value_text(r['post'] * f, r['post_sigma'] * f, u)}")
            if r["considered"]:
                text += "   (considered)"
            ax_t.text(0.03, y[k], text, va="center", fontsize=7.5, family="monospace", color="#333333")
        ax_t.set_title("a priori → estimate\n(position, velocity: change from the a priori)", loc="left", fontsize=8.5,
                       color="#555555")

        for ax in (ax_r, ax_m):
            ax.grid(axis="y", visible=False)
            ax.tick_params(axis="y", length=0)
        ax_r.set_yticks(y)
        ax_r.set_yticklabels([r["label"] for r in rows], fontsize=8.5)
        plt.setp(ax_m.get_yticklabels(), visible=False)
        ax_r.set_ylim(-0.7, n - 0.3 + 0.9)

        if title is None:
            frame_text = "position and velocity: " + _frame_axes(frame, center)[2]
            title = f"What the data determined  ·  {frame_text}"
        top = 1.0 - 0.1 / height
        fig.suptitle(title, x=0.17, ha="left", y=top, fontsize=11)
        handles = [Line2D([], [], ls="", marker="o", ms=5, color=dark, label="estimate"),
                   Line2D([], [], color=dark, lw=3, alpha=0.55, label="± 1σ posterior"),
                   plt.Rectangle((0, 0), 1, 1, color=shade, alpha=0.18, label="± 1σ a priori  /  prior-dominated (ratio > 0.7)")]
        if np.any(np.isfinite(true_move)):
            handles += [Line2D([], [], ls="", marker="D", ms=5, mfc="none", mec="#222222", label="truth"),
                        Line2D([], [], ls="", marker="D", ms=5, mfc="none", mec=flag, label="truth beyond 3σ")]
        if any(r["considered"] for r in rows):
            handles.append(Line2D([], [], ls="", marker="o", ms=5, color="#9a9a9a", label="considered"))
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.17, 0.05 / height), ncol=len(handles),
                   handletextpad=0.3, columnspacing=1.2)
    return fig


# ------------------------------------------------------------------------------------------------
# Covariance shape
# ------------------------------------------------------------------------------------------------

CHI2_2 = {"95%": 5.991, "99.73%": 11.83}  # χ² with 2 degrees of freedom


def _ellipse(P2, k):
    """Points of the k-σ ellipse of a 2x2 covariance."""
    values, vectors = np.linalg.eigh(P2)
    angle = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(angle), np.sin(angle)])
    return (vectors @ (k * np.sqrt(np.clip(values, 0, None))[:, None] * circle)).T


def _scale_unit(peak, quantity):
    if quantity == "position":
        return (1e3, "m") if peak < 2.0 else (1.0, "km")
    return (1e6, "mm/s") if peak < 2e-3 else (1e3, "m/s")


MARKERS = ["o", "s", "D", "^", "v", "P", "X", "h"]


def _sky_layer(ax, labels, colors, markers, covs, errors, factor):
    """Ellipses and errors of every solution on a plane-of-sky axis."""
    for lab in labels:
        c = colors[lab]
        P2 = covs[lab][1:3, 1:3]
        one, three = _ellipse(P2, 1) * factor, _ellipse(P2, 3) * factor
        ax.fill(one[:, 0], one[:, 1], color=c, alpha=0.12, lw=0)
        ax.plot(one[:, 0], one[:, 1], color=c, lw=1.4)
        ax.plot(three[:, 0], three[:, 1], color=c, lw=1.0, ls="--")
    for lab in labels:  # errors on top of every ellipse
        ax.plot(*(errors[lab][1:3] * factor), markers[lab], ms=7, color=colors[lab], mec="white", mew=0.8, zorder=4)
    ax.axhline(0, color="#4d4d4d", lw=0.6)
    ax.axvline(0, color="#4d4d4d", lw=0.6)


def plot_covariance_shape(at_epoch, index=0, title=None, width=11.0):
    """Compare the covariance of several solutions at one epoch, with their actual errors.

    Earth-based radiometric tracking measures along the line of sight and infers the plane of sky
    through the dynamics, so the two differ by orders of magnitude and get their own panels:

    - **left, plane of sky (PoS 1, PoS 2):** the 1σ (filled) and 3σ (dashed) ellipses of each solution,
      and its actual error (dot). The legend gives the 2-D χ² of the error (consistent: below 5.99
      for 95 %, 11.83 for 99.73 %);
    - **right, line of sight (LOS):** position and velocity, ±1σ (dark) and ±3σ (light) per solution,
      with the error. The line-of-sight velocity is what the first minutes of Doppler measure.

    Parameters
    ----------
    at_epoch : dict
        ``{label: state_errors(...)}`` in the default ``frame="LOS"``, all at the same epoch(s).
    index : int
        Which epoch of the ``state_errors`` output to use.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = list(at_epoch)
    colors = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(labels)}
    markers = {lab: MARKERS[i % len(MARKERS)] for i, lab in enumerate(labels)}
    errors = {lab: at_epoch[lab]["error"][index] for lab in labels}
    covs = {lab: at_epoch[lab]["covariance"][index] for lab in labels}
    axes_names = at_epoch[labels[0]]["axes"]
    if axes_names[0] != "LOS":
        raise ValueError("plot_covariance_shape needs state_errors(..., frame='LOS')")
    t = at_epoch[labels[0]]["t"][index]
    shade = "#4d4d4d"

    peak_sky = max(max(3 * np.sqrt(np.max(np.diag(covs[lab])[1:3])), np.nanmax(np.abs(errors[lab][1:3])))
                   for lab in labels)
    f_sky, u_sky = _scale_unit(peak_sky, "position")

    with _styled():
        height = 5.0
        fig = plt.figure(figsize=(width, height))
        grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], wspace=0.35, hspace=0.75,
                                top=1.0 - 0.9 / height, bottom=0.6 / height)
        ax_sky = fig.add_subplot(grid[:, 0])
        ax_pos = fig.add_subplot(grid[0, 1])
        ax_vel = fig.add_subplot(grid[1, 1])

        # Plane of sky
        _sky_layer(ax_sky, labels, colors, markers, covs, errors, f_sky)
        handles = []
        for lab in labels:
            P2, e2 = covs[lab][1:3, 1:3], errors[lab][1:3]
            chi2 = float(e2 @ np.linalg.solve(P2, e2)) if np.all(np.isfinite(e2)) else np.nan
            text = f"{lab}" + (f"   χ² = {chi2:.3g}" if np.isfinite(chi2) else "")
            handles.append(Line2D([], [], color=colors[lab], lw=1.4, marker=markers[lab], ms=6, mec="white", label=text))
        ax_sky.set_aspect("equal", adjustable="datalim")
        ax_sky.set_xlabel(f"T [{u_sky}]")
        ax_sky.set_ylabel(f"N [{u_sky}]")
        ax_sky.set_title("Plane of sky", loc="left", fontsize=10, color="#222222", pad=15)
        ax_sky.text(0, 1.02, "1σ filled, 3σ dashed, marker: actual error", transform=ax_sky.transAxes,
                    fontsize=8, color="#555555", va="bottom")

        # Zoom inset when one ellipse dwarfs the others
        extent = {lab: 3 * np.sqrt(np.max(np.linalg.eigvalsh(covs[lab][1:3, 1:3]))) for lab in labels}
        if max(extent.values()) > 5 * min(extent.values()):
            small = [lab for lab in labels if extent[lab] <= 5 * min(extent.values())]
            zoom = 1.3 * max(extent[lab] for lab in small) * f_sky
            fig.canvas.draw()  # final limits of the main panel
            x_lim, y_lim = ax_sky.get_xlim(), ax_sky.get_ylim()
            points = np.vstack([_ellipse(covs[lab][1:3, 1:3], 3) * f_sky for lab in labels])
            corners = {(0.6, 0.6): (1, 1), (0.02, 0.6): (-1, 1), (0.02, 0.02): (-1, -1), (0.6, 0.02): (1, -1)}
            crowd = {}
            for place, (sx, sy) in corners.items():  # least crowded corner of the main panel
                x0 = x_lim[0] + (place[0]) * (x_lim[1] - x_lim[0])
                y0 = y_lim[0] + (place[1]) * (y_lim[1] - y_lim[0])
                w, h = 0.38 * (x_lim[1] - x_lim[0]), 0.38 * (y_lim[1] - y_lim[0])
                crowd[place] = np.sum((points[:, 0] > x0) & (points[:, 0] < x0 + w)
                                      & (points[:, 1] > y0) & (points[:, 1] < y0 + h))
            place = min(crowd, key=crowd.get)
            inset = ax_sky.inset_axes([place[0], place[1], 0.38, 0.38])
            _sky_layer(inset, small, colors, markers, covs, errors, f_sky)
            inset.set_xlim(-zoom, zoom)
            inset.set_ylim(-zoom, zoom)
            inset.set_aspect("equal")
            inset.tick_params(labelsize=7)
            inset.set_facecolor("white")
            inset.set_title(f"zoom ±{zoom:.2g} {u_sky}", fontsize=7.5, color="#555555", pad=2)
            for spine in inset.spines.values():
                spine.set_visible(True)
                spine.set_color("#999999")

        # Line of sight: position and velocity
        y = np.arange(len(labels))[::-1]
        for ax, k, quantity, name in [(ax_pos, 0, "position", "Line-of-sight position (R)"),
                                      (ax_vel, 3, "velocity", "Line-of-sight velocity (R)")]:
            sig = np.array([np.sqrt(covs[lab][k, k]) for lab in labels])
            err = np.array([errors[lab][k] for lab in labels])
            factor, unit = _scale_unit(max(np.nanmax(3 * sig), np.nanmax(np.abs(err))), quantity)
            for row, lab in enumerate(labels):
                c = colors[lab]
                ax.plot([-3 * sig[row] * factor, 3 * sig[row] * factor], [y[row]] * 2, color=c, lw=5, alpha=0.18,
                        solid_capstyle="butt")
                ax.plot([-sig[row] * factor, sig[row] * factor], [y[row]] * 2, color=c, lw=5, alpha=0.45,
                        solid_capstyle="butt")
                outside = abs(err[row]) > 3 * sig[row]
                ax.plot(err[row] * factor, y[row], markers[lab], ms=6.5, color=c, mec="#D55E00" if outside else "white",
                        mew=1.4 if outside else 0.8, zorder=4)
            ax.axvline(0, color=shade, lw=0.6)
            limit = 1.1 * np.nanmax(np.r_[3 * sig, np.abs(err)]) * factor
            ax.set_xlim(-limit, limit)
            ax.set_yticks(y)
            ax.set_yticklabels([f"{e / s:+.1f}σ" for e, s in zip(err, sig)], fontsize=8, color="#555555")
            ax.grid(axis="y", visible=False)
            ax.tick_params(axis="y", length=0)
            ax.set_ylim(-0.6, len(labels) - 0.4)
            ax.set_xlabel(f"[{unit}]")
            ax.set_title(name, loc="left", fontsize=10, color="#222222")

        try:
            import scarabaeus as scb

            when = scb.SpiceManager.et2utc(float(t))[:16].replace("T", " ") + " UTC"
        except Exception:
            when = f"{t:.0f} s TDB"
        top = 1.0 - 0.1 / height
        fig.suptitle(title or f"Covariance and error at {when}  ·  line of sight and plane of sky from Earth",
                     x=0.125, ha="left", y=top, fontsize=11)
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.12, -0.04), ncol=min(len(handles), 4),
                   handletextpad=0.4, columnspacing=1.5)
    return fig


def _short_labels(components, axes):
    labels, groups = [], []
    for c in components:
        for k in range(c["dim"]):
            if c["name"] == "position":
                labels.append(f"pos {axes[k]}")
            elif c["name"] == "velocity":
                labels.append(f"vel {axes[k]}")
            else:
                labels.append(c["name"] + (f"[{k}]" if c["dim"] > 1 else ""))
            groups.append(c["name"])
    return labels, groups


def plot_correlation(source, frame="LOS", center="SUN", origin="SUN", annotate=None, title=None, size=None):
    """Correlation matrix of a saved solution at its epoch.

    Position and velocity are in ``frame`` (as in :func:`state_errors`). Only the lower triangle is drawn;
    cells are annotated when the matrix is small enough to read. Correlations near ±1 mark
    quantities the data cannot separate: they constrain a combination of them, and the a priori
    decides how it is shared. With considered parameters, the consider covariance is used.

    Parameters
    ----------
    source : str, Path or dict
        A saved solution (Scarabaeus 2.0 JSON).
    annotate : bool, optional
        Print the coefficients. Default: when the state has at most 16 elements.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if isinstance(source, (str, Path)):
        with open(source) as f:
            source = json.load(f)
    epoch_solution = _solution_at_epoch(source, frame, center, origin)
    P = epoch_solution["post_P"]
    labels, groups = _short_labels(epoch_solution["components"], epoch_solution["axes"])
    n = len(labels)
    d = np.sqrt(np.clip(np.diag(P), 1e-300, None))
    rho = P / np.outer(d, d)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    shown = np.where(mask, np.nan, rho)
    annotate = n <= 16 if annotate is None else annotate

    with _styled():
        side = size or min(9.0, 2.2 + 0.42 * n)
        fig, ax = plt.subplots(figsize=(side + 1.0, side))
        image = ax.imshow(shown, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=8.5)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.tick_params(length=0)
        # Group separators (between state components)
        for k in range(1, n):
            if groups[k] != groups[k - 1]:
                ax.plot([-0.5, k - 0.5], [k - 0.5, k - 0.5], color="white", lw=2)
                ax.plot([k - 0.5, k - 0.5], [k - 0.5, n - 0.5], color="white", lw=2)
        if annotate:
            for i in range(n):
                for j in range(i + 1):
                    if i == j:
                        continue
                    value = rho[i, j]
                    ax.text(j, i, f"{value:+.2f}".replace("-", "−"), ha="center",
                            va="center", fontsize=7.5, color="white" if abs(value) > 0.6 else "#333333")
        bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        bar.set_label("correlation", fontsize=8.5)
        bar.outline.set_visible(False)
        consider = source["final"]["epoch_solution"].get("covariance_consider") is not None
        what = "consider covariance" if consider else "covariance"
        frame_text = "position and velocity: " + _frame_axes(frame, center)[2]
        ax.set_title(title or f"Correlations of the solution ({what})\n{frame_text}", loc="left", fontsize=10.5,
                     color="#222222")
    return fig


# ------------------------------------------------------------------------------------------------
# Convergence
# ------------------------------------------------------------------------------------------------

def plot_convergence(source, by_dataset=True, title=None, width=11.0):
    """How the iterations of a fit converged, from a saved solution.

    Top row, one point per iteration:

    - **cost:** χ²/n of the pre-fit residuals (on that iteration's reference, hollow) and of the
      post-fit residuals (filled), on a log scale, with the white-noise band 1 ± √(2/n). The
      pre-fit cost of iteration 1 measures the a-priori error; once the two coincide, the reference
      has stopped moving in any way the data can see;
    - **deviation test:** the RMS of the state correction against its threshold;
    - **cost test:** the relative change of the post-fit cost against its threshold.

    Points that pass a test are filled; the title says which test stopped the fit (with the
    ``convergence_criterion`` of the run), or that it did not converge. The bottom row
    (``by_dataset``) shows the pre-fit RMS/σ of every dataset per iteration.

    The deviation test mixes units: it is the RMS of all state components in the filter's units
    (km, km/s, and whatever the parameters use), so its threshold is in practice a position
    threshold in km.

    Parameters
    ----------
    source : str, Path or dict
        A saved solution (Scarabaeus 2.0 JSON, which records the convergence test per iteration).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if isinstance(source, (str, Path)):
        with open(source) as f:
            source = json.load(f)
    meta = source["metadata"]
    summary = meta.get("convergence", {})
    keys = sorted((k for k in source if k.startswith("iter_")), key=lambda k: int(k.split("_")[1]))
    records = [source[k].get("convergence", {}) for k in keys]
    iters = np.arange(1, len(keys) + 1)

    def cost(block):
        z = np.concatenate([np.asarray(v, dtype=float)[:, 0] / np.asarray(v, dtype=float)[:, 1]
                            for v in block.values()])
        return float(np.mean(z**2)), z.size

    pre = np.array([cost(source[k]["prefits"])[0] for k in keys])
    post = np.array([cost(source[k]["postfits"])[0] for k in keys])
    n_res = cost(source[keys[-1]]["postfits"])[1]
    deviation = np.array([r.get("rms_deviation", np.nan) for r in records], dtype=float)
    change = np.array([np.nan if r.get("cost_relative_change") is None else r["cost_relative_change"]
                       for r in records], dtype=float)
    pass_dev = np.array([bool(r.get("converged_deviation")) for r in records])
    pass_cost = np.array([bool(r.get("converged_cost")) for r in records])
    dev_threshold = summary.get("rms_deviation_threshold")
    cost_threshold = summary.get("cost_relative_change_threshold")
    criterion = summary.get("criterion", "?")
    converged = summary.get("converged", bool(records and records[-1].get("converged")))
    ok, bad, shade = "#009E73", "#D55E00", "#4d4d4d"

    with _styled():
        height = 0.9 + 2.6 + (2.1 if by_dataset else 0.0)
        fig = plt.figure(figsize=(width, height))
        outer = fig.add_gridspec(2 if by_dataset else 1, 1, hspace=0.65, top=1.0 - 0.95 / height,
                                 bottom=0.55 / height)
        top_row = outer[0].subgridspec(1, 3, wspace=0.42)
        ax_cost, ax_dev, ax_change = (fig.add_subplot(top_row[0, k]) for k in range(3))

        # Cost of the pre- and post-fit residuals
        band = np.sqrt(2 / n_res)
        ax_cost.axhspan(1 - 3 * band, 1 + 3 * band, color=shade, alpha=0.12, lw=0)
        ax_cost.plot(iters, pre, "o-", color="#0072B2", mfc="white", ms=6, lw=1.2, label="pre-fit")
        ax_cost.plot(iters, post, "o-", color="#0072B2", ms=5, lw=1.2, label="post-fit")
        ax_cost.set_yscale("log")
        ax_cost.set_ylabel("χ²/n")
        ax_cost.set_title("Cost of the residuals", loc="left", fontsize=10, color="#222222", pad=15)
        ax_cost.text(0, 1.02, f"final post-fit {post[-1]:.3f}  (white noise 1 ± {band:.2f})",
                     transform=ax_cost.transAxes, fontsize=8, color="#555555", va="bottom")
        ax_cost.legend(loc="upper right")

        # The two convergence tests
        for ax, values, passed, threshold, name, what in [
            (ax_dev, deviation, pass_dev, dev_threshold, "Deviation test", "RMS of the correction"),
            (ax_change, change, pass_cost, cost_threshold, "Cost test", "relative change of the cost"),
        ]:
            finite = np.isfinite(values) & (values > 0)
            ax.plot(iters[finite], values[finite], "-", color="#999999", lw=1.0)
            for i in np.nonzero(finite)[0]:
                ax.plot(iters[i], values[i], "o", ms=6, color=ok if passed[i] else bad,
                        mfc=(ok if passed[i] else "white"), mew=1.3)
            if threshold:
                ax.axhline(threshold, color=shade, lw=0.9, ls="--")
                what += f"  ·  threshold {threshold:.0e} (dashed)"
            ax.set_yscale("log")
            ax.set_title(name, loc="left", fontsize=10, color="#222222", pad=15)
            ax.text(0, 1.02, what, transform=ax.transAxes, fontsize=8, color="#555555", va="bottom")
        ax_dev.set_ylabel("km, km/s (mixed)")
        for ax in (ax_cost, ax_dev, ax_change):
            ax.set_xticks(iters)
            ax.set_xlim(0.5, len(iters) + 0.5)
            ax.set_xlabel("iteration")

        # Pre-fit RMS per dataset
        if by_dataset:
            ax_ds = fig.add_subplot(outer[1])
            names = list(source[keys[0]]["prefits"])
            labels = {n: classify(n) for n in names}
            observers = list(dict.fromkeys(obs for _, obs in labels.values()))
            kinds = list(dict.fromkeys(kind for kind, _ in labels.values()))
            styles = ["-", "--", ":", "-."]
            for n in names:
                rms = [np.sqrt(np.mean((np.asarray(source[k]["prefits"][n], dtype=float)[:, 0]
                                        / np.asarray(source[k]["prefits"][n], dtype=float)[:, 1]) ** 2)) for k in keys]
                kind, obs = labels[n]
                ax_ds.plot(iters, rms, styles[kinds.index(kind) % len(styles)], marker="o", ms=4,
                           color=PALETTE[observers.index(obs) % len(PALETTE)], lw=1.3, label=n)
            ax_ds.axhspan(0, 1, color=shade, alpha=0.08, lw=0)
            ax_ds.set_yscale("log")
            ax_ds.set_xticks(iters)
            ax_ds.set_xlim(0.5, len(iters) + 0.5)
            ax_ds.set_xlabel("iteration")
            ax_ds.set_ylabel("pre-fit RMS / σ")
            ax_ds.set_title("Pre-fit residuals by dataset", loc="left", fontsize=10, color="#222222")
            ax_ds.legend(loc="upper right", ncol=min(3, len(names)), fontsize=7.5)

        if title is None:
            n_it = len(iters)
            if converged:
                tests = [t for t, passed in [("deviation", pass_dev[-1]), ("cost", pass_cost[-1])] if passed]
                status = f"converged after {n_it} iteration{'s' if n_it > 1 else ''} ({' and '.join(tests)} test)"
            else:
                status = f"NOT converged after {n_it} iterations"
            title = f"Convergence  ·  {status}  ·  criterion \"{criterion}\""
        top = 1.0 - 0.1 / height
        fig.suptitle(title, x=0.125, ha="left", y=top, fontsize=11, color="#222222" if converged else bad)
    return fig


# ------------------------------------------------------------------------------------------------
# Tracking timeline
# ------------------------------------------------------------------------------------------------

_COMPLEXES = {1: "Goldstone", 2: "Goldstone", 3: "Canberra", 4: "Canberra", 5: "Madrid", 6: "Madrid"}


def _complex(station_name):
    match = re.search(r"(\d+)", station_name)
    return _COMPLEXES.get(int(match.group(1)) // 10, "") if match else ""


def observation_epochs(measurements):
    """``{dataset: TDB seconds}`` from a ``MeasurementSpec`` collection."""
    return {spec["dataset_name"]: np.asarray(spec["observed_meas"].meas_time_et.times.values, dtype=float).ravel()
            for spec in measurements}


def plot_tracking_timeline(stations, target, epochs, observations=None, elevation_mask=10.0, geometry=True,
                           title=None, width=11.0):
    """When could each station see the spacecraft, and when did it take data?

    One row per station. The band shows when the spacecraft is above the elevation mask, shaded by
    elevation (darker is higher, less atmosphere on the path); a thin line marks the time above the
    horizon but below the mask. Ticks show the observations of each data type (range on top,
    range-rate or Doppler below). The right margin gives the fraction of time visible and the
    number of observations.

    With ``geometry=True``, a strip on top shows the Sun-Earth-probe (SEP) angle: below about 10-20°
    the solar corona adds noise and delay to radiometric data.

    Parameters
    ----------
    stations : dict or list
        ``{name: GroundStation}`` or a list of ``GroundStation``.
    target : Body
        The spacecraft (its ephemeris must be in the kernel pool).
    epochs : EpochArray
        Grid on which visibility and geometry are evaluated (e.g. every 10 minutes).
    observations : dict or MeasurementSpec collection, optional
        ``{dataset: epochs (TDB seconds or EpochArray)}``, or the filter's measurements. Datasets are
        assigned to stations through their names (``classify``).
    elevation_mask : float
        Degrees.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import scarabaeus as scb

    if not isinstance(stations, dict):
        stations = {gs.name: gs for gs in stations}
    if observations is not None and not isinstance(observations, dict):
        observations = observation_epochs(observations)
    observations = {n: np.asarray(getattr(t, "times", t).values if hasattr(t, "times") else t, dtype=float).ravel()
                    for n, t in (observations or {}).items()}
    t = np.asarray(epochs.times.values, dtype=float).ravel()
    t0, t_scale, t_label = _time_axis(t, "auto")
    x = (t - t0) / t_scale
    cmap = plt.get_cmap("Blues")

    rows = []
    for name, gs in stations.items():
        _, elevation, _ = gs.station_visibility(target, epochs, elevation_mask=elevation_mask, return_angles=True)
        elevation = np.asarray(elevation, dtype=float).ravel()
        mine = {n: v for n, v in observations.items() if classify(n)[1] == name.upper()}
        rows.append((name, elevation, mine))

    with _styled():
        row_h = 0.62
        height = 1.3 + (1.3 if geometry else 0.0) + row_h * len(rows)
        fig = plt.figure(figsize=(width, height))
        ratios = ([1.3] if geometry else []) + [row_h * len(rows)]
        grid = fig.add_gridspec(len(ratios), 2, height_ratios=ratios, width_ratios=[40, 1], hspace=0.35, wspace=0.02,
                                top=1.0 - 0.75 / height, bottom=0.6 / height, left=0.17, right=0.9)
        ax = fig.add_subplot(grid[-1, 0])

        if geometry:
            # one query per epoch: get_state takes one epoch per call (EpochArray support is planned)
            sc = np.array([target.get_state(epochs[k], reference_frame="J2000", origin="EARTH").quantity.values[:3]
                           for k in range(t.size)], dtype=float)
            sun = np.array([scb.CelestialBody.from_constants("SUN").get_state(epochs[k], reference_frame="J2000",
                                                                             origin="EARTH").quantity.values[:3]
                            for k in range(t.size)], dtype=float)
            cosine = np.einsum("ij,ij->i", sc, sun) / (np.linalg.norm(sc, axis=1) * np.linalg.norm(sun, axis=1))
            sep = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            distance_au = np.linalg.norm(sc, axis=1) / 1.495978707e8
            ax_geo = fig.add_subplot(grid[0, 0], sharex=ax)
            ax_geo.plot(x, sep, color="#D55E00", lw=1.4)
            if sep.min() < 25:
                ax_geo.axhspan(0, 15, color="#D55E00", alpha=0.08, lw=0)
            ax_geo.set_ylabel("SEP [deg]", color="#D55E00")
            ax_geo.tick_params(axis="y", colors="#D55E00")
            ax_dist = ax_geo.twinx()
            ax_dist.plot(x, distance_au, color="#555555", lw=1.0, ls="--")
            ax_dist.set_ylabel("Earth range [AU]", color="#555555")
            ax_dist.grid(False)
            ax_dist.spines["right"].set_visible(True)
            ax_geo.set_title("Geometry", loc="left", fontsize=10, color="#222222", pad=15)
            ax_geo.text(0, 1.04, f"Sun-Earth-probe angle {sep.min():.0f}-{sep.max():.0f}° (solid; corona noise below ~15°)"
                                 f"  ·  Earth range {distance_au.min():.3f}-{distance_au.max():.3f} AU (dashed)",
                        transform=ax_geo.transAxes, fontsize=8, color="#555555", va="bottom")
            plt.setp(ax_geo.get_xticklabels(), visible=False)

        # Station rows
        norm = plt.Normalize(0, 90)
        dx = np.median(np.diff(x)) if x.size > 1 else 1.0
        type_offsets = {"Range": 0.27, "Sequential ranging": 0.27}
        for row, (name, elevation, mine) in enumerate(rows):
            y = len(rows) - 1 - row
            above_horizon = elevation > 0
            ax.fill_between(x, y - 0.03, y + 0.03, where=above_horizon, color="#bbbbbb", lw=0, step="mid")
            visible = elevation >= elevation_mask
            ax.bar(x[visible], 0.36, width=dx, bottom=y - 0.18, color=cmap(0.25 + 0.75 * norm(elevation[visible])),
                   lw=0, align="center")
            n_obs = 0
            for dataset, t_obs in mine.items():
                kind = classify(dataset)[0]
                offset = type_offsets.get(kind, -0.27)
                ax.plot((t_obs - t0) / t_scale, np.full(t_obs.size, y + offset), "|", ms=5, mew=0.8,
                        color="#222222" if offset > 0 else "#E69F00")
                n_obs += t_obs.size
            complex_name = _complex(name)
            ax.text(-0.012, y + 0.12, f"{name}  {complex_name}".strip(), transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=8.5, color="#222222")
            summary = f"{visible.mean():.0%} visible" + (f" · {n_obs} obs" if mine else "")
            ax.text(-0.012, y - 0.18, summary, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=7.5, color="#555555")
        ax.set_yticks([])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(t_label)
        ax.set_xlim(x.min(), x.max())
        legend_text = f"band: above the {elevation_mask:g}° mask, shaded by elevation  ·  gray line: above the horizon"
        if observations:
            legend_text += "  ·  ticks: range (black, top), range-rate / Doppler (orange, bottom)"
        ax.set_title("Visibility and tracking" if observations else "Visibility", loc="left", fontsize=10,
                     color="#222222", pad=15)
        ax.text(0, 1.02, legend_text, transform=ax.transAxes, fontsize=8, color="#555555", va="bottom")

        # Elevation color scale
        cax = fig.add_subplot(grid[-1, 1])
        scale = plt.cm.ScalarMappable(norm=norm, cmap=plt.matplotlib.colors.ListedColormap(cmap(np.linspace(0.25, 1, 256))))
        bar = fig.colorbar(scale, cax=cax)
        bar.set_label("elevation [deg]", fontsize=8)
        bar.outline.set_visible(False)
        cax.tick_params(labelsize=7.5)
        if geometry:
            fig.add_subplot(grid[0, 1]).axis("off")

        top = 1.0 - 0.1 / height
        fig.suptitle(title or f"Tracking of {target.name}", x=0.17, ha="left", y=top, fontsize=11)
    return fig


def epochs_from_tracking_files(paths, blocks=(("2W_doppler", "Doppler"), ("2W_sranging", "ranging"))):
    """``{"DSS-nn <type>": TDB seconds}`` from converted tracking files (the JSON layout of chapter 9).

    Reading the epochs of a tracking file through the library needs a measurement model
    (``observed_measurements``), which needs its auxiliary data; this reads the file directly.
    """
    import scarabaeus as scb

    out = {}
    for path in paths:
        with open(path) as f:
            record = json.load(f)
        for block, label in blocks:
            if block not in record:
                continue
            data = record[block]
            spice_id = np.ravel(data["spice_id"])[0]
            station = f"DSS-{int(spice_id) % 1000}"
            et = np.asarray(scb.SpiceManager.YDS2et(data["year"], data["doy"], data["sec"]), dtype=float).ravel()
            key = f"{station} {label}"
            out[key] = np.concatenate([out.get(key, np.empty(0)), et])
    return out


# ------------------------------------------------------------------------------------------------
# Measurement editing
# ------------------------------------------------------------------------------------------------

def _huber_mad_trend(editor, name, mad_scale):
    """Per-pass Huber trend and robust scale of one dataset, as ``MeasurementEditing`` computes them.

    Mirrors ``MeasurementEditing._huber_mad_evaluate``, which returns only the flags and the
    refit σ. Once it also returns the trend and the scale (PLAN.md), call it instead.
    """
    from scarabaeus import MeasurementEditing
    from sklearn.linear_model import HuberRegressor

    dataset = editor._get_dataset(name)
    residuals = np.asarray(dataset.data["residuals"], dtype=float).ravel()
    times = np.asarray(dataset.data["t1"], dtype=float).ravel()
    keep = ~np.asarray(editor.outlier_flags[name], dtype=bool)
    segments = MeasurementEditing._pass_segments(times, gap_threshold_sec=editor._pass_gap_sec, keep=keep,
                                                 min_points=editor._pass_min_points)
    global_median = float(np.median(residuals[keep]))
    trend = np.full(residuals.shape, global_median)
    for segment in segments:
        fit_idx = segment[keep[segment]]
        if fit_idx.size < 3:
            continue
        t0 = float(np.mean(times[fit_idx]))
        t_scale = float(np.std(times[fit_idx])) or 1.0
        r_fit = residuals[fit_idx]
        r_mid = float(np.median(r_fit))
        r_scale = float(np.median(np.abs(r_fit - r_mid))) or (float(np.std(r_fit)) or 1.0)
        try:
            huber = HuberRegressor().fit(((times[fit_idx] - t0) / t_scale)[:, None], (r_fit - r_mid) / r_scale)
            trend[segment] = huber.predict(((times[segment] - t0) / t_scale)[:, None]) * r_scale + r_mid
        except Exception:
            trend[segment] = np.median(r_fit)
    detrended = residuals - trend
    center = float(np.median(detrended[keep]))
    scale = mad_scale * float(np.median(np.abs(detrended[keep] - center)))
    return times, residuals, trend + center, scale, keep


def plot_editing(editor, n_sigma=5.0, mad_scale=1.4826, dataset_names=None, units=None, title=None, width=11.0):
    """Preview of ``MeasurementEditing.huber_mad_filter`` on an editor that has not run it yet.

    One row per dataset:

    - **left:** the residuals against time with the robust trend of each pass (line) and the
      acceptance band, trend ± n_sigma × robust scale (shaded). Kept points are dots, points the
      filter would flag are vermillion crosses, points flagged before (e.g. no-data rows of the
      file) are gray crosses. The axis is fitted to the band; points far beyond it are drawn as edge
      markers and counted in the header, so a −4×10⁵ Hz outlier does not flatten the rest;
    - **right:** the detrended residuals in units of the robust scale, with the ±n_sigma limits
      and the normal curve.

    The header gives the counts, the robust scale (MAD × ``mad_scale``) and the σ the filter would
    re-estimate from the kept points, next to the σ the model assumes.

    Parameters
    ----------
    editor : MeasurementEditing
        Built on datasets that carry residuals (e.g. from ``generate_measurement_dataset`` against a
        reference trajectory), before ``huber_mad_filter`` is called.
    n_sigma, mad_scale : float
        As in ``huber_mad_filter``.
    units : dict, optional
        ``{row label: (factor, unit)}`` overriding ``DISPLAY_UNITS``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = list(dataset_names or editor.outlier_flags)
    unit_of = {**DISPLAY_UNITS, **(units or {})}
    rows = []
    for name in names:
        times, residuals, center, scale, keep = _huber_mad_trend(editor, name, mad_scale)
        # The library's own verdict, so the plot cannot disagree with the filter
        flags_new, sigma_new = editor._huber_mad_evaluate(name, n_sigma, mad_scale, True)
        mirrored = keep & (np.abs(residuals - center) > n_sigma * scale)
        if not np.array_equal(mirrored, flags_new):
            import warnings

            warnings.warn(f"{name}: the mirrored trend flags {int(mirrored.sum())} points, the filter "
                          f"{int(flags_new.sum())}; the band drawn may differ from the filter's")
        rows.append((name, times, residuals, center, scale, keep, np.asarray(flags_new, dtype=bool), sigma_new))

    t0, t_scale, t_label = _time_axis(np.concatenate([r[1] for r in rows]), "auto")
    shade, kept_color, flag_color, old_color = "#4d4d4d", "#0072B2", "#D55E00", "#9a9a9a"

    with _styled():
        height = 1.0 + 2.7 * len(rows)
        fig = plt.figure(figsize=(width, height))
        grid = fig.add_gridspec(len(rows), 2, width_ratios=[5, 1.2], wspace=0.08, hspace=0.75,
                                top=1.0 - 0.95 / height, bottom=0.55 / height)
        first = None
        for i, (name, times, residuals, center, scale, keep, flagged, sigma_new) in enumerate(rows):
            kind = classify(name)[0]
            factor, unit = unit_of.get(kind, (1.0, ""))
            x = (times - t0) / t_scale
            y, c, half = residuals * factor, center * factor, n_sigma * scale * factor
            ax = fig.add_subplot(grid[i, 0], sharex=first)
            first = first or ax
            hx = fig.add_subplot(grid[i, 1])

            order = np.argsort(x)
            ax.fill_between(x[order], (c - half)[order], (c + half)[order], color=shade, alpha=0.10, lw=0)
            ax.plot(x[order], c[order], color="#222222", lw=1.0)
            low, high = np.min(c - half), np.max(c + half)
            pad = 0.6 * (high - low)
            y_range = (low - pad, high + pad)
            kept = keep & ~flagged
            groups = [(kept, "o", kept_color, 3.0, "kept"), (flagged, "x", flag_color, 5.5, "flagged now"),
                      (~keep, "x", old_color, 5.0, "flagged before")]
            beyond_total = 0
            for mask, marker, color, size, _ in groups:
                inside = mask & (y >= y_range[0]) & (y <= y_range[1])
                ax.plot(x[inside], y[inside], marker, ms=size, color=color, mew=1.1 if marker == "x" else 0,
                        alpha=0.85, zorder=3)
                for edge, edge_marker in [(y_range[1], "^"), (y_range[0], "v")]:
                    clip = mask & ((y > edge) if edge_marker == "^" else (y < edge))
                    beyond_total += int(clip.sum())
                    ax.plot(x[clip], np.full(clip.sum(), edge - 0.03 * (y_range[1] - y_range[0]) * np.sign(edge - np.mean(y_range))),
                            edge_marker, ms=5, color=color, zorder=4)
            ax.set_ylim(*y_range)
            ax.set_ylabel(f"O − C [{unit}]" if unit else "O − C")
            if i < len(rows) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            extreme = y[(y < y_range[0]) | (y > y_range[1])]
            line1 = [f"n {residuals.size}", f"flagged now {int(flagged.sum())}", f"flagged before {int((~keep).sum())}"]
            if beyond_total:
                line1.append(f"{beyond_total} beyond the axis (largest {extreme[np.argmax(np.abs(extreme))]:.3g} {unit})")
            line2 = [f"robust scale {scale * factor:.3g} {unit}".rstrip(), f"band ±{n_sigma:g} × scale"]
            if sigma_new is not None:
                line2.append(f"σ refit from the kept points {sigma_new * factor:.3g} {unit}".rstrip())
            ax.set_title(name, loc="left", fontsize=10, color="#222222", pad=27)
            ax.text(0, 1.02, "  ·  ".join(line1) + "\n" + "  ·  ".join(line2), transform=ax.transAxes, fontsize=8,
                    color="#555555", va="bottom", linespacing=1.4)

            # Detrended residuals in units of the robust scale
            z = (residuals - center) / scale
            limit = 1.6 * n_sigma
            bins = np.linspace(-limit, limit, 41)
            hx.hist(np.clip(z[kept], -limit, limit), bins=bins, color=kept_color, alpha=0.55, lw=0, density=True,
                    label="kept")
            if flagged.any():
                hx.hist(np.clip(z[flagged], -limit, limit), bins=bins, color=flag_color, alpha=0.8, lw=0,
                        density=True, label="flagged now")
            grid_z = np.linspace(-limit, limit, 300)
            hx.plot(grid_z, np.exp(-0.5 * grid_z**2) / np.sqrt(2 * np.pi), color="#222222", lw=1.0)
            for edge in (-n_sigma, n_sigma):
                hx.axvline(edge, color=flag_color, lw=0.9, ls="--")
            hx.set_yticks([])
            hx.grid(False)
            hx.spines["left"].set_visible(False)
            hx.set_xlabel(f"detrended / scale\n(±{n_sigma:g} dashed, clipped at ±{limit:g})", fontsize=8)
        first.set_xlabel(t_label) if len(rows) == 1 else ax.set_xlabel(t_label)

        handles = [Line2D([], [], ls="", marker="o", ms=5, color=kept_color, label="kept"),
                   Line2D([], [], ls="", marker="x", ms=6, mew=1.2, color=flag_color, label="flagged now"),
                   Line2D([], [], ls="", marker="x", ms=6, mew=1.2, color=old_color, label="flagged before"),
                   Line2D([], [], color="#222222", lw=1.0, label="robust trend per pass"),
                   plt.Rectangle((0, 0), 1, 1, color=shade, alpha=0.15, label=f"acceptance band (±{n_sigma:g} × scale)")]
        top = 1.0 - 0.1 / height
        fig.suptitle(title or f"Measurement editing preview  ·  Huber trend + MAD threshold ({n_sigma:g}×)",
                     x=0.125, ha="left", y=top, fontsize=11)
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.9, top - 0.2 / height), ncol=5,
                   handletextpad=0.3, columnspacing=1.0, fontsize=7.5)
    return fig


# ------------------------------------------------------------------------------------------------
# Corner plots: a priori vs posterior, and Monte Carlo
# ------------------------------------------------------------------------------------------------

def _corner_axes(fig, n, top, bottom, left=0.1, right=0.97):
    """Lower-triangle grid of axes: ``{(row, col): Axes}`` for row >= col, shared columns."""
    grid = fig.add_gridspec(n, n, wspace=0.08, hspace=0.08, top=top, bottom=bottom, left=left, right=right)
    axes = {}
    for i in range(n):
        for j in range(i + 1):
            ax = fig.add_subplot(grid[i, j])
            axes[i, j] = ax
    for (i, j), ax in axes.items():
        if i < n - 1:
            ax.tick_params(labelbottom=False)
        if j > 0 or i == 0:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=7.5)
    return axes


def _gaussian(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _display(component, label):
    """(subtract the a priori?, factor, unit) for a state element in the corner plots."""
    name = component["name"]
    if name == "position":
        return True, 1.0, "km"
    if name == "velocity":
        return True, 1e6, "mm/s"
    unit = "RU" if "RU" in name else ("" if component["units"]["name"] == "unitless" else component["units"]["name"])
    return False, 1.0, unit


def plot_prior_posterior(source, variables=None, truth=None, truth_values=None, view="posterior", frame="LOS",
                         center="SUN", origin="SUN", title=None, size=None):
    """Corner plot of the a priori and the posterior of a saved solution (Gaussian, no samples).

    Each pair of variables shows the a-priori 1σ and 3σ ellipses (gray, dashed for 3σ), the
    posterior ones (filled 1σ, outlined 3σ), the estimate (dot) and the truth (star) when known,
    with the posterior correlation ρ in the corner. The diagonal shows the two 1-D Gaussians and
    σ_post / σ_prior. Position and velocity are changes from the a priori, in RTN about ``center``;
    parameters are absolute values.

    Parameters
    ----------
    source : str, Path or dict
        A saved solution (Scarabaeus 2.0 JSON).
    variables : list of str, optional
        Short labels to include (``"pos LOS"``, ``"vel PoS 1"``, ``"eta_srp"``, ...; with ``frame="RTN"``, ``"pos R"``, ``"vel T"``). Default: the
        position axes and every parameter, at most 6.
    truth, truth_values
        As in :func:`plot_parameter_learning`.
    view : {"posterior", "prior"}
        Frame each axis on the posterior (±4σ; the shape and the degeneracies show) or on the a
        priori (±3.5σ; how much the data reduced it shows).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if isinstance(source, (str, Path)):
        with open(source) as f:
            source = json.load(f)
    state = _solution_at_epoch(source, frame, center, origin, truth, truth_values)
    labels, groups = _short_labels(state["components"], state["axes"])
    component_of = [c for c in state["components"] for _ in range(c["dim"])]
    if variables is None:
        variables = [l for l, g in zip(labels, groups) if g == "position"] + \
                    [l for l, g in zip(labels, groups) if g not in ("position", "velocity")]
        variables = variables[:6]
    idx = [labels.index(v) for v in variables]
    n = len(idx)
    prior_x, prior_P, post_x, post_P, true_x = (state[k] for k in ("prior_x", "prior_P", "post_x", "post_P", "true_x"))

    # Display transform: position and velocity relative to the estimate (posterior view) or to the
    # a priori (prior view), in units set by the spread shown; parameters as absolute values
    center_x = post_x if view == "posterior" else prior_x
    spread = np.sqrt(np.diag(post_P if view == "posterior" else prior_P))
    ref, fac, units, axis_labels = np.zeros(n), np.ones(n), [], []
    for a, i in enumerate(idx):
        relative, _, unit = _display(component_of[i], labels[i])
        if relative:
            quantity = "velocity" if component_of[i]["name"] == "velocity" else "position"
            fac[a], unit = _scale_unit(4 * spread[i], quantity)
            ref[a] = center_x[i]
            axis_labels.append(f"{labels[i]} − {'estimate' if view == 'posterior' else 'a priori'}")
        else:
            axis_labels.append(labels[i])
        units.append(unit)
    def shown(x):
        return (np.asarray(x)[idx] - ref) * fac
    prior_m, post_m, true_m = shown(prior_x), shown(post_x), shown(true_x)
    S = np.diag(fac)
    prior_C = S @ prior_P[np.ix_(idx, idx)] @ S
    post_C = S @ post_P[np.ix_(idx, idx)] @ S
    prior_s, post_s = np.sqrt(np.diag(prior_C)), np.sqrt(np.diag(post_C))
    if view == "posterior":
        lims = [(post_m[a] - 4 * post_s[a], post_m[a] + 4 * post_s[a]) for a in range(n)]
    else:
        lims = [(prior_m[a] - 3.5 * prior_s[a], prior_m[a] + 3.5 * prior_s[a]) for a in range(n)]
    post_color, prior_color, truth_color = "#0072B2", "#9a9a9a", "#D55E00"

    with _styled():
        side = size or min(10.0, 1.9 * n + 1.2)
        fig = plt.figure(figsize=(side, side))
        axes = _corner_axes(fig, n, top=1.0 - 0.9 / side, bottom=0.6 / side + 0.05, left=0.12)
        for (i, j), ax in axes.items():
            if i == j:
                grid_x = np.linspace(*lims[i], 400)
                ax.fill_between(grid_x, _gaussian(grid_x, post_m[i], post_s[i]), color=post_color, alpha=0.25, lw=0)
                ax.plot(grid_x, _gaussian(grid_x, post_m[i], post_s[i]), color=post_color, lw=1.3)
                prior_curve = _gaussian(grid_x, prior_m[i], prior_s[i])
                ax2 = ax.twinx()  # the prior is far flatter: its own scale
                ax2.plot(grid_x, prior_curve, color=prior_color, lw=1.2, ls="--")
                ax2.set_yticks([])
                ax2.set_ylim(0, 1.15 * max(prior_curve.max(), 1e-300))
                ax2.grid(False)
                for spine in ax2.spines.values():
                    spine.set_visible(False)
                ax.set_ylim(0, 1.15 * _gaussian(post_m[i], post_m[i], post_s[i]))
                ax.set_yticks([])
                if np.isfinite(true_m[i]):
                    ax.axvline(true_m[i], color=truth_color, lw=1.0)
                ax.set_title(f"{variables[i]}\nσ post/prior {post_s[i] / prior_s[i]:.2g}", fontsize=8, color="#333333")
            else:
                for k, style in [(3, "--"), (1, "-")]:
                    e = _ellipse(prior_C[np.ix_([j, i], [j, i])], k) + prior_m[[j, i]]
                    ax.plot(e[:, 0], e[:, 1], color=prior_color, lw=0.9, ls=style)
                one = _ellipse(post_C[np.ix_([j, i], [j, i])], 1) + post_m[[j, i]]
                three = _ellipse(post_C[np.ix_([j, i], [j, i])], 3) + post_m[[j, i]]
                ax.fill(one[:, 0], one[:, 1], color=post_color, alpha=0.3, lw=0)
                ax.plot(three[:, 0], three[:, 1], color=post_color, lw=1.0)
                ax.plot(post_m[j], post_m[i], "o", ms=4, color=post_color, mec="white", mew=0.5, zorder=4)
                if np.isfinite(true_m[j]) and np.isfinite(true_m[i]):
                    ax.plot(true_m[j], true_m[i], "*", ms=9, color=truth_color, mec="white", mew=0.5, zorder=5)
                rho = post_C[i, j] / (post_s[i] * post_s[j])
                ax.text(0.04, 0.95, f"ρ {rho:+.2f}".replace("-", "−"), transform=ax.transAxes, fontsize=7.5,
                        va="top", color="#333333", bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))
                ax.set_ylim(*lims[i])
            ax.set_xlim(*lims[j])
            if i == n - 1:
                ax.set_xlabel(f"{axis_labels[j]}" + (f" [{units[j]}]" if units[j] else ""), fontsize=8)
            if j == 0 and i > 0:
                ax.set_ylabel(f"{axis_labels[i]}" + (f" [{units[i]}]" if units[i] else ""), fontsize=8)
            ax.ticklabel_format(useOffset=False, style="plain") if ax.get_xscale() == "linear" else None
        handles = [plt.Rectangle((0, 0), 1, 1, color=post_color, alpha=0.4, label="posterior 1σ (3σ outlined)"),
                   Line2D([], [], color=prior_color, lw=1.0, label="a priori 1σ, 3σ dashed")]
        if np.any(np.isfinite(true_m)):
            handles.append(Line2D([], [], ls="", marker="*", ms=9, color=truth_color, label="truth"))
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 1.0 - 0.95 / side), fontsize=8)
        frame_text = ("position, velocity: " + _frame_axes(frame, center)[2] + ", relative to the "
                      + ("estimate" if view == "posterior" else "a priori"))
        fig.suptitle(title or f"A priori and posterior  ·  {view} view", x=0.12, ha="left",
                     y=1.0 - 0.12 / side, fontsize=11)
        fig.text(0.12, 1.0 - 0.42 / side, frame_text, fontsize=8, color="#555555")
    return fig


def plot_monte_carlo(errors, covariances, labels=None, title=None, size=None):
    """Monte Carlo test of covariance realism: the errors of many fits against the filter covariance.

    Each run (new measurement noise and a-priori error) gives one estimation error. Every pair
    panel shows the runs as points, the 1σ and 3σ ellipses of the filter's average covariance and
    the sample mean (cross); the diagonal shows each error's histogram against N(0, σ).

    The header gives two tests:

    - **NEES** (normalized estimation error squared) eᵀP⁻¹e, averaged over the runs: for a
      consistent filter it follows χ²(N·d)/N, so its mean lies near d (the number of variables)
      with the 95 % interval shown. Above it, the covariance is optimistic; below, pessimistic;
    - **bias** per axis: the sample mean against its standard error σ/√N. Beyond 3, the errors are
      biased (a systematic error the covariance cannot describe), flagged in vermillion. The spread
      (sample standard deviation / filter σ) separates the two failure modes: a wrong spread is a
      wrong covariance, a right spread with an offset mean is an unmodeled systematic error.

    Parameters
    ----------
    errors : array (N, d)
        One error per run, e.g. ``state_errors(...)["error"][0]`` of each fit, in RTN.
    covariances : array (N, d, d) or (d, d)
        The filter's covariance of each run (or one for all).
    labels : list of str, optional
        Variable names; default ``R, T, N`` position then velocity.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.stats import chi2 as chi2_dist

    errors = np.asarray(errors, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    if covariances.ndim == 2:
        covariances = np.broadcast_to(covariances, (errors.shape[0],) + covariances.shape)
    N, d = errors.shape
    labels = labels or ["pos R", "pos T", "pos N", "vel R", "vel T", "vel N"][:d]
    mean_cov = covariances.mean(axis=0)
    sigma = np.sqrt(np.diag(mean_cov))
    nees = np.einsum("ni,ni->n", errors, np.linalg.solve(covariances, errors[..., None])[..., 0])
    nees_mean = float(np.mean(nees))
    nees_low, nees_high = chi2_dist.ppf([0.025, 0.975], N * d) / N
    sample_mean = errors.mean(axis=0)
    bias_z = sample_mean / (sigma / np.sqrt(N))
    spread = errors.std(axis=0, ddof=1) / sigma

    # Display units per variable: positions km -> m or km, velocities km/s -> mm/s
    fac, units = np.ones(d), []
    for a, label in enumerate(labels):
        quantity = "velocity" if label.startswith("vel") else "position"
        factor, unit = _scale_unit(max(3 * sigma[a], np.max(np.abs(errors[:, a]))), quantity)
        fac[a] = factor
        units.append(unit)
    E = errors * fac
    C = mean_cov * np.outer(fac, fac)
    s = sigma * fac
    m = sample_mean * fac
    lims = [(min(-3.6 * s[a], E[:, a].min()) * 1.05, max(3.6 * s[a], E[:, a].max()) * 1.05) for a in range(d)]
    point_color, cov_color, flag = "#0072B2", "#222222", "#D55E00"

    with _styled():
        side = size or min(10.5, 1.6 * d + 1.6)
        fig = plt.figure(figsize=(side, side))
        axes = _corner_axes(fig, d, top=1.0 - 1.15 / side, bottom=0.6 / side + 0.05, left=0.12)
        for (i, j), ax in axes.items():
            biased_j = abs(bias_z[j]) > 3
            if i == j:
                bins = np.linspace(*lims[i], 25)
                ax.hist(E[:, i], bins=bins, density=True, color=point_color, alpha=0.45, lw=0)
                grid_x = np.linspace(*lims[i], 300)
                ax.plot(grid_x, _gaussian(grid_x, 0.0, s[i]), color=cov_color, lw=1.1)
                ax.axvline(m[i], color=flag if biased_j else point_color, lw=1.2, ls="--")
                ax.set_yticks([])
                bias_text = f"{bias_z[i]:+.1f}" if abs(bias_z[i]) < 1e3 else f"{bias_z[i]:+.1e}"
                ax.set_title(f"{labels[i]}\nmean {bias_text} σ/√N\nspread {spread[i]:.2f}σ", fontsize=7.5,
                             color=flag if biased_j else "#333333")
            else:
                ax.plot(E[:, j], E[:, i], "o", ms=2.6, color=point_color, alpha=0.6, mew=0)
                for k, style in [(1, "-"), (3, "--")]:
                    e = _ellipse(C[np.ix_([j, i], [j, i])], k)
                    ax.plot(e[:, 0], e[:, 1], color=cov_color, lw=1.0, ls=style)
                biased = biased_j or abs(bias_z[i]) > 3
                ax.plot(m[j], m[i], "X", ms=7, color=flag if biased else point_color, mec="white", mew=0.6, zorder=5)
                ax.set_ylim(*lims[i])
            ax.set_xlim(*lims[j])
            if i == d - 1:
                ax.set_xlabel(f"{labels[j]} [{units[j]}]", fontsize=8)
            if j == 0 and i > 0:
                ax.set_ylabel(f"{labels[i]} [{units[i]}]", fontsize=8)

        biased_axes = [labels[a] for a in range(d) if abs(bias_z[a]) > 3]
        consistent = nees_low <= nees_mean <= nees_high
        if biased_axes:
            verdict = "biased: the errors are offset, which no covariance describes"
        elif consistent:
            verdict = "consistent"
        else:
            verdict = "covariance optimistic" if nees_mean > nees_high else "covariance pessimistic"
        lines = [f"{N} runs  ·  mean NEES {_fmt(nees_mean)} (consistent: {d}, 95 % interval {nees_low:.2f}–{nees_high:.2f})"
                 f"  ·  {verdict}",
                 f"biased axes (|mean| > 3 σ/√N): {', '.join(biased_axes) if biased_axes else 'none'}"
                 "  ·  spread: sample std / filter σ (consistent: about 1)"]
        fig.suptitle(title or "Monte Carlo: errors against the filter covariance", x=0.12, ha="left",
                     y=1.0 - 0.12 / side, fontsize=11)
        fig.text(0.12, 1.0 - 0.62 / side, "\n".join(lines), fontsize=8.5,
                 color="#222222" if consistent and not biased_axes else flag, linespacing=1.5)
        handles = [Line2D([], [], ls="", marker="o", ms=5, color=point_color, label="one run"),
                   Line2D([], [], color=cov_color, lw=1.0, label="filter 1σ (3σ dashed)"),
                   Line2D([], [], ls="", marker="X", ms=7, color=point_color, label="sample mean")]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 1.0 - 1.2 / side), fontsize=8)
    return fig


# ------------------------------------------------------------------------------------------------
# B-plane
# ------------------------------------------------------------------------------------------------

_KM6 = None  # [km, km, km, km/s, km/s, km/s], built on first use (needs scarabaeus)


def _state_units():
    global _KM6
    if _KM6 is None:
        import scarabaeus as scb

        km, sec = scb.Units.get_units(["km", "sec"])
        _KM6 = [km, km, km, km / sec, km / sec, km / sec]
    return _KM6


def _bplane_of(bplane, state):
    """[B·T, B·R, LTOF] of one (6,) state in km, km/s, one at a time.

    ``compute_hyperbolic_parameters`` accepts (N, 6) arrays, but ``ArrayWUnits.convert_to``
    truncates an (N, 6) array to (6, 6) once N >= 6, so batches are evaluated row by row.
    """
    import scarabaeus as scb

    return np.asarray(bplane.compute_hyperbolic_parameters(scb.ArrayWUnits(np.asarray(state, float), _state_units()))
                      .values, dtype=float)


def bplane_mapping(bplane, state, covariance, samples=0, seed=0):
    """Map a state and its covariance (relative to the flyby body, J2000, km and km/s) to the B-plane.

    The covariance is mapped with the Jacobian of ``Bplane.compute_hyperbolic_parameters`` with
    respect to the full state, by central differences. ``Bplane.target_covariance`` uses only the
    rotated position covariance, which drops the velocity uncertainty (the dominant term far from
    the body) and the B–LTOF correlation.

    Parameters
    ----------
    bplane : scarabaeus.Bplane
    state : array (6,)
        Relative state of the spacecraft (sc − body), J2000, [km, km/s].
    covariance : array (6, 6)
        Its covariance (an ``ArrayWUnits`` or a plain array in km and km/s).
    samples : int
        Number of Gaussian draws to map nonlinearly (each through the full conic), to check the
        linear ellipse. 0 skips them.

    Returns
    -------
    dict
        ``b`` (3,) [B·T km, B·R km, LTOF s], ``P`` (3, 3) its covariance, ``samples`` (n, 3) or None,
        ``vinf`` [km/s], ``mu`` [km³/s²].
    """
    import scarabaeus as scb

    x0 = np.asarray(getattr(state, "values", state), dtype=float)
    P = np.asarray(getattr(covariance, "values", covariance), dtype=float)
    b0 = _bplane_of(bplane, x0)
    J = np.zeros((3, 6))
    for i in range(6):
        h = 1e-8 * max(np.linalg.norm(x0[:3] if i < 3 else x0[3:]), 1e-3)
        plus, minus = x0.copy(), x0.copy()
        plus[i] += h
        minus[i] -= h
        J[:, i] = (_bplane_of(bplane, plus) - _bplane_of(bplane, minus)) / (2 * h)
    name = bplane.target_name.upper().replace("BARYCENTER", "").strip(" _")
    mu = float(getattr(scb.constants, name).GM.values)
    vinf = float(np.sqrt(np.dot(x0[3:], x0[3:]) - 2 * mu / np.linalg.norm(x0[:3])))
    drawn = None
    if samples:
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(P + 1e-30 * np.eye(6))
        drawn = np.array([_bplane_of(bplane, x0 + L @ rng.standard_normal(6)) for _ in range(samples)])
    return {"b": b0, "P": J @ P @ J.T, "samples": drawn, "vinf": vinf, "mu": mu, "body": name}


def _ellipse_axes(P2):
    """1σ semi-major, semi-minor axes and the orientation of the major axis from T toward R [deg]."""
    values, vectors = np.linalg.eigh(P2)
    major = vectors[:, 1]
    theta = np.degrees(np.arctan2(major[1], major[0]))
    theta = (theta + 90) % 180 - 90  # an axis, not a direction
    return np.sqrt(max(values[1], 0)), np.sqrt(max(values[0], 0)), theta


def plot_bplane(mappings, target=None, body_radius=None, sigma=3, title=None, width=11.0):
    """B-plane of one or several solutions: aimpoint, error ellipses, and the time of flight.

    Three panels:

    - **overview:** the flyby body (and its impact radius, the B magnitude below which the
      trajectory hits it, b = R √(1 + 2μ / (R v∞²))), the B-vector of each solution and the
      target. Ellipses smaller than the panel collapse to their dot;
    - **zoom:** B·T, B·R relative to the target (or the first solution), with the 1σ (filled) and
      ``sigma`` (dashed) ellipses, the mapped Monte Carlo samples if any, and the target;
    - **LTOF:** linearized time of flight to the B-plane relative to the same reference,
      ±1σ and ±``sigma``.

    Click a legend entry to hide or show that solution (interactive backends only; the zoom panel
    refits to what is shown). The legend gives each ellipse's 1σ semi-major and semi-minor axes, the orientation θ of the
    major axis (from T toward R) and σ(LTOF); with a target, also the miss in units of the 2-D
    covariance (√χ², 1 ≈ the 1σ ellipse). R points down, the usual B-plane convention.

    Parameters
    ----------
    mappings : dict
        ``{label: bplane_mapping(...)}``, e.g. one per data cutoff.
    target : sequence, optional
        Target ``(B·T, B·R)`` or ``(B·T, B·R, LTOF)`` [km, km, s].
    body_radius : float, optional
        Radius of the flyby body [km]. Default: the mean radius from ``scarabaeus.constants``.
    sigma : float
        The outer ellipse and interval.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = list(mappings)
    colors = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(labels)}
    markers = {lab: MARKERS[i % len(MARKERS)] for i, lab in enumerate(labels)}
    tags = {}  # label -> its legend text, which the clickable legend matches
    extents = {}
    first = mappings[labels[0]]
    if body_radius is None:
        try:
            import scarabaeus as scb

            body_radius = float(getattr(scb.constants, first["body"]).mean_radius.values)
        except Exception:
            body_radius = None
    ref = np.asarray(target if target is not None else first["b"], dtype=float)
    ref_ltof = ref[2] if ref.size > 2 else first["b"][2]
    ref_name = "target" if target is not None else labels[0]
    for lab in labels:  # legend text: the ellipse in numbers
        m = mappings[lab]
        d, P2 = m["b"][:2] - ref[:2], m["P"][:2, :2]
        smaa, smia, theta = _ellipse_axes(P2)
        text = f"{lab}   SMAA {smaa:.3g}, SMIA {smia:.3g} km, θ {theta:+.0f}°, σ LTOF {np.sqrt(m['P'][2, 2]):.3g} s"
        if target is not None:
            text += f"   miss {np.hypot(*d):.3g} km = {np.sqrt(d @ np.linalg.solve(P2, d)):.2g}σ"
        tags[lab] = text
    shade = "#4d4d4d"

    with _styled():
        height = 5.4
        fig = plt.figure(figsize=(width, height))
        grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 0.6], wspace=0.32,
                                top=1.0 - 0.95 / height, bottom=1.2 / height)
        ax_all, ax_zoom, ax_t = (fig.add_subplot(grid[0, k]) for k in range(3))

        # Overview: body, impact radius, B-vectors
        if body_radius:
            ax_all.add_patch(plt.Circle((0, 0), body_radius, color="#9ab7d3", lw=0, zorder=1))
            b_impact = body_radius * np.sqrt(1 + 2 * first["mu"] / (body_radius * first["vinf"] ** 2))
            ax_all.add_patch(plt.Circle((0, 0), b_impact, fill=False, ec=shade, ls=":", lw=0.9, zorder=1))
            ax_all.text(0, -b_impact, "impact radius", ha="center", va="bottom", fontsize=7, color="#555555")
        for lab in labels:
            bt, br = mappings[lab]["b"][:2]
            _tag(ax_all.plot([0, bt], [0, br], color=colors[lab], lw=0.8, alpha=0.6)[0], tags[lab])
            _tag(ax_all.plot(bt, br, markers[lab], ms=5, color=colors[lab], mec="white", mew=0.6, zorder=4)[0], tags[lab])
        if target is not None:
            _tag(ax_all.plot(ref[0], ref[1], "+", ms=11, mew=1.6, color="#222222", zorder=5)[0], "target")
        ax_all.plot(0, 0, ".", color=shade, ms=3)
        reach = max([np.hypot(*mappings[lab]["b"][:2]) for lab in labels] + [np.hypot(*ref[:2])]
                    + ([b_impact] if body_radius else []))
        ax_all.set_xlim(-1.15 * reach, 1.15 * reach)
        ax_all.set_ylim(1.15 * reach, -1.15 * reach)  # R down
        ax_all.set_aspect("equal")
        ax_all.set_xlabel("B·T [km]")
        ax_all.set_ylabel("B·R [km]")
        ax_all.set_title(f"B-plane, {first['body'].title()}-centered", loc="left", fontsize=10, color="#222222", pad=15)
        ax_all.text(0, 1.02, f"v∞ {first['vinf']:.2f} km/s  ·  R down", transform=ax_all.transAxes,
                    fontsize=8, color="#555555", va="bottom")

        # Zoom on the aimpoint
        handles = []
        for lab in labels:
            m, c = mappings[lab], colors[lab]
            d = m["b"][:2] - ref[:2]
            P2 = m["P"][:2, :2]
            one, outer = _ellipse(P2, 1) + d, _ellipse(P2, sigma) + d
            text = tags[lab]
            extents[lab] = float(np.max(np.abs(np.r_[outer.ravel(), d])))
            if m["samples"] is not None:
                s = m["samples"][:, :2] - ref[:2]
                _tag(ax_zoom.plot(s[:, 0], s[:, 1], "o", ms=1.6, mew=0, color=c, alpha=0.35, zorder=2)[0], text)
            _tag(ax_zoom.fill(one[:, 0], one[:, 1], color=c, alpha=0.15, lw=0, zorder=2)[0], text)
            _tag(ax_zoom.plot(one[:, 0], one[:, 1], color=c, lw=1.4, zorder=3)[0], text)
            _tag(ax_zoom.plot(outer[:, 0], outer[:, 1], color=c, lw=1.0, ls="--", zorder=3)[0], text)
            _tag(ax_zoom.plot(*d, markers[lab], ms=7, color=c, mec="white", mew=0.8, zorder=5)[0], text)
            handles.append(Line2D([], [], color=c, lw=1.4, marker=markers[lab], ms=6, mec="white", label=text))
        if target is not None:
            _tag(ax_zoom.plot(0, 0, "+", ms=13, mew=1.8, color="#222222", zorder=6)[0], "target")
            handles.append(Line2D([], [], ls="", marker="+", ms=10, mew=1.6, color="#222222", label="target"))
        ax_zoom.axhline(0, color=shade, lw=0.6)
        ax_zoom.axvline(0, color=shade, lw=0.6)

        def rescale(hidden):
            """Fit the zoom panel to the solutions still shown."""
            shown = [extents[lab] for lab in labels if tags[lab] not in hidden] or list(extents.values())
            extent = max(shown)
            ax_zoom.set_xlim(-1.08 * extent, 1.08 * extent)
            ax_zoom.set_ylim(1.08 * extent, -1.08 * extent)

        rescale(set())
        ax_zoom.set_aspect("equal")
        ax_zoom.set_xlabel(f"Δ B·T from {ref_name} [km]")
        ax_zoom.set_ylabel(f"Δ B·R from {ref_name} [km]")
        ax_zoom.set_title("Aimpoint dispersion", loc="left", fontsize=10, color="#222222", pad=15)
        note = f"1σ filled, {sigma:g}σ dashed" + (", dots: mapped samples" if any(
            mappings[lab]["samples"] is not None for lab in labels) else "")
        ax_zoom.text(0, 1.02, note, transform=ax_zoom.transAxes, fontsize=8, color="#555555", va="bottom")

        # Linearized time of flight
        x = np.arange(len(labels))
        dt = np.array([mappings[lab]["b"][2] - ref_ltof for lab in labels])
        sig_t = np.array([np.sqrt(mappings[lab]["P"][2, 2]) for lab in labels])
        for k, lab in enumerate(labels):
            c = colors[lab]
            _tag(ax_t.plot([k, k], [dt[k] - sigma * sig_t[k], dt[k] + sigma * sig_t[k]], color=c, lw=5, alpha=0.18,
                           solid_capstyle="butt")[0], tags[lab])
            _tag(ax_t.plot([k, k], [dt[k] - sig_t[k], dt[k] + sig_t[k]], color=c, lw=5, alpha=0.45,
                           solid_capstyle="butt")[0], tags[lab])
            _tag(ax_t.plot(k, dt[k], markers[lab], ms=6.5, color=c, mec="white", mew=0.8, zorder=4)[0], tags[lab])
            if mappings[lab]["samples"] is not None:
                spread = np.std(mappings[lab]["samples"][:, 2], ddof=1)
                for edge in (dt[k] + spread, dt[k] - spread):
                    _tag(ax_t.plot([k - 0.28, k + 0.28], [edge] * 2, color=c, lw=0.9)[0], tags[lab])
        ax_t.axhline(0, color=shade, lw=0.6)
        ax_t.set_xticks(x)
        ax_t.set_xticklabels([f"{i + 1}" for i in x])
        ax_t.set_xlim(-0.6, len(labels) - 0.4)
        ax_t.grid(axis="x", visible=False)
        ax_t.set_xlabel("solution (legend order)")
        ax_t.set_ylabel(f"Δ LTOF from {ref_name} [s]")
        ax_t.set_title("Time of flight", loc="left", fontsize=10, color="#222222", pad=15)
        ax_t.text(0, 1.02, f"±1σ dark, ±{sigma:g}σ light" + (", ticks: sample σ" if any(
            mappings[lab]["samples"] is not None for lab in labels) else ""), transform=ax_t.transAxes,
                  fontsize=8, color="#555555", va="bottom")

        top = 1.0 - 0.1 / height
        fig.suptitle(title or "B-plane targeting uncertainty", x=0.125, ha="left", y=top, fontsize=11)
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.12, 0.0), ncol=1, handletextpad=0.5,
                   fontsize=8)
        interactive_legend(fig, on_toggle=rescale)
    return fig
