# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""
Supplementary Plotting

Provides plotting functions for Scarabaeus (SCB) tutorials.

Some SCB tutorials provide plots that require complicated and/or 
irrelevant code. In order to keep the focus on SCB and not the 
details of plotting, some tutorials may choose to implement their 
plotting code here instead.
"""
import scarabaeus as scb

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches

import numpy as np

# shared colors for plotting
@dataclass
class colors:
    gold = '#CFB87C'
    blue = '#0277bd'
    planets = [plt.cm.tab10(i / 10) for i in range(10)]

def plot_epocharray_table(time_to_show: scb.EpochArray, tsc_path: str) -> None:
    """ Table of all EpochArray systems and representation combinations.

        Parameters
        ----------
        time_to_show : str or float
            The time time to pass to EpochArray for all combinations.
        
        tsc_path : str
            Path to the .tsc file to note where SCLK ticks come from.
    """
    # combinations
    systems         = ['TDB', 'ET', 'TT', 'TDT', 'UTC',
                       'TAI', 'GPS', 'SCLK']
    representations = ['NUM', 'AWU', 'CAL', 'DOY',
                       'ISOC', 'ISOD', 'JDTDB', 
                       'JED', 'JDTDT', 'JDUTC']

    # convert for all combos and store to table
    tbl_dict = {}
    base = time_to_show.to('TDB', 'NUM')
    for sy in systems:
        ep_sys = base.to(sys = sy)
        tbl_dict[sy] = {}
        for r in representations:
            # skip bad combos with an N/A
            try:
                ep_rep = ep_sys.to(rep=r)
                tbl_dict[sy][r] = str(ep_rep.times)
            except:
                tbl_dict[sy][r] = 'N/A'
                continue
                
    
    ## plot
    fig, ax = plt.subplots(figsize = (16, 8), constrained_layout = True)
    ax.axis('off')

    # create table
    cell_text = [[tbl_dict[sy][r] for r in representations]
                 for sy in systems]
    table = ax.table(cellText  = cell_text,
                     rowLabels = [f'sys = {sy}' + ('*' if sy == 'SCLK' else '')
                                  for sy in systems],
                     colLabels = [f'rep = {r}' for r in representations],
                     cellLoc   = 'center',
                     bbox      = [0, 0.04, 1, 0.96])

    # format it
    table.auto_set_column_width(col = list(range(len(representations))))
    table.set_fontsize(8)

    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            # row and column titles
            cell.set_facecolor(colors.gold)
            cell.set_text_props(color = 'black', fontweight = 'bold')
        else:
            # every other cell is white
            cell.set_facecolor('#f0f0f0' if row % 2 == 0 else 'white')
        cell.set_edgecolor('#cccccc')   # cell borders

    fig.suptitle('EpochArray Time System Conversions and Representations\n'
                 f'Epoch = {time_to_show.to(rep = "CAL")}')
    tsc_rel_path = Path(tsc_path).relative_to(Path.cwd())
    fig.text(0.55, 0.01, f'* SCLK coefficents defined in {tsc_rel_path}',
             va = 'bottom', ha = 'left', fontsize = 10)
    plt.show()
    
def plot_normals(normals, times):
    """ Queried attitude from CK with slider.
    
        Parameters
        ----------
        normals : 
            DESC
        times : 
            DESC
    """
    fig = plt.figure(figsize = (9, 8))
    axes = fig.subplot_mosaic([['3d', 'x'],
                               ['3d', 'x'],
                               ['3d', 'y'],
                               ['3d', 'y'],
                               ['3d', 'z'],
                               ['leg', 'z']],
                               per_subplot_kw = {'3d' : {'projection' : '3d'}})
    ax = axes['3d']
    fig.subplots_adjust(bottom = 0.25)

    ## 3d representation
    # first normal
    l1 = ax.plot([0, normals[0][0][0]], [0, normals[0][0][1]], [0, normals[0][0][2]], 
                 'b', lw = 2)
    p1 = ax.plot(normals[0][0][0], normals[0][0][1], normals[0][0][2], 'ro')

    # second normal
    l2 = ax.plot([0, normals[0][1][0]], [0, normals[0][1][1]], [0, normals[0][1][2]], 
                 'g', lw = 2)
    p2 = ax.plot(normals[0][1][0], normals[0][1][1], normals[0][1][2], 'mo')

    ## 2d representation
    # extract components for both normals
    axx, axy, axz = axes['x'], axes['y'], axes['z']
    x1s, x2s, y1s, y2s, z1s, z2s = [], [], [], [], [], []
    for normal in normals:
        frst, scnd = normal[0], normal[1]

        x1s.append(frst[0])
        y1s.append(frst[1])
        z1s.append(frst[2])

        x2s.append(scnd[0])
        y2s.append(scnd[1])
        z2s.append(scnd[2])
    
    # get times just as values
    start, end = times[0], times[-1]
    t_vals   = np.linspace(start.times.values, end.times.values, times.size)

    # plot x component
    axx.plot(t_vals, x1s, 'b')
    axx.plot(t_vals, x2s, 'g')
    marker_x1 = axx.axvline(t_vals[0], ls = '--', c = 'k')
    dot_x1    = axx.plot(t_vals[0], x1s[0], 'ro')
    dot_x2    = axx.plot(t_vals[0], x2s[0], 'mo')
    axx.grid()
    axx.set_title('X Component')

    # plot y component
    axy.plot(t_vals, y1s, 'b')
    axy.plot(t_vals, y2s, 'g')
    marker_y1 = axy.axvline(t_vals[0], ls = '--', c = 'k')
    dot_y1    = axy.plot(t_vals[0], y1s[0], 'ro')
    dot_y2    = axy.plot(t_vals[0], y2s[0], 'mo')
    axy.grid()
    axy.set_title('Y Component')

    # plot z component
    axz.plot(t_vals, z1s, 'b')
    axz.plot(t_vals, z2s, 'g')
    marker_z1 = axz.axvline(t_vals[0], ls = '--', c = 'k')
    dot_z1    = axz.plot(t_vals[0], z1s[0], 'ro')
    dot_z2    = axz.plot(t_vals[0], z2s[0], 'mo')
    axz.grid()
    axz.set_title('Z Component')

    ## formatting
    fig.suptitle(f'Queried Normals Between SCLK\n{t_vals[0]} & {t_vals[-1]}')
    fig.subplots_adjust(hspace = 1.25)

    # place legend in its own empty subplot
    ax_leg = axes['leg']
    ax_leg.set_axis_off()
    ax_leg.plot(0, 0, 'b', lw = 2, label = 'ORX_SA_PY_IG')
    ax_leg.plot(0, 0, 'g', lw = 2, label = 'ORX_SA_NY_IG')
    ax_leg.legend(loc = 'center')

    # define the values to use for snapping
    time_ticks = np.linspace(0, len(t_vals), len(t_vals)+1)
    ax_amp = fig.add_axes([0.225, 0.15, 0.65, 0.03])

    # make time slider
    time_slider = Slider(
        ax_amp, "SCLK Ticks", 0, len(t_vals)-1,
        valinit = 0, valstep = time_ticks,
        color = "green"
    )

    def update(_):
        time = int(time_slider.val)
        ## first normal
        # extract data
        x_pos = normals[time][0][0]
        y_pos = normals[time][0][1]
        z_pos = normals[time][0][2]
        # update plots
        l1[0].set_data_3d([0, x_pos], [0, y_pos], [0, z_pos])
        p1[0].set_data_3d([x_pos, x_pos], [y_pos, y_pos], [z_pos, z_pos])
        marker_x1.set_xdata([t_vals[time], t_vals[time]])
        dot_x1[0].set_xdata([t_vals[time], t_vals[time]])
        dot_x1[0].set_ydata([x1s[time], x1s[time]])
        marker_y1.set_xdata([t_vals[time], t_vals[time]])
        dot_y1[0].set_xdata([t_vals[time], t_vals[time]])
        dot_y1[0].set_ydata([y1s[time], y1s[time]])
        marker_z1.set_xdata([t_vals[time], t_vals[time]])
        dot_z1[0].set_xdata([t_vals[time], t_vals[time]])
        dot_z1[0].set_ydata([z1s[time], z1s[time]])

        ## second normal
        # extract data
        x_pos = normals[time][1][0]
        y_pos = normals[time][1][1]
        z_pos = normals[time][1][2]
        # update plots
        l2[0].set_data_3d([0, x_pos], [0, y_pos], [0, z_pos])
        p2[0].set_data_3d([x_pos, x_pos], [y_pos, y_pos], [z_pos, z_pos])
        dot_x2[0].set_xdata([t_vals[time], t_vals[time]])
        dot_x2[0].set_ydata([x2s[time], x2s[time]])
        dot_y2[0].set_xdata([t_vals[time], t_vals[time]])
        dot_y2[0].set_ydata([y2s[time], y2s[time]])
        dot_z2[0].set_xdata([t_vals[time], t_vals[time]])
        dot_z2[0].set_ydata([z2s[time], z2s[time]])
        
        fig.canvas.draw_idle()

    time_slider.on_changed(update)

    ax_reset = fig.add_axes([0.5, 0.05, 0.1, 0.04])
    button = Button(ax_reset, 'Reset', hovercolor='0.975')

    def reset(event):
        time_slider.reset()
    button.on_clicked(reset)

    plt.show()

def plot_planets(planets: list, planet_traj: dict, epochs: scb.EpochArray):
    """ Queried planetary ephemerides.
    
        Parameters
        ----------
        planets : list
            The planets to plot.
        planet_traj : dict
            Dictionary containing queried planet trajectories.
        epochs: EpochArray
            The epochs over which the planet trajectories were queried.
    """
    epochs = epochs.to(rep = 'CAL')

    fig, ax = plt.subplots(figsize = (6, 6))
    fig.suptitle(('Solar System Heliocentric Orbits\n'
                  f'{epochs.times[0]} | {epochs.times[-1]}'),
                fontweight='bold', fontsize=11)

    # full zoom
    for i, name in enumerate(planets):
        xy = planet_traj[name]
        ax.plot(xy[:, 0], xy[:, 1], color = colors.planets[i], alpha = 0.6, lw = 1.2)
        ax.scatter(xy[-1, 0], xy[-1, 1], color = colors.planets[i], 
                   marker = '.', s = 40, label = name)
    ax.plot(0, 0, '*', color = 'goldenrod', ms = 8)

    # inner planets detail
    det = ax.inset_axes([0.0, 0.03, 0.5, 0.5],
                        xlim = (-1.75, 1.75), ylim = (-1.75, 1.75),
                        xticklabels = [], yticklabels = [])
    for i, name in enumerate(planets[:4]):
        xy = planet_traj[name]
        det.plot(xy[:, 0], xy[:, 1], color = colors.planets[i], 
                 alpha = 0.6, lw = 1.2)
        det.scatter(xy[-1, 0], xy[-1, 1], color = colors.planets[i], 
                    marker = 'o', s = 20)
    det.plot(0, 0, '*', color = 'goldenrod', ms = 8)

    # formatting
    det.set_aspect('equal')
    ax.indicate_inset_zoom(det, edgecolor='k')

    ax.legend(ncol = 2, fontsize = 8)
    ax.set_aspect('equal')
    ax.set_xlabel('X [AU]  (J2000)')
    ax.set_ylabel('Y [AU]  (J2000)')
    plt.tight_layout()
    plt.show()

def plot_awu_awf_visuals(R):
    COLORS = [plt.cm.tab10(i / 10) for i in range(5)]

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    fig.suptitle('ArrayWUnits and ArrayWFrame Visualization',
                fontweight='bold', fontsize=11)

    # ── left: vector arithmetic (J2000, XY plane) ────────────────────
    v1  = np.array([1000.0, 200.0, 0.0])   # r1 XY
    v2  = np.array([ 500.0, 100.0, 0.0])   # r2 XY
    vs  = v1 + v2                           # r1 + r2

    ax = axes[0]
    for vec, lbl, c, ls in [(v1, 'r₁', COLORS[0], '-'),
                            (v2, 'r₂', COLORS[1], '-'),
                            (vs, 'r₁ + r₂', COLORS[2], '--')]:
        ax.annotate('', xy=(vec[0], vec[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=c, lw=2.2, ls=ls))
        ax.text(vec[0] * 1.04, vec[1] * 1.04, lbl, color=c,
                fontsize=10, fontweight='bold')

    ax.set_xlim(-100, 1700); ax.set_ylim(-100, 450)
    ax.set_xlabel('x  [km]  (J2000)'); ax.set_ylabel('y  [km]  (J2000)')
    ax.set_title('ArrayWFrame arithmetic — same-frame vectors sum correctly')
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.3)
    ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.3)

    # ── right: DCM rotation — J2000 vs IAUEARTH ──────────────────────
    # Re-use DCM and position from earlier cells
    p_j2k = np.array([7000.0, -1200.0, 3500.0])   # ISS-like position [km]
    p_ear  = R @ p_j2k                              # rotated to IAUEARTH

    ax2 = axes[1]
    # plot XZ plane (best separates the two frames visually)
    for vec, lbl, c in [(p_j2k, 'J2000',    COLORS[0]),
                        (p_ear, 'IAUEARTH', COLORS[3])]:
        ax2.annotate('', xy=(vec[0], vec[2]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=c, lw=2.2))
        ax2.text(vec[0] * 1.04, vec[2] * 1.04, f'{lbl}\n{np.round(vec, 0)[:2]} …',
                color=c, fontsize=8.5, fontweight='bold')

    lim = 8500
    ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
    ax2.set_xlabel('x  [km]'); ax2.set_ylabel('z  [km]')
    ax2.set_title('DCM rotation — same |r|, different frame components')
    ax2.axhline(0, color='k', lw=0.5, ls='--', alpha=0.3)
    ax2.axvline(0, color='k', lw=0.5, ls='--', alpha=0.3)
    ax2.set_aspect('equal')
    note = f'|r| = {np.linalg.norm(p_j2k):.1f} km (invariant)'
    ax2.text(0.97, 0.03, note, transform=ax2.transAxes, ha='right',
            fontsize=8, color='grey', style='italic')

    plt.tight_layout()
    plt.show()

def plot_simple_orbit(positions):
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle('Heliocentric Trajectory\nKeplerian Force Model')
    ax.plot(positions[:, 0], positions[:, 1], lw=1.5, label='ORCCA SC (Keplerian)')
    ax.plot(0, 0, '*', color = 'goldenrod', ms=14, label='Sun')
    ax.plot(positions[0, 0],  positions[0, 1],  'go', label='t₀  2021-Jan-01')
    ax.plot(positions[-1, 0], positions[-1, 1], 'ro', label='t_f  2021-Feb-01')
    ax.set_xlabel('X [AU]  (J2000)')
    ax.set_ylabel('Y [AU]  (J2000)')
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_ias15_vs_dop853(pos_dop, pos_ias, t_hrs, diff_km, n, AU_km):
    COLORS = [plt.cm.tab10(i / 10) for i in range(2)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('DOP853 vs IAS15 — Keplerian Force Model, 31-day Arc',
                fontweight='bold', fontsize=11)

    # left: trajectories (should overlap perfectly)
    axes[0].plot(pos_dop[:, 0] / AU_km, pos_dop[:, 1] / AU_km,
                color=COLORS[0], lw=2.0, label='DOP853')
    axes[0].plot(pos_ias[:n, 0] / AU_km, pos_ias[:n, 1] / AU_km,
                color=COLORS[1], lw=1.2, ls='--', label='IAS15')
    axes[0].set_xlabel('X [AU]  (J2000)')
    axes[0].set_ylabel('Y [AU]  (J2000)')
    axes[0].set_title('Trajectories (should overlap)')
    axes[0].set_aspect('equal')
    axes[0].legend()

    # right: position difference
    axes[1].semilogy(t_hrs, diff_km, color='purple', lw=1.5)
    axes[1].set_xlabel('Time from t₀ [hr]')
    axes[1].set_ylabel('|r_DOP853 − r_IAS15| [km]')
    axes[1].set_title('Position difference (numerical agreement)')

    plt.tight_layout()
    plt.show()

def plot_bplane_covariance(BT_ref, BR_ref, BT_pert, BR_pert, 
                           BT_lin, BR_lin, BT_nl, BR_nl, rho_btbr,
                           evec, ev, P2, maxsig, sigma_bt, sigma_br):
    def et2dt(et_arr):
        """Convert SPICE ET seconds (TDB from J2000) to list of datetime objects."""
        _J2000 = datetime(2000, 1, 1, 12, 0, 0)
        return [_J2000 + timedelta(seconds=float(t)) for t in np.atleast_1d(et_arr)]


    def _cov_ellipse(ax, cx, cy, P2, nsig, color, alpha, edgecolor='navy', lw=0.9):
        """Draw a 2D confidence ellipse from a 2×2 covariance matrix."""
        ev, evec = np.linalg.eigh(P2)
        ev       = np.maximum(ev, 0.0)
        angle    = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
        ax.add_patch(Ellipse(
            xy=(cx, cy),
            width =2 * nsig * np.sqrt(ev[1]),
            height=2 * nsig * np.sqrt(ev[0]),
            angle =angle,
            facecolor=color, edgecolor=edgecolor,
            linewidth=lw, alpha=alpha, zorder=2,
        ))
    
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("#fafafa")

    # ── sigma ellipses around the target ─────────────────────────────
    ell_colors  = ["#cfe8ff", "#7fb6ff", "#2c7be5"]
    ell_labels  = ["3-σ region", "2-σ region", "1-σ region"]
    for nsig, color, lbl in zip([3, 2, 1], ell_colors, ell_labels):
        _cov_ellipse(ax, BT_ref, BR_ref, P2, nsig, color, alpha=0.85,
                    edgecolor="navy", lw=1.4)

    # ── principal-axis arrows ─────────────────────────────────────────
    major = evec[:, 0];  Lmaj = 3.8 * np.sqrt(ev[0])
    minor = evec[:, 1];  Lmin = 3.8 * np.sqrt(ev[1])
    kw = dict(lw=2.2, alpha=0.7)
    ax.plot([BT_ref - Lmaj*major[0], BT_ref + Lmaj*major[0]],
            [BR_ref - Lmaj*major[1], BR_ref + Lmaj*major[1]], '--', color='#d62728', **kw)
    ax.plot([BT_ref - Lmin*minor[0], BT_ref + Lmin*minor[0]],
            [BR_ref - Lmin*minor[1], BR_ref + Lmin*minor[1]], '--', color='#2ca02c', **kw)

    # ── key points ────────────────────────────────────────────────────
    scatter_kw = dict(zorder=10, linewidths=1.4, edgecolors='k')
    ax.scatter(BT_ref,  BR_ref,  s=260, marker='X',  color='#ffcc00',  label='Target aimpoint (ref)',      **scatter_kw)
    ax.scatter(BT_pert, BR_pert, s=140, marker='o',  color='#d62728',  label='Perturbed (no TCM)',         **scatter_kw)
    ax.scatter(BT_lin,  BR_lin,  s=140, marker='^',  color='#2ca02c',  label='Post-TCM linear',            **scatter_kw)
    ax.scatter(BT_nl,   BR_nl,   s=140, marker='s',  color='#9467bd',  label='Post-TCM nonlinear',         **scatter_kw)

    # Earth at origin
    ax.scatter(0, 0, s=180, color='royalblue', marker='o', zorder=8,
            edgecolors='k', linewidths=1.2, label='Earth barycentre')

    # ── annotation lines to perturbed / corrected ─────────────────────
    for xp, yp, col in [(BT_pert, BR_pert, '#d62728'), (BT_lin, BR_lin, '#2ca02c'),
                        (BT_nl,  BR_nl,  '#9467bd')]:
        ax.annotate("", xy=(xp, yp), xytext=(BT_ref, BR_ref),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.3, alpha=0.6))

    # ── ell legend patches ────────────────────────────────────────────
    ell_patches = [mpatches.Patch(color=c, label=l, alpha=0.85)
                for c, l in zip(ell_colors[::-1], ell_labels[::-1])]

    # ── style ─────────────────────────────────────────────────────────
    ax.set_title("OSIRIS-REx Earth Gravity Assist — B-plane",
                fontsize=16, fontweight='bold', pad=16)
    ax.set_xlabel(r"$B \cdot \hat{T}$  [km]", fontsize=13)
    ax.set_ylabel(r"$B \cdot \hat{R}$  [km]", fontsize=13)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_xlim(BT_ref - maxsig, BT_ref + maxsig)
    ax.set_ylim(BR_ref + maxsig, BR_ref - maxsig)
    for sp in ax.spines.values(): sp.set_linewidth(1.3)

    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles + ell_patches, lbls + [p.get_label() for p in ell_patches],
            loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_fb_traj(r, v, hours, h_burn_start, h_burn_end):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for ax, vals, ylabel, labels in [
        (axes[0], r, 'Position [km]',   [r'$x$',  r'$y$',  r'$z$']),
        (axes[1], v, 'Velocity [km/s]', [r'$v_x$', r'$v_y$', r'$v_z$']),
    ]:
        for i, lbl in enumerate(labels):
            ax.plot(hours, vals[:, i], label=lbl)
        ax.axvspan(h_burn_start, h_burn_end, alpha=0.12, color='orange', label='Burn #0')
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right', fontsize=9)

    axes[1].set_xlabel('Time from coast start [hr]')
    axes[0].set_title('Heliocentric Trajectory — Finite-Burn Mission Sequence')
    plt.tight_layout()
    plt.show()