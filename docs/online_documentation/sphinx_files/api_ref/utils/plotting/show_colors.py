.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC
"""
    Script to display all colors in scb.Plotting.color_palettes property.
"""
import sys
sys.path.append("./src/")
import scarabaeus as scb

import math
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

def plot_colortable(colors, *, ncols=4, sort_colors=True):
    # NOTE: adapated from matplotlib
    #       https://matplotlib.org/stable/gallery/color/named_colors.html
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 32

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 100

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                    height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig

# plot for each color palette
for palette, colors in scb.Plotting.color_palettes.items():
    # extract colors from palette and convert to rgb
    colors_to_show = {}
    for i, color in enumerate(colors):
        hex_to_rgb = mcolors.to_rgb(color)
        colors_to_show[f'{i}'] = hex_to_rgb
    
    # plot
    plot_colortable(colors_to_show, ncols=2, sort_colors=False)
    plt.title(palette)
    plt.show()