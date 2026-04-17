# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:34:02 2026

@author: dhouten
"""

import numpy as np
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HARTREE_TO_EV = 27.21138602

# Dictionary to map plain text to LaTeX labels for nice axis labels
latex_labels = {
    "GAMMA": r'$\Gamma$',
    "G": r'$\Gamma$',
    "K": r'$K$',
    "M": r'$M$',
    "X": r'$X$',
    "Y": r'$Y$',
    "Z": r'$Z$',
    "L": r'$L$',
    "W": r'$W$',
    "U": r'$U$',
    # Add more if needed
}


def import_gnuplot_stitch(filename):
    """
    Read a .gnuplot band structure file and store the data in a dictionary.

    Parameters
    ----------
    filename : str
        Path to the gnuplot file.

    Returns
    -------
    data : dict
        Keys are (spin, band) tuples.
        Values are lists of (path_label, k_x, energy_eV).
    """
    data = defaultdict(list)  # {(spin, band): [(path, x, y), ...]}
    current_spin = None
    current_band = None
    current_path = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            # Header lines start with '#'
            if line.startswith('#'):
                # Example header: "# GAMMA-K spin 1 band 5"
                match = re.search(r'(\S+) spin (\d+) band (\d+)', line)
                if match:
                    current_path, current_spin, current_band = match.groups()
                    current_spin = int(current_spin)
                    current_band = int(current_band)

            # Data lines: numbers separated by spaces
            elif line:
                parts = line.split()
                if (len(parts) >= 2 and current_spin is not None and
                        current_band is not None):
                    x, y_hartree = map(float, parts[:2])
                    # Convert from Hartree to eV and apply shift ToV
                    y_eV = y_hartree * HARTREE_TO_EV
                    data[(current_spin, current_band)].append(
                        (current_path, x, y_eV)
                    )

    return data


def stitch(data):
    """
    Stitch together band structure k-path segments.

    Parameters
    ----------
    data : dict
        Output from import_gnuplot_stitch.

    Returns
    -------
    stitched_x : list
        The common x-axis values for all stitched bands.
    bands_y : np.ndarray
        2D array containing the energy values of all stitched bands.
    buf_max : float or None
        The maximum energy of the band that is fully below the Fermi level, or
        None if no such band exists.
    xticks : list
        The x-axis tick positions.
    xtick_labels : list
        The labels for the x-axis ticks.
    """

    # -------------------------------
    # Step 1: determine the order of the k-paths
    # -------------------------------
    path_order = []  # e.g. ["GAMMA-K", "K-M", ...] in the order they appear
    seen_paths = set()

    for (spin, band), points in data.items():
        for (path, _, _) in points:
            if path not in seen_paths:
                path_order.append(path)
                seen_paths.add(path)

    # -------------------------------
    # Step 2: compute shifts so each path is placed after the previous one
    # -------------------------------
    path_shifts = {}
    cumulative_shift = 0.0

    for path in path_order:
        xs = []
        for (spin, band), points in data.items():
            # collect all x-values belonging to this path
            xs.extend([x for (p, x, _) in points if p == path])

        if xs:
            xmin = min(xs)
            xmax = max(xs)

            # shift so that the leftmost point of this path is
            # at cumulative_shift
            path_shifts[path] = cumulative_shift - xmin

            # increase cumulative_shift by the width of this path
            cumulative_shift += (xmax - xmin)

    # -------------------------------
    # Step 3: build stitched bands (same x-grid, different y for each band)
    # -------------------------------

    bands_y = []  # list of lists: [band_0_y_values, band_1_y_values, ...]
    stitched_x = None   # will store the common stitched x-axis
    max_stitched_x = 0  # for x-axis limit

    for (spin, band), points in data.items():
        current_x = []
        current_y = []

        for (path, x, y) in points:
            shift = path_shifts[path]
            x_stitched = x + shift

            current_x.append(x_stitched)
            current_y.append(y)

        # store y-values of this band
        bands_y.append(current_y)

        # update global x-axis (we assume all bands share the same x-grid)
        if stitched_x is None:
            stitched_x = current_x

        # update maximum x-value
        if current_x:
            max_stitched_x = max(max_stitched_x, max(current_x))

    bands_y = np.array(bands_y)

    # Find the highest band that is fully below the Fermi level
    buf_index = None
    for idx, band_vals in enumerate(bands_y):
        # if all points of this band are below the Fermi level
        if np.all(band_vals < fermi):
            buf_index = idx
    if buf_index is None:
        buf_max = None
        print("No band found with all values below the Fermi level.")
    else:
        buf_max = max(bands_y[buf_index])
        print(f"Maximum energy of band below Fermi level: {buf_max:.3f} eV")

    # -------------------------------
    # Step 6: add vertical lines and labels at high-symmetry points
    # -------------------------------
    xticks = []
    xtick_labels = []

    for path in path_order:
        xs = []
        for (spin, band), points in data.items():
            xs.extend([x for (p, x, _) in points if p == path])

        if xs:
            xmin = min(xs)
            shift = path_shifts[path]
            stitched_position = xmin + shift

            xticks.append(stitched_position)

            # Label: take the left symbol of "GAMMA-K", i.e. "GAMMA"
            label_start = path.split('-')[0]
            label = latex_labels.get(label_start, label_start)
            xtick_labels.append(label)

    # Add final high-symmetry point label (the end of the last path)
    if stitched_x is not None:
        last_path = path_order[-1]
        last_label_end = last_path.split('-')[-1]
        xticks.append(max_stitched_x)
        xtick_labels.append(latex_labels.get(last_label_end, last_label_end))

    return stitched_x, bands_y, buf_max, xticks, xtick_labels


def plot_bands(stitched_x, bands_y, shift, xticks, xtick_labels,
               color='black', linewidth=1.5, ylim=None):
    """
    Stitch together band structure k-path segments and plot them.

    Parameters
    ----------
    stitched_x : list
        The common x-axis values for all stitched bands.
    bands_y : np.ndarray
        2D array containing the energy values of all stitched bands.
    shift : float
        The energy shift to apply to all bands.
    xticks : list
        The x-axis tick positions.
    xtick_labels : list
        The labels for the x-axis ticks.
    color : str, optional
        Currently not used for highlighted bands; background bands are gray.
    linewidth : float, optional
        Line width for highlighted bands.
    ylim : [ymin, ymax] or None
        Optional y-limits for the plot.

    Returns
    -------
    bands : np.ndarray
        The energy values of the bands after applying the shift.
    """

    fig, ax = plt.subplots(figsize=(6.0 / 2, 6.0 / 2))  # square figure
    bands_y = bands_y + shift  # apply energy shift to all bands

    # -------------------------------
    # Step 4: find the band that crosses (or lies very near) 0 eV
    # -------------------------------
    zero_band_index = None

    for idx, band_vals in enumerate(bands_y):
        # if any point of this band is in [-0.03, 0.03] eV,
        # treat it as the "zero" band
        if np.any((band_vals >= -0.03) & (band_vals <= 0.03)):
            zero_band_index = idx

    # If we found a zero band, rotate the array so that this band is first
    if zero_band_index is not None:
        bands_y = np.roll(bands_y, -zero_band_index, axis=0)
    else:
        print("No band found with values near zero.")

    # -------------------------------
    # Step 5: plot all bands
    # -------------------------------
    # Background bands (all) in dashed gray
    for band_vals in bands_y:
        plt.plot(stitched_x, band_vals, '--', color='gray', linewidth=0.5)

    # Highlight first two bands in solid colors (red, blue)
    if len(bands_y) > 0:
        plt.plot(stitched_x, bands_y[0], '-', color='red', linewidth=linewidth)
    if len(bands_y) > 1:
        plt.plot(stitched_x, bands_y[1], '-', color='blue',
                 linewidth=linewidth)

    # Draw vertical lines at each high-symmetry point
    for xtick in xticks:
        plt.axvline(x=xtick, color='black', linewidth=0.5)

    # Horizontal line at 0 eV
    plt.axhline(0, color='black', linewidth=0.5)

    # Make plot square
    ax.set_box_aspect(1)

    # Y-axis tick spacing (major every 0.5 eV, minor every 0.1 eV)
    ax.yaxis.set_major_locator(MultipleLocator(0.50))
    ax.yaxis.set_minor_locator(MultipleLocator(0.10))

    # Apply x-ticks and labels
    plt.xticks(xticks, xtick_labels)

    # Axis labels and limits
    plt.ylabel("Energy (eV)")
    plt.xlim(0, xticks[-1])

    if ylim:
        plt.ylim(ylim)

    plt.show()

    return bands_y


# -------------------------------
# Example usage
# -------------------------------
# filename = r"C:\Users\dhouten\PhD_David\WSe2_bands_GGA_PW91_SO_DZ.gnuplot"
# replace with your filename(s)
filename = r"./data/band.gnuplot"
fermi_filename = r"./data/band.csv"

fermi = np.loadtxt(fermi_filename, delimiter=',', skiprows=1)[0, 3] \
    * HARTREE_TO_EV
print('Fermi level (eV): ', fermi)

gnuplot_data = import_gnuplot_stitch(filename)

# Plot stitched bands, highlighting first two bands
stitched_x, bands_y, buf_max, xticks, xtick_labels = \
    stitch(gnuplot_data)

# Shift bands by energy in eV (e.g. to set VBM or Fermi level as zero)
# shift = 0.  # no shift
# shift = 5.06  # eV arbitrary value
# shift = -fermi  # Fermi level at 0 eV - useful for metals
shift = -buf_max  # Max band under Fermi at 0 eV - useful for semiconductors

bands = plot_bands(stitched_x, bands_y, shift, xticks, xtick_labels,
                   color='blue', linewidth=2, ylim=[-2, 3])
print(min(bands[1])-max(bands[0]))
print(r'BG_K = ', abs((bands[1][np.argmax(bands[0])]-max(bands[0]))*1), 'eV')
print(r'ΔSOC_VB = ', abs((max(bands[-1])-max(bands[0]))*1000), 'meV')
print(r'ΔSOC_CB = ',
      abs((bands[1][np.argmax(bands[0])]-bands[2][np.argmax(bands[0])])*1000),
      'meV')
print('ΔGamma-K = ', abs((bands[-1][0]-max(bands[0]))*1000), 'meV')
print('ΔK-Q_CB = ',
      (bands[1][np.argmax(bands[0])]-np.min(bands[1]))*1000,
      'meV')
