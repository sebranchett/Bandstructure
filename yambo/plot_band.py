from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import os
import re


def read_data(filename):
    """
    Reads data from a file and assigns column names based on a specified row.

    Parameters:
    filename (str): The path to the file to be read.

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the file with
    appropriate column names.
    """
    data = pd.read_csv(filename, sep=r'\s+', comment='#', header=None)
    # Find the row containing the string "(a.u.)"
    col_title_row = None
    title_line = None
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if "(a.u.)" in line:
                col_title_row = i
                title_line = line
                break
    if col_title_row is None:
        raise ValueError("Column title row not found in file: " + filename)

    # Read column names by splitting the title line on 2 or more spaces
    num_columns = data.shape[1]
    colnames = re.split(r'\s{2,}', title_line.lstrip('#').strip())
    colnames = [c.strip() for c in colnames if c.strip()]

    # If there is one more column than names, then the last column name is
    # "symmetry_label"
    if len(colnames) == num_columns - 1:
        colnames.append("symmetry_label")

    if len(colnames) != num_columns:
        raise ValueError(
            f"Number of column names ({len(colnames)}) does not match "
            f"number of columns in data ({num_columns})."
        )

    data.columns = colnames
    return data


def labels(data, prefix, omit_last=4):
    """
    Generate a list of labels for the columns of a DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the columns to label.
    prefix (str): The prefix to add to each label.
    omit_last (int, optional): The number of columns to omit from the end.
    Default is 4, but set to 3 for BSE as it does not have symmetry labels.

    Returns:
    list: A list of labels with the specified prefix, excluding the last
    'omit_last' columns.
    """
    return (prefix + " - " + data.columns[1:-omit_last]).to_list()


def plot_bands(file_dft="", file_gw="", file_bse="", plot_title="",
               output_file="band_structure.png", label_all_bands=False,
               ymin=None, ymax=None):
    """
    Plots the band structure from DFT, GW, and BSE data files.
    Parameters:
    file_dft (str, optional): Path to the DFT data file. Default is Empty.
    file_gw (str, optional): Path to the GW data file. Default is Empty.
    file_bse (str, optional): Path to the BSE data file. Default is Empty.
    plot_title (str, optional): Title of the plot. Default is an empty string.
    label_all_bands (bool, optional): Whether to label each band individually
    or just label the set (DFT, GW, BSE). Default is False (label set).
    output_file (str, optional): Path to save the output plot image. Default
    is "band_structure.png".
    ymin, ymax (float, optional): Y-axis limits. If None, limits are
    determined automatically based on the data. Default is None and None.
    """

    data_dft = None
    data_gw = None
    data_bse = None
    if file_dft:
        data_dft = read_data(file_dft)
    if file_gw:
        data_gw = read_data(file_gw)
    if file_bse:
        data_bse = read_data(file_bse)

    plt.figure(figsize=(7, 6))

    linewidth = 0.5
    ax = plt.gca()
    if label_all_bands:
        if file_dft:
            data_dft.plot(x=data_dft.columns[0], y=data_dft.columns[1:-4],
                          label=labels(data_dft, 'DFT'), color='black',
                          linewidth=linewidth, ax=ax)
        if file_gw:
            data_gw.plot(x=data_gw.columns[0], y=data_gw.columns[1:-4],
                         label=labels(data_gw, 'GW'), color='red',
                         linewidth=linewidth, ax=ax)
        if file_bse:
            data_bse.plot(x=data_bse.columns[0], y=data_bse.columns[1:-3],
                          label=labels(data_bse, 'BSE', omit_last=3),
                          color='blue', linewidth=linewidth, ax=ax)
    else:
        if file_dft:
            data_dft.plot(x=data_dft.columns[0], y=data_dft.columns[1:-4],
                          legend=False, color='black',
                          linewidth=linewidth, ax=ax)
        if file_gw:
            data_gw.plot(x=data_gw.columns[0], y=data_gw.columns[1:-4],
                         legend=False, color='red',
                         linewidth=linewidth, ax=ax)
        if file_bse:
            data_bse.plot(x=data_bse.columns[0], y=data_bse.columns[1:-3],
                          legend=False, color='blue',
                          linewidth=linewidth, ax=ax)

    plt.title(plot_title)
    # Axis formatting
    plt.ylabel("Energy - $E_{VBM}$ (eV)")
    ax.set_xlabel("k-path")
    if file_bse:
        ax2 = ax.secondary_xaxis('top')
        ax2.set_xlabel("q-path")

    # Plot vertical lines at the symmetry points and add labels
    symmetry_labels = False
    if file_dft:
        if "symmetry_label" in data_dft.columns:
            symmetry_labels = True
            symmetry_points = \
                data_dft[data_dft["symmetry_label"].notna()].iloc[:, [0, -1]]
    elif file_gw:
        if "symmetry_label" in data_gw.columns:
            symmetry_labels = True
            symmetry_points = \
                data_gw[data_gw["symmetry_label"].notna()].iloc[:, [0, -1]]
    if symmetry_labels:
        for _, sp in symmetry_points.iloc[:].iterrows():
            plt.axvline(sp.iloc[0], color='gray', linestyle='--', alpha=0.5)
            plt.xticks(symmetry_points.iloc[:, 0],
                       labels=symmetry_points.iloc[:, 1])
            if file_bse:
                ax2.set_xticks(symmetry_points.iloc[:, 0],
                               labels=symmetry_points.iloc[:, 1])

    # axis limits
    basis_data = None
    if data_dft is not None:
        basis_data = data_dft
    elif data_gw is not None:
        basis_data = data_gw
    else:
        basis_data = data_bse
    xmin = basis_data.iloc[:, 0].min()
    xmax = basis_data.iloc[:, 0].max()
    if ymin is None:
        ymin = min(0., basis_data.iloc[:, 1:-4].min().min() - 0.5)
    if ymax is None:
        ymax = basis_data.iloc[:, 1:-4].max().max() + 0.5
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.axhline(0, color="black", linestyle=":", linewidth=1.5)
    if label_all_bands:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        legend_handles = []
        if file_dft:
            legend_handles.append(
                Line2D([0], [0], color='black', lw=1.5, label='DFT')
            )
        if file_gw:
            legend_handles.append(
                Line2D([0], [0], color='red', lw=1.5, label='GW')
            )
        if file_bse:
            legend_handles.append(
                Line2D([0], [0], color='blue', lw=1.5, label='BSE')
            )
        if legend_handles:
            plt.legend(handles=legend_handles, loc='center left',
                       bbox_to_anchor=(1, 0.5))

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # data files to plot
    file_dft = "data/o.bands_interpolated_dft"
    file_gw = "data/o.bands_interpolated_gw"
    file_bse = "data/o-BSE.excitons_interpolated"

    # plot title and output file name
    plot_title = "MoS2 5x5x2 k-grid"
    output_file = os.path.join("output", plot_title.replace(" ", "_") + ".png")
    label_all_bands = False  # True to label each band, False to label set

    plot_bands(file_dft, file_gw, file_bse, plot_title, output_file,
               label_all_bands, ymin=None, ymax=None)
