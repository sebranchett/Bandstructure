from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from dataclasses import dataclass


# Dictionary to map plain text to LaTeX labels for nice axis labels
latex_labels = {
    "GAMMA": r'$\Gamma$',
    "G": r'$\Gamma$',
    "A": r'$A$',
    "F": r'$F$',
    "H": r'$H$',
    "K": r'$K$',
    "L": r'$L$',
    "M": r'$M$',
    "P": r'$P$',
    "Q": r'$Q$',
    "R": r'$R$',
    "U": r'$U$',
    "W": r'$W$',
    "X": r'$X$',
    "Y": r'$Y$',
    "Z": r'$Z$',
    # Add more if needed
}


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
    title_line = None
    with open(filename, 'r') as f:
        for line in f:
            if "(a.u.)" in line:
                title_line = line
                break
    if title_line is None:
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


def _build_band_labels(data, prefix, omit_last=4):
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


@dataclass
class Dataset:
    """Loaded dataset with metadata used for plotting."""

    name: str
    data: pd.DataFrame
    color: str
    omit_last: int


@dataclass(frozen=True)
class PlotMode:
    """High-level plot mode flags derived from enabled datasets."""

    has_bse: bool
    only_bse: bool


def _build_datasets(file_dft, file_gw, file_bse):
    """
    Load enabled datasets and attach plotting metadata.

    Parameters:
    file_dft, file_gw, file_bse (str): Optional input file paths.

    Returns:
    list[Dataset]: Loaded datasets and plotting metadata.
    """
    dataset_specs = [
        ("DFT", file_dft, "black", 4),
        ("GW", file_gw, "red", 4),
        ("BSE", file_bse, "blue", 3),
    ]

    datasets = []
    for name, file_path, color, omit_last in dataset_specs:
        if file_path:
            datasets.append(
                Dataset(
                    name=name,
                    data=read_data(file_path),
                    color=color,
                    omit_last=omit_last,
                )
            )

    if not datasets:
        raise ValueError(
            "At least one data file must be provided for plotting."
        )
    return datasets


def _plot_dataset_lines(ax, datasets, label_all_bands, linewidth):
    """
    Plot all selected datasets on the provided axes.

    Parameters:
    ax (matplotlib.axes.Axes): Axis used for plotting.
    datasets (list[Dataset]): Dataset descriptors from `_build_datasets`.
    label_all_bands (bool): Whether to label every plotted band.
    linewidth (float): Line width used for all curves.

    Returns:
    None
    """
    for dataset in datasets:
        data = dataset.data
        omit_last = dataset.omit_last
        plot_kwargs = {
            "x": data.columns[0],
            "y": data.columns[1:-omit_last],
            "color": dataset.color,
            "linewidth": linewidth,
            "ax": ax,
        }
        if label_all_bands:
            plot_kwargs["label"] = _build_band_labels(
                data, dataset.name, omit_last
            )
        else:
            plot_kwargs["legend"] = False
        data.plot(**plot_kwargs)


def _configure_x_axes(ax, mode):
    """
    Configure bottom and optional top x-axes.

    Parameters:
    ax (matplotlib.axes.Axes): Primary axis for the plot.
    mode (PlotMode): Plot mode flags from `_resolve_plot_mode`.

    Returns:
    matplotlib.axes._secondary_axes.SecondaryAxis | None:
    The top axis when needed (BSE overlay case), otherwise None.
    """
    bse_label = "Exciton Momentum q (a.u.)"
    if mode.only_bse:
        ax.set_xlabel(bse_label)
        return None

    ax.set_xlabel("k-path")
    if mode.has_bse:
        ax2 = ax.secondary_xaxis('top')
        ax2.set_xlabel(bse_label)
        return ax2
    return None


def _get_symmetry_source(datasets, symmetry_override=None):
    """
    Select the symmetry-label source dataset.

    Parameters:
    datasets (list[Dataset]): Dataset descriptors from `_build_datasets`.
    symmetry_override (pandas.DataFrame | None): Pre-loaded DataFrame to use
    as the symmetry source instead of searching datasets. Default is None.

    Returns:
    pandas.DataFrame | None:
    Override first, else DFT, else GW, else None if unavailable.
    """
    if symmetry_override is not None:
        return symmetry_override
    for preferred_name in ["DFT", "GW"]:
        for dataset in datasets:
            if dataset.name == preferred_name and \
                    "symmetry_label" in dataset.data.columns:
                return dataset.data
    return None


def _plot_symmetry_guides(ax, symmetry_source,
                          apply_labels=True,
                          draw_guides=True):
    """
    Draw vertical symmetry guides and apply tick labels.

    Parameters:
    ax (matplotlib.axes.Axes): Primary axis where guides are drawn.
    symmetry_source (pandas.DataFrame | None): Source with symmetry labels.
    apply_labels (bool): Whether to apply symmetry labels to bottom x-axis.
    draw_guides (bool): Whether to draw vertical guide lines.

    Returns:
    None
    """
    if symmetry_source is None:
        return

    symmetry_points = (
        symmetry_source[symmetry_source["symmetry_label"].notna()]
        .iloc[:, [0, -1]]
    )
    x_points = symmetry_points.iloc[:, 0]
    symmetry_labels = (
        symmetry_points.iloc[:, 1]
        .astype(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.strip()
        .map(lambda label: latex_labels.get(label, label))
    )

    if draw_guides:
        for k_value in x_points:
            ax.axvline(k_value, color='gray', linestyle='--', alpha=0.5)

    if apply_labels:
        ax.set_xticks(x_points, labels=symmetry_labels)


def _set_axis_limits(ax, datasets, ymin=None, ymax=None):
    """
    Set plot limits using the first dataset.

    Parameters:
    datasets (list[Dataset]): Dataset descriptors from `_build_datasets`.
    ymin, ymax (float | None): Optional explicit y-limits.

    Returns:
    None
    """
    basis_data = datasets[0].data
    basis_omit_last = datasets[0].omit_last
    xmin = basis_data.iloc[:, 0].min()
    xmax = basis_data.iloc[:, 0].max()
    if ymin is None:
        ymin = min(
            0.,
            basis_data.iloc[:, 1:-basis_omit_last].min().min() - 0.5
        )
    if ymax is None:
        ymax = basis_data.iloc[:, 1:-basis_omit_last].max().max() + 0.5

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axhline(0, color="black", linestyle=":", linewidth=1.5)


def _add_legend(ax, datasets, label_all_bands):
    """
    Add either full-band legend or concise dataset legend.

    Parameters:
    datasets (list[Dataset]): Dataset descriptors from `_build_datasets`.
    label_all_bands (bool): Controls full-band vs compact legend mode.

    Returns:
    None
    """
    if label_all_bands:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return

    legend_handles = [
        Line2D([0], [0], color=dataset.color, lw=1.5,
               label=dataset.name)
        for dataset in datasets
    ]
    ax.legend(handles=legend_handles, loc='center left',
              bbox_to_anchor=(1, 0.5))


def plot_bands(file_dft="", file_gw="", file_bse="", plot_title="",
               output_file="band_structure.png", label_all_bands=False,
               ymin=None, ymax=None, symmetry_from_file=""):
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
    symmetry_from_file (str, optional): Path to a file from which symmetry
    point positions and labels are read, without plotting its band data.
    Useful when plotting BSE-only data. Default is Empty.

    Returns:
    None
    """

    datasets = _build_datasets(file_dft, file_gw, file_bse)

    plt.figure(figsize=(7, 6))

    linewidth = 0.5
    ax = plt.gca()
    _plot_dataset_lines(ax, datasets, label_all_bands, linewidth)

    plt.title(plot_title)
    # Axis formatting
    plt.ylabel("Energy - $E_{VBM}$ (eV)")
    has_bse = any(dataset.name == "BSE" for dataset in datasets)
    mode = PlotMode(has_bse=has_bse, only_bse=(has_bse and len(datasets) == 1))
    _configure_x_axes(ax, mode)

    symmetry_override = (
        read_data(symmetry_from_file) if symmetry_from_file else None
    )
    symmetry_source = _get_symmetry_source(datasets, symmetry_override)
    _plot_symmetry_guides(
        ax,
        symmetry_source,
        apply_labels=not mode.only_bse,
        draw_guides=not mode.only_bse,
    )

    _set_axis_limits(ax, datasets, ymin, ymax)
    _add_legend(ax, datasets, label_all_bands)

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # data files to plot
    file_dft = "data/o.bands_interpolated_dft"
    file_gw = "data/o.bands_interpolated_gw"
    file_bse = "data/o-BSE.excitons_interpolated"
    label_all_bands = False  # True to label each band, False to label set

    # plot title and output file name
    plot_title = "MoS2 5x5x2 k-grid"
    output_file = os.path.join("output", plot_title.replace(" ", "_") + ".png")

    plot_bands(file_dft, file_gw, file_bse, plot_title, output_file,
               label_all_bands, ymin=None, ymax=None)
