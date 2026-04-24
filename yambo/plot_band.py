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
    Default is 4.

    Returns:
    list: A list of labels with the specified prefix, excluding the last
    'omit_last' columns.
    """
    return (prefix + " - " + data.columns[1:-omit_last]).to_list()


# data files to plot
file_dft = "data/o.bands_interpolated_dft"
file_gw = "data/o.bands_interpolated_gw"
file_bse = "data/o-BSE.excitons_interpolated"

data_dft = read_data(file_dft)
data_gw = read_data(file_gw)
data_bse = read_data(file_bse)

# plot title and output file name
plot_title = "MoS2 5x5x2 k-grid"
output_file = os.path.join("output", plot_title.replace(" ", "_") + ".png")

plt.figure(figsize=(7, 6))

linewidth = 0.5
ax = plt.gca()
data_dft.plot(x=data_dft.columns[0], y=data_dft.columns[1:-4],
              label=labels(data_dft, 'DFT'), color='black',
              linewidth=linewidth, ax=ax)
data_gw.plot(x=data_gw.columns[0], y=data_gw.columns[1:-4],
             label=labels(data_gw, 'GW'), color='red', linewidth=linewidth,
             ax=ax)
data_bse.plot(x=data_bse.columns[0], y=data_bse.columns[1:-3],
              label=labels(data_bse, 'BSE', omit_last=3), color='blue',
              linewidth=linewidth, ax=ax)

plt.title(plot_title)
# Axis formatting
plt.ylabel("Energy - $E_{VBM}$ (eV)")

# Add labels to both top and bottom x-axes
ax2 = ax.secondary_xaxis('top')
ax.set_xlabel("k-path")
ax2.set_xlabel("q-path")

# find the symmetry point labels in the last column of data_dft and plot them
symmetry_points = data_dft[data_dft.iloc[:, -1].notna()].iloc[:, [0, -1]]

# Plot vertical lines at the symmetry points and add labels
for _, sp in symmetry_points.iloc[:].iterrows():
    plt.axvline(sp.iloc[0], color='gray', linestyle='--', alpha=0.5)
    plt.xticks(symmetry_points.iloc[:, 0], labels=symmetry_points.iloc[:, 1])
    ax2.set_xticks(symmetry_points.iloc[:, 0],
                   labels=symmetry_points.iloc[:, 1])

# axis limits
plt.xlim(0, data_dft.iloc[:, 0].max())
plt.ylim(max(-4, data_dft.iloc[:, 1:-4].min().min() - 0.5),
         max(4, data_dft.iloc[:, 1:-4].max().max() + 0.5))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(False)
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()
