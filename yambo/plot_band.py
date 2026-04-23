import matplotlib.pyplot as plt
import pandas as pd
import os


def read_data(filename, col_title_row_nr=16, column_width=19):
    """
    Reads data from a file and assigns column names based on a specified row.

    Parameters:
    filename (str): The path to the file to be read.
    col_title_row_nr (int, optional): The row number where column titles are
    located. Default is 16.
    column_width (int, optional): The width of each column in the file.
    Default is 19.

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the file with
    appropriate column names.
    """
    data = pd.read_csv(filename, sep=r'\s+', comment='#', header=None)
    num_columns = data.shape[1]
    colnames = pd.read_fwf(filename, widths=[column_width]*num_columns,
                           skiprows=col_title_row_nr - 1, nrows=1)
    colnames.columns = colnames.columns.str.strip('# ')
    data.columns = colnames.columns
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

# describe these data files, will be used for title
plot_title = "MoS2 5x5x2 k-grid"

# make output file name
output_file = os.path.join("output", plot_title.replace(" ", "_") + ".png")

data_dft = read_data(file_dft)
data_gw = read_data(file_gw)
data_bse = read_data(file_bse)

linewidth = 0.5
data_dft.plot(x=data_dft.columns[0], y=data_dft.columns[1:-4],
              label=labels(data_dft, 'DFT'), color='black',
              linewidth=linewidth)
data_gw.plot(x=data_gw.columns[0], y=data_gw.columns[1:-4],
             label=labels(data_gw, 'GW'), color='red', linewidth=linewidth,
             ax=plt.gca())
data_bse.plot(x=data_bse.columns[0], y=data_bse.columns[1:-3],
              label=labels(data_bse, 'BSE', omit_last=3), color='blue',
              linewidth=linewidth, ax=plt.gca())

# adjust the y-axis limits to fit the data
# plt.gca().set_ylim(-5, 5)

# find the symmetry point labels in the last column of data_dft and plot them
symmetry_points = data_dft[data_dft.iloc[:, -1].notna()].iloc[:, [0, -1]]
y_placement = plt.gca().get_ylim()[0] - \
    0.13 * (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0])
for i in range(symmetry_points.shape[0]):
    plt.text(
        symmetry_points.iloc[i, 0], y_placement, symmetry_points.iloc[i, 1],
        verticalalignment='bottom', horizontalalignment='center'
    )

plt.xlabel(data_dft.columns[0])
plt.ylabel('Energy (eV)')
plt.title(plot_title)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()
