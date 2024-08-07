#! /usr/bin/env python3

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KernelDensity

BKGD_COLOR = 'darkgray'


if __name__ == '__main__':

    from s01_generate_data import (
        MultivariateNormalComponents, 
        ScaledData, 
        create_data_01_with_parameters, 
        create_data_02_with_parameters, 
        create_data_03_with_parameters, 
        create_data_04_with_parameters, 
        split_data_with_parameters,
        scale_data)

else:

    from src.s01_generate_data import (
        MultivariateNormalComponents, 
        ScaledData, 
        create_data_01_with_parameters, 
        create_data_02_with_parameters, 
        create_data_03_with_parameters, 
        create_data_04_with_parameters, 
        split_data_with_parameters,
        scale_data)


@dataclass
class DataAttr:
    path_stem: Path
    dir_name: str
    legend_label: str
    path: Path = Path('')

    def __post_init__(self):
        self.path = self.path_stem / self.dir_name


@dataclass
class XYDataPairs:

    pair_step_label: str

    y1: np.ndarray
    x1: np.ndarray

    y2: np.ndarray
    x2: np.ndarray

    def __post_init__(self):

        assert self.x1.shape == self.x2.shape
        assert self.y1.shape == self.y2.shape

        assert self.x1.ndim == 1
        assert self.x2.ndim == 1

        assert self.x1.shape[0] == self.y1.shape[0]
        assert self.x2.shape[0] == self.y2.shape[0]

        assert np.allclose(self.x1, self.x2, atol=1e-6)


@dataclass
class XYBins:

    x: np.ndarray
    y: np.ndarray
    bin: np.ndarray


def load_x_y_coords_for_data_pairs(
    input_path_1: Path, input_path_2: Path,
    input_x_npy_filename: str='line_xs.npy', 
    input_y_npy_filename: str='line_ys.npy') -> XYDataPairs:
    """
    Load 'x' and 'y' coordinates for two data sets, where 'x' and 'y' 
        coordinates from each data set are stored in separate Numpy files in 
        the same directory
    """

    step_number_1 = input_path_1.stem.split('_')[0]
    step_number_2 = input_path_2.stem.split('_')[0]
    assert step_number_1[0] == 's'
    assert step_number_2[0] == 's'
    pair_step_label = step_number_1 + '_' + step_number_2

    y_filepath_1 = input_path_1 / input_y_npy_filename
    x_filepath_1 = input_path_1 / input_x_npy_filename
    y_filepath_2 = input_path_2 / input_y_npy_filename
    x_filepath_2 = input_path_2 / input_x_npy_filename

    filepaths = [y_filepath_1, x_filepath_1, y_filepath_2]
    data = [np.loadtxt(filepath, delimiter=',') for filepath in filepaths]

    # some directories don't have an 'x' file, which should be same for all data 
    #   sets
    try:
        data_x2 = np.loadtxt(x_filepath_2, delimiter=',')
    except:
        print(f'No file of x-coordinates found for {x_filepath_2}')
        data_x2 = data[1]

    data.append(data_x2)

    x_y_data_pairs = XYDataPairs(
        pair_step_label, data[0], data[1], data[2], data[3])

    return x_y_data_pairs


def get_line_xs(input_path_stem: Path, data_str: str):
    """
    This function exists mainly to clarify and streamline the code that calls 
        it, even though it introduces some redundancy in that code
    Specifically, parts of the calling code analyze comparisons between data 
        sets and other parts rely on only a single data set; the latter needs
        only the 'line_xs' variable, which is already being called by code that
        pairs two data sets and benefits from checking that the two data sets'
        versions of 'line_xs' match
    However, to cleanly separate single-data-set code from the two-data-sets 
        code, this function wraps the two-data-sets calls so that the returned
        variables cannot be used; ergo, the two-data-sets code must call these
        functions again and not depend on the code in this funtion, which 
        precedes the one-data-set code
    Yes, the rationale is a bit odd, but it beats the alternatives for now; 
        the better solution will probably be to formally separate the 
        one-data-set and two-data-set code into separate functions, but it's
        too soon to do that yet
    """

    data_attr_1 = DataAttr(
        input_path_stem, 's03_bayes_stan' + data_str, 'Bayesian Linear')
    data_attr_2 = DataAttr(input_path_stem, 's04_quantile' + data_str, 'Linear')

    x_y_data_pairs = load_x_y_coords_for_data_pairs(
        data_attr_1.path, data_attr_2.path)
    line_xs = x_y_data_pairs.x1

    return line_xs


def bin_y_by_x(
    xs: np.ndarray, ys: np.ndarray, x_bin_centers: np.ndarray) -> XYBins:
    """
    Bin 'xs' where each 'line_xs' specifies the center of each bin, then group
        the bin indices with 'xs' and 'ys'
    """

    # 'xs' and 'ys' should be the same size and essentially one-dimensional
    assert xs.shape[0] == ys.shape[0]

    # 'xs' and 'ys' can have ndim > 1, but each of those "extra" dimensions 
    #   should be only '1' so that both arrays are one-dimensional
    assert (xs.shape[0] + xs.ndim - 1) == np.sum(xs.shape)
    assert (ys.shape[0] + ys.ndim - 1) == np.sum(ys.shape)

    # 'line_xs' should encompass all 'xs' values
    assert (xs >= x_bin_centers.min()).all()
    assert (xs <= x_bin_centers.max()).all()

    x_bin_cuts = get_bin_cuts_for_regular_1d_grid(x_bin_centers)
    x_bin_idx = np.digitize(xs, bins=x_bin_cuts)

    x_y_bins = XYBins(xs.reshape(-1), ys, x_bin_idx.reshape(-1))

    return x_y_bins


def plot_lines_comparison(
    line_xs: np.ndarray=np.array([]), 
    line_ys_1: np.ndarray=np.array([]), 
    line_ys_2: np.ndarray=np.array([]), 
    x_label: str='', y_label: str='', 
    line_label_1: str='', line_label_2: str='', 
    title: str='', output_filepath: Path=Path('plot.png')):
    """
    Plot two sets of lines with same x-coordinates and different y-coordinates
    """

    assert len(line_xs) == line_ys_1.shape[0]
    assert len(line_xs) == line_ys_2.shape[0]

    # assume that 'line_ys_1' and 'line_ys_2' have the same number of lines to
    #   plot
    assert line_ys_1.shape[1] == line_ys_2.shape[1]

    ax = plt.axes()
    ax.set_facecolor(BKGD_COLOR)

    for i in range(line_ys_1.shape[1]):
        # add 'label' for legend only once
        if i == 0:
            plt.plot(
                line_xs, line_ys_1[:, i], 
                color='blue', linestyle='solid', label=line_label_1,
                zorder=8)
        else:
            plt.plot(
                line_xs, line_ys_1[:, i], 
                color='blue', linestyle='solid', zorder=8)

    for i in range(line_ys_2.shape[1]):
        # add 'label' for legend only once
        if i == 0:
            plt.plot(
                line_xs, line_ys_2[:, i], 
                color='yellow', linestyle='dotted', label=line_label_2,
                zorder=9)
        else:
            plt.plot(
                line_xs, line_ys_2[:, i], 
                color='yellow', linestyle='dotted', zorder=9)

    plt.axhline(y=0, color='black', linestyle='solid', linewidth=0.5, zorder=1)
    plt.axvline(x=0, color='black', linestyle='solid', linewidth=0.5, zorder=1)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='upper left', fontsize='small')

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def get_bin_cuts_for_regular_1d_grid(grid_1d: np.ndarray) -> np.ndarray:
    """
    For a 1-D grid with regular intervals, return the values that are half-way 
        between each successive pair of grid values, plus a value that is a
        half-interval lower than the lowest value and another value that is a
        half-interval higher than the highest value
    """

    # if array is not sorted, sort it
    if not np.all(grid_1d[:-1] <= grid_1d[1:]):
        grid_1d = np.sort(grid_1d)

    x_diffs = (grid_1d[1:] - grid_1d[:-1])

    # verify that x-grid intervals are evenly spaced
    assert np.allclose(x_diffs, x_diffs[0], atol=1e-6)

    half_interval = x_diffs[0] / 2
    bin_cuts = grid_1d - half_interval
    bin_cuts = np.append(bin_cuts, grid_1d[-1] + half_interval)

    return bin_cuts


def plot_distributions_with_quantiles(
    arrs: list[np.ndarray], title: str, full_width: bool, cumulative: bool,
    output_filepath: Path):
    """
    Plot distributions of values in each array in 'arrs' along with their 
        quantiles as vertical lines
    """

    fig, ax = plt.subplots(1, sharex=False)

    quantiles = [i/10 for i in range(1, 10, 2)]
    quantile_sets = [np.quantile(arr, quantiles) for arr in arrs]

    all_values = np.concatenate(arrs)
    ends = np.quantile(all_values, [0.02, 0.98])

    palette = {
        0: 'blue', 1: 'green', 2: 'orange', 3: 'red', 4: 'purple'}

    _ = sns.kdeplot(
        ax=ax, data=arrs, alpha=0.5, cumulative=cumulative, palette=palette,
        common_norm=True, common_grid=False)

    for i, v_set in enumerate(quantile_sets):
        if i < 2:
            linestyle = 'solid'
        else:
            linestyle = 'dotted'
        for line in v_set:
            _ = ax.axvline(
                line, color=palette[i], linestyle=linestyle, zorder=8)

    # zoom in on x-axis
    if not full_width:
        ax.set_xlim(ends)

    plt.title(title)
    plt.tight_layout()

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_bin_idx(
    x: np.ndarray, y: np.ndarray, bin_idx: np.ndarray, output_filepath: Path):
    """
    Plot a scatter plot of 'x' and 'y' values, colored by bin indices 'bin_idx'
    """

    plt.scatter(x, y, alpha=0.1, c=bin_idx)
    plt.title('Scatter plot colored by bin indices')

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def select_subarray_by_index(
    x_y_bins: XYBins, total_idx_n: int, select_idx_n: int,
    include_end_bins: bool=True) -> XYBins:
    """
    Given a 2-D Numpy array 'arr' with a column of indices, select rows that 
        include a subset of the indices
    The subset of indices is selected at approximately evenly spaced intervals
        from the first index (assumed to be zero) to the last index 
        (total_idx_n - 1)
    UPDATE:  converted function to accept and return XYBins instead of Numpy 
        arrays, but otherwise, rest of documentation is the same

    'arr' - 2-D Numpy array 'arr' with a column of indices
    'arr_col_idx' - index of the column in 'arr' with the indices
    'total_idx_n' - total number of indices; needed as a parameter because
        arr[:, arr_col_idx] might not include all indices
    'select_idx_n' - number of indices to include in the subset

    Example:

        arr = array([
               [9, 0],
               [8, 1],
               [7, 2],
               [6, 3],
               [5, 4]])
        arr_col_idx = 1
        total_idx_n = 5
        select_idx_n = 3

        slice_idx = array([0, 2, 4])
        selected_arr = array([[9, 0],
                              [7, 2],
                              [5, 4]])
    """

    # convert XYBins to Numpy array to accommodate code as originally written
    arr = np.concatenate(
        (x_y_bins.x.reshape(-1, 1), 
         x_y_bins.y.reshape(-1, 1), 
         x_y_bins.bin.reshape(-1, 1)), axis=1)
    arr_col_idx = 2

    # verify that the subset of indices is no greater than the total number of 
    #   indices
    assert select_idx_n <= total_idx_n

    slice_idx = (
        np.round(np.linspace(0, total_idx_n-1, select_idx_n)).astype(int))
    if not include_end_bins:
        slice_idx = slice_idx[1:-1] 
    slice_mask = np.isin(arr[:, arr_col_idx], slice_idx)
    selected_arr = arr[slice_mask, :]

    # convert Numpy array to XYBins
    x_y_bins_selected = XYBins(
        selected_arr[:, 0], selected_arr[:, 1], selected_arr[:, 2])

    return x_y_bins_selected 


def plot_density_by_bin(
    x_y_bin_slices: XYBins,
    vertical_line_x_coord_1: np.ndarray, vertical_line_x_coord_2: np.ndarray,
    output_filepath: Path):
    """
    Plot density plots for each bin of 'y' values, with two sets of vertical 
        lines specified by 'vertical_line_x_coord_1' and 
        'vertical_line_x_coord_2'

    'x_y_bin_slices' - 2-D Numpy array with 'y' values in the first column
        and bin indices in the second column
    'bin_col_idx' - index of the column in 'x_y_bin_slices' with the bin 
        indices
    """

    bin_idx = np.unique(x_y_bin_slices.bin).astype(int)
    # reverse order to plot first bin at bottom and last bin at top
    bin_idx = bin_idx[::-1] 

    fig, ax = plt.subplots(len(bin_idx), sharex=False)

    for i, bin in enumerate(bin_idx):

        v_lines_1 = vertical_line_x_coord_1[bin, :]
        v_lines_2 = vertical_line_x_coord_2[bin, :]
        assert len(v_lines_1) == len(v_lines_2)

        bin_idx = np.where(x_y_bin_slices.bin == bin)
        temp_y = x_y_bin_slices.y[bin_idx]

        ax[i].set_facecolor(BKGD_COLOR)
        _ = sns.kdeplot(
            ax=ax[i], data=temp_y, fill=True, alpha=1.0, linewidth=1.5,
            bw_adjust=1)
        # _ = sns.histplot(
        #     ax=ax[i], data=temp_y, fill=True, alpha=0.5, linewidth=1.5)
        for v in range(len(v_lines_1)):
            _ = ax[i].axvline(
                v_lines_1[v], color='blue', linestyle='solid', zorder=8)
            _ = ax[i].axvline(
                v_lines_2[v], color='yellow', linestyle='dotted', zorder=9)
        _ = ax[i].xaxis.set_ticklabels([])
        _ = ax[i].yaxis.set_ticklabels([])
        _ = ax[i].set_ylabel(f'Bin {int(bin)}')

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def process_data_set_pairs(
    data_attr_1: DataAttr, data_attr_2: DataAttr, 
    data_str: str, scaled_data: ScaledData, output_path: Path, 
    line_xs: np.ndarray):
    """
    Analyze and plot data that compare two data sets
    """

    x_y_data_pairs = load_x_y_coords_for_data_pairs(
        data_attr_1.path, data_attr_2.path)

    
    output_filename = (
        x_y_data_pairs.pair_step_label + '_quantiles' + data_str + '.png')
    output_filepath = output_path / output_filename
    plot_lines_comparison(
        x_y_data_pairs.x1, x_y_data_pairs.y1, x_y_data_pairs.y2, 
        line_label_1=data_attr_1.legend_label, 
        line_label_2=data_attr_2.legend_label,
        output_filepath=output_filepath)


    # ys = scaled_data.test_y
    # xs = scaled_data.test_x
    ys = np.concatenate(
        [scaled_data.train_y, scaled_data.valid_y, scaled_data.test_y])
    xs = np.concatenate(
        [scaled_data.train_x, scaled_data.valid_x, scaled_data.test_x])
    x_y_bins = bin_y_by_x(xs, ys, line_xs)
    x_slice_n = 7
    x_n = line_xs.shape[0]
    x_y_bin_slices = select_subarray_by_index(
        x_y_bins, x_n, x_slice_n, False)
    output_filename = (
        x_y_data_pairs.pair_step_label + '_density_by_bin' + data_str + '.png')
    output_filepath = output_path / output_filename
    plot_density_by_bin(
        x_y_bin_slices, x_y_data_pairs.y1, x_y_data_pairs.y2, output_filepath)

    # (np.abs(x_y_data_pairs.y1 - x_y_data_pairs.y2)).sum()
    # (np.abs(x_y_data_pairs.y1 - x_y_data_pairs.y2)).mean()
    # x_y_data_pairs.x1.shape
    # x_y_data_pairs.y1.shape
    # x_y_data_pairs.x2.shape
    # x_y_data_pairs.y2.shape


def process_data(
    input_path_stem: Path, data_str: str, 
    mvn_components: MultivariateNormalComponents, scaled_data: ScaledData, 
    output_path: Path):


    output_path.mkdir(exist_ok=True, parents=True)

    # input_dir_name = 's03_bayes_stan_data01'
    # filename = 'fit_df.parquet'
    # filepath = input_path_stem / input_dir_name / filename
    # s03_fit_df = pl.read_parquet(filepath)


    # for a "real" problem, it would make more sense to look only at the test
    #   to evaluate results (or, at least, look at train and test data 
    #   separately), but in this case, since all the data is artificially 
    #   generated from the same distribution, I'll just lump it all together for
    #   convenience
    # ys = scaled_data.test_y
    # xs = scaled_data.test_x
    ys = np.concatenate(
        [scaled_data.train_y, scaled_data.valid_y, scaled_data.test_y])
    xs = np.concatenate(
        [scaled_data.train_x, scaled_data.valid_x, scaled_data.test_x])

    y_list = [ys, scaled_data.test_y]
    output_filename = 'y_distributions' + data_str + '.png'
    output_filepath = output_path / output_filename
    title = 'Distributions of all y-values and only test y-values'
    plot_distributions_with_quantiles(
        y_list, title, True, False, output_filepath)


    line_xs = get_line_xs(input_path_stem, data_str)
    x_y_bins = bin_y_by_x(xs, ys, line_xs)

    output_filename = 'x_y_with_bin_color' + data_str + '.png'
    output_filepath = output_path / output_filename
    plot_bin_idx(x_y_bins.x, x_y_bins.y, x_y_bins.bin, output_filepath)

    bin_idx = np.where(x_y_bins.bin == 20)
    bin20 = x_y_bins.y[bin_idx]
    bin_idx = np.where(x_y_bins.bin == 50)
    bin50 = x_y_bins.y[bin_idx]
    bin_idx = np.where(x_y_bins.bin == 80)
    bin80 = x_y_bins.y[bin_idx]

    y_list = [x_y_bins.y, bin20, bin50, bin80]
    output_filename = 'y_distributions_bins' + data_str + '.png'
    output_filepath = output_path / output_filename
    title = 'Distributions of all y-values and only bins 20, 50, 80'
    plot_distributions_with_quantiles(
        y_list, title, False, False, output_filepath)


    ##################################################

    data_attr_1 = DataAttr(
        input_path_stem, 's03_bayes_stan' + data_str, 'Bayesian Linear')
    data_attr_2 = DataAttr(input_path_stem, 's04_quantile' + data_str, 'Linear')
    process_data_set_pairs(
        data_attr_1, data_attr_2, data_str, scaled_data, output_path, line_xs)
    
    
    data_attr_1 = DataAttr(
        input_path_stem, 's03_bayes_stan' + data_str, 'Bayesian Linear')
    data_attr_2 = DataAttr(
        input_path_stem, 's06_multitask_nn' + data_str, 'Neural Network')
    process_data_set_pairs(
        data_attr_1, data_attr_2, data_str, scaled_data, output_path, line_xs)



    # for j in range(0, 100, 10):
    #     np.quantile(x_y_data_pairs.y1[j, :], [0.1, 0.3, 0.5, 0.7, 0.9])


    # filename_stem = 's06_multitask_nn_data02'
    # output_path = output_path_stem / (filename_stem + '_' + str(i))
    # mvn_components = create_data_02_with_parameters()
    # data = split_data_with_parameters(mvn_components.cases_data)
    # scaled_data = scale_data(
    #     data.train, data.valid, data.test, 
    #     mvn_components.predictors_column_idxs, 
    #     mvn_components.response_column_idx)
    # process_data(mvn_components, scaled_data, output_path)
    # compile_results_across_runs(output_path_stem, filename_stem)


def main():

    input_path_stem = Path.cwd() / 'output'
    output_path = Path.cwd() / 'output' / 's10_results'

    data_sets = [
        ('_data01', create_data_01_with_parameters),
        ('_data02', create_data_02_with_parameters),
        ('_data03', create_data_03_with_parameters),
        ('_data04', create_data_04_with_parameters)]

    for i in range(len(data_sets)):
        # i = 3
        data_str = data_sets[i][0]
        mvn_components = data_sets[i][1]()
        data = split_data_with_parameters(mvn_components.cases_data)
        scaled_data = scale_data(
            data.train, data.valid, data.test, 
            mvn_components.predictors_column_idxs, 
            mvn_components.response_column_idx)
        process_data(
            input_path_stem, data_str, mvn_components, scaled_data, output_path)


if __name__ == '__main__':
    main()
