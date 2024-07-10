#! /usr/bin/env python3

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KernelDensity



if __name__ == '__main__':

    from s01_generate_data import (
        MultivariateNormalComponents, 
        ScaledData, 
        create_data_01_with_parameters, 
        create_data_02_with_parameters, 
        split_data_with_parameters,
        scale_data)

    from common import (
        print_loop_status_with_elapsed_time,
        plot_scatter_regression_with_parameters,
        extract_data_from_dataloader_batches,
        log_loss_to_tensorboard,
        calculate_quantile_loss,
        extract_data_df_columns,
        bin_y_values_by_x_bins,
        plot_distribution_by_bin,
        compile_results_across_runs,
        evaluate_bin_uniformity)
else:

    from src.s01_generate_data import (
        MultivariateNormalComponents, 
        ScaledData, 
        create_data_01_with_parameters, 
        create_data_02_with_parameters, 
        split_data_with_parameters,
        scale_data)

    from src.common import (
        print_loop_status_with_elapsed_time,
        plot_scatter_regression_with_parameters,
        extract_data_from_dataloader_batches,
        log_loss_to_tensorboard,
        calculate_quantile_loss,
        extract_data_df_columns,
        bin_y_values_by_x_bins,
        plot_distribution_by_bin,
        compile_results_across_runs,
        evaluate_bin_uniformity)


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


def load_x_y_coords_for_data_pairs(
    input_path_1: Path, input_path_2: Path,
    input_x_npy_filename: str='line_xs.npy', 
    input_y_npy_filename: str='line_ys.npy') -> XYDataPairs:
    """
    Load 'x' and 'y' coordinates for two data sets, where 'x' and 'y' 
        coordinates from each data set are stored in separate Numpy files in 
        the same directory
    """

    y_filepath_1 = input_path_1 / input_y_npy_filename
    x_filepath_1 = input_path_1 / input_x_npy_filename
    y_filepath_2 = input_path_2 / input_y_npy_filename
    x_filepath_2 = input_path_2 / input_x_npy_filename

    filepaths = [y_filepath_1, x_filepath_1, y_filepath_2]
    data = [np.loadtxt(filepath, delimiter=',') for filepath in filepaths]

    # some directories don't have an 'x' file, which should be same for all
    try:
        data_x2 = np.loadtxt(x_filepath_2, delimiter=',')
    except:
        print(f'No file of x-coordinates found for {x_filepath_2}')
        data_x2 = data[1]

    data.append(data_x2)

    x_y_data_pairs = XYDataPairs(data[0], data[1], data[2], data[3])

    return x_y_data_pairs


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
    ax.set_facecolor('gray')

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


def select_subarray_by_index(
    arr: np.ndarray, arr_col_idx: int,
    total_idx_n: int, select_idx_n: int,
    include_end_bins: bool=True) -> np.ndarray:
    """
    Given a 2-D Numpy array 'arr' with a column of indices, select rows that 
        include a subset of the indices
    The subset of indices is selected at approximately evenly spaced intervals
        from the first index (assumed to be zero) to the last index 
        (total_idx_n - 1)

    'arr' - 2-D Numpy array 'arr' with a column of indices
    'arr_col_idx' - index of the column in 'arr' with the indices
    'total_idx_n' - total number of indices; needed as a parameter because
        arr[:, arr_col_idx] may not include all indices
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

    # verify that the subset of indices is no greater than the total number of 
    #   indices
    assert select_idx_n <= total_idx_n

    slice_idx = (
        np.round(np.linspace(0, total_idx_n-1, select_idx_n)).astype(int))
    if not include_end_bins:
        slice_idx = slice_idx[1:-1] 
    slice_mask = np.isin(arr[:, arr_col_idx], slice_idx)
    selected_arr = arr[slice_mask, :]

    return selected_arr


def plot_density_by_bin(
    y_and_bins_slices: np.ndarray, 
    vertical_line_x_coord_1: np.ndarray, vertical_line_x_coord_2: np.ndarray,
    output_filepath: Path):
    """
    Plot density plots for each bin of 'y' values, with two sets of vertical 
        lines specified by 'vertical_line_x_coord_1' and 
        'vertical_line_x_coord_2'

    'y_and_bins_slices' - 2-D Numpy array with 'y' values in the first column
        and bin indices in the second column
    """


    bin_idx = np.unique(y_and_bins_slices[:, 1]).astype(int)
    # reverse order to plot first bin at bottom and last bin at top
    bin_idx = bin_idx[::-1] 

    fig, ax = plt.subplots(len(bin_idx), sharex=False)

    for i, bin in enumerate(bin_idx):

        v_lines_1 = vertical_line_x_coord_1[bin, :]
        v_lines_2 = vertical_line_x_coord_2[bin, :]
        assert len(v_lines_1) == len(v_lines_2)

        temp_y = y_and_bins_slices[y_and_bins_slices[:, 1] == bin, 0]

        ax[i].set_facecolor('gray')
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


def process_data(
    input_path_stem: Path, data_str: str, 
    mvn_components: MultivariateNormalComponents, scaled_data: ScaledData, 
    output_path: Path):


    output_path.mkdir(exist_ok=True, parents=True)

    # input_dir_name = 's03_bayes_stan_data01'
    # filename = 'fit_df.parquet'
    # filepath = input_path_stem / input_dir_name / filename
    # s03_fit_df = pl.read_parquet(filepath)


    ##################################################

    data_attr_1 = DataAttr(
        input_path_stem, 's03_bayes_stan' + data_str, 'Bayesian Linear')
    data_attr_2 = DataAttr(input_path_stem, 's04_quantile' + data_str, 'Linear')
    x_y_data_pairs = load_x_y_coords_for_data_pairs(
        data_attr_1.path, data_attr_2.path)

    output_filename = 's03_s04_quantiles' + data_str + '.png'
    output_filepath = output_path / output_filename
    plot_lines_comparison(
        x_y_data_pairs.x1, x_y_data_pairs.y1, x_y_data_pairs.y2, 
        line_label_1=data_attr_1.legend_label, 
        line_label_2=data_attr_2.legend_label,
        output_filepath=output_filepath)


    line_xs = x_y_data_pairs.x1
    ys = scaled_data.test_y
    x_bin_cuts = get_bin_cuts_for_regular_1d_grid(line_xs)
    y_bin_idx = np.digitize(ys, bins=x_bin_cuts)
    y_and_bins = np.concatenate(
        (ys.reshape(-1, 1), y_bin_idx.reshape(-1, 1)), axis=1)

    x_slice_n = 7
    x_n = line_xs.shape[0]
    y_and_bins_slices = select_subarray_by_index(
        y_and_bins, 1, x_n, x_slice_n, False)

    output_filename = 's03_s04_density_by_bin' + data_str + '.png'
    output_filepath = output_path / output_filename
    plot_density_by_bin(
        y_and_bins_slices, x_y_data_pairs.y1, x_y_data_pairs.y2, 
        output_filepath)


    ##################################################

    data_attr_1 = DataAttr(
        input_path_stem, 's03_bayes_stan' + data_str, 'Bayesian Linear')
    data_attr_2 = DataAttr(
        input_path_stem, 's06_multitask_nn' + data_str, 'Neural Network')
    x_y_data_pairs = load_x_y_coords_for_data_pairs(
        data_attr_1.path, data_attr_2.path)

    output_filename = 's03_s06_quantiles' + data_str + '.png'
    output_filepath = output_path / output_filename
    plot_lines_comparison(
        x_y_data_pairs.x1, x_y_data_pairs.y1, x_y_data_pairs.y2, 
        line_label_1=data_attr_1.legend_label, 
        line_label_2=data_attr_2.legend_label,
        output_filepath=output_filepath)


    line_xs = x_y_data_pairs.x1
    ys = scaled_data.test_y
    x_bin_cuts = get_bin_cuts_for_regular_1d_grid(line_xs)
    y_bin_idx = np.digitize(ys, bins=x_bin_cuts)
    y_and_bins = np.concatenate(
        (ys.reshape(-1, 1), y_bin_idx.reshape(-1, 1)), axis=1)

    x_slice_n = 7
    x_n = line_xs.shape[0]
    y_and_bins_slices = select_subarray_by_index(
        y_and_bins, 1, x_n, x_slice_n, False)

    output_filename = 's03_s06_density_by_bin' + data_str + '.png'
    output_filepath = output_path / output_filename
    plot_density_by_bin(
        y_and_bins_slices, x_y_data_pairs.y1, x_y_data_pairs.y2, 
        output_filepath)



    (np.abs(x_y_data_pairs.y1 - x_y_data_pairs.y2)).sum()
    (np.abs(x_y_data_pairs.y1 - x_y_data_pairs.y2)).mean()



    x_y_data_pairs.x1.shape
    x_y_data_pairs.y1.shape
    x_y_data_pairs.x2.shape
    x_y_data_pairs.y2.shape

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

    input_path_stem = Path.cwd() / 'output_BIG'
    output_path = Path.cwd() / 'output' / 's10_results'

    data_str = '_data01'
    mvn_components = create_data_01_with_parameters()
    data = split_data_with_parameters(mvn_components.cases_data)
    scaled_data = scale_data(
        data.train, data.valid, data.test, 
        mvn_components.predictors_column_idxs, 
        mvn_components.response_column_idx)
    process_data(
        input_path_stem, data_str, mvn_components, scaled_data, output_path)


    data_str = '_data02'
    mvn_components = create_data_02_with_parameters()
    data = split_data_with_parameters(mvn_components.cases_data)
    scaled_data = scale_data(
        data.train, data.valid, data.test, 
        mvn_components.predictors_column_idxs, 
        mvn_components.response_column_idx)
    process_data(
        input_path_stem, data_str, mvn_components, scaled_data, output_path)



if __name__ == '__main__':
    main()
