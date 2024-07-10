#! /usr/bin/env python3

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass

import seaborn as sns


if __name__ == '__main__':

    from s01_generate_data import (
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
    total_idx_n: int, select_idx_n: int) -> np.ndarray:
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
    """

    # verify that the subset of indices is no greater than the total number of 
    #   indices
    assert select_idx_n <= total_idx_n

    x_slice_idx = (
        np.round(np.linspace(0, total_idx_n-1, select_idx_n)).astype(int))
    slice_mask = np.isin(arr[:, arr_col_idx], x_slice_idx)
    selected_arr = arr[slice_mask, :]

    return selected_arr


def main():

    output_path = Path.cwd() / 'output' / 's10_results'
    output_path.mkdir(exist_ok=True, parents=True)

    input_path_stem = Path.cwd() / 'output_BIG'

    input_dir_name = 's03_bayes_stan_data01'
    filename = 'fit_df.parquet'
    filepath = input_path_stem / input_dir_name / filename
    s03_fit_df = pl.read_parquet(filepath)


    ##################################################

    data_str = '_data01'
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



    ##################################################

    data_str = '_data01'
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


    data_str = '_data02'
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


    data_str = '_data02'
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



    (np.abs(x_y_data_pairs.y1 - x_y_data_pairs.y2)).sum()
    (np.abs(x_y_data_pairs.y1 - x_y_data_pairs.y2)).mean()



    # set up x bins for density plots
    # x = scaled_data.test_x
    # x_bin_n = x_y_data_pairs.x1.shape[0]
    # line_xs = np.linspace(x.min(), x.max(), x_bin_n)
    # x_bin_idxs = np.digitize(x, bins=line_xs)

    # y = x_y_data_pairs.y1
    # y.shape
    # x_slice_n = 7
    # # assumes y.shape[0] == x.shape[0], which was enforced in XYDataPairs
    # x_n = x_y_data_pairs.y1.shape[0]
    # assert x_slice_n < x_n
    # x_slice_idx = np.round(np.linspace(0, x_n-1, x_slice_n)).astype(int)
    # x_slice_y = y[x_slice_idx, :]
    # x_slice_y.shape
    # flat_x = x_slice_y.flatten()
    # index_column = np.repeat(np.arange(x_slice_n)[:, np.newaxis], y, axis=1).flatten()[:, np.newaxis]
    # np.repeat(np.arange(x_slice_n), y)

    x_y_data_pairs.x1.shape
    x_y_data_pairs.y1.shape
    x_y_data_pairs.x2.shape
    x_y_data_pairs.y2.shape

    mvn_components = create_data_01_with_parameters()
    data = split_data_with_parameters(mvn_components.cases_data)
    scaled_data = scale_data(
        data.train, data.valid, data.test, 
        mvn_components.predictors_column_idxs, 
        mvn_components.response_column_idx)

    scaled_data.test_x.shape
    scaled_data.test_y.shape

    line_xs = x_y_data_pairs.x1
    ys = scaled_data.test_y
    x_bin_cuts = get_bin_cuts_for_regular_1d_grid(line_xs)
    y_bin_idx = np.digitize(ys, bins=x_bin_cuts)
    y_and_bins = np.concatenate(
        (ys.reshape(-1, 1), y_bin_idx.reshape(-1, 1)), axis=1)

    x_slice_n = 7
    x_n = line_xs.shape[0]
    y_and_bins_slices = select_subarray_by_index(y_and_bins, 1, x_n, x_slice_n)




    # get scatter data
    # plot distributions vs quantiles conditional on x
    rs = np.random.RandomState(1979)
    x = rs.randn(500)
    g = np.tile(list("ABCDEFGHIJ"), 50)
    df = pd.DataFrame(dict(x=x, g=g))
    m = df.g.map(ord)
    df["x"] += m

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

    g.map(sns.kdeplot, "x",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)


    g.map(label, "x")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    g.savefig('plot.png')

    plt.clf()
    plt.close()

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


if __name__ == '__main__':
    main()
