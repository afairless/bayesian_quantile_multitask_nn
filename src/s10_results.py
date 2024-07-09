#! /usr/bin/env python3

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass


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


    # todo:  check 2nd data set
    # plot distributions vs quantiles conditional on x

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
