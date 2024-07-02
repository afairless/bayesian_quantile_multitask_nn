#! /usr/bin/env python3

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt


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


def plot_lines_comparison(
    line_xs: np.ndarray=np.array([]), 
    line_ys_1: np.ndarray=np.array([]), 
    line_ys_2: np.ndarray=np.array([]), 
    x_label: str='', y_label: str='', 
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
        plt.plot(
            line_xs, line_ys_1[:, i], 
            color='blue', linestyle='solid', zorder=8)

    for i in range(line_ys_2.shape[1]):
        plt.plot(
            line_xs, line_ys_2[:, i], 
            color='yellow', linestyle='dotted', zorder=9)

    plt.axhline(y=0, color='black', linestyle='solid', linewidth=0.5, zorder=1)
    plt.axvline(x=0, color='black', linestyle='solid', linewidth=0.5, zorder=1)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

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

    filename = 'line_xs.npy'
    filepath = input_path_stem / input_dir_name / filename
    s03_line_xs = np.loadtxt(filepath, delimiter=',')

    filename = 'line_ys.npy'
    filepath = input_path_stem / input_dir_name / filename
    s03_line_ys = np.loadtxt(filepath, delimiter=',')


    input_dir_name = 's04_quantile_data01'
    filename = 'line_xs.npy'
    filepath = input_path_stem / input_dir_name / filename
    s04_line_xs = np.loadtxt(filepath, delimiter=',')

    filename = 'line_ys.npy'
    filepath = input_path_stem / input_dir_name / filename
    s04_line_ys = np.loadtxt(filepath, delimiter=',')

    input_dir_name = 's06_multitask_nn_data01'
    filename = 'line_ys.npy'
    filepath = input_path_stem / input_dir_name / filename
    s06_line_ys = np.loadtxt(filepath, delimiter=',')

    assert (s03_line_xs == s04_line_xs).all()
    line_xs = s03_line_xs

    output_filename = 's03_s04_quantiles.png'
    output_filepath = output_path / output_filename
    plot_lines_comparison(
        line_xs, s03_line_ys, s04_line_ys, output_filepath=output_filepath)

    output_filename = 's03_s06_quantiles.png'
    output_filepath = output_path / output_filename
    plot_lines_comparison(
        line_xs, s03_line_ys, s06_line_ys, output_filepath=output_filepath)


    (np.abs(s03_line_ys - s04_line_ys)).sum()
    (np.abs(s03_line_ys - s04_line_ys)).mean()

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
