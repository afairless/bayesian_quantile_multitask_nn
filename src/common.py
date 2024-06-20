
import time
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable


##################################################
# FUNCTIONS FOR TEXT FILES
##################################################

def read_text_file(
    text_filename: str | Path, return_string: bool=False, 
    keep_newlines: bool=False):
    """
    Reads text file
    If 'return_string' is 'True', returns text in file as a single string
    If 'return_string' is 'False', returns list so that each line of text in
        file is a separate list item
    If 'keep_newlines' is 'True', newline markers '\n' are retained; otherwise,
        they are deleted

    :param text_filename: string specifying filepath or filename of text file
    :param return_string: Boolean indicating whether contents of text file
        should be returned as a single string or as a list of strings
    :return:
    """

    text_list = []

    try:
        with open(text_filename) as text:
            if return_string:
                # read entire text file as single string
                if keep_newlines:
                    text_list = text.read()
                else:
                    text_list = text.read().replace('\n', '')
            else:
                # read each line of text file as separate item in a list
                for line in text:
                    if keep_newlines:
                        text_list.append(line)
                    else:
                        text_list.append(line.rstrip('\n'))
            text.close()

        return text_list

    except:

        return ['There was an error when trying to read text file']


def write_list_to_text_file(
    a_list: list[str], text_filename: Path | str, overwrite: bool=False):
    """
    Writes a list of strings to a text file
    If 'overwrite' is 'True', any existing file by the name of 'text_filename'
        will be overwritten
    If 'overwrite' is 'False', list of strings will be appended to any existing
        file by the name of 'text_filename'

    :param a_list: a list of strings to be written to a text file
    :param text_filename: a string denoting the filepath or filename of text
        file
    :param overwrite: Boolean indicating whether to overwrite any existing text
        file or to append 'a_list' to that file's contents
    :return:
    """

    if overwrite:
        append_or_overwrite = 'w'
    else:
        append_or_overwrite = 'a'

    try:
        text_file = open(text_filename, append_or_overwrite, encoding='utf-8')
        for e in a_list:
            text_file.write(str(e))
            text_file.write('\n')

    finally:
        text_file.close()


##################################################
# FUNCTIONS FOR LOOP ITERATION COUNTING
##################################################


def seconds_to_formatted_time_string(seconds: float) -> str:
    """
    Given the number of seconds, returns a formatted string showing the time
        duration
    """

    hour = int(seconds / (60 * 60))
    minute = int((seconds % (60 * 60)) / 60)
    second = seconds % 60

    return '{}:{:>02}:{:>05.2f}'.format(hour, minute, second)


def print_loop_status_with_elapsed_time(
    the_iter: int, every_nth_iter: int, total_iter: int, start_time: float):
    """
    Prints message providing loop's progress for user

    :param the_iter: index that increments by 1 as loop progresses
    :param every_nth_iter: message should be printed every nth increment
    :param total_iter: total number of increments that loop will run
    :param start_time: starting time for the loop, which should be
        calculated by 'import time; start_time = time.time()'
    """

    current_time = time.ctime(int(time.time()))

    every_nth_iter_integer = max(round(every_nth_iter), 1)

    if the_iter % every_nth_iter_integer == 0:
        print('Processing loop iteration {i} of {t}, which is {p:0f}%, at {c}'
              .format(i=the_iter + 1,
                      t=total_iter,
                      p=(100 * (the_iter + 1) / total_iter),
                      c=current_time))
        elapsed_time = time.time() - start_time

        print('Elapsed time: {}'.format(seconds_to_formatted_time_string(
            elapsed_time)))


##################################################
# FUNCTIONS FOR CALCULATING LOSS
##################################################


def calculate_quantile_loss(
    quantile: float, true_values: torch.Tensor, 
    predicted_values: torch.Tensor) -> torch.Tensor:
    """
    Calculate quantile loss between a vector of true values and a vector of 
        predicted values
    """


    # input parameter pre-checks
    ##################################################

    assert quantile > 0
    assert quantile < 1

    assert true_values.ndim == 1 or true_values.shape[1] == 1 
    assert predicted_values.ndim == 1 or predicted_values.shape[1] == 1 
    assert true_values.shape == predicted_values.shape


    # calculate loss
    ##################################################

    errors = predicted_values - true_values 

    losses_1 = quantile * errors
    losses_2 = (quantile - 1) * errors
    losses = torch.max(losses_1, losses_2)

    loss = torch.mean(losses)

    return loss


##################################################
# FUNCTIONS FOR BINNING
##################################################


def extract_data_df_columns(
    data_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Establish data types for extracted Pandas DataFrame columns so linter
        doesn't complain
    """

    x_colname = 'x1'
    y_colname = 'y'
    x = data_df[x_colname].values
    assert isinstance(x, np.ndarray)
    y = data_df[y_colname].values
    assert isinstance(y, np.ndarray)

    return x, y


def enforce_bin_monotonicity(
    bin_cuts: np.ndarray, small_difference: float=1e-6) -> np.ndarray:
    """
    Bin cuts/borders theoretically cannot cross, so enforce monotonicity in case
        they do

    'small_difference' - a small amount by which to shift changed values to 
        ensure that float imprecision in evaluating "equal" values does not
        violate monotonicity
    """

    assert bin_cuts.ndim == 1

    # must have at least 4 numbers, i.e., 3 differences between numbers to have
    #   both a trend and a difference against that trend
    if len(bin_cuts) < 4:
        return bin_cuts

    bin_cuts = bin_cuts.copy()
    if np.issubdtype(bin_cuts.dtype, np.integer):
        bin_cuts = bin_cuts.astype(float)

    monotonically_increasing = bin_cuts[1:] >= bin_cuts[:-1]
    monotonically_decreasing = bin_cuts[1:] <= bin_cuts[:-1]
    mostly_increasing = monotonically_increasing.sum() > (len(bin_cuts) / 2)
    mostly_decreasing = monotonically_decreasing.sum() > (len(bin_cuts) / 2)

    if mostly_increasing and not np.all(monotonically_increasing):
        idx = np.where(~monotonically_increasing)[0] + 1
        for i in idx:
            bin_cuts[i] = bin_cuts[i-1] + small_difference

    if mostly_decreasing and not np.all(monotonically_decreasing):
        idx = np.where(~monotonically_decreasing)[0] + 1
        for i in idx:
            bin_cuts[i] = bin_cuts[i-1] - small_difference

    return bin_cuts


def bin_y_values_by_x_bins(
    x: np.ndarray, y: np.ndarray, x_bin_n: int, line_ys_func: Callable, **kwargs
    ) -> np.ndarray:
    """
    From x-y pairs, bin 'x' values into 'x_bin_n' number of bins; then for each
        'x' bin, bin 'y' values into bins with cut points produced by 
        'line_ys_func'
    """

    assert x.shape == y.shape
    assert x.ndim == y.ndim == 1

    line_xs = np.linspace(x.min(), x.max(), x_bin_n)
    x_bin_idxs = np.digitize(x, bins=line_xs)
    x_bin_idxs_y = np.column_stack((x_bin_idxs, y))
    # np.unique(x_bin_idxs_y[:, 0], return_counts=True)

    # is there an elegant way to vectorize this loop?
    x_binned_y_bin_idxs_compiled = np.array([], dtype=int)
    for i in range(x_bin_n):

        x_binned_ys = x_bin_idxs_y[x_bin_idxs_y[:, 0] == (i+1)][:, 1]

        line_ys = line_ys_func(line_xs=line_xs, **kwargs)
        assert line_xs.shape[0] == line_ys.shape[0] == x_bin_n

        # 'np.digitize' requires monotonicity
        bin_cuts = enforce_bin_monotonicity(line_ys[i, :])
        try:
            x_binned_y_bin_idxs = np.digitize(x_binned_ys, bins=bin_cuts)
        except:
            breakpoint()
        x_binned_y_bin_idxs_compiled = np.concatenate(
            (x_binned_y_bin_idxs_compiled, x_binned_y_bin_idxs))

    y_bin_counts = np.bincount(x_binned_y_bin_idxs_compiled)

    return y_bin_counts 


##################################################
# FUNCTIONS FOR SAVING RESULTS
##################################################

def plot_scatter_and_regression(
    x: pd.Series, y: pd.Series, 
    line_xs: np.ndarray=np.array([]), line_ys: np.ndarray=np.array([]), 
    x_label: str='', y_label: str='', 
    title: str='', alpha: float=0.8, 
    output_filepath: Path=Path('plot.png')):
    """
    Plot scatterplot of response variable 'y' over one predictor variable 'x'
        and any number of regression lines 'line_ys' plotted over 'line_xs'
    """

    plt.scatter(x, y, alpha=alpha, zorder=2)

    if line_xs.size > 0 and line_ys.size > 0:
        assert len(line_xs) == line_ys.shape[0]
        for i in range(line_ys.shape[1]):
            plt.plot(line_xs, line_ys[:, i], color='black', linestyle='dotted', zorder=9)

    plt.axhline(y=0, color='black', linestyle='solid', linewidth=0.5, zorder=1)
    plt.axvline(x=0, color='black', linestyle='solid', linewidth=0.5, zorder=1)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_scatter_regression_with_parameters(
    df: pd.DataFrame, x_colname: str, y_colname: str, line_xs_n: int, 
    scatter_n: int, scatter_n_seed: int, 
    line_ys_func: Callable, output_filepath: Path, **kwargs):
    """
    Plot scatterplot of response variable 'y' over one predictor variable 'x'
        and any number of regression lines 'line_ys' plotted over 'line_xs'

    'df' - DataFrame containing data
    'x_colname' - column name of predictor variable 'x'
    'line_xs_n' - number of points along x-axis for which to plot regression 
        line(s)
    'scatter_n' - number of points to plot in scatterplot
    'line_ys_func' - function to calculate y-values for regression line(s)
    'output_filepath' - file path at which to save plot
    """

    x_min = df[x_colname].min()
    x_max = df[x_colname].max()
    line_xs = np.linspace(x_min, x_max, line_xs_n)
    line_ys = line_ys_func(line_xs=line_xs, **kwargs)

    # save regression predicted 'x' and 'y' values
    output_path = output_filepath.parent
    array_output_filepath = output_path / 'line_xs.npy'
    np.savetxt(array_output_filepath, line_xs, delimiter=',')
    array_output_filepath = output_path / 'line_ys.npy'
    np.savetxt(array_output_filepath, line_ys, delimiter=',')

    # sample data points, so that not all have to be plotted 
    x = df[x_colname].sample(
        n=scatter_n, random_state=scatter_n_seed).reset_index(drop=True)
    assert isinstance(x, pd.Series)
    y = df[y_colname].sample(
        n=scatter_n, random_state=scatter_n_seed).reset_index(drop=True)
    assert isinstance(y, pd.Series)

    x_label = x_colname
    y_label = y_colname
    title = 'Regression quantiles and scatterplot of data sample'
    plot_scatter_and_regression(
        x, y, 
        line_xs=line_xs, line_ys=line_ys, 
        x_label=x_label, y_label=y_label, 
        title=title, alpha=0.05, 
        output_filepath=output_filepath)


def plot_distribution_by_bin(y_bin_counts: np.ndarray, output_filepath: Path):
    """
    Plot distribution of data points across quantile bins
    """

    y_label = 'Number of data points'
    x_label = 'Quantile bin index'
    title = 'Number of data points per quantile bin'

    plt.bar(range(len(y_bin_counts)), y_bin_counts)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def evaluate_bin_uniformity(y_bin_counts: np.ndarray, output_filepath: Path):
    """
    Calculate and save statistics concerning to what degree the bin counts
        follow a uniform distirbution
    """

    uniformity_summary = []
    uniformity_summary.append('Number of data points in each binned quantile')
    uniformity_summary.append(str(np.round(y_bin_counts, 2)))
    uniformity_summary.append('\n')

    uniformity_summary.append('Chi-square results for deviation from uniformity')
    uniform_arr = np.repeat(y_bin_counts.mean(), len(y_bin_counts))
    results = stats.chisquare(f_obs=y_bin_counts, f_exp=uniform_arr)
    uniformity_summary.append(str(results))
    uniformity_summary.append('\n')

    uniformity_summary.append('Lowest bin count divided by highest bin count')
    max_difference = y_bin_counts.min() / y_bin_counts.max()
    uniformity_summary.append(str(max_difference))

    write_list_to_text_file(uniformity_summary, output_filepath, True)


