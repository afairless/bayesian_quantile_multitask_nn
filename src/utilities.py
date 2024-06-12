
import numpy as np
import pandas as pd
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
# FUNCTIONS FOR PLOTTING
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

    x = df[x_colname].sample(
        n=scatter_n, random_state=scatter_n_seed).reset_index(drop=True)
    assert isinstance(x, pd.Series)
    y = df[y_colname].sample(
        n=scatter_n, random_state=scatter_n_seed).reset_index(drop=True)
    assert isinstance(y, pd.Series)

    plot_scatter_and_regression(
        x, y, line_xs=line_xs, line_ys=line_ys, alpha=0.05, 
        output_filepath=output_filepath)


