#! /usr/bin/env python3

import json
import pystan
import arviz as az
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from numpy.random import choice as np_choice
from numpy.random import normal as np_normal
from itertools import product as it_product
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import root_mean_squared_error


def write_list_to_text_file(
    a_list: list, text_filename: str, overwrite: bool=False):
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


def calculate_gaussian_kernel_density_bandwidth_silverman_rule(
    a_df: pd.DataFrame) -> pd.Series:
    """
    Calculate Gaussian kernel density bandwidth based on Silverman's rule from:
        Silverman, B. W. (1986).  Density Estimation for Statistics and Data
            Analysis.  London: Chapman & Hall/CRC. p. 45
            ISBN 978-0-412-24620-3

    Wikipedia is a useful reference:

        https://en.wikipedia.org/wiki/Kernel_density_estimation

    a_df:  a Pandas DataFrame where the Gaussian kernel density will be
        calculated for each column
    :return: scalar float representing bandwidth
    """

    # find interquartile range and divide it by 1.34
    iqr_div134 = (a_df.quantile(0.75) - a_df.quantile(0.25)) / 1.34

    # choose minimum of 'iqr_div134' and standard deviation for each variable
    a = pd.concat([iqr_div134, a_df.std()], axis=1).min(axis=1)

    h = 0.9 * a * len(a_df)**(-1/5)

    # check bandwidths/std on each variable

    return h


def resample_variables_by_gaussian_kernel_density(
    a_df: pd.DataFrame, sample_n: int) -> pd.DataFrame:
    """
    For each column in Pandas DataFrame 'a_df', calculates a new sample of that
        variable based on Gaussian kernel density

    a_df:  a Pandas DataFrame with columns of numerical data
    sample_n:  the number of new samples to calculate for each column
    :return: a Pandas DataFrame with 'sample_n' rows and the same number of
        columns as 'a_df'
    """

    bandwidths = calculate_gaussian_kernel_density_bandwidth_silverman_rule(a_df)
    re_sample = a_df.sample(n=sample_n, replace=True)
    density_re_sample = np_normal(
        loc=re_sample, scale=bandwidths, size=(sample_n, a_df.shape[1]))

    density_re_sample = pd.DataFrame(density_re_sample, columns=a_df.columns)

    return density_re_sample


def create_grid_regular_intervals_two_variables(
    a_df: pd.DataFrame, intervals_num: int) -> pd.DataFrame:
    """
    1) Accepts Pandas DataFrame where first two columns are numerical values
    2) Finds the range of each of these columns and divides each range into
        equally spaced intervals; the number of intervals is specified by
        'intervals_num'
    3) Creates new DataFrame with two columns where the rows represent the
        Cartesian product of the equally spaced intervals

    a_df:  a Pandas DataFrame where first two columns are numerical values
    intervals_num:  scalar integer; the number of equally spaced intervals
        to create for each column
    """

    intervals_df = a_df.apply(lambda d: np.linspace(
        start=d.min(), stop=d.max(), num=intervals_num))

    # the following code works much like 'expand.grid' in R, but it handles only
    #   two variables
    cartesian_product = list(
        it_product(intervals_df.iloc[:, 0], intervals_df.iloc[:, 1]))

    product_df = pd.DataFrame.from_records(
        cartesian_product, columns=a_df.columns)

    return product_df


def save_summaries(fit_df: pd.DataFrame, fit_model, output_path: Path):

    filename = 'fit_df.csv'
    output_filepath = output_path / filename
    fit_df.to_csv(output_filepath)

    filename = 'fit_df.parquet'
    output_filepath = output_path / filename
    fit_df.to_parquet(output_filepath)

    # text summary of means, sd, se, and quantiles for parameters, n_eff, & Rhat
    fit_stansummary = fit_model.stansummary()
    output_filepath = output_path / 'stansummary.txt'
    write_list_to_text_file([fit_stansummary], output_filepath.as_posix(), True)

    # same summary as for 'stansummary', but in matrix/dataframe form instead of text
    fit_summary_df = pd.DataFrame(
        fit_model.summary()['summary'],
        index=fit_model.summary()['summary_rownames'],
        columns=fit_model.summary()['summary_colnames'])
    output_filepath = output_path / 'stansummary.csv'
    fit_summary_df.to_csv(output_filepath, index=True)
    output_filepath = output_path / 'stansummary.parquet'
    fit_summary_df.to_parquet(output_filepath, index=True)


def save_plots(
    x: np.ndarray, y: np.ndarray, 
    fit_df: pd.DataFrame, fit_model, 
    output_path: Path):

    # plot parameters
    ##################################################

    az_stan_data = az.from_pystan(
        posterior=fit_model,
        posterior_predictive='predicted_y_given_x',
        observed_data=['y'])


    az.style.use('arviz-darkgrid')
    parameter_names = ['alpha', 'beta', 'sigma']
    show = False


    # plot chain autocorrelation
    ##################################################

    print('Plotting autocorrelation')
    az.plot_autocorr(az_stan_data, var_names=parameter_names, show=show)
    output_filepath = output_path / 'plot_autocorr.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_autocorr(
        az_stan_data, var_names=parameter_names, combined=True, show=show)
    output_filepath = output_path / 'plot_autocorr_combined_chains.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # plot parameter density
    ##################################################

    print('Plotting density')
    az.plot_density(
        az_stan_data, var_names=parameter_names, outline=False, shade=0.7,
        credible_interval=0.9, point_estimate='mean', show=show)
    output_filepath = output_path / 'plot_density.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # plot distribution
    ##################################################

    print('Plotting distribution')
    az.plot_dist(
        fit_df[parameter_names[1]+'[1]'], rug=True,
        quantiles=[0.25, 0.5, 0.75], show=show)
    output_filepath = output_path / 'plot_distribution.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_dist(
        fit_df[parameter_names[1]+'[1]'], rug=True, cumulative=True,
        quantiles=[0.25, 0.5, 0.75], show=show)
    output_filepath = output_path / 'plot_distribution_cumulative.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # plot ESS across local parts of distribution
    ##################################################

    print('Plotting ESS')
    az.plot_ess(
        az_stan_data, var_names=parameter_names, kind='local', n_points=10,
        rug=True, extra_methods=True, show=show)
    output_filepath = output_path / 'plot_ess_local.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_ess(
        az_stan_data, var_names=parameter_names, kind='quantile', n_points=10,
        rug=True, extra_methods=True, show=show)
    output_filepath = output_path / 'plot_ess_quantile.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_ess(
        az_stan_data, var_names=parameter_names, kind='evolution', n_points=10,
        rug=True, extra_methods=True, show=show)
    output_filepath = output_path / 'plot_ess_evolution.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # forest plots
    ##################################################

    print('Plotting forest plots')

    # look at model estimations of parameters, r-hat, and ess
    az.plot_forest(
        az_stan_data, kind='forestplot', var_names=parameter_names,
        linewidth=6, markersize=8,
        credible_interval=0.9, r_hat=True, ess=True, show=show)
    output_filepath = output_path / 'plot_forest.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    # look at model estimations of parameters, r-hat, and ess
    az.plot_forest(
        az_stan_data, kind='ridgeplot', var_names=parameter_names,
        credible_interval=0.9, r_hat=True, ess=True,
        ridgeplot_alpha=0.5, ridgeplot_overlap=2, ridgeplot_kind='auto',
        show=show)
    output_filepath = output_path / 'plot_forest_ridge.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # HPD plot
    ##################################################

    print('Plotting HPD plots')

    # look at model estimations of parameters, r-hat, and ess
    predicted_y_colnames = [e for e in fit_df.columns if 'y_given_x' in e]
    predicted_y_df = fit_df[predicted_y_colnames]

    for x_col_idx in range(x.shape[1]):
        plt.scatter(x[:, x_col_idx], y)
        az.plot_hpd(
            x[:, x_col_idx], predicted_y_df, credible_interval=0.5, show=show)
        az.plot_hpd(
            x[:, x_col_idx], predicted_y_df, credible_interval=0.9, show=show)
        filename = 'plot_hpd_x' + str(x_col_idx) + '.png'
        output_filepath = output_path / filename
        plt.savefig(output_filepath)
        plt.clf()
        plt.close()


    # plot KDE
    ##################################################

    print('Plotting KDE plots')

    az.plot_kde(
        fit_df[parameter_names[0]], fit_df[parameter_names[1]+'[1]'],
        contour=True, show=show)
    output_filepath = output_path / 'plot_kde_contour.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_kde(
        fit_df[parameter_names[0]], fit_df[parameter_names[1]+'[1]'],
        contour=False, show=show)
    output_filepath = output_path / 'plot_kde_no_contour.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # MCSE statistics and plots
    ##################################################

    print('Plotting MCSE plots')

    az.mcse(az_stan_data, var_names=parameter_names, method='mean')
    az.mcse(az_stan_data, var_names=parameter_names, method='sd')
    az.mcse(az_stan_data, var_names=parameter_names, method='quantile', prob=0.1)

    az.plot_mcse(
        az_stan_data, var_names=parameter_names, errorbar=True, n_points=10)
    output_filepath = output_path / 'plot_mcse_errorbar.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_mcse(
        az_stan_data, var_names=parameter_names, extra_methods=True,
        n_points=10)
    output_filepath = output_path / 'plot_mcse_extra_methods.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()



    # plot pair
    ##################################################

    print('Plotting pair plots')

    az.plot_pair(
        az_stan_data, var_names=parameter_names, kind='scatter',
        divergences=True, show=show)
    output_filepath = output_path / 'plot_pair_scatter.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_pair(
        az_stan_data, var_names=parameter_names, kind='kde',
        divergences=True, show=show)
    output_filepath = output_path / 'plot_pair_kde.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # plot parameters in parallel
    ##################################################

    print('Plotting parallel plots')

    az.plot_parallel(
        az_stan_data, var_names=parameter_names, colornd='blue', show=show)
    output_filepath = output_path / 'plot_parallel.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # plot parameters in parallel
    ##################################################

    az.plot_posterior(
        az_stan_data, var_names=parameter_names, credible_interval=0.9,
        point_estimate='mean', show=show)
    output_filepath = output_path / 'plot_posterior.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    """
    # plot predictive check
    ##################################################

    print('Plotting predictive checks')

    az.plot_ppc(
        az_stan_data, kind='kde', data_pairs={'y': 'predict_y_given_x'},
        random_seed=483742, show=show)
    output_filepath = output_path / 'plot_predictive_check_kde.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_ppc(
        az_stan_data, kind='cumulative', data_pairs={'y': 'predict_y_given_x'},
        random_seed=483742, show=show)
    output_filepath = output_path / 'plot_predictive_check_cumulative.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_ppc(
        az_stan_data, kind='scatter', data_pairs={'y': 'predict_y_given_x'},
        random_seed=483742, jitter=0.5, show=show)
    output_filepath = output_path / 'plot_predictive_check_scatter.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    """


    # plot chain rank order statistics
    ##################################################
    # each chain should show approximately a uniform distribution:
    #   https://arxiv.org/pdf/1903.08008
    #   Vehtari, Gelman, Simpson, Carpenter, Bürkner (2020)
    #   Rank-normalization, folding, and localization: An improved R for
    #       assessing convergence of MCMC

    az.plot_rank(
        az_stan_data, var_names=parameter_names, kind='bars', show=show)
    output_filepath = output_path / 'plot_rank_bars.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_rank(
        az_stan_data, var_names=parameter_names, kind='vlines', show=show)
    output_filepath = output_path / 'plot_rank_vlines.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # plot traces
    ##################################################

    print('Plotting traces')

    az.plot_trace(
        az_stan_data, var_names=parameter_names, legend=False, show=show)
    output_filepath = output_path / 'plot_trace.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # plot distributions on violin plot
    ##################################################

    print('Plotting violin plots')

    az.plot_violin(
        az_stan_data, var_names=parameter_names, rug=True,
        credible_interval=0.9, show=show)
    output_filepath = output_path / 'plot_violin.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_scatter_and_regression(
    x: pd.Series, y: pd.Series, 
    line_xs: np.ndarray=np.array([]), line_ys: np.ndarray=np.array([]), 
    x_label: str='', y_label: str='', 
    title: str='', alpha: float=0.8, 
    output_filepath: Path=Path('plot.png')):
    """
    Plot scatterplot of response variable 'y' over one predictor variable 'x'
        and any number of regression lines 'line_ys' plotted over 'line_xs'
    Copied from 'common' module because this module is run in a different
        virtual environment
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
    x: np.ndarray, y: np.ndarray, 
    x_colname: str, y_colname: str, 
    line_xs: np.ndarray, line_ys: np.ndarray,
    scatter_n: int, scatter_n_seed: int, 
    output_filepath: Path):
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

    Adapted from 'common' module because this module is run in a different
        virtual environment
    """

    # sample data points, so that not all have to be plotted 
    # use Pandas to match similar function 
    x_sample = pd.Series(x[:, 0]).sample(
        n=scatter_n, random_state=scatter_n_seed).reset_index(drop=True)
    assert isinstance(x_sample, pd.Series)
    y_sample = pd.Series(y).sample(
        n=scatter_n, random_state=scatter_n_seed).reset_index(drop=True)
    assert isinstance(y_sample, pd.Series)

    x_label = x_colname
    y_label = y_colname
    title = 'Regression quantiles and scatterplot of data sample'
    plot_scatter_and_regression(
        x_sample, y_sample, 
        line_xs=line_xs, line_ys=line_ys, 
        x_label=x_label, y_label=y_label, 
        title=title, alpha=0.05, 
        output_filepath=output_filepath)


def main():

    input_path = Path.cwd() / 'output' / 'data'
    input_filepaths = list(input_path.glob('*.json'))

    for e in input_filepaths:

        with open(e, 'r') as json_file:
            json_obj = json.load(json_file)

        globals()[e.stem] = np.array(json_obj)

    output_path = Path.cwd() / 'output' / 's02_bayes_stan'
    output_path.mkdir(parents=True, exist_ok=True)

    # variables are assigned by 'globals' above
    #   explicate here to reduce complaints from linter
    # scaled_data_train_y = scaled_data_train_y
    # scaled_data_train_x = scaled_data_train_x

    y = scaled_data_train_y
    x = scaled_data_train_x
    n = x.shape[0]
    k = x.shape[1]
    sample_n = 100
    sample_idx = np_choice(range(n), sample_n, replace=False)
    x_sample = x[sample_idx, :]
    y_sample = y[sample_idx]

    # set up regular intervals of 'x's for model to predict
    x_min = x.min()
    x_max = x.max()
    line_xs_n = 100
    line_xs: np.ndarray = np.linspace(x_min, x_max, line_xs_n)
    line_xs = line_xs.reshape(-1, 1)

    stan_data = {
        'N': n, 'K': k, 'x': x, 'y': y,
        'predict_y_given_x_n': x_sample.shape[0],
        'predict_y_given_x': x_sample,
        'predict_y_given_regular_x_n': line_xs.shape[0],
        'predict_y_given_regular_x': line_xs,
        }
    stan_filename = 's02_bayes_stan.stan'
    stan_filepath = Path.cwd() / 'src' / 'stan_code' / stan_filename

    stan_model = pystan.StanModel(file=stan_filepath.as_posix())
    fit_model = stan_model.sampling(
       data=stan_data, iter=300, chains=1, warmup=150, thin=1, seed=708869)
    #fit_model = stan_model.sampling(
    #    data=stan_data, iter=2000, chains=4, warmup=1000, thin=1, seed=22074)
    # fit_model = stan_model.sampling(
    #     data=stan_data, iter=2000, chains=4, warmup=1000, thin=2, seed=22074)

    # all samples for all parameters, predicted values, and diagnostics
    #   number of rows = number of 'iter' in 'StanModel.sampling' call
    fit_df = fit_model.to_dataframe()

    save_summaries(fit_df, fit_model, output_path)
    save_plots(x_sample, y_sample, fit_df, fit_model, output_path)

    line_ys = np.repeat(
        np.array([-0.3, 0, 0.6]).reshape(-1, 1), 
        len(line_xs), 
        axis=1).T
    x_colname = 'x1'
    x_label = x_colname
    y_label = 'y'
    output_filename = f'quantile_plot_{x_label}.png'
    output_filepath = output_path / output_filename

    plot_scatter_regression_with_parameters(
        x, y, x_label, y_label, 
        line_xs, line_ys,
        scatter_n=1000, scatter_n_seed=29344,
        output_filepath=output_filepath)




if __name__ == '__main__':
    main()
