#! /usr/bin/env python3

import copy
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

from src.s01_generate_data.generate_data import (
    create_data_with_parameters, 
    split_data_with_parameters,
    scale_data)

from src.utilities import write_list_to_text_file


@dataclass
class QuantileModelResults:

    summaries: list[str]
    params_df: pd.DataFrame

    def __post_init__(self):

        # assert pd.api.types.is_float_dtype(params_df.iloc[:, 0])
        assert (self.params_df.dtypes == np.float64).all()


def fit_summarize_quantile_model(
    model: sm.QuantReg, quantiles: np.ndarray=np.array([0.25, 0.50, 0.75]),
    ) -> QuantileModelResults:
    """
    Fit a series of quantile regression models and return the results summaries
        and parameter estimates
    """

    params = []
    summaries = []
    for q in quantiles:
        print(f'Fitting model for quantile {round(q, 2)}')
        results = model.fit(q=q)

        summaries.append('\n\n\n')
        summaries.append('******************************')
        summaries.append(f'Quantile {results.q}')
        summaries.append('******************************')
        summaries.append(results.summary())
        summaries.append('\n')
        summaries.append(results.summary2())

        params.append(results.params)

    params_df = pd.concat(params, axis=1)
    params_df.columns = [str(e) for e in quantiles]

    model_results = QuantileModelResults(
        summaries=summaries,
        params_df=params_df)

    return model_results


def calculate_quantile_prediction_vectors(
    regression_coefficients: np.ndarray, line_xs: np.ndarray) -> np.ndarray:
    """
    Calculate the quantile prediction vectors for given x-values and 
        coefficients
    """

    x_n = len(regression_coefficients) - 1
    line_xs_repeat = np.repeat(line_xs, x_n).reshape(-1, x_n)
    design_matrix = np.c_[np.ones_like(line_xs), line_xs_repeat]

    line_ys = design_matrix @ regression_coefficients

    return line_ys


def plot_regression(
    x: pd.Series, y: pd.Series, 
    line_xs: np.ndarray=np.array([]), line_ys: np.ndarray=np.array([]), 
    x_label: str='', y_label: str='', 
    title: str='', alpha: float=0.8, plot_axes: bool=True, 
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


def calculate_perpendicular_slope(slope: float) -> float:
    """
    Given the slope of a line, return the perpendicular slope
    """
    return - 1 / slope


def calculate_angle_given_slope(slope: float) -> float:
    """
    Given the slope of a line, return its angle of inclination in radians
    """
    return np.arctan(slope)


def create_direction_vector(theta):
    """
    Modified from ChatGPT
    Verified to work in 2-dimensional cases and cases where thetas all = 0 
        (see 'tests' directory)


    # for j in range(1, 5):
    #     theta = [1] * j
    #     theta_n = len(theta)
    #     B = np.ones((theta_n,))
    #     for i in range(theta_n-1):
    #     # for i in range(1):
    #         if i == 0:
    #             print('theta_n', theta_n)
    #             print('B start', B)
    #         print(i)
    #         print(B)
    #         B *= np.sin(theta[i])
    #         print(B)
    #         B = np.insert(B, 0, np.cos(theta[i]))
    #         print(B)

    #     print('\n')
    """

    theta_n = len(theta)
    B = np.ones((theta_n,))

    # original 'for' loop control from ChatGPT:
    # for i in range(theta_n-1):

    # for i in range(theta_n):

    # 2-dimensional cases and cases where thetas all = 0 require only once 
    #   through loop
    for i in range(1):
        B *= np.sin(theta[i])
        B = np.insert(B, 0, np.cos(theta[i]))

    B = B.reshape(-1, 1)

    return B


def project_matrix_to_line(
    a_matrix: np.ndarray, line_angle_radians: list[float]) -> np.ndarray:

    assert a_matrix.shape[1] - 1 == len(line_angle_radians)

    # abbreviate for more readable code
    thetas = line_angle_radians

    # works only for 2 dimensions, i.e., only one angle
    # projection_matrix = np.array(
    #     [np.sin(line_angle_radians), 
    #      np.cos(line_angle_radians)]).reshape(-1, 1)

    # works only for 3 dimensions, i.e., only two angles
    #   results verified only for cases where thetas all = 0
    # projection_matrix = np.array([
    #     np.cos(thetas[0]) * np.sin(thetas[1]), 
    #     np.sin(thetas)[0] * np.sin(thetas[1]), 
    #     np.cos(thetas[1])
    #     ]).reshape(-1, 1)

    projection_matrix = create_direction_vector(thetas)
    projection = a_matrix @ projection_matrix

    return projection


def main():

    output_path = Path.cwd() / 'output' / 's04_pytorch'
    output_path.mkdir(exist_ok=True, parents=True)

    mvn_components = create_data_with_parameters()
    data = split_data_with_parameters(mvn_components.cases_data)
    scaled_data = scale_data(
        data.train, data.valid, data.test, 
        mvn_components.predictors_column_idxs, 
        mvn_components.response_column_idx)


    ##################################################
    # 
    ##################################################


    model = nn.Sequential(
        nn.Linear(1, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_n = 1024
    train_x = torch.from_numpy(scaled_data.train_x[:train_n]).float()
    train_y = torch.from_numpy(scaled_data.train_y[:train_n]).float()
    valid_x = torch.from_numpy(scaled_data.valid_x[:train_n]).float()
    valid_y = torch.from_numpy(scaled_data.valid_y[:train_n]).float()
    test_x = torch.from_numpy(scaled_data.test_x[:train_n]).float()
    test_y = torch.from_numpy(scaled_data.test_y[:train_n]).float()

    n_epochs = 2
    batch_size = 32
    batch_n = train_x.shape[0] // batch_size
    batch_idxs = torch.arange(0, len(train_x), batch_size)

    best_metric = np.inf
    best_weights = None
    loss_log = []

    for epoch in range(n_epochs):
        model.train()
        # for i in range(batch_n):
        for i in range(10):
            batch_x = train_x[batch_idxs[i]:batch_idxs[i+1]]
            batch_y = train_y[batch_idxs[i]:batch_idxs[i+1]]
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        valid_y_pred = model(valid_x)
        valid_loss = loss_fn(valid_y_pred, valid_y)
        loss_log.append(valid_loss.item())
        if valid_loss < best_metric:
            best_metric = valid_loss
            best_weights = copy.deepcopy(model.state_dict())



    assert isinstance(best_weights, dict)
    model.load_state_dict(best_weights)




    ##################################################
    # FIT QUANTILE REGRESSION MODELS
    ##################################################

    colnames = [
        'x' + str(i+1) for i in range(mvn_components.cases_data.shape[1])]
    colnames[-1] = 'y'
    data_df = pd.DataFrame(
        np.concatenate(
            (scaled_data.train_x, 
             scaled_data.train_y.reshape(-1, 1)), 
            axis=1), 
        columns=colnames)

    formula = ['y ~ ' + ' + '.join(colnames[:-1])][0]
    model = smf.quantreg(formula, data=data_df)

    quantiles = np.arange(0.1, 0.91, 0.1)
    model_results = fit_summarize_quantile_model(model, quantiles)


    ##################################################
    # SAVE RESULTS OF QUANTILE REGRESSION MODELS
    ##################################################

    # save model results summaries
    ##################################################

    output_filename = 'summaries.txt'
    output_filepath = output_path / output_filename
    write_list_to_text_file(model_results.summaries, output_filepath, True)


    # plot scatterplot and quantile regression lines over each predictor
    ##################################################

    coefs = model_results.params_df.values

    x_colnames = [c for c in data_df.columns if c.startswith('x')]

    for x_colname in x_colnames:

        x_min = data_df[x_colname].min()
        x_max = data_df[x_colname].max()
        line_xs = np.linspace(x_min, x_max, 100)
        line_ys = calculate_quantile_prediction_vectors(coefs, line_xs)

        scatter_n = 1000
        x = data_df[x_colname][:scatter_n]
        assert isinstance(x, pd.Series)
        y = data_df['y'][:scatter_n]
        assert isinstance(y, pd.Series)

        output_filename = f'quantile_plot_{x_colname}.png'
        output_filepath = output_path / output_filename

        plot_regression(
            x, y, line_xs=line_xs, line_ys=line_ys, alpha=0.05, 
            output_filepath=output_filepath)


    ##################################################
    # Ad hoc inspection of quantile regression results
    ##################################################

    intercepts = coefs[0, :]
    regression_slope = coefs[1:, :].mean(axis=1)
    perpendicular_slope = calculate_perpendicular_slope(regression_slope)

    angle = calculate_angle_given_slope(perpendicular_slope)

    scaled_data_train = np.c_[scaled_data.train_x, scaled_data.train_y]

    # 'project_matrix_to_line' probably works for simple regression, but maybe 
    #   not for higher dimensions
    if scaled_data_train.shape[1] == 2: 
        projection = project_matrix_to_line(scaled_data_train, [angle])

        output_filename = 'quantile_regression_vs_histogram.png'
        output_filepath = output_path / output_filename
        plt.hist(projection, bins=100)
        for i in intercepts:
            plt.axvline(x=i, color='black', linestyle='dotted')
        plt.savefig(output_filepath)
        plt.clf()
        plt.close()

        output_filename = 'binned_quantiles.png'
        output_filepath = output_path / output_filename
        bins = np.digitize(projection, bins=intercepts)
        plt.hist(bins)
        plt.title('Ideally should be uniformly distributed')
        plt.savefig(output_filepath)
        plt.clf()
        plt.close()


        decile_summary = []
        decile_summary.append('Deciles estimated by quantile regression')
        decile_summary.append(str(np.round(intercepts, 2)))
        decile_summary.append(
            'Binned decile data points by quantile regression slope')
        decile_summary.append(
            str(np.round(np.quantile(projection, quantiles), 2)))

        output_filename = 'decile_summary.txt'
        output_filepath = output_path / output_filename
        write_list_to_text_file(decile_summary, output_filepath, True)



if __name__ == '__main__':
    main()