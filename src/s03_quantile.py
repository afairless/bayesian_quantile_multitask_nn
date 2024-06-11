#! /usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

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



def plot_regression(
    x: pd.Series, y: pd.Series, 
    line_xs: np.ndarray=np.array([]), line_ys: np.ndarray=np.array([]), 
    x_label: str='', y_label: str='', 
    title: str='', alpha: float=0.8, plot_axes: bool=True, 
    output_filepath: Path=Path('plot.png')):

    plt.scatter(x, y, alpha=alpha)

    if line_xs.size > 0 and line_ys.size > 0:
        assert len(line_xs) == line_ys.shape[0]
        for i in range(line_ys.shape[1]):
            plt.plot(line_xs, line_ys[:, i], color='black', linestyle='dotted')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def main():

    output_path = Path.cwd() / 'output' / 's03_quantile'
    output_path.mkdir(exist_ok=True, parents=True)

    mvn_components = create_data_with_parameters()
    data = split_data_with_parameters(mvn_components.cases_data)
    scaled_data = scale_data(
        data.train, data.valid, data.test, 
        mvn_components.predictors_column_idxs, 
        mvn_components.response_column_idx)

    # mvn_components.means
    # np.mean(mvn_components.cases_data, axis=0)
    # mvn_components.standard_deviations
    # np.std(mvn_components.cases_data, axis=0)
    # mvn_components.linear_regression_coefficients

    colnames = ['x1', 'x2', 'y']
    data_df = pd.DataFrame(
        np.concatenate(
            (scaled_data.train_x, 
             scaled_data.train_y.reshape(-1, 1)), 
            axis=1), 
        columns=colnames)

    # TODO: FIX FORMULA GENERATION TO ACCOMMODATE FLEXIBLE # OF XS
    model = smf.quantreg('y ~ x1 + x2', data=data_df)

    quantiles = np.arange(0.1, 0.91, 0.1)
    model_results = fit_summarize_quantile_model(model, quantiles)

    output_filename = 'summaries.txt'
    output_filepath = output_path / output_filename
    write_list_to_text_file(model_results.summaries, output_filepath)

    coefs = model_results.params_df.values
    # coefs = np.c_[params].transpose()

    x_colname = 'x1'
    x_min = data_df[x_colname].min()
    x_max = data_df[x_colname].max()
    line_xs = np.linspace(x_min, x_max, 100)

    # coefs = results.params
    x_n = data_df.shape[1] - 1
    line_xs_repeat = np.repeat(line_xs, x_n).reshape(-1, x_n)
    design_matrix = np.c_[np.ones_like(line_xs), line_xs_repeat]
    line_ys = np.dot(design_matrix, coefs)


    x = data_df[x_colname]
    assert isinstance(x, pd.Series)
    y = data_df['y']
    assert isinstance(y, pd.Series)
    output_filename = f'quantile_plot_{x_colname}.png'
    output_filepath = output_path / output_filename
    plot_regression(
        x, y, line_xs=line_xs, line_ys=line_ys, alpha=0.05, 
        output_filepath=output_filepath)





if __name__ == '__main__':
    main()
