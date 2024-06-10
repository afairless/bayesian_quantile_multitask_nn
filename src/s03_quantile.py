#! /usr/bin/env python3

import numpy as np
from pandas import DataFrame as pd_DataFrame

from src.s01_generate_data.generate_data import (
    create_data_with_parameters, 
    split_data_with_parameters,
    scale_data)

import statsmodels.formula.api as smf


def main():

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
    data_df = pd_DataFrame(
        np.concatenate(
            (scaled_data.train_x, 
             scaled_data.train_y.reshape(-1, 1)), 
            axis=1), 
        columns=colnames)
    model = smf.quantreg('y ~ x1 + x2', data=data_df)

    quantiles = np.arange(0.1, 0.91, 0.1)
    for q in quantiles:
        results = model.fit(q=q)
        # print(results.summary())
        # print(results.summary2())
        print(results.params)



if __name__ == '__main__':
    main()
