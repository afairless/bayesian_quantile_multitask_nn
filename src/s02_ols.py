#! /usr/bin/env python3

import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

from s01_generate_data import (
    MultivariateNormalComponents, 
    ScaledData, 
    create_data_01_with_parameters, 
    create_data_02_with_parameters, 
    split_data_with_parameters,
    scale_data)

import statsmodels.api as sm


def process_data(
    mvn_components: MultivariateNormalComponents, scaled_data: ScaledData, 
    output_path: Path):
    """
    Model and report results for data set
    """

    mvn_components = create_data_01_with_parameters()
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

    model = LinearRegression().fit(scaled_data.train_x, scaled_data.train_y)
    predictions_scaled = model.predict(scaled_data.test_x)

    # unscale predictions
    # first, concatenate the scaled predictors and the scaled predictions, so
    #   that 'scaler' can be conveniently used (one could unscale the 
    #   predictions manually, but the code is more maintainable this way in case
    #   the scaler is changed)
    x_predictions_scaled = np.concatenate(
        (scaled_data.test_x, predictions_scaled.reshape(-1, 1)), axis=1)
    x_predictions = scaled_data.scaler.inverse_transform(x_predictions_scaled)

    predictions = x_predictions[:, mvn_components.response_column_idx]
    test_y = data.test[:, mvn_components.response_column_idx]
    rmse = root_mean_squared_error(test_y, predictions)
    print(rmse)


    # RMSE without scaling below matches RMSE with scaling above
    # train_x = data.train[:, mvn_components.predictors_column_idxs]
    # train_y = data.train[:, mvn_components.response_column_idx]
    # model = LinearRegression().fit(train_x, train_y)
    # test_x = data.test[:, mvn_components.predictors_column_idxs]
    # test_y = data.test[:, mvn_components.response_column_idx]
    # predictions2 = model.predict(test_x)
    # rmse = root_mean_squared_error(test_y, predictions2)
    # rmse


def main():

    output_path = Path.cwd() / 'output' / 's02_ols_data01'
    mvn_components = create_data_01_with_parameters()
    data = split_data_with_parameters(mvn_components.cases_data)
    scaled_data = scale_data(
        data.train, data.valid, data.test, 
        mvn_components.predictors_column_idxs, 
        mvn_components.response_column_idx)
    process_data(mvn_components, scaled_data, output_path)

    output_path = Path.cwd() / 'output' / 's02_ols_data02'
    mvn_components = create_data_02_with_parameters()
    data = split_data_with_parameters(mvn_components.cases_data)
    scaled_data = scale_data(
        data.train, data.valid, data.test, 
        mvn_components.predictors_column_idxs, 
        mvn_components.response_column_idx)
    process_data(mvn_components, scaled_data, output_path)


if __name__ == '__main__':
    main()
