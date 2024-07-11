#! /usr/bin/env python3

import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

import statsmodels.api as sm


if __name__ == '__main__':
    from s01_generate_data import (
        MultivariateNormalComponents, 
        ScaledData, 
        create_data_01_with_parameters, 
        create_data_02_with_parameters, 
        create_data_03_with_parameters, 
        create_data_04_with_parameters, 
        split_data_with_parameters,
        scale_data)

    from common import write_list_to_text_file

else:
    from src.s01_generate_data import (
        MultivariateNormalComponents, 
        ScaledData, 
        create_data_01_with_parameters, 
        create_data_02_with_parameters, 
        create_data_03_with_parameters, 
        create_data_04_with_parameters, 
        split_data_with_parameters,
        scale_data)

    from src.common import write_list_to_text_file


def scaling_data_check(
    mvn_components: MultivariateNormalComponents, scaled_data: ScaledData):
    """
    Model and report results for data set

    NOTE:  Subsequently decided to work with only scaled data, but retaining 
        code here for reference
    """

    data = split_data_with_parameters(mvn_components.cases_data)

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

    # RMSE without scaling below matches RMSE with scaling above
    # train_x = data.train[:, mvn_components.predictors_column_idxs]
    # train_y = data.train[:, mvn_components.response_column_idx]
    # model = LinearRegression().fit(train_x, train_y)
    # test_x = data.test[:, mvn_components.predictors_column_idxs]
    # test_y = data.test[:, mvn_components.response_column_idx]
    # predictions2 = model.predict(test_x)
    # rmse = root_mean_squared_error(test_y, predictions2)
    # rmse


def process_data(
    mvn_components: MultivariateNormalComponents, 
    scaled_data: ScaledData, output_path: Path):
    """
    Model and report results for data set
    """

    output_path.mkdir(exist_ok=True, parents=True)


    # fit models
    ##################################################

    model_1 = LinearRegression().fit(scaled_data.train_x, scaled_data.train_y)
    predictions = model_1.predict(scaled_data.test_x)

    # offers convenient and detailed output summary
    model_2 = sm.OLS(
        scaled_data.train_y, 
        sm.add_constant(scaled_data.train_x)).fit()

    # check that models are equivalent
    assert (model_2.params[0] - model_1.intercept_) < 1e-6
    for i in range(len(model_1.coef_)):
        assert (model_2.params[i] - model_1.coef_[i]) < 1e-6


    # report model summary
    ##################################################

    model_2_summary = model_2.summary()

    output_filename = 'model_summary.txt'
    output_filepath = output_path / output_filename
    with open(output_filepath, 'w') as f:
        f.write(model_2_summary.as_text())


    # report predictions and metrics on test data
    ##################################################

    output_filename = 'predictions.npy'
    output_filepath = output_path / output_filename
    np.savetxt(output_filepath, predictions, delimiter=',')

    mae = mean_absolute_error(scaled_data.test_y, predictions)
    rmse = root_mean_squared_error(scaled_data.test_y, predictions)

    metrics_summary = [
        'Test data Mean Absolute Error', mae,
        'Test data Root Mean Squared Error', rmse]
    output_filename = 'metrics_summary.txt'
    output_filepath = output_path / output_filename
    write_list_to_text_file(metrics_summary, output_filepath, True)


def main():

    output_path = Path.cwd() / 'output' / 's02_regress_data01'
    mvn_components = create_data_01_with_parameters()
    data = split_data_with_parameters(mvn_components.cases_data)
    scaled_data = scale_data(
        data.train, data.valid, data.test, 
        mvn_components.predictors_column_idxs, 
        mvn_components.response_column_idx)
    process_data(mvn_components, scaled_data, output_path)

    output_path = Path.cwd() / 'output' / 's02_regress_data02'
    mvn_components = create_data_02_with_parameters()
    data = split_data_with_parameters(mvn_components.cases_data)
    scaled_data = scale_data(
        data.train, data.valid, data.test, 
        mvn_components.predictors_column_idxs, 
        mvn_components.response_column_idx)
    process_data(mvn_components, scaled_data, output_path)

    output_path = Path.cwd() / 'output' / 's02_regress_data03'
    mvn_components = create_data_03_with_parameters()
    data = split_data_with_parameters(mvn_components.cases_data)
    scaled_data = scale_data(
        data.train, data.valid, data.test, 
        mvn_components.predictors_column_idxs, 
        mvn_components.response_column_idx)
    process_data(mvn_components, scaled_data, output_path)

    output_path = Path.cwd() / 'output' / 's02_regress_data04'
    mvn_components = create_data_04_with_parameters()
    data = split_data_with_parameters(mvn_components.cases_data)
    scaled_data = scale_data(
        data.train, data.valid, data.test, 
        mvn_components.predictors_column_idxs, 
        mvn_components.response_column_idx)
    process_data(mvn_components, scaled_data, output_path)


if __name__ == '__main__':
    main()
