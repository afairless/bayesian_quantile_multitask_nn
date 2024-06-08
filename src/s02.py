#! /usr/bin/env python3

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

import statsmodels.api as sm

from src.s01_generate_data import (
    create_data_with_parameters, 
    split_data_with_parameters)



def main():

    mvn_components = create_data_with_parameters()
    data = split_data_with_parameters(mvn_components.cases_data)


    # calculate coefficients from data, include in mvnc
    # maybe add means to data
    # scale data
    # https://scikit-learn.org/stable/api/sklearn.preprocessing.html

    # cross-validate?  probably not

    train_x = data.train[:, mvn_components.predictors_column_idxs]
    train_y = data.train[:, mvn_components.response_column_idx]
    model = LinearRegression().fit(train_x, train_y)
    # model.coef_
    # model.intercept_

    test_x = data.test[:, mvn_components.predictors_column_idxs]
    test_y = data.test[:, mvn_components.response_column_idx]
    predictions = model.predict(test_x)
    rmse = root_mean_squared_error(test_y, predictions)

    regression_results = sm.OLS(train_y, sm.add_constant(train_x)).fit()


if __name__ == '__main__':
    main()
