#! /usr/bin/env python3

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split as skl_data_split 
from sklearn.preprocessing import StandardScaler

from src.utilities import (
    read_text_file, 
    write_list_to_text_file)



@dataclass
class MultivariateNormalComponents:

    means: np.ndarray
    standard_deviations: np.ndarray
    correlation_matrix: np.ndarray
    covariance: np.ndarray
    cases_data: np.ndarray
    predictors_column_idxs: np.ndarray = (
        field(default_factory=lambda: np.array([-1])))
    response_column_idx: int = -1
    linear_regression_coefficients: np.ndarray = (
        field(default_factory=lambda: np.array([-1])))

    def __post_init__(self):

        dimension_n = self.correlation_matrix.shape[0]

        # enforce array shapes
        assert self.correlation_matrix.shape[1] == dimension_n
        assert self.means.shape[0] == dimension_n
        assert self.standard_deviations.shape[0] == dimension_n
        assert self.covariance.shape[0] == dimension_n
        assert self.covariance.shape[1] == dimension_n
        assert self.cases_data.shape[1] == dimension_n

        # enforce standard deviation values
        assert (self.standard_deviations >= 0).all()

        # enforce correlation matrix element values
        assert (self.correlation_matrix >= 0).all()
        assert (self.correlation_matrix <= 1).all()
        assert (
            self.correlation_matrix.diagonal() == np.ones(dimension_n)).all()
        assert (np.linalg.eig(self.correlation_matrix).eigenvalues >= 0).all()

        # set column indices for 'cases_data'
        self.predictors_column_idxs = np.arange(dimension_n - 1)
        self.response_column_idx = dimension_n - 1

        # calculate linear regression coefficients
        x = self.covariance[:self.response_column_idx, :self.response_column_idx]
        y = self.covariance[:self.response_column_idx, self.response_column_idx]
        self.linear_regression_coefficients = np.linalg.inv(x) @ y


@dataclass
class SplitData:
    
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray

    def __post_init__(self):

        assert isinstance(self.train, np.ndarray)
        assert isinstance(self.valid, np.ndarray)
        assert isinstance(self.test, np.ndarray)

        assert self.train.shape[1] == self.valid.shape[1]
        assert self.train.shape[1] == self.test.shape[1]


@dataclass
class ScaledData:
    
    scaler: StandardScaler
    train_x: np.ndarray
    train_y: np.ndarray
    valid_x: np.ndarray
    valid_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray

    def __post_init__(self):

        # these assertions are all very straightforward and arguably omittable,
        #   so long as 'scaler' works as I expect
        # however, in the spirit of ensuring that my own understanding and
        #   implementation are correct, I retain them here
            
        assert isinstance(self.train_x, np.ndarray)
        assert isinstance(self.train_y, np.ndarray)
        assert isinstance(self.valid_x, np.ndarray)
        assert isinstance(self.valid_y, np.ndarray)
        assert isinstance(self.test_x, np.ndarray)
        assert isinstance(self.test_y, np.ndarray)

        assert self.train_x.shape[0] == self.train_y.shape[0]
        assert self.valid_x.shape[0] == self.valid_y.shape[0]
        assert self.test_x.shape[0] == self.test_y.shape[0]

        assert self.train_x.shape[1] == self.valid_x.shape[1]
        assert self.train_x.shape[1] == self.test_x.shape[1]


def create_correlation_matrix(dimension_n: int, seed: int) -> np.ndarray:
    """
    Given the number of dimensions, generate a random correlation matrix
    
    Github Copilot claims that this method guarantees positive 
        semi-definiteness; I haven't verified that mathematically, but tests
        with the Python package 'hypothesis' (see 'tests' directory) haven't
        found a counter-example
    """

    assert dimension_n >= 2

    np.random.seed(seed)

    A = np.random.rand(dimension_n, dimension_n)
    B = np.dot(A, A.T)
    D_inv = np.diag(1 / np.sqrt(np.diag(B)))
    correlation_matrix = np.dot(D_inv, np.dot(B, D_inv))
    np.fill_diagonal(correlation_matrix, 1)

    # notify user if matrix is not positive semi-definite
    eigs = np.linalg.eig(correlation_matrix)
    if (eigs.eigenvalues < 0).any():
        print('The correlation matrix has negative eigenvalues, meaning that '
          'it is not positive semi-definite.')

    return correlation_matrix


def create_multivariate_normal_data(
    cases_n: int, variables_n: int, seed: int, zero_centered: bool=True
    ) -> MultivariateNormalComponents:
    """
    Generate multivariate normal data with given numbers of cases and 
        variables, centered at the origin

    cases_n - number of cases or observations, i.e., rows in the generated table
    variables_n - number of variables, i.e., columns in the generated table
    """

    np.random.seed(seed)

    if zero_centered:
        mvn_means = np.zeros(variables_n)
    else:
        mvn_means = np.random.randint(-100, 100, variables_n)

    mvn_stds = np.random.randint(1, 100, variables_n)
    mvn_correlation = create_correlation_matrix(variables_n, seed+1)
    mvn_covariance = np.outer(mvn_stds, mvn_stds) * mvn_correlation

    # verify covariance calculation with alternative calculation
    mvn_covariance2 = np.diag(mvn_stds) @ mvn_correlation @ np.diag(mvn_stds)
    assert np.allclose(mvn_covariance, mvn_covariance2)

    mvn_data = np.random.multivariate_normal(mvn_means, mvn_covariance, cases_n)

    mvnc = MultivariateNormalComponents(
        correlation_matrix=mvn_correlation,
        means=mvn_means,
        standard_deviations=mvn_stds,
        covariance=mvn_covariance,
        cases_data=mvn_data)

    return mvnc


def create_data_with_parameters() -> MultivariateNormalComponents:
    """
    Create multivariate normal data with standard parameters
    """

    cases_n = 1_000_000
    predictors_n = 5
    variables_n = predictors_n + 1

    seed = 50319
    mvnc = create_multivariate_normal_data(cases_n, variables_n, seed, False)

    return mvnc


def split_data_3ways(data_array: np.ndarray, seed: int) -> SplitData:
    """
    Split data into training, validation, and testing sets
    """

    train_data, non_train_data = skl_data_split(
        data_array, train_size=0.6, random_state=seed)
    valid_data, test_data = skl_data_split(
        non_train_data, train_size=0.5, random_state=seed+1)

    assert isinstance(train_data, np.ndarray)
    assert isinstance(valid_data, np.ndarray)
    assert isinstance(test_data, np.ndarray)
    assert (
        data_array.shape[0] == 
        train_data.shape[0] + valid_data.shape[0] + test_data.shape[0])

    split_data = SplitData(train_data, valid_data, test_data)

    return split_data


def split_data_with_parameters(data_array: np.ndarray) -> SplitData:
    """
    Split data with standard parameter (i.e., 'seed')
    """

    seed = 411057
    split_data = split_data_3ways(data_array, seed)

    return split_data


def scale_data(
    train_data: np.ndarray, valid_data: np.ndarray, test_data: np.ndarray, 
    predictors_column_idxs: np.ndarray, response_column_idx: int) -> ScaledData:
    """
    Scale training, validation, and testing data
    """

    assert train_data.shape[1] == valid_data.shape[1]
    assert train_data.shape[1] == test_data.shape[1]

    scaler = StandardScaler().fit(train_data)
    train = scaler.transform(train_data)
    assert isinstance(train, np.ndarray)
    train_x = train[:, predictors_column_idxs]
    train_y = train[:, response_column_idx]

    valid = scaler.transform(valid_data)
    assert isinstance(valid, np.ndarray)
    valid_x = valid[:, predictors_column_idxs]
    valid_y = valid[:, response_column_idx]

    test = scaler.transform(test_data)
    assert isinstance(test, np.ndarray)
    test_x = test[:, predictors_column_idxs]
    test_y = test[:, response_column_idx]

    scaled_data = ScaledData(
        scaler, 
        train_x, train_y, 
        valid_x, valid_y, 
        test_x, test_y)

    return scaled_data


def save_data():
    """
    Serialize data to disk as JSON files for downstream processing with
        incompatible development environments
    """

    output_path = Path.cwd() / 'output' / 'data'
    output_path.mkdir(parents=True, exist_ok=True)

    mvn_components = create_data_with_parameters()
    data = split_data_with_parameters(mvn_components.cases_data)
    scaled_data = scale_data(
        data.train, data.valid, data.test, 
        mvn_components.predictors_column_idxs, 
        mvn_components.response_column_idx)

    data_filenames = [
        (scaled_data.train_x, 'scaled_data_train_x.json'),
        (scaled_data.train_y, 'scaled_data_train_y.json'),
        (scaled_data.test_x,  'scaled_data_test_x.json'),
        (mvn_components.predictors_column_idxs,  
         'predictors_column_idxs.json'),
        (np.array(mvn_components.response_column_idx),  
         'response_column_idx.json'),
        ]


    for e in data_filenames:

        data_array = e[0]
        filename = e[1]

        output_filepath = output_path / filename 
        array_list = data_array.tolist()

        with open(output_filepath, 'w') as json_file:
            json.dump(array_list, json_file)


if __name__ == '__main__':
    save_data()
