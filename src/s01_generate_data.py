#! /usr/bin/env python3

import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split as skl_data_split 
from pathlib import Path


@dataclass
class MultivariateNormalComponents:

    means: np.ndarray
    standard_deviations: np.ndarray
    correlation_matrix: np.ndarray
    covariance: np.ndarray
    cases_data: np.ndarray

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


def create_centered_multivariate_normal_data(
    cases_n: int, variables_n: int, seed: int) -> MultivariateNormalComponents:
    """
    Generate multivariate normal data with given numbers of cases and 
        variables, centered at the origin

    cases_n - number of cases or observations, i.e., rows in the generated table
    variables_n - number of variables, i.e., columns in the generated table
    """

    np.random.seed(seed)

    # variables are zero-centered
    mvn_means = np.zeros(variables_n)
    mvn_stds = np.random.randint(1, 100, variables_n) / 10
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

    cases_n = 1000
    predictors_n = 5
    variables_n = predictors_n + 1

    seed = 50315
    mvnc = create_centered_multivariate_normal_data(cases_n, variables_n, seed)

    return mvnc


def split_data_3ways(data_array: np.ndarray, seed: int) -> SplitData:

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

    seed = 411057
    split_data = split_data_3ways(data_array, seed)

    return split_data


if __name__ == '__main__':
    create_data_with_parameters()
