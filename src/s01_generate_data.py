#! /usr/bin/env python3

import numpy as np
from dataclasses import dataclass


@dataclass
class MultivariateNormalComponents:

    means: np.ndarray
    standard_deviations: np.ndarray
    correlation_matrix: np.ndarray
    covariance: np.ndarray
    data: np.ndarray

    def __post_init__(self):
        assert (
            self.correlation_matrix.shape[0] == 
            self.correlation_matrix.shape[1])
        assert self.means.shape[0] == self.correlation_matrix.shape[0]
        assert (
            self.standard_deviations.shape[0] == 
            self.correlation_matrix.shape[0])
        assert self.covariance.shape[0] == self.correlation_matrix.shape[0]
        assert self.covariance.shape[1] == self.correlation_matrix.shape[0]
        assert self.data.shape[1] == self.correlation_matrix.shape[0]


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
        data=mvn_data)

    return mvnc


def main():

    cases_n = 20
    x_n = 5
    variables_n = x_n + 1

    seed = 50315
    mvnc = create_centered_multivariate_normal_data(cases_n, variables_n, seed)


if __name__ == '__main__':
    main()
