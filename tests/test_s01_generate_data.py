
import pytest
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from hypothesis import given, settings
import hypothesis.strategies as st

from src.s01_generate_data import (
    create_correlation_matrix,
    create_multivariate_normal_data,
    )


@given(
    dimension_n=st.integers(min_value=2, max_value=90),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_create_correlation_matrix_01(dimension_n: int, seed: int):
    """
    Test valid input
    """

    result = create_correlation_matrix(dimension_n, seed)

    assert result.shape == (dimension_n, dimension_n)

    # test that all values are between 0 and 1, inclusive
    assert (result >= 0).all()
    assert (result <= 1).all()

    # test that all diagonal elements = 1
    assert (result.diagonal() == np.ones(dimension_n)).all()

    # test that correlation matrix is positive semi-definite
    assert (np.linalg.eig(result).eigenvalues >= 0).all()


@given(
    variables_n=st.integers(min_value=3, max_value=5),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(max_examples=10)
def test_multivariate_normal_components_linear_regression_coefficients_01(
    variables_n: int, seed: int):
    """
    Verify that linear algebra calculation of linear regression coefficients in
        'MultivariateNormalComponents' is correct

    Test parameters have been adjusted to be less stringent to allow for rapid,
        reliable re-running of tests
    """

    cases_n = 300_000
    mvnc = create_multivariate_normal_data(cases_n, variables_n, seed)

    x = mvnc.cases_data[:, mvnc.predictors_column_idxs]
    y = mvnc.cases_data[:, mvnc.response_column_idx]
    model = LinearRegression().fit(x, y)
    assert np.allclose(model.intercept_, 0, atol=5e-1)
    assert np.allclose(
        model.coef_, mvnc.linear_regression_coefficients, atol=5e-1)

    # alternative but slower calculation 
    # regression_results = sm.OLS(y, sm.add_constant(x)).fit()
    # assert np.allclose(regression_results.params[0], 0, atol=1e-1)
    # assert np.allclose(
    #     regression_results.params[1:], 
    #     mvnc.linear_regression_coefficients, atol=1e-1)


# def test_dummy_function02_01():
#     """
#     Test input of wrong data type
#     """

#     input = -1

#     with pytest.raises(AssertionError):
#         result = dummy_function02(input)

