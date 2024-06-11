
import pytest
import numpy as np

from hypothesis import given, settings
import hypothesis.strategies as st

from src.s03_quantile import (
    calculate_quantile_prediction_vectors,
    )


# def test_calculate_quantile_prediction_vectors_01():
#     """
#     Test valid input
#     """

#     x = np.array([[0, 0.5, 1], [0, 5, 10]])
#     coefs = np.array([1, 1, 1])

#     result = calculate_quantile_prediction_vectors(x, coefs, 3)

#     correct_result = np.array([[1, 1, 1, 1], [2, 2, 2, 2]])
#     breakpoint()

#     assert np.allclose(result, correct_result)



def test_calculate_quantile_prediction_vectors_01():
    """
    Test one set of coefficients
    """

    coefficients = np.array([1, 1, 1])
    line_xs = np.array([0, 0.5, 1])

    result = calculate_quantile_prediction_vectors(coefficients, line_xs)

    correct_result = np.array([[1, 2, 3]])

    assert np.allclose(result, correct_result)


def test_calculate_quantile_prediction_vectors_02():
    """
    Test >1 set of coefficients
    """

    coefficients = np.array([[1, 1, 1], [2, 2, 2]]).transpose()
    line_xs = np.array([0, 0.5, 1])

    result = calculate_quantile_prediction_vectors(coefficients, line_xs)

    correct_result = np.array([[1, 2, 3], [2, 4, 6]]).transpose()

    assert np.allclose(result, correct_result)





