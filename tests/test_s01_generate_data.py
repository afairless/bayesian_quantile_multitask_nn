 
import pytest
from numpy import ones as np_ones
from numpy.linalg import eig as np_eig

from hypothesis import given, settings
import hypothesis.strategies as st

from src.s01_generate_data import (
    create_correlation_matrix,
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
    assert (result.diagonal() == np_ones(dimension_n)).all()

    # test that correlation matrix is positive semi-definite
    assert (np_eig(result).eigenvalues >= 0).all()


# def test_dummy_function02_01():
#     """
#     Test input of wrong data type
#     """

#     input = -1

#     with pytest.raises(AssertionError):
#         result = dummy_function02(input)

