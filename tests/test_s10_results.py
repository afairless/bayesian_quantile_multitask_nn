
import pytest
import numpy as np

from src.s10_results import (
    get_bin_cuts_for_regular_1d_grid,
    select_subarray_by_index,
    )


def test_get_bin_cuts_for_regular_1d_grid_01():
    """
    Test input with irregular intervals
    """

    grid = np.array([-2, 0, 1, 2])

    with pytest.raises(AssertionError):
        result = get_bin_cuts_for_regular_1d_grid(grid)


def test_get_bin_cuts_for_regular_1d_grid_02():
    """
    Test sorted input
    """

    grid = np.array([-2, -1, 0, 1, 2])

    result = get_bin_cuts_for_regular_1d_grid(grid)

    correct_result = np.array([[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]])

    assert np.allclose(result, correct_result)


def test_get_bin_cuts_for_regular_1d_grid_03():
    """
    Test non-sorted input
    """

    grid = np.array([1, -2, 0, -1, 2])

    result = get_bin_cuts_for_regular_1d_grid(grid)

    correct_result = np.array([[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]])

    assert np.allclose(result, correct_result)


