
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


def test_select_subarray_by_index_01():
    """
    Test select_idx_n > total_idx_n
    """

    arr = np.array([
        [9, 8, 7, 6, 5],
        [0, 1, 2, 3, 4]]).transpose()
    arr_col_idx = 1

    total_idx_n = arr[:, arr_col_idx].max() + 1
    select_idx_n = 10

    with pytest.raises(AssertionError):
        result = select_subarray_by_index(
            arr, arr_col_idx, total_idx_n, select_idx_n)


def test_select_subarray_by_index_02():
    """
    Test valid input
    """

    arr = np.array([
        [9, 8, 7, 6, 5],
        [0, 1, 2, 3, 4]]).transpose()
    arr_col_idx = 1

    total_idx_n = arr[:, arr_col_idx].max() + 1
    select_idx_n = 3

    result = select_subarray_by_index(
        arr, arr_col_idx, total_idx_n, select_idx_n)

    correct_result = np.array([
        [9, 7, 5],
        [0, 2, 4]]).transpose()

    assert np.allclose(result, correct_result)


def test_select_subarray_by_index_03():
    """
    Test valid input
    """

    arr = np.array([
        [9, 8, 7, 6, 5],
        [0, 1, 0, 4, 4]]).transpose()
    arr_col_idx = 1

    total_idx_n = arr[:, arr_col_idx].max() + 1
    select_idx_n = 2

    result = select_subarray_by_index(
        arr, arr_col_idx, total_idx_n, select_idx_n)

    correct_result = np.array([
        [9, 7, 6, 5],
        [0, 0, 4, 4]]).transpose()

    assert np.allclose(result, correct_result)


def test_select_subarray_by_index_04():
    """
    Test valid input
    """

    arr = np.array([
        [0, 1, 0, 4, 4],
        [9, 8, 7, 6, 5],
        [5, 6, 7, 8, 9]]).transpose()
    arr_col_idx = 0

    total_idx_n = arr[:, arr_col_idx].max() + 1
    select_idx_n = 2

    result = select_subarray_by_index(
        arr, arr_col_idx, total_idx_n, select_idx_n)

    correct_result = np.array([
        [0, 0, 4, 4],
        [9, 7, 6, 5],
        [5, 7, 8, 9]]).transpose()

    assert np.allclose(result, correct_result)


