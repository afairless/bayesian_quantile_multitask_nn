
import pytest
import numpy as np
from dataclasses import fields

from src.s10_results import (
    get_bin_cuts_for_regular_1d_grid,
    XYBins,
    bin_y_by_x,
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


def test_bin_y_by_x_01():
    """
    Test valid input
    """

    x = np.array([0, 1, 2, 3, 4])
    y = np.array([9, 8, 7, 6, 5])
    x_bin_centers = np.array([-1, 2, 5])

    result = bin_y_by_x(x, y, x_bin_centers)

    correct_bins = np.array([1, 2, 2, 2, 3])
    correct_result = XYBins(x, y, correct_bins)

    assert np.allclose(result.x, correct_result.x)
    assert np.allclose(result.y, correct_result.y)
    assert np.allclose(result.bin, correct_result.bin)

    # more flexible way to compare dataclass instances
    for field in fields(result):
        field_name = field.name
        result_value = getattr(result, field_name) 
        correct_result_value = getattr(correct_result, field_name) 
        assert np.allclose(result_value, correct_result_value)


def test_select_subarray_by_index_01():
    """
    Test select_idx_n > total_idx_n
    """

    arr_1 = np.array([
        [-9, -8, -7, -6, -5],
        [ 9,  8,  7,  6,  5],
        [ 0,  1,  2,  3,  4]]).transpose()
    x_y_bins = XYBins(arr_1[:, 0], arr_1[:, 1], arr_1[:, 2])

    total_idx_n = x_y_bins.bin.max() + 1
    select_idx_n = 10

    with pytest.raises(AssertionError):
        result = select_subarray_by_index(x_y_bins, total_idx_n, select_idx_n)


def test_select_subarray_by_index_02():
    """
    Test valid input
    """

    arr_1 = np.array([
        [-9, -8, -7, -6, -5],
        [ 9,  8,  7,  6,  5],
        [ 0,  1,  2,  3,  4]]).transpose()
    x_y_bins = XYBins(arr_1[:, 0], arr_1[:, 1], arr_1[:, 2])

    total_idx_n = x_y_bins.bin.max() + 1
    select_idx_n = 3

    result = select_subarray_by_index(x_y_bins, total_idx_n, select_idx_n)

    arr_2 = np.array([
        [-9, -7, -5],
        [ 9,  7,  5],
        [ 0,  2,  4]]).transpose()
    correct_result = XYBins(arr_2[:, 0], arr_2[:, 1], arr_2[:, 2])

    for field in fields(result):
        field_name = field.name
        result_value = getattr(result, field_name) 
        correct_result_value = getattr(correct_result, field_name) 
        assert np.allclose(result_value, correct_result_value)


def test_select_subarray_by_index_03():
    """
    Test valid input
    """

    arr_1 = np.array([
        [-9, -8, -7, -6, -5],
        [ 9,  8,  7,  6,  5],
        [ 0,  1,  0,  4,  4]]).transpose()
    x_y_bins = XYBins(arr_1[:, 0], arr_1[:, 1], arr_1[:, 2])

    total_idx_n = x_y_bins.bin.max() + 1
    select_idx_n = 2

    result = select_subarray_by_index(x_y_bins, total_idx_n, select_idx_n)

    arr_2 = np.array([
        [-9, -7, -6, -5],
        [ 9,  7,  6,  5],
        [ 0,  0,  4,  4]]).transpose()
    correct_result = XYBins(arr_2[:, 0], arr_2[:, 1], arr_2[:, 2])

    for field in fields(result):
        field_name = field.name
        result_value = getattr(result, field_name) 
        correct_result_value = getattr(correct_result, field_name) 
        assert np.allclose(result_value, correct_result_value)


def test_select_subarray_by_index_04():
    """
    Test valid input
    """

    arr_1 = np.array([
        [0, 1, 0, 4, 4],
        [9, 8, 7, 6, 5],
        [5, 6, 7, 8, 9]]).transpose()
    x_y_bins = XYBins(arr_1[:, 1], arr_1[:, 2], arr_1[:, 0])

    total_idx_n = x_y_bins.bin.max() + 1
    select_idx_n = 2

    result = select_subarray_by_index(x_y_bins, total_idx_n, select_idx_n)

    arr_2 = np.array([
        [0, 0, 4, 4],
        [9, 7, 6, 5],
        [5, 7, 8, 9]]).transpose()
    correct_result = XYBins(arr_2[:, 1], arr_2[:, 2], arr_2[:, 0])

    for field in fields(result):
        field_name = field.name
        result_value = getattr(result, field_name) 
        correct_result_value = getattr(correct_result, field_name) 
        assert np.allclose(result_value, correct_result_value)


def test_select_subarray_by_index_05():
    """
    Test valid input
    """

    arr_1 = np.array([
        [-9, -8, -7, -6, -5],
        [ 9,  8,  7,  6,  5],
        [ 0,  1,  2,  3,  4]]).transpose()
    x_y_bins = XYBins(arr_1[:, 0], arr_1[:, 1], arr_1[:, 2])

    total_idx_n = x_y_bins.bin.max() + 1
    select_idx_n = 2

    result = select_subarray_by_index(x_y_bins, total_idx_n, select_idx_n)

    arr_2 = np.array([
        [-9, -5],
        [ 9,  5],
        [ 0,  4]]).transpose()
    correct_result = XYBins(arr_2[:, 0], arr_2[:, 1], arr_2[:, 2])

    for field in fields(result):
        field_name = field.name
        result_value = getattr(result, field_name) 
        correct_result_value = getattr(correct_result, field_name) 
        assert np.allclose(result_value, correct_result_value)


def test_select_subarray_by_index_06():
    """
    Test valid input
    """

    arr_1 = np.array([
        [-9, -8, -7, -6, -5],
        [ 9,  8,  7,  6,  5],
        [ 0,  1,  0,  4,  4]]).transpose()
    x_y_bins = XYBins(arr_1[:, 0], arr_1[:, 1], arr_1[:, 2])

    total_idx_n = x_y_bins.bin.max() + 1
    select_idx_n = 0

    result = select_subarray_by_index(x_y_bins, total_idx_n, select_idx_n)

    arr_2 = np.array([
        [],
        [],
        []]).transpose()
    correct_result = XYBins(arr_2[:, 0], arr_2[:, 1], arr_2[:, 2])

    for field in fields(result):
        field_name = field.name
        result_value = getattr(result, field_name) 
        correct_result_value = getattr(correct_result, field_name) 
        assert np.allclose(result_value, correct_result_value)


