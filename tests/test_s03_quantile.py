
import pytest
import numpy as np

from hypothesis import given, settings
import hypothesis.strategies as st

from src.s03_quantile import (
    calculate_quantile_prediction_vectors,
    calculate_perpendicular_slope,
    calculate_angle_given_slope,
    project_matrix_to_line,
    )


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


def test_calculate_perpendicular_slope_01():
    """
    Test valid input
    """
    slope = 1
    result = calculate_perpendicular_slope(slope)
    correct_result = -1
    assert result == correct_result


def test_calculate_perpendicular_slope_02():
    """
    Test valid input
    """
    slope = -1
    result = calculate_perpendicular_slope(slope)
    correct_result = 1
    assert result == correct_result


def test_calculate_perpendicular_slope_03():
    """
    Test valid input
    """
    slope = 4
    result = calculate_perpendicular_slope(slope)
    correct_result = -0.25
    assert result == correct_result


def test_calculate_perpendicular_slope_04():
    """
    Test valid input
    """
    slope = -4
    result = calculate_perpendicular_slope(slope)
    correct_result = 0.25
    assert result == correct_result


def test_calculate_perpendicular_slope_05():
    """
    Test valid input
    """
    slope = 0.25
    result = calculate_perpendicular_slope(slope)
    correct_result = -4
    assert result == correct_result


def test_calculate_perpendicular_slope_06():
    """
    Test valid input
    """
    slope = -0.25
    result = calculate_perpendicular_slope(slope)
    correct_result = 4
    assert result == correct_result


def test_calculate_angle_given_slope_01():
    """
    Test valid input
    """
    slope = 0
    result = calculate_angle_given_slope(slope)
    correct_result = 0
    assert np.allclose(result, correct_result)


def test_calculate_angle_given_slope_02():
    """
    Test valid input with result in radians
    """
    slope = 1
    result = calculate_angle_given_slope(slope)
    correct_result = 0.785398
    assert np.allclose(result, correct_result)


def test_calculate_angle_given_slope_03():
    """
    Test with result expressed more intuitively in degrees
    """
    slope = 1
    result = calculate_angle_given_slope(slope)
    correct_result = 45
    assert np.allclose(np.degrees(result), correct_result)


def test_calculate_angle_given_slope_04():
    """
    Test valid input with result in radians
    """
    slope = -1
    result = calculate_angle_given_slope(slope)
    correct_result = -0.785398
    assert np.allclose(result, correct_result)


def test_calculate_angle_given_slope_05():
    """
    Test with result expressed more intuitively in degrees
    """
    slope = -1
    result = calculate_angle_given_slope(slope)
    correct_result = -45
    assert np.allclose(np.degrees(result), correct_result)


def test_project_matrix_to_line_01():
    """
    Test 2 dimensions and 1 angle
    Test slope = 0
    """

    a_matrix = np.array([[-1, -1], [1, 1]])
    slope = 0
    angle = calculate_angle_given_slope(slope)
    result = project_matrix_to_line(a_matrix, [angle])

    correct_result = np.array([[-1], [1]])

    assert np.allclose(result, correct_result)


def test_project_matrix_to_line_02():
    """
    Test 2 dimensions and 1 angle
    Test slope = 1
    """

    a_matrix = np.array([[0, 0], [1, 1]])
    slope = 1
    angle = calculate_angle_given_slope(slope)
    result = project_matrix_to_line(a_matrix, [angle])

    correct_result = np.array([[0], [np.sqrt(2)]])

    assert np.allclose(result, correct_result)


def test_project_matrix_to_line_03():
    """
    Test 2 dimensions and 1 angle
    Test slope = 1
    """

    a_matrix = np.array([[-1, -1], [1, 1]])
    slope = 1
    angle = calculate_angle_given_slope(slope)
    result = project_matrix_to_line(a_matrix, [angle])

    correct_result = np.array([[-np.sqrt(2)], [np.sqrt(2)]])

    assert np.allclose(result, correct_result)


def test_project_matrix_to_line_04():
    """
    Test 3 dimensions and 2 angles
    Test slope = 0
    """

    a_matrix = np.array([[-1, -1, -1], [1, 1, 1]])
    slope = 0
    angle = calculate_angle_given_slope(slope)
    angles = [angle] * (a_matrix.shape[1] - 1)
    result = project_matrix_to_line(a_matrix, angles)

    correct_result = np.array([[-1], [1]])

    assert np.allclose(result, correct_result)


@pytest.mark.skip(reason='Unsure of correct answer')
def test_project_matrix_to_line_05():
    """
    Test 3 dimensions and 2 angles
    Test slope = 1

    """

    a_matrix = np.array([[0, 0, 0], [1, 1, 1]])
    slope = 1
    angle = calculate_angle_given_slope(slope)
    angles = [angle] * (a_matrix.shape[1] - 1)
    result = project_matrix_to_line(a_matrix, angles)

    correct_result = np.array([[0], [np.sqrt(3)]])
    # correct_result = np.array([[0], [1+(np.sqrt(2)/2)]])
    # correct_result = np.array([[0], [(3*np.sqrt(2))/2]])

    assert np.allclose(result, correct_result)


@pytest.mark.skip(reason='Unsure of correct answer')
def test_project_matrix_to_line_06():
    """
    Test 3 dimensions and 2 angles
    Test slope = 1
    """

    a_matrix = np.array([[-1, -1, -1], [1, 1, 1]])
    slope = 1
    angle = calculate_angle_given_slope(slope)
    angles = [angle] * (a_matrix.shape[1] - 1)
    result = project_matrix_to_line(a_matrix, angles)

    correct_result = np.array([[-np.sqrt(3)], [np.sqrt(3)]])

    assert np.allclose(result, correct_result)


def test_project_matrix_to_line_07():
    """
    Test 4 dimensions and 3 angles
    Test slope = 0
    """

    a_matrix = np.array([[-1, -1, -1, -1], [1, 1, 1, 1]])
    slope = 0
    angle = calculate_angle_given_slope(slope)
    angles = [angle] * (a_matrix.shape[1] - 1)
    result = project_matrix_to_line(a_matrix, angles)

    correct_result = np.array([[-1], [1]])

    assert np.allclose(result, correct_result)




