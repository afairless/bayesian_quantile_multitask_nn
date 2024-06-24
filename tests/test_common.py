
import pytest
import torch
import numpy as np

from src.common import (
    calculate_quantile_loss,
    enforce_bin_monotonicity,
    )


def test_calculate_quantile_loss_01():
    """
    Test quantile > 1
    """

    quantile = 1.01
    true_values = torch.Tensor([0.5, 0.5])
    predicted_values = torch.Tensor([0.5, 0.5])

    with pytest.raises(AssertionError):
        result = calculate_quantile_loss(quantile, true_values, predicted_values)


def test_calculate_quantile_loss_02():
    """
    Test quantile < 0
    """

    quantile = -0.01
    true_values = torch.Tensor([0.5, 0.5])
    predicted_values = torch.Tensor([0.5, 0.5])

    with pytest.raises(AssertionError):
        result = calculate_quantile_loss(quantile, true_values, predicted_values)


def test_calculate_quantile_loss_03():
    """
    Test input tensor with > 1 dimension
    """

    quantile = 0.5
    true_values = torch.Tensor([[0.5, 0.5], [0.5, 0.5]])
    predicted_values = torch.Tensor([0.5, 0.5])

    with pytest.raises(AssertionError):
        result = calculate_quantile_loss(quantile, true_values, predicted_values)


def test_calculate_quantile_loss_04():
    """
    Test input tensor with > 1 dimension
    """

    quantile = 0.5
    true_values = torch.Tensor([0.5, 0.5])
    predicted_values = torch.Tensor([[0.5, 0.5], [0.5, 0.5]])

    with pytest.raises(AssertionError):
        result = calculate_quantile_loss(quantile, true_values, predicted_values)


def test_calculate_quantile_loss_05():
    """
    Test input tensors with different shapes
    """

    quantile = 0.5
    true_values = torch.Tensor([0.5, 0.5])
    predicted_values = torch.Tensor([0.5, 0.5, 0.5])

    with pytest.raises(AssertionError):
        result = calculate_quantile_loss(quantile, true_values, predicted_values)


def test_calculate_quantile_loss_06():
    """
    Test input tensors with different shapes
    """

    quantile = 0.5
    true_values = torch.Tensor([0.5, 0.5, 0.5])
    predicted_values = torch.Tensor([0.5, 0.5, 0.5, 0.5])

    with pytest.raises(AssertionError):
        result = calculate_quantile_loss(quantile, true_values, predicted_values)


def test_calculate_quantile_loss_07():
    """
    Test valid input with no loss
    """

    quantile = 0.5
    true_values = torch.Tensor([0.5])
    predicted_values = torch.Tensor([0.5])

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_08():
    """
    Test valid input with no loss
    """

    quantile = 0.5
    true_values = torch.Tensor([0.5, 0.6])
    predicted_values = torch.Tensor([0.5, 0.6])

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_09():
    """
    Test valid input with no loss
    """

    quantile = 0.5
    true_values = torch.Tensor([0.4, 0.5, 0.6])
    predicted_values = torch.Tensor([0.4, 0.5, 0.6])

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_10():
    """
    Test valid input
    """

    quantile = 0.5
    true_values = torch.Tensor([0.5])
    predicted_values = torch.Tensor([0.6])

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0.05])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_11():
    """
    Test valid input
    """

    quantile = 0.5
    true_values = torch.Tensor([0.4])
    predicted_values = torch.Tensor([0.6])

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0.1])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_12():
    """
    Test valid input
    """

    quantile = 0.4
    true_values = torch.Tensor([0.4])
    predicted_values = torch.Tensor([0.6])

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0.08])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_13():
    """
    Test valid input
    """

    quantile = 0.8
    true_values = torch.Tensor([0.4])
    predicted_values = torch.Tensor([0.6])

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0.16])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_14():
    """
    Test valid input
    """

    quantile = 0.5
    true_values = torch.Tensor([0.5, 0.5])
    predicted_values = torch.Tensor([0.6, 0.6])

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0.05])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_15():
    """
    Test valid input
    """

    quantile = 0.5
    true_values = torch.Tensor([0.4, 0.5])
    predicted_values = torch.Tensor([0.4, 0.6])

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0.025])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_16():
    """
    Test valid input
    """

    quantile = 0.2
    true_values = torch.Tensor([0.4, 0.5])
    predicted_values = torch.Tensor([0.4, 0.6])

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0.01])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_17():
    """
    Test valid input
    """

    quantile = 0.6
    true_values = torch.Tensor([0.2, 0.4])
    predicted_values = torch.Tensor([0.6, 0.6])

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0.18])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_18():
    """
    Test valid input
    """

    quantile = 0.6
    true_values = torch.Tensor([0.2, 0.4])
    predicted_values = torch.Tensor([0.8, 0.2])

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0.22])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_19():
    """
    Test valid input with 2-dimensional shape
    """

    quantile = 0.5
    true_values = torch.Tensor([0.5, 0.5]).reshape(-1, 1)
    predicted_values = torch.Tensor([0.5, 0.5]).reshape(-1, 1)

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_20():
    """
    Test valid input
    """

    quantile = 0.6
    true_values = torch.Tensor([0.2, 0.4]).reshape(-1, 1)
    predicted_values = torch.Tensor([0.8, 0.2]).reshape(-1, 1)

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0.22])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_21():
    """
    Test valid input
    """

    quantile = 0.8
    true_values = torch.Tensor([10, 12, 15, 18, 20])
    predicted_values = torch.Tensor([11, 13, 14, 16, 19])

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0.48])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_22():
    """
    Test valid input
    """

    quantile = 0.5
    true_values = torch.Tensor([1, 2])
    predicted_values = torch.Tensor([2, 1])

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0.5])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_23():
    """
    Test penalization ratio with valid input
    """

    quantile = 0.8
    true_values = torch.Tensor([5])
    predicted_values = torch.Tensor([6])
    result_1 = calculate_quantile_loss(quantile, true_values, predicted_values)

    predicted_values = torch.Tensor([4])
    result_2 = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0.25])

    assert torch.allclose(result_2 / result_1, correct_result)


def test_calculate_quantile_loss_24():
    """
    Test penalization ratio with valid input
    """

    quantile = 0.3
    true_values = torch.Tensor([5])
    predicted_values = torch.Tensor([6])
    result_1 = calculate_quantile_loss(quantile, true_values, predicted_values)

    predicted_values = torch.Tensor([4])
    result_2 = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([7/3])

    assert torch.allclose(result_2 / result_1, correct_result)


def test_enforce_bin_monotonicity_01():
    """
    Test input too short to require any changes
    """

    bin_cuts = np.array([-2, -1])
    result = enforce_bin_monotonicity(bin_cuts)

    correct_result = np.array([-2, -1])
    assert np.allclose(result, correct_result)


def test_enforce_bin_monotonicity_02():
    """
    Test monotonically increasing input requiring no changes
    """

    bin_cuts = np.array([-2, -1, 0, 1, 2])
    result = enforce_bin_monotonicity(bin_cuts)

    correct_result = np.array([-2, -1, 0, 1, 2])
    assert np.allclose(result, correct_result)


def test_enforce_bin_monotonicity_03():
    """
    Test monotonically increasing input requiring no changes
    """

    bin_cuts = np.array([-2, -2, -1, 0, 0, 1, 2])
    result = enforce_bin_monotonicity(bin_cuts)

    correct_result = np.array([-2, -2, -1, 0, 0, 1, 2])
    assert np.allclose(result, correct_result)


def test_enforce_bin_monotonicity_04():
    """
    Test monotonically decreasing input requiring no changes
    """

    bin_cuts = np.array([2, 1, 0, -1, -2])
    result = enforce_bin_monotonicity(bin_cuts)

    correct_result = np.array([2, 1, 0, -1, -2])
    assert np.allclose(result, correct_result)


def test_enforce_bin_monotonicity_05():
    """
    Test monotonically decreasing input requiring no changes
    """

    bin_cuts = np.array([2, 2, 1, 0, 0, -1, -2])
    result = enforce_bin_monotonicity(bin_cuts)

    correct_result = np.array([2, 2, 1, 0, 0, -1, -2])
    assert np.allclose(result, correct_result)


def test_enforce_bin_monotonicity_06():
    """
    Test monotonically increasing input requiring one change
    """

    bin_cuts = np.array([-2, -1, 0, -1, 2])
    result = enforce_bin_monotonicity(bin_cuts)

    correct_result = np.array([-2, -1, 0, 0, 2])
    assert np.allclose(result, correct_result, atol=1e-4)


def test_enforce_bin_monotonicity_07():
    """
    Test monotonically increasing input requiring two changes
    """

    bin_cuts = np.array([-3, -2, -3, 0, 1, -1, 3])
    result = enforce_bin_monotonicity(bin_cuts)

    correct_result = np.array([-3, -2, -2, 0, 1, 1, 3])
    assert np.allclose(result, correct_result, atol=1e-4)


def test_enforce_bin_monotonicity_08():
    """
    Test monotonically increasing input requiring two consecutive changes
    """

    bin_cuts = np.array([-3, -2, -1, 0, -1, -3, 2, 3])
    result = enforce_bin_monotonicity(bin_cuts)

    correct_result = np.array([-3, -2, -1, 0, 0, 0, 2, 3])
    assert np.allclose(result, correct_result, atol=1e-4)


def test_enforce_bin_monotonicity_09():
    """
    Test monotonically decreasing input requiring one change
    """

    bin_cuts = np.array([2, 1, 0, 1, -2])
    result = enforce_bin_monotonicity(bin_cuts)

    correct_result = np.array([2, 1, 0, 0, -2])
    assert np.allclose(result, correct_result, atol=1e-4)


def test_enforce_bin_monotonicity_10():
    """
    Test monotonically increasing input requiring two changes
    """

    bin_cuts = np.array([3, 2, 3, 0, -1, 1, -3])
    result = enforce_bin_monotonicity(bin_cuts)

    correct_result = np.array([3, 2, 2, 0, -1, -1, -3])
    assert np.allclose(result, correct_result, atol=1e-4)


def test_enforce_bin_monotonicity_11():
    """
    Test monotonically increasing input requiring two consecutive changes
    """

    bin_cuts = np.array([3, 2, 1, 0, 1, 3, -2, -3])
    result = enforce_bin_monotonicity(bin_cuts)

    correct_result = np.array([3, 2, 1, 0, 0, 0, -2, -3])
    assert np.allclose(result, correct_result, atol=1e-4)


