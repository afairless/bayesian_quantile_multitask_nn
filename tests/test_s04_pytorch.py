
import pytest
import torch

from src.s04_pytorch import (
    calculate_quantile_loss,
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

    correct_result = torch.Tensor([0.12])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_13():
    """
    Test valid input
    """

    quantile = 0.8
    true_values = torch.Tensor([0.4])
    predicted_values = torch.Tensor([0.6])

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0.04])

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

    correct_result = torch.Tensor([0.04])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_17():
    """
    Test valid input
    """

    quantile = 0.6
    true_values = torch.Tensor([0.2, 0.4])
    predicted_values = torch.Tensor([0.6, 0.6])

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0.12])

    assert torch.allclose(result, correct_result)


def test_calculate_quantile_loss_18():
    """
    Test valid input
    """

    quantile = 0.6
    true_values = torch.Tensor([0.2, 0.4])
    predicted_values = torch.Tensor([0.8, 0.2])

    result = calculate_quantile_loss(quantile, true_values, predicted_values)

    correct_result = torch.Tensor([0.18])

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

    correct_result = torch.Tensor([0.18])

    assert torch.allclose(result, correct_result)


