
import pytest
import torch

from src.common import (
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


def test_pytorch_train_with_quantile_loss_01():
    """
    Test quantile regression with PyTorch
    """

    train_x = torch.Tensor([
        [-1], [-1], [-1], [-1], [-1], [-1],
         [0],  [0],  [0],  [0],  [0],  [0],
         [1],  [1],  [1],  [1],  [1],  [1]])

    train_y = torch.Tensor([0, 1, 2, 3, 4, 5]).reshape(-1, 1)

    model = torch.nn.Sequential(
        torch.nn.Linear(1, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 1))

    quantile = 0.2
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, maximize=False)

    epoch_n = 1000

    for _ in range(epoch_n):
        model.train()
        for i in range(3):
            batch_x = train_x[(6*i):(6*i+6)]
            batch_y = train_y

            y_pred = model(batch_x)

            loss = calculate_quantile_loss(quantile, y_pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    final_y_pred = model(train_x[0, :])

    correct_result = torch.Tensor([1])

    assert torch.allclose(final_y_pred, correct_result, atol=2e-2)


def test_pytorch_train_with_quantile_loss_02():
    """
    Test quantile regression with PyTorch
    """

    train_x = torch.Tensor([
        [-1], [-1], [-1], [-1], [-1], [-1],
         [0],  [0],  [0],  [0],  [0],  [0],
         [1],  [1],  [1],  [1],  [1],  [1]])

    train_y = torch.Tensor([0, 1, 2, 3, 4, 5]).reshape(-1, 1)

    model = torch.nn.Sequential(
        torch.nn.Linear(1, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 1))

    quantile = 0.8
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, maximize=False)

    epoch_n = 2000

    for _ in range(epoch_n):
        model.train()
        for i in range(3):
            batch_x = train_x[(6*i):(6*i+6)]
            batch_y = train_y

            y_pred = model(batch_x)

            loss = calculate_quantile_loss(quantile, y_pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    final_y_pred = model(train_x[0, :])

    correct_result = torch.Tensor([4])

    assert torch.allclose(final_y_pred, correct_result, atol=2e-2)


