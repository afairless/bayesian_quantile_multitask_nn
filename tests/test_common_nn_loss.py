
import torch

from src.common import (
    calculate_quantile_loss,
    )


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

    epoch_n = 3000

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

    epoch_n = 2500

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


