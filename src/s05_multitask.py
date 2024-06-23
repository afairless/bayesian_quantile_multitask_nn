#! /usr/bin/env python3

import time
import copy
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as torch_utils
from torch.utils.tensorboard.writer import SummaryWriter

torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':

    from s01_generate_data.generate_data import (
        create_data_with_parameters, 
        split_data_with_parameters,
        scale_data)

    from common import (
        print_loop_status_with_elapsed_time,
        plot_scatter_regression_with_parameters,
        calculate_quantile_loss,
        extract_data_df_columns,
        bin_y_values_by_x_bins,
        plot_distribution_by_bin,
        evaluate_bin_uniformity)
else:

    from src.s01_generate_data.generate_data import (
        create_data_with_parameters, 
        split_data_with_parameters,
        scale_data)

    from src.common import (
        print_loop_status_with_elapsed_time,
        plot_scatter_regression_with_parameters,
        calculate_quantile_loss,
        extract_data_df_columns,
        bin_y_values_by_x_bins,
        plot_distribution_by_bin,
        evaluate_bin_uniformity)


class SingleTasker(nn.Module):
    def __init__(self):

        super().__init__()
        self.l0 = nn.Linear(1, 12)
        self.l1 = nn.Linear(12, 6)
        self.l2 = nn.Linear(6, 1)

    def forward(self, x: torch.Tensor):
        x = self.l0(x)
        x = torch.relu(x)
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        return x


class MultiTasker(nn.Module):
    def __init__(self):

        super().__init__()
        self.l0 = nn.Linear(1, 12)
        self.l1 = nn.Linear(12, 6)
        self.l2_0 = nn.Linear(6, 1)
        self.l2_1 = nn.Linear(6, 1)
        self.l2_2 = nn.Linear(6, 1)
        self.l2_3 = nn.Linear(6, 1)
        self.l2_4 = nn.Linear(6, 1)
        self.l2_5 = nn.Linear(6, 1)
        self.l2_6 = nn.Linear(6, 1)
        self.l2_7 = nn.Linear(6, 1)
        self.l2_8 = nn.Linear(6, 1)

    def forward(self, x: torch.Tensor, task_id: int):
        x = self.l0(x)
        x = torch.relu(x)
        x = self.l1(x)
        x = torch.relu(x)
        if task_id == 0:
            x = self.l2_0(x)
        elif task_id == 1:
            x = self.l2_1(x)
        elif task_id == 2:
            x = self.l2_2(x)
        elif task_id == 3:
            x = self.l2_3(x)
        elif task_id == 4:
            x = self.l2_4(x)
        elif task_id == 5:
            x = self.l2_5(x)
        elif task_id == 6:
            x = self.l2_6(x)
        elif task_id == 7:
            x = self.l2_7(x)
        elif task_id == 8:
            x = self.l2_8(x)
        else:
            raise ValueError('Invalid task_id')
        return x


def extract_data_from_dataloader_batches(
    dataloader: torch_utils.DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract all predictor and response variable data from a DataLoader by 
        iterating through it
    """

    x_list = []
    y_list = []

    for batch_x, batch_y in dataloader:
        x_list.append(batch_x)
        y_list.append(batch_y)

    x = torch.concatenate(x_list, axis=0)
    y = torch.concatenate(y_list, axis=0)

    return x, y


def log_loss_to_tensorboard(
    loss: torch.Tensor, loss_name: str, writer: SummaryWriter, 
    epoch: int, loader: torch_utils.DataLoader, batch_idx: int):
    """
    Log a loss value to TensorBoard and print it to the console
    """

    print(
        f'Epoch {epoch}, '
        f'Batch {batch_idx+1}/{len(loader)}: '
        f'{loss_name}={loss.item()}')
    global_step_n = epoch * len(loader) + batch_idx
    writer.add_scalar(loss_name, loss.item(), global_step_n)


def train_model(
    quantiles: np.ndarray,
    train_x_y: torch_utils.TensorDataset, 
    valid_x_y: torch_utils.TensorDataset,
    output_path: Path) -> nn.Module:
    """
    Train a quantile regression model
    """

    tensorboard_path = output_path / 'runs'
    writer = SummaryWriter(tensorboard_path)

    model = MultiTasker()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, maximize=False)

    epoch_n = 2

    best_loss = np.inf
    best_state = None

    start_time = time.time()
    for epoch in range(epoch_n):

        i = 0
        print(f'Epoch {epoch+1}/{epoch_n}')
        print_loop_status_with_elapsed_time(epoch, 1, epoch_n, start_time)

        train_loader = torch_utils.DataLoader(
            train_x_y , shuffle=True, batch_size=64, num_workers=1)
        valid_loader = torch_utils.DataLoader(
            valid_x_y , shuffle=False, batch_size=64, num_workers=1)

        model.train()
        for i, (batch_x, batch_y) in enumerate(train_loader):

            quantile_losses = []
            for q in range(len(quantiles)):
                y_predict = model(batch_x, task_id=q).reshape(-1)
                loss = calculate_quantile_loss(quantiles[q], y_predict, batch_y)
                quantile_losses.append(loss)

            loss = quantile_losses[0]
            for l in range(1, len(quantile_losses)):
                loss += quantile_losses[l]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                log_loss_to_tensorboard(
                    loss, 'train_loss', writer, epoch, train_loader, i)

        model.eval()

        valid_x, valid_y = extract_data_from_dataloader_batches(valid_loader)

        y_predict = model(valid_x, task_id=0).reshape(-1)
        loss_0 = calculate_quantile_loss(quantiles[1], y_predict, valid_y)
        y_predict = model(valid_x, task_id=1).reshape(-1)
        loss_1 = calculate_quantile_loss(quantiles[-2], y_predict, valid_y)
        valid_loss = loss_0 + loss_1

        log_loss_to_tensorboard(
            valid_loss, 'valid_loss', writer, epoch, valid_loader, i)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())

        # delete loaders so that resources are properly released
        del train_loader
        del valid_loader



    assert isinstance(best_state, dict)
    model.load_state_dict(best_state)

    return model


def predict_line_ys_per_model(
    model: nn.Module, line_xs: np.ndarray, task_id: int) -> np.ndarray:
    """
    Calculate quantile predictions for given x-values for a model
    """

    line_xs_tensor = torch.from_numpy(line_xs).float().reshape(-1, 1)

    with torch.no_grad():
        line_ys_tensor = model(line_xs_tensor, task_id=task_id)

    line_ys = line_ys_tensor.numpy() 

    return line_ys  


def predict_line_ys(
    model: nn.Module, line_xs: np.ndarray, task_ids: list[int]) -> np.ndarray:
    """
    Calculate quantile predictions for given x-values for multiple models
    """

    line_ys_list = []
    for task_id in task_ids:
        line_ys_one_model = predict_line_ys_per_model(model, line_xs, task_id)
        line_ys_list.append(line_ys_one_model)

    line_ys = np.concatenate(line_ys_list, axis=1)

    return line_ys  


def main():

    output_path = Path.cwd() / 'output' / 's05_multitasking'
    output_path.mkdir(exist_ok=True, parents=True)

    mvn_components = create_data_with_parameters()
    data = split_data_with_parameters(mvn_components.cases_data)
    scaled_data = scale_data(
        data.train, data.valid, data.test, 
        mvn_components.predictors_column_idxs, 
        mvn_components.response_column_idx)


    ##################################################
    # 
    ##################################################


    train_n = scaled_data.train_x.shape[0]
    train_x = torch.from_numpy(scaled_data.train_x[:train_n]).float()
    train_y = torch.from_numpy(scaled_data.train_y[:train_n]).float()
    valid_x = torch.from_numpy(scaled_data.valid_x[:train_n]).float()
    valid_y = torch.from_numpy(scaled_data.valid_y[:train_n]).float()
    test_x = torch.from_numpy(scaled_data.test_x[:train_n]).float()
    test_y = torch.from_numpy(scaled_data.test_y[:train_n]).float()

    train_x_y = torch_utils.TensorDataset(train_x, train_y)
    valid_x_y = torch_utils.TensorDataset(valid_x, valid_y)

    quantiles = np.arange(0.1, 0.91, 0.1)
    task_ids = list(range(len(quantiles)))

    model = train_model(quantiles, train_x_y, valid_x_y, output_path)


    # plot scatterplot and quantile regression lines over each predictor
    ##################################################

    colnames = [f'x{i+1}' for i in range(scaled_data.test_x.shape[1])] + ['y']
    data_df = pd.DataFrame(
        np.concatenate(
            (scaled_data.test_x, 
             scaled_data.test_y.reshape(-1, 1)), 
            axis=1), 
        columns=colnames)

    x_colname = colnames[0]
    output_filename = f'model_plot_{x_colname}.png'
    output_filepath = output_path / output_filename

    plot_scatter_regression_with_parameters(
        data_df, x_colname, 'y', line_xs_n=100, 
        scatter_n=1000, scatter_n_seed=29344,
        line_ys_func=predict_line_ys, 
        output_filepath=output_filepath, model=model,
        task_ids=task_ids)



    ##################################################
    # 
    ##################################################

    x, y = extract_data_df_columns(data_df)
    y_bin_counts = bin_y_values_by_x_bins(
        x, y, 1000, line_ys_func=predict_line_ys, model=model, 
        task_ids=task_ids)

    output_filename = 'binned_quantiles_by_x_bins.png'
    output_filepath = output_path / output_filename
    plot_distribution_by_bin(y_bin_counts, output_filepath)

    output_filename = 'uniformity_summary.txt'
    output_filepath = output_path / output_filename
    evaluate_bin_uniformity(y_bin_counts, output_filepath)



if __name__ == '__main__':
    main()
