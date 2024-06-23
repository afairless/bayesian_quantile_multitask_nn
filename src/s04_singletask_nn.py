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
        create_data_01_with_parameters, 
        split_data_with_parameters,
        scale_data)

    from common import (
        print_loop_status_with_elapsed_time,
        plot_scatter_regression_with_parameters,
        extract_data_from_dataloader_batches,
        log_loss_to_tensorboard,
        calculate_quantile_loss,
        extract_data_df_columns,
        bin_y_values_by_x_bins,
        plot_distribution_by_bin,
        evaluate_bin_uniformity)
else:

    from src.s01_generate_data.generate_data import (
        create_data_01_with_parameters, 
        split_data_with_parameters,
        scale_data)

    from src.common import (
        print_loop_status_with_elapsed_time,
        plot_scatter_regression_with_parameters,
        extract_data_from_dataloader_batches,
        log_loss_to_tensorboard,
        calculate_quantile_loss,
        extract_data_df_columns,
        bin_y_values_by_x_bins,
        plot_distribution_by_bin,
        evaluate_bin_uniformity)


def train_model(
    quantile: float,
    train_x_y: torch_utils.TensorDataset, 
    valid_x_y: torch_utils.TensorDataset,
    output_path: Path) -> nn.Module:
    """
    Train a quantile regression model
    """

    tensorboard_path = output_path / 'runs'
    writer = SummaryWriter(tensorboard_path)

    model = nn.Sequential(
        nn.Linear(1, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1))

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

            y_predict = model(batch_x).reshape(-1)
            loss = calculate_quantile_loss(quantile, y_predict, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                log_loss_to_tensorboard(
                    loss, 'train_loss', writer, epoch, train_loader, i)

        model.eval()

        valid_x, valid_y = extract_data_from_dataloader_batches(valid_loader)
        valid_y_predict = model(valid_x).reshape(-1)

        valid_loss = calculate_quantile_loss(quantile, valid_y_predict, valid_y)

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
    model: nn.Module, line_xs: np.ndarray) -> np.ndarray:
    """
    Calculate quantile predictions for given x-values for a model
    """

    line_xs_tensor = torch.from_numpy(line_xs).float().reshape(-1, 1)

    with torch.no_grad():
        line_ys_tensor = model(line_xs_tensor)

    line_ys = line_ys_tensor.numpy() 

    return line_ys  


def predict_line_ys(models: list[nn.Module], line_xs: np.ndarray) -> np.ndarray:
    """
    Calculate quantile predictions for given x-values for multiple models
    """

    line_ys_list = []
    for model in models:
        line_ys_one_model = predict_line_ys_per_model(model, line_xs)
        line_ys_list.append(line_ys_one_model)

    line_ys = np.concatenate(line_ys_list, axis=1)

    return line_ys  


def main():

    output_path = Path.cwd() / 'output' / 's04_singletask_nn'
    output_path.mkdir(exist_ok=True, parents=True)

    mvn_components = create_data_01_with_parameters()
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

    models = []
    for quantile in quantiles:
        print(f'\nQuantile: {round(quantile, 2)}')
        model = train_model(quantile, train_x_y, valid_x_y, output_path)
        models.append(model)



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
        output_filepath=output_filepath, models=models)



    ##################################################
    # 
    ##################################################

    x, y = extract_data_df_columns(data_df)
    y_bin_counts = bin_y_values_by_x_bins(
        x, y, 1000, line_ys_func=predict_line_ys, models=models)

    output_filename = 'binned_quantiles_by_x_bins.png'
    output_filepath = output_path / output_filename
    plot_distribution_by_bin(y_bin_counts, output_filepath)

    output_filename = 'uniformity_summary.txt'
    output_filepath = output_path / output_filename
    evaluate_bin_uniformity(y_bin_counts, output_filepath)



if __name__ == '__main__':
    main()
