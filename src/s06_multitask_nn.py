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

from torchview import draw_graph


if __name__ == '__main__':

    from s01_generate_data import (
        MultivariateNormalComponents, 
        ScaledData, 
        create_data_01_with_parameters, 
        create_data_02_with_parameters, 
        create_data_03_with_parameters, 
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
        compile_results_across_runs,
        evaluate_bin_uniformity)
else:

    from src.s01_generate_data import (
        MultivariateNormalComponents, 
        ScaledData, 
        create_data_01_with_parameters, 
        create_data_02_with_parameters, 
        create_data_03_with_parameters, 
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
        compile_results_across_runs,
        evaluate_bin_uniformity)


class SingleTasker(nn.Module):

    def __init__(
        self, 
        input_size=1, 
        hidden_size_0=12, hidden_size_1=6, 
        output_size=1):

        super().__init__()
        self.l0 = nn.Linear(input_size, hidden_size_0)
        self.l1 = nn.Linear(hidden_size_0, hidden_size_1)
        self.l2 = nn.Linear(hidden_size_1, output_size)

    def forward(self, x: torch.Tensor):
        x = self.l0(x)
        x = torch.relu(x)
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        return x


class MultiTasker(nn.Module):

    def __init__(
        self, tasks_n=9, 
        input_size=1, 
        hidden_size_0=12, hidden_size_1=6, 
        output_size=1):
        
        super().__init__()
        self.l0 = nn.Linear(input_size, hidden_size_0)
        self.l1 = nn.Linear(hidden_size_0, hidden_size_1)
        self.l2 = nn.ModuleList(
            [nn.Linear(hidden_size_1, output_size) for _ in range(tasks_n)])

    def forward(self, x: torch.Tensor, task_id: int):
        x = self.l0(x)
        x = torch.relu(x)
        x = self.l1(x)
        x = torch.relu(x)
        
        if task_id < len(self.l2):
            x = self.l2[task_id](x)
        else:
            raise ValueError('Invalid task_id')
        
        return x


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

        # added to stop linter from complaining about 'i' being unbound for call 
        #   to 'log_loss_to_tensorboard'
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

    # save diagram of neural network
    graph_filename = 'torchview_diagram'
    model_graph = draw_graph(
        model, task_id=0, input_data=batch_x, 
        save_graph=True, 
        directory=output_path.__str__(), filename=graph_filename,
        hide_module_functions=False,
        hide_inner_tensors=True)

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


def process_data(
    mvn_components: MultivariateNormalComponents, scaled_data: ScaledData, 
    output_path: Path):
    """
    Model and report results for data set
    """

    output_path.mkdir(exist_ok=True, parents=True)


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


def main():

    output_path_stem = Path.cwd() / 'output'

    filename_stem = 's06_multitask_nn_data01'
    for i in range(2):
        output_path = output_path_stem / (filename_stem + '_' + str(i))
        mvn_components = create_data_01_with_parameters()
        data = split_data_with_parameters(mvn_components.cases_data)
        scaled_data = scale_data(
            data.train, data.valid, data.test, 
            mvn_components.predictors_column_idxs, 
            mvn_components.response_column_idx)
        process_data(mvn_components, scaled_data, output_path)

    compile_results_across_runs(output_path_stem, filename_stem)


    filename_stem = 's06_multitask_nn_data02'
    for i in range(2):
        output_path = output_path_stem / (filename_stem + '_' + str(i))
        mvn_components = create_data_02_with_parameters()
        data = split_data_with_parameters(mvn_components.cases_data)
        scaled_data = scale_data(
            data.train, data.valid, data.test, 
            mvn_components.predictors_column_idxs, 
            mvn_components.response_column_idx)
        process_data(mvn_components, scaled_data, output_path)

    compile_results_across_runs(output_path_stem, filename_stem)


    filename_stem = 's06_multitask_nn_data03'
    for i in range(2):
        output_path = output_path_stem / (filename_stem + '_' + str(i))
        mvn_components = create_data_03_with_parameters()
        data = split_data_with_parameters(mvn_components.cases_data)
        scaled_data = scale_data(
            data.train, data.valid, data.test, 
            mvn_components.predictors_column_idxs, 
            mvn_components.response_column_idx)
        process_data(mvn_components, scaled_data, output_path)

    compile_results_across_runs(output_path_stem, filename_stem)


if __name__ == '__main__':
    main()
