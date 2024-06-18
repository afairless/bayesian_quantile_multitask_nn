#! /usr/bin/env python3

import time
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

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
        evaluate_bin_uniformity)


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


def train_model(
    quantile: float,
    train_x: torch.Tensor, train_y: torch.Tensor,
    valid_x: torch.Tensor, valid_y: torch.Tensor) -> nn.Module:
    """
    Train a quantile regression model
    """

    model = nn.Sequential(
        nn.Linear(1, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1))

    # loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, maximize=False)

    epoch_n = 2
    batch_size = 64
    batch_n = train_x.shape[0] // batch_size
    batch_idxs = torch.arange(0, len(train_x), batch_size)

    best_metric = np.inf
    best_weights = None
    loss_log = []

    valid_y_preds = []
    x_min = valid_x.min()
    x_max = valid_x.max()
    line_xs = np.linspace(x_min, x_max , 100).reshape(-1, 1)
    # line_xs = np.array([-1, 0, 1]).reshape(-1, 1)

    for epoch in range(epoch_n):
        print(f'Starting epoch {epoch}')
        model.train()
        start_time = time.time()
        for i in range(batch_n-1):
            print_loop_status_with_elapsed_time(
                i, batch_n//20, batch_n-1, start_time)
            batch_x = train_x[batch_idxs[i]:batch_idxs[i+1]]
            batch_y = train_y[batch_idxs[i]:batch_idxs[i+1]].reshape(-1, 1)
            y_pred = model(batch_x)
            # loss = loss_fn(y_pred, batch_y)
            loss = calculate_quantile_loss(quantile, y_pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        valid_y_pred = model(valid_x)
        line_ys = model(torch.Tensor(line_xs))
        valid_y_preds.append(line_ys)
        # valid_loss = loss_fn(valid_y_pred, valid_y)
        valid_loss = calculate_quantile_loss(quantile, valid_y_pred, valid_y)
        loss_log.append(valid_loss.item())
        if valid_loss < best_metric:
            best_metric = valid_loss
            best_weights = copy.deepcopy(model.state_dict())


    assert isinstance(best_weights, dict)
    model.load_state_dict(best_weights)

    return model


def main():

    output_path = Path.cwd() / 'output' / 's04_pytorch'
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

    valid_y = valid_y.reshape(-1, 1)

    quantiles = np.arange(0.1, 0.91, 0.1)

    models = []
    for quantile in quantiles:
        model = train_model(quantile, train_x, train_y, valid_x, valid_y)
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
    plt.bar(range(len(y_bin_counts)), y_bin_counts)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    output_filename = 'uniformity_summary.txt'
    output_filepath = output_path / output_filename
    evaluate_bin_uniformity(y_bin_counts, output_filepath)



if __name__ == '__main__':
    main()
