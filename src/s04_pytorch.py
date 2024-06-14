#! /usr/bin/env python3

import time
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from s01_generate_data.generate_data import (
    create_data_with_parameters, 
    split_data_with_parameters,
    scale_data)

from utilities import (
    write_list_to_text_file,
    print_loop_status_with_elapsed_time,
    plot_scatter_regression_with_parameters)


def calculate_quantile_loss(
    quantile: float, true_values: torch.Tensor, 
    predicted_values: torch.Tensor) -> torch.Tensor:
    """
    Calculate quantile loss between a vector of true values and a vector of 
        predicted values
    """


    # input parameter pre-checks
    ##################################################

    assert quantile > 0
    assert quantile < 1

    assert true_values.ndim == 1 or true_values.shape[1] == 1 
    assert predicted_values.ndim == 1 or predicted_values.shape[1] == 1 
    assert true_values.shape == predicted_values.shape


    # calculate loss
    ##################################################

    errors = true_values - predicted_values

    losses_1 = quantile * errors
    losses_2 = (quantile - 1) * errors
    losses = torch.max(losses_1, losses_2)

    loss = torch.mean(losses)

    return loss


def predict_line_ys(model: nn.Module, line_xs: np.ndarray) -> np.ndarray:
    """
    Calculate quantile predictions for given x-values
    """

    line_xs_tensor = torch.from_numpy(line_xs).float().reshape(-1, 1)

    with torch.no_grad():
        line_ys_tensor = model(line_xs_tensor)

    line_ys = line_ys_tensor.numpy() 

    return line_ys  


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


    model = nn.Sequential(
        nn.Linear(1, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1))

    # loss_fn = nn.MSELoss()
    quantile = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_n = scaled_data.train_x.shape[0]
    train_x = torch.from_numpy(scaled_data.train_x[:train_n]).float()
    train_y = torch.from_numpy(scaled_data.train_y[:train_n]).float()
    valid_x = torch.from_numpy(scaled_data.valid_x[:train_n]).float()
    valid_y = torch.from_numpy(scaled_data.valid_y[:train_n]).float()
    test_x = torch.from_numpy(scaled_data.test_x[:train_n]).float()
    test_y = torch.from_numpy(scaled_data.test_y[:train_n]).float()

    valid_y = valid_y.reshape(-1, 1)

    n_epochs = 2
    batch_size = 4
    batch_n = train_x.shape[0] // batch_size
    batch_idxs = torch.arange(0, len(train_x), batch_size)

    best_metric = np.inf
    best_weights = None
    loss_log = []

    # valid_y_preds = []
    # x_min = valid_x.min()
    # x_max = valid_x.max()
    # line_xs = np.linspace(x_min, x_max , 100).reshape(-1, 1)

    for epoch in range(n_epochs):
        print(f'Starting epoch {epoch}')
        model.train()
        start_time = time.time()
        for i in range(batch_n-1):
        # for i in range(10):
            print_loop_status_with_elapsed_time(i, 200, batch_n-1, start_time)
            batch_x = train_x[batch_idxs[i]:batch_idxs[i+1]]
            batch_y = train_y[batch_idxs[i]:batch_idxs[i+1]].reshape(-1, 1)
            y_pred = model(batch_x)
            # loss = loss_fn(y_pred, batch_y)
            loss = calculate_quantile_loss(quantile, y_pred, batch_y)
            # if i < 40:
            #     print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        valid_y_pred = model(valid_x)
        # line_ys = model(torch.Tensor(line_xs))
        # valid_y_preds.append(line_ys)
        # valid_loss = loss_fn(valid_y_pred, valid_y)
        valid_loss = calculate_quantile_loss(quantile, valid_y_pred, valid_y)
        loss_log.append(valid_loss.item())
        if valid_loss < best_metric:
            best_metric = valid_loss
            best_weights = copy.deepcopy(model.state_dict())
            print('best_metric', best_metric)




    assert isinstance(best_weights, dict)
    model.load_state_dict(best_weights)


    
    # plt.scatter(valid_x[:100], valid_y[:100], alpha=0.2)
    # for e in valid_y_preds:
    #     plt.plot(line_xs, e.detach().numpy(), color='black', linestyle='dotted', zorder=9)
    # plt.savefig(output_path / 'iterplot.png')
    # plt.clf()
    # plt.close()



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
        output_filepath=output_filepath, model=model)



    ##################################################
    # 
    ##################################################

    """
    output_filename = 'quantile_regression_vs_histogram.png'
    output_filepath = output_path / output_filename
    plt.hist(projection, bins=100)
    for i in intercepts:
        plt.axvline(x=i, color='black', linestyle='dotted')
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    output_filename = 'binned_quantiles.png'
    output_filepath = output_path / output_filename
    bins = np.digitize(projection, bins=intercepts)
    plt.hist(bins)
    plt.title('Ideally should be uniformly distributed')
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    decile_summary = []
    decile_summary.append('Deciles estimated by quantile regression')
    decile_summary.append(str(np.round(intercepts, 2)))
    decile_summary.append(
        'Binned decile data points by quantile regression slope')
    decile_summary.append(
        str(np.round(np.quantile(projection, quantiles), 2)))

    output_filename = 'decile_summary.txt'
    output_filepath = output_path / output_filename
    write_list_to_text_file(decile_summary, output_filepath, True)
    """



if __name__ == '__main__':
    main()
