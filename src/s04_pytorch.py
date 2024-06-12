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

from src.s01_generate_data.generate_data import (
    create_data_with_parameters, 
    split_data_with_parameters,
    scale_data)

from src.utilities import (
    write_list_to_text_file,
    print_loop_status_with_elapsed_time,
    plot_scatter_regression_with_parameters)


def predict_line_ys(model: nn.Module, line_xs: np.ndarray) -> np.ndarray:

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

    loss_fn = nn.MSELoss()
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
    batch_size = 32
    batch_n = train_x.shape[0] // batch_size
    batch_idxs = torch.arange(0, len(train_x), batch_size)

    best_metric = np.inf
    best_weights = None
    loss_log = []

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
            loss = loss_fn(y_pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        valid_y_pred = model(valid_x)
        valid_loss = loss_fn(valid_y_pred, valid_y)
        loss_log.append(valid_loss.item())
        if valid_loss < best_metric:
            best_metric = valid_loss
            best_weights = copy.deepcopy(model.state_dict())
            print('best_metric', best_metric)



    assert isinstance(best_weights, dict)
    model.load_state_dict(best_weights)

    # predict_n = 4
    # with torch.no_grad():
    #     test_y_pred = model(test_x[:predict_n])
    #     test_loss = loss_fn(test_y_pred, test_y[:predict_n].reshape(-1, 1))





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
