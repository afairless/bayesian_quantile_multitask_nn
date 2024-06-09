#! /usr/bin/env python3

import json
import pystan
import arviz as az
import numpy as np
from pathlib import Path
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import root_mean_squared_error


def main():

    input_path = Path.cwd() / 'output' / 'data'
    input_filepaths = list(input_path.glob('*.json'))

    for e in input_filepaths:

        with open(e, 'r') as json_file:
            json_obj = json.load(json_file)

        globals()[e.stem] = np.array(json_obj)

    output_path = Path.cwd() / 'output' / 's02_bayes_stan'
    output_path.mkdir(parents=True, exist_ok=True)

    # variables are assigned by 'globals' above
    #   explicate here to reduce complaints from linter
    scaled_data_train_y = scaled_data_train_y
    scaled_data_train_x = scaled_data_train_x

    y = scaled_data_train_y
    x = scaled_data_train_x
    n = x.shape[0]
    k = x.shape[1]
    
    stan_data = {'N': n, 'K': k, 'x': x, 'y': y}
    stan_filename = 's02_bayes_stan0.stan'
    stan_filepath = Path.cwd() / 'src' / stan_filename

    stan_model = pystan.StanModel(file=stan_filepath.as_posix())

    fit_model = stan_model.sampling(
       data=stan_data, iter=300, chains=2, warmup=150, thin=1, seed=708869)
    #fit_model = stan_model.sampling(
    #    data=stan_data, iter=2000, chains=4, warmup=1000, thin=1, seed=22074)
    # fit_model = stan_model.sampling(
    #     data=stan_data, iter=2000, chains=4, warmup=1000, thin=2, seed=22074)

    # all samples for all parameters, predicted values, and diagnostics
    #   number of rows = number of 'iter' in 'StanModel.sampling' call
    fit_df = fit_model.to_dataframe()

    # text summary of means, sd, se, and quantiles for parameters, n_eff, & Rhat
    fit_stansummary = fit_model.stansummary()
    output_filepath = output_path / 'summary_stansummary.txt'
    # write_list_to_text_file([fit_stansummary], output_filepath, True)





if __name__ == '__main__':
    main()
