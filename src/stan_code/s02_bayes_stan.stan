data {
  int<lower=0> N;        // number of cases/data points
  int<lower=0> K;        // number of predictor variables
  matrix[N, K] x;        // matrix of predictor variables
  vector[N] y;           // outcome/response variable

  // predictor/x values at which to predict response/y values
  int<lower=0> predict_y_given_x_n;      // number of x values
  matrix[predict_y_given_x_n, K] predict_y_given_x;
}
parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> sigma;
}
model {
  alpha ~ cauchy(0, 10);
  beta ~ cauchy(0, 2.5);
  sigma ~ uniform(0, 5);
  y ~ normal(x * beta + alpha, sigma);
}
generated quantities {
  vector[predict_y_given_x_n] predicted_y_given_x;
  
  // generate response/y values at the given predictor/x values
  predicted_y_given_x = predict_y_given_x * beta + alpha + normal_rng(0, sigma);
}
