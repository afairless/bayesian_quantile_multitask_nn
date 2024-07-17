# # Q

# ## I

# + [markdown]
'''
Oftentimes, when we're making predictions from a model, a single number -- a 
point estimate -- isn't enough.  For example, if we predict that the price of a 
house is $407,833.00, are we really absolutely sure that that is exactly what
the house will sell for, down to the penny?  Well, no, of course not.  Maybe
it'll sell for $410,000.  Or maybe just for $399,000.  That doesn't mean our 
model was "wrong", exactly, because both of those prices are fairly close to
our prediction.  Really, it means that that one-number prediction was rather
bare-bones; it didn't give us a lot of information.  We'd often like to have
much more information from our predictions, and we can get that with a Bayesian 
model that offers a full probability distribution as a prediction.  For our 
house price example above, it might say that there's a 5% chance that the house
will sell at $407,833, a 2% chance that it'll sell at $407,000, a 1% chance that
it'll sell at $406,000, and so on until all our probabilities add up to 100%.
'''
# -

# ## Full Bayesian vs. Quantile Regression:  Linear Data

# + [markdown]
'''
When we need our models to produce a full posterior distribution for our 
predictions, we can bring to bear the full suite of tools from [Bayesian 
analysis](http://www.stat.columbia.edu/~gelman/book/).  Sometimes, though, we
need shortcuts:  using the full Bayesian model may exceed our computational
limitations, or we may not have the tooling and infrastructure to readily 
productionize the model.  In such cases, [quantile regression](quantile 
regression) can be a convenient alternative.

Quantile regression lets us predict one or a multitude of chosen [quantiles](
https://en.wikipedia.org/wiki/Quantiles) (e.g., the median, the 80th percentile)
of our predicted outcome variable's distribution.  This distribution isn't 
exactly the same as the predicted posterior distribution of a full Bayesian
model, because it doesn't account for a prior.  However, in cases where the
prior is very weak or the large number of cases in our data set lets the 
likelihood overwhelm the prior's effects on the posterior, the predicted 
quantiles can be reasonably close to the corresponding positions of the 
posterior.

To illustrate this similarity, I generated some positively correlated bivariate 
data with Gaussian noise in the response variable, so that I could run some
simple regression models on it.  Below is a scatterplot of a sample of the data
along with every decile (from the 10th to the 90th) of the predicted posterior 
distribution of a full Bayesian linear model with weak priors on its parameters.

![image](./output/s03_bayes_stan_data03/quantile_plot_x1.png)

Next is a scatterplot of the same data along with every decile from quantile
regression models, where each model predicts a single decile.

![image](./output/s04_quantile_data03/quantile_plot_x1.png)

The two plots appear to be very similar.  We can juxtapose the Bayesian and
non-Bayesian predictions on a single plot, so that we can compare them more
carefully.

![image](./output/s10_results/s03_s04_quantiles_data03.png)

The two sets of predictions aren't identical, but they're quite close.  We can
also look at slices along the x-axis, so that we can see the distributions and
the two sets of predictions.  For this purpose, we group the data into 100 bins
and show just a few bins along the x-axis.

![image](./output/s10_results/s03_s04_density_by_bin_data03.png)

The density plots show the Gaussian-distributed noise along the y-axis for each
bin.  Accordingly, we can now see the distribution and its predicted deciles
together.
'''
# -

# ## Full Bayesian vs. Quantile Regression:  Curvy Data

# + [markdown]
'''
The data for our examples above was generated from a linear model, so the 
Bayesian linear regression and quantile linear regression models readily fit to 
the data very well.  Our data is often more complicated, so our first attempt 
at fitting a linear model might not work well.  Here's an example of 
Bayesian linear regression predictions on a curvier data set.

![image](./output/s03_bayes_stan_data02/quantile_plot_x1.png)

And here's the quantile regression predictions for the same data.

![image](./output/s04_quantile_data03/quantile_plot_x1.png)

And here they are superimposed.

![image](./output/s10_results/s03_s04_quantiles_data02.png)

Clearly, the linear models are having difficulties.  Noticeably, the quantile
regression models choose slopes so that the predictions of the deciles 
separate, or fan out, while one looks from low x-values (left) to high x-values
(right).  That divergence might mean that each adjacent pair of deciles 
actually does bound ~10% of the data points overall, but the decile predictions
are clearly poor conditional on *x*.  We can see this even more clearly below
where the Gaussian distributions and the decile predictions markedly diverge
for several bins.

![image](./output/s10_results/s03_s04_density_by_bin_data02.png)

Of course, this isn't surprising.  If we really want our linear models to fit
non-linear data, we should make them more flexible by, for example, adding
polynomial terms or (when we have >1 predictor) interaction terms or the like.

An alternative is to use more flexible modeling algorithms like tree ensembles 
or neural networks.  The Python package [scikit-learn](
https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html)
has some convenient methods for doing this, but I chose to create my own
implementation in PyTorch.

While we can continue to create one model per predicted quantile, it might be
more efficient for both training and inference to incorporate all the predicted
deciles into a single model.  We can do this with [multi-task learning](
https://arxiv.org/abs/1706.05098).  

Here are the results from one training run of the model:

![image](./output/s10_results/s03_s04_density_by_bin_data02.png)

Clearly, the more flexible neural network model is adhering more closely to the
deciles condtional on *x*.  We can also see this when we compare the results to
the Bayesian linear regression.

![image](./output/s10_results/s03_s06_quantiles_data02.png)

![image](./output/s10_results/s03_s06_density_by_bin_data02.png)

While the neural network results aren't perfect, they clearly track the Gaussian
distributions in the x-bins more closely than the linear results do.
'''
# -
