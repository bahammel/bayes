import argparse
import matplotlib.pyplot as plt
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize  # noqa: F401

import pyro
from pyro.distributions import Bernoulli, Normal, Delta  # noqa: F401
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive, JitTrace_ELBO
from pyro.optim import Adam


"""
Bayesian Regression
Learning a function of the form:
    y = wx + b
"""


# generate toy dataset
def build_linear_dataset(N, p, noise_std=0.01):
    X = np.random.rand(N, p)
    # use random integer weights from [0, 7]
    w = np.random.randint(1, 4, size=p)
    print('w = {}'.format(w))
    y = np.matmul(X, w)  # + np.random.normal(0, noise_std, size=N)
    y = y.reshape(N, 1)
    X, y = torch.tensor(X), torch.tensor(y)
    data = torch.cat((X, y), 1)
    assert data.shape == (N, p + 1)
    return data


# NN with one linear layer
class RegressionModel(nn.Module):
    def __init__(self, p):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)

    def forward(self, x):
        # x * w + b
        return self.linear(x)


N = 100  # size of toy data
p = 2  # number of features

softplus = nn.Softplus()
regression_model = RegressionModel(p)


def model(data):
    # Create unit normal priors over the parameters
    loc = data.new_zeros(torch.Size((1, p)))
    scale = 2 * data.new_ones(torch.Size((1, p)))
    bias_loc = data.new_zeros(torch.Size((1,)))
    bias_scale = 2 * data.new_ones(torch.Size((1,)))
    w_prior = Normal(loc, scale).to_event(1)
    b_prior = Normal(bias_loc, bias_scale).to_event(1)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    with pyro.plate("map", N, subsample=data):
        x_data = data[:, :-1]
        y_data = data[:, -1]
        # run the regressor forward conditioned on inputs
        prediction_mean = lifted_reg_model(x_data).squeeze(-1)
        pyro.sample("obs", Normal(prediction_mean, 1),
                    obs=y_data)
        return prediction_mean


def guide(data):
    w_loc = torch.randn(1, p, dtype=data.dtype, device=data.device)
    w_log_sig = -3 + 0.05 * torch.randn(1, p, dtype=data.dtype, device=data.device)
    b_loc = torch.randn(1, dtype=data.dtype, device=data.device)
    b_log_sig = -3 + 0.05 * torch.randn(1, dtype=data.dtype, device=data.device)
    # register learnable params in the param store
    mw_param = pyro.param("guide_mean_weight", w_loc)
    sw_param = softplus(pyro.param("guide_log_scale_weight", w_log_sig))
    mb_param = pyro.param("guide_mean_bias", b_loc)
    sb_param = softplus(pyro.param("guide_log_scale_bias", b_log_sig))
    # gaussian guide distributions for w and b
    w_dist = Normal(mw_param, sw_param).to_event(1)
    b_dist = Normal(mb_param, sb_param).to_event(1)
    dists = {'linear.weight': w_dist, 'linear.bias': b_dist}
    # overloading the parameters in the module with random samples from the guide distributions
    lifted_module = pyro.random_module("module", regression_model, dists)
    # sample a regressor
    return lifted_module()


# get array of batch indices
def get_batch_indices(N, batch_size):
    all_batches = np.arange(0, N, batch_size)
    if all_batches[-1] != N:
        all_batches = list(all_batches) + [N]
    return all_batches


def main(args):
    pyro.clear_param_store()
    data = build_linear_dataset(N, p)
    if args.cuda:
        # make tensors and modules CUDA
        data = data.cuda()
        softplus.cuda()
        regression_model.cuda()

    # perform inference
    optim = Adam({"lr": 0.05})
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(model, guide, optim, loss=elbo)
    for j in range(args.num_epochs):
        if args.batch_size == N:
            # use the entire data set
            epoch_loss = svi.step(data)
        else:
            # mini batch
            epoch_loss = 0.0
            perm = torch.randperm(N) if not args.cuda else torch.randperm(N).cuda()
            # shuffle data
            data = data[perm]
            # get indices of each batch
            all_batches = get_batch_indices(N, args.batch_size)
            for ix, batch_start in enumerate(all_batches[:-1]):
                batch_end = all_batches[ix + 1]
                batch_data = data[batch_start: batch_end]
                epoch_loss += svi.step(batch_data)
        if j % 100 == 0:
            print("epoch avg loss {}".format(epoch_loss / float(N)))

    sum_ = analysis(svi, data)


def plot_data(x_data, y_data, title, ):
    fig = plt.figure(title)
    if fig.axes:
        ax = fig.axes[0]
    else:
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_data[:, 0], x_data[:, 1], y_data)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')


def summary(traces, sites):

    def get_marginal(traces, sites):
        return EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()

    marginal = get_marginal(traces, sites)

    site_stats = {}

    for i in range(marginal.shape[1]):
        site_name = sites[i]
        marginal_site = pd.DataFrame(marginal[:, i]).transpose()
        describe = partial(pd.Series.describe, percentiles=[.05, 0.25, 0.5, 0.75, 0.95])
        site_stats[site_name] = marginal_site.apply(describe, axis=1)[
            ["mean", "std", "5%", "25%", "50%", "75%", "95%"]
        ]

    return site_stats


def analysis(svi, data):

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))

    posterior = svi.run(data)

    def wrapped_model(data):
        pyro.sample("prediction", Delta(model(data)))

    trace_pred = TracePredictive(wrapped_model,
                                 posterior,
                                 num_samples=1000)

    post_pred = trace_pred.run(data)
    post_summary = summary(post_pred, sites=['prediction', 'obs'])
    mu = post_summary["prediction"]
    y = post_summary["obs"]
    x_data = data[:, :-1]
    y_data = data[:, -1]
    plot_data(x_data, y_data, "predictions")
    input()
    plot_data(x_data, y['mean'], "predictions")
    return post_summary


if __name__ == '__main__':
    assert pyro.__version__.startswith('0.3.1')
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int)
    parser.add_argument('-b', '--batch-size', default=N, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()
    main(args)
