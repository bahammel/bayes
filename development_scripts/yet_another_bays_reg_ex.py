import os
import numpy as np
import torch
import torch.nn as nn
import scipy.special as ssp

import pyro
from pyro.distributions import Normal, Bernoulli
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
# for CI testing
smoke_test = ('CI' in os.environ)
pyro.enable_validation(True)

def build_linear_dataset(N, p=1, noise_std=0.01):
    X = np.random.rand(N, p)
    # w = 3
    w = 3 * np.ones(p)
    # b = 1
    y = np.matmul(X, w) + np.repeat(1, N) + np.random.normal(0, noise_std, size=N)
    y = y.reshape(N, 1)
    X, y = torch.tensor(X).type(torch.Tensor), torch.tensor(y).type(torch.Tensor)
    data = torch.cat((X, y), 1)
    assert data.shape == (N, p + 1)
    return data

def build_logistic_dataset(N, p=1, noise_std=0.01):
    X = np.random.randn(N, p)
    
    w = np.random.randn(p)
    w += 2 * np.sign(w)

    y = np.round(ssp.expit(np.matmul(X, w) 
                           + np.repeat(1, N) 
                           + np.random.normal(0, noise_std, size=N)))
    y = y.reshape(N, 1)
    X, y = torch.tensor(X).type(torch.Tensor), torch.tensor(y).type(torch.Tensor)
    data = torch.cat((X, y), 1)
    assert data.shape == (N, p + 1)
    return data


class RegressionModel(nn.Module):
    def __init__(self, p):
        # p = number of features
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)
        self.non_linear = nn.Sigmoid()

    def forward(self, x):
        return self.non_linear(self.linear(x))


def model(data):
    # Create unit normal priors over the parameters
    loc, scale = torch.zeros(1, p), 10 * torch.ones(1, p)
    bias_loc, bias_scale = torch.zeros(1), 10 * torch.ones(1)
    w_prior = Normal(loc, scale).independent(1)
    b_prior = Normal(bias_loc, bias_scale).independent(1)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()
    with pyro.iarange("map", N):
        x_data = data[:, :-1]
        y_data = data[:, -1]
        
        model_logits = lifted_reg_model(x_data).squeeze(-1)
        pyro.sample("obs", Bernoulli(logits=model_logits, validate_args=True), obs=y_data.squeeze())

softplus = torch.nn.Softplus()

def guide(data):
    # define our variational parameters
    w_loc = torch.randn(1, p)
    # note that we initialize our scales to be pretty narrow
    w_log_sig = torch.tensor(-3.0 * torch.ones(1, p) + 0.05 * torch.randn(1, p))
    b_loc = torch.randn(1)
    b_log_sig = torch.tensor(-3.0 * torch.ones(1) + 0.05 * torch.randn(1))
    # register learnable params in the param store
    mw_param = pyro.param("guide_mean_weight", w_loc)
    sw_param = softplus(pyro.param("guide_log_scale_weight", w_log_sig))
    mb_param = pyro.param("guide_mean_bias", b_loc)
    sb_param = softplus(pyro.param("guide_log_scale_bias", b_log_sig))
    # guide distributions for w and b
    w_dist = Normal(mw_param, sw_param).independent(1)
    b_dist = Normal(mb_param, sb_param).independent(1)
    dists = {'linear.weight': w_dist, 'linear.bias': b_dist}
    # overload the parameters in the module with random samples
    # from the guide distributions
    lifted_module = pyro.random_module("module", regression_model, dists)
    # sample a regressor (which also samples w and b)
    return lifted_module()

N, p = 500, 100
optim = Adam({"lr": 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO())
regression_model = RegressionModel(p)

pyro.clear_param_store()
data = build_logistic_dataset(N, p)
num_iterations = 10000
for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(data)
    if j % (num_iterations / 10) == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / float(N)))

for name in pyro.get_param_store().get_all_param_names():
    #print("[%s]: %.3f" % (name, pyro.param(name).data.numpy()))
    print("[%s]: %s" % (name, pyro.param(name).data.numpy()))

scorecard = 0
trainingdata = data
for index, prediction in enumerate(torch.round(sampled_reg_model(data[:,:-1]))):
    if prediction.item() == data[index,-1].item():
        scorecard += 1
print('Final Score: %s / %s or %s' %(scorecard, N, round(scorecard / N,3)))    
