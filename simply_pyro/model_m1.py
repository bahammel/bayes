import torch.nn as nn
import torch
from pyro.distributions import Normal, Uniform
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro


class RegressionModel(nn.Module):

    def __init__(self, p):
        super().__init__()
        self.p = p  # number of features
        self.linear = nn.Linear(p, 1)

    def forward(self, x):
        return self.linear(x)


def model_fn(regression_model):

    def _model(x_data, y_data):
        # weight and bias priors
        priors = {
            'linear.weight': Normal(
                loc=torch.zeros_like(regression_model.linear.weight),
                scale=torch.ones_like(regression_model.linear.weight)
            ),
            'linear.bias': Normal(
                loc=torch.zeros_like(regression_model.linear.bias),
                scale=torch.ones_like(regression_model.linear.bias)
            ),
            'sigma': Uniform(
                loc=torch.zeros_like(regression_model.linear.bias),
                scale=2.0*torch.ones_like(regression_model.linear.bias)
            )
        }
        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module(
            "module", regression_model, priors
        )
        # sample a nn (which also samples w and b)
        lifted_reg_model = lifted_module()
        with pyro.plate("map", len(x_data)):
            # run the nn forward on data
            prediction_mean = lifted_reg_model(x_data)
            # condition on the observed data
            pyro.sample("obs",
                        Normal(prediction_mean, sigma),
                        obs=y_data)
            return prediction_mean
        # run the nn forward on data
        prediction_mean = lifted_reg_model(x_data)

        return prediction_mean

    return _model


def guide_fn(regression_model):
    softplus = torch.nn.Softplus()

    def _guide(x_data, y_data):

        # First layer weight distribution priors
        w_mu = torch.randn_like(regression_model.linear.weight)
        w_sigma = torch.randn_like(regression_model.linear.weight)
        w_mu_param = pyro.param("w_mu", w_mu)
        w_sigma_param = softplus(pyro.param("w_sigma", w_sigma))
        # First layer bias distribution priors
        b_mu = torch.randn_like(regression_model.linear.bias)
        b_sigma = torch.randn_like(regression_model.linear.bias)
        b_mu_param = pyro.param("b_mu", b_mu)
        b_sigma_param = softplus(pyro.param("b_sigma", b_sigma))

        priors = {
            'linear.weight': Normal(loc=w_mu_param, scale=w_sigma_param),
            'linear.bias': Normal(loc=b_mu_param, scale=b_sigma_param)
        }

        lifted_module = pyro.random_module("module", regression_model, priors)
        return lifted_module()

    return _guide


def get_pyro_model(return_all=False):
    regression_model = RegressionModel(p=1)
    model = model_fn(regression_model)
    guide = guide_fn(regression_model)
    AdamArgs = { 'lr': 1e-2 }
    optimizer = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': AdamArgs, 'gamma': 0.9995 })
    svi = SVI(model, guide, scheduler, loss=Trace_ELBO(), num_samples=1000)
    
    if return_all:
        return svi, model, guide
    else:
        return svi


if __name__ == '__main__':
    regression_model = RegressionModel(p=1)
    model = model_fn(regression_model)
    guide = guide_fn(regression_model)
