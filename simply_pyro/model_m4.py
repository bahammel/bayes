import torch.nn as nn
import torch
from torch.distributions import constraints
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
            )
        }
        scale = pyro.sample("sigma", Uniform(0.0, 200.0))
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
                        Normal(prediction_mean, scale),
                        obs=y_data)
            return prediction_mean

    return _model



from pyro.contrib.autoguide import AutoDiagonalNormal



def get_pyro_model(return_all=False):
    regression_model = RegressionModel(p=1)
    model = model_fn(regression_model)
    guide = AutoDiagonalNormal(model)
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
