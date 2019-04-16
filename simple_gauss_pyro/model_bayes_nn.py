import torch.nn as nn
import torch
import torch.nn.functional as F
from pyro.distributions import Normal
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro
import pdb
#pdb.set_trace()



class NN_Model(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(NN_Model,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output = self.fc1(x)
        #output = F.torch.sigmoid(output)
        output = F.relu(output)
        output = self.out(output)
        return output

def model_fn(nn_model):

    def _model(x_data, y_data):
        
        fc1w_prior = Normal(loc=torch.zeros_like(nn_model.fc1.weight), scale=torch.ones_like(nn_model.fc1.weight))
        fc1b_prior = Normal(loc=torch.zeros_like(nn_model.fc1.bias), scale=torch.ones_like(nn_model.fc1.bias))
        
        outw_prior = Normal(loc=torch.zeros_like(nn_model.out.weight), scale=torch.ones_like(nn_model.out.weight))
        outb_prior = Normal(loc=torch.zeros_like(nn_model.out.bias), scale=torch.ones_like(nn_model.out.bias))
        
        priors = {
            'fc1.weight': fc1w_prior, 
            'fc1.bias': fc1b_prior,  
            'out.weight': outw_prior, 
            'out.bias': outb_prior}
        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module(
            "module", nn_model, priors)
        # sample a regressor (which also samples w and b)
        lifted_reg_model = lifted_module()
        """
        lhat = log_softmax(lifted_reg_model(x_data))
        pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
        """
        # run the regressor forward conditioned on inputs
        prediction_mean = lifted_reg_model(x_data).squeeze(-1)
        pyro.sample("obs", Normal(prediction_mean, 1),
                    obs=y_data)
        return prediction_mean

    return _model

def guide_fn(nn_model):
    softplus = torch.nn.Softplus()

    def _guide(x_data, y_data):
        # First layer weight distribution priors
        fc1w_mu = torch.randn_like(nn_model.fc1.weight)
        fc1w_sigma = torch.randn_like(nn_model.fc1.weight)
        fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
        fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
        fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
        # First layer bias distribution priors
        fc1b_mu = torch.randn_like(nn_model.fc1.bias)
        fc1b_sigma = torch.randn_like(nn_model.fc1.bias)
        fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
        fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
        fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
        # Output layer weight distribution priors
        outw_mu = torch.randn_like(nn_model.out.weight)
        outw_sigma = torch.randn_like(nn_model.out.weight)
        outw_mu_param = pyro.param("outw_mu", outw_mu)
        outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
        outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)
        # Output layer bias distribution priors
        outb_mu = torch.randn_like(nn_model.out.bias)
        outb_sigma = torch.randn_like(nn_model.out.bias)
        outb_mu_param = pyro.param("outb_mu", outb_mu)
        outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
        outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
        priors = {'fc1.weight': fc1w_prior, 
                  'fc1.bias': fc1b_prior, 
                  'out.weight': outw_prior, 
                  'out.bias': outb_prior
        }

        lifted_module = pyro.random_module("module", nn_model, priors)
        return lifted_module()

    return _guide


def get_pyro_model(return_all=False):
    nn_model = NN_Model(input_size=100, hidden_size=400, output_size=1)
    model = model_fn(nn_model)
    guide = guide_fn(nn_model)
    AdamArgs = { 'lr': 1e-2 }
    optimizer = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': AdamArgs, 'gamma': 0.99995 })
    svi = SVI(model, guide, scheduler, loss=Trace_ELBO(), num_samples=1000)
    
    if return_all:
        return svi, model, guide
    else:
        return svi


if __name__ == '__main__':
    nn_model = NN_Model(input_size=100, hidden_size=400, output_size=1)
    model = model_fn(nn_model)
    guide = guide_fn(nn_model)
