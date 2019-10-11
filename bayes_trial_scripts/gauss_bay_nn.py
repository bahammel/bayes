import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from logger import Logger
from datetime import datetime
from sklearn.utils import shuffle
from tqdm import tqdm
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython import display
import pca_data_hu_ndl_bay as pca_data_hu
from torch.utils import data as data_utils
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from scipy.misc import imread

# %matplotlib inline
plt.ion()
plt.close('all')
"""
USE_GPU = torch.cuda.is_available()

device = torch.device('cuda' if USE_GPU else 'cpu')
torch.set_default_tensor_type(
    'torch.cuda.FloatTensor' if USE_GPU else 'torch.FloatTensor'
)

if USE_GPU:
    print("="*80)
    print("Model is using GPU")
    print("="*80)
    #model.cuda()
else:
    print("="*80)
    print("Model is using CPU")
    print("="*80)
"""
hu, PC, xtrain_pca, xtest_pca, Xtrain, Xtest, Ytrain, Ytest, X, Y = pca_data_hu.pca_data()
"""
train_loader = data_utils.TensorDataset(torch.from_numpy(xtrain_pca).float().to(device), torch.from_numpy(Ytrain).float().to(device))
test_loader = data_utils.TensorDataset(torch.from_numpy(xtest_pca).float().to(device), torch.from_numpy(Ytest).float().to(device))

x_data = data[:, :-1] # data[:, :-1].shape IS torch.Size([170, 2]) 170 pairs (Is Africa, Terrain Rug)
y_data = data[:, -1] #  data[:, -1].shape IS torch.Size([170]) 170 GDP values
"""

x_data = xtrain_pca
y_data = Ytrain

class NN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return output


net = NN(PC, 10, 1)

import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

log_softmax = nn.LogSoftmax(dim=1)


def model(x_data, y_data):
    
    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))
    
    outw_prior = Normal(loc=torch.zeros_like(net.out.weight), scale=torch.ones_like(net.out.weight))
    outb_prior = Normal(loc=torch.zeros_like(net.out.bias), scale=torch.ones_like(net.out.bias))
    
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,  'out.weight': outw_prior, 'out.bias': outb_prior}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()
    
    lhat = log_softmax(lifted_reg_model(x_data))
    
    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)

softplus = torch.nn.Softplus()
"""
def model(x_data, y_data):
    # weight and bias priors
    w_prior = Normal(torch.zeros(1, 2), torch.ones(1, 2)).to_event(1)
    b_prior = Normal(torch.tensor([[8.]]), torch.tensor([[1000.]])).to_event(1)
    f_prior = Normal(0., 1.)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior, 'factor': f_prior}
    scale = pyro.sample("sigma", Uniform(0., 10.))
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a nn (which also samples w and b)
    lifted_reg_model = lifted_module()
    with pyro.plate("map", len(x_data)):
        # run the nn forward on data
        prediction_mean = lifted_reg_model(x_data).squeeze(-1)
        # condition on the observed data
        pyro.sample("obs",
                    Normal(prediction_mean, scale),
                    obs=y_data)
        return prediction_mean
"""

def old_model(x_data, y_data):
    
    Scale2 = pyro.sample("sigma", Uniform(0., 10.))
    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))
    
    outw_prior = Normal(loc=torch.zeros_like(net.out.weight), scale=torch.ones_like(net.out.weight))
    outb_prior = Normal(loc=torch.zeros_like(net.out.bias), scale=torch.ones_like(net.out.bias))
    
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,  'out.weight': outw_prior, 'out.bias': outb_prior}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()
    
    prediction_mean = lifted_reg_model(x_data)

    pyro.sample("obs", Normal(prediction_mean, Scale2), obs=y_data)

softplus = torch.nn.Softplus()

def guide(x_data, y_data):
    
    # First layer weight distribution priors
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
    # First layer bias distribution priors
    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
    # Output layer weight distribution priors
    outw_mu = torch.randn_like(net.out.weight)
    outw_sigma = torch.randn_like(net.out.weight)
    outw_mu_param = pyro.param("outw_mu", outw_mu)
    outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
    outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)
    # Output layer bias distribution priors
    outb_mu = torch.randn_like(net.out.bias)
    outb_sigma = torch.randn_like(net.out.bias)
    outb_mu_param = pyro.param("outb_mu", outb_mu)
    outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
    outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
    
    lifted_module = pyro.random_module("module", net, priors)
    
    return lifted_module()

optim = Adam({"lr": 0.0001})
svi = SVI(model, guide, optim, loss=Trace_ELBO(), num_samples=1000)
#svi = SVI(model, guide, optim, ELBO())


EPOCHS = 20
BATCH_SZ = 100
loss = 0

pbar = tqdm(range(EPOCHS))

for epoch in pbar:
    Xtrain = torch.Tensor(xtrain_pca)
    Ytrain = torch.Tensor(Ytrain)
    Xtest = torch.Tensor(xtest_pca)
    Ytest = torch.Tensor(Ytest)

    train_loss = []
    #model.train()
    for batch_idx in range(len(xtrain_pca) // BATCH_SZ):
        x_batch = Xtrain[batch_idx*BATCH_SZ:(batch_idx+1)*BATCH_SZ]
        y_batch = Ytrain[batch_idx*BATCH_SZ:(batch_idx+1)*BATCH_SZ]

        # calculate the loss and take a gradient step
        loss += svi.step(Xtrain, Ytrain)
        #print("Y ", data[1])
    #normalizer_train = len(train_loader.dataset)
    normalizer_train = len(Xtrain)
    total_epoch_loss_train = loss / normalizer_train
    
    print("Epoch ", epoch, " Loss ", loss , "Total Loss ", total_epoch_loss_train)
