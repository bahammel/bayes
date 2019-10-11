import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.distributions import constraints

from logger import Logger
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython import display
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from scipy.misc import imread
import pdb
# %matplotlib inline
TENSORBOARD_DIR = '/usr/WS1/hammel1/proj/tensorboard/'
CHECKPOINT_DIR = '/usr/WS1/hammel1/proj/checkpoints/'
DATA_DIR = '/usr/WS1/hammel1/proj/data/'
experiment_id = datetime.now().isoformat()

torch.set_default_tensor_type('torch.cuda.FloatTensor')
USE_GPU = torch.cuda.is_available()
device = torch.device('cuda' if USE_GPU else 'cpu')

if USE_GPU:
    print("="*80)
    print("Model is using GPU")
    print("="*80)


class NN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output = self.fc1(x)
        #output = F.torch.sigmoid(output)
        output = F.relu(output)
        output = self.out(output)
        return output

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import time
import pca_data_hu_ndl2_train_test as pca_data_hu


class DataSet(Dataset):

    def __init__(self, X, Y, transform=False):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx].astype(np.float32)
        return torch.tensor(x, device=device), torch.tensor(y, device=device)


hu, PC, X_pca_test, X_pca, X, X_test, Y, Y_test = pca_data_hu.pca_data()
np.save(f"../data/{experiment_id}", [hu, PC, X_pca_test, X_pca, X, X_test, Y, Y_test])

bch_sz = 2048

training_set = DataSet(X_pca, Y, transform=False)
training_generator = DataLoader(training_set, batch_size=bch_sz, shuffle=True)

test_set = DataSet(X_pca_test, Y_test, transform=False)
test_generator = DataLoader(test_set, batch_size=bch_sz, shuffle=True)

img, lab = training_set.__getitem__(0)
print('PCA input shape at the first row : {}'.format(img.size()))      
print('label shape at the first row : {}'.format(lab.size()))      
print(np.array(training_set).shape)                                                                                                                                                 
#print(np.array(test_set).shape)

train_loader_check = DataLoader(training_set, batch_size=bch_sz, shuffle=True)
train_iter_check = iter(train_loader_check)
print(type(train_iter_check))

images, labels = train_iter_check.next()

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))
# Break
# import Ipython; Ipython.embed()

net = NN(PC, 100, 1)


if USE_GPU:
    net.cuda()

import pyro
from pyro.distributions import Normal, Uniform, Delta
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive
from pyro.infer.mcmc import MCMC, NUTS
import pyro.optim as optim

log_softmax = nn.LogSoftmax(dim=1)

def model(x_data, y_data):
    
    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), 
                        scale=torch.ones_like(net.fc1.weight))
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), 
                        scale=torch.ones_like(net.fc1.bias)
                       ).to_event(1)
    
    outw_prior = Normal(loc=torch.zeros_like(net.out.weight), 
                        scale=torch.ones_like(net.out.weight))
    outb_prior = Normal(loc=torch.zeros_like(net.out.bias), 
                        scale=torch.ones_like(net.out.bias)
                       ).to_event(1)
    
    priors = {
        'fc1.weight': fc1w_prior, 
        'fc1.bias': fc1b_prior,  
        'out.weight': outw_prior, 
        'out.bias': outb_prior}

    scale = pyro.sample('sigma', Uniform(0, 20))

    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module(
        "module", net, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()


    with pyro.plate("map", len(x_data)):

        # run the nn forward on data
        prediction_mean = lifted_reg_model(x_data).squeeze(-1)

        # condition on the observed data
        pyro.sample("obs",
                    Normal(prediction_mean, scale),
                    obs=y_data)

        return prediction_mean

softplus = torch.nn.Softplus()

"""
from pyro.contrib.autoguide import AutoDiagonalNormal 
guide = AutoDiagonalNormal(model)
"""
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
    sigma_loc = pyro.param('sigma_loc', torch.tensor(1.),
                           constraint=constraints.positive)
    sigma = pyro.sample("sigma", Normal(sigma_loc, torch.tensor(0.01)))
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
    
    lifted_module = pyro.random_module("module", net, priors)
    return lifted_module()



# Reducing Learning Rate.
# ReduceOnPlateau is not supported.
AdamArgs = { 'lr': 1e-2 }
optimizer = torch.optim.Adam
scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': AdamArgs, 'gamma': 0.99995 })
svi = SVI(model, guide, scheduler, loss=Trace_ELBO(), num_samples=1000)

"""

#Fixed Learning Rate
optimizer = Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
"""

print('Logging experiment as: ', experiment_id)

logger = Logger(os.path.join(TENSORBOARD_DIR, experiment_id))

num_iterations = 500
loss = 0
losses = []
for j in range(num_iterations):
    #print("Epoch ", j) 
    loss = 0
    for batch_id, data_train in enumerate(training_generator):
        # calculate the loss and take a gradient step
        loss += svi.step(data_train[0], data_train[1][:,-1])
        #loss += svi.step(x, y)
    normalizer_train = len(training_generator.dataset)
    total_epoch_loss_train = loss / normalizer_train
    
    losses.append(total_epoch_loss_train)
    print("Epoch ", j, " Loss ", total_epoch_loss_train)

fig = plt.figure(dpi=100, figsize=(5, 4))
plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("Epoch loss")

x_data = data_train[0]
y_data = data_train[1][:,-1]


posterior = svi.run(x_data, y_data)

# Break
#import Ipython; Ipython.embed()

# Save parameters
pyro.get_param_store().save(f'{experiment_id}_params.pt')

torch.save(model, os.path.join(CHECKPOINT_DIR, f'{experiment_id}_latest'))

#Save Parameters: preferred method
torch.save(net.state_dict(), f'{experiment_id}_state.pt')
#Save everything
torch.save(net,f'{experiment_id}_full.pt')

output = {  
    'model': model, 
    'guide': guide,  
    'state_dict': net.state_dict(), 
    'svi': svi, 
}  
torch.save(output, f'{experiment_id}_output.pt')              

#Print model's state_dic    
print("Model's state_dict:") 
for param_tensor in net.state_dict(): 
    print(param_tensor, "\t", net.state_dict()[param_tensor].size()) 

num_samples = 500
def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return mean.cpu().numpy()[:,0], yhats

print('Prediction for unshifted spectra')

correct = 0
total = 0


accept = []
tol = 0.1

for j, data in enumerate(training_generator):
    images, labels = data
    mean_predicted, predicted_list = predict(images)

    npredicted = np.array([_.cpu().numpy() for _ in predicted_list])[..., 0] 
    y_mu = npredicted.mean(axis=0)
    y_std = npredicted.std(axis=0)
    tolLo = y_mu - y_std
    tolHi = y_mu + y_std
    np_labels = labels.cpu().numpy()[:, 0]
    correct += np.sum((tolLo <= np_labels) & (np_labels <= tolHi))
    total += labels.size(0)

print(f"{100*correct/total:.2f}% of the labels are inside the std of the predictions")


fig = plt.figure(dpi=100, figsize=(5, 4))
plt.plot(np_labels[::100], 'o', c='b')  
plt.errorbar(range(len(y_mu[::100])), y_mu[::100], yerr=y_std[::100], fmt='o', c='r')
plt.ylabel('y_mu') 
plt.xlabel('sample')     

# print out some stats from the last data batch
exp = zip(npredicted.mean(axis=0), npredicted.std(axis=0))
for i,(m,s) in enumerate(exp): 
    print(f"{m:.3f} +/- {s:.3f}") 
    if i > 10: break

labels.data.shape 
#labels.shape 
#np.array(predicted).shape

from functools import partial
import pandas as pd

"""
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))
"""

for name, value in pyro.get_param_store().items(): 
    print(name, pyro.param(name).cpu().detach().numpy().mean()) 


get_marginal = lambda traces, sites:EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()

def summary(traces, sites):
    marginal = get_marginal(traces, sites)
    site_stats = {}
    for i in range(marginal.shape[1]):
        site_name = sites[i]
        marginal_site = pd.DataFrame(marginal[:, i]).transpose()
        describe = partial(pd.Series.describe, percentiles=[.05, 0.25, 0.5, 0.75, 0.95])
        site_stats[site_name] = marginal_site.apply(describe, axis=1) \
            [["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats


def wrapped_model_fn(model):
    def _wrapped_model(x_data, y_data):
        pyro.sample("prediction", Delta(model(x_data, y_data)))
    return _wrapped_model


def trace_summary(svi, model, x_data, y_data):

    posterior = svi.run(x_data, y_data)
    wrapped_model = wrapped_model_fn(model)

    # posterior predictive distribution we can get samples from
    trace_pred = TracePredictive(wrapped_model,
                                 posterior,
                                 num_samples=100)
    post_pred = trace_pred.run(x_data, None)
    post_summary = summary(post_pred, sites=['prediction', 'obs'])
    mu = post_summary["prediction"]
    obs = post_summary["obs"]


    #import pudb; pudb.set_trace()

    #x = x_data.cpu().numpy().ravel()
    #idx = np.argsort(x)

    y = y_data.cpu().numpy().ravel()
    idx = np.argsort(y)

    df = pd.DataFrame({
        #"x_data": x[idx],
        #"y_data": y_data.cpu().numpy().ravel()[idx],
        "Index": np.linspace(0, np.size(y), np.size(y)),
        "y_data": y[idx],
        #"obs": obs[idx],
        "mu_mean": mu["mean"][idx],
        "mu_std": mu["std"][idx],
        "mu_perc_5": mu["5%"][idx],
        "mu_perc_95": mu["95%"][idx],
        "obs_mean": obs["mean"][idx],
        "obs_std": obs["std"][idx],
        "obs_perc_5": obs["5%"][idx],
        "obs_perc_95": obs["95%"][idx],
    })

    print(df)

    plot_mu(df)
    plt.title('trace summary: mu')
    plot_obs(df)
    plt.title('trace summary: obs')

    #import pudb; pudb.set_trace()

def guide_summary(guide, x_data, y_data):
    #import pudb; pudb.set_trace()
    sampled_models = [guide(None, None) for _ in range(10000)]
    npredicted = np.asarray(
        [model(x_data).data.cpu().numpy()[:, 0] for model in sampled_models]
    )
    pred_mean = np.mean(npredicted, axis=0)
    pred_5q = np.percentile(npredicted, 5, axis=0)
    pred_95q = np.percentile(npredicted, 95, axis=0)

    #x = x_data.cpu().numpy().ravel()
    #idx = np.argsort(x)
    y = y_data.cpu().numpy().ravel()
    idx = np.argsort(y)

    df = pd.DataFrame({
        #"x_data": x[idx],
        "Index": np.linspace(0, np.size(y), np.size(y)),
        "y_data": y[idx],
        "mu_mean": pred_mean[idx],
        "mu_perc_5": pred_5q[idx],
        "mu_perc_95": pred_95q[idx],
    })
    plot_mu(df)
    plt.title('Guide summary')


def plot_mu(df):
    plt.figure()
    plt.plot(df['Index'], df['y_data'], 'o', color='C0', label='true')
    plt.plot(df['Index'], df['mu_mean'], 'o', color='C1', label='mu')
    plt.fill_between(df["Index"],
                     df["mu_perc_5"],
                     df["mu_perc_95"],
                     color='C1',
                     alpha=0.5)
    plt.legend()


def plot_obs(df):
    plt.figure()
    plt.plot(df['Index'], df['y_data'], 'o', color='C0', label='true')
    #plt.plot(df['x_data'], df['obs'], 'o', color='C5', label='obs')
    plt.plot(df['Index'], df['obs_mean'], 'o', color='C1', label='obs_mean')
    plt.fill_between(df["Index"],
                     df["obs_perc_5"],
                     df["obs_perc_95"],
                     color='C1',
                     alpha=0.5)
    plt.legend()



trace_summary(svi, model, x_data, y_data)
guide_summary(guide, x_data, y_data)

