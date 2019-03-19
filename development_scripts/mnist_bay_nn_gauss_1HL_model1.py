import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

# %matplotlib inline
TENSORBOARD_DIR = '/usr/WS1/hammel1/proj/tensorboard/'
CHECKPOINT_DIR = '/usr/WS1/hammel1/proj/checkpoints/'

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
        output = F.torch.sigmoid(output)
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

bch_sz = 1028

training_set = DataSet(X_pca, Y, transform=False)
training_generator = DataLoader(training_set, batch_size=bch_sz, shuffle=True)

test_set = DataSet(X_pca_test, Y_test, transform=False)
test_generator = DataLoader(test_set, batch_size=bch_sz, shuffle=True)

img, lab = training_set.__getitem__(0)
print('image shape at the first row : {}'.format(img.size()))      
print('label shape at the first row : {}'.format(lab.size()))      
print(np.array(training_set).shape)                                                                                                                                                 
#print(np.array(test_set).shape)
"""
Out[20]: (10000, 2)
Out[21]: (10000, 
"""
train_loader_check = DataLoader(training_set, batch_size=bch_sz, shuffle=True)
train_iter_check = iter(train_loader_check)
print(type(train_iter_check))

images, labels = train_iter_check.next()

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))
# Break
pdb.set_trace()


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
    
    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))
    
    outw_prior = Normal(loc=torch.zeros_like(net.out.weight), scale=torch.ones_like(net.out.weight))
    outb_prior = Normal(loc=torch.zeros_like(net.out.bias), scale=torch.ones_like(net.out.bias))
    
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,  'out.weight': outw_prior, 'out.bias': outb_prior}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)
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
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
    
    lifted_module = pyro.random_module("module", net, priors)
    return lifted_module()



# Reducing Learning Rate. ReduceOnPlateau is not supported.
#This code works but loss doesn't get lower than constant LR. Perhaps gamma should be closer to 1.0?
AdamArgs = { 'lr': 1e-2 }
optimizer = torch.optim.Adam
scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': AdamArgs, 'gamma': 0.99995 })
svi = SVI(model, guide, scheduler, loss=Trace_ELBO())

"""
optimizer = Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
"""

"""
num_iterations = 1
for j in range(num_iterations): 
    print("Epoch ", j) 
    for batch_id, data in enumerate(training_generator): 
        print("batch_id", batch_id, data[1][:,-1])
"""


experiment_id = datetime.now().isoformat()
print('Logging experiment as: ', experiment_id)

logger = Logger(os.path.join(TENSORBOARD_DIR, experiment_id))

num_iterations = 10
loss = 0

for j in range(num_iterations):
    #print("Epoch ", j) 
    loss = 0
    for batch_id, data in enumerate(training_generator):
        # calculate the loss and take a gradient step
        loss += svi.step(data[0], data[1][:,-1])
        #loss += svi.step(x, y)
    normalizer_train = len(training_generator.dataset)
    total_epoch_loss_train = loss / normalizer_train
    
    print("Epoch ", j, " Loss ", total_epoch_loss_train)


torch.save(model, os.path.join(CHECKPOINT_DIR, f'{experiment_id}_latest'))

num_samples = 100
def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return mean.cpu().numpy, yhats

#This instead ???
"""
def predict(x): 
    sampled_models = [guide(None, None) for _ in range(num_samples)] 
    yhats = [model[(x)] for model in sampled_models] 
    mean = torch.mean(torch.stack(yhats), 0) 
    print(mean.cpu().numpy.shape, yhats.shape) 
    return mean.cpu().numpy, yhats 
"""


print('Prediction for unshifted spectra')
"""
correct = 0
total = 0

#This still doesn't work
for j, data in enumerate(training_generator):
    images, labels = data
    predicted = predict(images)
    total += labels.size(0)
    #correct += (predicted == labels.flatten().numpy()).sum().item()
#print("accuracy: %d %%" % (100 * correct / total))
#print(images, predicted)





labels.data.shape 
#labels.shape 

np.array(predicted).shape
"""

from functools import partial
import pandas as pd

"""
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))
"""

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

def wrapped_model(x_data, y_data):
    pyro.sample("prediction", Delta(model(x_data, y_data)))

posterior = svi.run(data[0], data[1][:,-1])


# posterior predictive distribution we can get samples from
trace_pred = TracePredictive(wrapped_model,
                             posterior,
                             num_samples=100)
post_pred = trace_pred.run(data[0], None)
post_summary = summary(post_pred, sites= ['prediction', 'obs'])
mu = post_summary["prediction"]
y = post_summary["obs"]
y.insert(0, 'true', data[1].cpu().numpy())

print("sample y data:")
print(y.head(10))

df = pd.DataFrame(y) 
nx = df.reset_index()  #insert a first row in Dataframe for index
nx = nx.values  #Convert Dataframe to array
fig = plt.figure(dpi=100, figsize=(5, 4))
plt.scatter(nx[:,0],nx[:,1], c='b') 
plt.scatter(nx[:,0],nx[:,2], c='r') 
#plt.errorbar(nx[:,0],nx[:,2], yerr=nx[:,3], fmt='o', c='r')
#plt.ylim(0.5,0.7)
plt.ylabel('mu')
plt.xlabel('sample')


test_set = DataSet(X_pca_test, Y_test, transform=False)
test_generator = DataLoader(test_set, batch_size=1028, shuffle=True)
print('Prediction when data is shifted')

"""
correct = 0
total = 0

#This still doesn't work
for j, data in enumerate(test_generator):
    images, labels = data
    predicted = predict(images)
    total += labels.size(0)
    #correct += (predicted == labels.flatten().numpy()).sum().item()
#print("accuracy: %d %%" % (100 * correct / total))
#print(images, predicted)

labels.data.shape 
#labels.shape 

np.array(predicted).shape
"""

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

def wrapped_model(x_data, y_data):
    pyro.sample("prediction", Delta(model(x_data, y_data)))

posterior = svi.run(data[0], data[1][:,-1])


# posterior predictive distribution we can get samples from
trace_pred = TracePredictive(wrapped_model,
                             posterior,
                             num_samples=100)
post_pred = trace_pred.run(data[0], None)
post_summary = summary(post_pred, sites= ['prediction', 'obs'])
mu = post_summary["prediction"]
y = post_summary["obs"]
y.insert(0, 'true', data[1].cpu().numpy())

print("sample y data:")
print(y.head(10))
df = pd.DataFrame(y) 

nx = df.reset_index()  #insert a first row in Dataframe for index
nx = nx.values  #Convert Dataframe to array
fig = plt.figure(dpi=100, figsize=(5, 4))
plt.scatter(nx[:,0],nx[:,1], c='b') 
plt.scatter(nx[:,0],nx[:,2], c='r') 
#plt.errorbar(nx[:,0],nx[:,2], yerr=nx[:,3], fmt='o', c='r')
#plt.ylim(0.5,0.7)
plt.ylabel('mu')
plt.xlabel('sample')
