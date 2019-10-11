import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#from torch.utils.data.dataset import Dataset
from torch.utils import data as data_utils
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, SGD
import pyro.poutine as poutine
import random

#from logger import Logger
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle
from scipy.misc import imread
from tqdm import tqdm
import os

from IPython import display
from PIL import Image

import pca_data_2V as pca_data_hu

# enable validation (e.g. validate parameters of distributions)
assert pyro.__version__.startswith('0.3.1')
pyro.enable_validation(True)

EPOCHS = 40000

N = 5000
#hu = np.linspace(6, 14, 33)


hu, PC, xx, xtest_pca, xtrain, xtest, yy, ytest, X,Y = pca_data_hu.pca_data()

x = xx.astype('float32') 
y = yy.astype('float32') 
USE_GPU = torch.cuda.is_available()

device = torch.device('cuda' if USE_GPU else 'cpu')
torch.set_default_tensor_type(
    'torch.cuda.FloatTensor' if USE_GPU else 'torch.FloatTensor'
)
print("x.shape",x.shape)
print("y.shape",y.shape)
print("xx.shape", xx.shape)                                                                                                                       
print("xtest_pca.shape",xtest_pca.shape)                                                                                                                
print("yy.shape", yy.shape)                                                                                                                       
print("xtest_pca.shape", xtest_pca.shape)                                                                                                                
print("xtrain.shape", xtrain.shape)                                                                                                                   
print("xtest.shape", xtest.shape)                                                                                                                    
print("yy.shape", yy.shape)
print("y.shape", y.shape)                                                                                                                        


hidden_size = 100
model = torch.nn.Sequential(
    torch.nn.Linear(33, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, 2),
)

def pyromodel(x, y):
    priors = {}
    for name, par in model.named_parameters():
        priors[name] = dist.Normal(torch.zeros(*par.shape), 10 * torch.ones(*par.shape)).independent(par.dim())
        
        print("batch shape:", priors[name].batch_shape)
        print("event shape:", priors[name].event_shape)
        print("event dim:", priors[name].event_dim)    
        
    bayesian_model = pyro.random_module('bayesian_model', model, priors)
    sampled_model = bayesian_model()
    with pyro.iarange("map", len(x)):
        prediction_mean = sampled_model(x)
        pyro.sample("obs",
                    dist.Normal(prediction_mean,0.05 * torch.ones(x.size(0))),
                    obs=y)
    
softplus = torch.nn.Softplus()

def guide(x, y):
    dists = {}
    for name, par in model.named_parameters():
        loc = pyro.param(name + '.loc', torch.randn(*par.shape))
        scale = softplus(pyro.param(name + '.scale',
                                    -3.0 * torch.ones(*par.shape) + 0.05 * torch.randn(*par.shape)))
        dists[name] = dist.Normal(loc, scale).independent(par.dim())

        print("dists batch shape:", dists[name].batch_shape)
        print("dists event shape:", dists[name].event_shape)
        print("dists event dim:", dists[name].event_dim)    
        
    bayesian_model = pyro.random_module('bayesian_model', model, dists)
    return bayesian_model()

#optim = Adam({"lr": 0.05})
#svi = SVI(pyromodel, guide, optim, loss=Trace_ELBO())
AdamArgs = { 'lr': 0.05}
optimizer = torch.optim.Adam
scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': AdamArgs, 'gamma': 0.99995 })
svi = SVI(pyromodel, guide, scheduler, loss=Trace_ELBO(), num_samples=EPOCHS)

loss_hist = []
pyro.clear_param_store()
for j in range(EPOCHS):
    loss = svi.step(torch.tensor(x), torch.tensor(y)) 
    if j % 100 == 0:
        #print("[iteration %04d] loss: %.4f" % (j + 1, loss / float(N)))
        
        loss_hist.append(np.mean(loss))
        print(f"epoch {j}/{EPOCHS} :", loss_hist[-1])

plt.figure()
plt.plot(loss_hist)
plt.yscale('log')
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("Epoch loss")
        
      
print("x.shape",x.shape)
print("y.shape",y.shape)

trace = poutine.trace(pyromodel).get_trace(
    torch.tensor(x), torch.tensor(y)
)
trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
print(trace.format_shapes())
        
ys = []
amp = 1.
sig = 1.0
#xs = np.linspace(0, 5, 500, dtype='float32')
xs = torch.tensor(xtest_pca.astype('float32'))
for i in range(50):
    sampled_model = guide(None, None)
    ys += [sampled_model(xs).cpu().detach().numpy().flatten()]
ys = np.stack(ys).T

for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name).shape)

"""
plt.figure()
plt.yscale('linear')
plt.title("Training Data")
plt.xlabel("hu (mu = <10> is Y for NN)")
plt.ylabel("Intensity (X for NN)")
plt.plot(hu,x[::10,:].T)
"""
plt.figure()
plt.yscale('linear')
plt.title("Fit to mu")
plt.xlabel("1st PCA component")
plt.ylabel("mu")
plt.legend()
mean = ys.mean(1)                                          
std = ys.std(1)
#plt.scatter(xs[:,1,np.newaxis].cpu().numpy(), mean, c='g')
plt.errorbar(xs[:,1,np.newaxis].cpu().numpy(), mean, yerr=std, fmt='o', c='g')  #mean of predicted values w 1std error
plt.scatter(xs[:,1,np.newaxis].cpu().numpy(), ytest,     s=1, c='b', label='y test')

plt.scatter(xs[:,1,np.newaxis].cpu().numpy(), ys[:,0],   s=1, c='r', label='prediction')
plt.scatter(xs[:,1,np.newaxis].cpu().numpy(), ys[:,10],  s=1, c ='r', label='prediction')
plt.scatter(xs[:,1,np.newaxis].cpu().numpy(), ys[:,20],  s=1, c ='r', label='prediction')
plt.scatter(xs[:,1,np.newaxis].cpu().numpy(), ys[:,30],  s=1, c ='r', label='prediction')
plt.scatter(xs[:,1,np.newaxis].cpu().numpy(), ys[:,40],  s=1, c ='r', label='prediction')

"""
#print("x",x)
print("x[:,1].shape",x[:,1].shape)
#print("y",y)
print("y.shape",y.shape)
#print("xs",xs)
print("xs[:,1].shape",xs[:,1].shape)
#print("ys",ys)
print("ys.shape",ys.shape)
#print("xs[:, np.newaxis])",xs[:, np.newaxis])
print("xs[:,1 , np.newaxis]).shape",xs[:,1, np.newaxis].shape)

#print("torch.tensor(xs[:, np.newaxis]))",torch.tensor(xs[:, np.newaxis]))
print("torch.tensor(xs[:,1, np.newaxis])).shape",torch.tensor(xs[:,1, np.newaxis]).shape)
"""
