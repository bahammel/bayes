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
from pyro.distributions import Normal, Uniform
from torch.distributions import constraints
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

import pdb
import logging

import pca_data_cretin_m1_ndl_sym as pca_data_hu

# enable validation (e.g. validate parameters of distributions)
assert pyro.__version__.startswith('0.3.1')
pyro.enable_validation(True)

EPOCHS = 40000

#N = 50000
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


hidden_size = 300
model = torch.nn.Sequential(
    torch.nn.Linear(33, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, 11),
)

def pyromodel(x, y):
    priors = {}
    for name, par in model.named_parameters():
        priors[name] = dist.Normal(torch.zeros(*par.shape), 10 * torch.ones(*par.shape)).independent(par.dim())
        
        #print("batch shape:", priors[name].batch_shape)
        #print("event shape:", priors[name].event_shape)
        #print("event dim:", priors[name].event_dim)    
        
    bayesian_model = pyro.random_module('bayesian_model', model, priors)
    sampled_model = bayesian_model()
    sigma = pyro.sample('sigma', Uniform(0, 10))
    with pyro.iarange("map", len(x)):
        prediction_mean = sampled_model(x)
        logging.debug(f"prediction_mean: {prediction_mean.shape}")
        logging.debug(f"y_data: {y.shape}")

        d_dist = Normal(prediction_mean, sigma).to_event(1)

        logging.debug(f"y_data: {y.shape}")
        logging.debug(f"batch shape: {d_dist.batch_shape}")
        logging.debug(f"event shape: {d_dist.event_shape}")
        logging.debug(f"event dim: {d_dist.event_dim}")

        pyro.sample("obs",
                    d_dist,
                    obs=y)
    
softplus = torch.nn.Softplus()

def guide(x, y):
    dists = {}
    for name, par in model.named_parameters():
        loc = pyro.param(name + '.loc', torch.randn(*par.shape))
        scale = softplus(pyro.param(name + '.scale',
                                    -3.0 * torch.ones(*par.shape) + 0.05 * torch.randn(*par.shape)))
        dists[name] = dist.Normal(loc, scale).independent(par.dim())

        #print("dists batch shape:", dists[name].batch_shape)
        #print("dists event shape:", dists[name].event_shape)
        #print("dists event dim:", dists[name].event_dim)    
        
    sigma_loc = pyro.param('sigma_loc', torch.tensor(1.),
                             constraint=constraints.positive)
    sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))
    bayesian_model = pyro.random_module('bayesian_model', model, dists)
    return bayesian_model()

#optim = Adam({"lr": 0.05})
#svi = SVI(pyromodel, guide, optim, loss=Trace_ELBO())
AdamArgs = { 'lr': 0.0005}
optimizer = torch.optim.Adam
scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': AdamArgs, 'gamma': 0.999997 })
#scheduler = pyro.optim.ReduceLROnPlateau({'optimizer': optimizer, 'optim_args': AdamArgs, patience=10 })
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
        
ys =  np.zeros(shape=(1492,50,11))
#amp = 0.1
#sig = 0.5
#xs = np.linspace(0, 5, 500, dtype='float32')
xs = torch.tensor(xtest_pca.astype('float32'))

for i in range(50):
    sampled_model = guide(None, None)
    ys[:,i,:] = sampled_model(xs).cpu().detach().numpy()
#ys = np.moveaxis(ys,[0,1,2],[-1,-2,-3])

"""
[ins] In [17]: sampled_model(xs).cpu().detach().numpy()                                                      
Out[17]: 
array([[ 7.338881  ,  0.5836495 ],
       [ 6.8830543 ,  0.64003193],
       [ 8.482263  ,  1.6116515 ],
       ...,
       [ 5.8785515 ,  2.3745306 ],
       [12.157866  ,  0.7134296 ],
       [12.468794  ,  1.4159068 ]], dtype=float32)

[ins] In [34]: ys.shape                                                                                      
Out[34]: (2500, 2)

[ins] In [35]: xs.shape                                                                                      
Out[35]: torch.Size([2500, 33])

[ins] In [36]: sampled_model(xs).cpu().detach().numpy().shape                                                
Out[36]: (2500, 2)
"""

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


mean_0 = ys.mean(1)[:,0]                                           
std_0 = ys.std(1)[:,0] 
mean_1 = ys.mean(1)[:,1] 
std_1 = ys.std(1)[:,1]          
mean_2 = ys.mean(1)[:,2] 
std_2 = ys.std(1)[:,2]          
mean_3 = ys.mean(1)[:,3] 
std_3 = ys.std(1)[:,3]          
mean_4 = ys.mean(1)[:,4] 
std_4 = ys.std(1)[:,4]          
mean_5 = ys.mean(1)[:,5] 
std_5 = ys.std(1)[:,5]          
mean_6 = ys.mean(1)[:,6] 
std_6 = ys.std(1)[:,6]          
mean_7 = ys.mean(1)[:,7] 
std_7 = ys.std(1)[:,7]          
mean_8 = ys.mean(1)[:,8] 
std_8 = ys.std(1)[:,8]          
mean_9 = ys.mean(1)[:,9] 
std_9 = ys.std(1)[:,9]          

plt.figure()
plt.yscale('linear')
plt.title("Fit to t1")
plt.xlabel("average of PCA components")
plt.ylabel("t1")
plt.errorbar(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), mean_0, yerr=std_0, fmt='o', c='g', ms = 1, elinewidth=0.1,errorevery=1, label='mean of pred')  #mean of predicted values w 1std error
plt.scatter(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), ytest[:,0],     s=5, c='b', label='y test')
plt.legend()

plt.figure()
plt.yscale('linear')
plt.title("Fit to n1")
plt.xlabel("averae of PCA components")
plt.ylabel("n1")
plt.errorbar(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), mean_1, yerr=std_1, fmt='o', c='g', ms = 1, elinewidth=0.1,errorevery=1, label='mean of pred')  #mean of predicted values w 1std error
plt.scatter(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), ytest[:,1],     s=5, c='b', label='y test')
plt.legend()

plt.figure()
plt.yscale('linear')
plt.title("Fit to mhot")
plt.xlabel("averge of PCA components")
plt.ylabel("mhot")
plt.errorbar(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), mean_2, yerr=std_2, fmt='o', c='g', ms = 1, elinewidth=0.1,errorevery=1, label='mean of pred')  #mean of predicted values w 1std error
plt.scatter(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), ytest[:,2],     s=5, c='b', label='y test')
plt.legend()

plt.figure()
plt.yscale('linear')
plt.title("Fit to mix")
plt.xlabel("averge of PCA components")
plt.ylabel("mix")
plt.errorbar(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), mean_3, yerr=std_3, fmt='o', c='g', ms = 1, elinewidth=0.1,errorevery=1, label='mean of pred')  #mean of predicted values w 1std error
plt.scatter(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), ytest[:,3],     s=5, c='b', label='y test')
plt.legend()

plt.figure()
plt.yscale('linear')
plt.title("Fit to mixhot")
plt.xlabel("averge of PCA components")
plt.ylabel("mixhot")
plt.errorbar(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), mean_4, yerr=std_4, fmt='o', c='g', ms = 1, elinewidth=0.1,errorevery=1, label='mean of pred')  #mean of predicted values w 1std error
plt.scatter(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), ytest[:,4],     s=5, c='b', label='y test')
plt.legend()

plt.figure()
plt.yscale('linear')
plt.title("Fit to n2")
plt.xlabel("averge of PCA components")
plt.ylabel("n2")
plt.errorbar(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), mean_6, yerr=std_6, fmt='o', c='g', ms = 1, elinewidth=0.1,errorevery=1, label='mean of pred')  #mean of predicted values w 1std error
plt.scatter(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), ytest[:,6],     s=5, c='b', label='y test')
plt.legend()

plt.figure()
plt.yscale('linear')
plt.title("Fit to t2")
plt.xlabel("averge of PCA components")
plt.ylabel("t2")
plt.errorbar(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), mean_7, yerr=std_7, fmt='o', c='g', ms = 1, elinewidth=0.1,errorevery=1, label='mean of pred')  #mean of predicted values w 1std error
plt.scatter(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), ytest[:,7],     s=5, c='b', label='y test')
plt.legend()

plt.figure()
plt.yscale('linear')
plt.title("Fit to n3")
plt.xlabel("averge of PCA components")
plt.ylabel("n3")
plt.errorbar(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), mean_8, yerr=std_8, fmt='o', c='g', ms = 1, elinewidth=0.1,errorevery=1, label='mean of pred')  #mean of predicted values w 1std error
plt.scatter(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), ytest[:,8],     s=5, c='b', label='y test')
plt.legend()

plt.figure()
plt.yscale('linear')
plt.title("Fit to mch")
plt.xlabel("averge of PCA components")
plt.ylabel("mch")
plt.errorbar(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), mean_9, yerr=std_9, fmt='o', c='g', ms = 1, elinewidth=0.1,errorevery=1, label='mean of pred')  #mean of predicted values w 1std error
plt.scatter(np.average(xs[:,:,np.newaxis].cpu().numpy(),1), ytest[:,9],     s=5, c='b', label='y test')
plt.legend()








plt.figure() 
plt.yscale('linear') 
plt.title("mu vs sig") 
plt.xlabel("sig") 
plt.ylabel("mu") 
plt.scatter(ytest[:,1], ytest[:,0],     s=2, c='b', label='y test') 
plt.errorbar(mean_sig, mean_mu, xerr=std_sig, yerr=std_mu, fmt='o', c='g', ms = 1, elinewidth=0.1,errorevery=1, label='mean of pred')
plt.legend()


plt.figure() 
plt.yscale('linear') 
plt.title("mu vs amp") 
plt.xlabel("amp") 
plt.ylabel("mu") 
plt.scatter(ytest[:,2], ytest[:,0],     s=2, c='b', label='y test') 
plt.errorbar(mean_amp, mean_mu, xerr=std_amp, yerr=std_mu, fmt='o', c='g', ms = 1, elinewidth=0.1,errorevery=1, label='mean of pred')
plt.legend()

plt.figure() 
plt.yscale('linear') 
plt.title("sig vs amp") 
plt.xlabel("amp") 
plt.ylabel("sig") 
plt.scatter(ytest[:,2], ytest[:,1],     s=2, c='b', label='y test') 
plt.errorbar(mean_amp, mean_sig, xerr=std_amp, yerr=std_sig, fmt='o', c='g', ms = 1, elinewidth=0.1,errorevery=1, label='mean of pred')
plt.legend()


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
