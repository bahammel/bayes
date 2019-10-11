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
from datetime import datetime
import os

from IPython import display
from PIL import Image

import pdb
import logging

import pca_data_2V as pca_data_hu

# enable validation (e.g. validate parameters of distributions)
#assert pyro.__version__.startswith('0.3.1')
pyro.enable_validation(True)

#logging.basicConfig(level=logging.DEBUG)

#EPOCHS = 2000
EPOCHS = 50000
#EPOCHS = 80000

#N = 50000
#hu = np.linspace(6, 14, 33)


USE_GPU = torch.cuda.is_available()

device = torch.device('cuda' if USE_GPU else 'cpu')
torch.set_default_tensor_type(
    'torch.cuda.FloatTensor' if USE_GPU else 'torch.FloatTensor'
)


if USE_GPU:
    print("=" * 80)
    print("Model is using GPU")
    print("=" * 80)


HL1_size = 100

model = torch.nn.Sequential(
    torch.nn.Linear(33, HL1_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(HL1_size, 2),
)

def pyromodel(x, y):
    priors = {}
    for name, par in model.named_parameters():
        priors[name] = dist.Normal(torch.zeros(*par.shape), 50 * torch.ones(*par.shape)).independent(par.dim())
        
        #print("batch shape:", priors[name].batch_shape)
        #print("event shape:", priors[name].event_shape)
        #print("event dim:", priors[name].event_dim)    
        
    bayesian_model = pyro.random_module('bayesian_model', model, priors)
    sampled_model = bayesian_model()
    sigma = pyro.sample('sigma', Uniform(0, 50))
    with pyro.iarange("map", len(x)):
        prediction_mean = sampled_model(x)
        logging.debug(f"prediction_mean: {prediction_mean.shape}")

        if y is not None:
            logging.debug(f"y_data: {y.shape}")

        d_dist = Normal(prediction_mean, sigma).to_event(1)

        if y is not None:
            logging.debug(f"y_data: {y.shape}")

        logging.debug(f"batch shape: {d_dist.batch_shape}")
        logging.debug(f"event shape: {d_dist.event_shape}")
        logging.debug(f"event dim: {d_dist.event_dim}")

        pyro.sample("obs",
                    d_dist,
                    obs=y)

        return prediction_mean
    
softplus = torch.nn.Softplus()

from pyro.contrib.autoguide import AutoDiagonalNormal
guide = AutoDiagonalNormal(pyromodel)
"""
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.infer.autoguide import init_to_mean
guide = AutoMultivariateNormal(pyromodel, init_loc_fn=init_to_mean)
"""
def save():
    save_model = input("save model > ")

    if save_model.lower().startswith('y'):
        experiment_id = input("Enter exp name, press return to use datetime> ")
        if not experiment_id:
            experiment_id = datetime.now().isoformat()

        if os.environ['HOSTNAME'] == 'fractal':
            SAVE_PATH = f'/hdd/bdhammel/checkpoints/bayes/{experiment_id}'
        else:
            SAVE_PATH = f'/usr/WS1/hammel1/proj/checkpoints/bayes/{experiment_id}'

        print("Saving to :", SAVE_PATH)
        pyro.get_param_store().save(SAVE_PATH + '.params')
        np.save(SAVE_PATH + '.data', [hu, PC, xx, xtest_pca, xtrain, xtest, yy, ytest, X,Y])

    else:
        print(f"input was {save_model} not saving model")


if __name__ == '__main__':

    hu, PC, xx, xtest_pca, xtrain, xtest, yy, ytest, X,Y = pca_data_hu.pca_data()

    x = xx.astype('float32') 
    y = yy.astype('float32') 

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

    #optim = Adam({"lr": 0.05})
    #Rsvi = SVI(pyromodel, guide, optim, loss=Trace_ELBO())
    #AdamArgs = { 'lr': 0.0005}
    AdamArgs = { 'lr': 0.0002}
    optimizer = torch.optim.Adam
    #scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': AdamArgs, 'gamma': 0.999993 })
    scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': AdamArgs, 'gamma': 0.9995 })
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
    plt.title('Loss')
    plt.plot(loss_hist)
    plt.yscale('log')
    plt.xlabel("step")
    plt.ylabel("Epoch loss")
            
          
    print("x.shape",x.shape)
    print("y.shape",y.shape)

    trace = poutine.trace(pyromodel).get_trace(
        torch.tensor(x), torch.tensor(y)
    )
    trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
    print(trace.format_shapes())
            
    ys =  np.zeros(shape=(12500,50,2))
    #amp = 0.1
    #sig = 0.5
    #xs = np.linspace(0, 5, 500, dtype='float32')
    xs = torch.tensor(xtest_pca.astype('float32'))

    def predict(x):
        tr = poutine.trace(guide).get_trace(x)
        return poutine.replay(pyromodel, trace=tr)(x, None)

    save()

