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

import pca_data_1V as pca_data_hu

# enable validation (e.g. validate parameters of distributions)
#assert pyro.__version__.startswith('0.3.1')
pyro.enable_validation(True)

#logging.basicConfig(level=logging.DEBUG)

#EPOCHS = 2000
EPOCHS = 20000
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
    torch.nn.Linear(HL1_size, 1),
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
    sigma = pyro.sample('sigma', Uniform(0, 5000))
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

"""
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
    sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.1)))
    bayesian_model = pyro.random_module('bayesian_model', model, dists)
    return bayesian_model()

"""
from pyro.contrib.autoguide import AutoDiagonalNormal
guide = AutoDiagonalNormal(pyromodel)


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
    AdamArgs = { 'lr': 0.005}
    optimizer = torch.optim.Adam
    #scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': AdamArgs, 'gamma': 0.99995 })
    scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': AdamArgs, 'gamma': 0.9995 })
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
            
    ys =  np.zeros(shape=(12500,50,1))
    #amp = 0.1
    #sig = 0.5
    #xs = np.linspace(0, 5, 500, dtype='float32')
    xs = torch.tensor(xtest_pca.astype('float32'))

    def predict(x):
        tr = poutine.trace(guide).get_trace(x)
        return poutine.replay(pyromodel, trace=tr)(x, None)

    save()
    '''
    
    for i in range(50):
        ys[:,i,:] = predict(xs).cpu().detach().numpy()

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name).shape)

    mean_0 = ys.mean(1)[:,0]                                           
    std_0 = ys.std(1)[:,0] 
    """
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
    """
    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title('Fit')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel("t1_test")
    plt.ylabel("t1_pred")
    plt.errorbar(ytest[:,0], mean_0, yerr=std_0, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error
    """
    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.title("Fit to n1")
    plt.xlabel("n1_test")
    plt.ylabel("n1_pred")
    plt.errorbar(ytest[:,1], mean_1, yerr=std_1, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.title("Fit to mhot")
    plt.xlabel("mhot_test")
    plt.ylabel("mhot_pred")
    plt.errorbar(ytest[:,2], mean_2, yerr=std_2, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Fit to mix")
    plt.xlabel("mix_test")
    plt.ylabel("mix_pred")
    plt.errorbar(ytest[:,3], mean_3, yerr=std_3, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Fit to mixhot")
    plt.xlabel("mixhot_test")
    plt.ylabel("mixhot_pred")
    plt.errorbar(ytest[:,4], mean_4, yerr=std_4, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.title("Fit to n2")
    plt.xlabel("n2_test")
    plt.ylabel("n2_pred")
    plt.errorbar(ytest[:,6], mean_6, yerr=std_6, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.title("Fit to t2")
    plt.xlabel("t2_test")
    plt.ylabel("t2_pred")
    plt.errorbar(ytest[:,7], mean_7, yerr=std_7, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.title("Fit to n3")
    plt.xlabel("n3_test")
    plt.ylabel("n3_pred")
    plt.errorbar(ytest[:,8], mean_8, yerr=std_8, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.title("Fit to mch")
    plt.xlabel("mch_test")
    plt.ylabel("mch_pred")
    plt.errorbar(ytest[:,9], mean_9, yerr=std_9, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error
    """
    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("t1 hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,0] - mean_0)/std_0)
    """
    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("n1 hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,1] - mean_1)/std_1)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("mhot hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,2] - mean_2)/std_2)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("mix hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,3] - mean_3)/std_3)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("mixhot hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,4] - mean_4)/std_4)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("n2 hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,6] - mean_6)/std_6)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("t2 hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,7] - mean_7)/std_7)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("n3 hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,8] - mean_8)/std_8)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("mch hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,9] - mean_9)/std_9)
    """
    save()



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
    '''
