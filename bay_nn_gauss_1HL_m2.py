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

    scale = pyro.sample('sigma', Uniform(0, 200))

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
    sigma = pyro.sample("sigma", Normal(sigma_loc, torch.tensor(0.1)))
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

plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("Epoch loss")

posterior = svi.run(data_train[0], data_train[1][:,-1])

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

"""
ERROR: name 'params' is not defined
output = { 
    'guide': guide, 
    'state_dict': net.state_dict(), 
    'params': params 
} 
torch.save(output, f'{experiment_id}_output.pt')            
"""
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
    # tolLo = (1.- tol) * labels.cpu().data.flatten().numpy()
    # tolHi = (1 + tol) * labels.cpu().data.flatten().numpy()
    # accept += (tolLo.all() <= npredicted.all() <= tolHi.all())
    # correct += np.sum((tolLo <= mean_predicted) & (mean_predicted <= tolHi))
    #.sum().item()

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

def wrapped_model(x_data, y_data):
    pyro.sample("prediction", Delta(model(x_data, y_data)))


# Break
# import Ipython; Ipython.embed()

# posterior predictive distribution we can get samples from
trace_pred = TracePredictive(wrapped_model,
                             posterior,
                             num_samples=1000)
post_pred = trace_pred.run(data_train[0], None)  #inputing pca components?
post_summary = summary(post_pred, sites= ['prediction', 'obs'])
meuw = post_summary["prediction"]
y = post_summary["obs"]
meuw.insert(0, 'true', data_train[1].cpu().numpy())
y.insert(0, 'true', data_train[1].cpu().numpy())


print("sample meuw data:")
print(meuw.head(10))
#What's the difference between meuw and y? Means are the same but sigma is very different.
print("sample y data:")
print(y.head(10))

df = pd.DataFrame(meuw) 
nx = df.reset_index()  #insert a first row in Dataframe for index
nx = nx.values  #Convert Dataframe to array
fig = plt.figure(dpi=100, figsize=(5, 4))
plt.scatter(nx[::100,0],nx[::100,1], c='b') 
#plt.scatter(nx[:,0],nx[:,2], c='r') 
plt.errorbar(nx[::100,0],nx[::100,2], yerr=nx[::100,3], fmt='o', c='r')
#plt.ylim(0.5,0.7)
plt.ylabel('meuw')
plt.xlabel('sample')

# Break
import Ipython
from Ipython import embed;embed()


print('Prediction when data is shifted')

for j in range(num_iterations):
    #print("Epoch ", j) 
    for batch_id, data_test in enumerate(test_generator):   
        temp=batch_id


print('Starting Posterior')

posterior = svi.run(data_test[0], data_test[1][:,-1])

print('Finished Posterior')

correct = 0
total = 0


accept = []
tol = 0.1

#
for j, data in enumerate(test_generator):
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
plt.plot(np_labels[::10], 'o', c='b')  
plt.errorbar(range(len(y_mu[::10])), y_mu[::10], yerr=y_std[::10], fmt='o', c='r')
plt.ylabel('meuw') 
plt.xlabel('sample')     


# posterior predictive distribution we can get samples from
trace_pred = TracePredictive(wrapped_model,
                             posterior,
                             num_samples=100)
post_pred = trace_pred.run(data_test[0], None)
post_summary = summary(post_pred, sites= ['prediction', 'obs'])
meuw = post_summary["prediction"]
y = post_summary["obs"]
meuw.insert(0, 'true', data_test[1].cpu().numpy())
y.insert(0, 'true', data_test[1].cpu().numpy())

print("sample meuw data:")
print(meuw.head(10))
print("sample y data:")
print(y.head(10))
df = pd.DataFrame(meuw) 

nx = df.reset_index()  #insert a first row in Dataframe for index
nx = nx.values  #Convert Dataframe to array
fig = plt.figure(dpi=100, figsize=(5, 4))
plt.scatter(nx[:,0],nx[:,1], c='b') 
#plt.scatter(nx[:,0],nx[:,2], c='r') 
plt.errorbar(nx[:,0],nx[:,2], yerr=nx[:,3], fmt='o', c='r')
#plt.ylim(0.5,0.7)
plt.ylabel('meuw')
plt.xlabel('sample')

"""
R^2 (coefficient of determination) regression score function.
Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). 
A constant model that always predicts the expected value of y, 
disregarding the input features, would get a R^2 score of 0.0.


from sklearn.metrics import r2_score
preds = [] 
for i in range(1000): 
    sampled_reg_model = guide(None, None) 
    pred = sampled_reg_model(data_train[0]).cpu().data.numpy().flatten() 
    preds.append(pred) 

all_preds = np.stack(preds).mean(0) 
r2_score(data_train[1][:,-1].cpu(), all_preds)                                                                                                                                             

Out[23]: -0.12804690200144275     #Seems to vary between -0.1 and +0.1

all_preds.shape                                                                                                                                                                            
Out[24]: (848,)
data_train[1][:,-1].cpu().shape                                                                                                                                                            
Out[25]: torch.Size([848])

all_preds[1:30]                                                                                                                                                                            
Out[31]: 
array([6.020882 , 6.023583 , 6.0278573, 5.8618426, 6.014196 , 5.88622  ,
       5.971507 , 5.9299164, 6.025372 , 6.024828 , 6.0221314, 6.024089 ,
       5.971507 , 6.024828 , 5.9490256, 5.894002 , 6.0085297, 5.9859567,
       5.908943 , 5.844442 , 5.853278 , 5.923161 , 5.960731 , 5.9364977,
       5.994445 , 6.0225916, 5.9813895, 5.923161 , 5.9364977],
      dtype=float32)

data_train[1][:,-1].cpu()[1:30]                                                                                                                                                            
Out[32]: 
tensor([6.0184, 6.0527, 6.0404, 5.9449, 6.0110, 5.9522, 5.9841, 5.9669, 6.0257,
        6.0502, 6.0551, 6.0233, 5.9841, 6.0502, 5.9743, 5.9547, 6.0061, 5.9914,
        5.9596, 5.9400, 5.9424, 5.9645, 5.9792, 5.9694, 5.9963, 6.0208, 5.9890,
        5.9645, 5.9694], device='cpu')



*** If I do r2 score with identical values I DO get 1.0 ***
preds = [] 
for i in range(10000): 
    sampled_reg_model = guide(None, None) 
    pred = sampled_reg_model(data_train[0]).cpu().data.numpy().flatten() 
    preds.append(pred) 

all_preds = np.stack(preds).mean(0) 
r2_score(data_train[1][:,-1].cpu(), data_train[1][:,-1].cpu())                                                                                                                             
Out[34]: 1.0

preds = []
for _ in range(100): 
    guide_trace = poutine.trace(guide).get_trace(data[0], None) 
    # assuming that the original model took in data as (x, y) where y is observed 
    lifted_reg_model = poutine.replay(model, guide_trace) 
    preds.append(lifted_reg_model(data[0], None))
np.asarray(preds).shape

can't figure out how to show mean
[ins] In [46]: np.asarray(preds).shape                                                                                                                                       
Out[46]: (100,)

[ins] In [47]: np.asarray(preds)[0].shape                                                                                                                                   
Out[47]: torch.Size([848])
mean of one set (out of 100) of 848 values
[ins] In [67]: np.asarray(preds)[0].cpu().detach().numpy().mean()                                                                                                            
Out[67]: 6.187536
"""
#all_preds = np.stack(preds).mean(0)
