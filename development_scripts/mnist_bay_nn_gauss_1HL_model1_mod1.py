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
"""
image shape at the first row : torch.Size([30] #PCA input shape)
label shape at the first row : torch.Size([1]) #label shape
"""

train_loader_check = DataLoader(training_set, batch_size=bch_sz, shuffle=True)
train_iter_check = iter(train_loader_check)
print(type(train_iter_check))

images, labels = train_iter_check.next()

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))
"""
images shape on batch size = torch.Size([2048, 30])
labels shape on batch size = torch.Size([2048, 1])
"""
# Break
#pdb.set_trace()

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



# Reducing Learning Rate.
# ReduceOnPlateau is not supported.
AdamArgs = { 'lr': 1e-2 }
optimizer = torch.optim.Adam
scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': AdamArgs, 'gamma': 0.99995 })
svi = SVI(model, guide, scheduler, loss=Trace_ELBO())

"""

#Fixed Learning Rate
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


print('Logging experiment as: ', experiment_id)

logger = Logger(os.path.join(TENSORBOARD_DIR, experiment_id))

num_iterations = 50
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

torch.save(model, os.path.join(CHECKPOINT_DIR, f'{experiment_id}_latest'))

# Break
#pdb.set_trace()

"""
#Non-preferred method:
output = { 
          'guide': guide, 
          'state_dict': NN(PC,100,1).state_dict(), 
          'params': pyro.param 
} 
torch.save(output,os.path.join(CHECKPOINT_DIR, f'{experiment_id}_params')) 
"""
#preferred method
torch.save(NN(PC,100,1).state_dict(), f'{experiment_id}_params') 
#Look at state_dict

#Print model's state_dic    
print("Model's state_dict:") 
for param_tensor in NN(PC,100,1).state_dict(): 
    print(param_tensor, "\t", NN(PC,100,1).state_dict()[param_tensor].size()) 
"""
Model's state_dict:
fc1.weight       torch.Size([100, 30])
fc1.bias         torch.Size([100])
out.weight       torch.Size([1, 100])
out.bias         torch.Size([1])
"""
num_samples = 100
def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return mean.cpu().numpy, yhats

print('Prediction for unshifted spectra')

#This instead ???
"""
def predict(x): 
    sampled_models = [guide(None, None) for _ in range(num_samples)] 
    yhats = [model[(x)] for model in sampled_models] 
    mean = torch.mean(torch.stack(yhats), 0) 
    print(mean.cpu().numpy.shape, yhats.shape) 
    return mean.cpu().numpy, yhats 
"""
correct = 0
total = 0

"""
Need to change labels.flatten() to labels.cpu().flatten()
In [66]:  labels.cpu().data                                                                                                                                                                                                                                                                                  
Out[66]: 
tensor([[6.],
        ...,
        [6.]], device='cpu')

In [67]:  labels.cpu().data.shape                                                                                                                                                                                                                                                                            
Out[67]: torch.Size([1028, 1])

In [27]: labels.cpu().data.flatten().numpy()                                                                                                                                                                                                                                                                 
Out[27]: 
array([6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
       6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
In [28]: labels.cpu().data.flatten().numpy().sum().item()                                                                                                                                                                                                                                                    
Out[28]: 4488.0
"""

accept = []
tol = 0.1

#This still doesn't work
for j, data in enumerate(training_generator):
    images, labels = data
    predicted = predict(images)
    npredicted=np.array([_.cpu().numpy() for _ in predicted[1]])[:,:,0] 
    total += labels.size(0)
    tolLo = (1.- tol) * labels.cpu().data.flatten().numpy()
    tolHi = (1 + tol) * labels.cpu().data.flatten().numpy()
    accept += (tolLo.all() <= npredicted.all() <= tolHi.all())
    #.sum().item()
#print("accuracy: %d %%" % (100 * accept / total))

#print(images, predicted)

"""
In [67]:  labels.cpu().data.shape                                                                                                                                                                                                                                                                            
Out[67]: torch.Size([1028, 1])

In [68]: tolLow.shape                                                                                                                                                                                                                                                                                        
Out[68]: (1028,)

In [69]: tolHi.shape                                                                                                                                                                                                                                                                                         
Out[69]: (1028,)

J.Field's fix

[ins] In [91]: predicted[1][0].cpu().numpy()                                                                                                                                                                                                                                                                       
Out[91]: 
array([[7.4891357],
       ...,
       [7.4891357]], dtype=float32)

[ins] In [92]: predicted[1][0].cpu().numpy().shape                                                                                                                                                                                                                                                                 
Out[92]: (1028, 1)

[ins] In [93]: predicted[0]                                                                                                                                                                                                                                                                                        
Out[93]: <function Tensor.numpy>

[ins] In [94]: np.array([_.cpu().numpy() for _ in predicted[1]]).shape                                                                                                                                                                                                                                             
Out[94]: (100, 1028, 1)

[ins] In [95]: xx=np.array([_.cpu().numpy() for _ in predicted[1]])[:,:,0]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

sa[ins] In [96]: xx.shape                                                                                                                                                                                                                                                                                            
Out[96]: (100, 1028)

[ins] In [97]: xx                                                                                                                                                                                                                                                                                                  
Out[97]: 
array([[7.4891357, 7.4891357, 7.4891357, ..., 7.4891357, 7.4891357,
        7.4891357],
       ...,
       [4.506528 , 4.506528 , 4.506528 , ..., 4.506528 , 4.506528 ,
        4.506528 ]], dtype=float32)
"""

labels.data.shape 
#labels.shape 
np.array(predicted).shape

from functools import partial
import pandas as pd


for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))

for name, value in pyro.get_param_store().items(): 
    print(name, pyro.param(name).cpu().detach().numpy().mean()) 

"""
for name, value in pyro.get_param_store().items(): 
    print(name, pyro.param(name).cpu().detach().numpy().mean())                                                                                                                                                                                        
fc1w_mu 0.020490551
fc1w_sigma 0.49763533
fc1b_mu -0.95366895
fc1b_sigma -0.8260457
outw_mu 0.052733693
outw_sigma -0.46032602
outb_mu 1.2478061
outb_sigma -3.6989684
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




posterior = svi.run(data_train[0], data_train[1][:,-1])

# Break
#pdb.set_trace()

# posterior predictive distribution we can get samples from
trace_pred = TracePredictive(wrapped_model,
                             posterior,
                             num_samples=100)
post_pred = trace_pred.run(data_train[0], None)  #inputing pca components?
post_summary = summary(post_pred, sites= ['prediction', 'obs'])
mu = post_summary["prediction"]
y = post_summary["obs"]
mu.insert(0, 'true', data_train[1].cpu().numpy())
y.insert(0, 'true', data_train[1].cpu().numpy())


print("sample mu data:")
print(mu.head(10))
#What's the difference between mu and y? Means are the same but sigma is very different.
print("sample y data:")
print(y.head(10))

df = pd.DataFrame(mu) 
nx = df.reset_index()  #insert a first row in Dataframe for index
nx = nx.values  #Convert Dataframe to array
fig = plt.figure(dpi=100, figsize=(5, 4))
plt.scatter(nx[:,0],nx[:,1], c='b') 
#plt.scatter(nx[:,0],nx[:,2], c='r') 
plt.errorbar(nx[:,0],nx[:,2], yerr=nx[:,3], fmt='o', c='r')
#plt.ylim(0.5,0.7)
plt.ylabel('mu')
plt.xlabel('sample')

# Break
#pdb.set_trace()

print('Prediction when data is shifted')

for j in range(num_iterations):
    #print("Epoch ", j) 
    for batch_id, data_test in enumerate(test_generator):   
        temp=batch_id



posterior = svi.run(data_test[0], data_test[1][:,-1])


# posterior predictive distribution we can get samples from
trace_pred = TracePredictive(wrapped_model,
                             posterior,
                             num_samples=100)
post_pred = trace_pred.run(data_test[0], None)
post_summary = summary(post_pred, sites= ['prediction', 'obs'])
mu = post_summary["prediction"]
y = post_summary["obs"]
mu.insert(0, 'true', data_test[1].cpu().numpy())
y.insert(0, 'true', data_test[1].cpu().numpy())

print("sample mu data:")
print(mu.head(10))
print("sample y data:")
print(y.head(10))
df = pd.DataFrame(mu) 

nx = df.reset_index()  #insert a first row in Dataframe for index
nx = nx.values  #Convert Dataframe to array
fig = plt.figure(dpi=100, figsize=(5, 4))
plt.scatter(nx[:,0],nx[:,1], c='b') 
#plt.scatter(nx[:,0],nx[:,2], c='r') 
plt.errorbar(nx[:,0],nx[:,2], yerr=nx[:,3], fmt='o', c='r')
#plt.ylim(0.5,0.7)
plt.ylabel('mu')
plt.xlabel('sample')

"""
Unable to reload guide and parameters

In [8]: torch.load(os.path.join(CHECKPOINT_DIR, '2019-03-10T18:31:12.698383_params'))                                                                                                               
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-8-6872f8b14703> in <module>
----> 1 torch.load(os.path.join(CHECKPOINT_DIR, '2019-03-10T18:31:12.698383_params'))

/usr/WS1/hammel1/proj/gpu_venv/lib/python3.6/site-packages/torch/serialization.py in load(f, map_location, pickle_module)
    365         f = open(f, 'rb')
    366     try:
--> 367         return _load(f, map_location, pickle_module)
    368     finally:
    369         if new_fd:

/usr/WS1/hammel1/proj/gpu_venv/lib/python3.6/site-packages/torch/serialization.py in _load(f, map_location, pickle_module)
    536     unpickler = pickle_module.Unpickler(f)
    537     unpickler.persistent_load = persistent_load
--> 538     result = unpickler.load()
    539 
    540     deserialized_storage_keys = pickle_module.load(f)

AttributeError: Can't get attribute 'guide' on <module '__main__'>
"""


"""
R^2 (coefficient of determination) regression score function.
Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). 
A constant model that always predicts the expected value of y, 
disregarding the input features, would get a R^2 score of 0.0.

In [23]: preds = [] 
          ...:  
          ...: for i in range(1000): 
          ...:     sampled_reg_model = guide(None, None) 
          ...:     pred = sampled_reg_model(data_train[0]).cpu().data.numpy().flatten() 
          ...:     preds.append(pred) 
          ...:  
          ...: all_preds = np.stack(preds).mean(0) 
          ...: r2_score(data_train[1][:,-1].cpu(), all_preds)                                                                                                                                             
Out[23]: -0.12804690200144275     #Seems to vary between -0.1 and +0.1

[ins] In [24]: all_preds.shape                                                                                                                                                                            
Out[24]: (848,)
[ins] In [25]: data_train[1][:,-1].cpu().shape                                                                                                                                                            
Out[25]: torch.Size([848])

[ins] In [31]: all_preds[1:30]                                                                                                                                                                            
Out[31]: 
array([6.020882 , 6.023583 , 6.0278573, 5.8618426, 6.014196 , 5.88622  ,
       5.971507 , 5.9299164, 6.025372 , 6.024828 , 6.0221314, 6.024089 ,
       5.971507 , 6.024828 , 5.9490256, 5.894002 , 6.0085297, 5.9859567,
       5.908943 , 5.844442 , 5.853278 , 5.923161 , 5.960731 , 5.9364977,
       5.994445 , 6.0225916, 5.9813895, 5.923161 , 5.9364977],
      dtype=float32)

[ins] In [32]: data_train[1][:,-1].cpu()[1:30]                                                                                                                                                            
Out[32]: 
tensor([6.0184, 6.0527, 6.0404, 5.9449, 6.0110, 5.9522, 5.9841, 5.9669, 6.0257,
        6.0502, 6.0551, 6.0233, 5.9841, 6.0502, 5.9743, 5.9547, 6.0061, 5.9914,
        5.9596, 5.9400, 5.9424, 5.9645, 5.9792, 5.9694, 5.9963, 6.0208, 5.9890,
        5.9645, 5.9694], device='cpu')



*** If I do r2 score with identical values I DO get 1.0 ***
[ins] In [34]: preds = [] 
          ...:  
          ...: for i in range(10000): 
          ...:     sampled_reg_model = guide(None, None) 
          ...:     pred = sampled_reg_model(data_train[0]).cpu().data.numpy().flatten() 
          ...:     preds.append(pred) 
          ...:  
          ...: all_preds = np.stack(preds).mean(0) 
          ...: r2_score(data_train[1][:,-1].cpu(), data_train[1][:,-1].cpu())                                                                                                                             
Out[34]: 1.0
"""
