import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pyro
from pyro.distributions import Normal, Uniform, Delta
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive
from pyro.infer.mcmc import MCMC, NUTS
import pyro.optim as optim

USE_GPU = torch.cuda.is_available()

#exp_id = '2019-03-11T10:02:52.595718'
#exp_id = '2019-04-06T10:48:00.497689'
exp_id = '2019-04-08T09:16:00.525043'
torch.set_default_tensor_type(
    'torch.cuda.FloatTensor' if USE_GPU else 'torch.FloatTensor'
)

CHECKPOINT = f'/usr/WS1/hammel1/proj/checkpoints/{exp_id}.pt'
STATE = f'/usr/WS1/hammel1/proj/bayes/{exp_id}_state.pt'
PARMS = f'/usr/WS1/hammel1/proj/bayes/{exp_id}_params.pt'
FULL = f'/usr/WS1/hammel1/proj/bayes/{exp_id}_full.pt'
OUTPUT = f'/usr/WS1/hammel1/proj/bayes/{exp_id}_output.pt'
DIR = f'2019-04-06T10:48:00.497689_state.pt'

print('exp_id is:', exp_id)
print('CHECKPOINT is:', CHECKPOINT)

torch.set_default_tensor_type('torch.cuda.FloatTensor')
USE_GPU = torch.cuda.is_available()
device = torch.device('cuda' if USE_GPU else 'cpu')

if USE_GPU:
    print("="*80)
    print("Model is using GPU")
    print("="*80)

# Break
#pdb.set_trace()
model_output = torch.load(OUTPUT)                                                                                                                                                                                                
model_full = torch.load(FULL)                                                                                                                                                                                                
pyro.get_param_store().load(PARMS)            

"""
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))
"""

for name, value in pyro.get_param_store().items(): 
    print(name, pyro.param(name).cpu().detach().numpy().mean()) 

"""
This doesn't work
pyro.module('module', nn, update_module_params=True)
"""

"""
This doesn't work
#Print model's state_dic    
print("Model's state_dict:") 
for param_tensor in model.state_dict(): 
    print(param_tensor, "\t", model.state_dict()[param_tensor].size()) 
"""

AdamArgs = { 'lr': 1e-2 }
#optimizer = torch.optim.Adam
scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': AdamArgs, 'gamma': 0.99995 })
svi = SVI(model, guide, scheduler, loss=Trace_ELBO())

"""
# Print optimizer's state_dict
#Gives Error
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer().state_dict()[var_name])   # Print optimizer's state_dict
"""


num_samples = 100
def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return mean.cpu().numpy, yhats

print('Prediction for unshifted spectra')


from functools import partial
import pandas as pd

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


for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))

for name, value in pyro.get_param_store().items(): 
    print(name, pyro.param(name).cpu().detach().numpy().mean()) 



posterior = svi.run(Xtrain, Ytrain)

# Break
#pdb.set_trace()

# posterior predictive distribution we can get samples from
trace_pred = TracePredictive(wrapped_model,
                             posterior,
                             num_samples=100)
post_pred = trace_pred.run(Xtrain, None)  #inputing pca components?
post_summary = summary(post_pred, sites= ['prediction', 'obs'])
meuw = post_summary["prediction"]
y = post_summary["obs"]
meuw.insert(0, 'true', np.array(Ytrain.cpu()))
y.insert(0, 'true', np.array(Ytrain.cpu()))


print("sample meuw data:")
print(meuw.head(10))
#What's the difference between mu and y? Means are the same but sigma is very different.
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

