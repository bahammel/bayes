#import pca_data_cretin_m1 as pca_data_cretin
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
exp_id = '2019-03-13T11:41:39.463032'
torch.set_default_tensor_type(
    'torch.cuda.FloatTensor' if USE_GPU else 'torch.FloatTensor'
)

CHECKPOINT = f'/usr/WS1/hammel1/proj/checkpoints/{exp_id}_params'
TEMP = f'/usr/WS1/hammel1/proj/bayes/{exp_id}_params.pt'
print('exp_id is:', exp_id)
print('CHECKPOINT is:', CHECKPOINT)

hu, PC, X_pca_test, X_pca, X, X_test, Y, Y_test = np.load(f'/usr/WS1/hammel1/proj/data/{exp_id}.npy')

#Only evaluate every 1000th point
Xtrain = torch.Tensor(X_pca)[::1000]
Xtest = torch.Tensor(X_pca_test)[::1000]
Ytrain = torch.Tensor(Y)[::1000]
Ytest = torch.Tensor(Y_test)[::1000]


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



torch.set_default_tensor_type('torch.cuda.FloatTensor')
USE_GPU = torch.cuda.is_available()
device = torch.device('cuda' if USE_GPU else 'cpu')

if USE_GPU:
    print("="*80)
    print("Model is using GPU")
    print("="*80)

# Break
#pdb.set_trace()


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
    
    
    # run the regressor forward conditioned on inputs
    prediction_mean = lifted_reg_model(x_data).squeeze(-1)
    pyro.sample("obs", Normal(prediction_mean, 1),
                obs=y_data)
    return prediction_mean


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


net = NN(PC,100,1)
net.load_state_dict(torch.load(TEMP))
"""
torch.load(CHECKPOINT)
state = torch.load(CHECKPOINT)
net.load_state_dict(state['state_dict'])
optimizer.load_state_dict(state['optimizer'])
#https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
"""

#Print model's state_dic    
print("Model's state_dict:") 
for param_tensor in NN(PC,100,1).state_dict(): 
    print(param_tensor, "\t", NN(PC,100,1).state_dict()[param_tensor].size()) 


AdamArgs = { 'lr': 1e-2 }
optimizer = torch.optim.Adam
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


for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))

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

