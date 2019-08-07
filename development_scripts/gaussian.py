import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, SGD
import pyro.poutine as poutine
import random

# enable validation (e.g. validate parameters of distributions)
assert pyro.__version__.startswith('0.3.1')
pyro.enable_validation(True)

N = 500
#hu = np.linspace(0.1, 1.0, 10)
hu = 10.0
np.random.seed(1234)
shift = 1. + 0.1 * (np.random.rand(N, 1) - 0.5)
amp = 1.
std = 1.0
mu = 10.0
x = ((amp / std*np.sqrt(2.*np.pi)) * np.exp(-0.5*(hu - shift*mu)/std)**2.).astype('float32') 
y = mu * np.ones(500).astype('float32') 
#y = (shift*mu).squeeze(-1).astype('float32') 


hidden_size = 8
model = torch.nn.Sequential(
    torch.nn.Linear(1, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, 1),
)

def pyromodel(x, y):
    priors = {}
    for name, par in model.named_parameters():
        priors[name] = dist.Normal(torch.zeros(*par.shape), 10 * torch.ones(*par.shape)).independent(par.dim())
    bayesian_model = pyro.random_module('bayesian_model', model, priors)
    sampled_model = bayesian_model()
    with pyro.iarange("map", len(x)):
        prediction_mean = sampled_model(x).squeeze(-1)
        pyro.sample("obs",
                    dist.Normal(prediction_mean,
                                0.05 * torch.ones(x.size(0))),
                    obs=y)
    
softplus = torch.nn.Softplus()

def guide(x, y):
    dists = {}
    for name, par in model.named_parameters():
        loc = pyro.param(name + '.loc', torch.randn(*par.shape))
        scale = softplus(pyro.param(name + '.scale',
                                    -3.0 * torch.ones(*par.shape) + 0.05 * torch.randn(*par.shape)))
        dists[name] = dist.Normal(loc, scale).independent(par.dim())
    bayesian_model = pyro.random_module('bayesian_model', model, dists)
    return bayesian_model()

optim = Adam({"lr": 0.05})
svi = SVI(pyromodel, guide, optim, loss=Trace_ELBO())

pyro.clear_param_store()
for j in range(1000):
    loss = svi.step(torch.tensor(x), torch.tensor(y)) 
    if j % 100 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / float(N)))
        
        
      
print("x.shape",x.shape)
print("y.shape",y.shape)

#trace = poutine.#trace(pyromodel).get_trace(x, y)
#trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
#print(trace.format_shapes())
        
ys = []
#xs = amp * (1.0) / (std*np.sqrt(2.*np.pi)) * np.exp(-((hu - 6.0)**2./(2.* std**2.))**1.0)
xs = np.linspace(0, 5, 500, dtype='float32')
for i in range(50):
    sampled_model = guide(None, None)
    #ys += [sampled_model(torch.tensor(xs)).detach().numpy().flatten()]
    ys += [sampled_model(torch.tensor(xs[:, np.newaxis])).detach().numpy().flatten()]
ys = np.stack(ys).T

for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))


plt.scatter(x, y, s=2, c='r')
plt.scatter(xs, ys[:,1], s=2, c='b')
mean = ys.mean(1)
std = ys.std(1)
plt.fill_between(xs, mean - std, mean + std, facecolor='gray', alpha=0.2, interpolate=True)
plt.fill_between(xs, mean - std * 2, mean + std * 2, facecolor='gray', alpha=0.2, interpolate=True)
plt.fill_between(xs, mean - std * 3, mean + std * 3, facecolor='gray', alpha=0.2, interpolate=True)
plt.plot(xs, mean, c='g')


#print("x",x)
print("x.shape",x.shape)
#print("y",y)
print("y.shape",y.shape)
#print("xs",xs)
print("xs.shape",xs.shape)
#print("ys",ys)
print("ys.shape",ys.shape)
#print("xs[:, np.newaxis])",xs[:, np.newaxis])
print("xs[:, np.newaxis]).shape",xs[:, np.newaxis].shape)

#print("torch.tensor(xs[:, np.newaxis]))",torch.tensor(xs[:, np.newaxis]))
print("torch.tensor(xs[:, np.newaxis])).shape",torch.tensor(xs[:, np.newaxis]).shape)

