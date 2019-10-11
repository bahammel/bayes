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

EPOCHS = 60000

N = 500
#hu = np.linspace(6, 14, 9)
hu = np.linspace(6, 14, 33)
#hu = 10.0
np.random.seed(1234)
shift = 1. + 0.1 * (np.random.rand(N, 1) - 0.5)
amp = 1.
sig = 1.0
mu = 10.0
x = ((amp / sig*np.sqrt(2.*np.pi)) * np.exp(-0.5*(((hu - shift*mu)/sig))**2.)).astype('float32') 
#y = mu * np.ones(500).astype('float32') 
y = (shift*mu).squeeze(-1).astype('float32') 


hidden_size = 100
model = torch.nn.Sequential(
    torch.nn.Linear(33, hidden_size),
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
xs = ((amp / sig*np.sqrt(2.*np.pi)) * np.exp(-0.5*(((hu - shift*mu)/sig))**2.)).astype('float32') 
for i in range(50):
    sampled_model = guide(None, None)
    #ys += [sampled_model(torch.tensor(xs)).detach().numpy().flatten()]
    ys += [sampled_model(torch.tensor(xs)).detach().numpy().flatten()]
ys = np.stack(ys).T

for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))


plt.figure()
plt.yscale('linear')
plt.title("Training Data")
plt.xlabel("hu (mu = <10> is Y for NN)")
plt.ylabel("Intensity (X for NN)")
plt.plot(hu,x[::10,:].T)

plt.figure()
plt.yscale('linear')
plt.title("Fit to mu")
plt.xlabel("Intersity (center is(x[:,16]))")
plt.ylabel("mu")
plt.legend()
plt.scatter(x[:,0], y, s=2, c='r', label='training data x[:,0]')
plt.scatter(x[:,8], y, s=2, c='r', label='training data x[:,8]')
plt.scatter(x[:,16], y, s=2, c='m', label='training data x[:16] CENTER')
plt.scatter(x[:,24], y, s=2, c='r', label='training data x[:24]')
plt.scatter(x[:,32], y, s=2, c='r', label='training data x[:,32]')

plt.scatter(xs[:,16], ys[:,1], s=2, c='b')
mean = ys.mean(1)
std = ys.std(1)
plt.scatter(xs[:,16], mean - std, mean + std, c='grey')
plt.scatter(xs[:,16], mean - std * 2, mean + std * 2, c='grey')
plt.scatter(xs[:,16], mean - std * 3, mean + std * 3, c='grey')
plt.scatter(xs[:,16], mean + std, mean + std, c='grey')
plt.scatter(xs[:,16], mean + std * 2, mean + std * 2, c='grey')
plt.scatter(xs[:,16], mean + std * 3, mean + std * 3, c='grey')
plt.scatter(xs[:,16], mean, c='g')


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

