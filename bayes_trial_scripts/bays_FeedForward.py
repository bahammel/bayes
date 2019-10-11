import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, SGD
import pyro.poutine as poutine

# enable validation (e.g. validate parameters of distributions)
assert pyro.__version__.startswith('0.3.1')
pyro.enable_validation(True)


N = 500
x = (np.random.normal(size=N)).astype('float32')
y = (x ** 2 + np.random.normal(scale=0.05, size=N)).astype('float32')

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

"""
This doesn't work
trace = poutine.trace(poutine.enum(pyromodel, first_available_dim=-3)).get_trace()
trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
print(trace.format_shapes())
"""
pyro.clear_param_store()
for j in range(1000):
    loss = svi.step(torch.tensor(x[:, np.newaxis]), torch.tensor(y))
    if j % 100 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / float(N)))
        
        
ys = []
xs = np.linspace(-5, 5, 500, dtype='float32')
for i in range(50):
    sampled_model = guide(None, None)
    ys += [sampled_model(torch.tensor(xs[:, np.newaxis])).detach().numpy().flatten()]
ys = np.stack(ys).T

plt.scatter(x, y, s=2, c='r')
mean = ys.mean(1)
std = ys.std(1)
plt.fill_between(xs, mean - std, mean + std, facecolor='gray', alpha=0.2, interpolate=True)
plt.fill_between(xs, mean - std * 2, mean + std * 2, facecolor='gray', alpha=0.2, interpolate=True)
plt.fill_between(xs, mean - std * 3, mean + std * 3, facecolor='gray', alpha=0.2, interpolate=True)
plt.plot(xs, mean)
