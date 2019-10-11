import torch
import pyro
import pyro.optim
import matplotlib.pyplot as plt
from torch.distributions import constraints
from pyro.distributions import Normal, Categorical
import torch.nn as nn

# The true model, with randomly generated weights and fixed noise level
def build_linear_dataset(N,p,noise_std=0.01):
    w = torch.randn(1)
    b = torch.randn(1)
    xs = torch.randn(N)
    ys = w * xs + b
    ys += torch.distributions.Normal(0,noise_std).sample(ys.shape)
    plt.plot(xs.numpy(), ys.numpy(),'+')
    print('w is %g, b is %g, noise_level is %g' % (w,b,noise_level))
    data = torch.cat((xs,ys),1)
    return data

# NN with one linear layer
class RegressionModel(nn.Module):
    def __init__(self, p):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)

    def forward(self, x):
        # x * w + b
        return self.linear(x)


N = 100  # size of toy data
p = 2  # number of features

softplus = nn.Softplus()
regression_model = RegressionModel(p)

def model(x,y):
    # Create unit normal priors over the parameters
    loc = torch.zeros(1,2)
    scale = torch.zeros(1,2)
    bias_loc = torch.zeros(1,2)
    bias_scale = torch.zeros(1,2)
    w_prior = Normal(loc, scale).to_event(1)
    b_prior = Normal(bias_loc, bias_scale).to_event(1)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    with pyro.plate("map", N, subsample=x):
        # run the regressor forward conditioned on inputs
        prediction_mean = lifted_reg_model(x).squeeze(-1)
        pyro.sample("obs", Normal(prediction_mean, 1),
                    obs=y)




def guide(x,y):
    w_loc = torch.randn(1, p)
    w_log_sig = -3 + 0.05 * torch.randn(1, p)
    b_loc = torch.randn(1, p)
    b_log_sig = -3 + 0.05 * torch.randn(1, p)
    # register learnable params in the param store
    mw_param = pyro.param("guide_mean_weight", w_loc)
    sw_param = softplus(pyro.param("guide_log_scale_weight", w_log_sig))
    mb_param = pyro.param("guide_mean_bias", b_loc)
    sb_param = softplus(pyro.param("guide_log_scale_bias", b_log_sig))
    # gaussian guide distributions for w and b
    w_dist = Normal(mw_param, sw_param).to_event(1)
    b_dist = Normal(mb_param, sb_param).to_event(1)
    dists = {'linear.weight': w_dist, 'linear.bias': b_dist}
    # overloading the parameters in the module with random samples from the guide distributions
    lifted_module = pyro.random_module("module", regression_model, dists)
    # sample a regressor
    return lifted_module()

optimizer = pyro.optim.SGD({"lr": 1e-4, "momentum": 0.1})
pyro.clear_param_store()
svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO())
rec_loss = []
for i in range(100):
    loss = svi.step(xs,ys)
    rec_loss.append(loss)
    print('Iter %d, loss = %g' % (i,loss))

fig = plt.figure(figsize = (18,6))
plt.subplot(131)
plt.plot(rec_loss)

plt.subplot(132)
plt.plot(xs.numpy(),ys.numpy(),'+')
for i in range(10):
    wb, prec = guide(xs,ys)
    py = wb[0] * xs + wb[1]
    plt.plot(xs.numpy(), py.detach().numpy(), 'g', alpha = 0.1)


plt.subplot(133)
noise  = []
for i in range(1000):
    wb,prec = guide(xs,ys)
    noise.append(1/torch.sqrt(prec).item())
plt.hist(noise)   

v0 = torch.eye(2)
n   = xs.shape[0]
x_aug = torch.ones(n,2)
x_aug[:,0] = xs

Vn = torch.inverse(torch.eye(2) + x_aug.t().mm(x_aug))
wn = Vn.mv(x_aug.t().mv(ys))
alpha_n = 5 + n / 2
beta_n   = 5 + 0.5 * (ys.dot(ys) - wn.dot(Vn.inverse().mv(wn)))

w_sampler = pyro.distributions.MultivariateNormal(wn,Vn)
prec_sampler = pyro.distributions.Gamma(alpha_n,beta_n)

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(xs.numpy(), ys.numpy(), '+', alpha = 0.2)
for i in range(10):
    wb = w_sampler.sample()
    py = wb[0]  * xs + wb[1]
    plt.plot(xs.numpy(), py.detach().numpy(), 'g', alpha = 0.3)
    
prec = torch.sqrt(1/prec_sampler.sample((1000,)))
plt.subplot(122)
plt.hist(prec.numpy())


"""
# The variational distribution
def new_guide(x,y):
    wb   = pyro.sample("wb", pyro.distributions.MultivariateNormal(wn, Vn))
    prec = pyro.sample("precision", pyro.distributions.Gamma(alpha_n, beta_n))
    return wb, prec

new_svi = pyro.infer.SVI(model, new_guide, optimizer, pyro.infer.Trace_ELBO())
print(new_svi.step(xs,ys))
"""
