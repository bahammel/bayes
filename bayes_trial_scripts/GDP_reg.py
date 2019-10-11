import os
from functools import partial
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import pyro
from pyro.distributions import Normal, Uniform, Delta
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.distributions.util import logsumexp
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive
from pyro.infer.mcmc import MCMC, NUTS
import pyro.optim as optim
import pyro.poutine as poutine

# for CI testing
smoke_test = ('CI' in os.environ)
#assert pyro.__version__.startswith('0.3.0')
pyro.enable_validation(True)
pyro.set_rng_seed(1)
pyro.enable_validation(True)


DATA_URL = "https://d2fefpcigoriu7.cloudfront.net/datasets/rugged_data.csv"
data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
df = data[["cont_africa", "rugged", "rgdppc_2000"]]
df = df[np.isfinite(df.rgdppc_2000)]
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

"""
data[:,0]   # Africa OR Not                                                                                                                                                                           
tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0.,
. . .        
        0., 0., 0., 0., 0., 1., 1., 1.])

data[:,1]    # Ruggedness Index                                                                                                                                                                          
tensor([8.5800e-01, 3.4270e+00, 7.6900e-01, 7.7500e-01, 2.6880e+00, 6.0000e-03,
. . .
    5.3300e-01, 1.1940e+00])

data[:,2]      # log(GDP)                                                                                                                                                                        
tensor([ 7.4926,  8.2169,  9.9333,  9.4070,  7.7923,  9.2125, 10.1432, 10.2746,
. . .
    6.6516,  7.8237])
"""


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
african_nations = data[data["cont_africa"] == 1]
non_african_nations = data[data["cont_africa"] == 0]
sns.scatterplot(non_african_nations["rugged"],
            np.log(non_african_nations["rgdppc_2000"]),
            ax=ax[0])
ax[0].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="Non African Nations")
sns.scatterplot(african_nations["rugged"],
            np.log(african_nations["rgdppc_2000"]),
            ax=ax[1])
ax[1].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="African Nations")

class RegressionModel(nn.Module):
    def __init__(self, p):
        # p = number of features
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)
        self.factor = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        return self.linear(x) + (self.factor * x[:, 0] * x[:, 1]).unsqueeze(1)
    # The second term above is to allow for cross correleation between Country and Ruggedness

p = 2  # number of features
regression_model = RegressionModel(p)


loss_fn = torch.nn.MSELoss(reduction='sum')
optim = torch.optim.Adam(regression_model.parameters(), lr=0.05)
num_iterations = 1000 if not smoke_test else 2
data = torch.tensor(df.values, dtype=torch.float)
x_data, y_data = data[:, :-1], data[:, -1]

"""
[ins] In [12]: x_data.shape                                                                                                                                                                          
Out[12]: torch.Size([170, 2])

[ins] In [13]: y_data.shape                                                                                                                                                                          
Out[13]: torch.Size([170])

[ins] In [10]: x_data                                                                                                                                                                                
Out[10]: 
tensor([[1.0000e+00, 8.5800e-01],
        [0.0000e+00, 3.4270e+00],
        [0.0000e+00, 7.6900e-01],
        [0.0000e+00, 7.7500e-01],
[ins] In [11]: y_data                                                                                                                                                                                
Out[11]: 
tensor([ 7.4926,  8.2169,  9.9333,  9.4070,  7.7923,  9.2125, 10.1432, 10.2746,
         7.8520,  6.4324, 10.2148,  6.8661,  6.9062,  7.2992,  8.6960,  9.6758,
         9.7396,  8.5745,  8.4768,  8.6775,  7.7827,  8.8957,  9.6350,  8.9493,



"""

def main():
    x_data = data[:, :-1]
    y_data = data[:, -1]
    for j in range(num_iterations):
        # run the model forward on the data
        y_pred = regression_model(x_data).squeeze(-1)
        # calculate the mse loss
        loss = loss_fn(y_pred, y_data)
        # initialize gradients to zero
        optim.zero_grad()
        # backpropagate
        loss.backward()
        # take a gradient step
        optim.step()
        if (j + 1) % 50 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))
    # Inspect learned parameters
    print("Learned parameters:")
    for name, param in regression_model.named_parameters():
        print(name, param.data.numpy())

main()


"""
Output

[ins] In [1]: %run GDP_reg.py                                                                                                                                                                        
[iteration 0050] loss: 3723.7849
. . . 
[iteration 1000] loss: 147.9397
Learned parameters:
factor 0.37248382  #This is the correleation factor between Country and Ruggedness
linear.weight [[-1.90511    -0.18619268]]  #This is the two slopes for linear fit (Country, Ruggedness)
linear.bias [9.188872]  # This is the off-set for the linear fit (Why only one value?)
"""
