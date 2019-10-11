import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pca_data_hu_ndl_bay as pca_data_hu
from torch.utils import data as data_utils

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython import display
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from scipy.misc import imread

# %matplotlib inline
"""
USE_GPU = torch.cuda.is_available()

device = torch.device('cuda' if USE_GPU else 'cpu')
torch.set_default_tensor_type(
    'torch.cuda.FloatTensor' if USE_GPU else 'torch.FloatTensor'
)

if USE_GPU:
    print("="*80)
    print("Model is using GPU")
    print("="*80)
    #net.cuda()
"""

class NN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return output

hu, PC, xtrain_pca, xtest_pca, Xtrain, Xtest, Ytrain, Ytest, X, Y = pca_data_hu.pca_data()

#data = xtrain_pca, Ytrain
Xtrain_pca = torch.from_numpy(xtrain_pca)                                                                                                                                                         
Ytrain = torch.from_numpy(Ytrain[:,0])                                                                                                                                                                             

Xtrain_pca = Xtrain_pca.type(torch.FloatTensor)
Ytrain = Ytrain.type(torch.FloatTensor)

"""
[ins] In [8]: xtrain_pca                                                                                                                                                                                                          
Out[8]: 
array([[ 6.62943480e-01,  1.31268632e-04, -3.60591161e-06, ...,
         4.97148464e-05, -5.72742316e-05, -7.27695971e-06],
       [ 6.63002130e-01, -5.75280890e-05, -1.15661437e-05, ...,
        -2.00387909e-04, -7.48236293e-05,  4.09897400e-05],
       [-6.62909440e-01, -1.11439025e-05, -9.09003091e-06, ...,
        -4.27010632e-05,  1.74825065e-05, -1.76335502e-05],
       ...,
       [ 6.62844604e-01,  2.59299274e-04,  2.00394289e-04, ...,
        -1.87430159e-05, -1.72265167e-04,  1.46268924e-06],
       [-6.62946765e-01,  1.01955843e-05, -3.81483561e-05, ...,
         1.70382013e-05, -1.31019581e-05, -1.53496732e-05],
       [ 6.62963571e-01,  6.50297250e-05,  6.26137859e-05, ...,
         8.20093643e-05,  1.20424331e-04, -2.03586922e-04]])

[ins] In [9]: xtrain_pca.shape                                                                                                                                                                                                    
Out[9]: (750, 10)

[ins] In [10]: Ytrain    
Out[11]: 
array([[0.5],
       [0.5],
       [2. ],
       [2. ],
       [2. ],
       [2. ],
       [2. ],
       [0.5],
       [2. ],
       [2. ],
       [0.5],
       [2. ],
etc. . .

In [13]: Ytrain.shape                                                                                                                                                                                                       
Out[13]: (750, 1)

Tried This:
In [24]: torch.from_numpy(Ytrain).view(750,-1).shape                                                                                                                                                                        
Out[24]: torch.Size([750, 1])

In [25]: torch.from_numpy(Ytrain).shape                                                                                                                                                                                     
Out[25]: torch.Size([750, 1])


This looks right:
In [28]: torch.from_numpy(Ytrain[:,0]).shape                                                                                                                                                                                
Out[28]: torch.Size([750])

[ins] In [2]: Ytrain                                                                                                                                                                                                              
Out[2]: 
tensor([0.5000, 0.5000, 2.0000, 2.0000, 2.0000, 0.5000, 0.5000, 2.0000, 2.0000,
        0.5000, 0.5000, 2.0000, 2.0000, 0.5000, 2.0000, 2.0000, 2.0000, 2.0000,
        0.5000, 0.5000, 0.5000, 2.0000, 0.5000, 2.0000, 0.5000, 2.0000, 2.0000,
        2.0000, 2.0000, 2.0000, 2.0000, 0.5000, 2.0000, 2.0000, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000, 2.0000, 0.5000, 2.0000, 0.5000, 2.0000, 0.5000,
. . . 
        0.5000, 2.0000, 0.5000, 2.0000, 2.0000, 2.0000, 0.5000, 2.0000, 2.0000,
        2.0000, 2.0000, 0.5000, 2.0000, 0.5000, 2.0000, 0.5000, 0.5000, 0.5000,
        0.5000, 2.0000, 2.0000, 0.5000, 2.0000, 0.5000, 0.5000, 2.0000, 2.0000,
        2.0000, 0.5000, 2.0000, 2.0000, 2.0000, 0.5000, 2.0000, 0.5000, 0.5000,
        0.5000, 2.0000, 0.5000], dtype=torch.float64)

*****
So I did this:
Xtrain_pca = torch.from_numpy(xtrain_pca)                                                                                                                                                         
Ytrain = torch.from_numpy(Ytrain[:,0])                                                                                                                                                                             
Xtrain_pca = Xtrain_pca.type(torch.FloatTensor)
Ytrain = Ytrain.type(torch.FloatTensor)
*****


[ins] In [2]: Xtrain_pca                                                                                                                                                                                                          
Out[2]: 
tensor([[ 6.6487e-01,  2.1642e-05,  1.1468e-04,  ...,  1.0358e-04,
         -1.2248e-05,  2.1390e-05],
        [-6.6114e-01, -1.2135e-05, -1.7397e-05,  ..., -4.7786e-05,
          6.9052e-08, -2.8681e-05],
        [ 6.6468e-01,  3.7801e-05, -1.0179e-04,  ..., -1.2996e-04,
          1.5446e-04,  6.3970e-05],
        ...,
        [-6.6112e-01, -9.0545e-06,  2.4147e-05,  ..., -3.2966e-05,
          3.1447e-06,  9.2491e-06],
        [ 6.6448e-01,  1.6039e-04,  2.0468e-04,  ...,  7.2432e-05,
         -3.3571e-05, -1.0029e-04],
        [ 6.6469e-01, -1.3184e-04, -1.5170e-04,  ...,  8.4855e-05,
          6.5450e-05,  2.5622e-04]])

[ins] In [3]: Xtrain_pca.shape                                                                                                                                                                                                    
Out[3]: torch.Size([750, 10])

[ins] In [4]: Ytrain                                                                                                                                                                                                              
Out[4]: 
tensor([0.5000, 2.0000, 0.5000, 2.0000, 2.0000, 0.5000, 0.5000, 0.5000, 0.5000,
        2.0000, 0.5000, 2.0000, 2.0000, 2.0000, 0.5000, 2.0000, 0.5000, 2.0000,
        2.0000, 2.0000, 2.0000, 0.5000, 2.0000, 0.5000, 0.5000, 2.0000, 2.0000,

. . . . .
        0.5000, 2.0000, 2.0000, 2.0000, 0.5000, 2.0000, 0.5000, 0.5000, 2.0000,
        0.5000, 0.5000, 2.0000, 2.0000, 0.5000, 0.5000, 2.0000, 0.5000, 2.0000,
        2.0000, 0.5000, 0.5000])

[ins] In [5]: Ytrain.shape                                                                                                                                                                                                        
Out[5]: torch.Size([750])


Needs to look like the MNIST data below:

MNIST Data looks like this:
In [2]: data[0]                                                                                                                                                                                                             
Out[2]: 
tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]],

        ...,
.

        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]]])
    
 data[0].shape                                                                                                                                                                              
 torch.Size([128, 1, 28, 28])
 
NOTE: Before data[0] gets used in the training loop below, the shape is changed 
 data[0].view(-1,28*28).shape                                                                                                                                                              
 torch.Size([128, 784])


In [3]: data[1]                                                                                                                                                                                                             
Out[3]: 
tensor([6, 0, 6, 0, 9, 6, 3, 6, 3, 8, 6, 9, 0, 4, 6, 0, 3, 7, 4, 1, 7, 5, 1, 9,
        1, 5, 7, 6, 8, 7, 2, 9, 3, 3, 4, 3, 3, 7, 0, 8, 9, 6, 5, 3, 0, 4, 5, 1,
        8, 3, 8, 0, 2, 1, 1, 8, 9, 5, 9, 3, 2, 7, 1, 8, 0, 7, 5, 8, 6, 0, 9, 4,
        6, 8, 8, 0, 0, 4, 8, 5, 0, 6, 2, 8, 3, 8, 7, 7, 3, 7, 2, 2, 4, 6, 3, 4,
        3, 9, 1, 4, 1, 4, 4, 9, 4, 7, 3, 2, 2, 4, 4, 1, 4, 7, 6, 8, 9, 2, 3, 3,
        7, 9, 7, 8, 2, 8, 7, 1])

data[1].shape                                                                                                                                                                             
 torch.Size([128])

In [6]: data[1].type                                                                                                                                                                                                        
Out[6]: <function Tensor.type>

"""



net = NN(10, 10, 1)

import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

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
    
    lhat = log_softmax(lifted_reg_model(x_data))
    
    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)

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

optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO())


num_iterations = 5
loss = 0

for j in range(num_iterations):
    loss = 0
    loss += svi.step(Xtrain_pca, Ytrain)
    #normalizer_train = len(train_loader.dataset)
    #total_epoch_loss_train = loss / normalizer_train
    total_epoch_loss_train = loss
    
    print("Epoch ", j, " Loss ", total_epoch_loss_train)






num_samples = 10
def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return np.argmax(mean.numpy(), axis=1)

print('Prediction when network is forced to predict')
correct = 0
total = 0
for j, data in enumerate(test_loader):
    images, labels = data
    predicted = predict(images.view(-1,28*28))
    total += labels.size(0)
    correct += (predicted == labels.numpy()).sum().item()
print("accuracy: %d %%" % (100 * correct / total))

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')
"""
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #plt.imshow(npimg,  cmap='gray')
    #fig.show(figsize=(1,1))
    
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow(npimg,  cmap='gray', interpolation='nearest')
    plt.show()

num_samples = 100

def give_uncertainities(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [F.log_softmax(model(x.view(-1,28*28)).data, 1).detach().numpy() for model in sampled_models]
    return np.asarray(yhats)
    #mean = torch.mean(torch.stack(yhats), 0)
    #return np.argmax(mean, axis=1)

def test_batch(images, labels, plot=True):
    y = give_uncertainities(images)
    predicted_for_images = 0
    correct_predictions=0

    for i in range(len(labels)):
    
        if(plot):
            print("Real: ",labels[i].item())
            fig, axs = plt.subplots(1, 10, sharey=True,figsize=(20,2))
    
        all_digits_prob = []
    
        highted_something = False
    
        for j in range(len(classes)):
        
            highlight=False
        
            histo = []
            histo_exp = []
        
            for z in range(y.shape[0]):
                histo.append(y[z][i][j])
                histo_exp.append(np.exp(y[z][i][j]))
            
            prob = np.percentile(histo_exp, 50) #sampling median probability
        
            if(prob>0.2): #select if network thinks this sample is 20% chance of this being a label
                highlight = True #possibly an answer
        
            all_digits_prob.append(prob)
            
            if(plot):
            
                N, bins, patches = axs[j].hist(histo, bins=8, color = "lightgray", lw=0,  weights=np.ones(len(histo)) / len(histo), density=False)
                axs[j].set_title(str(j)+" ("+str(round(prob,2))+")") 
        
            if(highlight):
            
                highted_something = True
                
                if(plot):

                    # We'll color code by height, but you could use any scalar
                    fracs = N / N.max()

                    # we need to normalize the data to 0..1 for the full range of the colormap
                    norm = colors.Normalize(fracs.min(), fracs.max())

                    # Now, we'll loop through our objects and set the color of each accordingly
                    for thisfrac, thispatch in zip(fracs, patches):
                        color = plt.cm.viridis(norm(thisfrac))
                        thispatch.set_facecolor(color)

    
        if(plot):
            plt.show()
    
        predicted = np.argmax(all_digits_prob)
    
        if(highted_something):
            predicted_for_images+=1
            if(labels[i].item()==predicted):
                if(plot):
                    print("Correct")
                correct_predictions +=1.0
            else:
                if(plot):
                    print("Incorrect :()")
        else:
            if(plot):
                print("Undecided.")
        
        if(plot):
            imshow(images[i].squeeze())
        
    
    if(plot):
        print("Summary")
        print("Total images: ",len(labels))
        print("Predicted for: ",predicted_for_images)
        print("Accuracy when predicted: ",correct_predictions/predicted_for_images)
        
    return len(labels), correct_predictions, predicted_for_images




# Prediction when network can decide not to predict

print('Prediction when network can refuse')
correct = 0
total = 0
total_predicted_for = 0
for j, data in enumerate(test_loader):
    images, labels = data
    
    total_minibatch, correct_minibatch, predictions_minibatch = test_batch(images, labels, plot=False)
    total += total_minibatch
    correct += correct_minibatch
    total_predicted_for += predictions_minibatch

print("Total images: ", total)
print("Skipped: ", total-total_predicted_for)
print("Accuracy when made predictions: %d %%" % (100 * correct / total_predicted_for))



# preparing for evaluation

dataiter = iter(test_loader)
images, labels = dataiter.next()

test_batch(images[:100], labels[:100])
"""    
