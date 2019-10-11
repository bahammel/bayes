import logging
import os
import torch
import pyro
from pyro.distributions import Delta
from pyro.infer import EmpiricalMarginal, TracePredictive
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import glob
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
import numpy as np

import gaussian_1V_M1 as cretin_nn
import pca_data_1V as pca_data_hu

# logging.basicConfig(level=logging.DEBUG)

USE_GPU = torch.cuda.is_available()
device = torch.device('cuda' if USE_GPU else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if USE_GPU else 'torch.FloatTensor')

if os.environ['HOSTNAME'] == 'fractal':
    PARAM_FILES = '/hdd/bdhammel/checkpoints/bayes/*.params'
else:
    PARAM_FILES = '/usr/WS1/hammel1/proj/checkpoints/bayes/*.params'

if os.environ['HOSTNAME'] == 'fractal':
    DATA_FILES = '/hdd/bdhammel/checkpoints/bayes/*.npy'
else:
    DATA_FILES = '/usr/WS1/hammel1/proj/checkpoints/bayes/*.npy'


#experiment_id = '2019-09-11T16:39:29.282283' 

if USE_GPU:
    print("=" * 80)
    print("Model is using GPU")
    print("=" * 80)


def get_marginal(traces, sites):
    return EmpiricalMarginal(
        traces, sites
    )._get_samples_and_weights()[0].detach().cpu().numpy()


def summary(traces, sites):
    # returns 100
    marginal = get_marginal(traces, sites)
    site_stats = {}
    for i in range(marginal.shape[1]):
        site_name = sites[i]
        marginal_site = pd.DataFrame(marginal[:, i, :, 0]).transpose()
        describe = partial(
            pd.Series.describe, percentiles=[0.16, 0.5, 0.84]
        )
        site_stats[site_name] = marginal_site.apply(
            describe, axis=1
        )[["mean", "std", "16%", "50%", "84%"]]

    return site_stats

def summary2(traces, sites):
    # returns 100
    marginal = get_marginal(traces, sites)
    return marginal


def wrapped_model_fn(model):
    def _wrapped_model(x_data, y_data):
        pyro.sample("prediction", Delta(model(x_data, y_data)))
    return _wrapped_model


def trace_summary(svi, model, x_data, y_data):
    optim = Adam({"lr": 0.03})
    # svi = SVI(model, guide, optim, loss=Trace_ELBO(), num_samples=1000)
    posterior = svi.run(x_data, y_data)
    wrapped_model = wrapped_model_fn(model)

    # posterior predictive distribution we can get samples from
    trace_pred = TracePredictive(wrapped_model,
                                 posterior,
                                 num_samples=nsamples)
    post_pred = trace_pred.run(x_data, None)
    post_summary = summary2(post_pred, sites=['prediction', 'obs'])
    #import pdb; pdb.set_trace()    
    pred = post_summary[:,0,:]
    obs = post_summary[:,1,:]

    pred_mean = pred.mean(axis=0)
    pred_std = pred.std(axis=0)
    obs_mean = obs.mean(axis=0)
    obs_std = obs.std(axis=0)


    #import pdb; pdb.set_trace()
    plot_pred(y_data, pred_mean, pred_std, experiment_id)
    plot_obs(y_data, obs_mean, pred_mean, obs_std, pred_std, experiment_id)

def plot_dists(df):
    from scipy.stats import norm

    x = np.linspace(2, 16, 100) 
    for test_val in np.unique(df['y_test']):
        plt.figure(f"{test_val}")
        mus = df['pred_mean'][df['y_test']==test_val]
        stds = df['pred_std'][df['y_test']==test_val]
        plt.hist(mus, normed=True)
        pdf = norm(loc=mus.mean(), scale=stds.mean())
        plt.plot(x, pdf.pdf(x))


    #import pudb; pudb.set_trace()

def guide_summary(guide, x_data, y_data, experiment_id):
    #import pudb; pudb.set_trace()
    sampled_models = [guide(None, None) for _ in range(1000)]
    npredicted = np.asarray(
        [model(x_data).data.cpu().numpy()[:, 0] for model in sampled_models]
    )
    pred_mean = np.mean(npredicted, axis=0)
    pred_std = np.std(npredicted, axis=0)
    pred_16q = np.percentile(npredicted, 16, axis=0)
    pred_84q = np.percentile(npredicted, 84, axis=0)
    #[ins] In [7]: npredicted.shape
    #Out[7]: (1000, 21000)
    # Therefore, average is over the 1000 model samples

    #idx = np.argsort(y_data[:,0].squeeze())
    plot_pred(y_data, pred_mean, pred_std, experiment_id)


def plot_pred(x_data,y1_data,y1_error, experiment_id):
    plt.figure()
    plt.title('Guide summary: y[:,0] mu  \n' +  experiment_id)
    plt.errorbar(x_data[:,0], y1_data[:,0], yerr=y1_error[:,0], fmt='o', c='b', ms = 0.5, elinewidth=0.2,errorevery=1)  #mean of predicted values w 1std error
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.legend()

def plot_obs(x_data,y1_data,y2_data,y1_error,y2_error, experiment_id):
    plt.figure()
    plt.title('Trace summary: y[:,0] mu  \n' +  experiment_id)    
    plt.errorbar(x_data[:,0], y1_data[:,0], yerr=y1_error[:,0], fmt='+', c='r', ms = 1, elinewidth=0.5,errorevery=1, label='obs')  #mean of predicted values w 1std error
    plt.errorbar(x_data[:,0], y2_data[:,0], yerr=y2_error[:,0], fmt='o', c='b', ms = 0.5, elinewidth=0.2,errorevery=1, label='pred')  #mean of predicted values w 1std error
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.legend()


    #import pdb; pdb.set_trace()
if __name__ == '__main__':

    model = cretin_nn.pyromodel
    guide = cretin_nn.guide
    optim = Adam({"lr": 0.05})
    svi = SVI(model, guide, optim, loss=Trace_ELBO(), num_samples=1000)

    saved_param_files = glob.glob(PARAM_FILES)
    saved_param_files.sort(key=os.path.getmtime, reverse=True)
    print(*saved_param_files, sep='\n')
    idx = int(input("file? (0 for most recent exp) > "))
    pyro.get_param_store().load(saved_param_files[idx])

    saved_data_files = glob.glob(DATA_FILES)
    saved_data_files.sort(key=os.path.getmtime, reverse=True)
    print(*saved_data_files, sep='\n')
    idx = int(input("file? (0 for most recent data) > "))


    experiment_id = saved_data_files[idx].rsplit('/', 1)[1]


    hu, PC, xx, xtest_pca, xtrain, xtest, yy, ytest, X,Y = np.load(saved_data_files[idx])

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name).mean(0).mean(0))

    nsamples = 1000

    xs = torch.tensor(xtest_pca.astype('float32'))
    ys =  np.zeros(shape=(12500,nsamples,2))


    #guide_summary(guide, xs, ytest)
    trace_summary(svi, model, xs, ytest)


    for i in range(nsamples):
        sampled_model = guide(None, None)
        ys[:,i,:] = sampled_model(xs).cpu().detach().numpy()

    mean_0 = ys.mean(1)[:,0]                                           
    std_0 = ys.std(1)[:,0] 

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.title("Fit to y[0]")
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.errorbar(ytest[:,0], mean_0, yerr=std_0, fmt='o', c='r', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("y[0] hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,0] - mean_0)/std_0)


