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


experiment_id = '2019-09-11T16:39:29.282283' 

RESTORE_PATH = f'/usr/WS1/hammel1/proj/checkpoints/bayes/{experiment_id}'
DATA_FILE = RESTORE_PATH + '.data.npy'
PARAM_FILE = RESTORE_PATH + '.params'

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


def wrapped_model_fn(model):
    def _wrapped_model(x_data, y_data):
        pyro.sample("prediction", Delta(model(x_data, y_data)))
    return _wrapped_model


def trace_summary(svi, model, x_data, y_data):
    optim = Adam({"lr": 0.03})
    # svi = SVI(model, guide, optim, loss=Trace_ELBO(), num_samples=10000)
    posterior = svi.run(x_data, y_data)
    wrapped_model = wrapped_model_fn(model)

    # posterior predictive distribution we can get samples from
    trace_pred = TracePredictive(wrapped_model,
                                 posterior,
                                 num_samples=nsamples)
    post_pred = trace_pred.run(x_data, None)
    post_summary = summary(post_pred, sites=['prediction', 'obs'])
    pred = post_summary["prediction"]
    obs = post_summary["obs"]


    #x = x_data.cpu().numpy().ravel()
    idx = np.argsort(y_data.squeeze())

    df = pd.DataFrame({
        "y_test": y_data[idx].squeeze(),
        #"obs": obs[idx],
        "pred_mean": pred["mean"][idx].values,
        "pred_std": pred["std"][idx].values,
        "pred_16%": pred["16%"][idx].values,
        "pred_84%": pred["84%"][idx].values,
        "obs_mean": obs["mean"][idx].values,
        "obs_std": obs["std"][idx].values,
        "obs_16%": obs["16%"][idx].values,
        "obs_84%": obs["84%"][idx].values,
    })

    print(df)

    plot_pred(df)
    plt.title('trace summary: pred')
    plot_obs(df)
    plt.title('trace summary: obs')
    #plot_dists(df)


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

def guide_summary(guide, x_data, y_data):
    #import pudb; pudb.set_trace()
    sampled_models = [guide(None, None) for _ in range(10000)]
    npredicted = np.asarray(
        [model(x_data).data.cpu().numpy()[:, 0] for model in sampled_models]
    )
    pred_mean = np.mean(npredicted, axis=0)
    pred_std = np.std(npredicted, axis=0)
    pred_16q = np.percentile(npredicted, 16, axis=0)
    pred_84q = np.percentile(npredicted, 84, axis=0)
    #[ins] In [7]: npredicted.shape
    #Out[7]: (10000, 2500)
    # Therefore, average is over the 10000 model samples


    #x = x_data.cpu().numpy().ravel()
    idx = np.argsort(y_data.squeeze())

    df = pd.DataFrame({
        "pred_mean": pred_mean[idx],
        "pred_std": pred_std[idx],
        "pred_16%": pred_16q[idx],
        "pred_84%": pred_84q[idx],
        "y_test": y_data.squeeze()[idx],
    })

    plot_pred(df)
    plt.title('Guide summary')


def plot_pred(df):
    plt.figure()
    plt.plot(df['y_test'], df['pred_mean'], 'o', color='C0', label='pred_mean')    
    plt.fill_between(df["y_test"],
                     df["pred_16%"],
                     df["pred_84%"],
                     color='C1',
                     alpha=0.5)    
    plt.legend()


def plot_obs(df):
    plt.figure()
    plt.plot(df['y_test'], df['pred_mean'], 'o', color='C0', label='pred_mean')
    plt.plot(df['y_test'], df['obs_mean'], '+', color='C1', label='obs_mean')
    plt.fill_between(df["y_test"],
                     df["obs_16%"],
                     df["obs_84%"],
                     color='C1',
                     alpha=0.5)
    plt.legend()


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
    hu, PC, xx, xtest_pca, xtrain, xtest, yy, ytest, X,Y = np.load(saved_data_files[idx])

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name).mean(0).mean(0))

    nsamples = 1000

    xs = torch.tensor(xtest_pca.astype('float32'))
    ys =  np.zeros(shape=(2500,nsamples,1))

    trace_summary(svi, model, xs, ytest)
    guide_summary(guide, xs, ytest)


    for i in range(nsamples):
        sampled_model = guide(None, None)
        ys[:,i,:] = sampled_model(xs).cpu().detach().numpy()

    mean_0 = ys.mean(1)[:,0]                                           
    std_0 = ys.std(1)[:,0] 

    """
    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.title("Fit to y")
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    """
    """
    plt.figure(1)
    plt.errorbar(ytest[:,0], mean_0, yerr=std_0, fmt='o', c='r', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error
    """

    """
    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("y hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,0] - mean_0)/std_0)
    """
