import os
import torch
import pyro
from pyro.distributions import Delta
from pyro.infer import EmpiricalMarginal, TracePredictive
from model_bayes_nn import get_pyro_model
from data_gauss_bayes import get_dataset
import glob
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
import numpy as np

torch.set_default_tensor_type('torch.cuda.FloatTensor')
USE_GPU = torch.cuda.is_available()
device = torch.device('cuda' if USE_GPU else 'cpu')
"""
if os.environ['HOSTNAME'] == 'fractal':
    MODEL_FILES = '/hdd/bdhammel/checkpoints/bayes/*.params'
else:
    MODEL_FILES = '/usr/WS1/hammel1/proj/checkpoints/bayes/*.params'

if os.environ['HOSTNAME'] == 'fractal':
    DATA_FILES = '/hdd/bdhammel/checkpoints/bayes/*.npy'
else:
    DATA_FILES = '/usr/WS1/hammel1/proj/checkpoints/bayes/*.npy'
"""

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
        marginal_site = pd.DataFrame(marginal[:, i, :]).transpose()
        describe = partial(
            pd.Series.describe, percentiles=[.05, 0.25, 0.5, 0.75, 0.95]
        )
        site_stats[site_name] = marginal_site.apply(
            describe, axis=1
        )[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]

    return site_stats


def wrapped_model_fn(model):
    def _wrapped_model(x_data, y_data):
        pyro.sample("prediction", Delta(model(x_data, y_data)))
    return _wrapped_model

def trace_summary(svi, model, x_data, y_data):

    posterior = svi.run(x_data, y_data)
    wrapped_model = wrapped_model_fn(model)

    # posterior predictive distribution we can get samples from
    trace_pred = TracePredictive(wrapped_model,
                                 posterior,
                                 num_samples=1000)
    post_pred = trace_pred.run(x_data, None)
    post_summary = summary(post_pred, sites=['prediction', 'obs'])
    mu = post_summary["prediction"]
    obs = post_summary["obs"]


    x = x_data.cpu().numpy().ravel()
    idx = np.argsort(x)

    df = pd.DataFrame({
        "x_data": x[idx],
        "y_data": y_data.cpu().numpy().ravel()[idx],
        #"obs": obs[idx],
        "mu_mean": mu["mean"][idx],
        "mu_std": mu["std"][idx],
        "mu_perc_5": mu["5%"][idx],
        "mu_perc_95": mu["95%"][idx],
        "obs_mean": obs["mean"][idx],
        "obs_std": obs["std"][idx],
        "obs_perc_5": obs["5%"][idx],
        "obs_perc_95": obs["95%"][idx],
    })

    print(df)

    plot_mu(df)
    plt.title('trace summary: mu')
    plot_obs(df)
    plt.title('trace summary: obs')


def guide_summary(guide, x_data, y_data):
    sampled_models = [guide(None, None) for _ in range(10000)]
    npredicted = np.asarray(
        [model(x_data).data.cpu().numpy()[:, 0] for model in sampled_models]
    )
    pred_mean = np.mean(npredicted, axis=0)
    pred_5q = np.percentile(npredicted, 5, axis=0)
    pred_95q = np.percentile(npredicted, 95, axis=0)

    x = x_data.cpu().numpy().ravel()
    idx = np.argsort(x)

    df = pd.DataFrame({
        "x_data": x[idx],
        "mu_mean": pred_mean[idx],
        "mu_perc_5": pred_5q[idx],
        "mu_perc_95": pred_95q[idx],
        "y_data": y_data.cpu().numpy().ravel()[idx],
    })

    plot_mu(df)
    plt.title('Guide summary')


def plot_mu(df):
    plt.figure()
    plt.plot(df['x_data'], df['y_data'], 'o', color='C0', label='true')
    plt.plot(df['x_data'], df['mu_mean'], color='C1', label='mu')
    plt.fill_between(df["x_data"],
                     df["mu_perc_5"],
                     df["mu_perc_95"],
                     color='C1',
                     alpha=0.5)
    plt.legend()


def plot_obs(df):
    plt.figure()
    plt.plot(df['x_data'], df['y_data'], 'o', color='C0', label='true')
    plt.plot(df['x_data'], df['obs_mean'], color='C1', label='obs')
    plt.fill_between(df["x_data"],
                     df["obs_perc_5"],
                     df["obs_perc_95"],
                     color='C1',
                     alpha=0.5)
    plt.legend()



