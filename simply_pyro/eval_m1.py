import os
import torch
import pyro
from pyro.distributions import Delta
from pyro.infer import EmpiricalMarginal, TracePredictive
from model import get_pyro_model
from data import get_dataset
import glob
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
import numpy as np

torc.set_default_tensor_type('torch.cuda.FloatTensor')
USE_GPU = torch.cuda.is_available()
device = torch.device('cuda' if USE_GPU else 'cpu')
#SAVE_DIR = '/usr/WS1/hammel1/proj/checkpoints/bayes/*'
exp_id = '2019-04-11T09:30:21.884475' 
SAVE_DIR = f'/usr/WS1/hammel1/proj/checkpoints/bayes/{exp_id}'
DATA_DIR = f'/usr/WS1/hammel1/proj/data/{exp_id}'

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
            pd.Series.describe, percentiles=[.05, 0.25, 0.5, 0.75, 0.95]
        )
        site_stats[site_name] = marginal_site.apply(
            describe, axis=1
        )[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]

    return site_stats


def wrapped_model(x_data, y_data):
    pyro.sample("prediction", Delta(model(x_data, y_data)))


def trace_summary(svi, xdata, ydata):

    posterior = svi.run(x_data, y_data)

    # posterior predictive distribution we can get samples from
    trace_pred = TracePredictive(wrapped_model,
                                 posterior,
                                 num_samples=1000)
    post_pred = trace_pred.run(x_data, None)
    post_summary = summary(post_pred, sites=['prediction', 'obs'])
    mu = post_summary["prediction"]
    y = post_summary["obs"]

    x = x_data.cpu().numpy().ravel()
    idx = np.argsort(x)

    df = pd.DataFrame({
        "x_data": x[idx],
        "mu_mean": mu["mean"][idx],
        "mu_perc_5": mu["5%"][idx],
        "mu_perc_95": mu["95%"][idx],
        "y_mean": y["mean"][idx],
        "y_perc_5": y["5%"][idx],
        "y_perc_95": y["95%"][idx],
        "y_data": y_data.cpu().numpy().ravel()[idx],
    })

    plot_mu(df,'solid','C0','C1','o','mu_TRACE')
    #plt.title('trace summary: mu')
    plot_y(df,'solid','C0','C4','o','y_TRACE')
    #plt.title('trace summary: y')


def guide_summary(guide, x_data, y_data):
    sampled_models = [guide(None, None) for _ in range(1000)]
    npredicted = np.asarray(
        [model(x_data).data.cpu().numpy()[:, 0] for model in sampled_models]
    )
    y_mean = np.mean(npredicted, axis=0)
    y_5q = np.percentile(npredicted, 5, axis=0)
    y_95q = np.percentile(npredicted, 95, axis=0)

    x = x_data.cpu().numpy().ravel()
    idx = np.argsort(x)

    df = pd.DataFrame({
        "x_data": x[idx],
        "y_mean": y_mean[idx],
        "y_perc_5": y_5q[idx],
        "y_perc_95": y_95q[idx],
        "y_data": y_data.cpu().numpy().ravel()[idx],
    })
    plot_y(df,'dashed','C2','C3','+r','y_GUIDE')
    plt.title('Summary')


def plot_mu(df,LS,COL1,COL2,MARK,LAB):
    plt.figure()
    plt.plot(df['x_data'], df['y_data'], MARK, color=COL1, label='true')
    plt.plot(df['x_data'], df['mu_mean'], color=COL2, label=LAB, linestyle=LS)
    plt.fill_between(df["x_data"],
                     df["mu_perc_5"],
                     df["mu_perc_95"],
                     color=COL2,
                     alpha=0.1)
    plt.legend()


def plot_y(df,LS,COL1,COL2,MARK,LAB):
    plt.figure(1)
    plt.plot(df['x_data'], df['y_data'], MARK, color=COL1, label='true')
    plt.plot(df['x_data'], df['y_mean'], color=COL2, label=LAB, linestyle=LS)
    plt.fill_between(df["x_data"],
                     df["y_perc_5"],
                     df["y_perc_95"],
                     color=COL2,
                     alpha=0.1)
    plt.legend()


if __name__ == '__main__':
    svi, model, guide = get_pyro_model(return_all=True)

#    saved_data_files = glob.glob(DATA_DIR)
#    saved_data_files.sort(key=os.path.getmtime)
#    print(*saved_data_files, sep='\n')
#    idx = int(input("file?> "))
#    torch.load(saved_data_files[idx])
    torch.load(DATA_DIR)
#    training_generator = np.load(saved_data_files[idx])
#    training_generator = iter(get_dataset())
#    x_data, y_data = next(training_generator)

#    saved_param_files = glob.glob(SAVE_DIR)
#    saved_param_files.sort(key=os.path.getmtime)
#    print(*saved_param_files, sep='\n')
#    idx = int(input("file?> "))
#    pyro.get_param_store().load(saved_param_files[idx])
    pyro.get_param_store().load(SAVE_DIR)
    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))

    trace_summary(svi, x_data, y_data)
    guide_summary(guide, x_data, y_data)
