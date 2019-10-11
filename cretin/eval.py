import os
import torch
import pyro
from pyro.distributions import Delta
from pyro.infer import EmpiricalMarginal, TracePredictive
import glob
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
import numpy as np

import cretin_11V_3HL_m1 as cretin_nn
import pca_data_cretin_m1_ndl_sym as pca_data_hu

USE_GPU = torch.cuda.is_available()
device = torch.device('cuda' if USE_GPU else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if USE_GPU else 'torch.FloatTensor')

# TODO: make this save under experiment ID
#PARAM_FILE = './test.pt'

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
                                 num_samples=10000)
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

    #import pudb; pudb.set_trace()

def guide_summary(guide, x_data, y_data):
    #import pudb; pudb.set_trace()
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
    #plt.plot(df['x_data'], df['obs'], 'o', color='C5', label='obs')
    plt.plot(df['x_data'], df['obs_mean'], color='C1', label='obs_mean')
    plt.fill_between(df["x_data"],
                     df["obs_perc_5"],
                     df["obs_perc_95"],
                     color='C1',
                     alpha=0.5)
    plt.legend()


if __name__ == '__main__':

    model = cretin_nn.pyromodel
    guide = cretin_nn.guide

    pyro.get_param_store().load(PARAM_FILE)

    #hu, PC, xx, xtest_pca, xtrain, xtest, yy, ytest, X,Y = np.load(
    #    '/usr/WS1/hammel1/proj/data/cretin_pca_data.npy')
    hu, PC, xx, xtest_pca, xtrain, xtest, yy, ytest, X,Y = np.load(DATA_FILE)

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))

    xs = torch.tensor(xtest_pca.astype('float32'))
    ys =  np.zeros(shape=(1492,50,11))

    # trace_summary(svi, model, x_data, y_data)
    # guide_summary(guide, xs, ytest)


    for i in range(50):
        sampled_model = guide(None, None)
        ys[:,i,:] = sampled_model(xs).cpu().detach().numpy()

    mean_0 = ys.mean(1)[:,0]                                           
    std_0 = ys.std(1)[:,0] 

    mean_1 = ys.mean(1)[:,1] 
    std_1 = ys.std(1)[:,1]          

    mean_2 = ys.mean(1)[:,2] 
    std_2 = ys.std(1)[:,2]          

    mean_3 = ys.mean(1)[:,3] 
    std_3 = ys.std(1)[:,3]          

    mean_4 = ys.mean(1)[:,4] 
    std_4 = ys.std(1)[:,4]          

    mean_5 = ys.mean(1)[:,5] 
    std_5 = ys.std(1)[:,5]          

    mean_6 = ys.mean(1)[:,6] 
    std_6 = ys.std(1)[:,6]          

    mean_7 = ys.mean(1)[:,7] 
    std_7 = ys.std(1)[:,7]          

    mean_8 = ys.mean(1)[:,8] 
    std_8 = ys.std(1)[:,8]          

    mean_9 = ys.mean(1)[:,9] 
    std_9 = ys.std(1)[:,9]          


    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.title("Fit to t1")
    plt.xlabel("t1_test")
    plt.ylabel("t1_pred")
    plt.errorbar(ytest[:,0], mean_0, yerr=std_0, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.title("Fit to n1")
    plt.xlabel("n1_test")
    plt.ylabel("n1_pred")
    plt.errorbar(ytest[:,1], mean_1, yerr=std_1, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.title("Fit to mhot")
    plt.xlabel("mhot_test")
    plt.ylabel("mhot_pred")
    plt.errorbar(ytest[:,2], mean_2, yerr=std_2, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Fit to mix")
    plt.xlabel("mix_test")
    plt.ylabel("mix_pred")
    plt.errorbar(ytest[:,3], mean_3, yerr=std_3, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Fit to mixhot")
    plt.xlabel("mixhot_test")
    plt.ylabel("mixhot_pred")
    plt.errorbar(ytest[:,4], mean_4, yerr=std_4, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.title("Fit to n2")
    plt.xlabel("n2_test")
    plt.ylabel("n2_pred")
    plt.errorbar(ytest[:,6], mean_6, yerr=std_6, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.title("Fit to t2")
    plt.xlabel("t2_test")
    plt.ylabel("t2_pred")
    plt.errorbar(ytest[:,7], mean_7, yerr=std_7, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.title("Fit to n3")
    plt.xlabel("n3_test")
    plt.ylabel("n3_pred")
    plt.errorbar(ytest[:,8], mean_8, yerr=std_8, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.title("Fit to mch")
    plt.xlabel("mch_test")
    plt.ylabel("mch_pred")
    plt.errorbar(ytest[:,9], mean_9, yerr=std_9, fmt='o', c='b', ms = 1, elinewidth=0.1,errorevery=1)  #mean of predicted values w 1std error




    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("t1 hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,0] - mean_0)/std_0)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("n1 hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,1] - mean_1)/std_1)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("mhot hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,2] - mean_2)/std_2)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("mix hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,3] - mean_3)/std_3)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("mixhot hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,4] - mean_4)/std_4)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("n2 hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,6] - mean_6)/std_6)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("t2 hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,7] - mean_7)/std_7)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("n3 hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,8] - mean_8)/std_8)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.title("mch hist")
    plt.xlabel("std")
    plt.ylabel("# (pred-actual)/std]")
    plt.hist(np.abs(ytest[:,9] - mean_9)/std_9)
