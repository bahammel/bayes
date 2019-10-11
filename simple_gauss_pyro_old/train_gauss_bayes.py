import pyro
import numpy as np
from model_bayes_nn_m1 import NN_Model, get_pyro_model
import matplotlib.pyplot as plt
import torch
from data_gauss_bayes import get_dataset, seed_everything
from eval_gauss_bayes_m2 import trace_summary
from tqdm import tqdm
from datetime import datetime
import os
import pdb
import torch.nn.functional as F
from pyro.infer import Trace_ELBO, TraceEnum_ELBO, config_enumerate
import pyro.poutine as poutine

# enable validation (e.g. validate parameters of distributions)
assert pyro.__version__.startswith('0.3.1')
#pyro.enable_validation(True)

# We'll ue this helper to check our models are correct.
def test_model(model, guide, loss):
    pyro.clear_param_store()
    loss.loss(model, guide)

#pdb.set_trace()

EPOCHS = 1000


def train_nn(training_generator):
    regression_model = RegressionModel(p=1)
    optim = torch.optim.Adam(regression_model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    for e in range(EPOCHS):
        losses = []
        for x_data, y_data in tqdm(training_generator):
            # calculate the loss and take a gradient step
            y_pred = regression_model(x_data)
            loss = loss_fn(y_pred, y_data)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
        print(np.mean(losses))

    for name, param in regression_model.named_parameters():
        print(name, param.data.cpu().numpy())


def train_bayes(training_generator):
    svi, model, guide = get_pyro_model(return_all=True)

    loss_hist = []
    for e in range(EPOCHS):
        losses = []
        for x_data, y_data in tqdm(training_generator):
            losses.append(svi.step(x_data, y_data))
            
        #test_model(model(x_data,y_data), model(x_data,y_data), Trace_ELBO())

        loss_hist.append(np.mean(losses))
        print(f"epoch {e}/{EPOCHS} :", loss_hist[-1])

    trace = poutine.trace(model).get_trace(x_data, y_data)
    trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
    print(trace.format_shapes())

    plt.plot(loss_hist)
    plt.yscale('log')
    plt.title("ELBO")
    plt.xlabel("step")
    plt.ylabel("Epoch loss")

    trace_summary(svi, model, x_data, y_data)

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))


def save():
    save_model = input("save model > ")

    if save_model.lower().startswith('y'):
        experiment_id = input("Enter exp name, press return to use datetime> ")
        if not experiment_id:
            experiment_id = datetime.now().isoformat()

        if os.environ['HOSTNAME'] == 'fractal':
            SAVE_PATH = f'/hdd/bdhammel/checkpoints/bayes/{experiment_id}'
        else:
            SAVE_PATH = f'/usr/WS1/hammel1/proj/checkpoints/bayes/{experiment_id}'

        print("Saving to :", SAVE_PATH)
        pyro.get_param_store().save(SAVE_PATH + '.params')

        save_data = input("save data > ")
        if save_data.lower().startswith('y'):
            dataset = training_generator.dataset
            dataset.save(SAVE_PATH)
    else:
        print(f"input was {save_model} not saving model")


if __name__ == '__main__':
    seed_everything()
    pyro.clear_param_store()
    training_generator = get_dataset(batch_size=256)

    #evaluate()
    # train_nn(training_generator)
    train_bayes(training_generator)
    save()
