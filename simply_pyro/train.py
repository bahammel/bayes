import pyro
import numpy as np
from model import RegressionModel, get_pyro_model
import torch
from data import get_dataset
from tqdm import tqdm
from datetime import datetime

experiment_id = datetime.now().isoformat()

EPOCHS = 50
SAVE_DIR = f'/hdd/bdhammel/checkpoints/bayes/{experiment_id}'


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
    svi = get_pyro_model()

    for e in range(EPOCHS):
        losses = []
        for x_data, y_data in tqdm(training_generator):
            # calculate the loss and take a gradient step
            losses.append(svi.step(x_data, y_data))
        print(np.mean(losses))

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))

    save_model = input("save model > ")
    if save_model.lower().startswith('y'):
        pyro.get_param_store().save(SAVE_DIR)


if __name__ == '__main__':
    pyro.clear_param_store()
    training_generator = get_dataset()
    # train_nn(training_generator)
    train_bayes(training_generator)
