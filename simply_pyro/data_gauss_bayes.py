import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import random

torch.set_default_tensor_type('torch.cuda.FloatTensor')
USE_GPU = torch.cuda.is_available()
device = torch.device('cuda' if USE_GPU else 'cpu')


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if USE_GPU:
    print("=" * 80)
    print("Model is using GPU")
    print("=" * 80)


class DataSet(Dataset):

    def __init__(self, mu, seed):
        hu = np.linspace(1, 10, 100)
        np.random.seed(seed)
        #shift = np.random.choice(np.linspace(-1, 1, 5000))
        shift = np.random.rand(5000, 1) - 1.0
        #mu = np.random.choice(np.linspace(5.8, 6.2, 2))
        self.X = x = mu * (1. + shift/1.e2)  
        std = 0.2
        #std = np.random.choice(np.linspace(2, 3, 2))
        amp = 0.5
        #amp = np.random.choice(np.linspace(0.1, 1.0,20))
        self.Y = amp * (1.0) / (std*np.sqrt(2.*np.pi)) * np.exp(-((hu - mu)**2./(2.* std**2.))**1.0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx].astype(np.float32)
        return torch.tensor(x, device=device), torch.tensor(y, device=device)

    def save(self, path):
        data = np.c_[self.X, self.Y]
        np.save(path, data)

    def load(self, path):
        X, Y = np.load(path).T
        self.X = X[:, None]
        self.Y = Y[:, None]


def get_dataset(mu=6.0, batch_size=128, seed=None, data_file=None):
    print(f'Fitting Gaussian with: mu = {mu}')
    training_set = DataSet(mu=mu,seed=seed)
    if data_file is not None:
        training_set.load(data_file)

    return DataLoader(training_set, batch_size=batch_size)


if __name__ == '__main__':
    training_set = DataSet(0, 2, 3)
    training_generator = DataLoader(training_set, batch_size=50, shuffle=True)

    for x, y in training_generator:
        plt.plot(x.cpu().numpy().ravel(), y.cpu().numpy().ravel(), 'o')
        input()
        plt.draw()
