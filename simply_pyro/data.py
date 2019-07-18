import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import random

USE_GPU = torch.cuda.is_available()
device = torch.device('cuda' if USE_GPU else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if USE_GPU else 'torch.FloatTensor')


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

    def __init__(self, m, b, epsilon, seed):
        np.random.seed(seed)
        self.X = x = 10 * np.random.rand(256*200, 1)  # shape (cases, features)
        self.Y = m * x + b + epsilon * np.random.randn(*x.shape)  # shape (cases, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx].astype(np.float32)
        return torch.tensor(x, device=device), torch.tensor(y, device=device).squeeze(-1)

    def save(self, path):
        data = np.c_[self.X, self.Y]
        np.save(path, data)

    def load(self, path):
        X, Y = np.load(path).T
        self.X = X[:, None]
        self.Y = Y


def get_dataset(m=0, b=5.0, epsilon=1.00, batch_size=128, seed=None, data_file=None):
    print(f'Fitting line: y={m}x+{b}')
    training_set = DataSet(m=m, b=b, epsilon=epsilon, seed=seed)
    if data_file is not None:
        training_set.load(data_file)

    return DataLoader(training_set, batch_size=batch_size, drop_last=True)


if __name__ == '__main__':
    training_set = DataSet(0, 2, 0.1)
    training_generator = DataLoader(training_set, batch_size=50, shuffle=True)

    for x, y in training_generator:
        plt.plot(x.cpu().numpy().ravel(), y.cpu().numpy().ravel(), 'o')
        input()
        plt.draw()
