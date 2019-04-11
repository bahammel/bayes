import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

torch.set_default_tensor_type('torch.cuda.FloatTensor')
USE_GPU = torch.cuda.is_available()
device = torch.device('cuda' if USE_GPU else 'cpu')

if USE_GPU:
    print("=" * 80)
    print("Model is using GPU")
    print("=" * 80)


class DataSet(Dataset):

    def __init__(self, m, b, epsilon):
        self.X = x = 10 * np.random.rand(100, 1)
        self.Y = m * x + b + epsilon * np.random.randn(*x.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx].astype(np.float32)
        return torch.tensor(x, device=device), torch.tensor(y, device=device)


def get_dataset(m=0, b=2, epsilon=3, batch_size=128):
    print(f'Fitting line: y={m}x+{b}')
    training_set = DataSet(m=m, b=b, epsilon=epsilon)
    return DataLoader(training_set, batch_size=batch_size)


if __name__ == '__main__':
    training_set = DataSet(0, 2, 3)
    training_generator = DataLoader(training_set, batch_size=50, shuffle=True)

    for x, y in training_generator:
        plt.plot(x.cpu().numpy().ravel(), y.cpu().numpy().ravel(), 'o')
        input()
        plt.draw()