import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import random
import pdb
#pdb.set_trace()

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

    def __init__(self, m, b, epsilon, seed=None):
        Z = np.linspace(0.1, 1.0, 100)
        np.random.seed(seed)
        shift = epsilon * (2*np.random.rand(5000, 1) - 1.0)
        bsh = b * (1. + shift)  
        #self.Y = b*np.ones(5000)
        self.Y = bsh
        self.X = (m * Z) + bsh


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx].astype(np.float32)
        return torch.tensor(x, device=device), torch.tensor(y, device=device).squeeze(-1)

    def save(self, path):
        np.save(path, {'X':self.X, 'Y':self.Y})

    def load(self, path):
        data = np.load(path)
        self.X = data.item()['X']
        self.Y = data.item()['Y']


def get_dataset(m=0.0,b=2.0, epsilon=1.e-10, batch_size=256, seed=None, data_file=None):
    print(f'Fitting Linear with: m={m}, b = {b}, epsilon = {epsilon}')
    training_set = DataSet(m=m,b=b,epsilon=epsilon,seed=seed)

    img, lab = training_set.__getitem__(0)
    print('input shape at the first row : {}'.format(img.size()))      
    print('label shape at the first row : {}'.format(lab.size()))      
    print('training set shape:', np.array(training_set).shape)                                                                                                                                                 

    train_loader_check = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    train_iter_check = iter(train_loader_check)
    print(type(train_iter_check))
    images, labels = train_iter_check.next()
    print('images shape on batch size = {}'.format(images.size()))
    print('labels shape on batch size = {}'.format(labels.size()))
    #print(np.array(test_set).shape)

    if data_file is not None:
        training_set.load(data_file)
        print(training_set.X.shape)
        print(training_set.Y.shape)

    return DataLoader(training_set, batch_size=batch_size, drop_last=True)


if __name__ == '__main__':
    training_set = DataSet(0.0, 2.0, 2.0)
    training_generator = DataLoader(training_set, batch_size=50, shuffle=True)
    for x, y in training_generator:
        plt.plot(x.cpu().numpy().ravel(), y.cpu().numpy().ravel(), 'o')
        input()
        plt.draw()
