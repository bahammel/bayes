import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.datasets import load_digits
from tqdm import tqdm


def noise(X):
    pct_noise = 0.1    #10% noise
    print("Noise multiplier (%) is:", pct_noise)

    nX = X * (1 + pct_noise/100.*(np.random.random_sample((X[:,1].size,X[1,:].size)) - 0.5))
    plt.figure()
    plt.title('data w & w/o noise')
    plt.plot(np.transpose(X[::1000,:]))
    plt.plot(np.transpose(nX[::1000,:]))

    X = nX
    
    return X

def gauss_data_2():
    hu = np.linspace(1, 10, 200) 
    #num_of_gauss = np.random.randint(1, 2)
    num_of_cases = 20000
    num_of_gauss = 10

    X = []
    Y = []
    for cases in range(num_of_cases):
        #Create num_of_cases different fake spectra, each made from a set of (mu, std, amp)
        I = 0.0; mu = 0.0; std = 0.0; amp = 0.0; mc = 0.0;
        mu = np.random.choice(np.linspace(2, 9, 20))
        std = np.random.choice(np.linspace(0.1, 3.0, 20))
        amp = np.random.choice(np.linspace(0.1, 1.0,20))
        for gauss in range(num_of_gauss):
            mc = mc + 1
            mult = mc * 0.1
            #Add up num_of_gauss different Gaussians, where mu and std are scaled by mult, to make fake spectrum
            I = I + amp * (1.0 - hu/12.0) / (std*np.sqrt(2.*np.pi)) * np.exp(-(hu - mult*mu)**2./(2.*mult*std**2.))
        Y.append([mu,std,amp])
        X.append(I)
    Y = np.array(Y)
    print(Y.shape)
    X = np.array(X)
    print(X.shape)

    X = noise(X)


def gauss_data_bay():
    hu = np.linspace(1, 10, 200) 
    num_of_cases = 10000

    X = []
    Y = []
    for cases in range(num_of_cases):
        #Create num_of_cases different fake spectra, each made from a set of (mu, std, amp)
        I = 0.0; mu = 0.0; std = 0.0; amp = 0.0; mc = 0.0; mult = 1.0;
        mu = np.random.choice(np.linspace(4, 8, 2))
        #mu = 5.5
        #std = np.random.choice(np.linspace(2, 3, 2))
        std = 0.5
#        amp = np.random.choice(np.linspace(0.1, 1.0,20))
        amp = 0.5
        #Add up num_of_gauss different Gaussians, where mu and std are scaled by mult, to make fake spectrum
        I = amp * (1.0) / (std*np.sqrt(2.*np.pi)) * np.exp(-(hu - mu)**2./(2.* std**2.))
        Y.append([mu])
        X.append(I)
    Y = np.array(Y)
    print(Y.shape)
#    print(Y)
    X = np.array(X)
    print(X.shape)
#    print(X)

    X = noise(X)
    
    
    plt.figure()
    plt.title('data')
    plt.plot(hu,np.transpose(X[::50]))

    return hu, X, np.asarray(Y, dtype=np.float32)
