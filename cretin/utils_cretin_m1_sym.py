import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.datasets import load_digits
from tqdm import tqdm
plt.ion()

def shift(z):
    pct_shift = 200.0    #10% noise
    print("Shift multiplier (%) is:", pct_shift)

    X = z['spectra']   
    #z['inputs'] = z['inputs']
    #nX = X * ((np.random.random_sample((X[:,1].size,X[1,:].size)) - 0.5)*pct_noise/100. + 1.)    
    nX = X * (1 + pct_shift/100)
    z['spectra'] = nX
    plt.figure()
    plt.title('data w & w/o noise')
    plt.plot(np.transpose(X[::1000,:]))
    plt.plot(np.transpose(nX[::1000,:])) 
    return z

def noise(z):
    pct_noise = 0.0    #10% noise
    print("Noise multiplier (%) is:", pct_noise)

    X = z['spectra']   
    #z['inputs'] = z['inputs']
    #nX = X * ((np.random.random_sample((X[:,1].size,X[1,:].size)) - 0.5)*pct_noise/100. + 1.)    
    nX = X * (1 + pct_noise/100.*(np.random.random_sample((X[:,1].size,X[1,:].size)) - 0.5))
    z['spectra'] = nX
    plt.figure()
    plt.title('data w & w/o noise')
    plt.plot(np.transpose(X[::1000,:]))
    plt.plot(np.transpose(nX[::1000,:])) 
    return z


def normalize(_z, debug=True):
    z = dict()

    limits = [(2000.0, 4600.0),
              (0.8e+24, 4.0e+24),
              (0.1, 0.6),
              (1.0, 1000.0),
              (1.0, 100.0),
              (174.0, 175.0),
              (0.8e+24, 5.0e+24),
              (100.0, 200.0),
              (0.8e+24, 5.0e+24),
              (1.e-4, 5.e-4),
              (0.03, 0.1),
             ]


    conversion_op = [
        lambda x: x / 1.e4,
        lambda x: (x)/1.e25,
        #lambda x: np.sqrt(x)/1e-2,
        lambda x: (x)/1.0,
        #lambda x: np.sqrt(x)/10,
        lambda x: (x)/1.e3,
        #lambda x: np.sqrt(x)/10,
        lambda x: (x)/100,
        lambda x: x / 100,
        lambda x: (x)/1.e25,
        lambda x: (x)/200.,
        lambda x: (x)/1.e25,
        lambda x: x / 4.e-4,
        lambda x: x * 10,
        ]

    if debug:
        for i, lim in enumerate(limits):
            print(conversion_op[i](lim[0]), conversion_op[i](lim[1])) 

    inputs = []
    for i, input_data in enumerate(_z['inputs'].T):
        data = conversion_op[i](input_data)
        print(f"{data.min():.4f}, {data.max():.4f}")
        inputs.append(data)
    
    z['inputs'] = np.transpose(inputs)
    z['spectra'] = _z['spectra']

    return z


def clip_data(z, debug=True):

    clip_range = [
        (3000.0, 6000.0),
        (1.0e+25, 5.0e+25),
        (3.0e-06, 1e-05),
        (1.0, 1000.0),
        (1.0, 2.0),
        (10.0, 20.0),
        (2.0e+25, 1.0e+26),
        (50.0, 300.0),
        (1.0e+24, 5.0e+24),
        (0.00010, 0.00055),
        (0.05, 0.1),
    ]
    

    Y = z['inputs'] 
    X = z['spectra'] 
    for i in range(Y.shape[-1]):
        _idx = (Y[:, i] > clip_range[i][0]) & (Y[:, i] < clip_range[i][1])
        try:
            idx = np.logical_and(idx, _idx)
        except:
            idx = _idx
            print('first pass')
    
    _z = dict()
    _z['inputs'] = z['inputs'][idx]
    _z['spectra'] = X[idx]

    if debug:
        print('Clipping data to range:')
        for i in _z['inputs'].T:
            print(f'\t{i.min():.2e}, {i.max():.2e}')

    return _z

  
def cretin_data():
    # energy binning ranges from 9000 to 12500 ev
    energy_bins = np.linspace(10000, 13000, 250)
    
    # input variables and limits
    variables = ['t1', 'n1', 'mhot', 'mix', 'mix_hot',
    	     'theta', 'n2', 't3', 'n3', 'mch', 'rmax']
    
    # the simulation data
    #z = np.load('cretin_data.npz')
    #z = np.load('/usr/WS1/hammel1/proj/Cretin_runs/cretin_data.npz')
    z = np.load('/p/lustre1/hammel1/proj/Cretin_runs/my_uqp_directories/cretin_data.npz')
    # reduce the range of data
    #z = clip_data(z)

    z = normalize(z)
    #z = noise(z)
    #z = shift(z)

    """
    Y = np.zeros([5000, 11])
    Y[:,0] = z['inputs'][:,0]/1.e3
    Y[:,1] = np.sqrt(z['inputs'][:,1])/1.e12
    Y[:,2] = np.sqrt(z['inputs'][:,2])/1.e-2
    Y[:,3] = z['inputs'][:,3]/1.e2
    Y[:,4] = z['inputs'][:,4]/1.e2
    Y[:,5] = z['inputs'][:,5]/10.
    Y[:,6] = np.sqrt(z['inputs'][:,6])/1.e12
    Y[:,7] = z['inputs'][:,7]/1.e2
    Y[:,8] = np.sqrt(z['inputs'][:,8])/1.e12
    Y[:,9] = z['inputs'][:,9]/1.e-4
    Y[:,10] = z['inputs'][:,10] * 10.
    """
    
    # z has spectra and inputs
    #print('The first full spectrum')
    #print(z['spectra'][0])
    #spectra.shape
    print('The inputs for the first spectrum')
    print(list(zip(variables,z['inputs'][0])))
    #print('The Normalized inputs for the first spectrum')
    #print(Y[0,:])
    
    
    # prints the max of the mix variable
    print('Max of the mix varible: should be under 1000:')
    print(z['inputs'][:,variables.index('mix')].max())

    X = z['spectra']
    didx = np.where(X[:,100] > 1.e10)
    print("Deleting indices:", didx)
    kidx = np.where(X[:,100] < 1.e10)
    print("Keeping indices:", kidx)

    X = X[kidx]
    Y = z['inputs'] 
    Y = Y[kidx]
    print("New X and Y shape is:", X.shape, Y.shape)

    hu = energy_bins

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.05)

    print("ytrain shape is:", ytrain.shape)
    print("ytest shape is:", ytest.shape)
    
    plt.figure()
    plt.title('data w & w/o noise')
    plt.plot(hu,np.transpose(X[::1000,:]))
    #plt.plot(hu,np.transpose(nX[::1000,:]))
    #return map(np.asarray, [X,Y])
    # return X,Y
    #return map(np.asarray, [hu, xtrain, xtest, ytrain, ytest])
    return hu, xtrain, ytrain, xtest, ytest, X, Y

