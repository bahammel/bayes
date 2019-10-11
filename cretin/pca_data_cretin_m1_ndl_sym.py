import numpy as np
import matplotlib.pyplot as plt
#import utils_cretin_m1_sym as utils_cretin
import utils_cretin_m1_PySh as utils_cretin
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from tqdm import tqdm


def pca_data():
    plt.close('all')

    hu, xtrain, ytrain, xtest, ytest, X, Y = utils_cretin.cretin_data()

    scaler = StandardScaler()
    # print(scaler.fit(np.array(X)))
    #>>>>xtrain_std = scaler.fit_transform(np.array(xtrain))  

    # Fit on training set only.
    scaler.fit(xtrain)
    # Apply transform to both the training set and the test set.
    #xtrain_use = scaler.transform(xtrain)
    #xtest_use = scaler.transform(xtest)
    xtrain_use = xtrain
    xtest_use = xtest


    pca = PCA(n_components=33) 
    #pca = PCA(0.999)
    pca.fit(xtrain_use)
    #xtrain_hat = pca.transform([xtrain[0]])
    PC = pca.n_components_ 
    print(f"Data decomposed into {PC} components")

    evecs = pca.components_[pca.explained_variance_.argsort()][::-1]
    evals = pca.explained_variance_[pca.explained_variance_.argsort()][::-1]

    plt.figure()
    [plt.plot(vec) for vec in evecs]
    plt.title("eigen vectors")

    fig, axes = plt.subplots(pca.n_components_, 1, figsize=(6, 10))
    plt.title("individual eigen vectors")
    for i, ax in enumerate(axes.flat):
        ax.plot(evecs[i])

    xtrain_pca = pca.transform(xtrain_use)
    xtest_pca = pca.transform(xtest_use)

    plt.figure()
    plt.plot(xtrain_use[0], label='data')
    #plt.plot(pca.mean_ + np.sum(xtrain_pca[0].T * pca.components_, axis=0), label='manual reconstruction')   A
    plt.plot(pca.inverse_transform(xtrain_pca)[0], linestyle='dashed',label='pca reconstruction')
    #xtrain_pca = scaler.transform(xtrain)
    #xtest_pca = scaler.transform(xtest)
    plt.legend()
    plt.title("reconstructions")

    plt.show()

    import pickle
    s = pickle.dumps(pca)
    from datetime import datetime                                                                                                                                                          

    experiment_id = datetime.now().isoformat() 
    filename = f"/usr/WS1/hammel1/proj/data/{experiment_id}_pca_pickle"    
    outfile = open(filename,'wb')
    pickle.dump(pca, outfile)
    outfile.close()

    return hu, PC, xtrain_pca, xtest_pca, xtrain, xtest, ytrain, ytest, X, Y
