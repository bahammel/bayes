import numpy as np
import matplotlib.pyplot as plt
import utils_gauss_hu_ndl2_train_test as utils_gauss_hu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from tqdm import tqdm

def pca_data():
    plt.close('all')

    hu, X, X_test, Y, Y_test  = utils_gauss_hu.gauss_data_bay()
    #hu, X, Y  = utils_gauss_hu.gauss_data_2()

    scaler = StandardScaler()
    # print(scaler.fit(np.array(X)))
    #>>>>xtrain_std = scaler.fit_transform(np.array(xtrain))  

    # Fit on training set only.
    scaler.fit(X)
    # Apply transform to both the training set and the test set.
    #xtrain_use = scaler.transform(xtrain)
    #xtest_use = scaler.transform(xtest)
    X_use = X_test

    pca = PCA(n_components=30) 
    #pca = PCA(0.9999)
    pca.fit(X_use)
    #xtrain_hat = pca.transform([xtrain[0]])
    PC = pca.n_components_ 
    print(f"Data decomposed into {PC} components")

    evecs = pca.components_[pca.explained_variance_.argsort()][::-1]
    evals = pca.explained_variance_[pca.explained_variance_.argsort()][::-1]
    X_pca_test = pca.transform(X_use)
    X_pca = pca.transform(X)


    import pickle
    s = pickle.dumps(pca)
    from datetime import datetime                                                                                                                                                          

    experiment_id = datetime.now().isoformat() 
    filename = f"/usr/WS1/hammel1/proj/data/{experiment_id}_pca_pickle"    
    outfile = open(filename,'wb')
    pickle.dump(pca, outfile)
    outfile.close()

    return hu, PC, X_pca_test, X_pca, X, X_test, Y, Y_test


"""
    plt.figure()
    [plt.plot(vec) for vec in evecs]
    plt.title("eigen vectors")

    fig, axes = plt.subplots(pca.n_components_, 1, figsize=(6, 10))
    plt.title("individual eigen vectors")
    for i, ax in enumerate(axes.flat):
        ax.plot(evecs[i])

    plt.figure()
    plt.plot(X_use[0], label='data')
    #plt.plot(pca.mean_ + np.sum(xtrain_pca[0].T * pca.components_, axis=0), label='manual reconstruction')   A
    plt.plot(pca.inverse_transform(X_pca)[0], linestyle='dashed',label='pca reconstruction')
    #xtrain_pca = scaler.transform(xtrain)
    #xtest_pca = scaler.transform(xtest)
    plt.legend()
    plt.title("reconstructions")
"""
